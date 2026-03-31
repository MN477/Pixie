"""
fuzzy_risk_engine.py
====================
Moteur de scoring clinique par logique floue (skfuzzy).
Lit behavior_summary.csv → génère final_clinical_scores.csv.

Dépendances :
    pip install scikit-fuzzy numpy pandas matplotlib

Usage :
    python fuzzy_risk_engine.py
    python fuzzy_risk_engine.py --input behavior_summary.csv --output final_clinical_scores.csv --plot
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION CLINIQUE — Ajustez ces seuils selon le profil de l'enfant
# (ex : enfant de 8 ans vs adolescent de 14 ans, contexte classe vs récré)
# ══════════════════════════════════════════════════════════════════════════════

# ── Agitation (agitation_per_min : nombre d'incréments d'agitation par minute) ──
# À AJUSTER : pour un enfant de 8 ans en classe, >30/min est déjà élevé.
#             Pour un ado, hausser le seuil "High" vers 80–100.
AGITATION_RANGE = (0, 200)          # (min, max) de l'univers
AGITATION_CALM_PEAK     =  10       # /min → clairement calme
AGITATION_MODERATE_PEAK =  50       # /min → modérément agité
AGITATION_HIGH_START    =  80       # /min → début zone agitation élevée
AGITATION_HIGH_PEAK     = 200       # /min → agitation maximale (borne univers)

# ── Stimming (stimming_pct : % de frames avec oscillation répétitive détectée) ──
# À AJUSTER : 5 % peut déjà être cliniquement significatif pour certains profils TSA.
STIMMING_RANGE = (0, 100)
STIMMING_RARE_PEAK      =   5       # % → rarement observé
STIMMING_FREQ_PEAK      =  30       # % → fréquent, surveillance recommandée
STIMMING_SEVERE_START   =  55       # % → début zone sévère
STIMMING_SEVERE_PEAK    = 100       # % → quasi-permanent

# ── Speed (mean_frame_speed_px : vitesse moyenne des articulations en px/frame) ──
# À AJUSTER : dépend de la résolution vidéo. Pour 720p, 8 px/frame ≈ naturel.
#             Pour 1080p, monter les seuils proportionnellement.
SPEED_RANGE = (0, 50)
SPEED_SLOW_PEAK    =  3             # px/frame → très lent / immobile
SPEED_NATURAL_PEAK = 10             # px/frame → mouvement naturel
SPEED_FAST_START   = 18             # px/frame → début zone agitation motrice
SPEED_FAST_PEAK    = 50             # px/frame → mouvement très rapide

# ── Risk Score (sortie) ──
RISK_RANGE = (0, 10)                # Score clinique normalisé sur 10

# ── Clamp : clip les valeurs hors-univers au lieu de crasher ──
CLAMP_INPUTS = True


# ══════════════════════════════════════════════════════════════════════════════
# CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════

class FuzzyBehaviorAnalyzer:
    """
    Moteur de logique floue pour le scoring clinique comportemental.

    Antécédents (entrées) :
        agitation  — agitation_per_min    de behavior_summary.csv
        stimming   — stimming_pct         de behavior_summary.csv
        speed      — mean_frame_speed_px  de behavior_summary.csv

    Conséquent (sortie) :
        risk_score — Score clinique [0–10]
                     0–3  : Low      (surveillance standard)
                     3–6  : Elevated (alerte, consulter un professionnel)
                     6–10 : Critical (intervention recommandée)
    """

    def __init__(self):
        self._build_universe()
        self._build_membership_functions()
        self._build_rules()
        self._build_control_system()

    # ──────────────────────────────────────────────
    # ÉTAPE 1 : Univers de discours (domaines numériques)
    # ──────────────────────────────────────────────
    def _build_universe(self):
        """Crée les variables floues d'entrée et de sortie."""

        # Antécédents (inputs)
        self.agitation = ctrl.Antecedent(
            np.linspace(*AGITATION_RANGE, 500), "agitation"
        )
        self.stimming = ctrl.Antecedent(
            np.linspace(*STIMMING_RANGE, 500), "stimming"
        )
        self.speed = ctrl.Antecedent(
            np.linspace(*SPEED_RANGE, 500), "speed"
        )

        # Conséquent (output)
        self.risk_score = ctrl.Consequent(
            np.linspace(*RISK_RANGE, 500), "risk_score",
            defuzzify_method="centroid"   # Centre de gravité — le plus précis
        )

    # ──────────────────────────────────────────────
    # ÉTAPE 2 : Fonctions d'appartenance (Membership Functions)
    # ──────────────────────────────────────────────
    def _build_membership_functions(self):
        """
        Définit les MFs trapézoïdales/triangulaires pour chaque variable.
        Utilisez fuzz.trapmf pour des zones plates (ex: extrêmes)
        et fuzz.trimf pour des pics triangulaires (ex: zones centrales).
        """

        uni_a = np.linspace(*AGITATION_RANGE, 500)
        uni_s = np.linspace(*STIMMING_RANGE,  500)
        uni_sp = np.linspace(*SPEED_RANGE,     500)
        uni_r = np.linspace(*RISK_RANGE,       500)

        # ── Agitation ──────────────────────────────
        # Calm    : [0 → 0 → CALM_PEAK → MODERATE_PEAK/2]
        self.agitation["Calm"] = fuzz.trapmf(
            uni_a,
            [AGITATION_RANGE[0], AGITATION_RANGE[0],
             AGITATION_CALM_PEAK,
             AGITATION_MODERATE_PEAK * 0.6]
        )
        # Moderate : triangle centré sur MODERATE_PEAK
        self.agitation["Moderate"] = fuzz.trimf(
            uni_a,
            [AGITATION_CALM_PEAK,
             AGITATION_MODERATE_PEAK,
             AGITATION_HIGH_START]
        )
        # High     : [HIGH_START → HIGH_PEAK → HIGH_PEAK]  (plateau à droite)
        self.agitation["High"] = fuzz.trapmf(
            uni_a,
            [AGITATION_MODERATE_PEAK,
             AGITATION_HIGH_START,
             AGITATION_RANGE[1],
             AGITATION_RANGE[1]]
        )

        # ── Stimming ───────────────────────────────
        self.stimming["Rare"] = fuzz.trapmf(
            uni_s,
            [STIMMING_RANGE[0], STIMMING_RANGE[0],
             STIMMING_RARE_PEAK,
             STIMMING_FREQ_PEAK * 0.5]
        )
        self.stimming["Frequent"] = fuzz.trimf(
            uni_s,
            [STIMMING_RARE_PEAK,
             STIMMING_FREQ_PEAK,
             STIMMING_SEVERE_START]
        )
        self.stimming["Severe"] = fuzz.trapmf(
            uni_s,
            [STIMMING_FREQ_PEAK,
             STIMMING_SEVERE_START,
             STIMMING_RANGE[1],
             STIMMING_RANGE[1]]
        )

        # ── Speed ──────────────────────────────────
        self.speed["Slow"] = fuzz.trapmf(
            uni_sp,
            [SPEED_RANGE[0], SPEED_RANGE[0],
             SPEED_SLOW_PEAK,
             SPEED_NATURAL_PEAK * 0.5]
        )
        self.speed["Natural"] = fuzz.trimf(
            uni_sp,
            [SPEED_SLOW_PEAK,
             SPEED_NATURAL_PEAK,
             SPEED_FAST_START]
        )
        self.speed["Fast"] = fuzz.trapmf(
            uni_sp,
            [SPEED_NATURAL_PEAK,
             SPEED_FAST_START,
             SPEED_RANGE[1],
             SPEED_RANGE[1]]
        )

        # ── Risk Score (output) ────────────────────
        self.risk_score["Low"] = fuzz.trapmf(
            uni_r, [0, 0, 2, 4]
        )
        self.risk_score["Elevated"] = fuzz.trimf(
            uni_r, [2, 5, 8]
        )
        self.risk_score["Critical"] = fuzz.trapmf(
            uni_r, [6, 8, 10, 10]
        )

    # ──────────────────────────────────────────────
    # ÉTAPE 3 : Règles floues expertes
    # ──────────────────────────────────────────────
    def _build_rules(self):
        """
        Règles IF-THEN définies par des experts cliniciens.
        Priorité de haut en bas (les règles ne sont pas exclusives
        en logique floue — toutes contribuent au résultat).

        ── Comment lire et modifier ces règles ──
        Chaque règle a la forme :
            ctrl.Rule(antécédent(s), conséquent)
        Combinez des antécédents avec & (AND) ou | (OR).
        Négation : ~self.agitation["Calm"]

        Pour ajouter une règle, copiez un bloc et adaptez-le.
        Pour pondérer une règle, ajoutez : .view() ou utilisez
        le paramètre `label=` pour le débogage.
        """

        a = self.agitation
        s = self.stimming
        sp = self.speed
        r  = self.risk_score

        self.rules = [

            # ════════ RÈGLES CRITIQUES ════════
            # Combinaison maximale : tout est élevé → score critique
            ctrl.Rule(a["High"] & s["Severe"],                  r["Critical"]),
            ctrl.Rule(a["High"] & sp["Fast"],                   r["Critical"]),
            ctrl.Rule(s["Severe"] & sp["Fast"],                 r["Critical"]),
            ctrl.Rule(a["High"] & s["Frequent"] & sp["Fast"],   r["Critical"]),

            # Stimming sévère seul suffit (indicateur TSA fort)
            ctrl.Rule(s["Severe"],                              r["Critical"]),

            # ════════ RÈGLES ÉLEVÉES ════════
            # Agitation haute avec stimming modéré
            ctrl.Rule(a["High"] & s["Frequent"],                r["Elevated"]),
            # Agitation modérée avec vitesse élevée
            ctrl.Rule(a["Moderate"] & sp["Fast"],               r["Elevated"]),
            # Stimming fréquent avec agitation modérée
            ctrl.Rule(s["Frequent"] & a["Moderate"],            r["Elevated"]),
            # Vitesse rapide seule (hyperactivité motrice)
            ctrl.Rule(sp["Fast"] & ~a["Calm"],                  r["Elevated"]),
            # Agitation haute mais stimming rare (TDAH sans TSA)
            ctrl.Rule(a["High"] & s["Rare"],                    r["Elevated"]),

            # ════════ RÈGLES BASSES ════════
            # Calme sur toutes les dimensions
            ctrl.Rule(a["Calm"] & s["Rare"] & sp["Slow"],       r["Low"]),
            ctrl.Rule(a["Calm"] & s["Rare"] & sp["Natural"],    r["Low"]),
            ctrl.Rule(a["Calm"] & s["Rare"],                    r["Low"]),
            # Agitation modérée + mouvement naturel
            ctrl.Rule(a["Moderate"] & sp["Natural"],            r["Low"]),
            # Calme dominant
            ctrl.Rule(a["Calm"],                                r["Low"]),
        ]

    # ──────────────────────────────────────────────
    # ÉTAPE 4 : Système de contrôle
    # ──────────────────────────────────────────────
    def _build_control_system(self):
        """Assemble le système de contrôle flou Mamdani."""
        self._ctrl_system    = ctrl.ControlSystem(self.rules)
        self._simulation     = ctrl.ControlSystemSimulation(self._ctrl_system)

    # ──────────────────────────────────────────────
    # API PUBLIQUE
    # ──────────────────────────────────────────────
    def compute_risk(
        self,
        agitation_per_min:    float,
        stimming_pct:         float,
        mean_frame_speed_px:  float,
    ) -> dict:
        """
        Calcule le score de risque clinique pour une ligne de behavior_summary.

        Paramètres
        ----------
        agitation_per_min    : valeur de la colonne agitation_per_min
        stimming_pct         : valeur de la colonne stimming_pct (0–100)
        mean_frame_speed_px  : valeur de la colonne mean_frame_speed_px

        Retourne
        --------
        dict avec :
            risk_score     : float [0–10]  — score défuzzifié (centroïde)
            risk_level     : str           — "Low" / "Elevated" / "Critical"
            agitation_in   : float         — valeur utilisée (après clamp)
            stimming_in    : float         — valeur utilisée (après clamp)
            speed_in       : float         — valeur utilisée (après clamp)
        """
        # ── Clamp des entrées hors-univers ──
        ag  = float(agitation_per_min)
        st  = float(stimming_pct)
        sp  = float(mean_frame_speed_px)

        if CLAMP_INPUTS:
            ag = np.clip(ag, AGITATION_RANGE[0] + 1e-6, AGITATION_RANGE[1] - 1e-6)
            st = np.clip(st, STIMMING_RANGE[0]  + 1e-6, STIMMING_RANGE[1]  - 1e-6)
            sp = np.clip(sp, SPEED_RANGE[0]     + 1e-6, SPEED_RANGE[1]     - 1e-6)

        # ── Inférence floue ──
        sim = ctrl.ControlSystemSimulation(self._ctrl_system)
        sim.input["agitation"] = ag
        sim.input["stimming"]  = st
        sim.input["speed"]     = sp

        try:
            sim.compute()
            score = float(sim.output["risk_score"])
        except Exception as e:
            # Fallback : règle heuristique simple si la défuzzification échoue
            score = self._fallback_score(ag, st, sp)
            print(f"  [WARN] Fuzzy compute failed ({e}), using fallback score={score:.2f}")

        # ── Catégorisation linguistique ──
        if score < 3.5:
            level = "Low"
        elif score < 6.5:
            level = "Elevated"
        else:
            level = "Critical"

        return {
            "risk_score":   round(score, 3),
            "risk_level":   level,
            "agitation_in": ag,
            "stimming_in":  st,
            "speed_in":     sp,
        }

    @staticmethod
    def _fallback_score(ag, st, sp) -> float:
        """Score heuristique de secours (régression linéaire normalisée)."""
        s  = (ag / AGITATION_RANGE[1]) * 4.0
        s += (st / STIMMING_RANGE[1])  * 4.0
        s += (sp / SPEED_RANGE[1])     * 2.0
        return float(np.clip(s, 0, 10))

    # ──────────────────────────────────────────────
    # VISUALISATION (optionnelle)
    # ──────────────────────────────────────────────
    def plot_membership_functions(self):
        """Affiche les fonctions d'appartenance pour inspection visuelle."""
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            fig.suptitle("Fuzzy Behavior Analyzer — Membership Functions", fontsize=14)

            vars_to_plot = [
                (self.agitation,  "Agitation (per min)",   ["Calm", "Moderate", "High"]),
                (self.stimming,   "Stimming (%)",          ["Rare", "Frequent", "Severe"]),
                (self.speed,      "Speed (px/frame)",      ["Slow", "Natural", "Fast"]),
                (self.risk_score, "Risk Score [0–10]",     ["Low", "Elevated", "Critical"]),
            ]
            colors = ["#2ecc71", "#f39c12", "#e74c3c"]

            for ax, (var, title, terms) in zip(axes.flat, vars_to_plot):
                x = var.universe
                for term, color in zip(terms, colors):
                    mf = fuzz.interp_membership(x, var[term].mf, x)
                    ax.plot(x, mf, label=term, color=color, linewidth=2)
                    ax.fill_between(x, 0, mf, alpha=0.12, color=color)
                ax.set_title(title, fontweight="bold")
                ax.set_ylim(-0.05, 1.1)
                ax.legend(loc="upper right", fontsize=9)
                ax.set_xlabel("Value")
                ax.set_ylabel("Membership")
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig("membership_functions.png", dpi=150)
            plt.show()
            print("[INFO] Plot saved → membership_functions.png")

        except ImportError:
            print("[WARN] matplotlib not installed — skipping plot.")

    def plot_risk_distribution(self, results_df: pd.DataFrame):
        """Bar chart du score de risque par track_id."""
        try:
            import matplotlib.pyplot as plt

            colors_map = {"Low": "#2ecc71", "Elevated": "#f39c12", "Critical": "#e74c3c"}
            bar_colors = [colors_map.get(lv, "#95a5a6") for lv in results_df["risk_level"]]

            fig, ax = plt.subplots(figsize=(max(8, len(results_df) * 1.2), 5))
            bars = ax.bar(
                results_df["track_id"].astype(str),
                results_df["risk_score"],
                color=bar_colors, edgecolor="white", linewidth=0.8
            )
            ax.axhline(3.5, color="#f39c12", linestyle="--", linewidth=1, label="Low / Elevated threshold")
            ax.axhline(6.5, color="#e74c3c", linestyle="--", linewidth=1, label="Elevated / Critical threshold")
            ax.set_ylim(0, 10.5)
            ax.set_xlabel("Track ID (Enfant)", fontsize=11)
            ax.set_ylabel("Clinical Risk Score", fontsize=11)
            ax.set_title("Clinical Risk Score per Child", fontsize=13, fontweight="bold")
            ax.legend(fontsize=9)

            for bar, row in zip(bars, results_df.itertuples()):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{row.risk_score:.1f}\n{row.risk_level}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold"
                )

            plt.tight_layout()
            plt.savefig("risk_distribution.png", dpi=150)
            plt.show()
            print("[INFO] Plot saved → risk_distribution.png")

        except ImportError:
            print("[WARN] matplotlib not installed — skipping plot.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Lecture CSV → Scoring → Export
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Fuzzy Clinical Risk Engine — behavior_summary.csv → final_clinical_scores.csv"
    )
    parser.add_argument("--input",  default="behavior_summary.csv",      help="CSV d'entrée")
    parser.add_argument("--output", default="final_clinical_scores.csv", help="CSV de sortie")
    parser.add_argument("--plot",   action="store_true",                 help="Afficher les graphiques")
    args = parser.parse_args()

    # ── Vérification de l'entrée ──
    if not os.path.exists(args.input):
        print(f"[ERROR] Fichier introuvable : {args.input}")
        print("        Lancez d'abord behavior_classifier.py pour générer behavior_summary.csv")
        return

    # ── Chargement ──
    print(f"[INFO] Lecture de {args.input} ...")
    df = pd.read_csv(args.input)

    required_cols = ["track_id", "agitation_per_min", "stimming_pct", "mean_frame_speed_px"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[ERROR] Colonnes manquantes dans le CSV : {missing}")
        print(f"        Colonnes disponibles : {list(df.columns)}")
        return

    print(f"[INFO] {len(df)} enfants (track_ids) à analyser.")

    # ── Initialisation du moteur flou ──
    print("[INFO] Initialisation du moteur flou ...")
    analyzer = FuzzyBehaviorAnalyzer()

    if args.plot:
        analyzer.plot_membership_functions()

    # ── Scoring de chaque enfant ──
    print("[INFO] Calcul des scores de risque clinique ...")
    results = []
    for _, row in df.iterrows():
        tid = int(row["track_id"])

        result = analyzer.compute_risk(
            agitation_per_min   = row["agitation_per_min"],
            stimming_pct        = row["stimming_pct"],
            mean_frame_speed_px = row["mean_frame_speed_px"],
        )

        # Fusionner avec les données sources pour un CSV riche
        merged = {
            "track_id":           tid,
            "duration_sec":       row.get("duration_sec",          np.nan),
            "hand_raise_pct":     row.get("hand_raise_pct",        np.nan),
            "dominant_posture":   row.get("dominant_posture",       "unknown"),
            "agitation_per_min":  result["agitation_in"],
            "stimming_pct":       result["stimming_in"],
            "mean_frame_speed_px": result["speed_in"],
            "risk_score":         result["risk_score"],
            "risk_level":         result["risk_level"],
        }
        results.append(merged)

        print(f"  Track {tid:>3} | "
              f"agit={result['agitation_in']:6.1f}/min | "
              f"stim={result['stimming_in']:5.1f}% | "
              f"spd={result['speed_in']:5.2f}px | "
              f"→ Risk={result['risk_score']:.2f} [{result['risk_level']}]")

    # ── Export ──
    results_df = pd.DataFrame(results).sort_values("risk_score", ascending=False)
    results_df.to_csv(args.output, index=False)
    print(f"\n[OK] Scores exportés → {args.output}")

    # ── Résumé console ──
    print("\n" + "═" * 60)
    print("  RÉSUMÉ CLINIQUE")
    print("═" * 60)
    level_counts = results_df["risk_level"].value_counts()
    for level in ["Critical", "Elevated", "Low"]:
        n = level_counts.get(level, 0)
        pct = n / len(results_df) * 100 if len(results_df) > 0 else 0
        icon = {"Critical": "🔴", "Elevated": "🟡", "Low": "🟢"}.get(level, "⚪")
        print(f"  {icon}  {level:<10} : {n} enfant(s)  ({pct:.0f}%)")
    print(f"\n  Score moyen   : {results_df['risk_score'].mean():.2f} / 10")
    print(f"  Score max     : {results_df['risk_score'].max():.2f}  "
          f"(Track {int(results_df.iloc[0]['track_id'])})")
    print("═" * 60 + "\n")

    if args.plot:
        analyzer.plot_risk_distribution(results_df)


if __name__ == "__main__":
    main()


// test sarra