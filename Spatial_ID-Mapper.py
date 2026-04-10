"""
=============================================================================
behaviour_classifier_visual.py  —  v11.2  (+ Spatial ID Mapper)
=============================================================================
CONTEXT : 2 students seated behind a desk, fixed camera.

SPATIAL ID MAPPER (new in this version)
========================================
Problem: YOLO tracker assigns unstable IDs across frames (ID switching),
producing parasitic IDs (3, 4, 5…) for the same two physical students.

Solution: SpatialIDMapper overrides tracker IDs using spatial position.
  - Detection center_x < frame_w / 2  → forced to ID 1 (Left student)
  - Detection center_x ≥ frame_w / 2  → forced to ID 2 (Right student)
  - Any other detection is silently discarded (no 3rd ID ever stored)

Details:
  - Center X computed from shoulders (priority), hips, or all visible joints
  - Ambiguity zone (±5% of frame width around midpoint): resolved using the
    last known X position for each zone (spatial memory)
  - Conflict resolution (two detections same side): keep the one farthest
    from the center; discard the other as parasitic
  - Minimum keypoint filter: detections with < 3 valid keypoints are ignored
  - track_states dict is pre-initialised with exactly {1: ..., 2: ...}
    and never grows beyond 2 entries
  - Final console report shows exactly 2 students: Left (ID:1) / Right (ID:2)

CHANGES v11 → v11.2 (behaviour detection)
==========================================
1. SLOUCHING: HEAD_BETWEEN_SHOULDERS signal (weight 0.40) — detects head
   bent down between shoulders, independent of desk_y and hips.
2. BODY ROCKING: 18-frame persistence counter + CV stability window.
3. FIDGET WRIST: suppressed when hand_raised; ZCR gate for hand_movement.
4. All DECAY/EXIT frame counts = 30 (~1s at 30fps).

PRIORITY: hand_raised > bounding > fidgeting > bouncing > posture
=============================================================================
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd


# ==============================================================================
# COCO KEYPOINT INDICES  (YOLOv8-pose, 17 joints)
# ==============================================================================

KP_NOSE           = 0
KP_LEFT_EYE       = 1
KP_RIGHT_EYE      = 2
KP_LEFT_EAR       = 3
KP_RIGHT_EAR      = 4
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW     = 7
KP_RIGHT_ELBOW    = 8
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10
KP_LEFT_HIP       = 11
KP_RIGHT_HIP      = 12
KP_LEFT_KNEE      = 13
KP_RIGHT_KNEE     = 14
KP_LEFT_ANKLE     = 15
KP_RIGHT_ANKLE    = 16
N_KP              = 17


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class CFG:
    # I/O (overridden by CLI)
    VIDEO_PATH:  str = "aya2.mov"
    BODY_CSV:    str = "raw_body_multi.csv"
    RAW_OUT_CSV: str = "behaviour_raw_frames.csv"
    SUM_OUT_CSV: str = "behaviour_summary.csv"

    # Keypoint confidence gate
    KP_CONF_MIN:   float = 0.30
    KP_INTERP_MAX: int   = 6       # max frames to extrapolate a missing keypoint

    # Kalman filter noise
    KP_KALMAN_Q: float = 3e-4     # process noise (higher = faster response)
    KP_KALMAN_R: float = 8e-3     # measurement noise (higher = smoother)

    # Body ruler fallback when both shoulders invisible
    FALLBACK_SW: float = 100.0    # pixels

    # ══════════════════════════════════════════════════════════════════════
    # CLASSROOM-SPECIFIC DETECTION — no knees required
    # All pixel distances are normalised by shoulder_width (body ruler)
    # so thresholds are camera-distance independent.
    # ══════════════════════════════════════════════════════════════════════

    # ── Desk detection (automatic, per-track) ─────────────────────────────
    DESK_ESTIMATE_WIN:     int   = 60
    DESK_MIN_SAMPLES:      int   = 20

    # ── Behaviour 1 : Sitting ─────────────────────────────────────────────
    SITTING_SHOULDER_ABOVE_DESK_NORM: float = 0.20

    # ── Behaviour 2 : Slouching  (score-based) ────────────────────────────
    # All slouch weights, thresholds, and geometry constants are defined in
    # the "Knee-based posture / Slouching v11" section below (around line 370).
    # They are grouped with the other classroom-calibrated parameters.

    # ── Behaviour 3 : Standing (score-based) ─────────────────────────────
    STANDING_SHOULDER_ABOVE_DESK_NORM: float = 1.20
    STAND_W_SHOULDER_DESK: float = 0.50
    STAND_W_HEAD_DESK:     float = 0.25
    STAND_W_VERT_VEL:      float = 0.05
    STAND_W_BODY_HEIGHT:   float = 0.20
    STAND_VEL_NORM:        float = 0.08
    STAND_BODY_HT_SIT:   float = 0.80
    STAND_BODY_HT_STAND: float = 1.10
    STAND_SCORE_HIGH: float = 0.42
    STAND_SCORE_LOW:  float = 0.20
    STAND_SCORE_EMA:  float = 0.40

    # ── Behaviour 4 : Bouncing (multi-signal, seated oscillation) ────────
    BOUNCE_FFT_WIN:        int   = 32
    BOUNCE_PEAK_THRESH:    float = 0.26
    BOUNCE_MIN_AMP_NORM:   float = 0.025
    BOUNCE_VARIANCE_WIN:   int   = 20
    BOUNCE_VARIANCE_MIN:   float = 1e-4
    BOUNCE_ZCR_WIN:        int   = 16
    BOUNCE_ZCR_MIN:        float = 0.10
    BOUNCE_W_FFT:          float = 0.45
    BOUNCE_W_VARIANCE:     float = 0.25
    BOUNCE_W_ZCR:          float = 0.20
    BOUNCE_W_AMP:          float = 0.10
    BOUNCE_SCORE_HIGH:     float = 0.35
    BOUNCE_SCORE_LOW:      float = 0.15
    BOUNCE_SCORE_EMA:      float = 0.40
    BOUNCE_CONFIRM_FRAMES: int   = 4

    # ── Behaviour 4b: Bounding (full-body displacement — jumping/leaping) ──
    BOUND_DISP_NORM:      float = 0.25
    BOUND_CONFIRM_FRAMES: int   = 3
    BOUND_DECAY_FRAMES:   int   = 8
    BOUND_HISTORY_LEN:    int   = 5

    # ══════════════════════════════════════════════════════════════════════
    # ── Behaviour 4c : FidgetDetector — v9 precision tuning ──────────────
    # ══════════════════════════════════════════════════════════════════════
    #
    # Historique des versions :
    #   v7 → v8 : seuils relevés (VAR_MIN 3e-4→1.5e-3, SCORE_HIGH 0.32→0.45,
    #             CONFIRM 5→12, EMA 0.35→0.25)
    #   v8 → v9 : gate de sortie à deux niveaux, classification de type avec
    #             pondération de visibilité, cohérence inter-joints pour
    #             leg_shake, FFT par région, nouvelles colonnes CSV de debug.

    # ── Fenêtre temporelle ────────────────────────────────────────────────
    # 50 frames (~1.67s à 30fps). Plus long = résolution FFT meilleure,
    # variance de drift plus stable. Le compromis : latence accrue mais
    # on veut de la précision, pas de la réactivité.
    # v7=30, v8=40, v9=50
    FIDGET_WIN:          int   = 50

    # ── Porte de stillness (variance minimale) ────────────────────────────
    # Toujours 1.5e-3 : correspond à ~0.04 sw d'amplitude pic-à-pic,
    # soit ~4% de la largeur épaule. En dessous = bruit Kalman.
    # INCHANGÉ depuis v8.
    FIDGET_VAR_MIN:      float = 1.5e-3

    # ── Plafond de normalisation de variance ──────────────────────────────
    # INCHANGÉ depuis v8.
    FIDGET_VAR_NORM:     float = 1.2e-2

    # ── FFT : peakedness minimum pour signal rythmique ────────────────────
    # Relevé de 0.30 → 0.33 : le pic dominant doit représenter ≥33% de
    # l'énergie totale dans la bande 1–8 Hz.
    # Raison : à 50 frames de fenêtre la résolution FFT est meilleure,
    # donc les pics vrais sont plus nets et les signaux bruités ont des
    # spectres encore plus plats.
    # v7=0.22, v8=0.30, v9=0.33
    FIDGET_FFT_PEAK_MIN: float = 0.33

    # ── ZCR : plage physiologique ────────────────────────────────────────
    # Inchangée par rapport à v8 (0.08–0.50).
    FIDGET_ZCR_MIN:      float = 0.08
    FIDGET_ZCR_MAX:      float = 0.50

    # ── Variance latérale X ───────────────────────────────────────────────
    # Normalisée à 1e-2 (vs 8e-3 en v8) : encore plus stricte.
    # Raison : la composante X des hanches est très bruitée à distance.
    # v8=8e-3, v9=1e-2
    FIDGET_LAT_VAR_NORM: float = 1e-2

    # ── Poids de combinaison ──────────────────────────────────────────────
    # On renforce encore la variance (signal primaire) et on réduit ZCR.
    # La somme doit faire 1.0.
    # v8: VAR=0.45 FFT=0.30 ZCR=0.18 LAT=0.07
    # v9: VAR=0.48 FFT=0.30 ZCR=0.16 LAT=0.06
    FIDGET_W_VAR:        float = 0.48
    FIDGET_W_FFT:        float = 0.30
    FIDGET_W_ZCR:        float = 0.16
    FIDGET_W_LAT:        float = 0.06

    # ── Seuils d'hysteresis ───────────────────────────────────────────────
    # SCORE_HIGH : 0.45 → 0.52. Le score composite doit être fort sur
    # variance + FFT + ZCR simultanément pour déclencher.
    # Cela élimine les situations où seulement la variance est haute
    # (écriture rapide, changement de posture brusque) sans rhythmicité.
    # v7=0.32, v8=0.45, v9=0.52
    FIDGET_SCORE_HIGH:     float = 0.52

    # SCORE_LOW : 0.22 → 0.25. La sortie se fait dès que le score tombe
    # sous 0.25 — légèrement plus strict que v8.
    # v8=0.22, v9=0.25
    FIDGET_SCORE_LOW:      float = 0.25

    # SCORE_FLOOR : seuil de sortie rapide (NOUVEAU en v9).
    # Si le score tombe sous ce plancher absolu, le decay est forcé à
    # FIDGET_DECAY_FAST frames (sortie quasi-immédiate).
    # Rationale : un signal qui s'effondre à < 0.12 signifie que la personne
    # a arrêté de bouger — inutile d'attendre les 18 frames de décroissance.
    FIDGET_SCORE_FLOOR:    float = 0.12

    # EMA : 0.25 → 0.20. Encore plus lente = moins réactive aux transitoires
    # (levée de main, changement de siège).
    # v7=0.35, v8=0.25, v9=0.20
    FIDGET_SCORE_EMA:      float = 0.20

    # ── Confirmation et décroissance ──────────────────────────────────────
    # CONFIRM_FRAMES : 12 → 15 (= 0.5s à 30fps).
    # v7=5, v8=12, v9=15
    FIDGET_CONFIRM_FRAMES: int   = 15

    # DECAY_FRAMES : maintenu à 18 (= 0.6s) pour les sorties normales
    # (score entre SCORE_FLOOR et SCORE_LOW). L'objectif est de conserver
    # la continuité sur un fidgeting intermittent mais réel.
    # INCHANGÉ depuis v8.
    FIDGET_DECAY_FRAMES:   int   = 18

    # DECAY_FAST : 3 frames (= ~0.1s). Sortie rapide déclenchée quand le
    # score tombe sous SCORE_FLOOR. NOUVEAU en v9.
    # Justification : si quelqu'un stoppe net, le score s'effondre sous 0.12
    # en 1–2 frames. L'attendre 18 frames crée des épisodes artificiellement
    # longs. 3 frames = marge minimale anti-glitch.
    FIDGET_DECAY_FAST:     int   = 3

    # EXIT_LOW_FRAMES : nombre de frames CONSÉCUTIVES sous SCORE_LOW
    # requis pour déclencher la sortie via le chemin "decay normal".
    # NOUVEAU en v9 — remplace l'ancienne décrémentation simple.
    # 8 frames consécutives sous 0.25 = ~0.27s. Tolère de brèves remontées.
    FIDGET_EXIT_LOW_FRAMES: int  = 8

    # ── Classification de type ────────────────────────────────────────────
    # Marge relative requise entre la zone dominante et la 2e.
    # Inchangée à 30% (légèrement plus stricte que les 20% de v8).
    # v8=0.20, v9=0.30
    FIDGET_TYPE_MARGIN:    float = 0.30

    # Pénalité de visibilité partielle (NOUVEAU en v9).
    # Une région dont seulement 1 joint sur N est visible reçoit sa variance
    # multipliée par ce facteur (< 1) avant la comparaison de type.
    # Valeur 0.5 = la variance d'un joint unique compte 50% de sa valeur réelle.
    # Raison : évite qu'une cheville partiellement visible batte un poignet
    # clairement observé qui écrit.
    FIDGET_PARTIAL_VIS_PENALTY: float = 0.50

    # Nombre minimum de joints actifs pour valider le type leg_shake
    # (NOUVEAU en v9). Doit avoir ≥ 2 joints jambe/cheville avec variance
    # > VAR_MIN pour que "leg_shake" soit confirmé.
    FIDGET_LEG_MIN_ACTIVE_JOINTS: int = 2

    # ── Knee-based posture (fires ONLY when knees are truly reliable) ────
    #
    # v11 : deux nouveaux filtres de rejet immédiat ajoutés à _knees_reliable() :
    #   1. Confiance cheville brute < ANKLE_RAW_CONF_MIN → rejet immédiat
    #   2. Angle genou > KNEE_IMPOSSIBLE_ANGLE (170°) → extrapolation certaine
    # Ces filtres sont plus agressifs que ceux de v10 et garantissent le
    # basculement en mode desk pour les élèves assis derrière un bureau.
    #
    KNEE_SITTING_MAX:    float = 140.0
    KNEE_SITTING_EXIT:   float = 150.0
    KNEE_STANDING_MIN:   float = 158.0
    KNEE_STANDING_EXIT:  float = 148.0
    KNEE_CONFIRM_FRAMES: int   = 4
    KNEE_PRIORITY:       float = 1.0

    # Confiance brute minimale pour GENOU ET CHEVILLE (inchangé v10).
    KNEE_RAW_CONF_MIN:   float = 0.60

    # Confiance brute minimale spécifique à la CHEVILLE.
    # Les chevilles sont les joints les plus souvent extrapolés sous le bureau.
    # On exige une confiance plus haute que pour le genou.
    # v11 NEW — si ankle_conf < 0.50 → forcer mode desk immédiatement.
    ANKLE_RAW_CONF_MIN:  float = 0.50

    # Angle de genou physiquement impossible pour un être humain debout.
    # Un angle > 170° signifie une jambe quasi-parfaitement droite : c'est
    # typiquement ce que produit l'IA quand elle extrapole une jambe
    # invisible à travers un bureau. Aucun humain ne se tient debout avec
    # un angle genou de 172°.
    # v11 NEW — si avg_knee_angle > KNEE_IMPOSSIBLE_ANGLE → forcer mode desk.
    KNEE_IMPOSSIBLE_ANGLE: float = 170.0

    # Tolérance verticale pour le critère hanche > genou > cheville (v10).
    KNEE_VERT_TOL_NORM:  float = 0.15

    # Marge au-dessus du bureau pour rejeter les genoux "fantômes" (v10).
    KNEE_DESK_MARGIN_NORM: float = 0.30

    # ── Behaviour 2 : Slouching — refonte v11 ─────────────────────────────
    #
    # PROBLÈME v10 : spine_tilt utilisait le vecteur nez→épaules calculé par
    # _tilt_from_vertical(nose, sh_mid). Mais sh_mid dépend des hanches pour
    # le calcul du corps, et quand les hanches sont mal placées par l'IA de
    # pose, spine_tilt ≈ 0 — ce qui annule le signal de tilt.
    #
    # SOLUTION v11 : on remplace TOTALEMENT spine_tilt par deux signaux
    # robustes basés UNIQUEMENT sur le nez et les épaules :
    #
    #   Signal A — HEAD_DESK (maintenu, poids renforcé) :
    #     distance verticale nez→bureau normalisée par sw.
    #     Si la tête descend < HEAD_FLOOR sw au-dessus du bureau → slouching.
    #     Ce signal fonctionne indépendamment des hanches.
    #
    #   Signal B — HEAD_TILT (NOUVEAU) :
    #     angle entre le vecteur nez→milieu-épaules et la VERTICALE descendante.
    #     Calculé avec _tilt_from_vertical(sh_mid, nose) :
    #       sh_mid = top (épaules), nose = bottom → vecteur "tête par rapport épaules"
    #     Un angle > 15° = tête penchée en avant = signe de slouching.
    #     Ce signal NE DÉPEND PAS des hanches.
    #
    #   Signal C — FORWARD_SHIFT (maintenu) :
    #     décalage horizontal nez→épaules.
    #
    # Poids v11 : HEAD_DESK=0.50, HEAD_TILT=0.35, FORWARD_SHIFT=0.15
    # (HEAD_TILT remplace SPINE_TILT avec un poids plus fort car plus fiable)
    SLOUCH_W_HEAD_DESK:      float = 0.50   # v10=0.55, v11=0.50
    SLOUCH_W_HEAD_TILT:      float = 0.35   # NOUVEAU v11 (remplace SPINE_TILT)
    SLOUCH_W_FORWARD_SHIFT:  float = 0.15   # v10=0.20, v11=0.15
    # Note : SLOUCH_W_SPINE_TILT n'est plus utilisé mais conservé pour
    # compatibilité CSV (colonne spine_tilt_deg toujours loguée).
    SLOUCH_W_SPINE_TILT:     float = 0.0    # désactivé en v11

    # HEAD_TILT : seuils d'angle tête→épaules (vecteur sh_mid→nose)
    # 0° = tête parfaitement au-dessus des épaules (posture droite)
    # 20° = légère inclinaison vers l'avant
    # 40°+ = clairement penché sur le bureau
    SLOUCH_HEAD_TILT_CLEAR:  float = 15.0   # deg — pas de contribution en dessous
    SLOUCH_HEAD_TILT_MAX:    float = 50.0   # deg — contribution maximale

    SLOUCH_HEAD_FLOOR:       float = 0.35   # sw — tête à cette distance du bureau = score max
    SLOUCH_FORWARD_SHIFT_CLEAR: float = 0.10
    SLOUCH_FORWARD_SHIFT_MAX:   float = 0.45

    # Hysteresis slouch — confirmé v10 avec seuils assouplis pour salle de classe
    SLOUCH_SCORE_HIGH: float = 0.35   # v9=0.40, v10=0.35
    SLOUCH_SCORE_LOW:  float = 0.18   # v9=0.20, v10=0.18
    SLOUCH_SCORE_EMA:  float = 0.30
    SLOUCH_HEAD_CLEAR: float = 1.00   # v9=1.20, v10=1.00

    # Exit frames pour le slouching (stabilité v11)
    # 30 frames = ~1s à 30fps avant de quitter "slouching" → étiquette stable
    SLOUCH_EXIT_LOW_FRAMES:  int   = 30   # v11 NEW — remplace le compteur ≥5

    # ── DECAY_FRAMES global — stabilité v11 ───────────────────────────────
    # DEMANDE UTILISATEUR : augmenter DECAY_FRAMES à 30 pour tous les
    # comportements pour éviter le clignotement des étiquettes.
    # ~1s à 30fps : une étiquette confirmée reste active au moins 1s après
    # que le signal tombe sous le seuil.
    # Affecte : FIDGET_DECAY_FRAMES, BOUND_DECAY_FRAMES, BOUNCE confirm
    # (Les compteurs de décroissance dans PostureStateMachine sont aussi
    # augmentés via SLOUCH_EXIT_LOW_FRAMES et STAND_EXIT_LOW_FRAMES.)
    FIDGET_DECAY_FRAMES:    int   = 30   # v10=18, v11=30
    BOUND_DECAY_FRAMES:     int   = 30   # v10=8,  v11=30
    STAND_EXIT_LOW_FRAMES:  int   = 30   # v11 NEW — frames consécutives sous LOW pour quitter "standing"

    # ══════════════════════════════════════════════════════════════════════
    # ── v11.2 NEW PARAMETERS ─────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════

    # ── Fix 1 : Slouching — signal HEAD_BETWEEN_SHOULDERS ────────────────
    #
    # PROBLÈME v11 : le signal HEAD_TILT (angle épaules→nez) ne capte pas
    # la position "tête enfoncée entre les épaules" (fatigue, écriture intense).
    # Dans cette posture, le nez peut descendre AU NIVEAU ou EN DESSOUS de
    # la ligne imaginaire des épaules. L'angle HEAD_TILT serait alors faible
    # (tête vers l'avant) mais les valeurs restent incertaines.
    #
    # SOLUTION : signal géométrique direct.
    # Si nose.y >= sh_mid.y - THRESHOLD × sw → tête "entre" ou sous les épaules.
    # THRESHOLD = 0.0 signifie tête exactement au niveau des épaules.
    # THRESHOLD = 0.2 signifie tête à 20% de sw AU-DESSUS de la ligne épaules.
    # → Score plein (1.0) quand nose.y >= sh_mid.y (nez au niveau des épaules)
    # → Score nul quand nose.y <= sh_mid.y - THRESHOLD × sw
    # Ce signal est indépendant du bureau et des hanches.
    SLOUCH_HEAD_BETWEEN_SHOULDERS_THRESHOLD: float = 0.25
    # sw — si nez < sh_mid.y − 0.25×sw : contribution nulle (tête haute)
    # si nez ≥ sh_mid.y              : contribution maximale (1.0)
    # Poids de ce nouveau signal dans le score slouch combiné :
    SLOUCH_W_HEAD_BETWEEN:   float = 0.40   # fort : signal géométrique direct
    # Les poids v11 sont redistribués pour accueillir le nouveau signal :
    # HEAD_DESK=0.30, HEAD_TILT=0.15, FORWARD_SHIFT=0.10, HEAD_BETWEEN=0.40
    # Note : HEAD_BETWEEN remplace une grande partie du poids HEAD_DESK
    # car il est plus discriminant (géométrie pure sans dépendre du bureau).
    # Les 4 constantes SLOUCH_W_* ci-dessous REMPLACENT celles définies plus
    # haut dans la section v11 (Python gardera la dernière valeur).

    # ── Fix 2 : Body Rocking — persistance et fenêtre glissante ──────────
    #
    # PROBLÈME : body_rocking se déclenchait sur 2 frames isolées (pic de
    # variance transitoire, changement de position brusque).
    # SOLUTION :
    #   a) Compteur de persistance : ROCKING_PERSIST_FRAMES (18) frames
    #      consécutives où hip_wv > seuil avant de confirmer le rocking.
    #   b) Fenêtre glissante de variance latérale (_rocking_stability_buf,
    #      longueur ROCKING_STABILITY_WIN = 20 frames). La confirmation
    #      exige que la variance dans cette fenêtre soit STABLE (std/mean < 0.6)
    #      et non impulsionnelle. Un pic isolé a une grande déviation relative.
    ROCKING_PERSIST_FRAMES:   int   = 18    # ~0.6s à 30fps avant confirmation
    ROCKING_STABILITY_WIN:    int   = 20    # fenêtre pour test de stabilité
    ROCKING_STABILITY_MAX_CV: float = 0.60  # coefficient de variation max (std/mean)
    # Si CV > 0.60, le signal est impulsionnel (changement de position) → rejet.

    # ── Fix 3 : Fidgeting poignet — gate ZCR haute fréquence ─────────────
    #
    # PROBLÈME : les mouvements intentionnels (écriture, frappe au clavier)
    # produisent une haute variance des poignets mais à fréquence variable et
    # non-rythmique, ce qui pollue le fidgeting.
    # SOLUTION :
    #   a) Si hand_raised → mettre wrist_wv = 0 (ignorer poignets).
    #   b) Le type "hand_movement" n'est validé que si s_zcr > WRIST_ZCR_MIN.
    #      Un ZCR faible = mouvement directionnel (écriture) pas répétitif.
    #      Un ZCR élevé = mouvement oscillant rapide = vrai fidgeting nerveux.
    FIDGET_WRIST_ZCR_MIN: float = 0.15  # ZCR minimum pour valider hand_movement

    # ── Redistribution poids slouch v11.2 (somme = 1.0) ──────────────────
    # Écrase les valeurs définies dans la section slouch v11 plus haut.
    # HEAD_BETWEEN (0.40) + HEAD_DESK (0.30) + HEAD_TILT (0.20) +
    # FORWARD_SHIFT (0.10) + SPINE_TILT (0.00) = 1.00
    SLOUCH_W_HEAD_DESK:      float = 0.30   # v11=0.50 → v11.2=0.30
    SLOUCH_W_HEAD_TILT:      float = 0.20   # v11=0.35 → v11.2=0.20
    SLOUCH_W_FORWARD_SHIFT:  float = 0.10   # v11=0.15 → v11.2=0.10
    SLOUCH_W_SPINE_TILT:     float = 0.0    # désactivé (inchangé)

    # ── Behaviour 5 : Hand raise  (score-based, hysteresis) ──────────────────
    HAND_W_WRIST_VS_SHOULDER: float = 0.38
    HAND_W_WRIST_VS_NOSE:     float = 0.30
    HAND_W_ELBOW_VS_SHOULDER: float = 0.17
    HAND_W_UPWARD_VEL:        float = 0.05
    HAND_W_ELBOW_ANGLE:       float = 0.10
    HAND_HEIGHT_NORM:  float = 1.5
    HAND_VEL_NORM:     float = 0.12
    HAND_SCORE_HIGH:   float = 0.38
    HAND_SCORE_LOW:    float = 0.16
    HAND_SCORE_EMA:    float = 0.45
    HAND_RAISE_HOLD:   int   = 2
    HAND_LOWER_HOLD:   int   = 8
    HAND_ELBOW_RAISE_ANGLE_MIN: float = 100.0

    # ── Display ────────────────────────────────────────────────────────────
    WINDOW_NAME: str = "Behaviour Classifier v11.2  —  slouch+rocking+fidget fixes"

    # ── Spatial ID Mapper — 2 élèves fixes Gauche/Droite ─────────────────
    # L'image est divisée en deux zones par la ligne verticale X = frame_w / 2.
    # Toute détection dont le centre X < midpoint → ID forcé = STUDENT_ID_LEFT
    # Toute détection dont le centre X ≥ midpoint → ID forcé = STUDENT_ID_RIGHT
    # Toute détection hors de ces deux IDs (3e ID parasite) est IGNORÉE.
    STUDENT_ID_LEFT:  int = 1   # Élève côté gauche de l'image
    STUDENT_ID_RIGHT: int = 2   # Élève côté droit de l'image
    # Noms affichés dans les rapports
    STUDENT_LABEL:    dict = {1: "Left  (ID:1)", 2: "Right (ID:2)"}

    # Behaviour colour map (BGR)
    # v11 : labels fidget mis à jour (body_rocking, hand_movement)
    BEHAVIOUR_COLORS: dict = {
        "sitting":                  (0,   200, 255),
        "standing":                 (0,   220,  80),
        "slouching":                (50,   50, 255),   # rouge-orange — problème posture
        "bouncing":                 (255, 140,   0),
        "bounding":                 (0,  230, 255),
        # Fidgeting sous-types — couleurs distinctes par membre
        "fidgeting":                (0,  100, 255),    # fallback orange
        "fidgeting:leg_shake":      (0,   40, 255),    # orange intense — jambe
        "fidgeting:body_rocking":   (0,  160, 180),    # turquoise — corps
        "fidgeting:hand_movement":  (0,  200, 140),    # vert-jaune — main
        "fidgeting:generic":        (0,  100, 255),    # orange standard
        "hand_raised":              (255,   0, 200),
        "unknown":                  (140, 140, 140),
    }

    DESK_LINE_COLOR: tuple = (0, 255, 220)

    ID_PALETTE = [
        (0, 200, 255), (0, 255, 128), (255, 128, 0),
        (200, 0, 255), (0, 128, 255), (255, 0, 128),
        (128, 255, 0), (255, 200, 0), (0, 255, 220),
    ]


# ==============================================================================
# 2-D KALMAN FILTER  (one per keypoint)
# ==============================================================================

class KalmanKP:
    """
    State = [x, vx, y, vy]  (constant-velocity model).
    Smooths detection jitter; extrapolates through brief occlusions via
    predict_only() which advances the filter without a measurement.
    """

    def __init__(self) -> None:
        self.x = np.zeros(4, dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 10.0
        dt = 1.0
        self.F = np.array([
            [1, dt, 0,  0],
            [0,  1, 0,  0],
            [0,  0, 1, dt],
            [0,  0, 0,  1],
        ], dtype=np.float64)
        self.H  = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
        q       = CFG.KP_KALMAN_Q
        self.Q  = np.diag([q, q * 10, q, q * 10])
        self.R  = np.eye(2) * CFG.KP_KALMAN_R
        self._init          = False
        self.missing_frames = 0

    def predict_only(self) -> tuple[float, float]:
        if not self._init:
            return 0.0, 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.missing_frames += 1
        return float(self.x[0]), float(self.x[2])

    def update(self, mx: float, my: float) -> tuple[float, float]:
        if not self._init:
            self.x[:] = [mx, 0.0, my, 0.0]
            self._init          = True
            self.missing_frames = 0
            return mx, my
        xp  = self.F @ self.x
        Pp  = self.F @ self.P @ self.F.T + self.Q
        z   = np.array([[mx], [my]], dtype=np.float64)
        S   = self.H @ Pp @ self.H.T + self.R
        K   = Pp @ self.H.T @ np.linalg.inv(S)
        self.x = (xp.reshape(-1,1) + K @ (z - self.H @ xp.reshape(-1,1))).flatten()
        self.P = (np.eye(4) - K @ self.H) @ Pp
        self.missing_frames = 0
        return float(self.x[0]), float(self.x[2])

    @property
    def vy_up(self) -> float:
        return -float(self.x[3])


# ==============================================================================
# KEYPOINT BUFFER  (17 KalmanKP + confidence gate + interpolation)
# ==============================================================================

class KeypointBuffer:
    def __init__(self) -> None:
        self.kf     = [KalmanKP() for _ in range(N_KP)]
        self.smooth = np.full((N_KP, 2), np.nan, dtype=np.float64)
        self.valid  = np.zeros(N_KP, dtype=bool)

    def update(self, kp_array: np.ndarray) -> None:
        for i in range(N_KP):
            x_raw = float(kp_array[i, 0])
            y_raw = float(kp_array[i, 1])
            conf  = float(kp_array[i, 2]) if kp_array.shape[1] > 2 else 1.0

            missing = (
                conf < CFG.KP_CONF_MIN
                or math.isnan(x_raw)
                or (x_raw == 0.0 and y_raw == 0.0)
            )

            kf = self.kf[i]
            if not missing:
                sx, sy = kf.update(x_raw, y_raw)
                self.smooth[i] = [sx, sy]
                self.valid[i]  = True
            elif kf._init and kf.missing_frames < CFG.KP_INTERP_MAX:
                sx, sy = kf.predict_only()
                self.smooth[i] = [sx, sy]
                self.valid[i]  = True
            else:
                if kf._init:
                    kf.missing_frames += 1
                self.smooth[i] = [np.nan, np.nan]
                self.valid[i]  = False

    def get(self, idx: int) -> Optional[tuple[float, float]]:
        if self.valid[idx]:
            return float(self.smooth[idx, 0]), float(self.smooth[idx, 1])
        return None

    def kpf(self, idx: int) -> KalmanKP:
        return self.kf[idx]


# ==============================================================================
# GEOMETRY HELPERS
# ==============================================================================

def _tilt_from_vertical(top: tuple, bottom: tuple) -> float:
    dx, dy = bottom[0] - top[0], bottom[1] - top[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return 0.0
    return float(math.degrees(math.acos(max(-1.0, min(1.0, dy / length)))))


def _body_ruler(kpb: KeypointBuffer) -> float:
    ls = kpb.get(KP_LEFT_SHOULDER)
    rs = kpb.get(KP_RIGHT_SHOULDER)
    if ls and rs:
        sw = math.hypot(rs[0]-ls[0], rs[1]-ls[1])
        if sw > 5.0:
            return sw

    for sh_idx, hp_idx in [
        (KP_LEFT_SHOULDER,  KP_LEFT_HIP),
        (KP_RIGHT_SHOULDER, KP_RIGHT_HIP),
    ]:
        sh = kpb.get(sh_idx)
        hp = kpb.get(hp_idx)
        if sh and hp:
            torso = math.hypot(hp[0]-sh[0], hp[1]-sh[1])
            if torso > 5.0:
                return torso * 0.8

    return CFG.FALLBACK_SW


def _hip_centroid(kpb: KeypointBuffer) -> Optional[tuple[float, float]]:
    lh = kpb.get(KP_LEFT_HIP)
    rh = kpb.get(KP_RIGHT_HIP)
    if lh and rh:
        return (lh[0]+rh[0])/2.0, (lh[1]+rh[1])/2.0
    return lh or rh


def _knees_reliable(
    kpb: KeypointBuffer,
    kp_array_raw: np.ndarray,
    desk_y: float,
) -> bool:
    """
    Retourne True SEULEMENT si au moins une chaîne jambe complète (hanche-genou-
    cheville) est RÉELLEMENT observée — pas extrapolée ou derrière le bureau.

    v11 : DEUX FILTRES DE REJET IMMÉDIAT ajoutés en tête :
      A) Confiance cheville brute < ANKLE_RAW_CONF_MIN (0.50) → rejet immédiat.
         Les chevilles sont le joint le plus souvent extrapolé sous le bureau.
      B) Angle de genou > KNEE_IMPOSSIBLE_ANGLE (170°) → rejet immédiat.
         Un angle > 170° est physiquement impossible pour un humain debout.
         C'est la signature exacte d'une jambe extrapolée à travers un bureau.
         Ces deux filtres garantissent le passage en mode desk pour un élève assis.

    CRITÈRES COMPLETS (chain evaluation, tous doivent passer pour 1 jambe) :

    0. FILTRE ANGLE IMPOSSIBLE (v11 NEW) — vérifié en premier
       Si l'angle hip-knee-ankle > KNEE_IMPOSSIBLE_ANGLE : rejet.

    1. CONFIANCE BRUTE GENOU ≥ KNEE_RAW_CONF_MIN ET CHEVILLE ≥ ANKLE_RAW_CONF_MIN

    2. COHÉRENCE ANATOMIQUE VERTICALE : hip_y < knee_y + tol ET knee_y < ankle_y + tol

    3. POSITION RELATIVE AU BUREAU : genou pas au-dessus de desk_y − margin

    Note : si desk_y n'est pas encore estimé, seuls les critères 0–2 s'appliquent.
    """
    desk_ready = not math.isnan(desk_y)
    sw_approx  = _body_ruler(kpb)

    for h_idx, k_idx, a_idx in [
        (KP_LEFT_HIP,  KP_LEFT_KNEE,  KP_LEFT_ANKLE),
        (KP_RIGHT_HIP, KP_RIGHT_KNEE, KP_RIGHT_ANKLE),
    ]:
        # ── Confiance brute genou et cheville ─────────────────────────────
        k_conf = float(kp_array_raw[k_idx, 2]) if kp_array_raw.shape[1] > 2 else 0.0
        a_conf = float(kp_array_raw[a_idx, 2]) if kp_array_raw.shape[1] > 2 else 0.0

        # Critère 1 : confiance suffisante (genou ET cheville)
        if k_conf < CFG.KNEE_RAW_CONF_MIN or a_conf < CFG.ANKLE_RAW_CONF_MIN:
            continue

        # ── Positions lissées ──────────────────────────────────────────────
        h = kpb.get(h_idx)
        k = kpb.get(k_idx)
        a = kpb.get(a_idx)
        if h is None or k is None or a is None:
            continue

        # ── Critère 0 : angle genou physiquement impossible (v11 NEW) ─────
        # Un angle > 170° = jambe quasi-droite = extrapolation derrière bureau.
        # On calcule l'angle hip-knee-ankle directement ici.
        knee_ang = _angle_deg(h, k, a)
        if knee_ang > CFG.KNEE_IMPOSSIBLE_ANGLE:
            continue   # jambe extrapolée — forcer mode desk

        # ── Critère 2 : ordre vertical anatomique ─────────────────────────
        vert_tol = CFG.KNEE_VERT_TOL_NORM * sw_approx
        if not (h[1] < k[1] + vert_tol and k[1] < a[1] + vert_tol):
            continue

        # ── Critère 3 : genou pas derrière le bureau ──────────────────────
        if desk_ready:
            margin = CFG.KNEE_DESK_MARGIN_NORM * sw_approx
            if k[1] < desk_y - margin:
                continue

        return True   # tous les critères passés pour cette jambe

    return False


def _angle_deg(A: tuple, B: tuple, C: tuple) -> float:
    ba = np.array([A[0]-B[0], A[1]-B[1]], dtype=np.float64)
    bc = np.array([C[0]-B[0], C[1]-B[1]], dtype=np.float64)
    na, nc = np.linalg.norm(ba), np.linalg.norm(bc)
    if na < 1e-6 or nc < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0))))


def _ema(prev: float, new_val: float, alpha: float) -> float:
    if math.isnan(prev):
        return new_val
    if math.isnan(new_val):
        return prev
    return alpha * new_val + (1.0 - alpha) * prev


# ==============================================================================
# DESK ESTIMATOR  (per-track, automatic)
# ==============================================================================

class DeskEstimator:
    def __init__(self) -> None:
        self._obs_buf:    deque = deque(maxlen=CFG.DESK_ESTIMATE_WIN)
        self.desk_y:      float = float("nan")
        self._init_max:   float = float("nan")
        self._n_samples:  int   = 0

    def update(self, kpb: KeypointBuffer, sw: float, hand_raised: bool) -> None:
        if hand_raised:
            return

        candidates: list[float] = []
        for idx in (KP_LEFT_WRIST, KP_RIGHT_WRIST,
                    KP_LEFT_ELBOW, KP_RIGHT_ELBOW):
            pt = kpb.get(idx)
            if pt:
                candidates.append(pt[1])

        if not candidates:
            return

        sample = max(candidates)

        if not math.isnan(self.desk_y):
            if abs(sample - self.desk_y) > 1.5 * sw:
                return

        if math.isnan(self._init_max) or sample > self._init_max:
            self._init_max = sample

        self._obs_buf.append(sample)
        self._n_samples += 1

        if self._n_samples >= CFG.DESK_MIN_SAMPLES:
            self.desk_y = float(np.median(list(self._obs_buf)))
        else:
            self.desk_y = self._init_max

    @property
    def is_ready(self) -> bool:
        return not math.isnan(self.desk_y) and self._n_samples >= 3


# ==============================================================================
# BEHAVIOUR 1, 2, 3 — CLASSROOM POSTURE CLASSIFIER  (score-based)
# ==============================================================================

@dataclass
class PostureResult:
    label:              str   = "unknown"
    spine_tilt_deg:     float = float("nan")
    head_desk_norm:     float = float("nan")
    shoulder_desk_norm: float = float("nan")
    forward_shift_norm: float = float("nan")
    body_height_norm:   float = float("nan")
    slouch_score:       float = 0.0
    stand_score:        float = 0.0
    confidence:         float = 0.0


# ==============================================================================
# KNEE-BASED POSTURE CLASSIFIER
# ==============================================================================

@dataclass
class KneePostureResult:
    label:            str   = "unknown"
    knee_angle_left:  float = float("nan")
    knee_angle_right: float = float("nan")
    avg_knee_angle:   float = float("nan")
    confidence:       float = 0.0


class KneePostureClassifier:
    def __init__(self) -> None:
        self._state        = "unknown"
        self._candidate    = "unknown"
        self._hold_counter = 0

    def classify(self, kpb: KeypointBuffer) -> KneePostureResult:
        res = KneePostureResult()

        angles: list[float] = []
        for h_idx, k_idx, a_idx in [
            (KP_LEFT_HIP,  KP_LEFT_KNEE,  KP_LEFT_ANKLE),
            (KP_RIGHT_HIP, KP_RIGHT_KNEE, KP_RIGHT_ANKLE),
        ]:
            h = kpb.get(h_idx); k = kpb.get(k_idx); a = kpb.get(a_idx)
            if h and k and a:
                ang = _angle_deg(h, k, a)
                if not math.isnan(ang):
                    angles.append(ang)

        if not angles:
            return res

        res.knee_angle_left  = angles[0] if len(angles) >= 1 else float("nan")
        res.knee_angle_right = angles[1] if len(angles) >= 2 else float("nan")
        min_angle = float(np.min(angles))
        res.avg_knee_angle = min_angle

        if min_angle < CFG.KNEE_SITTING_MAX:
            raw = "sitting"
        elif min_angle > CFG.KNEE_STANDING_MIN:
            raw = "standing"
        else:
            raw = self._state if self._state != "unknown" else "sitting"

        if self._state == "sitting" and min_angle > CFG.KNEE_SITTING_EXIT:
            raw = "standing" if min_angle > CFG.KNEE_STANDING_MIN else self._state
        if self._state == "standing" and min_angle < CFG.KNEE_STANDING_EXIT:
            raw = "sitting" if min_angle < CFG.KNEE_SITTING_MAX else self._state

        if raw == self._state:
            self._hold_counter = 0
        else:
            if raw == self._candidate:
                self._hold_counter += 1
            else:
                self._candidate    = raw
                self._hold_counter = 1
            if self._hold_counter >= CFG.KNEE_CONFIRM_FRAMES:
                self._state        = raw
                self._hold_counter = 0

        res.label = self._state

        if res.label == "sitting":
            res.confidence = min(1.0,
                (CFG.KNEE_SITTING_MAX - min_angle) / 35.0)
        elif res.label == "standing":
            res.confidence = min(1.0,
                (min_angle - CFG.KNEE_STANDING_MIN) / 15.0)
        else:
            res.confidence = 0.0

        return res


# ==============================================================================
# BOUNDING DETECTOR  (large hip displacement — jumping / leaping)
# ==============================================================================

class BoundingDetector:
    def __init__(self) -> None:
        self._prev_hip:    Optional[tuple[float, float]] = None
        self._disp_buf:    deque = deque(maxlen=CFG.BOUND_HISTORY_LEN)
        self._decay_count: int   = 0

    def update(self, kpb: KeypointBuffer, sw: float) -> tuple[float, int]:
        hip = _hip_centroid(kpb)
        if hip is None:
            self._decay_count = max(0, self._decay_count - 1)
            return 0.0, int(self._decay_count > 0)
        if self._prev_hip is not None:
            disp = math.hypot(
                hip[0] - self._prev_hip[0],
                hip[1] - self._prev_hip[1]
            ) / max(sw, 1.0)
        else:
            disp = 0.0
        self._prev_hip = hip
        self._disp_buf.append(disp)
        smooth = float(np.max(self._disp_buf)) if self._disp_buf else 0.0
        score  = float(np.clip(smooth / (CFG.BOUND_DISP_NORM * 2.0), 0.0, 1.0))
        if smooth >= CFG.BOUND_DISP_NORM:
            self._decay_count = CFG.BOUND_DECAY_FRAMES
        else:
            self._decay_count = max(0, self._decay_count - 1)
        return score, int(self._decay_count > 0)


# ==============================================================================
# FIDGET DETECTOR  (repetitive motion — leg shaking, rocking, wrist fidgeting)
# v9 : two-tier exit gate, visibility-weighted type classification,
#      leg_shake inter-joint check, per-region FFT, debug CSV columns
# ==============================================================================

class FidgetDetector:
    """
    Détecte les MOUVEMENTS RÉPÉTITIFS DE FAIBLE AMPLITUDE (fidgeting) à l'aide
    de quatre signaux complémentaires calculés sur une fenêtre glissante.

    ARCHITECTURE GÉNÉRALE
    ─────────────────────
    Quatre signaux composites → weighted sum → EMA → two-tier hysteresis gate
    → label confirmé.

    Signaux :
      1. Variance Y détreddée du joint le plus actif (poids 0.48)
      2. Peakedness FFT sur la bande 1–8 Hz du même joint (poids 0.30)
      3. Zero-crossing rate du même joint (poids 0.16)
      4. Variance X latérale hanches+épaules (poids 0.06)

    NOUVEAUTÉS v9
    ─────────────
    A. Gate de sortie à deux niveaux
       ─────────────────────────────
       Niveau 1 – sortie rapide (SCORE_FLOOR) :
         Si score < FIDGET_SCORE_FLOOR (0.12), decay forcé à FIDGET_DECAY_FAST
         (3 frames). La personne a clairement arrêté de bouger.
       Niveau 2 – sortie normale :
         EXIT_LOW_FRAMES (8) frames consécutives sous SCORE_LOW (0.25) avant
         de sortir. Tolère les micro-pauses dans un fidgeting continu.
       Résultat : les épisodes s'arrêtent dès que le mouvement cesse, sans
       attendre les 18 frames de v8 qui créaient des épisodes artificiels.

    B. Classification de type avec pondération de visibilité
       ──────────────────────────────────────────────────────
       Chaque région reçoit un score = variance_brute × visibilité_ratio.
       visibilité_ratio = n_joints_visibles / n_joints_total_région.
       Si ratio < 0.5 → pénalité additionnelle FIDGET_PARTIAL_VIS_PENALTY.
       Exemple : 1 cheville visible sur 4 joints jambe → ratio=0.25,
       variance multipliée par 0.25 × 0.50 = 0.125.
       Cela empêche qu'une cheville partiellement visible batte un poignet
       clairement observable (étudiant en train d'écrire).

    C. Cohérence inter-joints pour leg_shake
       ───────────────────────────────────────
       Pour valider "leg_shake", il faut désormais que ≥ FIDGET_LEG_MIN_ACTIVE_JOINTS
       (= 2) joints inférieurs (chevillles + genoux) aient une variance
       individuelle > FIDGET_VAR_MIN. Un seul joint actif = artefact possible.
       Si la condition n'est pas remplie, le type tombe à "generic".

    D. FFT par région pour la classification
       ──────────────────────────────────────
       On calcule le score FFT du meilleur joint de chaque région, et on
       l'utilise comme multiplicateur booléen sur le score de variance de
       la région. Une région avec un pic FFT clair est préférée à une région
       avec une haute variance mais un spectre plat (écriture = haute variance,
       spectre plat).

    E. Colonnes de debug CSV exposées
       ────────────────────────────────
       lower_var_debug, hip_var_debug, wrist_var_debug, fft_score_debug
       sont stockées comme attributs publics, lues par TrackState.process()
       et écrites dans behaviour_raw_frames.csv.
    """

    # Keypoints par région
    _LOWER_KPS   = (KP_LEFT_ANKLE, KP_RIGHT_ANKLE, KP_LEFT_KNEE, KP_RIGHT_KNEE)
    _HIP_KPS     = (KP_LEFT_HIP,   KP_RIGHT_HIP)
    _WRIST_KPS   = (KP_LEFT_WRIST, KP_RIGHT_WRIST)
    _ALL_KPS     = _LOWER_KPS + _HIP_KPS + _WRIST_KPS
    _LATERAL_KPS = (KP_LEFT_HIP, KP_RIGHT_HIP,
                    KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER)

    def __init__(self) -> None:
        # Historiques Y normalisés (position / sw) par joint
        self._y_hist: dict[int, deque] = {
            kp: deque(maxlen=CFG.FIDGET_WIN) for kp in self._ALL_KPS
        }
        # Historiques X normalisés pour composante latérale
        self._x_hist: dict[int, deque] = {
            kp: deque(maxlen=CFG.FIDGET_WIN) for kp in self._LATERAL_KPS
        }

        # Score EMA
        self._score_ema:    float = 0.0
        # Compteur de confirmation (entrée)
        self._high_cnt:     int   = 0
        # Décroissance (sortie)
        self._decay_count:  int   = 0
        # Compteur de frames consécutives sous SCORE_LOW (sortie niveau 2)
        self._low_cnt:      int   = 0
        # État confirmé
        self._fidgeting:    bool  = False

        # ── v11.2 : Persistance body_rocking ──────────────────────────────
        # Compteur de frames consécutives où body_rocking est le type candidat.
        # Seulement quand ce compteur atteint ROCKING_PERSIST_FRAMES, le type
        # "body_rocking" est confirmé et gardé dans fidget_type.
        self._rocking_persist_cnt: int   = 0
        # Fenêtre glissante des valeurs hip_wv pour test de stabilité.
        # Un mouvement impulsionnel a une grande variance dans cette fenêtre.
        # Un vrai balancement est stable (valeurs proches).
        self._rocking_stability_buf: deque = deque(
            maxlen=CFG.ROCKING_STABILITY_WIN
        )
        # Dernier type rocking confirmé (préservé pendant la décroissance)
        self._rocking_confirmed: bool = False

        # ── Attributs publics — valeurs courantes ──────────────────────────
        self.fidget_score:      float = 0.0
        self.fidget_type:       str   = "none"
        self.best_joint_var:    float = 0.0
        # Colonnes de debug CSV (v9)
        self.lower_var_debug:   float = 0.0
        self.hip_var_debug:     float = 0.0
        self.wrist_var_debug:   float = 0.0
        self.fft_score_debug:   float = 0.0

    # ──────────────────────────────────────────────────────────────────────
    # Helpers internes
    # ──────────────────────────────────────────────────────────────────────

    def _detrend(self, arr: np.ndarray) -> np.ndarray:
        """
        Supprime la tendance linéaire (dérive posturale lente) par régression
        moindres-carrés.  Repli sur soustraction de la moyenne si n < 4.
        """
        n = len(arr)
        if n < 4:
            return arr - arr.mean()
        x = np.arange(n, dtype=np.float64)
        return arr - np.polyval(np.polyfit(x, arr, 1), x)

    def _raw_var(self, kp: int) -> float:
        """Variance détreddée brute (non normalisée) du joint kp."""
        h = self._y_hist[kp]
        min_hist = max(12, CFG.FIDGET_WIN // 2)
        if len(h) < min_hist:
            return 0.0
        return float(np.var(self._detrend(np.array(h, dtype=np.float64))))

    def _variance_score(self, sig: np.ndarray) -> float:
        """Variance normalisée → [0, 1].  Sous VAR_MIN → 0."""
        d   = self._detrend(sig)
        var = float(np.var(d))
        if var < CFG.FIDGET_VAR_MIN:
            return 0.0
        return float(np.clip(
            (var - CFG.FIDGET_VAR_MIN)
            / max(CFG.FIDGET_VAR_NORM - CFG.FIDGET_VAR_MIN, 1e-9),
            0.0, 1.0
        ))

    def _fft_score(self, sig: np.ndarray, fps: float = 30.0) -> float:
        """
        Peakedness FFT dans la bande physiologique 1–8 Hz → [0, 1].
        Retourne 0 si signal trop court, aucun pic dans la bande, ou
        peakedness < FIDGET_FFT_PEAK_MIN.
        Normalisation plafonnée à 0.75 (conservateur) pour v9.
        """
        n = len(sig)
        if n < 8:
            return 0.0
        d     = self._detrend(sig)
        spec  = np.abs(np.fft.rfft(d))[1:]          # skip DC
        freqs = np.arange(1, len(spec) + 1) * fps / n
        mask  = (freqs >= 1.0) & (freqs <= 8.0)
        if not mask.any():
            return 0.0
        total = float(spec.sum()) + 1e-9
        peak  = float(spec[mask].max())
        pkness = peak / total
        if pkness < CFG.FIDGET_FFT_PEAK_MIN:
            return 0.0
        return float(np.clip(pkness / 0.75, 0.0, 1.0))

    def _zcr_score(self, sig: np.ndarray) -> float:
        """
        Taux de passages à zéro du signal détreddé → [0, 1].
        Plage physiologique : [FIDGET_ZCR_MIN, FIDGET_ZCR_MAX].
        Score maximal au centre de la plage.
        """
        if len(sig) < 4:
            return 0.0
        d       = self._detrend(sig)
        n_cross = int(np.sum(np.diff(np.sign(d)) != 0))
        zcr     = n_cross / max(len(d) - 1, 1)
        if zcr < CFG.FIDGET_ZCR_MIN or zcr > CFG.FIDGET_ZCR_MAX:
            return 0.0
        mid = (CFG.FIDGET_ZCR_MIN + CFG.FIDGET_ZCR_MAX) / 2.0
        return float(np.clip(
            1.0 - abs(zcr - mid) / (mid - CFG.FIDGET_ZCR_MIN + 1e-9),
            0.0, 1.0
        ))

    def _vis_weighted_var(
        self,
        kp_list:     tuple,
        joint_var:   dict[int, float],
        kpb_valid:   set[int],
    ) -> tuple[float, float]:
        """
        Retourne (variance_pondérée, fft_score_meilleur_joint) pour une région.

        Pondération de visibilité (v9) :
          ratio = n_joints_visibles / n_joints_total
          Si ratio < 0.5 : variance multipliée par FIDGET_PARTIAL_VIS_PENALTY
          Sinon           : variance multipliée par ratio (proportionnel)

        FFT calculée sur le meilleur joint visible de la région.
        """
        n_total   = len(kp_list)
        n_visible = sum(1 for kp in kp_list if kp in kpb_valid)
        if n_visible == 0:
            return 0.0, 0.0

        # Variance brute max de la région
        vars_in_region = [joint_var.get(kp, 0.0) for kp in kp_list]
        max_var = max(vars_in_region)
        if max_var == 0.0:
            return 0.0, 0.0

        # Ratio de visibilité
        vis_ratio = n_visible / n_total
        if vis_ratio < 0.5:
            # Pénalité pour visibilité partielle
            weighted = max_var * vis_ratio * CFG.FIDGET_PARTIAL_VIS_PENALTY
        else:
            weighted = max_var * vis_ratio

        # FFT sur le meilleur joint visible de cette région
        best_kp_in_region = max(
            (kp for kp in kp_list if kp in kpb_valid),
            key=lambda kp: joint_var.get(kp, 0.0),
            default=None,
        )
        fft_s = 0.0
        if best_kp_in_region is not None:
            h = self._y_hist[best_kp_in_region]
            min_hist = max(12, CFG.FIDGET_WIN // 2)
            if len(h) >= min_hist:
                fft_s = self._fft_score(np.array(h, dtype=np.float64))

        return weighted, fft_s

    def _classify_type(
        self,
        joint_var:   dict[int, float],
        s_lat:       float,
        raw_score:   float,
        kpb_valid:   set[int],
        hand_raised: bool,
        s_zcr:       float,
    ) -> str:
        """
        Classification du type de fidgeting — v11.2.

        NOUVEAUTÉS v11.2 :
        1. Compteur de persistance body_rocking (ROCKING_PERSIST_FRAMES = 18).
           Le type "body_rocking" n'est retourné que si la variance hanches
           reste haute pendant 18 frames consécutives, ET que la variance
           est stable dans la fenêtre glissante (CV < ROCKING_STABILITY_MAX_CV).
           Un pic isolé (changement de position) a un CV élevé et est rejeté.

        2. Suppression poignets si main levée (hand_raised).
           Si hand_raised = True, wrist_wv est forcé à 0.0. Le fidgeting de
           type "hand_movement" ne peut pas se déclencher pendant une levée
           de main (évite de confondre levée de main et fidgeting nerveux).

        3. Gate ZCR pour "hand_movement".
           Le type "hand_movement" n'est confirmé que si s_zcr >=
           FIDGET_WRIST_ZCR_MIN (0.15). Un ZCR faible = mouvement directionnel
           (écriture, frappe) pas répétitif = pas du fidgeting.
        """
        if raw_score < CFG.FIDGET_VAR_MIN * 2:
            self.lower_var_debug = 0.0
            self.hip_var_debug   = 0.0
            self.wrist_var_debug = 0.0
            self._rocking_persist_cnt = 0
            return "none"

        # ── Variance pondérée par visibilité ──────────────────────────────
        lower_wv, lower_fft = self._vis_weighted_var(
            self._LOWER_KPS, joint_var, kpb_valid)
        hip_wv,   hip_fft   = self._vis_weighted_var(
            self._HIP_KPS,   joint_var, kpb_valid)
        wrist_wv, wrist_fft = self._vis_weighted_var(
            self._WRIST_KPS, joint_var, kpb_valid)

        # Fix 3a : si main levée → ignorer la variance des poignets
        if hand_raised:
            wrist_wv  = 0.0
            wrist_fft = 0.0

        # Expose pour CSV
        self.lower_var_debug = round(lower_wv, 6)
        self.hip_var_debug   = round(hip_wv,   6)
        self.wrist_var_debug = round(wrist_wv, 6)

        max_var = max(lower_wv, hip_wv, wrist_wv)
        if max_var < CFG.FIDGET_VAR_MIN:
            self._rocking_persist_cnt = 0
            return "none"

        TIE_THRESHOLD = 0.10  # 10% d'écart minimum pour label franc

        # ── Test leg_shake ─────────────────────────────────────────────────
        if lower_wv >= hip_wv and lower_wv >= wrist_wv:
            second = max(hip_wv, wrist_wv)
            if second > 0 and (lower_wv - second) / max(lower_wv, 1e-9) < TIE_THRESHOLD:
                if lower_fft >= 0.20:
                    self._rocking_persist_cnt = 0
                    return "leg_shake"
                self._rocking_persist_cnt = 0
                return "generic"
            self._rocking_persist_cnt = 0
            return "leg_shake"

        # ── Test body_rocking avec persistance (Fix 2) ────────────────────
        if hip_wv >= lower_wv and hip_wv >= wrist_wv:
            second = max(lower_wv, wrist_wv)
            is_dominant = not (
                second > 0 and
                (hip_wv - second) / max(hip_wv, 1e-9) < TIE_THRESHOLD
                and hip_fft < 0.20
            )
            if is_dominant:
                # Alimenter la fenêtre de stabilité avec hip_wv courant
                self._rocking_stability_buf.append(hip_wv)

                # Test de stabilité : coefficient de variation dans la fenêtre
                buf = list(self._rocking_stability_buf)
                if len(buf) >= 4:
                    buf_arr = np.array(buf, dtype=np.float64)
                    mean_v  = float(buf_arr.mean())
                    std_v   = float(buf_arr.std())
                    cv      = std_v / max(mean_v, 1e-9)
                    stable  = cv < CFG.ROCKING_STABILITY_MAX_CV
                else:
                    stable = True   # pas assez de données → optimiste

                if stable:
                    self._rocking_persist_cnt += 1
                else:
                    # Signal impulsionnel (changement de position) → reset
                    self._rocking_persist_cnt = 0
                    return "generic"

                # Confirmation seulement après ROCKING_PERSIST_FRAMES
                if self._rocking_persist_cnt >= CFG.ROCKING_PERSIST_FRAMES:
                    return "body_rocking"
                else:
                    # En attente de confirmation — retourner le type précédent
                    # ou "generic" pendant la période de validation
                    return "generic"
            else:
                self._rocking_persist_cnt = 0
                return "generic"

        # ── Test hand_movement avec gate ZCR (Fix 3b) ─────────────────────
        if wrist_wv >= lower_wv and wrist_wv >= hip_wv:
            second = max(lower_wv, hip_wv)
            if second > 0 and (wrist_wv - second) / max(wrist_wv, 1e-9) < TIE_THRESHOLD:
                if wrist_fft >= 0.20 and s_zcr >= CFG.FIDGET_WRIST_ZCR_MIN:
                    self._rocking_persist_cnt = 0
                    return "hand_movement"
                self._rocking_persist_cnt = 0
                return "generic"
            # Vérifier la gate ZCR même pour le gagnant clair
            if s_zcr < CFG.FIDGET_WRIST_ZCR_MIN:
                # ZCR trop faible : mouvement directionnel (écriture) pas répétitif
                self._rocking_persist_cnt = 0
                return "generic"
            self._rocking_persist_cnt = 0
            return "hand_movement"

        self._rocking_persist_cnt = 0
        return "generic"

    # ──────────────────────────────────────────────────────────────────────
    # Main update
    # ──────────────────────────────────────────────────────────────────────

    def update(
        self, kpb: KeypointBuffer, sw: float, hand_raised: bool = False
    ) -> tuple[float, str, float, int]:
        """
        Met à jour le détecteur pour une frame.

        Paramètres :
            kpb         — KeypointBuffer courant
            sw          — shoulder width (body ruler)
            hand_raised — True si une main est levée (supprime wrist fidgeting)

        Retourne :
            fidget_score   float [0,1]  — score EMA combiné
            fidget_type    str          — "leg_shake"|"body_rocking"|"hand_movement"|"generic"|"none"
            best_joint_var float        — variance brute du joint le plus actif (debug)
            fidgeting      int  {0,1}   — label confirmé
        """
        # ── 1. Mise à jour des historiques de position normalisée ──────────
        for kp in self._ALL_KPS:
            pt = kpb.get(kp)
            if pt is not None:
                self._y_hist[kp].append(pt[1] / max(sw, 1.0))

        for kp in self._LATERAL_KPS:
            pt = kpb.get(kp)
            if pt is not None:
                self._x_hist[kp].append(pt[0] / max(sw, 1.0))

        # ── 2. Ensemble des joints valides (pour pondération visibilité) ────
        # Un joint est "valide" s'il a un historique suffisant ET que le
        # KeypointBuffer le considère actif (not interpolated / missing).
        min_hist  = max(12, CFG.FIDGET_WIN // 2)
        kpb_valid = {kp for kp in self._ALL_KPS if kpb.valid[kp]}

        # ── 3. Variance Y brute par joint ──────────────────────────────────
        joint_var: dict[int, float] = {}
        for kp in self._ALL_KPS:
            v = self._raw_var(kp)
            if v > 0.0:
                joint_var[kp] = v

        # ── 4. Calcul des quatre signaux ───────────────────────────────────
        if not joint_var:
            raw_score = 0.0
            self.fidget_type     = "none"
            self.best_joint_var  = 0.0
            self.fft_score_debug = 0.0
            # Reset colonnes debug
            self.lower_var_debug = 0.0
            self.hip_var_debug   = 0.0
            self.wrist_var_debug = 0.0
        else:
            # Meilleur joint global (variance brute max)
            best_kp  = max(joint_var, key=joint_var.__getitem__)
            best_var = joint_var[best_kp]
            self.best_joint_var = round(best_var, 6)

            all_arr = np.array(list(self._y_hist[best_kp]), dtype=np.float64)

            # Signal 1 : variance normalisée du meilleur joint
            s_var = self._variance_score(all_arr)

            # Signal 2 : peakedness FFT du meilleur joint
            s_fft = self._fft_score(all_arr) if len(all_arr) >= 8 else 0.0
            self.fft_score_debug = round(s_fft, 4)

            # Signal 3 : ZCR du meilleur joint
            s_zcr = self._zcr_score(all_arr) if len(all_arr) >= 8 else 0.0

            # Signal 4 : variance latérale X (hanches + épaules)
            lat_vars = []
            for kp in self._LATERAL_KPS:
                h = self._x_hist[kp]
                if len(h) >= min_hist:
                    arr_x = np.array(h, dtype=np.float64)
                    lat_vars.append(float(np.var(self._detrend(arr_x))))
            if lat_vars:
                s_lat = float(np.clip(
                    max(lat_vars) / max(CFG.FIDGET_LAT_VAR_NORM, 1e-9),
                    0.0, 1.0
                ))
            else:
                s_lat = 0.0

            # ── Combinaison pondérée ──────────────────────────────────────
            raw_score = (
                CFG.FIDGET_W_VAR * s_var
                + CFG.FIDGET_W_FFT * s_fft
                + CFG.FIDGET_W_ZCR * s_zcr
                + CFG.FIDGET_W_LAT * s_lat
            )

            # ── Classification de type (v11.2) ───────────────────────────
            # Passe hand_raised pour supprimer wrist fidgeting si main levée,
            # et s_zcr pour la gate ZCR du type hand_movement.
            self.fidget_type = self._classify_type(
                joint_var, s_lat, raw_score, kpb_valid, hand_raised, s_zcr
            )

        # ── 5. EMA ────────────────────────────────────────────────────────
        self._score_ema   = _ema(self._score_ema, raw_score, CFG.FIDGET_SCORE_EMA)
        self.fidget_score = round(self._score_ema, 4)

        # ── 6. Hysteresis à deux niveaux (v9) ─────────────────────────────
        if not self._fidgeting:
            # ── Entrée : accumulation du compteur de confirmation ──────────
            if self._score_ema >= CFG.FIDGET_SCORE_HIGH:
                self._high_cnt += 1
                self._low_cnt   = 0   # reset sortie
                if self._high_cnt >= CFG.FIDGET_CONFIRM_FRAMES:
                    self._fidgeting   = True
                    self._decay_count = CFG.FIDGET_DECAY_FRAMES
                    self._high_cnt    = 0
            else:
                # Décrémentation ×2 : le crédit accumulé se perd vite si
                # le score redescend entre deux séquences.
                self._high_cnt = max(0, self._high_cnt - 2)
                self._low_cnt  = 0
        else:
            # ── Sortie niveau 1 : score_floor (sortie rapide) ─────────────
            if self._score_ema < CFG.FIDGET_SCORE_FLOOR:
                # Le signal s'est effondré. On force le decay à 3 frames.
                self._decay_count = min(self._decay_count, CFG.FIDGET_DECAY_FAST)
                self._low_cnt     = CFG.FIDGET_EXIT_LOW_FRAMES  # force sortie niveau 2
            # ── Sortie niveau 2 : frames consécutives sous SCORE_LOW ───────
            elif self._score_ema < CFG.FIDGET_SCORE_LOW:
                self._low_cnt += 1
            else:
                # Score au-dessus de LOW → reset compteur de sortie
                self._low_cnt     = 0
                self._decay_count = CFG.FIDGET_DECAY_FRAMES   # recharge le decay

            # ── Décision de sortie ─────────────────────────────────────────
            if self._low_cnt >= CFG.FIDGET_EXIT_LOW_FRAMES:
                # Assez de frames basses consécutives → démarrer la décroissance
                self._decay_count = max(0, self._decay_count - 1)
                if self._decay_count == 0:
                    self._fidgeting = False
                    self._low_cnt   = 0
                    self._high_cnt  = 0
            # Si low_cnt < EXIT_LOW_FRAMES : pas encore déclenché → on attend

        fidgeting = int(self._fidgeting)
        if not self._fidgeting:
            self.fidget_type = "none"

        return self.fidget_score, self.fidget_type, self.best_joint_var, fidgeting


class PostureStateMachine:
    """
    Maintains EMA-smoothed scores for slouching and standing,
    and applies independent hysteresis to each behaviour.
    """

    _UPRIGHT  = 0
    _SLOUCHING = 1
    _STANDING  = 2

    def __init__(self) -> None:
        self._slouch_score_ema: float = 0.0
        self._stand_score_ema:  float = 0.0
        self._slouch_high_cnt:  int   = 0
        self._slouch_low_cnt:   int   = 0
        self._stand_high_cnt:   int   = 0
        self._stand_low_cnt:    int   = 0
        self._slouching: bool = False
        self._standing:  bool = False
        self._nose_y_hist: deque = deque(maxlen=8)

    def update(
        self,
        kpb:    KeypointBuffer,
        sw:     float,
        desk_y: float,
    ) -> PostureResult:
        res        = PostureResult()
        desk_ready = not math.isnan(desk_y)

        nose = kpb.get(KP_NOSE)
        ls   = kpb.get(KP_LEFT_SHOULDER)
        rs   = kpb.get(KP_RIGHT_SHOULDER)

        if ls and rs:
            sh_mid: Optional[tuple] = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
        elif ls:
            sh_mid = ls
        elif rs:
            sh_mid = rs
        else:
            sh_mid = None

        # ── Spine tilt (conservé pour le CSV, mais non utilisé dans slouch) ─
        # v11 : remplacé par head_tilt dans le calcul du score slouch.
        # On le calcule quand même pour la colonne CSV spine_tilt_deg.
        if nose and sh_mid:
            res.spine_tilt_deg = _tilt_from_vertical(nose, sh_mid)

        if sh_mid and desk_ready:
            res.shoulder_desk_norm = (desk_y - sh_mid[1]) / sw

        if nose and desk_ready:
            res.head_desk_norm = (desk_y - nose[1]) / sw

        if nose and sh_mid:
            res.forward_shift_norm = abs(nose[0] - sh_mid[0]) / sw

        if nose and sh_mid:
            res.body_height_norm = math.hypot(
                nose[0]-sh_mid[0], nose[1]-sh_mid[1]
            ) / sw

        if nose:
            self._nose_y_hist.append(nose[1] / sw)
        vert_vel_up = 0.0
        if len(self._nose_y_hist) >= 3:
            vert_vel_up = max(0.0,
                -(float(self._nose_y_hist[-1]) - float(self._nose_y_hist[-3])) / 2.0
            )

        # ══ SLOUCH SCORE v11.2 ════════════════════════════════════════════
        #
        # 4 signaux, tous basés sur nez + épaules uniquement (pas de hanches) :
        #
        # Signal A — HEAD_BETWEEN_SHOULDERS (poids 0.40) — NOUVEAU v11.2
        #   Si nose.y >= sh_mid.y − THRESHOLD×sw → score 1.0 (tête au niveau
        #   ou en dessous des épaules = fatigue/écriture intense).
        #   Ce signal répond à la demande utilisateur : détecter la tête baissée
        #   ENTRE les épaules, indépendamment du bureau.
        #
        # Signal B — HEAD_DESK (poids 0.30) : distance tête→bureau.
        #
        # Signal C — HEAD_TILT (poids 0.20) : angle épaules→nez vs vertical.
        #
        # Signal D — FORWARD_SHIFT (poids 0.10) : décalage horizontal nez→épaules.

        # ── Signal A : HEAD_BETWEEN_SHOULDERS ─────────────────────────────
        if nose and sh_mid:
            # threshold_px : la zone "entre les épaules" commence ici
            threshold_px = sh_mid[1] - CFG.SLOUCH_HEAD_BETWEEN_SHOULDERS_THRESHOLD * sw
            # nose.y croît vers le bas en pixels.
            # Quand nose.y >= sh_mid.y (nez au niveau épaules) → score = 1.0
            # Quand nose.y <= threshold_px → score = 0.0
            # Interpolation linéaire entre les deux.
            s_between = float(np.clip(
                (nose[1] - threshold_px) / max(sh_mid[1] - threshold_px, 1.0),
                0.0, 1.0
            ))
        else:
            s_between = 0.0

        # ── Signal B : HEAD_DESK ───────────────────────────────────────────
        if desk_ready and not math.isnan(res.head_desk_norm):
            hdg = res.head_desk_norm
            s_head = float(np.clip(
                (CFG.SLOUCH_HEAD_CLEAR - hdg)
                / max(CFG.SLOUCH_HEAD_CLEAR - CFG.SLOUCH_HEAD_FLOOR, 0.01),
                0.0, 1.0
            ))
        else:
            s_head = 0.0

        # ── Signal C : HEAD_TILT ───────────────────────────────────────────
        if nose and sh_mid:
            head_tilt_deg = _tilt_from_vertical(sh_mid, nose)
        else:
            head_tilt_deg = float("nan")

        if not math.isnan(head_tilt_deg):
            s_head_tilt = float(np.clip(
                (head_tilt_deg - CFG.SLOUCH_HEAD_TILT_CLEAR)
                / max(CFG.SLOUCH_HEAD_TILT_MAX - CFG.SLOUCH_HEAD_TILT_CLEAR, 0.01),
                0.0, 1.0
            ))
        else:
            s_head_tilt = 0.0

        # ── Signal D : FORWARD_SHIFT ───────────────────────────────────────
        if not math.isnan(res.forward_shift_norm):
            fwd = res.forward_shift_norm
            s_fwd = float(np.clip(
                (fwd - CFG.SLOUCH_FORWARD_SHIFT_CLEAR)
                / max(CFG.SLOUCH_FORWARD_SHIFT_MAX - CFG.SLOUCH_FORWARD_SHIFT_CLEAR, 0.01),
                0.0, 1.0
            ))
        else:
            s_fwd = 0.0

        # ── Combinaison v11.2 ──────────────────────────────────────────────
        # HEAD_BETWEEN=0.40, HEAD_DESK=0.30, HEAD_TILT=0.20, FORWARD_SHIFT=0.10
        raw_slouch = (
            CFG.SLOUCH_W_HEAD_BETWEEN * s_between
            + CFG.SLOUCH_W_HEAD_DESK  * s_head
            + CFG.SLOUCH_W_HEAD_TILT  * s_head_tilt
            + CFG.SLOUCH_W_FORWARD_SHIFT * s_fwd
        )
        self._slouch_score_ema = _ema(
            self._slouch_score_ema, raw_slouch, CFG.SLOUCH_SCORE_EMA
        )
        res.slouch_score = self._slouch_score_ema

        # ══ STAND SCORE ═══════════════════════════════════════════════════

        if not math.isnan(res.shoulder_desk_norm):
            sdg = res.shoulder_desk_norm
            s_sh = float(np.clip(
                (sdg - CFG.STANDING_SHOULDER_ABOVE_DESK_NORM + 0.40) / 0.80,
                0.0, 1.0
            ))
        else:
            s_sh = 0.0

        if not math.isnan(res.head_desk_norm):
            hdg2 = res.head_desk_norm
            s_hd = float(np.clip(
                (hdg2 - 1.50) / 1.00,
                0.0, 1.0
            ))
        else:
            s_hd = 0.0

        s_vel = float(np.clip(vert_vel_up / max(CFG.STAND_VEL_NORM, 1e-6), 0.0, 1.0))

        if not math.isnan(res.body_height_norm):
            bh = res.body_height_norm
            s_bh = float(np.clip(
                (bh - CFG.STAND_BODY_HT_SIT)
                / max(CFG.STAND_BODY_HT_STAND - CFG.STAND_BODY_HT_SIT, 0.01),
                0.0, 1.0
            ))
        else:
            s_bh = 0.0

        raw_stand = (
            CFG.STAND_W_SHOULDER_DESK * s_sh
            + CFG.STAND_W_HEAD_DESK   * s_hd
            + CFG.STAND_W_VERT_VEL    * s_vel
            + CFG.STAND_W_BODY_HEIGHT * s_bh
        )
        self._stand_score_ema = _ema(
            self._stand_score_ema, raw_stand, CFG.STAND_SCORE_EMA
        )
        res.stand_score = self._stand_score_ema

        # ══ HYSTERESIS v11 — stabilité améliorée ══════════════════════════
        #
        # Entrée : inchangée (2 frames au-dessus de HIGH pour confirmer).
        # Sortie : utilise SLOUCH_EXIT_LOW_FRAMES (30) et STAND_EXIT_LOW_FRAMES
        # (30) — frames CONSÉCUTIVES sous LOW requis avant de quitter l'état.
        # Cela évite le clignotement quand l'élève oscille à la limite.

        if not self._slouching:
            if self._slouch_score_ema >= CFG.SLOUCH_SCORE_HIGH:
                self._slouch_high_cnt += 1
                if self._slouch_high_cnt >= 2:
                    self._slouching      = True
                    self._slouch_low_cnt = 0
            else:
                self._slouch_high_cnt = max(0, self._slouch_high_cnt - 1)
        else:
            if self._slouch_score_ema < CFG.SLOUCH_SCORE_LOW:
                self._slouch_low_cnt += 1
                if self._slouch_low_cnt >= CFG.SLOUCH_EXIT_LOW_FRAMES:
                    self._slouching      = False
                    self._slouch_high_cnt = 0
            else:
                self._slouch_low_cnt = max(0, self._slouch_low_cnt - 1)

        if not self._standing:
            if self._stand_score_ema >= CFG.STAND_SCORE_HIGH:
                self._stand_high_cnt += 1
                if self._stand_high_cnt >= 2:
                    self._standing      = True
                    self._stand_low_cnt = 0
            else:
                self._stand_high_cnt = max(0, self._stand_high_cnt - 1)
        else:
            if self._stand_score_ema < CFG.STAND_SCORE_LOW:
                self._stand_low_cnt += 1
                if self._stand_low_cnt >= CFG.STAND_EXIT_LOW_FRAMES:
                    self._standing       = False
                    self._stand_high_cnt = 0
            else:
                self._stand_low_cnt = max(0, self._stand_low_cnt - 1)

        # ══ LABEL ═════════════════════════════════════════════════════════
        if self._standing and desk_ready:
            res.label      = "standing"
            res.confidence = round(self._stand_score_ema, 3)
        elif self._slouching:
            res.label      = "slouching"
            res.confidence = round(self._slouch_score_ema, 3)
        elif not math.isnan(res.shoulder_desk_norm) and \
                res.shoulder_desk_norm > CFG.SITTING_SHOULDER_ABOVE_DESK_NORM:
            res.label      = "sitting"
            span = max(
                CFG.STANDING_SHOULDER_ABOVE_DESK_NORM - CFG.SITTING_SHOULDER_ABOVE_DESK_NORM,
                0.01
            )
            res.confidence = min(1.0,
                (res.shoulder_desk_norm - CFG.SITTING_SHOULDER_ABOVE_DESK_NORM) / span)
        elif not desk_ready and not math.isnan(res.spine_tilt_deg):
            res.label      = "sitting"
            res.confidence = 0.20
        else:
            res.label = "unknown"

        return res


# ==============================================================================
# BEHAVIOUR 4 — BOUNCING DETECTOR
# ==============================================================================

class BouncingDetector:
    def __init__(self) -> None:
        self._head_y_buf: deque = deque(maxlen=CFG.BOUNCE_FFT_WIN * 2)
        self._score_ema:  float = 0.0
        self._high_cnt:   int   = 0
        self._low_cnt:    int   = 0
        self._bouncing:   bool  = False

    def _get_head_y(self, kpb: KeypointBuffer, sw: float) -> Optional[float]:
        nose = kpb.get(KP_NOSE)
        if nose:
            return nose[1] / sw
        ls = kpb.get(KP_LEFT_SHOULDER); rs = kpb.get(KP_RIGHT_SHOULDER)
        if ls and rs:  return ((ls[1]+rs[1])/2.0) / sw
        if ls:         return ls[1] / sw
        if rs:         return rs[1] / sw
        return None

    def update(self, kpb: KeypointBuffer, sw: float) -> tuple[float, int]:
        head_y = self._get_head_y(kpb, sw)
        if head_y is None:
            self._score_ema = _ema(self._score_ema, 0.0, CFG.BOUNCE_SCORE_EMA)
            return self._score_ema, int(self._bouncing)

        self._head_y_buf.append(head_y)

        if len(self._head_y_buf) < CFG.BOUNCE_FFT_WIN:
            return 0.0, 0

        sig_full = np.array(list(self._head_y_buf), dtype=np.float32)
        sig = sig_full[-CFG.BOUNCE_FFT_WIN:]
        sig_d = sig - sig.mean()

        amp = float(np.max(np.abs(sig_d)))
        if amp < CFG.BOUNCE_MIN_AMP_NORM:
            raw_score = 0.0
        else:
            spec = np.abs(np.fft.rfft(sig_d))[1:]
            if spec.max() > 0:
                fft_score = float(np.clip(
                    float(spec.max()) / float(spec.sum()), 0.0, 1.0
                ))
                fft_score = fft_score if fft_score >= CFG.BOUNCE_PEAK_THRESH else 0.0
            else:
                fft_score = 0.0

            var_sig = sig_full[-CFG.BOUNCE_VARIANCE_WIN:]
            var_sig_d = var_sig - var_sig.mean()
            variance = float(np.var(var_sig_d))
            var_score = float(np.clip(
                (variance - CFG.BOUNCE_VARIANCE_MIN)
                / max(CFG.BOUNCE_VARIANCE_MIN * 20, 1e-8),
                0.0, 1.0
            ))

            zcr_sig = sig_full[-CFG.BOUNCE_ZCR_WIN:]
            zcr_d   = zcr_sig - zcr_sig.mean()
            n_crossings = int(np.sum(np.diff(np.sign(zcr_d)) != 0))
            zcr = n_crossings / max(len(zcr_d) - 1, 1)
            zcr_score = float(np.clip(
                (zcr - CFG.BOUNCE_ZCR_MIN)
                / max(0.40 - CFG.BOUNCE_ZCR_MIN, 0.01),
                0.0, 1.0
            ))

            amp_score = float(np.clip(
                amp / (CFG.BOUNCE_MIN_AMP_NORM * 8),
                0.0, 1.0
            ))

            raw_score = (
                CFG.BOUNCE_W_FFT      * fft_score
                + CFG.BOUNCE_W_VARIANCE * var_score
                + CFG.BOUNCE_W_ZCR      * zcr_score
                + CFG.BOUNCE_W_AMP      * amp_score
            )

        self._score_ema = _ema(self._score_ema, raw_score, CFG.BOUNCE_SCORE_EMA)

        if not self._bouncing:
            if self._score_ema >= CFG.BOUNCE_SCORE_HIGH:
                self._high_cnt += 1
                if self._high_cnt >= CFG.BOUNCE_CONFIRM_FRAMES:
                    self._bouncing = True
                    self._low_cnt  = 0
            else:
                self._high_cnt = max(0, self._high_cnt - 1)
        else:
            if self._score_ema < CFG.BOUNCE_SCORE_LOW:
                self._low_cnt += 1
                if self._low_cnt >= CFG.BOUNCE_CONFIRM_FRAMES * 2:
                    self._bouncing = False
                    self._high_cnt = 0
            else:
                self._low_cnt = max(0, self._low_cnt - 1)

        return round(self._score_ema, 4), int(self._bouncing)


# ==============================================================================
# BEHAVIOUR 5 — HAND RAISE
# ==============================================================================

class HandRaiseSM:
    _IDLE = 0; _CANDIDATE = 1; _RAISED = 2; _LOWERING = 3

    def __init__(self) -> None:
        self._state        = self._IDLE
        self._rcnt         = 0
        self._lcnt         = 0
        self._score_ema    = 0.0
        self.hand_score:          float = 0.0
        self.wrist_above_shoulder: bool = False
        self.wrist_vel_up:         bool = False
        self.raised:               int  = 0

    def _sub_wrist_vs_shoulder(self, ls, rs, lw, rw, sw):
        scores = []
        for sh, wr in [(ls, lw), (rs, rw)]:
            if sh and wr:
                gap = (sh[1] - wr[1]) / sw
                scores.append(float(np.clip(
                    (gap + 0.5) / (CFG.HAND_HEIGHT_NORM + 0.5), 0.0, 1.0
                )))
        return max(scores) if scores else 0.0

    def _sub_wrist_vs_nose(self, nose, lw, rw, sw):
        if nose is None:
            return 0.0
        scores = []
        for wr in [lw, rw]:
            if wr:
                gap = (nose[1] - wr[1]) / sw
                scores.append(float(np.clip((gap + 0.3) / 0.8, 0.0, 1.0)))
        return max(scores) if scores else 0.0

    def _sub_elbow_vs_shoulder(self, ls, rs, le, re, sw):
        scores = []
        for sh, el in [(ls, le), (rs, re)]:
            if sh and el:
                gap = (sh[1] - el[1]) / sw
                scores.append(float(np.clip((gap + 0.3) / 0.8, 0.0, 1.0)))
        return max(scores) if scores else 0.0

    def _sub_upward_velocity(self, kpb, sw):
        lv = kpb.kpf(KP_LEFT_WRIST).vy_up
        rv = kpb.kpf(KP_RIGHT_WRIST).vy_up
        return float(np.clip(max(lv, rv) / (sw * max(CFG.HAND_VEL_NORM, 1e-6)), 0.0, 1.0))

    def _sub_elbow_angle(self, ls, rs, le, re, lw, rw):
        angles = []
        for sh, el, wr in [(ls, le, lw), (rs, re, rw)]:
            if sh and el and wr:
                angles.append(_angle_deg(sh, el, wr))
        if not angles:
            return 0.0
        return float(np.clip(
            (max(angles) - CFG.HAND_ELBOW_RAISE_ANGLE_MIN)
            / (180.0 - CFG.HAND_ELBOW_RAISE_ANGLE_MIN),
            0.0, 1.0
        ))

    def update(self, kpb: KeypointBuffer, sw: float) -> None:
        ls  = kpb.get(KP_LEFT_SHOULDER);  rs  = kpb.get(KP_RIGHT_SHOULDER)
        lw  = kpb.get(KP_LEFT_WRIST);     rw  = kpb.get(KP_RIGHT_WRIST)
        le  = kpb.get(KP_LEFT_ELBOW);     re  = kpb.get(KP_RIGHT_ELBOW)
        nose = kpb.get(KP_NOSE)

        s1 = self._sub_wrist_vs_shoulder(ls, rs, lw, rw, sw)
        s2 = self._sub_wrist_vs_nose(nose, lw, rw, sw)
        s3 = self._sub_elbow_vs_shoulder(ls, rs, le, re, sw)
        s4 = self._sub_upward_velocity(kpb, sw)
        s5 = self._sub_elbow_angle(ls, rs, le, re, lw, rw)

        raw_score = (
            CFG.HAND_W_WRIST_VS_SHOULDER   * s1
            + CFG.HAND_W_WRIST_VS_NOSE     * s2
            + CFG.HAND_W_ELBOW_VS_SHOULDER * s3
            + CFG.HAND_W_UPWARD_VEL        * s4
            + CFG.HAND_W_ELBOW_ANGLE       * s5
        )

        self._score_ema = _ema(self._score_ema, raw_score, CFG.HAND_SCORE_EMA)
        self.hand_score = round(self._score_ema, 3)
        self.wrist_above_shoulder = s1 > 0.5
        self.wrist_vel_up         = s4 > 0.3

        if self._state == self._IDLE:
            if self._score_ema >= CFG.HAND_SCORE_HIGH:
                self._state = self._CANDIDATE; self._rcnt = 1
        elif self._state == self._CANDIDATE:
            if self._score_ema >= CFG.HAND_SCORE_HIGH:
                self._rcnt += 1
                if self._rcnt >= CFG.HAND_RAISE_HOLD:
                    self._state = self._RAISED
            else:
                self._state = self._IDLE; self._rcnt = 0
        elif self._state == self._RAISED:
            if self._score_ema < CFG.HAND_SCORE_LOW:
                self._state = self._LOWERING; self._lcnt = 1
        elif self._state == self._LOWERING:
            if self._score_ema >= CFG.HAND_SCORE_HIGH:
                self._state = self._RAISED; self._lcnt = 0
            else:
                self._lcnt += 1
                if self._lcnt >= CFG.HAND_LOWER_HOLD:
                    self._state = self._IDLE; self._rcnt = self._lcnt = 0

        self.raised = int(self._state in (self._RAISED, self._LOWERING))


# ==============================================================================
# PER-TRACK STATE
# ==============================================================================

@dataclass
class TrackState:
    track_id:    int
    kpb:         KeypointBuffer        = field(default_factory=KeypointBuffer)
    desk:        DeskEstimator         = field(default_factory=DeskEstimator)
    bounce:      BouncingDetector      = field(default_factory=BouncingDetector)
    bounding:    BoundingDetector      = field(default_factory=BoundingDetector)
    fidget:      FidgetDetector        = field(default_factory=FidgetDetector)
    hand_sm:     HandRaiseSM           = field(default_factory=HandRaiseSM)
    posture_sm:  PostureStateMachine   = field(default_factory=PostureStateMachine)
    knee_cls:    KneePostureClassifier = field(default_factory=KneePostureClassifier)

    def process(self, kp_array: np.ndarray) -> dict:
        self.kpb.update(kp_array)
        sw = _body_ruler(self.kpb)

        self.hand_sm.update(self.kpb, sw)
        self.desk.update(self.kpb, sw=sw, hand_raised=bool(self.hand_sm.raised))

        # ── TOUJOURS calculer le chemin desk (haut du corps) ──────────────
        # v10 : le PostureStateMachine tourne à chaque frame, même si on
        # utilise ensuite les genoux. Cela garantit que :
        #   a) le DeskEstimator se remplit correctement dès les premières frames
        #   b) le score slouch est toujours disponible, même en mode knee
        #   c) le slouching est détecté correctement pour 2 élèves assis
        dr = self.posture_sm.update(self.kpb, sw, self.desk.desk_y)

        # ── Mode knee : UNIQUEMENT si les jambes sont vraiment fiables ─────
        # _knees_reliable() filtre les jambes extrapolées derrière le bureau
        # (faux positifs "standing" typiques en salle de classe).
        use_knees = _knees_reliable(self.kpb, kp_array, self.desk.desk_y)

        if use_knees:
            kr = self.knee_cls.classify(self.kpb)
            posture_label      = kr.label
            posture_confidence = kr.confidence
            posture_method     = "knee"
            knee_angle_left    = kr.knee_angle_left
            knee_angle_right   = kr.knee_angle_right
            avg_knee_angle     = kr.avg_knee_angle
        else:
            # Mode desk exclusif (haut du corps seulement)
            posture_label      = dr.label
            posture_confidence = dr.confidence
            posture_method     = "desk"
            knee_angle_left    = float("nan")
            knee_angle_right   = float("nan")
            avg_knee_angle     = float("nan")

        # ── Slouching : toujours prioritaire sur le label posture ──────────
        # Quel que soit le mode (knee ou desk), si le score slouch est fort,
        # on bascule en "slouching". Cela garantit la détection du slouch
        # même quand le mode knee est actif et donnerait "sitting".
        # Le standing est protégé : un élève debout ne peut pas être "slouching".
        if dr.slouch_score > CFG.SLOUCH_SCORE_HIGH and posture_label != "standing":
            posture_label      = "slouching"
            posture_confidence = round(dr.slouch_score, 3)

        slouch_score       = round(dr.slouch_score, 3)
        stand_score        = round(dr.stand_score, 3)
        spine_tilt_deg     = dr.spine_tilt_deg
        head_desk_norm     = dr.head_desk_norm
        shoulder_desk_norm = dr.shoulder_desk_norm
        forward_shift_norm = dr.forward_shift_norm
        body_height_norm   = dr.body_height_norm

        bounce_score, is_bouncing = self.bounce.update(self.kpb, sw)
        bound_score,  is_bounding = self.bounding.update(self.kpb, sw)

        fidget_score, fidget_type, best_joint_var, is_fidgeting = \
            self.fidget.update(self.kpb, sw, hand_raised=bool(self.hand_sm.raised))

        def _f(v: float) -> object:
            return round(v, 3) if not math.isnan(v) else ""

        return {
            "posture_method":       posture_method,
            "shoulder_width_px":    round(sw, 2),
            "desk_y_px":            round(self.desk.desk_y, 1)
                                    if self.desk.is_ready else "",
            "knee_angle_left":      _f(knee_angle_left),
            "knee_angle_right":     _f(knee_angle_right),
            "avg_knee_angle":       _f(avg_knee_angle),
            "spine_tilt_deg":       _f(spine_tilt_deg),
            "head_desk_norm":       _f(head_desk_norm),
            "shoulder_desk_norm":   _f(shoulder_desk_norm),
            "forward_shift_norm":   _f(forward_shift_norm),
            "body_height_norm":     _f(body_height_norm),
            "slouch_score":         slouch_score,
            "stand_score":          stand_score,
            "posture":              posture_label,
            "posture_confidence":   round(posture_confidence, 3),
            "bounce_score":         round(bounce_score, 3),
            "bouncing":             is_bouncing,
            "bound_score":          round(bound_score, 3),
            "bounding":             is_bounding,
            "fidget_score":         round(fidget_score, 4),
            "fidget_type":          fidget_type,
            "best_joint_var":       round(best_joint_var, 6),
            "fidgeting":            is_fidgeting,
            # ── Colonnes de debug fidget (v9) — par-région ─────────────────
            # lower_var_debug : variance pondérée visibilité de la région jambe
            #   (chevilles + genoux). Haute = mouvement de jambe probable.
            # hip_var_debug   : variance pondérée région hanche.
            # wrist_var_debug : variance pondérée région poignet.
            # fft_score_debug : peakedness FFT brute du meilleur joint global.
            #   Proche de 1.0 = signal très rythmique (vrai leg_shake).
            #   Proche de 0.0 = bruit non-rythmique (écriture, bougé de chaise).
            "fidget_lower_var":     round(self.fidget.lower_var_debug, 6),
            "fidget_hip_var":       round(self.fidget.hip_var_debug,   6),
            "fidget_wrist_var":     round(self.fidget.wrist_var_debug, 6),
            "fidget_fft_score":     round(self.fidget.fft_score_debug, 4),
            "hand_score":           self.hand_sm.hand_score,
            "wrist_above_shoulder": int(self.hand_sm.wrist_above_shoulder),
            "wrist_vel_up":         int(self.hand_sm.wrist_vel_up),
            "hand_raised":          self.hand_sm.raised,
        }


# ==============================================================================
# SUMMARY BUILDER
# ==============================================================================

@dataclass
class Episode:
    track_id: int; behaviour: str
    start_frame: int; end_frame: int
    conf_sum: float = 0.0; n_frames: int = 0

    def extend(self, fid: int, conf: float) -> None:
        self.end_frame = fid; self.conf_sum += conf; self.n_frames += 1

    @property
    def confidence_avg(self) -> float:
        return self.conf_sum / max(1, self.n_frames)

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


class SummaryBuilder:
    GAP_TOLERANCE = 5

    def __init__(self) -> None:
        self._open:       dict[int, Episode] = {}
        self._last_frame: dict[int, int]     = {}
        self._done:       list[Episode]      = []

    def ingest(self, fid: int, tid: int, beh: str, conf: float) -> None:
        gap = fid - self._last_frame.get(tid, -9999)
        self._last_frame[tid] = fid
        ep = self._open.get(tid)
        if ep is None or ep.behaviour != beh or gap > self.GAP_TOLERANCE:
            if ep: self._done.append(ep)
            self._open[tid] = Episode(tid, beh, fid, fid, conf, 1)
        else:
            ep.extend(fid, conf)

    def flush(self) -> None:
        for ep in self._open.values(): self._done.append(ep)
        self._open.clear()

    @property
    def episodes(self) -> list[Episode]:
        return sorted(self._done, key=lambda e: (e.track_id, e.start_frame))


def dominant_behaviour(feat: dict) -> tuple[str, float]:
    """
    Priorité : hand_raised > bounding > fidgeting > bouncing > posture

    v11 : labels fidget mis à jour.
      "fidgeting:leg_shake"    → FIDGET  LEG SHAKE
      "fidgeting:body_rocking" → FIDGET  BODY ROCKING
      "fidgeting:hand_movement"→ FIDGET  HAND MOVEMENT
      "fidgeting:generic"      → FIDGETING
    """
    if feat["hand_raised"]:
        return "hand_raised", 1.0
    if feat["bounding"]:
        return "bounding", float(feat.get("bound_score", 0.0))
    if feat["fidgeting"]:
        ftype = str(feat.get("fidget_type", "generic"))
        if ftype in ("none", ""):
            ftype = "generic"
        return f"fidgeting:{ftype}", float(feat.get("fidget_score", 0.0))
    if feat["bouncing"]:
        return "bouncing", float(feat.get("bounce_score", 0.0))
    return str(feat["posture"]), float(feat["posture_confidence"])



# ==============================================================================
# SPATIAL ID MAPPER
# ==============================================================================

def _detection_center_x(kp_array: np.ndarray) -> float:
    """
    Calcule la coordonnée X du centre d'une détection de personne.

    Stratégie (par ordre de priorité) :
    1. Moyenne X des ÉPAULES visibles (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER).
       Les épaules sont le repère le plus stable et le plus haut du corps,
       rarement occludé par le bureau.
    2. Fallback : moyenne X des HANCHES visibles.
    3. Fallback : moyenne X de TOUS les keypoints visibles (conf > 0.1).
    4. Fallback ultime : 0.0 (détection ignorée par le mapper).

    La visibilité est lue directement depuis kp_array[:, 2] (confiance brute
    du modèle de pose), sans filtre Kalman, pour avoir la vraie position
    géographique brute.
    """
    CONF_MIN = 0.20   # seuil minimal pour qu'un keypoint compte

    # Priorité 1 : épaules
    shoulder_xs = []
    for idx in (KP_LEFT_SHOULDER, KP_RIGHT_SHOULDER):
        if kp_array[idx, 2] >= CONF_MIN and kp_array[idx, 0] > 0:
            shoulder_xs.append(kp_array[idx, 0])
    if shoulder_xs:
        return float(np.mean(shoulder_xs))

    # Priorité 2 : hanches
    hip_xs = []
    for idx in (KP_LEFT_HIP, KP_RIGHT_HIP):
        if kp_array[idx, 2] >= CONF_MIN and kp_array[idx, 0] > 0:
            hip_xs.append(kp_array[idx, 0])
    if hip_xs:
        return float(np.mean(hip_xs))

    # Priorité 3 : tous les joints visibles
    all_xs = [kp_array[i, 0] for i in range(N_KP)
              if kp_array[i, 2] >= CONF_MIN and kp_array[i, 0] > 0]
    if all_xs:
        return float(np.mean(all_xs))

    return 0.0   # aucun keypoint valide → sera ignoré


class SpatialIDMapper:
    """
    Mappe les IDs instables du tracker YOLO vers deux IDs fixes :
      ID 1 = Élève côté GAUCHE  (X < frame_w / 2)
      ID 2 = Élève côté DROITE  (X ≥ frame_w / 2)

    POURQUOI CE SYSTÈME
    ───────────────────
    YOLOv8/v11 attribue des IDs basés sur la continuité du tracking. Dans une
    salle de classe avec une caméra fixe et deux élèves statiques, l'ID peut
    changer entre les frames si le tracker "perd" brièvement une personne
    (occlusion, faible confiance). On obtient alors des IDs parasites (3, 4, 5…)
    pour les mêmes personnes physiques.

    PRINCIPE
    ────────
    On utilise la POSITION SPATIALE comme critère d'identité, pas le tracking :
    - Gauche de l'image  → toujours ID 1
    - Droite de l'image  → toujours ID 2

    RÉSOLUTION DE CONFLITS (deux détections du même côté)
    ──────────────────────────────────────────────────────
    Si, dans une même frame, deux détections tombent du même côté (cas rare mais
    possible si un élève se penche vers le centre), on garde celle dont le centre X
    est le plus éloigné du milieu de l'image — c'est-à-dire la plus "clairement"
    dans sa zone. L'autre est ignorée.

    MÉMOIRE SPATIALE (anti-glitch)
    ───────────────────────────────
    On maintient le dernier X connu pour chaque zone. Si une détection apparaît
    très près du centre (dans la zone d'ambiguïté ±AMBIGUITY_ZONE_FRAC), on
    utilise la position précédente pour décider de l'assignation.

    FILTRE DE TAILLE MINIMALE
    ──────────────────────────
    Une détection avec très peu de keypoints valides (< MIN_VALID_KPS) est
    probablement un artefact (reflet, personne au fond) → ignorée.

    RÉSULTAT
    ────────
    `map_frame(persons, frame_w)` retourne un dict {1: kp_array, 2: kp_array}
    avec exactement 0, 1 ou 2 entrées. Jamais d'autre ID.
    """

    # Fraction de la largeur image définissant la zone d'ambiguïté centrale.
    # Ex: 0.10 = les 10% centraux (5% de chaque côté du milieu) sont ambigus.
    AMBIGUITY_ZONE_FRAC: float = 0.10

    # Nombre minimum de keypoints valides pour qu'une détection soit retenue.
    MIN_VALID_KPS: int = 3

    # Confidence minimum pour compter un keypoint comme "valide"
    KP_CONF_MIN: float = 0.20

    def __init__(self) -> None:
        # Dernier X connu par zone (pour résoudre les ambiguïtés)
        self._last_x: dict[int, float] = {
            CFG.STUDENT_ID_LEFT:  0.0,   # initialisé à 0 = côté gauche
            CFG.STUDENT_ID_RIGHT: float("inf"),  # initialisé à ∞ = côté droit
        }
        # Compteur de frames ignorées (debug)
        self.ignored_count: int = 0
        self.remapped_count: int = 0

    def _count_valid_kps(self, kp_array: np.ndarray) -> int:
        """Nombre de keypoints avec confiance suffisante."""
        return int(np.sum(kp_array[:, 2] >= self.KP_CONF_MIN))

    def map_frame(
        self,
        persons:  dict[int, np.ndarray],
        frame_w:  int,
    ) -> dict[int, np.ndarray]:
        """
        Entrée  : {tracker_id: kp_array(17,3)} — N détections quelconques
        Sortie  : {1: kp_array, 2: kp_array}   — exactement les 2 élèves connus

        Étapes :
        1. Filtrer les détections avec trop peu de keypoints valides.
        2. Calculer le center_x de chaque détection.
        3. Assigner chaque détection à un ID fixe (1 ou 2) selon sa zone.
        4. Résoudre les conflits (deux détections pour le même ID).
        5. Mettre à jour la mémoire spatiale.
        """
        if not persons or frame_w <= 0:
            return {}

        midpoint    = frame_w / 2.0
        ambig_half  = frame_w * self.AMBIGUITY_ZONE_FRAC / 2.0
        left_bound  = midpoint - ambig_half   # limite gauche de la zone ambiguë
        right_bound = midpoint + ambig_half   # limite droite de la zone ambiguë

        # ── Étape 1 & 2 : filtrer et calculer les centres ─────────────────
        candidates: list[tuple[int, np.ndarray, float]] = []
        for orig_tid, kp_arr in persons.items():
            n_valid = self._count_valid_kps(kp_arr)
            if n_valid < self.MIN_VALID_KPS:
                # Trop peu de joints → artefact, ignoré
                self.ignored_count += 1
                continue
            cx = _detection_center_x(kp_arr)
            if cx <= 0:
                self.ignored_count += 1
                continue
            candidates.append((orig_tid, kp_arr, cx))

        if not candidates:
            return {}

        # ── Étape 3 : assignation zone → ID fixe ─────────────────────────
        # Zone d'ambiguïté : on utilise la mémoire pour décider.
        left_cands:  list[tuple[int, np.ndarray, float]] = []
        right_cands: list[tuple[int, np.ndarray, float]] = []

        for orig_tid, kp_arr, cx in candidates:
            if cx < left_bound:
                # Clairement à gauche
                left_cands.append((orig_tid, kp_arr, cx))
            elif cx >= right_bound:
                # Clairement à droite
                right_cands.append((orig_tid, kp_arr, cx))
            else:
                # Zone ambiguë : utiliser la mémoire
                # Attribuer à la zone dont le dernier centre connu est le plus proche
                dist_left  = abs(cx - self._last_x[CFG.STUDENT_ID_LEFT])
                dist_right = abs(cx - self._last_x[CFG.STUDENT_ID_RIGHT])
                if dist_left <= dist_right:
                    left_cands.append((orig_tid, kp_arr, cx))
                else:
                    right_cands.append((orig_tid, kp_arr, cx))

        # ── Étape 4 : résoudre les conflits (deux détections même côté) ──
        def _pick_best(
            cands: list[tuple[int, np.ndarray, float]],
            is_left: bool,
        ) -> Optional[tuple[np.ndarray, float]]:
            """Garde la détection la plus éloignée du centre (la plus ancrée)."""
            if not cands:
                return None
            if len(cands) == 1:
                return cands[0][1], cands[0][2]
            # Trier par distance au centre : le plus éloigné du midpoint gagne
            if is_left:
                best = min(cands, key=lambda t: t[2])   # X minimal = plus à gauche
            else:
                best = max(cands, key=lambda t: t[2])   # X maximal = plus à droite
            # Les autres sont ignorés (parasites)
            self.ignored_count += len(cands) - 1
            return best[1], best[2]

        result: dict[int, np.ndarray] = {}

        left_pick = _pick_best(left_cands, is_left=True)
        if left_pick is not None:
            kp_arr, cx = left_pick
            result[CFG.STUDENT_ID_LEFT] = kp_arr
            self._last_x[CFG.STUDENT_ID_LEFT] = cx

        right_pick = _pick_best(right_cands, is_left=False)
        if right_pick is not None:
            kp_arr, cx = right_pick
            result[CFG.STUDENT_ID_RIGHT] = kp_arr
            self._last_x[CFG.STUDENT_ID_RIGHT] = cx

        # Compter les remappages (IDs YOLO qui ont changé)
        original_ids = set(persons.keys())
        new_ids      = set(result.keys())
        if original_ids != new_ids or len(original_ids) > 2:
            self.remapped_count += 1

        return result


# ==============================================================================
# CSV LOADER
# ==============================================================================

def load_body_csv(body_csv: str) -> tuple[dict[int, dict[int, np.ndarray]], list[int]]:
    print(f"[CSV]   Loading {body_csv}", end="", flush=True)

    body_index: dict[int, dict[int, np.ndarray]] = defaultdict(dict)
    chunk_size  = 17 * 20 * 500

    for chunk in pd.read_csv(
        body_csv,
        dtype={
            "frame_id":     "Int64",
            "track_id":     "Int64",
            "landmark_idx": "Int64",
            "x":             float,
            "y":             float,
            "visibility":    float,
        },
        chunksize=chunk_size,
    ):
        chunk = chunk.dropna(subset=["landmark_idx"])
        chunk["landmark_idx"] = chunk["landmark_idx"].astype(int)

        for (fid, tid), grp in chunk.groupby(["frame_id", "track_id"]):
            fid = int(fid); tid = int(tid)
            arr = np.zeros((N_KP, 3), dtype=np.float64)

            for _, row in grp.iterrows():
                idx = int(row["landmark_idx"])
                if not (0 <= idx < N_KP):
                    continue
                try:
                    arr[idx, 0] = float(row["x"])          if not pd.isna(row.get("x",          np.nan)) else 0.0
                    arr[idx, 1] = float(row["y"])          if not pd.isna(row.get("y",          np.nan)) else 0.0
                    arr[idx, 2] = float(row["visibility"]) if not pd.isna(row.get("visibility", np.nan)) else 0.0
                except (ValueError, TypeError):
                    pass

            body_index[fid][tid] = arr

        print(".", end="", flush=True)

    frame_ids = sorted(body_index.keys())
    print(f"\n[CSV]   {len(frame_ids)} frames loaded, "
          f"frame range [{frame_ids[0]} … {frame_ids[-1]}]")
    return body_index, frame_ids


# ==============================================================================
# OVERLAY RENDERER
# ==============================================================================

class Renderer:
    FONT      = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

    _LINKS_UPPER = [
        (KP_LEFT_SHOULDER,  KP_LEFT_ELBOW,    None,            2),
        (KP_LEFT_ELBOW,     KP_LEFT_WRIST,    None,            2),
        (KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW,   None,            2),
        (KP_RIGHT_ELBOW,    KP_RIGHT_WRIST,   None,            2),
        (KP_LEFT_SHOULDER,  KP_RIGHT_SHOULDER,(170, 170, 170), 2),
        (KP_LEFT_SHOULDER,  KP_LEFT_HIP,      (130, 130, 130), 1),
        (KP_RIGHT_SHOULDER, KP_RIGHT_HIP,     (130, 130, 130), 1),
        (KP_NOSE,           KP_LEFT_EYE,      (100, 100, 100), 1),
        (KP_NOSE,           KP_RIGHT_EYE,     (100, 100, 100), 1),
        (KP_LEFT_EYE,       KP_LEFT_EAR,      (100, 100, 100), 1),
        (KP_RIGHT_EYE,      KP_RIGHT_EAR,     (100, 100, 100), 1),
    ]
    _LINKS_LEGS = [
        (KP_LEFT_HIP,   KP_LEFT_KNEE,   (100, 160, 220), 2),
        (KP_LEFT_KNEE,  KP_LEFT_ANKLE,  (100, 160, 220), 2),
        (KP_RIGHT_HIP,  KP_RIGHT_KNEE,  (100, 160, 220), 2),
        (KP_RIGHT_KNEE, KP_RIGHT_ANKLE, (100, 160, 220), 2),
        (KP_LEFT_HIP,   KP_RIGHT_HIP,   (130, 130, 130), 1),
    ]
    _ARM_INDICES = {
        KP_LEFT_SHOULDER, KP_LEFT_ELBOW,  KP_LEFT_WRIST,
        KP_RIGHT_SHOULDER, KP_RIGHT_ELBOW, KP_RIGHT_WRIST,
    }

    @staticmethod
    def _beh_col(beh: str) -> tuple:
        return CFG.BEHAVIOUR_COLORS.get(beh, CFG.BEHAVIOUR_COLORS["unknown"])

    @classmethod
    def draw_skeleton(
        cls,
        frame:       np.ndarray,
        kpb:         KeypointBuffer,
        beh:         str,
        hand_raised: bool,
    ) -> tuple[int, int, int, int]:
        beh_col = cls._beh_col(beh)
        arm_col = (0, 255, 0) if hand_raised else beh_col
        xs, ys  = [], []

        def _pt(idx: int) -> Optional[tuple[int, int]]:
            p = kpb.get(idx)
            if p:
                xs.append(int(p[0])); ys.append(int(p[1]))
                return int(p[0]), int(p[1])
            return None

        for a_idx, b_idx, fixed_col, thick in cls._LINKS_UPPER:
            col = arm_col if a_idx in cls._ARM_INDICES else fixed_col
            pa  = _pt(a_idx); pb = _pt(b_idx)
            if pa and pb:
                cv2.line(frame, pa, pb, col, thick, cv2.LINE_AA)

        for a_idx, b_idx, leg_col, thick in cls._LINKS_LEGS:
            pa = _pt(a_idx); pb = _pt(b_idx)
            if pa and pb:
                cv2.line(frame, pa, pb, leg_col, thick, cv2.LINE_AA)

        for idx in range(N_KP):
            pt = _pt(idx)
            if pt:
                is_wrist = idx in (KP_LEFT_WRIST, KP_RIGHT_WRIST)
                is_knee  = idx in (KP_LEFT_KNEE, KP_RIGHT_KNEE)
                dot_col  = (0, 255, 0)   if (hand_raised and is_wrist) else \
                           (100, 220, 255) if is_knee else beh_col
                r = 5 if is_wrist else (4 if is_knee else 3)
                cv2.circle(frame, pt, r, dot_col, -1, cv2.LINE_AA)

        return (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0)

    @classmethod
    def draw_badge(
        cls,
        frame: np.ndarray,
        tid:   int,
        beh:   str,
        conf:  float,
        cx:    int,
        top_y: int,
    ) -> None:
        # ── Convertit le label interne en texte lisible (v11) ───────────────
        _FIDGET_LABELS = {
            "fidgeting:leg_shake":    "FIDGET  LEG SHAKE",
            "fidgeting:body_rocking": "FIDGET  BODY ROCKING",
            "fidgeting:hand_movement":"FIDGET  HAND MOVEMENT",
            "fidgeting:generic":      "FIDGETING",
        }
        if beh in _FIDGET_LABELS:
            display_beh = _FIDGET_LABELS[beh]
        else:
            display_beh = beh.upper().replace("_", " ")

        label = f"ID:{tid}  {display_beh}  {int(conf*100)}%"
        fs    = 0.50
        (tw, th), _ = cv2.getTextSize(label, cls.FONT_BOLD, fs, 1)

        pad_x, pad_y = 10, 5
        bx1 = max(0, cx - tw//2 - pad_x)
        bx2 = bx1 + tw + pad_x * 2
        by1 = max(0, top_y - th - pad_y*2 - 6)
        by2 = top_y - 6

        if by2 <= by1:
            by1 = max(0, by2 - th - pad_y*2)

        beh_col = cls._beh_col(beh)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), beh_col, cv2.FILLED)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 20, 20), 1)
        cv2.putText(frame, label, (bx1 + pad_x, by1 + pad_y + th),
                    cls.FONT_BOLD, fs, (0, 0, 0), 1, cv2.LINE_AA)

        bw     = bx2 - bx1
        filled = max(0, min(bw, int(bw * conf)))
        cv2.rectangle(frame, (bx1, by2),           (bx2,         by2+4), (40,40,40),  cv2.FILLED)
        cv2.rectangle(frame, (bx1, by2),           (bx1+filled,  by2+4), beh_col,     cv2.FILLED)

    @classmethod
    def draw_diagnostics(
        cls,
        frame:   np.ndarray,
        feat:    dict,
        x_right: int,
        y_top:   int,
    ) -> None:
        def _s(key: str, dec: int = 2) -> str:
            v = feat.get(key, "")
            if v == "" or (isinstance(v, float) and math.isnan(v)):
                return "n/a"
            return str(v) if isinstance(v, str) else f"{float(v):.{dec}f}"

        method = feat.get("posture_method", "?")
        # Indicateur visuel du mode posture avec avertissement knee
        mode_str = method
        if method == "knee":
            mode_str = "knee ⚠"   # avertissement : mode knee actif (rare en classe)

        lines = [
            f"mode  : {mode_str}",
            f"knee  : {_s('avg_knee_angle')}°",
            f"slouch: {_s('slouch_score')}",   # remonté en 3e position — priorité visuelle
            f"desk_y: {_s('desk_y_px',1)}px",
            f"spine : {_s('spine_tilt_deg')}°",
            f"hd_dk : {_s('head_desk_norm')}",
            f"stand : {_s('stand_score')}",
            f"bounce: {_s('bounce_score')}",
            f"bound : {_s('bound_score')}",
            f"fidget: {_s('fidget_score')} [{feat.get('fidget_type','none')[:8]}]",
            f"l/h/w : {_s('fidget_lower_var',4)}/{_s('fidget_hip_var',4)}/{_s('fidget_wrist_var',4)}",
            f"hand  : {_s('hand_score')}",
        ]
        lh = 13; pw = 195; ph = lh * len(lines) + 6
        px1 = min(x_right + 4, frame.shape[1] - pw - 2)
        py1 = max(0, y_top)
        px2 = min(frame.shape[1] - 1, px1 + pw)
        py2 = min(frame.shape[0] - 1, py1 + ph)

        sub = frame[py1:py2, px1:px2]
        if sub.size > 0:
            frame[py1:py2, px1:px2] = cv2.addWeighted(
                sub, 0.20, np.zeros_like(sub), 0.80, 0
            )

        # Couleur du mode : jaune = desk (normal), orange = knee (attention)
        mode_col = (80, 200, 255) if method == "desk" else (0, 140, 255)
        slouch_active = bool(feat.get("posture") == "slouching")
        fidget_active = bool(feat.get("fidgeting", 0))

        # Mapping ligne → clé score pour highlight dynamique
        score_keys = {
            2: "slouch_score",
            6: "stand_score",
            7: "bounce_score",
            8: "bound_score",
            9: "fidget_score",
            11: "hand_score",
        }

        for i, line in enumerate(lines):
            ty = py1 + lh * (i + 1)
            if ty >= frame.shape[0]:
                break
            col = mode_col if i == 0 else (200, 200, 200)

            # Slouch row : rouge-orange si actif
            if i == 2:
                try:
                    v = float(feat.get("slouch_score", 0.0))
                    if slouch_active:
                        col = (50, 50, 255)    # rouge vif → slouching confirmé
                    elif v > 0.25:
                        col = (80, 180, 255)   # cyan → score élevé mais pas confirmé
                except (ValueError, TypeError):
                    pass
            elif i in score_keys:
                try:
                    val = float(feat.get(score_keys[i], 0.0))
                    if val > 0.30:
                        col = (80, 220, 255)
                except (ValueError, TypeError):
                    pass

            # Fidget row : orange vif si actif
            if i == 9 and fidget_active:
                col = (0, 140, 255)

            cv2.putText(frame, line, (px1 + 3, ty),
                        cls.FONT, 0.35, col, 1, cv2.LINE_AA)

    @staticmethod
    def draw_desk_line(
        frame:    np.ndarray,
        desk_y:   float,
        x_min:    int,
        x_max:    int,
        track_id: int,
    ) -> None:
        if math.isnan(desk_y):
            return
        y = int(desk_y)
        col = CFG.DESK_LINE_COLOR

        seg_on, seg_off = 10, 6
        x = x_min
        while x < x_max:
            x_end = min(x + seg_on, x_max)
            cv2.line(frame, (x, y), (x_end, y), col, 1, cv2.LINE_AA)
            x = x_end + seg_off

        cv2.putText(frame, f"desk{track_id}", (x_min, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)

    @classmethod
    def draw_hand_banner(cls, frame, tid, cx, top_y, tick) -> None:
        if (tick // 15) % 2 == 0:
            label = f"! HAND RAISED  ID:{tid} !"
            fs    = 0.52
            (tw, th), _ = cv2.getTextSize(label, cls.FONT_BOLD, fs, 2)
            bx1 = max(0, cx - tw//2 - 8)
            by1 = max(0, top_y - th - 44)
            bx2 = bx1 + tw + 16
            by2 = by1 + th + 10
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (220, 0, 200), cv2.FILLED)
            cv2.putText(frame, label, (bx1+8, by1+th+4),
                        cls.FONT_BOLD, fs, (255,255,255), 2, cv2.LINE_AA)

    @classmethod
    def draw_bounding_banner(cls, frame, tid, cx, top_y, tick) -> None:
        if (tick // 10) % 2 == 0:
            label = f">> BOUNDING  ID:{tid} <<"
            fs    = 0.52
            (tw, th), _ = cv2.getTextSize(label, cls.FONT_BOLD, fs, 2)
            bx1 = max(0, cx - tw//2 - 8)
            by1 = max(0, top_y - th - 64)
            bx2 = bx1 + tw + 16
            by2 = by1 + th + 10
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 200, 255), cv2.FILLED)
            cv2.putText(frame, label, (bx1+8, by1+th+4),
                        cls.FONT_BOLD, fs, (0, 0, 0), 2, cv2.LINE_AA)

    @classmethod
    def draw_fidgeting_banner(cls, frame, tid, cx, top_y, tick, fidget_type) -> None:
        if (tick // 12) % 2 == 0:
            type_str = fidget_type.upper().replace("_", " ") if fidget_type != "none" else ""
            label = f"~ FIDGET {type_str}  ID:{tid} ~"
            fs    = 0.50
            (tw, th), _ = cv2.getTextSize(label, cls.FONT_BOLD, fs, 2)
            bx1 = max(0, cx - tw//2 - 10)
            by1 = max(0, top_y - th - 82)
            bx2 = bx1 + tw + 20
            by2 = by1 + th + 10
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 140, 255), cv2.FILLED)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0,  80, 180), 2)
            cv2.putText(frame, label, (bx1 + 10, by1 + th + 4),
                        cls.FONT_BOLD, fs, (255, 255, 255), 2, cv2.LINE_AA)

    @staticmethod
    def draw_hud(frame, frame_id, fps, n_persons, paused, speed, progress) -> None:
        h, w = frame.shape[:2]
        hud_lines = [
            f"Frame : {frame_id}",
            f"FPS   : {fps:.1f}",
            f"People: {n_persons}",
            f"Speed : {speed:.1f}x",
            "|| PAUSED" if paused else "> PLAYING",
        ]
        hud_h, hud_w = 18*len(hud_lines)+8, 155
        sub = frame[0:hud_h, 0:hud_w]
        if sub.size > 0:
            frame[0:hud_h, 0:hud_w] = cv2.addWeighted(sub, 0.2,
                                                        np.zeros_like(sub), 0.8, 0)
        for i, line in enumerate(hud_lines):
            col = (0, 60, 255) if "PAUSED" in line else (0, 255, 180)
            cv2.putText(frame, line, (6, 18+i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, col, 1, cv2.LINE_AA)

        bar_h  = 6
        filled = int(w * progress)
        cv2.rectangle(frame, (0, h-bar_h), (w, h),       (40,40,40),   cv2.FILLED)
        cv2.rectangle(frame, (0, h-bar_h), (filled, h),  (0,200,255),  cv2.FILLED)

        hint = "SPACE=pause/step  Q=quit  S=snapshot  +/-=speed"
        (hw, hh), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.37, 1)
        cv2.putText(frame, hint, (w-hw-6, h-bar_h-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (150,150,150), 1, cv2.LINE_AA)

    @classmethod
    def render_all(
        cls, frame, results, frame_id, fps, paused, speed, progress, tick
    ) -> None:
        for tid, (track, feat) in results.items():
            beh, conf    = dominant_behaviour(feat)
            hand_up      = bool(feat["hand_raised"])
            is_bounding  = bool(feat["bounding"])
            is_fidgeting = bool(feat["fidgeting"])
            # fidget_type : extrait depuis le label composé "fidgeting:xxx"
            # ou depuis feat directement si disponible
            if ":" in beh:
                fidget_type = beh.split(":", 1)[1]   # "leg_shake", "rocking", …
            else:
                fidget_type = str(feat.get("fidget_type", "none"))
            # Comportement "de base" pour le squelette (on normalise le compound)
            beh_base = beh.split(":")[0]   # "fidgeting", "sitting", etc.
            method   = feat.get("posture_method", "?")

            x_min, y_min, x_max, y_max = cls.draw_skeleton(
                frame, track.kpb, beh_base, hand_up
            )

            if x_max == 0 and y_max == 0:
                continue

            cx = (x_min + x_max) // 2

            # Ligne de bureau : toujours affichée quand disponible
            if track.desk.is_ready:
                cls.draw_desk_line(frame, track.desk.desk_y, x_min, x_max, tid)

            cls.draw_badge(frame, tid, beh, conf, cx, y_min)

            if hand_up:
                cls.draw_hand_banner(frame, tid, cx, y_min, tick)
            if is_bounding:
                cls.draw_bounding_banner(frame, tid, cx, y_min, tick)
            if is_fidgeting and not is_bounding:
                cls.draw_fidgeting_banner(frame, tid, cx, y_min, tick, fidget_type)

            cls.draw_diagnostics(frame, feat, x_max, y_min)

        cls.draw_hud(frame, frame_id, fps, len(results), paused, speed, progress)


# ==============================================================================
# CSV OUTPUT COLUMNS
# ==============================================================================

_RAW_COLS = [
    "frame_id", "track_id",
    "posture_method",
    "shoulder_width_px", "desk_y_px",
    "knee_angle_left", "knee_angle_right", "avg_knee_angle",
    "spine_tilt_deg", "head_desk_norm", "shoulder_desk_norm",
    "forward_shift_norm", "body_height_norm",
    "slouch_score", "stand_score", "posture", "posture_confidence",
    "bounce_score", "bouncing",
    "bound_score", "bounding",
    "fidget_score", "fidget_type", "best_joint_var", "fidgeting",
    # ── Colonnes de debug fidget v9 ───────────────────────────────────────
    # Utiles pour analyser les faux positifs dans pandas/Excel :
    #   - Comparer lower_var vs wrist_var pour voir si leg_shake ou écriture
    #   - fft_score proche de 0 = signal non-rythmique (faux positive probable)
    "fidget_lower_var",   # max variance pondérée visibilité — région jambe
    "fidget_hip_var",     # max variance pondérée visibilité — région hanche
    "fidget_wrist_var",   # max variance pondérée visibilité — région poignet
    "fidget_fft_score",   # peakedness FFT brute du meilleur joint [0,1]
    "hand_score", "wrist_above_shoulder", "wrist_vel_up", "hand_raised",
]
_SUM_COLS = [
    "track_id", "behaviour",
    "start_frame", "end_frame", "duration_frames", "confidence_avg",
]


# ==============================================================================
# MAIN
# ==============================================================================

def run(
    video_path: str,
    body_csv:   str,
    raw_out:    str,
    sum_out:    str,
    speed:      float,
    step_mode:  bool,
    save_video: str,
) -> None:

    for p, name in [(video_path, "Video"), (body_csv, "Body CSV")]:
        if not Path(p).exists():
            print(f"[ERROR] {name} not found: {p}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Behaviour Classifier v11.2  —  slouch HEAD_BETWEEN + rocking persist + fidget ZCR")
    print(f"{'='*60}\n")

    body_index, frame_ids = load_body_csv(body_csv)

    if not frame_ids:
        print("[ERROR] Body CSV is empty or has no valid rows.")
        sys.exit(1)

    total_csv_frames = len(frame_ids)
    print(f"[CSV]   {total_csv_frames} frames will be processed.\n")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Video] {Path(video_path).name}  "
          f"{frame_w}x{frame_h}  {video_fps:.1f}fps  {total_video_frames} frames")
    print(f"[Mode]  Speed={speed}x  Step={step_mode}")
    print(f"[Out]   {raw_out}  |  {sum_out}\n")
    print("  Controls: SPACE=pause/step  Q/ESC=quit  S=snapshot  +/-=speed\n")

    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_video, fourcc, video_fps, (frame_w, frame_h))
        print(f"[Save]  Annotated video → {save_video}")

    raw_fh = open(raw_out, "w", newline="", encoding="utf-8")
    raw_w  = csv.DictWriter(raw_fh, fieldnames=_RAW_COLS)
    raw_w.writeheader()

    track_states: dict[int, TrackState] = {
        CFG.STUDENT_ID_LEFT:  TrackState(track_id=CFG.STUDENT_ID_LEFT),
        CFG.STUDENT_ID_RIGHT: TrackState(track_id=CFG.STUDENT_ID_RIGHT),
    }
    summary    = SummaryBuilder()
    renderer   = Renderer()
    id_mapper  = SpatialIDMapper()   # ← Spatial ID Mapper v11.2

    def get_track(tid: int) -> TrackState:
        # Seuls les IDs 1 et 2 sont autorisés — garanti par le mapper
        return track_states[tid]

    fps_buf:  deque = deque(maxlen=30)
    prev_t    = time.perf_counter()

    paused         = step_mode
    tick           = 0
    rows_written   = 0
    snapshot_count = 0
    frame_delay_ms = max(1, int(1000 / (video_fps * speed)))

    last_rendered: Optional[np.ndarray] = None

    cv2.namedWindow(CFG.WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CFG.WINDOW_NAME, min(frame_w, 1400), min(frame_h, 860))

    csv_frame_idx = 0

    while csv_frame_idx < total_csv_frames:

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):
            print("\n[Info]  User quit.")
            break

        if key == ord(' '):
            if not step_mode:
                paused = not paused

        if key == ord('s'):
            if last_rendered is not None:
                snap = f"snapshot_{snapshot_count:04d}.png"
                cv2.imwrite(snap, last_rendered)
                print(f"[Snap]  {snap}")
                snapshot_count += 1

        if key in (ord('+'), ord('=')):
            speed = min(speed * 1.5, 32.0)
            frame_delay_ms = max(1, int(1000 / (video_fps * speed)))
            print(f"[Speed] {speed:.2f}x")

        if key == ord('-'):
            speed = max(speed / 1.5, 0.05)
            frame_delay_ms = max(1, int(1000 / (video_fps * speed)))
            print(f"[Speed] {speed:.2f}x")

        if paused and key != ord(' '):
            if last_rendered is not None:
                cv2.imshow(CFG.WINDOW_NAME, last_rendered)
            cv2.waitKey(30)
            continue

        frame_id = frame_ids[csv_frame_idx]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()

        if not ret:
            print(f"[Warn]  Could not read video frame {frame_id} — skipping.")
            csv_frame_idx += 1
            continue

        persons = body_index.get(frame_id, {})

        if not persons:
            now = time.perf_counter(); dt = now - prev_t; prev_t = now
            if dt > 0: fps_buf.append(1.0/dt)
            disp_fps = float(np.mean(fps_buf)) if fps_buf else 0.0
            progress = csv_frame_idx / max(1, total_csv_frames-1)
            Renderer.draw_hud(frame, frame_id, disp_fps, 0, paused, speed, progress)
            cv2.imshow(CFG.WINDOW_NAME, frame)
            last_rendered = frame.copy()
            if writer: writer.write(frame)
            csv_frame_idx += 1; tick += 1
            cv2.waitKey(frame_delay_ms)
            continue

        frame_results: dict[int, tuple] = {}

        # ── Spatial ID Mapper : écrase les IDs YOLO par les zones spatiales ──
        # `persons` contient {tracker_id: kp_array} avec des IDs instables.
        # `mapped_persons` contient {1: kp_array, 2: kp_array} — IDs fixes.
        # Toute détection parasite (3e ID, artefact) est silencieusement ignorée.
        mapped_persons = id_mapper.map_frame(persons, frame_w)

        for track_id, kp_array in mapped_persons.items():
            track = get_track(track_id)
            feat  = track.process(kp_array)

            raw_w.writerow({"frame_id": frame_id, "track_id": track_id, **feat})
            rows_written += 1

            beh_label, beh_conf = dominant_behaviour(feat)
            summary.ingest(frame_id, track_id, beh_label, beh_conf)

            frame_results[track_id] = (track, feat)

        if csv_frame_idx % 60 == 0:
            raw_fh.flush()

        now = time.perf_counter(); dt = now - prev_t; prev_t = now
        if dt > 0: fps_buf.append(1.0/dt)
        disp_fps = float(np.mean(fps_buf)) if fps_buf else 0.0
        progress = csv_frame_idx / max(1, total_csv_frames - 1)

        renderer.render_all(
            frame, frame_results, frame_id,
            disp_fps, paused, speed, progress, tick
        )

        cv2.imshow(CFG.WINDOW_NAME, frame)
        last_rendered = frame.copy()

        if writer:
            writer.write(frame)

        csv_frame_idx += 1
        tick          += 1

        cv2.waitKey(frame_delay_ms)

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    raw_fh.flush()
    raw_fh.close()

    print(f"\n[Done]  CSV 1 — {rows_written} rows → {raw_out}")
    print(f"[Mapper] {id_mapper.remapped_count} frames had ID remapping  |  "
          f"{id_mapper.ignored_count} parasitic detections discarded")

    summary.flush()
    episodes = summary.episodes
    with open(sum_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_SUM_COLS)
        w.writeheader()
        for ep in episodes:
            w.writerow({
                "track_id":        ep.track_id,
                "behaviour":       ep.behaviour,
                "start_frame":     ep.start_frame,
                "end_frame":       ep.end_frame,
                "duration_frames": ep.duration_frames,
                "confidence_avg":  round(ep.confidence_avg, 4),
            })

    print(f"[Done]  CSV 2 — {len(episodes)} episodes → {sum_out}")

    # ── Rapport final : uniquement ID 1 et ID 2 ───────────────────────────
    # Aucun ID parasite ne peut apparaître ici car le mapper les a tous filtrés.
    for student_id in (CFG.STUDENT_ID_LEFT, CFG.STUDENT_ID_RIGHT):
        label   = CFG.STUDENT_LABEL[student_id]
        student_eps = [ep for ep in episodes if ep.track_id == student_id]

        print(f"\n{'='*64}")
        print(f"  ÉLÈVE {label}")
        print(f"{'='*64}")
        if not student_eps:
            print(f"  (aucune détection pour cet élève)")
        else:
            print(f"  {'Behaviour':<18} {'Start':>7} {'End':>7} {'Frames':>7} {'Conf':>6}")
            print(f"  {'-'*54}")
            for ep in student_eps:
                beh_display = ep.behaviour.replace("fidgeting:", "fidget:").upper()
                print(f"  {beh_display:<18} "
                      f"{ep.start_frame:>7} {ep.end_frame:>7} "
                      f"{ep.duration_frames:>7} {ep.confidence_avg:>6.3f}")
        print(f"{'='*64}")

    print(f"\n  Total : 2 élèves  |  {len(episodes)} épisodes  |  "
          f"{rows_written} frames analysées\n")


# ==============================================================================
# CLI
# ==============================================================================

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Behaviour Classifier v11.2 — slouch head-between + rocking persist + fidget ZCR gate"
    )
    p.add_argument("--video",      default=CFG.VIDEO_PATH)
    p.add_argument("--body",       default=CFG.BODY_CSV)
    p.add_argument("--raw-out",    dest="raw_out",   default=CFG.RAW_OUT_CSV)
    p.add_argument("--sum-out",    dest="sum_out",   default=CFG.SUM_OUT_CSV)
    p.add_argument("--speed",      type=float, default=1.0)
    p.add_argument("--step",       action="store_true")
    p.add_argument("--save-video", dest="save_video", default="")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    run(
        video_path = args.video,
        body_csv   = args.body,
        raw_out    = args.raw_out,
        sum_out    = args.sum_out,
        speed      = args.speed,
        step_mode  = args.step,
        save_video = args.save_video,
    )
