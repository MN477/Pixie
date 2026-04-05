"""
config.py
=========
Central configuration for all labeling pipelines.
Stores paths, global settings, and all empirical thresholds.
"""

class Paths:
    # Head Pose
    HEAD_POSE_INPUT_CSV = "raw_head_pose_multi.csv"
    HEAD_POSE_OUTPUT_CSV = "labeled_head_pose_multi.csv"

    # Gaze
    GAZE_INPUT_CSV = "raw_gaze_multi.csv"
    GAZE_HEAD_POSE_CSV = "labeled_head_pose_multi.csv"
    GAZE_OUTPUT_CSV = "labeled_gaze_multi.csv"

    # Action Units
    AU_INPUT_CSV = "raw_action_units_multi.csv"
    AU_HEAD_POSE_CSV = "labeled_head_pose_multi.csv"
    AU_OUTPUT_CSV = "labeled_action_units_multi.csv"
    AU_EVENTS_CSV = "collective_events.csv"

    # Body Gestures
    BODY_CSV = "raw_body_multi.csv"
    BODY_RAW_OUT_CSV = "behaviour_raw_frames.csv"
    BODY_SUM_OUT_CSV = "behaviour_summary.csv"


class GlobalConfig:
    # Common across multiple pipelines
    FPS = 30
    OPENFACE_CONFIDENCE_THRESH = 0.85
    MEDIAN_WINDOW = 5
    INTERPOLATE_LIMIT = 3


class HeadPoseConfig:
    # Hard Thresholds (Degrees)
    PITCH_UP_THRESH = 15.0
    PITCH_DOWN_THRESH = -15.0
    
    YAW_LEFT_THRESH = 20.0
    YAW_RIGHT_THRESH = -20.0
    
    ROLL_LEFT_THRESH = 15.0
    ROLL_RIGHT_THRESH = -15.0

    # OpenFace Reliability Limits
    YAW_RELIABLE_LIMIT = 35.0
    PITCH_RELIABLE_LIMIT = 20.0

    # Margin (Degrees) past threshold required to reach maximum confidence (1.0)
    MARGIN = 10.0


class GazeConfig:
    # Horizontal gaze thresholds (radians)
    GAZE_H_CENTER_MAX = 0.15
    GAZE_H_MARGIN = 0.15

    # Vertical gaze thresholds (radians)
    GAZE_V_LEVEL_MAX = 0.12
    GAZE_V_MARGIN = 0.12

    # Gaze stability & Duration
    STABILITY_WINDOW = 15
    GAZE_MIN_DURATION = 0.3      # seconds

    # Confidence scaling margin
    CONFIDENCE_MARGIN = 0.15     # radians


class ActionUnitConfig:
    # Duration thresholds (seconds)
    SMILE_MIN_DURATION = 0.5
    FATIGUE_MIN_DURATION = 2.0
    YAWNING_MIN_DURATION = 0.2
    TALKING_MIN_DURATION = 0.5

    # Collective events
    COLLECTIVE_WINDOW_SEC = 5.0
    COLLECTIVE_MIN_TRACKS = 3

    CONFIDENCE_MARGIN = 1.5

    # Genuine Smile
    SMILE_AU06_THRESH = 1.0
    SMILE_AU12_THRESH = 1.0

    # Fatigue
    FATIGUE_AU45_ROLLING_THRESH = 0.3
    FATIGUE_AU05_UPPER_THRESH = 0.5
    FATIGUE_AU15_THRESH = 0.5
    FATIGUE_AU01_THRESH = 1.0

    # Yawning
    YAWNING_AU25_THRESH = 1.5
    YAWNING_AU26_THRESH = 1.5
    YAWNING_AU27_THRESH = 1.0

    # Talking Flag
    TALKING_AU25_THRESH = 0.5
    TALKING_AU26_THRESH = 0.5

    # Expressiveness Score
    EXPRESSIVENESS_AUS = [
        "AU06_r", "AU12_r", "AU04_r", "AU07_r", "AU09_r",
        "AU10_r", "AU15_r", "AU17_r", "AU20_r", "AU23_r"
    ]
    EXPRESSIVENESS_ACTIVITY_THRESH = 1.0


class BodyGesturesConfig:
    # Keypoint confidence gate
    KP_CONF_MIN = 0.25
    KP_INTERP_MAX = 10

    # Kalman filter noise
    KP_KALMAN_Q = 2e-4
    KP_KALMAN_R = 6e-3

    # Body ruler fallback when both shoulders invisible
    FALLBACK_SW = 100.0

    # Behaviour 1 & 3 : Sitting / Standing (with hysteresis)
    SITTING_KNEE_ANGLE_MAX = 140.0
    SITTING_KNEE_ANGLE_EXIT = 150.0    # Original CFG has this
    STANDING_KNEE_ANGLE_MIN = 158.0
    STANDING_KNEE_ANGLE_EXIT = 148.0   # Original CFG has this
    POSTURE_CONFIRM_FRAMES = 4

    # Upper-body fallback (desk-occluded lower body)
    KNEE_VIS_MIN = 0.30
    UB_BBOX_AR_SITTING_MAX = 1.6
    UB_BBOX_AR_STANDING_MIN = 2.2
    UB_SHOULDER_Y_SITTING_MIN = 0.30
    UB_SHOULDER_Y_STANDING_MAX = 0.22
    UB_TORSO_COMPRESS_THRESH = 0.08
    UB_HEAD_PITCH_SLOUCH = -18.0

    # Behaviour 2 : Slouching
    SLOUCH_SPINE_TILT_MAX = 12.0
    SLOUCH_NOSE_DROP_NORM = 0.15
    SLOUCH_HIP_FORWARD_NORM = 0.10

    # Behaviour 4 : Bounding (large body displacement)
    BOUND_DISP_NORM = 0.25
    BOUND_CONFIRM_FRAMES = 3
    BOUND_DECAY_FRAMES = 8
    BOUND_HISTORY_LEN = 5

    # Behaviour 5 : Hand raise
    HAND_ABOVE_MARGIN_NORM = 0.08
    ELBOW_ABOVE_SHOULDER = True
    HAND_VEL_NORM = 0.06
    HAND_RAISE_HOLD = 6
    HAND_LOWER_HOLD = 15
