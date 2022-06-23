"""
Parameters used in TCM scripts
"""

# ----------------- PATHS ----------------- #
PATH_TRAINING_OUTPUT_DIR_SEGMENTATION = r"Z:\TCM\trening\segmentacja"
PATH_TRAINING_OUTPUT_DIR_EXTRACTION = r"Z:\TCM\trening\ekstrakcja"
PATH_MODELS_EXTRACTION = r"Z:\TCM\modele\ekstrakcja"
PATH_MODELS_SEGMENTATION = r"Z:\TCM\modele\segmentacja"
PATH_TRAINING_DATA_SEGMENTATION = r"Z:\TCM\trening\DATASETY\segmentacja"
PATH_TRAINING_DATA_EXTRACTION = r"Z:\TCM\trening\DATASETY\ekstrakcja"
PATH_DASH_SQL_DIR = r"Z:\TCM\DATA\dash_sql"
PATH_EXTRACTED_TEETH_DIR = r"Z:\TCM\DATA\extracted_teeth"

# ----------------- FAILURES ----------------- #
DICTIONARY_FAILURES = {
    0:"wykruszenie",
    1:"narost",
    2:"stepienie",
    3:"zatarcie"   
}

# ----------------- DEFAULT MODEL PARAMETERS ----------------- #
DEFAULT_THRESH_EXTRACTION = 0.70
DEFAULT_THRESH_SEGMENTATION = 0.85

# ----------------- AUTOMATIC LABELING ----------------- #
LABELING_EPS_VALUE = 0.003

# ----------------- NEPTUNE AI ----------------- #
NEPTUNE_SEGMENTATION_PROJECT_PATH = 'kacper-marciniak/TCM-segmentation'
NEPTUNE_EXTRACTION_PROJECT_PATH = 'kacper-marciniak/TCM-extraction'

# ----------------- SQL ----------------- #
SQL_SERVER = r'DESKTOP-5LI4OVK\SQLEXPRESS'
SQL_DATABASE_NAME = r'TCM_database'
SQL_PORT = 1433

# ----------------- BACKBONES DETECTRON ----------------- #
BACKBONES_EXTRACTION = [
    "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml",
    "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml",
    "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml",
    "COCO-Detection/faster_rcnn_R_50_C4_3x.yaml",
    "COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml",
    "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
    "COCO-Detection/faster_rcnn_R_101_C4_3x.yaml",
    "COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml",
    "COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml",
    "COCO-Detection/faster_rcnn_X_101_3x.yaml"
]

BACKBONES_SEGMENTATION = [
    "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml",
    "COCO-InstanceSegmentation/mask_rcnn_X_101_FPN_3x.yaml",
]