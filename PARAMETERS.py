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

# ----------------- AUTOMATIC LABELING ----------------- #
LABELING_EPS_VALUE = 0.003

# ----------------- NEPTUNE AI ----------------- #
NEPTUNE_SEGMENTATION_PROJECT_PATH = 'kacper-marciniak/TCM-segmentation'
NEPTUNE_EXTRACTION_PROJECT_PATH = 'kacper-marciniak/TCM-extraction'