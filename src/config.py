# config.py
"""
Central configuration file for file paths, model names, and training parameters.
"""

import os

# --- PATHS ---
# Using os.path.join for cross-platform compatibility
INPUT_DIR = '/kaggle/input/hilabs'
MODEL_DIR = './models'
OUTPUT_DIR = './output'

# Input files
RXNORM_FILE_PATH = os.path.join(INPUT_DIR, 'rxnorm_all_data.parquet')
SNOMED_FILE_PATH = os.path.join(INPUT_DIR, 'snomed_all_data.parquet')
TEST_FILE_PATH = os.path.join(INPUT_DIR, 'Test.xlsx')

# Output files and directories
# Ensure output directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

FINETUNED_MODEL_PATH = os.path.join(MODEL_DIR, 'fine_tuned_clinical_model')
BM25_INDEX_FILE = os.path.join(MODEL_DIR, 'bm25_index_FULL.pkl')
MAPPING_FILE = os.path.join(MODEL_DIR, 'index_to_info_FULL.pkl')
OUTPUT_CSV_FILE = os.path.join(OUTPUT_DIR, 'df_result_FULL_hybrid_ensemble.csv')


# --- MODEL CONFIGURATION ---
BASE_MODEL_NAME = 'all-MiniLM-L6-v2'

# --- TRAINING PARAMETERS ---
NUM_TRAINING_EPOCHS = 1
TRAIN_BATCH_SIZE = 16
MAX_TRAINING_SAMPLES = 100000

# --- HARMONIZATION PARAMETERS ---
TOP_N_CANDIDATES = 50
CONFIDENCE_THRESHOLD = 0.70

# Dynamic weighting parameters for the ensemble model
SHORT_QUERY_THRESHOLD = 3
SEMANTIC_WEIGHT_SHORT = 0.60
FUZZY_WEIGHT_SHORT = 0.40
SEMANTIC_WEIGHT_LONG = 0.80
FUZZY_WEIGHT_LONG = 0.20

# --- PREPROCESSING ---
# List of noise terms to remove during aggressive preprocessing.
NOISE_TERMS = [
    'vfc-', 'take 1 tablet(s)', 'by oral route', 'every day',
    'as needed', 'hfa breath activated', 'dose', 'oral', 'tablet',
    'mg', 'ml', 'injection', 'solution', '(six)', 'sig:', 'tabs'
]
