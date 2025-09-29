# data_loader.py
"""
Contains the DataLoader class for loading and preprocessing all datasets.
"""

import pandas as pd
import re
import gc
from config import RXNORM_FILE_PATH, SNOMED_FILE_PATH, TEST_FILE_PATH, NOISE_TERMS

class DataLoader:
    """Handles loading, combining, and preprocessing of datasets."""
    def __init__(self):
        self.knowledge_base_df = None
        self.test_df = None

    def load_knowledge_base(self):
        """Loads and combines RxNorm and SNOMED CT datasets."""
        print("--- Loading and Preparing Knowledge Base Data ---")
        try:
            rxnorm_df = pd.read_parquet(RXNORM_FILE_PATH)
            rxnorm_df.columns = [col.upper() for col in rxnorm_df.columns]
            rxnorm_df['SYSTEM'] = 'RxNorm'

            snomed_df = pd.read_parquet(SNOMED_FILE_PATH)
            snomed_df.columns = [col.upper() for col in snomed_df.columns]
            snomed_df['SYSTEM'] = 'SNOMED CT'

            self.knowledge_base_df = pd.concat([rxnorm_df, snomed_df], ignore_index=True)
            print(f"Full knowledge base loaded with {len(self.knowledge_base_df):,} rows.")

            # Clean data by dropping rows with missing STR values
            initial_rows = len(self.knowledge_base_df)
            self.knowledge_base_df.dropna(subset=['STR'], inplace=True)
            print(f"Removed {initial_rows - len(self.knowledge_base_df)} rows with missing STR values.")

            del rxnorm_df, snomed_df
            gc.collect()
            return self.knowledge_base_df

        except Exception as e:
            print(f"FATAL ERROR during knowledge base loading: {e}")
            raise

    def load_test_data(self):
        """Loads the test data from the Excel file."""
        print("--- Loading Test Data ---")
        try:
            self.test_df = pd.read_excel(TEST_FILE_PATH)
            print("Test file loaded successfully.")
            return self.test_df
        except Exception as e:
            print(f"FATAL ERROR during test data loading: {e}")
            raise

    @staticmethod
    def preprocess_text(text):
        """
        An aggressive cleaning function to remove clinical noise.
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()
        for noise in NOISE_TERMS:
            text = text.replace(noise, '')

        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())

        return text

    def apply_preprocessing_to_kb(self):
        """Applies the aggressive preprocessing to the knowledge base."""
        if self.knowledge_base_df is None:
            self.load_knowledge_base()
        print("Applying aggressive preprocessing to knowledge base...")
        self.knowledge_base_df['STR_PROCESSED'] = self.knowledge_base_df['STR'].apply(self.preprocess_text)
        return self.knowledge_base_df
