# harmonizer.py
"""
Contains the Harmonizer class for building the search index, performing matching,
and ensembling the results.
"""
import os
import pickle
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util
from thefuzz import fuzz
from tqdm import tqdm
from config import (
    FINETUNED_MODEL_PATH, BM25_INDEX_FILE, MAPPING_FILE,
    TOP_N_CANDIDATES, CONFIDENCE_THRESHOLD, SHORT_QUERY_THRESHOLD,
    SEMANTIC_WEIGHT_SHORT, FUZZY_WEIGHT_SHORT, SEMANTIC_WEIGHT_LONG, FUZZY_WEIGHT_LONG
)
from data_loader import DataLoader

class Harmonizer:
    """Handles the harmonization process using a hybrid ensemble method."""

    def __init__(self, knowledge_base_df, test_df):
        self.knowledge_base_df = knowledge_base_df
        self.test_df = test_df
        self.model = None
        self.bm25 = None
        self.index_to_info = None

    def _build_or_load_indexes(self):
        """Builds or loads the BM25 index and the info mapping file."""
        print("\n--- Building or Loading Indexes ---")
        if os.path.exists(BM25_INDEX_FILE) and os.path.exists(MAPPING_FILE):
            print("Loading existing BM25 index and mapping file...")
            with open(BM25_INDEX_FILE, 'rb') as f:
                self.bm25 = pickle.load(f)
            with open(MAPPING_FILE, 'rb') as f:
                self.index_to_info = pickle.load(f)
            if 'STR_PROCESSED' not in self.index_to_info[0]:
                print("\nFATAL ERROR: Stale index files detected. Please delete and re-run.")
                exit()
        else:
            print("Building new BM25 index and mapping file...")
            unique_df = self.knowledge_base_df.drop_duplicates(subset=['STR_PROCESSED']).reset_index(drop=True)
            self.index_to_info = unique_df[['CUI', 'STR', 'STR_PROCESSED', 'CODE', 'SYSTEM']].to_dict('records')
            
            tokenized_corpus = [doc.split() for doc in unique_df['STR_PROCESSED']]
            self.bm25 = BM25Okapi(tokenized_corpus)

            print("Saving indexes for future use...")
            with open(BM25_INDEX_FILE, 'wb') as f:
                pickle.dump(self.bm25, f)
            with open(MAPPING_FILE, 'wb') as f:
                pickle.dump(self.index_to_info, f)

    def _load_model(self):
        """Loads the fine-tuned sentence transformer model."""
        print("\n--- Loading Fine-Tuned Model ---")
        try:
            self.model = SentenceTransformer(FINETUNED_MODEL_PATH)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"FATAL ERROR: Could not load the fine-tuned model from '{FINETUNED_MODEL_PATH}'. {e}")
            raise

    def harmonize(self):
        """Runs the full harmonization pipeline on the test data."""
        self._build_or_load_indexes()
        self._load_model()
        
        print("\n--- Processing Test Data with Hybrid Ensemble Method ---")
        results = []
        for _, row in tqdm(self.test_df.iterrows(), total=len(self.test_df), desc="Harmonizing entities"):
            input_text = str(row['Input Entity Description'])
            processed_input = DataLoader.preprocess_text(input_text)
            
            tokenized_query = processed_input.split()
            candidate_indices = self.bm25.get_top_n(tokenized_query, range(len(self.index_to_info)), n=TOP_N_CANDIDATES)
            
            if not candidate_indices:
                results.append({"Input": input_text, "Ensemble Score": 0.0})
                continue

            semantic_weight, fuzzy_weight = (
                (SEMANTIC_WEIGHT_SHORT, FUZZY_WEIGHT_SHORT) if len(input_text.split()) < SHORT_QUERY_THRESHOLD
                else (SEMANTIC_WEIGHT_LONG, FUZZY_WEIGHT_LONG)
            )

            candidate_processed_strings = [self.index_to_info[i]['STR_PROCESSED'] for i in candidate_indices]
            
            input_embedding = self.model.encode(processed_input, convert_to_tensor=True)
            candidate_embeddings = self.model.encode(candidate_processed_strings, convert_to_tensor=True)
            semantic_scores = util.cos_sim(input_embedding, candidate_embeddings).flatten().tolist()
            
            best_candidate = None
            highest_score = -1

            for i, idx in enumerate(candidate_indices):
                semantic_score = semantic_scores[i]
                original_candidate_str = self.index_to_info[idx]['STR']
                fuzzy_score = fuzz.token_set_ratio(input_text, original_candidate_str) / 100.0
                
                ensemble_score = (semantic_score * semantic_weight) + (fuzzy_score * fuzzy_weight)
                
                if ensemble_score > highest_score:
                    highest_score = ensemble_score
                    best_candidate = {
                        "Input": input_text, "Standard Description": original_candidate_str,
                        "Predicted CUI": self.index_to_info[idx]['CUI'], "Standard Code": self.index_to_info[idx]['CODE'],
                        "Coding System": self.index_to_info[idx]['SYSTEM'], "Semantic Score": semantic_score,
                        "Fuzzy Score": fuzzy_score, "Ensemble Score": ensemble_score
                    }
            results.append(best_candidate)
            
        return pd.DataFrame(results)
