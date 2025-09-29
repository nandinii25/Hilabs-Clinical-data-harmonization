# main.py
"""
Main script to orchestrate the entire data harmonization pipeline.
"""
import pandas as pd
import numpy as np
from data_loader import DataLoader
from model_trainer import ModelTrainer
from harmonizer import Harmonizer
from config import OUTPUT_CSV_FILE, CONFIDENCE_THRESHOLD

def main():
    """Executes the complete workflow."""
    # 1. Load and Preprocess Data
    data_loader = DataLoader()
    kb_df = data_loader.load_knowledge_base()
    test_df = data_loader.load_test_data()

    # 2. Train (Fine-Tune) the Model
    # This step is computationally expensive. You might run it once and then
    # comment it out for subsequent runs to just use the saved model.
    # trainer = ModelTrainer(kb_df)
    # trainer.train()

    # 3. Harmonize Data
    # Apply preprocessing to the knowledge base for the harmonizer
    kb_df_processed = data_loader.apply_preprocessing_to_kb()
    
    harmonizer = Harmonizer(kb_df_processed, test_df)
    results_df = harmonizer.harmonize()

    # 4. Format and Save Final Results
    print("\n--- Saving Final Harmonized Results ---")
    results_df.fillna(0, inplace=True)
    results_df['REVIEW_FLAG'] = np.where(results_df['Ensemble Score'] < CONFIDENCE_THRESHOLD, 'YES', 'NO')

    for col in ['Semantic Score', 'Fuzzy Score', 'Ensemble Score']:
        results_df[col] = results_df[col].map('{:.2f}'.format)

    final_columns = [
        'Input', 'Standard Description', 'Predicted CUI', 'Standard Code',
        'Coding System', 'Ensemble Score', 'Semantic Score', 'Fuzzy Score', 'REVIEW_FLAG'
    ]
    results_df = results_df.reindex(columns=final_columns).fillna('N/A')

    results_df.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"Results successfully saved to '{OUTPUT_CSV_FILE}'")
    print("\n--- SCRIPT FINISHED SUCCESSFULLY ---")


if __name__ == '__main__':
    main()
