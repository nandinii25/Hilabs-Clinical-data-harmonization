# Clinical Text Harmonization Engine  

This project provides a complete pipeline for harmonizing clinical text descriptions by matching them to standard medical terminologies (RxNorm and SNOMED CT). It uses a fine-tuned sentence transformer model combined with traditional text matching for robust performance.  

## Project Structure  

The project is organized into several Python modules for clarity and maintainability:  

- **`config.py`**: A centralized file for all configurations, including file paths, model names, and algorithm parameters.  
- **`data_loader.py`**: Contains the `DataLoader` class responsible for loading the knowledge base (RxNorm, SNOMED) and test datasets, and for text preprocessing.  
- **`model_trainer.py`**: Contains the `ModelTrainer` class, which handles the fine-tuning of the SentenceTransformer model on the clinical terminology data.  
- **`harmonizer.py`**: Contains the `Harmonizer` class, which implements the core logic for matching input text to the knowledge base using a hybrid approach (BM25 for candidate retrieval, fine-tuned model for semantic scoring, and fuzzy matching).  
- **`main.py`**: The main entry point of the application that orchestrates the entire workflow from data loading to saving the final results.  

## How to Run  

### Installation:  
Make sure you have all the required libraries installed.  

```bash  
pip install pandas pyarrow sentence-transformers faiss-cpu rank_bm25 thefuzz openpyxl  
```  

### Data:  
Place your input files (`rxnorm_all_data.parquet`, `snomed_all_data.parquet`, `Test.xlsx`) in the directory specified in `config.py` (default is `/kaggle/input/hilabs/`).  

### Execution:  
Run the main script from your terminal.  

```bash  
python main.py  
```  

- The first time you run the script, it will fine-tune the model and build the necessary indexes. This can be time-consuming.  
- On subsequent runs, the script will load the saved model and indexes, making the process much faster. If you want to retrain, you can either delete the saved model files or uncomment the training call in `main.py`.  

## Workflow Overview  

1. **Data Loading & Preprocessing**:  
    The `DataLoader` loads the RxNorm and SNOMED CT datasets, combines them into a single knowledge base, and applies aggressive text cleaning to both the knowledge base and the input text.  

2. **Model Fine-Tuning (Optional on subsequent runs)**:  
    The `ModelTrainer` creates training pairs from synonyms in the knowledge base and fine-tunes a SentenceTransformer model to better understand the semantics of clinical text. The trained model is saved to disk.  

3. **Indexing**:  
    The `Harmonizer` creates a fast BM25 keyword index from the preprocessed knowledge base to quickly retrieve relevant candidates for any given input term.  

4. **Harmonization & Ensembling**:  
    For each term in the test set, the `Harmonizer`:  
    - Uses the BM25 index to fetch the top N most likely candidates.  
    - Calculates a semantic score for these candidates using the fine-tuned model.  
    - Calculates a fuzzy match score.  
    - Combines these scores using a dynamic weighting system to produce a final ensemble score.  

5. **Output**:  
    The final results, including the best match, scores, and a flag for manual review, are saved to a CSV file.  

This structured approach makes the code easier to understand, test, and extend in the future.  