# Clinical Text Harmonization Engine  

![Banner](images/banner.png)

This project provides a complete pipeline for harmonizing clinical text descriptions by matching them to standard medical terminologies (RxNorm and SNOMED CT). It uses a fine-tuned sentence transformer model combined with traditional text matching for robust performance.  

## Project Structure  

- **`config.py`**: Centralized configuration for all file paths, model names, etc.  
- **`data_loader.py`**: Handles loading and preprocessing of all data.  
- **`model_trainer.py`**: Handles the fine-tuning of the AI model.  
- **`harmonizer.py`**: Implements the core matching and scoring logic.  
- **`main.py`**: The main script for batch processing an entire test file.  
- **`app.py`**: A web interface using Gradio for interactive testing.  
- **`requirements.txt`**: A list of all necessary Python packages.  

## Local Setup Instructions  

Follow these steps to set up and run the project on your local machine.  

### 1. Create a Virtual Environment  

It's highly recommended to use a virtual environment to avoid conflicts with other projects.  

```bash  
# Create a virtual environment named 'venv'  
python -m venv venv  

# Activate the virtual environment  
# On Windows:  
venv\Scripts\activate  
# On macOS/Linux:  
source venv/bin/activate  
```  

### 2. Install Dependencies  

Install all the required packages using the `requirements.txt` file.  

```bash  
pip install -r requirements.txt  
```  

### 3. Download Data  

Create a directory named `data` in the root of the project. Place your input files inside this `data` directory:  

- `rxnorm_all_data.parquet`  
- `snomed_all_data.parquet`  
- `Test.xlsx`  

## How to Run  

After setting up the environment, you can run the project in two modes.  

### A) First-Time Setup (Model Training)  

The very first time you run the project, you need to train the model.  

1. Open `main.py`.  
2. Uncomment the lines for the `ModelTrainer`:  

```python  
# print("\nStarting model training...")  
# trainer = ModelTrainer(kb_df)  
# trainer.train()  
```  

3. Run the script. This will train the model and save it to the `./models` directory. It will also build the search indexes. This is a one-time, potentially long-running step.  

```bash  
python main.py  
```  

4. Once training is complete, you can re-comment the training lines in `main.py` so it doesn't run every time.  

### B) Running the Application  

Once the model and indexes are created, you can run either the batch processing script or the interactive web app.  

#### Batch Processing  

To process the entire `Test.xlsx` file and generate a CSV output:  

```bash  
python main.py  
```  

The results will be saved in the `./output` directory.  

#### Interactive Web Interface  

![gradio app](images/gradio1.png)
To launch a user-friendly web app for testing individual terms:  

```bash  
python app.py  
```  



Open your web browser and navigate to the local URL shown in the terminal (usually `http://127.0.0.1:7860`).  