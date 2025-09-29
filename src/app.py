# app.py
"""
A simple web interface using Gradio to interact with the Harmonizer model.
"""

import gradio as gr
import pandas as pd
from data_loader import DataLoader
from harmonizer import Harmonizer

print("Initializing application... This may take a moment.")

# --- 1. Load data and initialize harmonizer ONCE at startup ---
try:
    data_loader = DataLoader()
    # Preprocessing the KB is needed for index building
    kb_df_processed = data_loader.apply_preprocessing_to_kb()
    
    # The harmonizer will load the model and indexes on initialization
    harmonizer = Harmonizer(kb_df_processed)
    print("Initialization complete. Gradio is starting.")
except Exception as e:
    print(f"FATAL ERROR during initialization: {e}")
    # Set harmonizer to None if setup fails, so the UI can show an error
    harmonizer = None

# --- 2. Define the prediction function that Gradio will call ---
def harmonize_interface(input_text):
    """
    Takes a single text input, harmonizes it, and returns a formatted DataFrame.
    """
    if not harmonizer:
        return pd.DataFrame({"Error": ["Application failed to initialize. Please check the console logs."]})
        
    if not input_text or not input_text.strip():
        return pd.DataFrame() # Return an empty dataframe if the input is empty

    # Get the dictionary of the best match from the harmonizer
    result_dict = harmonizer.harmonize_single(input_text)

    if result_dict is None:
        return pd.DataFrame({"Status": ["No suitable candidates found for the given input."]})

    # Convert the single result dictionary to a DataFrame for clean display
    df = pd.DataFrame([result_dict])

    # Format score columns for better readability
    for col in ['Semantic Score', 'Fuzzy Score', 'Ensemble Score']:
        df[col] = df[col].map('{:.2f}'.format)
        
    # Reorder columns for a clean presentation in the UI
    display_columns = [
        'Standard Description', 'Ensemble Score', 'Predicted CUI', 'Standard Code',
        'Coding System', 'Semantic Score', 'Fuzzy Score'
    ]
    df = df[display_columns]
    
    return df

# --- 3. Create and launch the Gradio interface ---
iface = gr.Interface(
    fn=harmonize_interface,
    inputs=gr.Textbox(
        lines=2, 
        placeholder="Enter a clinical term here (e.g., 'mri pelvis', 'aspirin 81 mg cap')...",
        label="Input Clinical Term"
    ),
    outputs=gr.DataFrame(
        headers=['Standard Description', 'Score', 'CUI', 'Code', 'System', 'Semantic', 'Fuzzy'],
        datatype=['str', 'number', 'str', 'str', 'str', 'number', 'number'],
        label="Harmonization Result"
    ),
    title="Clinical Text Harmonization Engine",
    description="""
    Enter a clinical text description to match it against standard medical terminologies (RxNorm and SNOMED CT).
    The engine uses a fine-tuned AI model and fuzzy string matching to find the best fit.
    """,
    allow_flagging="never",
    examples=[
        ["mri pelvis"],
        ["aspirin 81 mg cap"],
        ["sore thraot"],
        ["doxycycline 100mg bid"]
    ]
)

if __name__ == "__main__":
    iface.launch()
