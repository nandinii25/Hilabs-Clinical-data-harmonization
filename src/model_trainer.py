# model_trainer.py
"""
Contains the ModelTrainer class to handle the fine-tuning of the SentenceTransformer model.
"""
import os
import random
from itertools import combinations
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from config import BASE_MODEL_NAME, FINETUNED_MODEL_PATH, NUM_TRAINING_EPOCHS, TRAIN_BATCH_SIZE, MAX_TRAINING_SAMPLES

class ModelTrainer:
    """Handles the fine-tuning of the sentence transformer model."""

    def __init__(self, knowledge_base_df):
        self.knowledge_base_df = knowledge_base_df
        self.model = None

    def _create_training_pairs(self):
        """Generates positive pairs of synonyms for the same medical concept (CUI)."""
        print("Generating positive training pairs from the knowledge base...")
        cui_groups = self.knowledge_base_df.groupby('CUI')['STR'].apply(list)
        training_pairs = []

        for cui, synonyms in cui_groups.items():
            unique_synonyms = list(set(synonyms))
            if len(unique_synonyms) > 1:
                for pair in combinations(unique_synonyms, 2):
                    training_pairs.append(InputExample(texts=[pair[0], pair[1]]))

        print(f"Created {len(training_pairs):,} total positive pairs.")
        return training_pairs

    def train(self):
        """Configures and runs the model fine-tuning pipeline."""
        print("\n--- Step 3: Fine-Tuning the Model ---")
        os.environ["WANDB_DISABLED"] = "true"

        training_examples = self._create_training_pairs()

        if not training_examples:
            print("FATAL ERROR: No training pairs could be generated.")
            return

        random.shuffle(training_examples)
        if len(training_examples) > MAX_TRAINING_SAMPLES:
            print(f"Limiting training examples to {MAX_TRAINING_SAMPLES:,} from {len(training_examples):,}")
            training_examples = training_examples[:MAX_TRAINING_SAMPLES]

        print(f"Loading base model: '{BASE_MODEL_NAME}'")
        self.model = SentenceTransformer(BASE_MODEL_NAME)

        train_dataloader = DataLoader(training_examples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)
        warmup_steps = int(len(train_dataloader) * NUM_TRAINING_EPOCHS * 0.1)

        print("Starting the fine-tuning process...")
        print(f"Epochs: {NUM_TRAINING_EPOCHS}, Batch Size: {TRAIN_BATCH_SIZE}, Warmup Steps: {warmup_steps}")
        print("This can take a significant amount of time depending on the data size and hardware.")

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=NUM_TRAINING_EPOCHS,
            warmup_steps=warmup_steps,
            output_path=FINETUNED_MODEL_PATH,
            show_progress_bar=True
        )

        print("\n--- Fine-Tuning Complete! ---")
        print(f"Your specialized model has been saved to: '{FINETUNED_MODEL_PATH}'")
