# Create training model

from transformers import (
    AutoModelForSequenceClassification, # A model loader specifically for classification
    TrainingArguments, # A configuration object that holds all training settings
    Trainer # actual training engine that does the work
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class ModelTrainer:
    """Handle model training and evaluation"""

    def __init__(self, model_name, num_labels=2):
        """
        Initialize Trainer

        Args:
            model_name: Pretrained model to fine-tune
            num_labels: Number of classes (2 for binary sentiment)
        """

        self.model_name = model_name
        self.num_labels = num_labels

        # Load pretrained model
        print(f"loading model: {model_name}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
    
    def setup_training(self, output_dir="../models/finetuned_model"):
        """
        Configure training parameters

        Args:
            output_dir: Where to save model checkpoints
        
        Returns:
            TrainingArguments object
        """

        training_args = TrainingArguments(
            output_dir=output_dir,

            # Training hyperparameters
            num_train_epochs=3,             # Number of full passes through data
            per_device_train_batch_size=8,   # How many examples to process (reviews) at once during training
            per_device_eval_batch_size=16,  # Batch size for evaluation (testing). Larger for evaluation (no gradient calculation)
            learning_rate=2e-5,             # How much to update the model's weights after each batch (2e-5 because the model is already pre-trained)
            weight_decay=0.1,              # Regularization technique to prevent overfitting

            # Evaluation and logging
            eval_strategy="epoch",    # Evaluate after each epoch
            save_strategy="epoch",          # Save checkpoint after each epoch
            logging_steps=100,              # Log metrics every 100 steps

            # Performance optimizations
            fp16=False,                     # Use 16-bit floating point instead of 32-bit. Set to False because it requires GPU with tensor cores (modern NVIDIA GPUs)
            dataloader_num_workers=2,       # Number of parallel processes for loading data

            # Save best model
            load_best_model_at_end=True,    # After training, load the checkpoint with best validation performance
            metric_for_best_model="accuracy",

            # Misc
            report_to="none",               # Where to send training logs, "none" prints to console
            seed=42                         # Sets random seed for reproducibility
        )

        return training_args

    def compute_metrics(self, eval_pred):
        """calculate metrics for evaluation

        Args:
            eval_pred: Tuple of (predictions, labels)

        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred

        # Get predicted class (highest logit)
        predictions = np.argmax(predictions, axis=1) # Argmax - Returns the index of the maximum value.

        # calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='binary'
        )

        return{
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, train_dataset, eval_dataset, training_args):
        """
        Train the model

        Args:
            train_dataset: Training data
            eval_dataset: Validation data
            training_args: Training configuration

        Returns:
            Trained model
        """

        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )

        # Train
        print("Starting training...")
        trainer.train()

        trainer.save_model()
        # Save the final model
        print(f"Model saved to {training_args.output_dir}")

        return trainer