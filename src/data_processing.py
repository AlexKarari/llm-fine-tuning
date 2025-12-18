# Import libraries

from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

class DataPreprocessor:
    """Handle dataset loading and preproessing for fine-tuning."""

    def __init__(self, model_name, max_length=512):
        """Initialize preprocessor

        Args:
            model_name: Name of the model (for tokenizer)
            max_length: Maximum sequence length for tokenization. Defaults to 512.
        """
        self.model_name = model_name
        self.max_length = max_length # short texts will be padded (filled with empty tokens). Long texts truncated.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) # smart loader that automatically picks the right tokenizer for the model

        # some models do not have a pad token, so we add one:
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_and_prepare_data(self, dataset_name, subset_size=None):
        """
        Load dataset and prepare for training.

        Args:
            dataset_name: _descriptName of dataset on HuggungFace
            subset_size: Optional - use smaller subset for testing

        Returns:
            Processed dataset ready for training
        """

        # Load dataset
        print(f"Loading {dataset_name}...")
        dataset = load_dataset(dataset_name)

        # Optional: Use subset for faster experimentation
        if subset_size:
            # SHUFFLE FIRST to get balanced data!
            dataset['train'] = dataset['train'].shuffle(seed=42).select(range(subset_size))
            dataset['test'] = dataset['test'].shuffle(seed=42).select(range(subset_size // 5))

        # Tokenize the dataset
        print("Tokenizing the dataset...")
        
        tokenized_dataset = dataset.map(
            self._tokenize_function,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        return tokenized_dataset
    
    def _tokenize_function(self, examples):
        """
        Tokenize text data

        Args:
            examples: Batch of examples from dataset

        Returns:
            Tokenized examples
        """

        # Tokenize the texts
        tokenized = self.tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors=None # Return lists, not tensor
        )

        # Add labels (for classification)
        tokenized['labels'] = examples['label']

        return tokenized


