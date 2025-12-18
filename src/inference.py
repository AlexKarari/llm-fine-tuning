from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class SentimentClassifier:
    """Load fine-tuned model and make predictions"""

    def __init__(self, model_path):
        """
        Load fine-tuned model

        Args:
            model_path: Path to saved model directory
        """
        print(f"Loading model from {model_path}")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Move model to GPU if available
        self.device = torch.device("cude" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval() # Set to evaluation mode

        # Label mapping
        self.labels = {0: "Negative", 1: "Positive"}

    def predict(self, text):
        """
        Predict sentiment of text

        Args:
            text: Input text to classify

        Returns:
            Dictionary with prediction and confidence
        """

        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        ).to(self.device)

        # Get prediction
        with torch.no_grad(): # disables gradient calculation (faster inference)
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1) # Converts logits to probabilities
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        return {
            "label": self.labels[predicted_class],
            "confidence": confidence,
            "probabilities": {
                "Negative": probabilities[0][0].item(),
                "Positive": probabilities[0][1].item()
            }
        }
    
    def predict_batch(self, texts):
        """
        Predict sentiment for multiple texts
        
        Args:
            texts: List of texts to classify
        
        Returns:
            List of predictions
        """
        return [self.predict(text) for text in texts]