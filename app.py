# Build Gradio Interface

# Import libraries
import gradio as gr
from src.inference import SentimentClassifier
import os

# Load model
MODEL_PATH = "./models/sentiment_model"
            

# Check if model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Please train the model first."
    )

classifier = SentimentClassifier(MODEL_PATH)

def predict_sentiment(text):
    """
    Wrapper function for gradio

    Args:
        text: User input text

    Returns:
        Formatted prediction string
    """
    if not text.strip():
        return "Please enter some text to analyze."
    
    result = classifier.predict(text)

    # format output
    output = f"""
    **Prediction:** {result['label']}
    **Confidence:** {result['confidence']:.2%}

    **Probability Breakdown:**
    - Negative: {result['probabilities']['Negative']:.2%}
    - Positive: {result['probabilities']['Positive']:.2%}
    """
    return output

# Create Gradio interface
demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter a movie review here...",
        label="Movie Review"
    ),
    outputs=gr.Markdown(label="Sentiment Analysis Result"),
    title="🎬 Movie Review Sentiment Analyzer",
    description="""
    This model was fine-tuned on the IMDB dataset to classify movie reviews as positive or negative.
    Enter a movie review to see the sentiment prediction!
    """,
    examples=[
        ["This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout."],
        ["Waste of time. Poor acting, terrible script, and boring storyline."],
        ["It was decent. Some good moments but overall pretty average."]
    ],
    # theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=False) 
