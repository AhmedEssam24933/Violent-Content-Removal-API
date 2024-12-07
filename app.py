from flask import Flask, request, jsonify
import torch
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer

# Initialize the Flask application
app = Flask(__name__)

# Specify the model path
model_path = r"E:\Fassla Projects\Remove Violet Content Model"

# Load the model and tokenizer
model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)

# Define class labels
class_labels = ["Not Hate Speech", "Hate Speech"]  # Updated labels

@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()
    
    # Extract `postId` and `text` fields
    post_id = data.get('postId')
    sample_text = data.get('text', '')

    # Encode the input text
    encoded_input = tokenizer(sample_text, return_tensors='pt')

    # Get model output
    output = model(**encoded_input)

    # Extract logits and convert them to probabilities using softmax
    logits = output.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the predicted class index and corresponding label
    predicted_class_index = torch.argmax(probabilities, dim=-1).item()
    predicted_label = class_labels[predicted_class_index]

    # Prepare the response to return with `postId` and predicted label
    result = {
        'postId': post_id,
        'label': predicted_label
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
