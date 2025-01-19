from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.utils import pad_sequences
import numpy as np
import json

# Load the tokenizer from the JSON file
try:
    with open("tokenizer.json", "r") as json_file:
        tokenizer_json = json.load(json_file)
    tokenizer = tokenizer_from_json(tokenizer_json)
except Exception as e:
    raise Exception(f"Error loading tokenizer: {str(e)}")

# Load the trained model
try:
    model = load_model("model.keras")
except Exception as e:
    raise Exception(f"Error loading model: {str(e)}")

# Define the labels for predictions
labels = ['AI-Generated', 'Human-Written', 'Plagiarized']

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "This is the home page for the Text Detection API!"

@app.route("/detect", methods=["POST"])
def detect():
    try:
        # Parse the JSON input
        data = request.get_json()
        if data is None or "text" not in data:
            return jsonify({"Error": "Invalid input. Please provide data in the format {'text': 'your text here'}."}), 400

        text = data["text"].strip()
        if not text:
            return jsonify({"Error": "Input text is empty. Please provide valid text for detection."}), 400

        # Tokenize and pad the input text
        tokenized_sequence = tokenizer.texts_to_sequences([text])
        text_sequences = pad_sequences(tokenized_sequence, maxlen=200, padding='post')

        # Predict using the loaded model
        predictions = model.predict(text_sequences)
        predicted_label = labels[np.argmax(predictions)]

        # Return the result as JSON
        json_data = {"Output": predicted_label, "Confidence": float(np.max(predictions))}
        return jsonify(json_data)

    except Exception as e:
        return jsonify({"Error": f"An error occurred: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
