from flask import Flask, jsonify, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.utils import pad_sequences
import numpy as np
import json
import PyPDF2
import io

# Initialize Flask app
app = Flask(__name__)

# Define labels and constants
labels = ['AI-Generated', 'Human-Written', 'Plagiarized']
MAX_SEQUENCE_LENGTH = 200

# Load model and tokenizer once at startup
model = load_model("model2.keras")
with open("tokenizer.json", "r") as json_file:
    tokenizer_json = json.load(json_file)
    tokenizer = tokenizer_from_json(tokenizer_json)

@app.route("/")
def home():
    return "Welcome to the Text Detection API!"

@app.route("/detect", methods=["POST"])
def detect():
    try:
        # Case 1: If file is uploaded (PDF)
        if "file" in request.files:
            file = request.files["file"]
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        else:
            # Case 2: If JSON body contains text
            data = request.get_json()
            text = data.get("text", "").strip()

        if not text:
            return jsonify({"Error": "Invalid input or empty text. Provide text or upload a PDF."}), 400

        # Tokenize and pad the input text
        tokenized_sequence = tokenizer.texts_to_sequences([text])
        text_sequences = pad_sequences(tokenized_sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

        # Predict using the model
        predictions = model.predict(text_sequences, verbose=0)
        predicted_label = labels[np.argmax(predictions)]

        # Return the result
        return jsonify({"Output": predicted_label, "Confidence": float(np.max(predictions))})

    except Exception as e:
        return jsonify({"Error": f"An error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=False, port=5000)
