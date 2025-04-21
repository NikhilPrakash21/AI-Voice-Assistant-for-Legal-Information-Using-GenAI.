from flask import Flask, request, jsonify
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize
import json
from flask_cors import CORS
from deep_translator import GoogleTranslator

app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "http://localhost:3000"}})

# Load model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FILE = "data.pth"
intents_file = "intents.json"

# Load model parameters and initialize model
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

# Load intents
with open(intents_file, 'r') as f:
    intents = json.load(f)

# Translator setup
def translate_to_english(text):
    return GoogleTranslator(source='auto', target='en').translate(text)

def translate_to_tamil(text):
    return GoogleTranslator(source='en', target='ta').translate(text)

# Chat route
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data['message']
    print(message)
    # Translate input to English if necessary
    translated_message = translate_to_english(message)
    print(translated_message)
    sentence = tokenize(translated_message)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    # Model prediction
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Probability threshold
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["intent"]:
                response = intent["response"]

                # Translate response to Tamil if the input was Tamil
                if message != translated_message:  # Indicates original input was not in English
                    response = translate_to_tamil(response)
                
                return jsonify({"message": response})

    fallback_response = "I'm not sure how to respond to that."
    if message != translated_message:
        fallback_response = translate_to_tamil(fallback_response)

    return jsonify({"message": fallback_response})

if __name__ == "__main__":
    app.run(debug=True)
