import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# MODEL CONFIGURATION (EXACTLY from your notebook)
VOC_SIZE = 5000
MAX_LEN = 20  # sent_length from your notebook
EMBEDDING_DIM = 40

# Load model ONCE at startup
try:
    model = load_model('fake_news_model.h5')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Model load error: {e}")
    model = None

# Load or recreate tokenizer (EXACTLY as in your notebook)
try:
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print("✅ Tokenizer loaded")
except:
    print("⚠️ Creating tokenizer from scratch (voc_size=5000)")
    tokenizer = Tokenizer(num_words=VOC_SIZE, oov_token="<OOV>")

def preprocess_text(text):
    """EXACT preprocessing from your FAKE_NEWS_CLASSIFIER.ipynb"""
    # Tokenize
    sequence = tokenizer.texts_to_sequences([text])
    # Pad to exact length (20) from your notebook
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    return padded

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" in JSON body'}), 400
        
        text = data['text']
        if not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        processed = preprocess_text(text)
        
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Predict (sigmoid output [0,1])
        prediction = model.predict(processed, verbose=0)
        probability = float(prediction[0][0])
        label = 1 if probability > 0.6 else 0  # Your threshold from notebook
        
        result = {
            'prediction': int(label),
            'probability': probability,
            'label_text': 'FAKE' if label == 1 else 'REAL',
            'confidence': round(max(probability, 1-probability), 4),
            'input_length': len(text.split())
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction for multiple texts"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'Missing "texts" array'}), 400
        
        processed = preprocess_text(texts)
        predictions = model.predict(processed, verbose=0)
        
        results = []
        for i, prob in enumerate(predictions):
            label = 1 if prob[0] > 0.6 else 0
            results.append({
                'text': texts[i][:100] + '...' if len(texts[i]) > 100 else texts[i],
                'prediction': int(label),
                'probability': float(prob[0]),
                'label_text': 'FAKE' if label == 1 else 'REAL'
            })
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Fake News API on http://0.0.0.0:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
