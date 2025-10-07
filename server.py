#!/usr/bin/env python3
"""
EmbeddingGemma-300M Service
Runs the model locally with proper HuggingFace authentication
"""

import os
import json
from flask import Flask, request, jsonify
from transformers import AutoModel, AutoTokenizer
import torch
import redis

app = Flask(__name__)

# Environment variables
PORT = int(os.getenv('PORT', 3000))
REDIS_URL = os.getenv('REDIS_URL')
HF_TOKEN = os.getenv('HF_TOKEN')

if not REDIS_URL:
    raise ValueError('REDIS_URL environment variable is required')

if not HF_TOKEN:
    raise ValueError('HF_TOKEN environment variable is required')

# Initialize Redis
redis_client = redis.from_url(REDIS_URL)

# Login to HuggingFace
from huggingface_hub import login
print('Logging into HuggingFace...')
login(token=HF_TOKEN)

# Load model and tokenizer with authentication
print('Loading EmbeddingGemma-300M model...')
model_name = 'google/embeddinggemma-300m'

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN,
    trust_remote_code=True
)

print('Loading model...')
model = AutoModel.from_pretrained(
    model_name,
    token=HF_TOKEN,
    torch_dtype=torch.float16,  # Use FP16 for memory efficiency
    device_map='auto',  # Automatically use GPU if available
    trust_remote_code=True
)

print(f'EmbeddingGemma-300M model loaded successfully (768 dimensions)')
print(f'Device: {model.device}')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        redis_client.ping()
        redis_connected = True
    except:
        redis_connected = False

    return jsonify({
        'status': 'healthy',
        'model': 'embeddinggemma-300m',
        'dimensions': 768,
        'redis_connected': redis_connected
    })

@app.route('/generate-embedding', methods=['POST'])
def generate_embedding():
    """Generate 768-dimensional embedding for input text"""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400

        text = data['text']

        # Tokenize and generate embedding
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling over token embeddings
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            # Normalize
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)

        # Convert to list
        embedding_list = embedding.cpu().tolist()

        if len(embedding_list) != 768:
            return jsonify({
                'error': f'Unexpected embedding dimension: {len(embedding_list)}'
            }), 500

        return jsonify({'embedding': embedding_list})

    except Exception as e:
        print(f'Error generating embedding: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f'Starting EmbeddingGemma service on port {PORT}')
    app.run(host='0.0.0.0', port=PORT)
