#!/usr/bin/env python3
"""
EmbeddingGemma-300M Service
Uses HuggingFace Inference API for reliable production deployment
"""

import os
import json
import requests
from flask import Flask, request, jsonify
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

# HuggingFace Inference API configuration
HF_API_URL = 'https://api-inference.huggingface.co/pipeline/feature-extraction/google/embeddinggemma-300m'

print('Using HuggingFace Inference API for EmbeddingGemma-300M')
print('Model: google/embeddinggemma-300m (768 dimensions)')
print('API Endpoint:', HF_API_URL)

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
    """Generate 768-dimensional embedding via HuggingFace API"""
    try:
        data = request.get_json()

        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400

        text = data['text']

        # Call HuggingFace Inference API
        response = requests.post(
            HF_API_URL,
            headers={
                'Authorization': f'Bearer {HF_TOKEN}',
                'Content-Type': 'application/json'
            },
            json={'inputs': text},
            timeout=30
        )

        if not response.ok:
            error_text = response.text
            print(f'HF API error: {response.status_code} - {error_text}')
            return jsonify({
                'error': f'HuggingFace API error: {response.status_code}',
                'details': error_text
            }), 500

        embedding = response.json()

        # Validate embedding dimensions
        if isinstance(embedding, list) and len(embedding) == 768:
            return jsonify({'embedding': embedding})
        else:
            print(f'Unexpected embedding format: {type(embedding)}')
            return jsonify({
                'error': f'Unexpected embedding format from API'
            }), 500

    except requests.Timeout:
        return jsonify({'error': 'HuggingFace API timeout'}), 504
    except Exception as e:
        print(f'Error generating embedding: {e}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f'Starting EmbeddingGemma service on port {PORT}')
    app.run(host='0.0.0.0', port=PORT)
