#!/usr/bin/env python3
"""
EmbeddingGemma-300M Service
HuggingFace Inference API version - lightweight, no local model loading
"""

import os
import sys
import requests
from flask import Flask, request, jsonify
import redis
import logging

# Configure logging to stderr to avoid corrupting HTTP responses
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)

app = Flask(__name__)

# Disable Flask's default logger to prevent it from writing to stdout
import logging as flask_logging
werkzeug_logger = flask_logging.getLogger('werkzeug')
werkzeug_logger.handlers = []
werkzeug_logger.addHandler(flask_logging.StreamHandler(sys.stderr))
app.logger.handlers = []
app.logger.addHandler(flask_logging.StreamHandler(sys.stderr))

# Global error handler to ensure JSON responses
@app.errorhandler(Exception)
def handle_error(error):
    """Ensure all errors return JSON"""
    response = {
        'error': 'Internal server error',
        'details': str(error)
    }
    return jsonify(response), 500

@app.errorhandler(400)
def handle_bad_request(error):
    """Handle bad request errors"""
    response = {
        'error': 'Bad request',
        'details': str(error)
    }
    return jsonify(response), 400

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

# HuggingFace Inference API settings
HF_API_URL = "https://api-inference.huggingface.co/models/google/embeddinggemma-300m"
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

logging.info('Using HuggingFace Inference API for embeddings')
logging.info('Model: google/embeddinggemma-300m')
logging.info('Expected dimensions: 768')

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
        'api': 'huggingface-inference',
        'dimensions': 768,
        'redis_connected': redis_connected
    })

@app.route('/generate-embedding', methods=['POST'])
def generate_embedding():
    """Generate 768-dimensional embedding using HuggingFace Inference API"""
    try:
        # Handle JSON parsing errors
        try:
            data = request.get_json(force=True)
        except Exception as json_error:
            return jsonify({
                'error': 'Invalid JSON in request body',
                'details': str(json_error)
            }), 400

        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400

        text = data['text']

        # Validate text is a string
        if not isinstance(text, str):
            return jsonify({'error': 'Text field must be a string'}), 400

        if not text.strip():
            return jsonify({'error': 'Text field cannot be empty'}), 400

        # Call HuggingFace Inference API
        payload = {"inputs": text}

        logging.info(f'Calling HF Inference API for text: {text[:50]}...')
        response = requests.post(HF_API_URL, headers=HF_HEADERS, json=payload, timeout=30)

        if response.status_code != 200:
            logging.error(f'HF API error: {response.status_code} - {response.text}')
            return jsonify({
                'error': 'HuggingFace API error',
                'details': f'Status {response.status_code}: {response.text[:200]}'
            }), 500

        # Parse the response
        result = response.json()

        # HF Inference API returns the embedding directly as an array
        if isinstance(result, list):
            embedding_list = result
        else:
            # Some models return {"embedding": [...]}
            embedding_list = result.get('embedding', result)

        # Validate dimensions
        if len(embedding_list) != 768:
            return jsonify({
                'error': f'Unexpected embedding dimension: {len(embedding_list)}, expected 768'
            }), 500

        # Check for NaN or Inf values
        import math
        if any(math.isnan(x) or math.isinf(x) for x in embedding_list):
            logging.error(f'Invalid embedding values detected (NaN/Inf). Input text: {text[:100]}')
            return jsonify({
                'error': 'Model generated invalid embedding values (NaN/Inf)',
                'details': 'This may indicate a model configuration issue. Contact support.'
            }), 500

        return jsonify({'embedding': embedding_list}), 200

    except requests.exceptions.Timeout:
        logging.error('HF Inference API timeout')
        return jsonify({
            'error': 'Request timeout',
            'details': 'HuggingFace API took too long to respond'
        }), 504
    except requests.exceptions.RequestException as e:
        logging.error(f'HF Inference API request error: {e}')
        return jsonify({
            'error': 'API request failed',
            'details': str(e)
        }), 500
    except Exception as e:
        logging.error(f'Error generating embedding: {e}', exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    logging.info(f'Starting EmbeddingGemma service on port {PORT}')
    logging.info('Using HuggingFace Inference API (lightweight mode)')
    # Disable Flask development server logging to stdout
    import logging as log_module
    log = log_module.getLogger('werkzeug')
    log.setLevel(log_module.ERROR)
    app.run(host='0.0.0.0', port=PORT)
