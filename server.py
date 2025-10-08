#!/usr/bin/env python3
"""
EmbeddingGemma-300M Service
Local model loading with PyTorch 2.5.0+ and transformers 4.50.0+
"""

import os
import sys
import torch
from flask import Flask, request, jsonify
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login
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

# Login to HuggingFace
logging.info('Logging into HuggingFace...')
login(token=HF_TOKEN)

# Load model and tokenizer
logging.info('Loading EmbeddingGemma-300M model...')
model_name = 'google/embeddinggemma-300m'

logging.info('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    token=HF_TOKEN,
    trust_remote_code=True
)

logging.info('Loading model (this may take a few minutes)...')
model = AutoModel.from_pretrained(
    model_name,
    token=HF_TOKEN,
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True
)

logging.info(f'Model loaded successfully on device: {model.device}')
logging.info('Model produces 768-dimensional embeddings')

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
        'device': str(model.device),
        'redis_connected': redis_connected
    })

@app.route('/generate-embedding', methods=['POST'])
def generate_embedding():
    """Generate 768-dimensional embedding locally"""
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

        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        )

        # Move inputs to model device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate embedding
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling over sequence dimension
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            # L2 normalization
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)

        # Convert to list and return
        embedding_list = embedding.cpu().tolist()

        # Validate dimensions
        if len(embedding_list) != 768:
            return jsonify({
                'error': f'Unexpected embedding dimension: {len(embedding_list)}, expected 768'
            }), 500

        return jsonify({'embedding': embedding_list}), 200

    except Exception as e:
        logging.error(f'Error generating embedding: {e}', exc_info=True)
        return jsonify({
            'error': 'Internal server error',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    logging.info(f'Starting EmbeddingGemma service on port {PORT}')
    # Disable Flask development server logging to stdout
    import logging as log_module
    log = log_module.getLogger('werkzeug')
    log.setLevel(log_module.ERROR)
    app.run(host='0.0.0.0', port=PORT)
