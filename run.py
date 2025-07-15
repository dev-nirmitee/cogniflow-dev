import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from config import DevelopmentConfig, ProductionConfig
from app.routes.run_models import run_imh_model, run_guarvis_model


# initialize Flask app with CORS support and configuration
load_dotenv(override=True)
app = Flask(__name__)
CORS(app)
if os.getenv('FLASK_ENV') == 'production':
    app.config.from_object(ProductionConfig)
else:
    app.config.from_object(DevelopmentConfig)

# application routes
@app.route('/')
def index():
    return jsonify({"message": "Welcome to the Cogniflow API!"})

@app.route('/api/imh', methods=['POST'])
def imh_model():
    try:
        result = run_imh_model()
        return jsonify({"status": "success", "data": result, "message": "IMH model executed successfully"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/guarvis', methods=['POST'])
def guarvis_model():
    try:
        result = run_guarvis_model()
        return jsonify({"status": "success", "data": result, "message": "Guarvis model executed successfully"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
