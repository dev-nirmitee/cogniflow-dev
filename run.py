import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin

from config import DevelopmentConfig, ProductionConfig
from authenticator import require_api_key, project_ids_for_data_extraction
from app.routes.run_models import run_imh_model, run_guarvis_model
from data_extraction.mongodb_connector import MongoDBConnector
from data_extraction.mongodb_query_builder import data_query
from data_extraction.data_reformatting import guarvis_input_data, imh_input_data


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
@require_api_key
def index():
    return jsonify({"message": "Welcome to the Cogniflow API!"})

@app.route('/api/imh', methods=['POST'])
@require_api_key
def imh_model():
    req_json = request.get_json()
    user_id = req_json.get('user_id')
    project_id = req_json.get('project_id')
    # data fetching for selected project_ids
    if project_id in project_ids_for_data_extraction():
        mongodb_uri = os.getenv('mongo_uri')
        mongo_db_name = 'imh'
        mongodb_connector = MongoDBConnector(mongodb_uri, mongo_db_name)
        query = data_query(user_id, project_id)
        activity_data = mongodb_connector.fetch_data('activity', query)
        activity_type_data = mongodb_connector.fetch_data('activity_type', query)
        location_data = mongodb_connector.fetch_data('location', query)
        mongodb_connector.close()
        input_data = imh_input_data(activity_data, activity_type_data, location_data)
    try:
        result = run_imh_model(input_data)
        insert_result = mongodb_connector.insert_result(result, 'imh_results', user_id, project_id)
        return jsonify({"status": "success", "data": result, "message": "IMH model executed successfully"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/guarvis', methods=['POST'])
@require_api_key
def guarvis_model():
    req_json = request.get_json()
    user_id = req_json.get('user_id')
    project_id = req_json.get('project_id')
    # data fetching for selected project_ids
    if project_id in project_ids_for_data_extraction():
        mongodb_uri = os.getenv('mongo_uri')
        mongo_db_name = 'guarvis'
        mongodb_connector = MongoDBConnector(mongodb_uri, mongo_db_name)
        query = data_query(user_id, project_id)
        activity_data = mongodb_connector.fetch_data('activity', query)
        sleep_data = mongodb_connector.fetch_data('sleep', query)
        mongodb_connector.close()
        input_data = guarvis_input_data(activity_data, sleep_data)
    try:
        result = run_guarvis_model(input_data)
        insert_result = mongodb_connector.insert_result(result, 'guarvis_results', user_id, project_id)
        return jsonify({"status": "success", "data": result, "message": "Guarvis model executed successfully"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
