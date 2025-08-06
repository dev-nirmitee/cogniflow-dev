# Imports
import os
from functools import wraps

from flask import jsonify, request

def load_api_keys():
    # get the API keys from environment variables
    api_keys = os.getenv('API_KEYS', '').split(',')
    api_keys = {key.split(':')[0]: key.split(':')[1] for key in api_keys if ':' in key}
    return api_keys

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Apikey '):
            return jsonify({"status": "error", "message": "Missing or Invalid Authorization"}), 401
        api_key = auth_header.split(' ')[1]
        api_keys = load_api_keys()
        project_id = api_keys.get(api_key)
        if not project_id:
            return jsonify({"status": "error", "message": "Authorization failed"}), 403
        req_json = request.get_json()
        if not req_json:
            return jsonify({"status": "error", "message": "Request body is required"}), 400
        if req_json.get('project_id') != project_id:
            return jsonify({"status": "error", "message": "Authorization failed"}), 403
        return f(*args, **kwargs)    
    return decorated_function

def project_ids_for_data_extraction():
    # get the project IDs for data extraction from environment variables
    project_ids = os.getenv('EXTRACT_DATA', '').split(',')
    project_ids = [project_id.strip() for project_id in project_ids if project_id.strip()]
    return project_ids