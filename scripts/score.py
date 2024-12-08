from flask import request, jsonify
import joblib
import os

def init():
    global rf_model
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'random_forest_model.pkl')
    rf_model = joblib.load(model_path)

def run(data):
    try:
        data = request.get_json(force=True)

        # Convertir el género a one-hot encoding
        gender = data['gender']
        gender_female = 1.0 if gender == 'Female' else 0.0
        gender_male = 1.0 if gender == 'Male' else 0.0
        
        # Formatear los datos en el formato que el modelo espera
        features = [
            data['age'], data['annual_income'], data['total_spent'], 
            data['num_purchases'], data['avg_purchase_value'], data['online_activity_score'], 
            data['loyalty_program'], data['days_since_last_purchase'], data['num_site_visits'], 
            gender_female, gender_male
        ]
        
        # Realiza la predicción
        result = rf_model.predict([features])
        return jsonify({'prediction': result.tolist()})
        
    except Exception as e:
        error = jsonify({'error': str(e)}), 500
        return error