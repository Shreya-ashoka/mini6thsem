from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # This enables CORS for all routes

# Load the pre-trained model
with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from the incoming JSON data
    features = [
        data.get('Smokes', 0),
        data.get('HormonalContraceptives', 0),
        data.get('IUD', 0),
        data.get('STDs', 0),
        data.get('STDs_condylomatosis', 0),
        data.get('STDs_cervical_condylomatosis', 0),
        data.get('STDs_vaginal_condylomatosis', 0),
        data.get('STDs_vulvo_perineal_condylomatosis', 0),
        data.get('STDs_syphilis', 0),
        data.get('STDs_pelvic_inflammatory_disease', 0),
        data.get('STDs_genital_herpes', 0),
        data.get('STDs_molluscum_contagiosum', 0),
        data.get('STDs_AIDS', 0),
        data.get('STDs_HIV', 0),
        data.get('STDs_Hepatitis_B', 0),
        data.get('STDs_HPV', 0),
        data.get('Age', 0),
        data.get('Number_of_sexual_partners', 0),
        data.get('First_sexual_intercourse', 0),
        data.get('Num_of_pregnancies', 0),
        data.get('Smokes_years', 0),
        data.get('Smokes_packs_per_year', 0),
        data.get('HormonalContraceptives_years', 0),
        data.get('IUD_years', 0),
        data.get('STDs_number', 0),
        data.get('STDs_Number_of_diagnoses', 0),
        data.get('Dx_Cancer', 0),
        data.get('Dx_CIN', 0),
        data.get('Dx_HPV', 0),
        data.get('Dx', 0),
        data.get('Hinselmann', 0),
        data.get('Schiller', 0),
        data.get('Citology', 0)
    ]

    features = np.array(features).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(features)
    
    # Send prediction result
    result = int(prediction[0])
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
