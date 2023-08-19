import pickle as pkl
import numpy as np
import json
from flask import Flask, request, jsonify
import pandas as pd
import requests
import random;
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from flask import Flask
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
kmeans = KMeans(n_clusters=5, random_state=42)
drug_names=['metformin', 'repaglinide', 'nateglinide', 'glimepiride',
                         'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone',
                         'insulin', 'glyburide-metformin']

   
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data=data['new']
    data=np.array([data])
    
    def predict_drug_recommendation(new_patient_pca):
        patient_cluster = kmeans.predict(new_patient_pca.reshape(1, -1))[0]
        cosine_scores = cosine_sim_matrix[patient_cluster]
        predicted_dosages = np.dot(cosine_scores, collaborative_matrix)
        
        # Get indices of drugs with highest predicted dosages
        top_drug_indices = predicted_dosages.argsort()[-4:][::-1]  # Change 10 to the desired number of recommendations
        
        # Get the corresponding drug names
        recommended_drugs = [drug_names[i] for i in top_drug_indices if i < len(drug_names)]
        
        return recommended_drugs
    kmeans=pkl.load(open("781670.f1/model.pkl","rb"))
    scaler=pkl.load(open("781670.f1/scaler.pkl","rb"))
    pca=pkl.load(open("781670.f1/pca.pkl","rb"))
    collaborative_matrix=pkl.load(open("781670.f1/collaborative_matrix.pkl","rb"))
    cosine_sim_matrix=pkl.load(open("781670.f1/cosine_sim_matrix.pkl","rb"))
    new_patient_data_scaled = scaler.transform(data)
    new_patient = pca.transform(new_patient_data_scaled)
    prediction = predict_drug_recommendation(new_patient)
    return jsonify({"indices": prediction},)

if __name__ == "__main__":
    app.run()

# Second part of code

'''import pickle as pkl
import numpy as np
import json
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from flask_cors import CORS

app = Flask(__name__)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


drug_names = ['metformin', 'repaglinide', 'nateglinide', 'glimepiride',
              'glipizide', 'glyburide', 'pioglitazone', 'rosiglitazone',
              'insulin', 'glyburide-metformin']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    data = data['new']
    data = np.array(data)

    def predict_drug_recommendation(new_patient_pca):
        patient_cluster = kmeans.predict(new_patient_pca.reshape(1, -1))[0]
        cosine_scores = cosine_sim_matrix[patient_cluster]
        predicted_dosages = np.dot(cosine_scores, collaborative_matrix)

        # Get indices of drugs with highest predicted dosages
        top_drug_indices = predicted_dosages.argsort()[-4:][::-1]  # Change 10 to the desired number of recommendations

        # Get the corresponding drug names
        recommended_drugs = [drug_names[i] for i in top_drug_indices if i < len(drug_names)]

        return recommended_drugs

    scaler = pkl.load(open("781670.f1/scaler.pkl", "rb"))
    #781670.f1\scaler.pkl
    pca = pkl.load(open("781670.f1/pca.pkl", "rb"))
    collaborative_matrix = pkl.load(open("781670.f1/collaborative_matrix.pkl", "rb"))
    cosine_sim_matrix = pkl.load(open("781670.f1/cosine_sim_matrix.pkl", "rb"))

    new_patient_data_scaled = scaler.transform(data.reshape(1, -1))
    new_patient = pca.transform(new_patient_data_scaled)
    prediction = predict_drug_recommendation(new_patient)
    
    return jsonify({"indices": prediction})

if __name__ == "__main__":
    scaler = pkl.load(open("781670.f1/scaler.pkl", "rb"))
    #781670.f1\scaler.pkl
    pca = pkl.load(open("781670.f1/pca.pkl", "rb"))
    collaborative_matrix = pkl.load(open("781670.f1/collaborative_matrix.pkl", "rb"))
    cosine_sim_matrix = pkl.load(open("781670.f1/cosine_sim_matrix.pkl", "rb"))
    
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(collaborative_matrix)  # Fit the KMeans model with the collaborative matrix
    
    app.run(host='0.0.0.0', port=80)'''

# third part of code 
