"""
API Flask pour les prédictions Iris
Cette API permet de faire des prédictions via des requêtes POST
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pandas as pd

app = Flask(__name__)
CORS(app)  # Permet les requêtes cross-origin depuis Streamlit

# Charger et entraîner les modèles au démarrage
def initialize_models():
    # Charger les données
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Normaliser
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Entraîner les modèles
    models = {
        'knn': KNeighborsClassifier(n_neighbors=5),
        'logistic': LogisticRegression(max_iter=1000),
        'decision_tree': DecisionTreeClassifier(),
        'svm': SVC(probability=True),
        'neural_network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)
    }
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
    
    return models, scaler

# Initialiser au démarrage
models, scaler = initialize_models()

# Mapping des classes
class_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

@app.route('/')
def home():
    return jsonify({
        'message': 'API Iris Prediction',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Faire une prédiction',
            '/models': 'GET - Liste des modèles disponibles',
            '/health': 'GET - Vérifier l\'état de l\'API'
        }
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'models_loaded': len(models)})

@app.route('/models', methods=['GET'])
def get_models():
    return jsonify({
        'available_models': list(models.keys()),
        'default_model': 'knn'
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données de la requête
        data = request.get_json()
        
        # Extraire les caractéristiques
        sepal_length = float(data.get('sepal_length', 0))
        sepal_width = float(data.get('sepal_width', 0))
        petal_length = float(data.get('petal_length', 0))
        petal_width = float(data.get('petal_width', 0))
        model_name = data.get('model', 'knn').lower()
        
        # Vérifier que le modèle existe
        if model_name not in models:
            return jsonify({
                'error': f'Modèle {model_name} non disponible',
                'available_models': list(models.keys())
            }), 400
        
        # Préparer les données
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        input_scaled = scaler.transform(input_data)
        
        # Faire la prédiction
        model = models[model_name]
        prediction = model.predict(input_scaled)[0]
        prediction_name = class_names[prediction]
        
        # Calculer les probabilités si possible
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_scaled)[0]
            probabilities = {
                'setosa': float(proba[0]),
                'versicolor': float(proba[1]),
                'virginica': float(proba[2])
            }
        
        # Retourner la réponse
        response = {
            'success': True,
            'model_used': model_name,
            'input': {
                'sepal_length': sepal_length,
                'sepal_width': sepal_width,
                'petal_length': petal_length,
                'petal_width': petal_width
            },
            'prediction': {
                'class': int(prediction),
                'species': prediction_name
            }
        }
        
        if probabilities:
            response['probabilities'] = probabilities
            response['confidence'] = max(probabilities.values())
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Prédire pour plusieurs échantillons à la fois"""
    try:
        data = request.get_json()
        samples = data.get('samples', [])
        model_name = data.get('model', 'knn').lower()
        
        if model_name not in models:
            return jsonify({
                'error': f'Modèle {model_name} non disponible'
            }), 400
        
        # Préparer les données
        input_data = np.array([
            [s['sepal_length'], s['sepal_width'], s['petal_length'], s['petal_width']]
            for s in samples
        ])
        input_scaled = scaler.transform(input_data)
        
        # Prédictions
        model = models[model_name]
        predictions = model.predict(input_scaled)
        
        # Formater les résultats
        results = []
        for i, pred in enumerate(predictions):
            result = {
                'sample_index': i,
                'prediction': {
                    'class': int(pred),
                    'species': class_names[pred]
                }
            }
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_scaled[i:i+1])[0]
                result['probabilities'] = {
                    'setosa': float(proba[0]),
                    'versicolor': float(proba[1]),
                    'virginica': float(proba[2])
                }
                result['confidence'] = float(max(proba))
            
            results.append(result)
        
        return jsonify({
            'success': True,
            'model_used': model_name,
            'total_samples': len(samples),
            'predictions': results
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🚀 API Flask démarrée sur http://localhost:5000")
    print("📋 Endpoints disponibles:")
    print("   GET  /          - Informations sur l'API")
    print("   GET  /health    - État de l'API")
    print("   GET  /models    - Liste des modèles")
    print("   POST /predict   - Faire une prédiction")
    print("   POST /predict_batch - Prédictions multiples")
    app.run(debug=True, host='0.0.0.0', port=5000)
