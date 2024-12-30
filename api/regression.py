from flask import Flask, request, jsonify
import pickle
import pandas as pd

# Charger votre modèle de régression (ex: un modèle Pickle)
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Charger les données JSON envoyées par le client
    df = pd.DataFrame(data)  # Convertir les données en DataFrame pandas
    
    # Appliquez le modèle sur les données (assurez-vous que les colonnes correspondent)
    predictions = model.predict(df)
    
    return jsonify(predictions.tolist())  # Retourner les résultats sous forme de JSON

if __name__ == "__main__":
    app.run(debug=True)

