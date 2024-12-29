from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Charger le modèle
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "Bienvenue dans l'API de régression !"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Charger les données JSON envoyées par le client
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)  # Prédire avec le modèle
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
