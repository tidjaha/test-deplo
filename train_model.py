import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Génération de données factices
X = np.random.rand(100, 1) * 10  # 100 valeurs entre 0 et 10
y = 3.5 * X + np.random.randn(100, 1)  # y = 3.5x + bruit

# Entraînement du modèle
model = LinearRegression()
model.fit(X, y)

# Sauvegarder le modèle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Modèle entraîné et sauvegardé dans 'model.pkl'")
