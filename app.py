from flask import Flask, request, jsonify
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
 
app = Flask(__name__)
 
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
 
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
 
accuracy = accuracy_score(y_test, model.predict(X_test_scaled))
CLASS_NAMES = wine.target_names.tolist()
FEATURE_NAMES = wine.feature_names
 
 
@app.route("/")
def home():
    return jsonify({
        "app": "Wine Classifier API",
        "description": "A simple ML API that classifies wine types using a Random Forest model.",
        "model_accuracy": f"{accuracy * 100:.2f}%",
        "endpoints": {
            "GET  /": "App info",
            "GET  /health": "Health check",
            "GET  /features": "List required input features",
            "POST /predict": "Predict wine class from feature values"
        }
    })
 
 
@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model": "RandomForestClassifier", "trained": True})
 
 
@app.route("/features")
def features():
    return jsonify({
        "description": "Send these 13 features as a JSON array to /predict",
        "features": FEATURE_NAMES,
        "example_request": {
            "features": [13.2, 2.77, 2.51, 18.5, 96.6, 1.04, 2.55, 0.57, 1.47, 6.2, 1.05, 3.33, 820]
        }
    })
 
 
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
 
    if not data or "features" not in data:
        return jsonify({"error": "Please provide a 'features' key with a list of 13 numeric values."}), 400
 
    features = data["features"]
 
    if len(features) != 13:
        return jsonify({"error": f"Expected 13 features, got {len(features)}."}), 400
 
    try:
        input_array = np.array(features).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
 
        return jsonify({
            "predicted_class": int(prediction),
            "wine_type": CLASS_NAMES[prediction],
            "confidence": f"{max(probabilities) * 100:.2f}%",
            "class_probabilities": {
                CLASS_NAMES[i]: f"{prob * 100:.2f}%"
                for i, prob in enumerate(probabilities)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
 
 
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
 