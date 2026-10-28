
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("student_result_model.pkl")


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Student Result Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}

    required_fields = [
        "attendance",
        "internal_marks",
        "assignment_marks",
        "previous_score"
    ]

    if not all(field in data for field in required_fields):
        return jsonify({"error": "Missing required fields"}), 400

    try:
        features = np.array([[
            float(data[field]) for field in required_fields
        ]])

        prediction = model.predict(features)[0]
        result = "PASS" if prediction == 1 else "FAIL"

        return jsonify({"prediction": result})

    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input values"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
