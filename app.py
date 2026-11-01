
from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
import sys

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("app.log")
    ]
)

logger = logging.getLogger(__name__)

# Load trained ML model
model = joblib.load("student_result_model.pkl")
logger.info("Student Result Prediction model loaded successfully")


# Home API
@app.route("/", methods=["GET"])
def home():
    logger.info("Home API accessed")

    return jsonify({
        "message": "Student Result Prediction API is running"
    })


# Health Check API
@app.route("/health", methods=["GET"])
def health():
    logger.info("Health check requested")

    return jsonify({
        "status": "healthy"
    })


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():
    logger.info("Prediction request received")

    data = request.get_json(silent=True) or {}

    required_fields = [
        "attendance",
        "internal_marks",
        "assignment_marks",
        "previous_score"
    ]

    if not isinstance(data, dict):
        logger.warning("Invalid request: JSON object expected")
        return jsonify({
            "error": "Invalid JSON input"
        }), 400

    if not all(field in data for field in required_fields):
        logger.warning("Prediction failed: Missing required fields")
        return jsonify({
            "error": "Missing required fields"
        }), 400

    try:
        features = np.array([[
            float(data[field]) for field in required_fields
        ]])

        prediction = model.predict(features)[0]
        result = "PASS" if prediction == 1 else "FAIL"

        logger.info("Prediction completed successfully: %s", result)

        return jsonify({
            "prediction": result
        })

    except (ValueError, TypeError):
        logger.warning("Prediction failed: Invalid input values")

        return jsonify({
            "error": "Invalid input values"
        }), 400

    except Exception:
        logger.exception("Unexpected error during prediction")

        return jsonify({
            "error": "Internal server error"
        }), 500


# Run Flask application
if __name__ == "__main__":
    logger.info("Starting Student Result Prediction API")

    app.run(host="0.0.0.0", port=5000)
