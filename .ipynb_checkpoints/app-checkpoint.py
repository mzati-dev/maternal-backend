from flask import Flask, request, jsonify
from datetime import datetime
import time
from flask_cors import CORS
import logging
import pandas as pd
from joblib import load
import sklearn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from typing import Dict, Any, Union

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info(f"Using scikit-learn version: {sklearn.__version__}")

# Constants
MALAWI_DISTRICTS = [
    'Balaka', 'Blantyre', 'Chikwawa', 'Chiradzulu', 'Chitipa',
    'Dedza', 'Dowa', 'Karonga', 'Kasungu', 'Likoma',
    'Lilongwe', 'Machinga', 'Mangochi', 'Mchinji', 'Mulanje',
    'Mwanza', 'Mzimba', 'Neno', 'Nkhata Bay', 'Nkhotakota',
    'Nsanje', 'Ntcheu', 'Ntchisi', 'Phalombe', 'Rumphi',
    'Salima', 'Thyolo', 'Zomba'
]

RECOMMENDATIONS = {
    'High': [
        "Immediate consultation with obstetric specialist required",
        "Increased frequency of antenatal visits",
        "Continuous fetal monitoring recommended",
        "Consider hospitalization for close observation",
        "Strict blood pressure monitoring",
        "Bed rest may be advised",
        "Emergency contact numbers provided"
    ],
    'Low': [
        "Continue with regular antenatal check-ups",
        "Maintain balanced diet with adequate protein and iron",
        "Moderate exercise recommended",
        "Monitor blood pressure weekly",
        "Attend all prenatal education classes",
        "Maintain hydration and proper rest",
        "Report any unusual symptoms immediately"
    ]
}

# Model and metadata initialization
model = None
encoder = None
# Define default feature names and column types for the model
FEATURE_NAMES = [
    'Age', 'Location', 'ChronicalCondition', 'PreviousPregnancyComplication',
    'GestationAge', 'Gravidity', 'Parity', 'AntenatalVisit', 'Systolic',
    'Dystolic', 'PulseRate', 'SpecificComplication', 'DeliveryMode',
    'StaffConductedDelivery'
]
CATEGORICAL_COLS = [
    'Location', 'ChronicalCondition', 'PreviousPregnancyComplication',
    'SpecificComplication', 'DeliveryMode', 'StaffConductedDelivery'
]
NUMERICAL_COLS = [
    'Age', 'GestationAge', 'Gravidity', 'Parity', 'AntenatalVisit',
    'Systolic', 'Dystolic', 'PulseRate'
]


FEATURE_MAPPING = {
    'age': 'Age',
    'location': 'Location',
    'chronicCondition': 'ChronicalCondition',
    'previousPregnancyComplication': 'PreviousPregnancyComplication',
    'gestationAge': 'GestationAge',
    'gravidity': 'Gravidity',
    'parity': 'Parity',
    'antenatalVisit': 'AntenatalVisit',
    'systolic': 'Systolic',
    'diastolic': 'Dystolic', # Corrected from Dystolic to Diastolic for API input mapping
    'pulseRate': 'PulseRate',
    'specificComplication': 'SpecificComplication',
    'deliveryMode': 'DeliveryMode',
    'staffConductedDelivery': 'StaffConductedDelivery'
}


def load_model_with_fallback(filepath):
    """Attempt to load model with various fallback strategies"""
    try:
        loaded_data = load(filepath)

        if isinstance(loaded_data, dict):
            model_obj = loaded_data.get('model')
            encoder_obj = loaded_data.get('encoder')
            feature_names = loaded_data.get('feature_names', FEATURE_NAMES)
            categorical_cols = loaded_data.get('categorical_cols', CATEGORICAL_COLS)
            numerical_cols = loaded_data.get('numerical_cols', NUMERICAL_COLS)
            return model_obj, encoder_obj, feature_names, categorical_cols, numerical_cols
        elif hasattr(loaded_data, 'predict'):
            logger.warning("Loaded file contains only model object, using default feature names")
            return loaded_data, None, FEATURE_NAMES, CATEGORICAL_COLS, NUMERICAL_COLS
        else:
            raise ValueError("Unknown model file format")
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        # We don't re-raise here immediately to allow the app to start,
        # but the `model` variable will remain None, triggering fallback behavior.
        return None, None, FEATURE_NAMES, CATEGORICAL_COLS, NUMERICAL_COLS

# Load the model globally when the app starts
try:
    model, encoder, FEATURE_NAMES, CATEGORICAL_COLS, NUMERICAL_COLS = load_model_with_fallback(
        'model/maternal_risk_predictor(1).pkl'
    )
    if model:
        logger.info("Model and metadata loaded successfully.")
        # Perform a quick test prediction
        try:
            # Create proper test data with correct values, aligning with FEATURE_NAMES
            test_data_dict = {
                'Age': 25,
                'Location': 'Urban',
                'ChronicalCondition': 'No',
                'PreviousPregnancyComplication': 'No',
                'GestationAge': 38,
                'Gravidity': 2,
                'Parity': 1,
                'AntenatalVisit': 4,
                'Systolic': 120,
                'Dystolic': 80, # This maps to 'diastolic' in API input but 'Dystolic' in model features
                'PulseRate': 70,
                'SpecificComplication': 'No',
                'DeliveryMode': 'Spontaneous Vertex Delivery',
                'StaffConductedDelivery': 'Skilled'
            }
            test_df = pd.DataFrame([test_data_dict])

            if encoder:
                # Ensure the test_df has the categorical columns for transformation
                encoded_categorical = encoder.transform(test_df[CATEGORICAL_COLS])
                encoded_df = pd.DataFrame(encoded_categorical,
                                          columns=encoder.get_feature_names_out(CATEGORICAL_COLS))
                numerical_df = test_df[NUMERICAL_COLS]
                test_df_processed = pd.concat([numerical_df, encoded_df], axis=1)
            else:
                test_df_processed = test_df[NUMERICAL_COLS] # If no encoder, only numerical is possible without errors

            
            # This handles cases where the model expects more features than what's available
            # due to dynamic one-hot encoding or specific training setups.
            # Create a full DataFrame with all expected features, initialized to 0
            full_test_df = pd.DataFrame(0, index=[0], columns=FEATURE_NAMES if not encoder else test_df_processed.columns)
            # Update with actual values from processed test_df
            for col in test_df_processed.columns:
                if col in full_test_df.columns:
                    full_test_df[col] = test_df_processed[col]

            # Reorder columns to match the model's expected feature order
            if hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
                final_test_df = full_test_df[model.feature_names_in_]
            else:
                final_test_df = full_test_df # Fallback if model doesn't expose feature_names_in_

            prediction = model.predict(final_test_df)
            logger.info(f"Model test successful. Prediction: {prediction}")
        except Exception as e:
            logger.error(f"Model test failed during prediction: {str(e)}", exc_info=True)
            model = None # Set model to None to ensure fallback logic is used
    else:
        logger.warning("Model was not loaded. Falling back to rule-based risk assessment.")
except Exception as e:
    logger.error(f"Error during model initialization or initial test: {str(e)}", exc_info=True)
    model = None # Ensure model is None if there's any loading error

def validate_input(data: Dict[str, Any]) -> Union[None, Dict[str, str]]:
    """Validate input data against model requirements"""
    # Use FEATURE_MAPPING values as expected fields in the model context
    required_fields = set(FEATURE_MAPPING.keys())
    missing_fields = required_fields - set(data.keys())
    if missing_fields:
        return {'error': f'Missing required fields: {", ".join(missing_fields)}'}

    type_errors = []
    # These are the keys from the incoming API request
    numeric_fields = ['age', 'gestationAge', 'gravidity', 'parity', 'antenatalVisit',
                      'systolic', 'diastolic', 'pulseRate']

    for field in numeric_fields:
        try:
            # Ensure values are convertible to float
            float(data[field])
        except (ValueError, TypeError):
            type_errors.append(f"{field} must be a number")

    # Range validations
    if 'age' in data and (float(data['age']) < 10 or float(data['age']) > 60):
        type_errors.append("Age must be between 10 and 60 years")
    if 'gestationAge' in data and (float(data['gestationAge']) < 0 or float(data['gestationAge']) > 45):
        type_errors.append("Gestation age must be between 0 and 45 weeks")
    if 'systolic' in data and (float(data['systolic']) < 50 or float(data['systolic']) > 300):
        type_errors.append("Systolic BP must be between 50 and 300 mmHg")
    if 'diastolic' in data and (float(data['diastolic']) < 30 or float(data['diastolic']) > 200):
        type_errors.append("Diastolic BP must be between 30 and 200 mmHg")
    if 'pulseRate' in data and (float(data['pulseRate']) < 30 or float(data['pulseRate']) > 220):
        type_errors.append("Pulse rate must be between 30 and 220 bpm")
    if data.get('location') not in MALAWI_DISTRICTS:
        type_errors.append("Invalid district selected")

    if type_errors:
        return {'error': " | ".join(type_errors)}

    return None

def prepare_input_data_for_model(api_data: Dict[str, Any]) -> pd.DataFrame:
    """Convert API input to model-ready DataFrame, applying one-hot encoding."""
    # Map API input keys to model's expected feature names
    mapped_data = {
        FEATURE_MAPPING[k]: v
        for k, v in api_data.items()
        if k in FEATURE_MAPPING
    }

    # Create DataFrame from the mapped single record
    input_df = pd.DataFrame([mapped_data])

    if encoder:
        # Separate numerical and categorical columns
        numerical_df = input_df[NUMERICAL_COLS]
        categorical_df = input_df[CATEGORICAL_COLS]

        # One-hot encode categorical features
        # Handle potential KeyError if a category not seen during training appears
        try:
            encoded_categorical = encoder.transform(categorical_df)
            encoded_df = pd.DataFrame(encoded_categorical.toarray(), # .toarray() for sparse matrices
                                      columns=encoder.get_feature_names_out(CATEGORICAL_COLS),
                                      index=input_df.index) # Maintain index for concat
        except ValueError as e:
            logger.error(f"Error during encoding: {e}. This might be due to unseen categories. "
                         f"Input categorical data: {categorical_df.to_dict('records')}")
            # If an unseen category causes an error, it's safer to return an empty DataFrame
            # or handle more robustly based on application needs.
            # For this context, we'll re-raise as it's a critical preprocessing step.
            raise ValueError(f"Failed to encode categorical features: {e}. "
                             "Ensure all categories in request data were present during model training.")


        # Concatenate numerical and encoded categorical features
        processed_df = pd.concat([numerical_df, encoded_df], axis=1)
    else:
        # If no encoder, assume model only uses numerical features or handles categorical internally
        # This branch might be less robust if the model actually needs encoding but encoder wasn't loaded
        processed_df = input_df[NUMERICAL_COLS]
        logger.warning("No encoder available. Only numerical columns will be used for prediction.")


    # Ensure all expected features (from model's training) are present and in correct order
    # This is crucial if the model expects a fixed number of features in a specific order
    if model and hasattr(model, 'feature_names_in_') and model.feature_names_in_ is not None:
        # Create a DataFrame with all expected features, initialized to 0 or NaN
        final_input_df = pd.DataFrame(0, index=processed_df.index, columns=model.feature_names_in_)
        # Fill with actual processed data
        for col in processed_df.columns:
            if col in final_input_df.columns:
                final_input_df[col] = processed_df[col]
        # Return the DataFrame with columns in the exact order the model expects
        return final_input_df[model.feature_names_in_]
    else:
        # If model doesn't expose feature_names_in_ or model not loaded,
        # assume processed_df is sufficient. This might be risky.
        logger.warning("Model's feature_names_in_ not found or model not loaded. "
                       "Proceeding with available processed features. This might lead to prediction errors.")
        return processed_df

def calculate_risk(patient_data):
    """Calculate risk based on patient data using logical rules"""
    # Use get() with default values to prevent KeyError if a field is missing (though validation should catch this)
    age = patient_data.get('age', 0)
    gravidity = patient_data.get('gravidity', 0)
    systolic = patient_data.get('systolic', 0)
    diastolic = patient_data.get('diastolic', 0)
    antenatal_visit = patient_data.get('antenatalVisit', 0)
    delivery_mode = patient_data.get('deliveryMode', '')
    specific_complication = patient_data.get('specificComplication', '')
    pulse_rate = patient_data.get('pulseRate', 0)
    chronic_condition = patient_data.get('chronicCondition', '')
    previous_complication = patient_data.get('previousPregnancyComplication', '')
    staff_conducted_delivery = patient_data.get('staffConductedDelivery', '')

    # High-Risk Factors (will trigger High Risk if true)
    age_risk = age <= 16 or age >= 35
    gravidity_risk = gravidity >= 4
    bp_risk = systolic >= 140 or diastolic >= 90
    few_antenatal_visits = antenatal_visit < 4
    cesarean_risk = delivery_mode == "Caesarean Section"
    complication_risk = specific_complication == "Yes"
    pulseRateRisk = pulse_rate <= 60 or pulse_rate >= 100

    # Combine high-risk factors for initial risk level
    is_high_risk = (age_risk or gravidity_risk or bp_risk or
                    few_antenatal_visits or cesarean_risk or complication_risk or pulseRateRisk)

    risk_level = 'High' if is_high_risk else 'Low'

    # Probability calculation (simplified for logical rules)
    probability = 0.0 # Default low probability

    if cesarean_risk:
        probability = 0.95
    elif complication_risk:
        probability = 0.9
    elif bp_risk:
        probability = 0.85
    elif age_risk:
        probability = 0.8
    elif gravidity_risk:
        probability = 0.75
    elif few_antenatal_visits:
        probability = 0.7
    elif pulseRateRisk:
        probability = 0.8

    # Low-risk factors slightly increase probability for 'Low' risk
    elif chronic_condition == "Yes":
        probability = max(probability, 0.4) # Ensure it doesn't decrease a high prob
    elif previous_complication == "Yes":
        probability = max(probability, 0.35)
    elif staff_conducted_delivery == "Unskilled":
        probability = max(probability, 0.3)
    else:
        # If no specific rules triggered, and not high risk, assign a low probability
        if not is_high_risk:
            probability = 0.2

    # If is_high_risk is true, ensure probability is at least 0.5 (or higher based on specific rule)
    if is_high_risk and probability < 0.5:
        probability = 0.5 # A baseline for high risk

    return risk_level, probability

def override_risk_based_on_rules(input_data: Dict[str, Any], current_risk: str, current_prob: float) -> tuple:
    """
    Override the model's risk assessment based on specific high-risk rules.
    Returns tuple of (updated_risk, updated_probability)
    """
    try:
        # Convert all inputs to appropriate types
        # Using .get() for safety against missing keys, though validation should catch most
        age = float(input_data.get('age', 0))
        gravidity = int(float(input_data.get('gravidity', 0)))
        parity = int(float(input_data.get('parity', 0)))
        systolic = float(input_data.get('systolic', 0))
        diastolic = float(input_data.get('diastolic', 0))
        specific_complication = input_data.get('specificComplication', 'No') # Assume 'No' if not present

        logger.info(f"Checking override rules with - Age: {age}, Gravidity: {gravidity}, Parity: {parity}, BP: {systolic}/{diastolic}, Complication: {specific_complication}")

        # Rule 1: Very young mother with existing children
        if age <= 16 and (parity >= 1 or gravidity > 1):
            logger.info("RULE TRIGGERED: Young mother with existing children -> High Risk (0.95)")
            return ("High", 0.95)

        # Rule 2: Abnormal blood pressure (High or Low)
        if systolic >= 140 or diastolic >= 90:
            logger.info(f"RULE TRIGGERED: High BP ({systolic}/{diastolic}) -> High Risk (0.9)")
            return ("High", 0.9)

        if systolic < 90 or diastolic < 60:
            logger.info(f"RULE TRIGGERED: Low BP ({systolic}/{diastolic}) -> High Risk (0.85)")
            return ("High", 0.85)

        # Rule 3: Specific Complication present
        if specific_complication == "Yes":
            logger.info("RULE TRIGGERED: Specific Complication present -> High Risk (0.98)")
            return ("High", 0.98) # Very high probability if explicit complication

        logger.info("No override rules triggered, using model prediction (or initial logic prediction).")
        return (current_risk, current_prob)

    except Exception as e:
        logger.error(f"Error in risk override: {str(e)}", exc_info=True)
        # If an error occurs in the override logic, fall back to the original assessment
        return (current_risk, current_prob)


@app.route('/')
def home():
    return jsonify({
        "status": "active",
        "service": "Maternal Health Risk Prediction API",
        "model_loaded": model is not None,
        "expected_features": FEATURE_NAMES,
        "categorical_columns": CATEGORICAL_COLS,
        "numerical_columns": NUMERICAL_COLS,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def assess_risk():
    """Endpoint for assessing maternal risk, using model with rule-based override or fallback logic."""
    start_time = time.time()

    try:
        data = request.get_json()
        if not data:
            logger.error("No JSON data received.")
            return jsonify({'error': 'No JSON data received'}), 400

        logger.info(f"Received request data: {data}")

        # Validate input data
        validation_error = validate_input(data)
        if validation_error:
            logger.error(f"Validation error: {validation_error['error']}")
            return jsonify(validation_error), 400

        original_risk = None
        original_probability = None
        risk_override_applied = False

        # Attempt model prediction first if model is loaded
        if model:
            try:
                processed_input_df = prepare_input_data_for_model(data)
                logger.info(f"Prepared input for model. Columns: {processed_input_df.columns.tolist()}")

                predicted_class = model.predict(processed_input_df)[0]
                prediction_proba = model.predict_proba(processed_input_df)[0][1] # Probability of 'High' risk

                original_risk = "High" if predicted_class == 1 else "Low"
                original_probability = float(prediction_proba)

                logger.info(f"Model predicted: Risk Level = {original_risk}, Probability = {original_probability}")

                # Apply rule-based override on model's prediction
                current_risk, current_probability = override_risk_based_on_rules(data, original_risk, original_probability)
                if current_risk != original_risk or current_probability != original_probability:
                    risk_override_applied = True
                    logger.info("Model prediction overridden by rules.")

            except Exception as e:
                logger.error(f"Error during model prediction or preprocessing: {str(e)}. Falling back to logic-based assessment.", exc_info=True)
                # Fallback to logic-based assessment if model prediction fails
                current_risk, current_probability = calculate_risk(data)
                risk_override_applied = False # No override if model failed
                original_risk = None # Clear original model predictions
                original_probability = None
        else:
            logger.warning("Model not loaded or failed to load. Using logic-based risk assessment.")
            # Use logic-based approach if model is not available
            current_risk, current_probability = calculate_risk(data)
            risk_override_applied = False # No override as no model prediction to override

        response = {
            "patientId": data.get('patientId', f"patient-{int(time.time() * 1000)}"),
            "patientName": data.get('name', 'N/A'), # Assuming 'name' is used for patientName
            "riskLevel": current_risk,
            "probability": round(current_probability, 4),
            "recommendations": RECOMMENDATIONS[current_risk],
            "timestamp": datetime.utcnow().isoformat() + 'Z',
            "inputFeatures": data,
            "riskOverrideApplied": risk_override_applied,
            "originalRisk": original_risk if risk_override_applied else None,
            "originalProbability": round(original_probability, 4) if risk_override_applied else None,
            "processingTimeMs": int((time.time() - start_time) * 1000)
        }

        logger.info(f"Final assessment response: {response}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"An unexpected error occurred during risk assessment: {str(e)}", exc_info=True)
        return jsonify({
            'error': True,
            'message': 'An internal server error occurred',
            'details': str(e),
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'processingTimeMs': int((time.time() - start_time) * 1000)
        }), 500

@app.route('/api/districts', methods=['GET'])
def get_districts():
    """Endpoint to get list of Malawi districts"""
    return jsonify({
        'districts': MALAWI_DISTRICTS
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)