import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from flask import Flask, request, jsonify
from sklearn.linear_model import LogisticRegression

# 1. Synthetic data features
feature_names = [
    'heart_rate',            # Patient's heart rate
    'blood_pressure',        # Patient's blood pressure
    'gait_speed',            # Speed of the patient's gait (movement)
    'movement_variability',  # Variation in the patient's movement patterns
    'balance_score',         # Balance assessment score
    'oxygen_level'           # Patient's blood oxygen level
]

X, y = make_classification(
    n_samples=1000,           # Creates 1000 samples
    n_features=len(feature_names),  # Number of features matches our defined list
    n_informative=4,          # 4 of these features are informative for classification
    n_redundant=0,            # No redundant features are generated
    n_classes=3,              # Two classes: e.g., 0 (Low Risk) and 1 (Medium Risk) and 2 (High Risks)
    random_state=42           # Seed for reproducibility
)

# Creates a DataFrame to organize the synthetic features with proper column names
df = pd.DataFrame(X, columns=feature_names)
df['risk'] = y

# For demo, we are only taking the first features to decide prediction 
sample_features = df.iloc[0][feature_names].to_dict()
print(sample_features)

# Splitting the data -------------------

# 80% for training and 20% for testing the model's performance.
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names],                       # The input feautures   
    df['risk'],                              # The target variable
    test_size=0.2,                           # The validation set
    random_state=42
)

# Logistic Regression model for multiclass classification
model = LogisticRegression(random_state=42, max_iter=1000, multi_class='multinomial') # 'multinomial' uses the softmax function to support > 2 classes
model.fit(X_train, y_train)


# Model Evaluation -----------------

# Predict the risk values for the test dataset
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)       # Proportion of correct predictions
precision = precision_score(y_test, y_pred, average='weighted')     # Meausures how many items that were selected were relevant 
recall = recall_score(y_test, y_pred, average='weighted')           # Measures how many relevant items were selected
f1 = f1_score(y_test, y_pred, average='weighted')                     # Balance between precision and recall 

print("Model Evaluation Metrics:")
print(f"Accuracy : {accuracy}")
print(f"Precision: {precision}")
print(f"Recall   : {recall}")
print(f"F1-score :  {f1}")

# Save the trained model
joblib.dump(model, 'fall_risk_model.pkl')
print("Model saved as fall_risk_model.pkl")



app = Flask(__name__)           # New instance of Flask class

# Home route  
@app.route('/') 
def index():
    html_content = """
    <html>
      <head>
        <title>Fall Risk Prediction API</title>
        <script>
        
          async function predictSample() {               // JavaScript function to make a GET request to /predict_sample endpoint
            try {
              const response = await fetch('/predict_sample');
              const result = await response.json();
              document.getElementById('sampleResult').innerText = 
                'Prediction: ' + result.risk_level + ' (' + result.risk_label + ')';
            } catch (error) {
              document.getElementById('sampleResult').innerText = 'Error: ' + error;
            }
          }
          
          // To display metrics 
          async function scores() {  // JavaScript function to make a GET request to /scores endpoint
            try {
                const response = await fetch('/scores');
                const result = await response.json();
                document.getElementById('scoreResult').innerHTML =
                'Accuracy: ' + result.accuracy.toFixed(4) + '<br>' +
                'Precision: ' + result.precision.toFixed(4) + '<br>' +
                'Recall: ' + result.recall.toFixed(4) + '<br>' +
                'F1 Score: ' + result.f1.toFixed(4);
            } catch (error) {
                document.getElementById('scoreResult').innerText = 'Error: ' + error;
            }
         }

        </script>
      </head>
      
        <body style="background-color: rgb(231, 255, 224);">   
            <h1 style="color: rgb(18, 43, 36);">Fall Risk Prediction using Synthetic Data</h1>  <!-- Dark green header -->
            <button onclick="predictSample()" style="background-color: rgb(14, 125, 62); color: white; padding: 10px; border: none; border-radius: 5px;">
                Predict using Sample Data
            </button>
            <div id="sampleResult" style="margin-top: 20px; font-weight: bold; color: blue;"></div>
            
            <h1 style="color: rgb(18, 43, 36);">Model Evaluation Scores</h1>
            <button onclick="scores()" style="background-color: rgb(14, 125, 62); color: white; padding: 10px; border: none; border-radius: 5px;">
                Show Scores
            </button>
            <div id="scoreResult" style="margin-top: 20px; font-weight: bold; color: blue;"></div>
       </body>

    </html>
    """
    return html_content

# Route to display evaluation metrics on a web page
@app.route('/scores')
def scores():
    return jsonify({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# Load the model for /predict endpoint
model = joblib.load('fall_risk_model.pkl')

@app.route('/predict_sample', methods=['GET'])
def predict_sample():
    try:
        
        features = [float(sample_features[feat]) for feat in feature_names]    # Converts the sample_features dictionary values to a list of floats in the correct order
        prediction = model.predict([features])                                 # Predicts risk for the sample data
        risk = int(prediction[0])
       
        # Determines the risk label 
        if risk == 0:
            risk_label = 'Low Risk'
        elif risk == 1:
            risk_label = 'Medium Risk'
        elif risk == 2:
            risk_label = 'High Risk'
        else:
            risk_label = 'Unknown Risk'
            
        return jsonify({'risk_level': risk, 'risk_label': risk_label, 'sample_data': sample_features}) # Returns prediction and sample response as JSON response 
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
