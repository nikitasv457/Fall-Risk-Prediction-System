# Fall Risk Prediction System

The Fall Risk Prediction API is a project that uses synthetic data to predict a patient's fall risk based on vital signs and movement patterns. The project leverages Python, scikit-learn, and Flask to build a multiclass classification model (predicting Low, Medium, or High Risk) and exposes REST API endpoints for making predictions and displaying evaluation metrics (Accuracy, Precision, Recall, and F1-score).

## Features

- **Synthetic Data Generation:**  
  The model generates 1,000 samples of synthetic patient data (e.g., heart rate, blood pressure, gait speed, movement variability, balance score, oxygen level) using `make_classification`.

- **Multiclass Classification:**  
  Trains a Logistic Regression model (with multinomial option) to predict three risk levels:
  - 0: Low Risk
  - 1: Medium Risk
  - 2: High Risk

- **Model Evaluation:**  
  Calculates evaluation metrics (accuracy, precision, recall, and F1-score) using a test split from the synthetic data.

- **Flask API Endpoints:**  
  Provides a simple web interface with the following endpoints:
  - **`/`**: Home page with buttons to show a sample prediction and display evaluation scores.
  - **`/predict_sample`**: Returns a risk prediction using a predefined synthetic sample.
  - **`/scores`**: Returns the evaluation metrics as a JSON object.


## Installation

### Prerequisites

- **Python 3.x**  

### Steps

1. **Clone the Repository:**

   ```
   git clone https://github.com/nikitasv457/Fall-Risk-Prediction-System.git
   ```
2.  **Install the required packages:**
    ```
    python install_libraries.py
    ```

### Usage 
1. **Run the application:**

    ```
    python app.py
    ```
#### This will start the Flask development server (by default on ``` http://127.0.0.1:5000/``` ).

2. Access the Web Interface:

- Open your web browser and navigate to:
- Home Page: ```http://127.0.0.1:5000/``` 
- This page contains buttons to trigger a sample prediction and to display model evaluation scores.
- API Endpoints:
    - GET ```/predict_sample```: Returns a risk prediction for a predefined (first row for this model) synthetic sample.
    - GET ```/scores```: Returns model evaluation metrics (accuracy, precision, recall, F1-score) as JSON.

### API Endpoints
- Home Page (```/```):
    - Provides a simple interface with two buttons:

        - Predict using Sample Data: Triggers the ```/predict_sample``` endpoint.
        - Displays Scores: Triggers the ```/scores``` endpoint.

    - GET ```/predict_sample```:
        - Returns a JSON response with:
            - risk_level: Numeric risk prediction.
            - risk_label: Descriptive risk label (Low Risk, Medium Risk, High Risk).
            - sample_data: The synthetic data sample used for the prediction.
    
    - GET ```/scores```:
            - Returns a JSON object with the model's evaluation metrics:
                - accuracy
                - Precision
                - Recall
                - F1-score 

