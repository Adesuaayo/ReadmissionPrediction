from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('decision_tree_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define descriptive messages for prediction output
PREDICTION_MESSAGES = {
    0: "Prediction: The patient is unlikely to be readmitted. This suggests that based on the provided data, there is a low probability of readmission.",
    1: "Prediction: The patient is likely to be readmitted. Based on the data, there's a higher chance the patient may need to return for further care."
}

# Label mappings for categorical fields
label_mappings = {
    'race': {'AfricanAmerican': 0, 'Asian': 1, 'Caucasian': 2, 'Hispanic': 3, 'Other': 4},
    'gender': {'Female': 0, 'Male': 1},
    'age': {'[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9},
    'admission_type': {'Elective': 0, 'Emergency': 1, 'Newborn': 2, 'Other': 3},
    'admission_source': {'Emergency Room': 0, 'Other': 1, 'Referral': 2, 'Transfer': 3, 'nan': 4},
    'diag_1': {'Circulatory': 0, 'Diabetes': 1, 'Digestive': 2, 'Genitourinary': 3, 'Infectious Diseases': 4, 'Injury': 5, 'Musculoskeletal': 6, 'Neoplasms': 7, 'Other': 8, 'Respiratory': 9},
    'diag_2': {'Circulatory': 0, 'Diabetes': 1, 'Digestive': 2, 'Genitourinary': 3, 'Infectious Diseases': 4, 'Injury': 5, 'Musculoskeletal': 6, 'Neoplasms': 7, 'Other': 8, 'Respiratory': 9},
    'diag_3': {'Circulatory': 0, 'Diabetes': 1, 'Digestive': 2, 'Genitourinary': 3, 'Infectious Diseases': 4, 'Injury': 5, 'Musculoskeletal': 6, 'Neoplasms': 7, 'Other': 8, 'Respiratory': 9},
    'metformin': {'Down': 0, 'No': 1, 'Steady': 2, 'Up': 3},
    'glipizide': {'Down': 0, 'No': 1, 'Steady': 2, 'Up': 3},
    'glyburide': {'Down': 0, 'No': 1, 'Steady': 2, 'Up': 3},
    'insulin': {'Down': 0, 'No': 1, 'Steady': 2, 'Up': 3}
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Process form data with label mappings for categorical fields
    features = []
    for field, value in request.form.items():
        if field in label_mappings:
            mapped_value = label_mappings[field].get(value, None)
            if mapped_value is None:
                return render_template('index.html', prediction_text="Invalid input for field: " + field)
            features.append(mapped_value)
        else:
            features.append(float(value))

    input_data = np.array(features).reshape(1, -1)

    # Make the prediction
    prediction = model.predict(input_data)[0]
    prediction_text = PREDICTION_MESSAGES[prediction]
    
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
