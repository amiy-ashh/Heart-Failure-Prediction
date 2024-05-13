from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the trained model from the pickle file
with open(r"D:\Heart Failure Prediction\Pickle File\Random Forest Classifier.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from the form
    age = float(request.form['age'])
    anaemia = int(request.form['anaemia'])
    creatinine_phosphokinase = float(request.form['creatinine_phosphokinase'])
    diabetes = int(request.form['diabetes'])
    ejection_fraction = float(request.form['ejection_fraction'])
    high_blood_pressure = int(request.form['high_blood_pressure'])
    platelets = float(request.form['platelets'])
    serum_creatinine = float(request.form['serum_creatinine'])
    serum_sodium = float(request.form['serum_sodium'])
    sex = int(request.form['sex'])
    smoking = int(request.form['smoking'])
    time = float(request.form['time'])
    
    # Make prediction using the loaded model
    prediction = model.predict([[age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                                 high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex,
                                 smoking, time]])
    
    # Predicted class (0: no heart failure, 1: heart failure)
    predicted_class = "Dead" if prediction[0] == 1 else "Alive"
    
    
    return render_template('index.html', predicted_class=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
