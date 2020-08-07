from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Model
filename = 'clf.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        creditscore = int(request.form['CreditScore'])
        geography = int(request.form['Geography'])
        gender = int(request.form['Gender'])
        age = int(request.form['Age'])
        tenure = int(request.form['Tenure'])
        balance = float(request.form['Balance'])
        numofproducts = int(request.form['NumOfProducts'])
        hascrcard = int(request.form['HasCrCard'])
        isactivemember = int(request.form['IsActiveMember'])
        estimatedsalary = float(request.form['EstimatedSalary'])

        input_data = np.array([[creditscore, geography, gender, age, tenure, balance, numofproducts, hascrcard, isactivemember, estimatedsalary]])

        my_prediction = classifier.predict(input_data)

        return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)