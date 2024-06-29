from flask import Flask, request, render_template
import pickle
import os
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST', 'GET'])
def submit():
    if request.method == 'POST':
        # Reading the inputs given by the user
        input_feature = [int(x) for x in request.form.values()]
        input_feature = np.array(input_feature)
        print(input_feature)
        
        # Defining the column names
        names = [
            'Gender', 'Age', 'Education Level', 'Institution Type', 'IT Student',
            'Location', 'Load-shedding', 'Financial Condition', 'Internet Type', 'Network Type','Class Duration',
            'Self Lms', 'Device'
        ]
        
        # Creating a DataFrame with the input features
        data = pd.DataFrame([input_feature], columns=names)
        print(data)
        
        # Predictions using the loaded model file
        prediction = model.predict(data)
        print(prediction)
        
        # Displaying the prediction results in the UI
        if prediction == 0:
            return render_template('output.html', result="Adaptivity Level is High")
        elif prediction == 1:
            return render_template('output.html', result="Adaptivity Level is Medium")
        else:
            return render_template('output.html', result="Adaptivity Level is Low")
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

