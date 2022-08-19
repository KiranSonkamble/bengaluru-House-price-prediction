import pandas as pd
import pickle
from flask import Flask, render_template, request
import sklearn
import numpy as np


app =Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("LinearRegression.pkl",'rb'))

@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict(): 

    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')

    print(location,bhk,bath,sqft)
    input = pd.DataFrame( [[location,bhk,sqft,bath]],columns=['location','bhk','total_sqft','bath'])
    prediction = pipe.predict(input)[0]
    
    return f"{prediction:2,.2f} lakh"    


if __name__== "__main__":
    app.run(debug=True,port=5001)

    