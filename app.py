from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import pickle 

app = Flask(__name__)
model = pickle.load(open('price_prediction_lrmodel.pkl','rb')) #read mode
  
@app.route("/predict", methods=['GET','POST'])
def predict():

    if request.method == 'POST':
        #access the data from form
        print('Posted')
        Displacement=float(request.form["Displacement"])
        Cylinders=float(request.form["Cylinders"])
        Fuel_Tank_Capacity=float(request.form["Fuel_Tank_Capacity"])
        Wheelbase=float(request.form["Wheelbase"])
        Highway_Mileage=float(request.form["Highway_Mileage"])
        Seating_Capacity=float(request.form["Seating_Capacity"])
        Number_of_Airbags=float(request.form["Number_of_Airbags"])
        Hill_Assist=False#request.form["Hill_Assist"]
        Electronic_Stability_Program=True#request.form["Electronic_Stability_Program"]
        Rain_Sensing_Wipers=True#request.form["Rain_Sensing_Wipers"]
        Leather_Wrapped_Steering=True#request.form["Leather_Wrapped_Steering"]
        Automatic_Headlamps=True#request.form["Automatic_Headlamps"]
        Traction_Control=True#request.form["Traction_Control"]
        Cruise_Control=True#request.form["Cruise_Control"]
        # get prediction
        
        input_cols = [[Displacement,Cylinders,Fuel_Tank_Capacity,Wheelbase,Highway_Mileage,Seating_Capacity,Number_of_Airbags,Hill_Assist,Electronic_Stability_Program,Rain_Sensing_Wipers,Leather_Wrapped_Steering,Automatic_Headlamps,Traction_Control,Cruise_Control]]
        prediction = model.predict(input_cols)
        output = round(prediction[0], 2)
        print(output)
        return render_template("predict.html", prediction_text='Your predicted annual Healthcare Expense is $ {}'.format(output))
    return render_template('predict.html')

@app.route("/visualize")
def visualize():
    return render_template('visualization.html')

@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/features")
def features():
    df1=pd.read_csv('float_column_features.csv')
    df2=pd.read_csv('bool_column_features.csv')
    df3=pd.read_csv('object_column_features.csv')
    return render_template('features.html', f_dict=df1,b_dict=df2, o_dict=df3)

if __name__ == "__main__":
    app.run(debug=True)
