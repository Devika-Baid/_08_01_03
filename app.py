from flask import Flask, render_template, request, url_for
import pandas as pd
import numpy as np
import pickle 
import os

app = Flask(__name__)
model = pickle.load(open('price_prediction_model.pkl','rb')) #read mode

path='no_null_df.csv'
df = pd.read_csv(path)
df = df[['Displacement','Cylinders','Fuel_Tank_Capacity','Wheelbase','Highway_Mileage'
    ,'Seating_Capacity','Number_of_Airbags','Hill_Assist','ESP_(Electronic_Stability_Program)',
    'Rain_Sensing_Wipers','Leather_Wrapped_Steering','Automatic_Headlamps','ASR_Traction_Control'
    ,'Cruise_Control']]
df=df.describe()
df=df.transpose()
df=df[['mean','std']]
df['low']=round(df['mean']-df['std'],0)
df['high']=round(df['mean']+df['std'],0)
df=df[['low','high']]
df=df.transpose()
    # print(df) 
@app.route("/predict", methods=['GET','POST'])
def predict():
    prediction_text=""
    if request.method == 'POST':
        #access the data from form
        # print('Posted')
        Displacement=float(request.form["Displacement"])
        Cylinders=float(request.form["Cylinders"])
        Fuel_Tank_Capacity=float(request.form["Fuel_Tank_Capacity"])
        Wheelbase=float(request.form["Wheelbase"])
        Highway_Mileage=float(request.form["Highway_Mileage"])
        Seating_Capacity=float(request.form["Seating_Capacity"])
        Number_of_Airbags=float(request.form["Number_of_Airbags"])
        Hill_Assist=bool(request.form.getlist("Hill_Assist"))
        Electronic_Stability_Program=bool(request.form.getlist("Electronic_Stability_Program"))
        Rain_Sensing_Wipers=bool(request.form.getlist("Rain_Sensing_Wipers"))
        Leather_Wrapped_Steering=bool(request.form.getlist("Leather_Wrapped_Steering"))
        Automatic_Headlamps=bool(request.form.getlist("Automatic_Headlamps"))
        Traction_Control=bool(request.form.getlist("Traction_Control"))
        Cruise_Control=bool(request.form.getlist("Cruise_Control"))
        # get prediction
        input_cols = [[Displacement,Cylinders,Fuel_Tank_Capacity,Wheelbase,Highway_Mileage,Seating_Capacity,Number_of_Airbags,Hill_Assist,Electronic_Stability_Program,Rain_Sensing_Wipers,Leather_Wrapped_Steering,Automatic_Headlamps,Traction_Control,Cruise_Control]]
        print(input_cols)
        prediction = model.predict(input_cols)
        output = round(prediction[0], 2)
        print(output)
        warning=""
        if(output<0):
            warning="The prediction is negative, the input values are beyond the scope of prediction for this model."
        return render_template("predict.html", prediction_text='Your predicted annual Healthcare Expense is Rs {}'.format(output),ranges=df,warning=warning)
    return render_template('predict.html',prediction_text="",ranges=df)

@app.route("/visualize")
def visualize():
    cwd=os.getcwd()
    types=['bar','pie','box','pivot','waffle','regplot']
    dict={}
    for t in types:
        l=os.listdir(cwd+"\\static\\img\\"+t)
        dict[t]=["/img/"+t+"/"+i for i in l]
    # print(dict)
    return render_template('visualization.html',dict=dict)

@app.route("/index")
def index():
    return render_template('index.html')

@app.route("/features")
def features():
    #reading the dataset
    path='no_null_df.csv'
    df = pd.read_csv(path)

    object_columns=df.select_dtypes(include=['object']).columns
    float_columns=df.select_dtypes(include=['float64']).columns
    bool_columns=df.select_dtypes(include=['bool']).columns

    b_dict=list(bool_columns)
    o_dict={}
    for c in object_columns:
        l=list(df[c].unique())
        l.sort()
        o_dict[c]=l

    df1=df[float_columns].describe()
    df2=df1.transpose()
    df3=df2[['min','max','mean']]
    df3=df3.round(0)
    df3=df3.transpose()
    # print(df3.to_dict())
    return render_template('features.html', f_dict=df3,b_dict=b_dict, o_dict=o_dict)

if __name__ == "__main__":
    app.run(debug=True)
