from flask import Flask, render_template, request, url_for
import pickle 
app = Flask(__name__)
model = pickle.load(open('price_prediction_lrmodel.pkl','rb')) #read mode
@app.route("/predict")
def home():
    return render_template('predict.html')

  
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
        # return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)
