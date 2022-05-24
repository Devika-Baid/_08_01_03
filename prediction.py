import pickle 
model = pickle.load(open('price_prediction_model.pkl','rb')) #read mode
input_cols = [[1850,4,52,2630,20,5,2,True,True,True,True,True,True,True]]
prediction = model.predict(input_cols)
output = round(prediction[0], 2)
print(output)

input_cols = [[1550,4,52,2630,20,5,2,False,False,False,False,False,False,False]]
prediction = model.predict(input_cols)
output = round(prediction[0], 2)
print(output)