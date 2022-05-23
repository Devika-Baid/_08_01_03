import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle
lm=LinearRegression()

#reading the dataset
path='no_null_df.csv'
df = pd.read_csv(path)


#independent and dependent columns
x = df[['Displacement','Cylinders','Fuel_Tank_Capacity','Wheelbase','Highway_Mileage'
    ,'Seating_Capacity','Number_of_Airbags','Hill_Assist','ESP_(Electronic_Stability_Program)',
    'Rain_Sensing_Wipers','Leather_Wrapped_Steering','Automatic_Headlamps','ASR_/_Traction_Control'
    ,'Cruise_Control']]
y = df['Ex-Showroom_Price']

#split in train and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

#model training
lm.fit(x_train, y_train)

#model testing
predictions = lm.predict(x_test)
print(lm.score(x_test,y_test))

#save the model
file = open("price_prediction_model.pkl", 'wb')
pickle.dump(lm, file)