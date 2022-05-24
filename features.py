import pandas as pd
import numpy as np

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

from flask import Flask, render_template
app = Flask(__name__)


@app.route("/features")
def features():
    return render_template('features.html', f_dict=df3,b_dict=b_dict, o_dict=o_dict)


if __name__ == '__main__':
    app.run(debug=True)
# df.to_json('object.json') #new dataframe
# df.to_json('float.json') #new dataframe
# df.to_json('bool.json') #new dataframe
