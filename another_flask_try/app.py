import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from flask import Flask, request, render_template
import  pickle
app = Flask("__name__")

q = ""

@app.route("/")
def loadpage():
    return render_template('home.html', query="")


@app.route("/", methods=["POST"])
def cancerPrediction():
    dataset_url = "https://raw.githubusercontent.com/apogiatzis/breast-cancer-azure-ml-notebook/master/breast-cancer-data.csv"
    df = pd.read_csv(dataset_url)

    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']

    filename = 'bcan_model.sav'
    model = pickle.load(open(filename, 'rb'))

    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5]]
    new_df = pd.DataFrame(data, columns = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean'])

    single = model.predict(new_df)
    probability = model.predict_proba(new_df)[:,1]

    if single == 1:
        o1 = "The patient has been diagnosed with Breast Cancer"
        o2 = "Confidence: {}".format(probability*100)
    else:
        o1 = "The patient has not been diagnosed with Breast Cancer"
        o2 = "Confidence: {}".format(probability*100)
    

    return render_template('home.html', output1=o1, output2=o2,
    query1=request.form['query1'], query2=request.form['query2'])

app.run()