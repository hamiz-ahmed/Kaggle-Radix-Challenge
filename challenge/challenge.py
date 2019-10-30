"""
Challenge.py.

The script to start flask web service.
"""

from flask import Flask, request, Response
import pandas as pd
from pandas.io.json import json_normalize
from challenge.statistical_model import StatisticalModel

app = Flask(__name__)


model_stat = StatisticalModel()


@app.route('/genres/train', methods=['POST'])
def train():
    """
    Entry point for training flask.

    This method is the entry point for flask service genres/train
    :return:
    """
    print("/genres/train")

    # decode the request
    data = request.data.decode("utf-8")

    # write data from req in local csv file
    file_train_data = "train_local.csv"
    f = open(file_train_data, "a")
    f.write(data)
    f.close()

    # read the training csv as dataframe
    train_df = pd.read_csv(file_train_data)

    model_stat.train_model(train_df)

    # return the response that model is trained successfully
    return Response(
        status=200,
        headers={
            "message": "The model has been successfully trained"
        })


@app.route('/genres/predict', methods=['POST'])
def predict():
    """
    Entry point for getting predictions.

    This method is the entry point for flask service genres/predict
    :return:
    """
    print('/genres/predict')

    file_test_data = "test_local.csv"

    if request.is_json:
        json = request.json
        data = json_normalize(json)

    else:
        data = request.data.decode("utf-8")

        f = open(file_test_data, "a")
        f.write(data)
        f.close()

        data = pd.read_csv(file_test_data)
        data = data.dropna(axis=0)

    df_submission = model_stat.predict(data)
    df_submission.to_csv("submission.csv", index=False)

    print(df_submission)

    return Response(
        df_submission.to_csv(index=False),
        status=200,
        headers={
            "description": "The top 5 predicted movie genres",
            "Content-Type": "text/csv",

        })


if __name__ == '__main__':
    app.run(debug=True)
