"""
statistical_model.py.

The script is responsible for handling all machine learning methods.
"""

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.multiclass import OneVsRestClassifier
from nltk.corpus import stopwords
from challenge.data import Data
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier


class StatisticalModel:
    """All statistical machine learning model methods."""

    def __init__(self):
        self.models = list
        self.data = Data()

    def train_model(self, df_train):
        """
        Train statistical machine learning models.

        This method, first filters the data and then initializes multiple
        machine learning methods and trains them.
        :param df_train:
        :return:
        """
        y_train = self.data.get_encoded_genres(df_train)
        x_train = Data.filter_synopsis(df_train['synopsis'])

        # create model pipelines
        model1 = Pipeline([
            ('vectorizer1',
             CountVectorizer(stop_words=stopwords.words('english'))),
            ('clf1', OneVsRestClassifier(MultinomialNB()))])

        model2 = Pipeline([
            ('vectorizer2',
             CountVectorizer(stop_words=stopwords.words('english'))),
            ('clf2', OneVsRestClassifier(LogisticRegression(solver='sag')))])

        model3 = Pipeline([('vectorizer3',
                            CountVectorizer(
                                stop_words=stopwords.words('english'))),
                           ('clf3',
                            OneVsRestClassifier(SGDClassifier(loss='log')))])

        model4 = Pipeline([('vectorizer4',
                            CountVectorizer(
                                stop_words=stopwords.words('english'))),
                           ('clf4',
                            OneVsRestClassifier(
                                RandomForestClassifier(n_estimators=50)))])

        self.models = [model1, model2, model3, model4]

        for i, model in enumerate(self.models):
            model.fit(x_train, y_train)
            print("Training Model ", i)

    def predict(self, df_test):
        """
        Get predictions from trained statistical machine learning models.

        This method, first filters the data and then gets predictions from
        trained models.
        :param df_test:
        :return:
        """
        x_test = Data.filter_synopsis(df_test['synopsis'])

        prediction_genres_encoded = []

        # get predictions from all the models in the array
        for model in self.models:
            prediction_encoded = model.predict_proba(x_test)
            prediction_genres_encoded.append(prediction_encoded)

        # take average of all the predictions
        predicted_genres_encoded = (prediction_genres_encoded[0] +
                                    prediction_genres_encoded[1] +
                                    prediction_genres_encoded[2] +
                                    prediction_genres_encoded[3]) / 4

        predicted_genres = self.get_unencoded_genres(predicted_genres_encoded)

        df_test['predicted_genres'] = predicted_genres
        df_submission = df_test[['movie_id', 'predicted_genres']]

        df_submission.to_csv("submission.csv", index=False)
        print("prediction done")
        return df_submission

    def get_unencoded_genres(self, predicted_genres):
        """
        Get unencoded genres from binarizer.

        This method sorts the predictions that are received from the model
        to get top 5 genres.
        :param predicted_genres:
        :return:
        """
        genres = np.asarray(self.data.binarizer.classes_)
        final_prediction = []
        for predicted_row in predicted_genres:
            ind = np.argpartition(predicted_row, -4)[-5:]
            ind = np.flip(ind)
            prediction_text = " ".join(genres[ind])
            final_prediction.append(prediction_text)

        return np.asarray(final_prediction)
