"""
data.py.

The script containing class Data.py.
"""


from sklearn.preprocessing import MultiLabelBinarizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download('stopwords')
nltk.download('wordnet')


class Data:
    """Main class for data preprocessing."""

    def __init__(self):
        self.binarizer = MultiLabelBinarizer()

    @classmethod
    def filter_synopsis(self, synopsis_series):
        """
        Filter the synopsis of training and testing df.

        This function removes stop and stem words from the synopsis
        :param synopsis_series:
        :return:
        """
        # covert to lower case
        synopsis_series = synopsis_series.str.lower()
        stop = stopwords.words('english')
        # stemmer = SnowballStemmer("english")
        wordnet_lemmatizer = WordNetLemmatizer()
        # remove stop words
        series = synopsis_series.apply(
            lambda x: ' '.join(
                [word for word in x.split() if word not in stop]))

        # remove stem words
        # series = series.apply(
        #     lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

        # lematize stem words
        series = series.apply(
            lambda x: ' '.join([wordnet_lemmatizer.lemmatize(word, pos="v")
                                for word in x.split()]))
        return series

    def get_encoded_genres(self, df):
        """
        Encode the genres from df.

        This method initializes binarizer to transform genres to one hot
        encoding.
        :param df:
        :return:
        """
        # covert
        df['genres'] = df['genres'].str.split()

        one_hot_genres = self.binarizer.fit_transform(df['genres'].tolist())
        return one_hot_genres
