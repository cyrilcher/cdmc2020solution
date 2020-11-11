import pandas as pd
import base64
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from scipy.sparse import hstack


def prepare_bits_vocab(X_train_bytes):
    bits_vocab = []
    for i in tqdm([1, 2, 4, 8, 16]):
        vectorizer = CountVectorizer(ngram_range=(i, i), max_df=0.9, min_df=0.1)
        vectorizer.fit(X_train_bytes)
        bits_vocab += list(vectorizer.vocabulary_)
    return bits_vocab


def tfidf_transform(X_train_bytes, X_test_bytes, bits_vocab):
    vocab = {w: j for j, w in enumerate(bits_vocab)}
    tfidf = TfidfVectorizer(ngram_range=(1, 16), max_df=0.8, vocabulary=vocab, min_df=0.2)
    X_train_transformed = tfidf.fit_transform(X_train_bytes)
    X_test_transformed = tfidf.transform(X_test_bytes)
    return X_train_transformed, X_test_transformed


def ohe_transform(train_df, test_df):
    ohe = OneHotEncoder()
    X_train_ohe = ohe.fit_transform(train_df[['CPU']])
    X_test_ohe = ohe.transform(test_df[['CPU']])
    return X_train_ohe, X_test_ohe


def prepare_data():
    train_df = pd.read_csv('data/CDMC2020IoTMalware.train.csv')
    test_df = pd.read_csv('data/CDMC2020IoTMalware.test.csv')
    X_train_bytes = train_df['ByteSequence'].apply(lambda x: ' '.join([str(i) for i in base64.b64decode(x)]))
    bits_vocab = prepare_bits_vocab(X_train_bytes)
    X_test_bytes = test_df['ByteSequence'].apply(lambda x: ' '.join([str(i) for i in base64.b64decode(x)]))
    X_train_transformed, X_test_transformed = tfidf_transform(X_train_bytes, X_test_bytes, bits_vocab)
    X_train_ohe, X_test_ohe = ohe_transform(train_df, test_df)
    X_train_full = hstack([X_train_transformed, X_train_ohe])
    X_test_full = hstack([X_test_transformed, X_test_ohe])
    return X_train_full, X_test_full, train_df, test_df


def fit_predict_knn(X_train_full, X_test_full, train_df):
    le = LabelBinarizer()
    le.fit(train_df[['Family']])
    y_train = le.transform(train_df[['Family']])
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=-1, p=1)
    knn.fit(X_train_full, y_train)
    knn_preds = knn.predict(X_test_full)
    return le.inverse_transform(knn_preds)


if __name__ == '__main__':
    print('Preprocessing takes about 10 minutes')
    X_train_full, X_test_full, train_df, test_df  = prepare_data()
    print('Training and predicting')
    preds = fit_predict_knn(X_train_full, X_test_full, train_df)
    submission_df = pd.DataFrame(index=test_df.index+1, data=preds)
    submission_df.to_csv('task2_submission.csv', header=False)