from tensorflow import keras
from keras import models
from sklearn.metrics import classification_report
import pickle


def evaluate_model(X_test, y_test):
    model = models.load_model("snn_model.h5")
    predictions = (model.predict(X_test) > 0.4).astype("int32")
    print("REPORT:",classification_report(y_test, predictions))

with open('preprocessed_data.pkl', 'rb') as f:
    train_data, val_data, test_data, train_labels, val_labels, test_labels = pickle.load(f)

# Assign X_test and y_test
X_test = test_data
y_test = test_labels
evaluate_model(X_test, y_test)