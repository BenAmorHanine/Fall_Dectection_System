import tensorflow as tf
from tensorflow import keras

from keras import layers
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
#from keras import load_model

def create_snn(input_shape):
    model = keras.Sequential([
        layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'),
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Adjust output layer based on your task
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


"""
"""

"""model = keras.Sequential([
        layers.Dense(32, activation='relu', input_dim=input_dim),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Sigmoid for binary classification
    ])
    if you use this one fl build call fl training baddel :    
        input_dim = X_train.shape[1]
        model = create_snn(input_dim)
    
    """