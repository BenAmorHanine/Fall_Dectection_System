from snn_model import create_snn
#from random_forest import create_random_forest
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import models
from keras import layers


def train_model():
    # Load preprocessed data from pickle file
    with open('preprocessed_data.pkl', 'rb') as f:
        train_data, val_data, test_data, train_labels, val_labels, test_labels = pickle.load(f)

    # Normalize the data using StandardScaler
    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    val_data = scaler.transform(val_data)
    test_data = scaler.transform(test_data)

    # Print data shapes for debugging
    print(f"Train data shape before reshaping: {train_data.shape}")
    print(f"Validation data shape before reshaping: {val_data.shape}")

    # Reshape the data for Conv1D or other deep learning models
    # Adjust the shape based on the model's expected input format
    train_data = np.expand_dims(train_data, axis=-1)  # Adding feature dimension
    val_data = np.expand_dims(val_data, axis=-1)
    test_data = np.expand_dims(test_data, axis=-1)

    print(f"Train data shape after reshaping: {train_data.shape}")
    print(f"Validation data shape after reshaping: {val_data.shape}")

    # Ensure the input shape matches the model's expected input shape
    input_shape = (train_data.shape[1], train_data.shape[2])
    print(f"Input shape for the model: {input_shape}")
    

    # Create the model using the `create_snn` function
    model = create_snn(input_shape=input_shape)
    #model = create_random_forest()

    
    # Train the model
    model.fit(
        train_data, 
        train_labels, 
        epochs=10, 
        batch_size=32, 
        validation_data=(val_data, val_labels)
    )
    """model.fit(
    train_data, 
    train_labels,
    )"""
    
    # Save the trained model
    model.save("snn_model.h5")
    print("Model saved to 'snn_model.h5'")
    

    #model.save("rf_model.h5")
    #print("Model saved to 'rf_model.h5'")#

    # Optionally save the scaler for preprocessing test data during inference
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler saved to 'scaler.pkl'")

train_model()