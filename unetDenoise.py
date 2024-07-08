import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
import os

# i know this uses a different framework (keras) instead of pytorch as for main.py, but i did it before i knew i could use pytorch for this project
def model(input_shape):
    """
    Creates a U-Net denoising model.

    Args:
        input_shape (tuple): The shape of the input tensor.

    Returns:
        keras.models.Model: The U-Net denoising model.
    """
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    # Bottleneck
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    # Decoder
    up5 = UpSampling2D((2, 2))(conv4)
    up5 = concatenate([conv3, up5], axis=3)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)

    up6 = UpSampling2D((2, 2))(conv5)
    up6 = concatenate([conv2, up6], axis=3)
    conv6 = Conv2D(16, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv6)

    up7 = UpSampling2D((2, 2))(conv6)
    up7 = concatenate([conv1, up7], axis=3)
    conv7 = Conv2D(8, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv7)

    outputs = Conv2D(3, (1, 1), activation='linear')(conv7)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def simpler_model(input_shape):
    """
    Creates a simpler U-Net denoising model.

    Args:
        input_shape (tuple): The shape of the input tensor.

    Returns:
        keras.models.Model: The U-Net denoising model.
    """
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    # Bottleneck
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv3)

    # Decoder
    up5 = UpSampling2D((2, 2))(conv3)
    up5 = concatenate([conv1, up5], axis=3)
    conv5 = Conv2D(8, (3, 3), activation='relu', padding='same')(up5)
    conv5 = Conv2D(8, (3, 3), activation='relu', padding='same')(conv5)

    outputs = Conv2D(3, (1, 1), activation='linear')(conv5)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def train(model, X_train, Y_train, X_val, Y_val, batch_size=32, epochs=10, save_path=None, root_dir='unet_precomputed'):
    """
    Trains a U-Net model on the given training data.

    Args:
        model (tf.keras.Model): The U-Net model to train.
        X_train (numpy.ndarray): The input training data.
        Y_train (numpy.ndarray): The target training data.
        X_val (numpy.ndarray): The input validation data.
        Y_val (numpy.ndarray): The target validation data.
        batch_size (int, optional): The batch size for training. Defaults to 32.
        epochs (int, optional): The number of epochs to train for. Defaults to 10.
        save_path (str, optional): The path to save the trained model. Defaults to None. Remember to save with .keras extension.

    Returns:
        tf.keras.Model: The trained U-Net model.
    """
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, Y_val))
    if save_path:
        path = os.path.join(root_dir, save_path)
        model.save(path)
    return model

def load_model(model_path, input_shape, optimizer='adam', loss='mse'):
    """
    Load a trained U-Net model from a file.

    Args:
        model_path (str): The path to the saved model.

    Returns:
        tf.keras.Model: The loaded U-Net model.
    """
    unet_model = simpler_model(input_shape)
    unet_model.compile(optimizer=optimizer, loss=loss)
    unet_model.load_weights(model_path)
    return unet_model

def predict(sample, model):
    return model.predict(sample)