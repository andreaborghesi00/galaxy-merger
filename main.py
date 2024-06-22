# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# AUTHOR: MATTEO FOSSATI
#
# NOTEBOOK FOR CLASSIFYING galaxy mergers with galaxy images in various bands and using CNNs
#
# You will see an exercise of building, compiling, and training a CNN on syntetic astronomical imaging data.
#
# Load the data and visualize a sample of the data.
#
# 1) Divide the data into training, validation, and testing sets.
#
# 2) Build a CNN in Keras.
#
# 3) Compile the CNN.
#
# 4) Train the CNN to perform a classification task.
#
# 5) Evaluate the results.
#
# CNNs can be applied to a wide range of image recognition tasks, including classification and regression. In this notebook, we will build, compile, and train CNN to classify whether a galaxy has undergone a merger, using simulated Hubble and James Webb Space Telescopes images of galaxies.
#
#

# %%
# arrays and math
import numpy as np
from scipy import fftpack

# data handling
import pickle

# fits
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.visualization import simple_norm


# plotting
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


# keras
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input, Conv2DTranspose, Dropout
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback, ModelCheckpoint


# sklearn (for machine learning)
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf

# denoise
import gmmDenoise
import morphologicalDenoise
import fourierDenoise
import unetDenoise

# load data
y = np.load('datasets/labels.npy')

# %%
X = np.load('datasets/dataset.npy')

# %%
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

# %% [markdown]
# # Data retrieval

# %%
# %%time
version = 'noisy' # pristine or noisy. Pristine has infinite S/N, noisy has realistic S/N
file_url = 'https://archive.stsci.edu/hlsps/deepmerge/hlsp_deepmerge_hst-jwst_acs-wfc3-nircam_illustris-z2_f814w-f160w-f356w_v1_sim-'+version+'.fits'


# %%
hdu = fits.open(download_file(file_url, cache=True, show_progress=True))

# %% [markdown]
# # Sampling and data visualization

# %% [markdown]
# Filters are F814W (814 nm, Optical Wavelenghts) from the Advanced Camera for Surveys, F160W (1600 nm, near-infrared) from the Wide Field Camera 3 and F356W (3560 nm, medium-infrared) from NirCAM on JWST

# %%
# set the random seed to get the same random set of images each time, or comment it out to get different ones!
np.random.seed(206265)

# select 16 random image indices:
example_ids = np.random.choice(hdu[1].data.shape[0], 16)
# pull the F160W image (index=1) from the simulated dataset for these selections | near infrared
examples = [hdu[0].data[j, 1, :, :] for j in example_ids]

# initialize your figure
fig = plt.figure(figsize=(12, 12))

# loop through the randomly selected images and plot with labels
for i, image in enumerate(examples):
    ax = fig.add_subplot(4, 4, i+1)
    # norm = simple_norm(image, 'log', max_percent=99.75)
    # contrast stretching

    ax.imshow(image, aspect='equal', cmap='binary_r')
    ax.set_title('Merger='+str(bool(hdu[1].data[example_ids[i]][0])))

    ax.axis('off')

plt.show()

# %%
# set the random seed to get the same random set of images each time, or comment it out to get different ones!
np.random.seed(206265)

# select 16 random image indices:
example_ids = np.random.choice(hdu[1].data.shape[0], 3)
# pull the F814W image (index=0) from the simulated dataset for these selections
examples = [hdu[0].data[j, :, :, :] for j in example_ids] # not really, we're pulling all bands

# initialize your figure
fig = plt.figure(figsize=(12, 12))

# loop through the randomly selected images and plot with labels
imgcnt = 1
for i, image in enumerate(examples):
    for band in range (3):
        ax = fig.add_subplot(3, 3, imgcnt)
        norm = simple_norm(image[band], 'log', max_percent=99.75)

        ax.imshow(image[band], aspect='equal', cmap='binary_r')
        ax.set_title('Merger='+str(bool(hdu[1].data[example_ids[i]][0])))

        ax.axis('off')
        imgcnt += 1

plt.show()
print(example_ids)

# %% [markdown]
# # Preprocessing

# %% [markdown]
# ### Data preparation

# %%
# X = np.asarray(hdu[0].data).astype('float32')
# y = np.asarray(hdu[1].data).astype('float32')

# %%
# switch to channel last format
# X = np.moveaxis(X, 1, -1)
X.shape

# %%
# np.save('dataset.npy', X)
# np.save('labels.npy', y)

# %% [markdown]
# ## Denoising

# %% [markdown]
# ### Plotting and utility functions

# %%
def image_entropy(image):
    """
    Compute the entropy of an image.

    Parameters:
    - image: numpy.ndarray
        The input image as a 2D array.

    Returns:
    - entropy: float
        The entropy of the image.
    """
    histogram, _ = np.histogram(image.flatten(), bins=256, density=True)
    
    histogram = histogram[histogram != 0] # Remove zero probabilities to avoid errors in the log calculation

    probabilities = histogram / np.sum(histogram) # Normalize the histogram to get probabilities
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy

# %%
def plot_orig_samples(seed=206265, x=X, y=y, num_samples=3):
    # set the random seed to get the same random set of images each time, or comment it out to get different ones!
    np.random.seed(seed)

    # select 16 random image indices:
    example_ids = np.random.choice(x.shape[0], num_samples)
    # pull the F814W image (index=0) from the simulated dataset for these selections
    examples = [x[j, :, :, :] for j in example_ids] # not really, we're pulling all bands

    # initialize your figure
    fig = plt.figure(figsize=(12, 12))

    # loop through the randomly selected images and plot with labels
    imgcnt = 1
    for i, image in enumerate(examples):
        for band in range (3):
            ax = fig.add_subplot(num_samples, 3, imgcnt)
            ax.imshow(image[:,:,band], aspect='equal', cmap='binary_r')
            ax.set_title(f'Merger={str(bool(y[example_ids[i]]))}\nEntropy={image_entropy(image[:,:,band]):.2f}')

            ax.axis('off')
            imgcnt += 1
    plt.show()

# %%
def plot_transformed_samples(fun, seed=206265, x=X, y=y, num_samples=3, band_wise_transform=False,*args, **kwargs):
        # set the random seed to get the same random set of images each time, or comment it out to get different ones!
    np.random.seed(seed)

    # select 16 random image indices:
    example_ids = np.random.choice(x.shape[0], num_samples)
    # pull the F814W image (index=0) from the simulated dataset for these selections
    examples = [x[j, :, :, :] for j in example_ids] # not really, we're pulling all bands

    # initialize your figure
    fig = plt.figure(figsize=(12, 12))

    # loop through the randomly selected images and plot with labels
    imgcnt = 1
    if not band_wise_transform:
        for i, image in enumerate(examples):
            filtered_image = fun(image, *args, **kwargs)
            for band in range (3):
                filtered_entropy = image_entropy(filtered_image[:,:,band])
                unfiltered_entropy = image_entropy(image[:,:,band])
                diff = filtered_entropy-unfiltered_entropy

                ax = fig.add_subplot(num_samples, 3, imgcnt)
                ax.imshow(filtered_image[:,:,band], aspect='equal', cmap='binary_r')
                ax.set_title(f'Merger={str(bool(y[example_ids[i]]))}\nEntropy={filtered_entropy:.2f} [{"+" if diff > 0 else ""}{diff:.2f}]')

                ax.axis('off')
                imgcnt += 1
    else:
        for i, image in enumerate(examples):
            for band in range (3):
                filtered_image = fun(image[:,:,band], *args, **kwargs)
                filtered_entropy = image_entropy(filtered_image)
                unfiltered_entropy = image_entropy(image[:,:,band])
                diff = filtered_entropy-unfiltered_entropy
                
                ax = fig.add_subplot(num_samples, 3, imgcnt)
                ax.imshow(filtered_image, aspect='equal', cmap='binary_r')
                ax.set_title(f'Merger={str(bool(y[example_ids[i]]))}\nEntropy={filtered_entropy:.2f} [{"+" if diff > 0 else ""}{diff:.2f}]')

                ax.axis('off')
                imgcnt += 1
    plt.show()

# %% [markdown]
#
# ### Subtract the median

# %%
def sub_median(image):
    median = np.median(image[0:10, :])
    print(median)
    return image - median

# %%
X_sub_median = np.array([np.array([image[:, :, band] - np.median(image[:, :, band]) for band in range(3)]).T for image in X])

# %%
plot_transformed_samples(sub_median, x=X, y=y, num_samples=3, band_wise_transform=True)

# %% [markdown]
# ### Fourier Transform 
# The rationale and hypothesis here is that noise is that the noise is concentrated in high frequencies. Hence we decompose each channel of each image in its consituent frequencies and keep only a percentage of the lower frequencies discarding the higher ones.

# %% [markdown]
# The procedure is explained on a single sample image

# %%
plot_transformed_samples(fourierDenoise.denoise_sample, x=X, y=y, num_samples=1, band_wise_transform=False)

# %%
X_fft = fourierDenoise.denoise_dataset(X)

# %% [markdown]
# #### Fourier explanation

# %%
fig = plt.figure(figsize=(12, 12))
sample = X_fft[4399]
for band in range(3):
    ax = fig.add_subplot(1, 3, band+1)
    img_fft = fftpack.fft2(sample[:,:,band])    # 2 dimensional fast fourier transform of a single band of the image
    ax.title.set_text('Band '+str(band))
    plt.imshow(np.abs(img_fft), norm=LogNorm(vmin=5), cmap='viridis')
    plt.colorbar(ax=ax, orientation='horizontal')
plt.show()

# %%
img_fft = fftpack.fft2(sample[:,:,0])
keep_fraction = 0.38 # keep only 38% of the original frequencies. The lower the fraction, the more aggressive the filtering -> more noise removed but less detail
im_fft2 = img_fft.copy()
r, c = im_fft2.shape
im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0 # set the "inner" rows to zero
im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0 # set the "inner" columns to zero
plt.imshow(np.abs(im_fft2), norm=LogNorm(vmin=5), cmap='viridis')
plt.colorbar()
plt.title('Filtered Spectrum')
plt.show()

# %%
im_new = fftpack.ifft2(im_fft2).real # inverse fourier transform, reconstruct the image
orig_entropy = image_entropy(sample)
filtered_entropy = image_entropy(im_new)
plt.figure(figsize=(12, 12))
ax = plt.subplot(1, 2, 1)
ax.imshow(sample[:,:,0], cmap='binary_r')
ax.title.set_text(f'Original Image: Entropy={orig_entropy:.2f}')
ax.axis('off')
ax = plt.subplot(1, 2, 2)
ax.imshow(im_new, cmap='binary_r')
ax.title.set_text(f'Filtered Image: Entropy={filtered_entropy:.2f} (~{filtered_entropy-orig_entropy:.2f})')
ax.axis('off')
plt.show()

# %% [markdown]
# The following data visualization shows how critical is the choice of the percentage of kept frequencies. Keeping a low 

# %% [markdown]
# ### Morphology-based denoising

# %%
plot_transformed_samples(morphologicalDenoise.rolling_ball_background_subtraction, x=X, y=y, num_samples=1, band_wise_transform=True, radius=5)

# %%
plot_transformed_samples(morphologicalDenoise.top_hat_transform, seed=206265, x=X, y=y, num_samples=1, band_wise_transform=True, radius=2)

# %% [markdown]
# ### Mixture of models

# %%
plot_transformed_samples(gmmDenoise.background_subtraction, seed=206265, x=X, y=y, num_samples=1, band_wise_transform=False)

# %%
X_gmm = gmmDenoise.denoise_dataset(X)

# %% [markdown]
# ### Denoising with U-Net

# %%
random_state = 0
X = X[:, :72, :72, :] # crop the images to 72x72 pixels to properly fit the U-Net model

X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=random_state)
X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=random_state)

input_shape = X_train.shape[1:]

# %%
model = unetDenoise.load_model('unet_precomputed/unet_model_4epochs.keras', input_shape)

# %%
X_unet = unetDenoise.predict(model, X)

# %%
plot_orig_samples(x=X_unet, y=y, num_samples=1)

# %% [markdown]
# Next, reshape the image array as follows: (number_of_images, image_width, image_length, 3). This is a “channels last” approach, where the final axis denotes the number of “spectral bands”. CNN’s will work with an arbitrary number of channels.

# %% [markdown]
# # Models

# %% [markdown]
# Now, we will build the CNN.
#
# Further details about Conv2D, MaxPooling2D, BatchNormalization, Dropout, and Dense layers can be found in the Keras Layers Documentation. https://keras.io/api/layers/
#
# Further details about the sigmoid and softmax activation function can be found in the Keras Activation Function Documentation. https://keras.io/api/layers/activations/

# %%
from keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.regularizers import l2


def heavy_cnn(initial_input_shape):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(6, 6), activation='gelu', input_shape=initial_input_shape, padding='same'))
    model.add(Conv2D(32, kernel_size=(6, 6), activation='gelu', input_shape=initial_input_shape, padding='same'))
    model.add(BatchNormalization())
    # model.add(Dropout(.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    model.add(Conv2D(64, kernel_size=(5, 5), activation='gelu', padding='same'))
    model.add(Conv2D(64, kernel_size=(5, 5), activation='gelu', padding='same'))
    model.add(BatchNormalization())
    # model.add(Dropout(.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, kernel_size=(3, 3), activation='gelu', padding='same'))
    # model.add(BatchNormalization())
    # model.add(Dropout(.2))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='gelu', padding='same'))
    model.add(BatchNormalization())
    # model.add(Dropout(.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, kernel_size=(3, 3), activation='gelu', padding='same'))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='gelu', padding='same'))
    model.add(BatchNormalization())
    # model.add(Dropout(.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='gelu'))
    model.add(Dropout(.2))
    model.add(Dense(1024, activation='gelu'))
    model.add(Dropout(.2))
    
    model.add(Dense(1, activation='sigmoid'))
    
    # opt = SGD(learning_rate=0.01, momentum=0.9)
    opt = AdamW(learning_rate=0.0001)
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    
    return model
    
    

# %%
# #Place your CNN here
# from keras.optimizers import SGD
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l2


def deep_merge(initial_input_shape):
    
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(5, 5), activation='relu', input_shape=initial_input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(Dropout(.5))
    
    
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(Dropout(.5))
    
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(Dropout(.5))
    
    model.add(Flatten())
    model.add(Dense(64, activation='softmax', kernel_regularizer=tf.keras.regularizers.L2(0.0001)))
    model.add(Dense(32, activation='softmax', kernel_regularizer=tf.keras.regularizers.L2(0.0001)))
    
    model.add(Dense(1, activation='sigmoid'))
    
    # opt = SGD(learning_rate=0.01, momentum=0.9)
    opt = Adam(learning_rate=0.0001)
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    
    return model
    
    

# %% [markdown]
# Next, we compile the model. You can use the Adam opmimizer and the binary cross entropy loss function (as this is a binary classification problem).
#
# You can learn more about optimizers and more about loss functions for regression tasks in the Keras documentation.

# %% [markdown]
# # Training

# %%
# Choose dataset
#
random_state = 42
X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.8, random_state=random_state)
X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=random_state)

# %%
from keras.optimizers import Adam
from AugmentationCallback import AugmentationCallback

nb_epoch = 800
batch_size = 128
shuffle = True

stop_early = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)
augmentation_cb = AugmentationCallback()
model = heavy_cnn(X_train.shape[1:])


# %%
with tf.device('/GPU:0'):
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=nb_epoch, batch_size=batch_size, shuffle=shuffle)

# %% [markdown]
# To visualize the performance of the CNN, we plot the evolution of the accuracy and loss as a function of training epochs, for the training set and for the validation set.

# %% [markdown]
# # Testing and result analysis

# %%
# Plot (accuracy, val accuracy, loss and val loss) vs epoch

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model accuracy and loss')
plt.ylabel('loss/accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'val_acc', 'train_loss', 'val_loss'], loc='upper left')
plt.show()

# %% [markdown]
# Observe how the loss for the validation set is higher than for the training set (and conversely, the accuracy for the validation set is lower than for the training set), suggesting that this model is suffering from overfitting.
#
# Over-fitting of the network model is mitigated by the use of regularization through dropout of some % during training, applied after all convolutional layers.
#
# You can also recompile the model, train for many more epochs, and include a callback, in cnn.train e.g.,
# callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)

# %% [markdown]
# Now Plot a confusion Matrix (Merger vs non-merger)

# %%
# Evaluate the network performance here
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Get predictions
y_pred = model.predict(X_test)
y_pred = np.round(y_pred)

# Generate confusion matrix
cm = metrics.confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(12, 8))

# Plot non-normalized confusion matrix
plt.subplot(1, 2, 1)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Non-Merger', 'Merger'])
plt.yticks(tick_marks, ['Non-Merger', 'Merger'])
thresh = cm.max() / 4.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=22)

# Plot normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.subplot(1, 2, 2)
plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Normalized Confusion Matrix')
plt.colorbar()
plt.xticks(tick_marks, ['Non-Merger', 'Merger'])
plt.yticks(tick_marks, ['Non-Merger', 'Merger'])
for i in range(cm_normalized.shape[0]):
    for j in range(cm_normalized.shape[1]):
        plt.text(j, i, format(cm_normalized[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black",
                 fontsize=22)

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# %% [markdown]
# Effect of noise? Try re-training the network with “noisy” data (i.e., modify
# the version to “noisy” and download the associated data product). Do the results change? If so, how and why? What are the pros and cons of using noisy vs. pristine data to train a ML model?

# %% [markdown]
# Can I try a different model? You can try adding layers, changing the activation functions, swapping out the loss function, or trying a different optimizer or learning rate. Experiment and see what model changes give the best results.

# %%



