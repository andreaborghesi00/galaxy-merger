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
# # Imports and Initialization

# %%
# arrays and math
import numpy as np
from scipy import fftpack

# data handling
import pickle
import os
import albumentations as albus
from albumentations.pytorch import ToTensorV2
import gc


# fits
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.visualization import simple_norm


# plotting
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

# keras
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Activation, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, Input, Conv2DTranspose, Dropout, GlobalAveragePooling2D, AveragePooling2D, Flatten
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import Adam
from augmentation_callback import AugmentationCallback
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import models

# torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, ConvTranspose2d, BatchNorm2d, Dropout2d, ReLU, Sigmoid, GELU, Module, Sequential, Linear
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
import torchmetrics as tm
from torchsummary import summary

# sklearn (for machine learning)
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf

# denoise
import denoise.gmm_denoise as gmmDenoise
import denoise.morphological_denoise as morphological_denoise
import denoise.fourier_denoise as fourierDenoise
import denoise.unet_denoise as unet_denoise


# load data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
X = np.load('datasets/dataset.npy')
y = np.load('datasets/labels.npy')

# %%
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

# %% [markdown]
# ## Data retrieval

# %%
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
# X.shape

# %%
# np.save('datasets/dataset.npy', X)
# np.save('datasets/labels.npy', y)

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
def dataset_entropy(X):
    """
    Compute the entropy of a dataset.

    Parameters:
    - X: numpy.ndarray
        The input dataset as a 4D array.

    Returns:
    - entropy_mean: float
        The entropy of the dataset.
    - entropy_std: float
        The standard deviation of the entropy of the dataset.
    """
    entropies = np.array([image_entropy(image) for image in X])
    entropy_mean = np.mean(entropies)
    entropy_std = np.std(entropies)

    return entropy_mean, entropy_std


# %%
def dataset_entropy_diff(X, Y):
    """
    Compute the difference in entropy between two datasets.

    Parameters:
    - X: numpy.ndarray
        The first input dataset
    - Y: numpy.ndarray
        The second input dataset

    Returns:
    - entropy_diff: float
        The difference in entropy between the two datasets.
    """
    entropy_mean_X, _ = dataset_entropy(X)
    entropy_mean_Y, _ = dataset_entropy(Y)

    entropy_diff = entropy_mean_X - entropy_mean_Y

    return entropy_diff


# %%
def denoise_info(filename, fun, verbose=True, *args, **kwargs):
    """
    Load or generate denoised information from a dataset file.

    Parameters:
    - filename (str): The name of the dataset file.
    - fun (function): The function used to generate denoised information if the file doesn't exist.
    - verbose (bool): Whether to print the entropy and entropy difference information.
    - *args: Additional positional arguments to pass to the `fun` function.
    - **kwargs: Additional keyword arguments to pass to the `fun` function.

    Returns:
    - entropy (tuple): A tuple containing the entropy value and its standard deviation.
    - entropy_diff (float): The difference in entropy compared to the original dataset.

    """
    path = 'datasets/'+filename+'.npy'
    try:
        Y = np.load(path)
    except:
        Y = fun(X, *args, **kwargs)
        np.save(path, Y)
    
    entropy, entropy_diff = dataset_entropy(Y), dataset_entropy_diff(Y, X)
    if verbose: print(f'{filename}: Entropy={entropy[0]:.2f} +/- {entropy[1]:.2f}, Diff={entropy_diff:.2f}')
    return entropy, entropy_diff


# %%
def plot_orig_samples(seed=206265, x=X, y=y, num_samples=3):
    """
    Plot original samples from a simulated dataset.

    Parameters:
    - seed (int): Random seed to get the same random set of images each time. Default is 206265.
    - x (numpy.ndarray): Input images dataset. Default is X.
    - y (numpy.ndarray): Labels for the input images dataset. Default is y.
    - num_samples (int): Number of random samples to plot. Default is 3.

    Returns:
    None
    """

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
    """
    Plot transformed samples.

    Parameters:
    - fun: The transformation function to apply to the images.
    - seed: The random seed to get the same random set of images each time.
    - x: The input dataset of images.
    - y: The labels for the images.
    - num_samples: The number of random images to select and plot.
    - band_wise_transform: Whether to apply the transformation function band-wise or to the entire image.
    - *args, **kwargs: Additional arguments to pass to the transformation function.

    Returns:
    None
    """

    # set the random seed to get the same random set of images each time, or comment it out to get different ones!
    np.random.seed(seed)

    # select 16 random image indices:
    example_ids = np.random.choice(x.shape[0], num_samples)
    print(example_ids)
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
# ### Fourier Transform 
# The rationale and hypothesis here is that noise is that the noise is concentrated in high frequencies. Hence we decompose each channel of each image in its consituent frequencies and keep only a percentage of the lower frequencies discarding the higher ones.

# %% [markdown]
# The procedure is explained on a single sample image

# %%
plot_transformed_samples(fourierDenoise.denoise_sample, x=X, y=y, num_samples=1, band_wise_transform=False)

# %%
denoise_info('dataset_fft', fourierDenoise.denoise_dataset, verbose=True);

# %% [markdown]
# #### Fourier explanation

# %%
fig = plt.figure(figsize=(12, 12))
sample = X[4369]
for band in range(3):
    ax = fig.add_subplot(1, 3, band+1)
    img_fft = fftpack.fft2(sample[:,:,band])    # 2 dimensional fast fourier transform of a single band of the image
    ax.title.set_text('Band '+str(band))
    plt.imshow(np.abs(img_fft), norm=LogNorm(vmin=5), cmap='viridis')
    plt.colorbar(ax=ax, orientation='horizontal')
plt.show()

# %%
img_fft = fftpack.fft2(sample[:,:,0])
frac = 0.35
im_fft2 = img_fft.copy()
r, c = im_fft2.shape
im_fft2[int(r*frac):int(r*(1-frac))] = 0 # set the "inner" rows to zero
im_fft2[:, int(c*frac):int(c*(1-frac))] = 0 # set the "inner" columns to zero
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
ax.title.set_text(f'Denoised Image: Entropy={filtered_entropy:.2f} (~{filtered_entropy-orig_entropy:.2f})')
ax.axis('off')
plt.show()

# %%
plot_transformed_samples(fourierDenoise.denoise_sample, x=X, y=y, num_samples=3, band_wise_transform=False)

# %% [markdown]
# The following data visualization shows how critical is the choice of the percentage of kept frequencies. Keeping a low 

# %% [markdown]
# ### Morphology-based denoising

# %% [markdown]
# #### Rollling ball background subtraction

# %%
plot_transformed_samples(morphological_denoise.rolling_ball_background_subtraction, x=X, y=y, num_samples=3, band_wise_transform=False, radius=5)

# %%
from scipy.ndimage import minimum_filter
from skimage.morphology import disk

sample = X[4369]
im_new  = sample[:,:,band] - minimum_filter(sample[:,:,band], footprint=disk(5))
orig_entropy = image_entropy(sample)
filtered_entropy = image_entropy(im_new)
plt.figure(figsize=(12, 12))
ax = plt.subplot(1, 2, 1)
ax.imshow(sample[:,:,2], cmap='binary_r')
ax.title.set_text(f'Original Image: Entropy={orig_entropy:.2f}')
ax.axis('off')
ax = plt.subplot(1, 2, 2)
ax.imshow(im_new, cmap='binary_r')
ax.title.set_text(f'Denoised Image: Entropy={filtered_entropy:.2f} (~{filtered_entropy-orig_entropy:.2f})')
ax.axis('off')
plt.show()

# %%
denoise_info('dataset_bg_sub', morphological_denoise.rolling_ball_background_subtraction, radius=5);

# %% [markdown]
# #### Rolling ball top hat

# %%
plot_transformed_samples(morphological_denoise.top_hat_transform, x=X, y=y, num_samples=3, band_wise_transform=False, radius=5)


# %%
sample = X[4369]
im_new = morphological_denoise.top_hat_transform_single_band(sample, 5, 0)# inverse fourier transform, reconstruct the image
orig_entropy = image_entropy(sample)
filtered_entropy = image_entropy(im_new)
plt.figure(figsize=(12, 12))
ax = plt.subplot(1, 2, 1)
ax.imshow(sample[:,:,2], cmap='binary_r')
ax.title.set_text(f'Original Image: Entropy={orig_entropy:.2f}')
ax.axis('off')
ax = plt.subplot(1, 2, 2)
ax.imshow(im_new, cmap='binary_r')
ax.title.set_text(f'Denoised Image: Entropy={filtered_entropy:.2f} (~{filtered_entropy-orig_entropy:.2f})')
ax.axis('off')
plt.show()

# %%
denoise_info('dataset_tophat', morphological_denoise.top_hat_transform_dataset, radius=5);

# %% [markdown]
# ### Mixture of models

# %%
plot_transformed_samples(gmmDenoise.background_subtraction, seed=206265, x=X, y=y, num_samples=3, band_wise_transform=False)

# %%
denoise_info('dataset_gmm', gmmDenoise.background_subtraction_dataset, verbose=True);

# %% [markdown]
# ### Autoencoders: U-Net

# %%
random_state = 0
X = X[:, :72, :72, :] # crop the images to 72x72 pixels to properly fit the U-Net model

X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=random_state)
X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=random_state)

input_shape = X_train.shape[1:]

# %%
model = unet_denoise.simpler_model(input_shape)
model.summary()

# %%
model = unet_denoise.train(model, X_train, X_train, X_val, X_val, batch_size=32, epochs=4, save_path='simple_unet_4epochs.h5')

# %%
X_test_unet = unet_denoise.predict(X_test, model)
plot_orig_samples(seed=206261, x=X_test_unet, y=y_test, num_samples=3)

# %%
plot_orig_samples(seed=206261, x=X_test, y=y_test, num_samples=3)

# %%
model = unet_denoise.load_model('unet_precomputed/unet_model_4epochs.keras', input_shape)

# %%
denoise_info('dataset_unet', unet_denoise.predict, model=model, verbose=True);

# %%
X_unet = np.load('datasets/dataset_unet.npy')
plot_orig_samples(x=X_unet, y=y, num_samples=3)
del X_unet

# %% [markdown]
# Next, reshape the image array as follows: (number_of_images, image_width, image_length, 3). This is a “channels last” approach, where the final axis denotes the number of “spectral bands”. CNN’s will work with an arbitrary number of channels.

# %% [markdown]
# # Pytorch dataset and loaders

# %%
# Choose dataset
# Dataset root: 'datasets/'
# Available dataset types: '', 'fft', 'bg_sub', 'tophat', 'gmm', 'unet'
dataset_type = 'fft'

experiment_name = f"{dataset_type}_heavy_augmented"
random_state = 42
dataset_path = f'datasets/dataset{f"_{dataset_type}" if dataset_type != "" else ""}.npy'

X = np.load(dataset_path)
print(dataset_path)

# split 70:10:20
X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=random_state)
X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.33333, random_state=random_state)


# %%
class GalaxyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = np.array(X)
        self.y = np.array(y)
        self.transform = transform
        # channel last to channel first
        if transform is None:
            self.X = np.moveaxis(self.X, -1, 1)
        self.y = np.expand_dims(self.y, axis=1)
        # x to tensor
        if transform is None:
            self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(image=self.X[idx])['image'], self.y[idx]
        return self.X[idx], self.y[idx]


# %% [markdown]
# ### Data augmentation

# %%
augmentations = albus.Compose([
    albus.HorizontalFlip(p=0.5),
    albus.VerticalFlip(p=0.5),
    albus.RandomRotate90(p=0.5),
    # albus.RandomBrightnessContrast(p=0.5), # they're already pretty noisy, so perhaps dont
    ToTensorV2()
])

# %%
train_ds = GalaxyDataset(X_train, y_train, transform=augmentations)
val_ds = GalaxyDataset(X_val, y_val)
test_ds = GalaxyDataset(X_test, y_test)

# %%
train_ds.__getitem__(19)[1]

# %%
batch_size = 256

train_dl = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4
)

val_dl = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)

test_dl = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4
)


# %% [markdown]
# # Models

# %% [markdown]
# Now, we will build the CNN.
#
# Further details about Conv2D, MaxPooling2D, BatchNormalization, Dropout, and Dense layers can be found in the Keras Layers Documentation. https://keras.io/api/layers/
#
# Further details about the sigmoid and softmax activation function can be found in the Keras Activation Function Documentation. https://keras.io/api/layers/activations/

# %%
class HeavyCNN(nn.Module):
    def __init__(self):
        super(HeavyCNN, self).__init__()
        self.conv1 = nn.Sequential(
            Conv2d(3, 32, kernel_size=6, padding='same'),
            BatchNorm2d(32),
            GELU(),
            Conv2d(32, 32, kernel_size=6, padding='same'),
            BatchNorm2d(32),
            GELU(),
            MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(
            Conv2d(32, 64, kernel_size=5, padding='same'),
            BatchNorm2d(64),
            GELU(),
            Conv2d(64, 64, kernel_size=5, padding='same'),
            BatchNorm2d(64),
            GELU(),
            MaxPool2d(2)
        )
        
        self.conv3 = nn.Sequential(
            Conv2d(64, 128, kernel_size=3, padding='same'),
            BatchNorm2d(128),
            GELU(),
            Conv2d(128, 128, kernel_size=3, padding='same'),
            BatchNorm2d(128),
            GELU(),
            Conv2d(128, 128, kernel_size=3, padding='same'),
            BatchNorm2d(128),
            GELU(),
            MaxPool2d(2)
        )

        self.conv4 = nn.Sequential(
            Conv2d(128, 256, kernel_size=3, padding='same'),
            BatchNorm2d(256),
            GELU(),
            Conv2d(256, 256, kernel_size=3, padding='same'),
            BatchNorm2d(256),
            GELU(),
            Conv2d(256, 256, kernel_size=3, padding='same'),
            BatchNorm2d(256),
            GELU(),
            MaxPool2d(2)
        )

        self.fc1 = nn.Sequential(
            Linear(256*4*4, 1024),
            GELU(),
            Dropout2d(0.5),
            Linear(1024, 512),
            GELU(),
            Dropout2d(0.5),
            Linear(512, 1),
            Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(-1, 256*4*4)
        x = self.fc1(x)
        return x



# %%
class DeepMerge(Module):
    def __init__(self):
        super(DeepMerge, self).__init__()
        self.conv1 = Sequential(
            Conv2d(3, 8, kernel_size=5, padding=2),
            ReLU(),
            BatchNorm2d(8),
            MaxPool2d(2),
            Dropout2d(0.5)
        )  # output size 37x37

        self.conv2 = Sequential(
            Conv2d(8, 16, kernel_size=3, padding=1),
            ReLU(),
            BatchNorm2d(16),
            MaxPool2d(2),
            Dropout2d(0.5)
        )  # output size 18x18

        self.conv3 = Sequential(
            Conv2d(16, 32, kernel_size=3, padding=1),
            ReLU(),
            BatchNorm2d(32),
            MaxPool2d(2),
            Dropout2d(0.5)
        )  # output size 9x9

        self.fc1 = Sequential(
            Linear(32*9*9, 64),
            nn.Softmax(dim=1),
            Linear(64, 32),
            nn.Softmax(dim=1),
            Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.fc1(x)
        return x


# %%
class ClassifierHead(Module):
    def __init__(self, in_features):
        super(ClassifierHead, self).__init__()
        self.fc1 = nn.Sequential(
            Linear(in_features, 1024),
            GELU(),
            Dropout2d(0.5),
            Linear(1024, 512),
            GELU(),
            Dropout2d(0.5),
            Linear(512, 1),
            Sigmoid()
        )

    def forward(self, x):
        return self.fc1(x)


# %%
from torchvision.models import resnet50, ResNet50_Weights

resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

# %%
for child in resnet.children():
    print(child)

# %%
for child in resnet.named_children():
    print(child)

# %%

# %%
in_features = resnet.fc.in_features
resnet.fc = ClassifierHead(in_features)


# %%
resnet = resnet.to(device)
resnet.fc

# %%
summary(resnet, (3, 75, 75));

# %%
unfreeze_layers = ['layer3','layer4', 'fc']
for name, param in resnet.named_parameters():
    if any(layer_name in name for layer_name in unfreeze_layers):
        param.requires_grad = True
    else:
        param.requires_grad = False

# %%
# from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier
# import timm
# # or FinetuneableZoobotRegressor, or FinetuneableZoobotTree

# import torch.nn as nn

# class CustomHead(nn.Module):
#     def __init__(self, input_dim, dropout_rate=0.3):
#         super().__init__()
#         self.gelu = nn.GELU()

#         self.fc1 = nn.Linear(input_dim, 1024)
#         self.dropout = nn.Dropout(dropout_rate)
        
#         self.fc2 = nn.Linear(1024, 512)
#         self.dropout = nn.Dropout(dropout_rate)
        
#         self.fc3 = nn.Linear(512, 1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.gelu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         x = self.gelu(x)
#         x = self.dropout(x)
#         x = self.fc3(x)
#         x = self.sigmoid(x)
#         return x
    


# class GalaxyMergerClassifier(nn.Module):
#     def __init__(self, encoder):
#         super().__init__()
#         self.encoder = encoder
        
#         # Freeze the encoder parameters
#         for param in self.encoder.encoder.parameters():
#             param.requires_grad = False
        
#         # Unfreeze the custom head parameters
#         for param in self.encoder.head.parameters():
#             param.requires_grad = True

#     def forward(self, x):
#         # x should be your input tensor of shape (batch_size, 3, 75, 75)
#         features = self.encoder.forward_features(x)
#         output = self.encoder.head(features)
#         return output

# %%
import timm
class GalaxyMergerClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(69, 1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        print(features.shape)
        return self.classifier(features)

# 2. Create the model
# encoder = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_nano', pretrained=True, num_classes=2)


# %%
def validate(model, dl):
    model.eval()
    val_acc = tm.Accuracy(task='binary', average='micro').to(device)
    val_bce = nn.BCELoss()
    val_loss_hist = []

    for x, y in dl:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            out = model(x)
            val_acc.update(out, y)
            val_loss = val_bce(out, y)
            val_loss_hist.append(val_loss.item())

    return val_acc.compute().item(), np.mean(val_loss_hist)


# %%
def train(model, train_dl, val_dl, epochs, optimizer, scheduler, criterion):
    pbar = tqdm(total=epochs*len(train_dl))
    model.train()
    train_acc = tm.Accuracy(task='binary', average='micro').to(device)
    last_val_acc = -1

    val_acc_hist = []
    train_acc_hist = []

    val_loss_hist = []
    train_loss_hist = []
    train_acc 
    for epoch in range(epochs):
        local_train_acc_hist = []
        local_train_loss_hist = []
        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_acc.update(out, y)
            local_train_acc_hist.append(train_acc.compute().item())
            local_train_loss_hist.append(loss.item())
            pbar.set_description(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {local_train_acc_hist[-1]:.4f}, val_acc (previous): {last_val_acc:.4f} | best_val_acc: {max(val_acc_hist) if len(val_acc_hist) > 0 else -1:.4f} at epoch {np.argmax(val_acc_hist)+1 if len(val_acc_hist) > 0 else -1}')
            pbar.update(1)
            
        train_loss_hist.append(np.mean(local_train_loss_hist))
        train_acc_hist.append(np.mean(local_train_acc_hist))

        last_val_acc, last_val_loss = validate(model, val_dl)
        val_acc_hist.append(last_val_acc)
        val_loss_hist.append(last_val_loss)
    return train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist



# %%
# model = GalaxyMergerClassifier(encoder).to(device)

# encoder = FinetuneableZoobotClassifier(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', num_classes=2)
# encoder_output_dim = encoder.encoder_dim

# # Create and set the new custom head
# new_head = CustomHead(input_dim=encoder_output_dim)
# encoder.head = new_headcriterion = nn.BCELoss()
model = HeavyCNN().to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
epochs = 250
criterion = nn.BCELoss()

# %%
summary(model, (3, 75, 75));

# %%
train_loss, val_loss, train_acc, val_acc = train(resnet, train_dl, val_dl, epochs, optimizer, scheduler, criterion)

history = {
    'train_loss': train_loss,
    'val_loss': val_loss,
    'train_acc': train_acc,
    'val_acc': val_acc
}

i = 0
while os.path.exists(f'results/history_{experiment_name}_{i}.pkl'):
    i += 1
pickle.dump(history, open(f'results/history_{experiment_name}_{i}.pkl', 'wb'))
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'model': model,
    'model_name': model.__class__.__name__,
    'optimizer': optimizer,
    'optimizer_name': optimizer.__class__.__name__,
    'scheduler': scheduler,
    'scheduler_name': scheduler.__class__.__name__,
    'history': history
}, f'results/model_{experiment_name}_{i}.pth')

# %%
# pretty plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='train')
plt.plot(val_loss, label='val')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label='train')
plt.plot(val_acc, label='val')
plt.title('Accuracy')
plt.legend()
plt.show()

# %%
validate(model, test_dl)


# %%
def bulk_train_test(model_classes, dataset_types, augment = True, batch_size = 256, epochs = 250, random_state = 42, optimizer = None, scheduler = None, criterion = None, lr = 2e-5):
    for model_class in model_classes:
        print(f"########## {model_class.__name__} ##########")
        model = None
        for dataset_type in dataset_types:
            if model: 
                model.cpu()
                del model
                gc.collect()
                torch.cuda.empty_cache()
                
            model = model_class().to(device)
            dataset_name = dataset_type if dataset_type != "" else "noisy"
            print(f"########## {dataset_name} ##########")
            experiment_name = f"{dataset_name}_{model_class.__name__}_augmented"
            random_state = 42
            dataset_path = f'datasets/dataset{f"_{dataset_type}" if dataset_type != "" else ""}.npy'

            X = np.load(dataset_path)

            # split 70:10:20
            X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.3, random_state=random_state)
            X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.33333, random_state=random_state)
            
            train_ds, val_ds, test_ds = GalaxyDataset(X_train, y_train), GalaxyDataset(X_val, y_val), GalaxyDataset(X_test, y_test)
            train_dl, val_dl, test_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4), DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4), DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
            
            optimizer = AdamW(model.parameters(), lr=2e-5) if optimizer is None else optimizer
            scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6) if scheduler is None else scheduler
            epochs = epochs
            criterion = nn.BCELoss() if criterion is None else criterion
            
            train_loss, val_loss, train_acc, val_acc = train(model, train_dl, val_dl, epochs, optimizer, scheduler, criterion)

            history = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }

            i = 0
            while os.path.exists(f'results/history_{experiment_name}_{i}.pkl'):
                i += 1
            # pickle.dump(history, open(f'results/history_{experiment_name}_{i}.pkl', 'wb'))
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'model': model,
                'model_name': model.__class__.__name__,
                'optimizer': optimizer,
                'optimizer_name': optimizer.__class__.__name__,
                'scheduler': scheduler,
                'scheduler_name': scheduler.__class__.__name__,
                'history': history
            }, f'results/model_{experiment_name}_{i}.pth')
            
            # pretty plots
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.plot(train_loss, label='train')
            plt.plot(val_loss, label='val')
            plt.title('Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(train_acc, label='train')
            plt.plot(val_acc, label='val')
            plt.title('Accuracy')
            plt.legend()
            plt.show()
            test_acc, test_loss = validate(model, test_dl)
            print(f"Model: {model.__class__.__name__}, Experiment: {experiment_name}, Dataset: {dataset_type}, Test (micro) accuracy: {test_acc}, Test loss: {test_loss}")
            
                        
            

# %%
bulk_train_test([HeavyCNN], ['', 'fft', 'bg_sub', 'tophat', 'gmm', 'unet'], epochs=250)

# %% [markdown]
# # Compare results

# %%
hist_a = pickle.load(open('results/history_fft_heavy_augmented_1.pkl', 'rb'))
hist_b = pickle.load(open('results/history_tophat_heavy_augmented_0.pkl', 'rb'))

# %%
# pretty comparing plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(hist_a['train_loss'], label='train FFT')
plt.plot(hist_a['val_loss'], label='val FFT')
plt.plot(hist_b['train_loss'], label='train Tophat')
plt.plot(hist_b['val_loss'], label='val Tophat')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_a['train_acc'], label='train FFT')
plt.plot(hist_a['val_acc'], label='val FFT')
plt.plot(hist_b['train_acc'], label='train Tophat')
plt.plot(hist_b['val_acc'], label='val Tophat')
plt.title('Accuracy')
plt.legend()
plt.show()

# %% [markdown]
# # Tensorflow stuff

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



