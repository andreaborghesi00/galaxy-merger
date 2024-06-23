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
# # Imports and initialization

# %%
# arrays and math
import numpy as np
from scipy import fftpack

# data handling
import pickle
import os
import albumentations as albus
from albumentations.pytorch import ToTensorV2

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
from AugmentationCallback import AugmentationCallback
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
import gmmDenoise
import morphologicalDenoise
import fourierDenoise
import unetDenoise


# load data
y = np.load('datasets/labels.npy')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%
X = np.load('datasets/dataset.npy')

# %%
# gpu_devices = tf.config.experimental.list_physical_devices('GPU')
# for device in gpu_devices:
#     tf.config.experimental.set_memory_growth(device, True)

# %% [markdown]
# ## Data retrieval

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
plot_orig_samples(x=X_unet, y=y, num_samples=5)

# %% [markdown]
# Next, reshape the image array as follows: (number_of_images, image_width, image_length, 3). This is a “channels last” approach, where the final axis denotes the number of “spectral bands”. CNN’s will work with an arbitrary number of channels.

# %% [markdown]
# # Pytorch dataset and loaders

# %%
# Choose dataset
experiment_name = "gmm_heavy"
random_state = 42
X = np.load('datasets/dataset.npy')
X = gmmDenoise.background_subtraction_dataset(X)
X_train, X_valtest, y_train, y_valtest = train_test_split(X, y, test_size=0.2, random_state=random_state)
X_test, X_val, y_test, y_val = train_test_split(X_valtest, y_valtest, test_size=0.5, random_state=random_state)
X_train[0].shape


# %%
class GalaxyDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = np.array(X)
        self.y = np.array(y)
        self.transform = transform
        # channel last to channel first
        if transform is None:
            print('Transforming images')
            self.X = np.moveaxis(self.X, -1, 1)
        self.y = np.expand_dims(self.y, axis=1)
        # x to tensor
        # self.X = torch.tensor(self.X, dtype=torch.float32)
        # self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.transform is not None:
            return self.transform(image=self.X[idx])['image'], self.y[idx]
        return self.X[idx], self.y[idx]


# %%
augmentations = albus.Compose([
    albus.HorizontalFlip(p=0.5),
    albus.VerticalFlip(p=0.5),
    albus.RandomRotate90(p=0.5),
    # albus.RandomBrightnessContrast(p=0.5), # they're already pretty noisy, so perhaps dont
    ToTensorV2()
])

# %%
train_ds = GalaxyDataset(X_train, y_train)
val_ds = GalaxyDataset(X_val, y_val)
test_ds = GalaxyDataset(X_test, y_test)

# %%
train_ds.__getitem__(0)[0].shape

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
            pbar.set_description(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {local_train_acc_hist[-1]:.4f}, val_acc (previous): {last_val_acc:.4f}')
            pbar.update(1)
        train_acc_hist.append(np.mean(local_train_acc_hist))

        last_val_acc, last_val_loss = validate(model, val_dl)
        val_acc_hist.append(last_val_acc)
        val_loss_hist.append(last_val_loss)
    return train_loss_hist, val_loss_hist, train_acc_hist, val_acc_hist



# %%
import timm
class GalaxyMergerClassifier(nn.Module):
    def __init__(self, encoder, num_classes=2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(encoder.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

# 2. Create the model
encoder = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_nano', pretrained=True, num_classes=0)
model = GalaxyMergerClassifier(encoder)

# %%
model = HeavyCNN().to(device)
criterion = nn.BCELoss()
optimizer = AdamW(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
epochs = 30

# %%
summary(model, (3, 72, 72));

# %%
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
pickle.dump(history, open(f'results/history_{experiment_name}_{i}.pkl', 'wb'))
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'model': model,	
    'optimizer': optimizer,
    'scheduler': scheduler,
    'history': history
}, f'results/model_{experiment_name}_{i}.pt')

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

# %% [markdown]
# # Compare results

# %%
hist_a = pickle.load(open('results/history_heavy_augmented_0.pkl', 'rb'))
hist_b = pickle.load(open('results/history_heavy_0.pkl', 'rb'))

# %%
# pretty comparing plots
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(hist_a['train_loss'], label='train_augmented')
plt.plot(hist_a['val_loss'], label='val_augmented')
plt.plot(hist_b['train_loss'], label='train')
plt.plot(hist_b['val_loss'], label='val')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(hist_a['train_acc'], label='train_augmented')
plt.plot(hist_a['val_acc'], label='val_augmented')
plt.plot(hist_b['train_acc'], label='train')
plt.plot(hist_b['val_acc'], label='val')
plt.title('Accuracy')
plt.legend()
plt.show()

# %% [markdown]
# # Tensorflow stuff

# %%
from keras.optimizers import SGD
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.regularizers import l2


def heavy_cnn(initial_input_shape):
    
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(6, 6), activation='gelu', input_shape=initial_input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(32, kernel_size=(6, 6), activation='gelu', input_shape=initial_input_shape, padding='same'))
    model.add(BatchNormalization())
    # model.add(Dropout(.2))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    
    model.add(Conv2D(64, kernel_size=(5, 5), activation='gelu', padding='same'))
    model.add(BatchNormalization())
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
    model.add(BatchNormalization())
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
    opt = AdamW(learning_rate=0.00005)
    
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    
    return model
    
    

# %%
def create_galaxy_merger_model(input_shape=(72, 72, 3), trainable_base=False):
    # Load the pretrained EfficientNetB0 model
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the base model if trainable_base is False
    base_model.trainable = trainable_base
    
    # Create the new model
    model = models.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation='gelu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

def preprocess_image(image):
    # image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image



# %%
import timm
from huggingface_hub import login

login("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")


# %%
encoder = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_nano', pretrained=True, num_classes=0)

# %%
import tensorflow as tf
import torch
import timm

class TFZoobotEncoder(tf.keras.Model):
    def __init__(self, pytorch_model):
        super(TFZoobotEncoder, self).__init__()
        self.pytorch_model = pytorch_model
        
    @tf.function
    def call(self, inputs):
        # Convert TensorFlow tensor to PyTorch tensor
        x = tf.transpose(inputs, [0, 3, 1, 2])  # NHWC to NCHW
        x = tf.cast(x, tf.float32)
        
        # Run inference
        features = tf.py_function(self._run_pytorch_model, [x], tf.float32)
        
        # Ensure the output shape is set
        features.set_shape([None, 768, None, None])  # Adjust the 768 if your model outputs a different number of channels
        return features
    
    def _run_pytorch_model(self, x):
        x_torch = torch.from_numpy(x.numpy())
        with torch.no_grad():
            features = self.pytorch_model(x_torch)
        return features.numpy()

# Load the PyTorch model
encoder_pt = timm.create_model('hf_hub:mwalmsley/zoobot-encoder-convnext_nano', pretrained=True, num_classes=0)
encoder_pt.eval()

# Create TensorFlow wrapper
encoder_tf = TFZoobotEncoder(encoder_pt)

# Create the full model
def create_galaxy_merger_model(encoder, num_classes=2):
    inputs = tf.keras.Input(shape=(72, 72, 3))
    x = encoder(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_galaxy_merger_model(encoder_tf)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

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
# ## Training

# %%
import functools

# %%
# data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,  # Random rotation between 0 and 20 degrees
    width_shift_range=0.2,  # Random horizontal shift
    height_shift_range=0.2,  # Random vertical shift
    horizontal_flip=True,  # Random horizontal flip
    zoom_range=0.2,  # Random zoom
    shear_range=0.2,  # Random shear
    fill_mode='nearest',  # Strategy for filling in newly created pixels
)

# X_train = map(preprocess_image, X_train)
# X_train = np.array(list(X_train))

# X_val = map(preprocess_image, X_val)
# X_val = np.array(list(X_val))
# Assume x_train is your training data (shape: (num_samples, height, width, channels))
# and y_train is your labels

# Fit the ImageDataGenerator to your data
datagen.fit(X_train)

# %%
nb_epoch = 40
shuffle = True

# stop_early = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
augmented_data_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
model = heavy_cnn(X_train.shape[1:])

# model = create_galaxy_merger_model(X_train.shape[1:], trainable_base=True)


# %%
X.shape

# %%
with tf.device('/GPU:0'):
    history = model.fit(augmented_data_generator, validation_data=(X_val, y_val), epochs=nb_epoch, batch_size=batch_size, shuffle=shuffle)

# %% [markdown]
# To visualize the performance of the CNN, we plot the evolution of the accuracy and loss as a function of training epochs, for the training set and for the validation set.

# %% [markdown]
# ## Testing and result analysis

# %%
# Plot (accuracy, val accuracy, loss and val loss) vs epoch

pickle.dump(history.history, open(f'results/history_{experiment_name}.pkl', 'wb'))

plt.title('model loss')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.show()
plt.title('model accuracy')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
# plt.ylabel('loss/accuracy')
plt.xlabel('epoch')
plt.legend(['train_acc', 'val_acc', 'train_loss', 'val_loss'], loc='upper left')
plt.show()

# %%
load_hist = pickle.load(open(f'results/history_{experiment_name}.pkl', 'rb'))

plt.title('model loss')
plt.plot(load_hist['accuracy'], label='train')
plt.plot(load_hist['val_accuracy'], label='val')
plt.show()

plt.title('model accuracy')
plt.plot(load_hist['loss'], label='train')
plt.plot(load_hist['val_loss'], label='val')
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



