import os
import numpy as np
import gc

import morphologicalDenoise
import fourierDenoise
import unetDenoise

from astropy.io import fits
from astropy.utils.data import download_file

import torch
from torch.utils.data import Dataset, DataLoader

datasets_root = 'datasets'

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
    
def get_dataloader(X, y, transform, *args, **kwargs):
    dataset = GalaxyDataset(X, y, transform)
    return DataLoader(dataset, *args, **kwargs)

def generate_dataset(X, func, *args, **kwargs):
    """
    Generate a dataset using the provided function.

    Parameters:
    - X: The input dataset. If None, it will be loaded from a default source.
    - func: The function used to generate the dataset.
    - *args: Additional positional arguments to be passed to the function.
    - **kwargs: Additional keyword arguments to be passed to the function.

    Returns:
    - Y: The generated dataset.

    """
    global datasets_root

    if X is None:
        X, _ = load_dataset()

    Y = func(X, *args, **kwargs)
    return Y

def load_dataset(dataset_type="noisy"):
    """
    Load the dataset from the specified dataset_type. If the dataset has not been yet generated, it will be generated.

    Args:
        dataset_type (str): The type of dataset to load.
            Available options:  "noisy" -> noisy dataset,
                                "pristine" -> pristine dataset,
                                "fft" -> FFT transformed dataset,
                                "bg_sub" -> Background subtracted dataset,
                                "top_hat" -> Top-hat transformed dataset,
                                "unet" -> U-Net transformed dataset.

    Returns:
        tuple: A tuple containing the loaded dataset X and the corresponding labels y.
    """
    global datasets_root
    os.makedirs(datasets_root, exist_ok=True)
    
    dataset_options = ["noisy", "fft", "bg_sub", "top_hat", "unet", "pristine"]
    if dataset_type not in dataset_options:
        raise ValueError(f"Invalid dataset type. Available options: {dataset_options}")
    else:
        try:
            dataset_path = f'dataset{f"_{dataset_type}" if dataset_type != "noisy" else ""}.npy'
            dataset_path = os.path.join(datasets_root, dataset_path)
            X = np.load(dataset_path, allow_pickle=True)
            y = np.load(os.path.join(datasets_root, 'labels.npy'), allow_pickle=True)
            return X, y
        except FileNotFoundError:
            if dataset_type == "noisy":
                version = 'noisy' # pristine or noisy. Pristine has infinite S/N, noisy has realistic S/N
                file_url = 'https://archive.stsci.edu/hlsps/deepmerge/hlsp_deepmerge_hst-jwst_acs-wfc3-nircam_illustris-z2_f814w-f160w-f356w_v1_sim-'+version+'.fits'
                hdu = fits.open(download_file(file_url, cache=True, show_progress=True))
                X = np.asarray(hdu[0].data).astype('float32')
                y = np.asarray(hdu[1].data).astype('float32')

                X = np.moveaxis(X, 1, -1) # channel first to channel last format
                np.save('datasets/dataset.npy', X)
                np.save('datasets/labels.npy', y)
                return X, y
            
            elif dataset_type == "pristine":
                version = 'pristine'
                file_url = 'https://archive.stsci.edu/hlsps/deepmerge/hlsp_deepmerge_hst-jwst_acs-wfc3-nircam_illustris-z2_f814w-f160w-f356w_v1_sim-'+version+'.fits'
                hdu = fits.open(download_file(file_url, cache=True, show_progress=True))
                X_pristine = np.asarray(hdu[0].data).astype('float32')
                y_pristine = np.asarray(hdu[1].data).astype('float32')
                X_pristine = np.moveaxis(X_pristine, 1, -1)
                np.save('datasets/dataset_pristine.npy', X_pristine)
                np.save('datasets/labels_pristine.npy', y_pristine)
                return X_pristine, y_pristine
            
            elif dataset_type == "fft":
                X, y = load_dataset()
                X_fft = generate_dataset(X, fourierDenoise.denoise_dataset)
                np.save(os.path.join(datasets_root, 'dataset_fft.npy'), X_fft)
                return X_fft, y

            elif dataset_type == "bg_sub":
                X, y = load_dataset()
                X_bg_sub = generate_dataset(X, morphologicalDenoise.rolling_ball_background_subtraction_dataset, radius=5)
                np.save(os.path.join(datasets_root, 'dataset_bg_sub.npy'), X_bg_sub)
                return X_bg_sub, y       

            elif dataset_type == "top_hat":
                X, y = load_dataset()
                X_top_hat = generate_dataset(X, morphologicalDenoise.top_hat_transform_dataset, radius=5)
                np.save(os.path.join(datasets_root, 'dataset_top_hat.npy'), X_top_hat)
                return X_top_hat, y

            elif dataset_type == "unet":
                X, y = load_dataset()
                X = X[:, :72, :72, :]
                input_shape = (72, 72, 3)
                model = unetDenoise.load_model('unet_precomputed/unet_model_4epochs.keras', input_shape)
                X_unet = generate_dataset(X, unetDenoise.predict, model=model)
                np.save(os.path.join(datasets_root, 'dataset_unet.npy'), X_unet)
                
                torch.cuda.empty_cache()

                return X_unet, y