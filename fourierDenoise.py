import numpy as np
from scipy import fftpack

def denoise_sample(img, keep_fraction=0.35):
    """
    Apply FFT-based denoising to an image.

    Parameters:
    - img: numpy.ndarray
        The input image to be denoised, it expects a 3D array with shape (n_rows, n_cols, n_bands).
    - keep_fraction: float, optional
        The fraction of high-frequency components to keep in the Fourier domain.
        Default is 0.35

    Returns:
    - filtered_img: numpy.ndarray
        The denoised image.

    """
    filtered_img = np.zeros_like(img)
    for band in range(3):
        img_fft = fftpack.fft2(img[:, :, band])
        im_fft2 = img_fft.copy()
        r, c = im_fft2.shape
        im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0
        im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
        filtered_img[:, :, band] = fftpack.ifft2(im_fft2).real
    return filtered_img

def denoise_dataset(dataset, keep_fraction=0.35):
    """
    Apply FFT-based denoising to a dataset of images.

    Parameters:
    - dataset: numpy.ndarray
        The input dataset of images to be denoised, it expects a 4D array with shape (n_samples, n_rows, n_cols, n_bands).
    - keep_fraction: float, optional
        The fraction of high-frequency components to keep in the Fourier domain.
        Default is 0.35

    Returns:
    - filtered_dataset: numpy.ndarray
        The denoised dataset of images.

    """
    filtered_dataset = np.zeros_like(dataset)
    for i in range(dataset.shape[0]):
        filtered_dataset[i] = denoise_sample(dataset[i], keep_fraction)
    return filtered_dataset