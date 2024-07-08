from skimage.morphology import disk, white_tophat
from scipy.ndimage import minimum_filter
import numpy as np

def rolling_ball_background_subtraction(image, radius=5):
    """
    Perform background subtraction using the rolling ball algorithm.

    Parameters:
    - image: numpy.ndarray
        The input image as a grayscale image (single-channel).
    - radius: int
        The radius of the rolling ball structuring element.

    Returns:
    - background_subtracted_image: numpy.ndarray
        The background-subtracted image.
    """
    selem = disk(radius)
    closed_image = np.zeros_like(image)
    
    for band in range(3):
        closed_image[:, :, band] = minimum_filter(image[:,:,band], footprint=disk(radius))

    return image - closed_image # subtract background approximation from original image

def top_hat_transform(image, radius):
    """
    Apply the top-hat transform to an image.

    Parameters:
    - image: numpy.ndarray
        The input grayscale image.
    - radius: int
        The radius of the structuring element.

    Returns:
    - top_hat_image: numpy.ndarray
        The result of the top-hat transform.
    """
    # Create a disk-shaped structuring element
    top_hat = np.zeros_like(image)
    for band in range(3):
        top_hat[:,:,band] = white_tophat(image[:,:,band], disk(radius))
    return top_hat

def top_hat_transform_single_band(image, radius, band=0):
    return white_tophat(image[:, :, band], disk(radius))

def rolling_ball_background_subtraction_dataset(dataset, radius):
    """
    Perform background subtraction using the rolling ball algorithm on a dataset of images.

    Parameters:
    - dataset: numpy.ndarray
        The input dataset of images to be processed.
    - radius: int
        The radius of the rolling ball structuring element.

    Returns:
    - background_subtracted_dataset: numpy.ndarray
        The background-subtracted dataset of images.
    """
    background_subtracted_dataset = np.zeros_like(dataset)
    for i in range(dataset.shape[0]):
        background_subtracted_dataset[i] = rolling_ball_background_subtraction(dataset[i], radius)
        
    return background_subtracted_dataset

def top_hat_transform_dataset(dataset, radius):
    """
    Apply the top-hat transform to a dataset of images.

    Parameters:
    - dataset: numpy.ndarray
        The input dataset of images to be processed.
    - radius: int
        The radius of the structuring element.

    Returns:
    - top_hat_dataset: numpy.ndarray
        The result of the top-hat transform applied to the dataset.
    """
    top_hat_dataset = np.zeros_like(dataset)
    for i in range(dataset.shape[0]):
            top_hat_dataset[i] = top_hat_transform(dataset[i], radius)
    return top_hat_dataset