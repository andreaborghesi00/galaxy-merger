from skimage.morphology import disk, binary_erosion, white_tophat
from scipy.ndimage import minimum_filter
import numpy as np

def rolling_ball_background_subtraction(image, radius):
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
    # Create a spherical structuring element (rolling ball)
    selem = disk(radius)

    # Compute the morphological closing of the image using the rolling ball
    closed_image = minimum_filter(image, footprint=selem)

    # Compute the background-subtracted image
    background_subtracted_image = image - closed_image

    return background_subtracted_image

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
    selem = disk(radius)

    # Apply the top-hat transform
    top_hat_image = white_tophat(image, selem)

    return top_hat_image

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