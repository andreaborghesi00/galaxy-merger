from sklearn.mixture import GaussianMixture
import numpy as np
import pickle

def background_modeling(dataset, num_components=3):
    """
    Perform background modeling using Gaussian Mixture Model (GMM).

    Parameters:
    - dataset: numpy.ndarray
        Input dataset of images (4D array with shape [num_images, height, width, num_channels]).
    - num_components: int, optional
        Number of Gaussian components in the mixture model.
        Default is 3.

    Returns:
    - background_model: GaussianMixture
        Trained Gaussian Mixture Model representing the background.
    """
    # Reshape dataset to 2D array (num_pixels x num_channels)
    num_images, height, width, num_channels = dataset.shape
    flattened_dataset = dataset.reshape(-1, num_channels)

    # Fit Gaussian Mixture Model to the flattened dataset
    gmm = GaussianMixture(n_components=num_components, covariance_type='full')
    gmm.fit(flattened_dataset)

    return gmm

def background_subtraction(image, gmm=None, gmm_type='precomputed_full', channel_last=True):
    """
    Perform background subtraction using a pre-trained Gaussian Mixture Model (GMM).

    Parameters:
    - gmm: GaussianMixture
        Trained Gaussian Mixture Model representing the background.
    - image: numpy.ndarray
        The input image to be processed.

    Returns:
    - background_subtracted_image: numpy.ndarray
        The background-subtracted image.
    """
    if gmm is None:
        gmm_types = ['precomputed_5k', 'precomputed_full']
        if gmm_type not in gmm_types:
            raise ValueError(f'Invalid GMM type. Expected one of: {gmm_types}')
        elif gmm_type == 'precomputed_5k':
            gmm = pickle.load(open('gmm_precomputed/gmm_model_5k.pkl', 'rb'))
        elif gmm_type == 'precomputed_full':
            gmm = pickle.load(open('gmm_precomputed/gmm_model_full.pkl', 'rb'))

    # Reshape the image to a 2D array (num_pixels x num_channels)
    height, width, num_channels = image.shape
    flattened_image = image.reshape(-1, num_channels)

    # Compute the log-likelihood of each pixel
    log_likelihood = gmm.score_samples(flattened_image)

    # Reshape the log-likelihood to the original image shape
    bg_estimate = log_likelihood.reshape(height, width)
    bg_subtracted_img = np.array([image[:,:,band] - bg_estimate for band in range(3)])
    if channel_last: bg_subtracted_img = np.moveaxis(bg_subtracted_img, 0, -1)
    return bg_subtracted_img

def background_subtraction_dataset(dataset, gmm=None, gmm_type='precomputed_full', channel_last=True):
    """
    Apply Gaussian Mixture Model (GMM) background subtraction to a dataset.

    Args:
        dataset (ndarray): The input dataset to apply background subtraction to.
        gmm (GaussianMixture, optional): The pre-trained GMM model. If not provided, a new model will be trained.
        gmm_type (str, optional): The type of GMM model to use. Valid options are 'precomputed_5k', 'precomputed_full', and 'new'.
            Defaults to 'precomputed_full'.
        channel_last (bool, optional): Whether the channel dimension is the last dimension in the dataset. Defaults to True.

    Returns:
        ndarray: The background-subtracted dataset.

    Raises:
        ValueError: If an invalid GMM type is provided.

    """
    gmm_types = ['precomputed_5k', 'precomputed_full', 'new']
    if gmm_type not in gmm_types:
        raise ValueError(f'Invalid GMM type. Expected one of: {gmm_types}')
    elif gmm_type == 'new':
        gmm = background_modeling(dataset)
    elif gmm_type == 'precomputed_5k':
        gmm = pickle.load(open('precomputed/gmm_model_5k.pkl', 'rb'))
    elif gmm_type == 'precomputed_full':
        gmm = pickle.load(open('precomputed/gmm_model.pkl', 'rb'))

    background_subtracted_dataset = np.zeros_like(dataset)
    for i in range(dataset.shape[0]):
        background_subtracted_dataset[i] = background_subtraction(dataset[i], gmm, channel_last)
    return background_subtracted_dataset