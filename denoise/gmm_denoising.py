import numpy as np
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
import pickle

def gmm_background_modeling(dataset, num_components=3):
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

def gmm_background_subtraction(gmm, image, channel_last=True):
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
    # Reshape the image to a 2D array (num_pixels x num_channels)
    height, width, num_channels = image.shape
    flattened_image = image.reshape(-1, num_channels)
    print(flattened_image.shape)
    # Compute the log-likelihood of each pixel
    log_likelihood = gmm.score_samples(flattened_image)
    print(log_likelihood.shape)

    pred = gmm.predict(flattened_image)
    print(pred.shape)
    # Reshape the log-likelihood to the original image shape
    bg_estimate = pred.reshape(height, width)
    bg_subtracted_img = np.array([image[:,:,band] - bg_estimate for band in range(3)])
    if channel_last: bg_subtracted_img = np.moveaxis(bg_subtracted_img, 0, -1)
    return bg_subtracted_img

if __name__ == '__main__':
    dataset = np.load('dataset.npy')
    # dataset = dataset[:5000]

    # background_model = gmm_background_modeling(dataset)
    # pickle.dump(background_model, open('gmm_model_full.pkl', 'wb'))

    background_model = pickle.load(open('gmm_model_full.pkl', 'rb'))

    image = dataset[4399]
    background_subtracted_image = gmm_background_subtraction(background_model, image)
    print(background_subtracted_image.shape)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image[:, :, 0], cmap='binary_r')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(background_subtracted_image[:,:,0], cmap='binary_r')
    plt.title('Background Subtracted Image')
    plt.axis('off')

    plt.show()