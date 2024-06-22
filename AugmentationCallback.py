import tensorflow as tf
import numpy as np
class AugmentationCallback(tf.keras.callbacks.Callback):
    """
    Callback class for applying data augmentation during training.

    Args:
        augmentation_prob (float, optional): Probability of applying augmentation. Defaults to 0.5.
        local_seed (int, optional): Local seed for random number generation. Defaults to None.
    """

    def __init__(self, augmentation_prob=.5, local_seed=None):
        super(AugmentationCallback, self).__init__()
        self.augmentation_prob = augmentation_prob
        self.seed = local_seed
        if self.seed is None:
            self.seed = np.random.randint(0, 10000)

    def on_batch_begin(self, batch, logs=None):
        """
        Method called at the beginning of each batch during training.

        Args:
            batch: The current batch of data.
            logs (dict, optional): Dictionary of logs. Defaults to None.
        """
        augmentation_functions = [self.rotate_image, self.flip_image]
        for i, func in enumerate(augmentation_functions):
            if(tf.random.uniform(1, seed=self.seed+i) < self.augmentation_prob):
                func(batch)

    def rotate_image(self, batch):
        """
        Rotate the images in the batch.

        Args:
            batch: The batch of images to be rotated.
        """
        for i in range(len(batch)):
            batch[i] = tf.contrib.image.rotate(batch[i], tf.random.uniform(1, minval=-np.pi/4, maxval=np.pi/4))

    def flip_image(self, batch):
        """
        Flip the images in the batch.

        Args:
            batch: The batch of images to be flipped.
        """
        for i in range(len(batch)):
            batch[i] = tf.image.flip_left_right(batch[i])

    def zoom_image(self, batch):
        """
        Zoom the images in the batch.

        Args:
            batch: The batch of images to be zoomed.
        """
        pass

    def color_jitter(self, batch):
        """
        Apply color jitter to the images in the batch.

        Args:
            batch: The batch of images to apply color jitter to.
        """
        pass