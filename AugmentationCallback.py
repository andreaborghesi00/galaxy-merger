import tensorflow as tf
import numpy as np
class AugmentationCallback(tf.keras.callbacks.Callback):

    def __init__(self, augmentation_prob=.5, local_seed=None):
        super(AugmentationCallback, self).__init__()
        self.augmentation_prob = augmentation_prob
        self.seed = local_seed
        if self.seed is None:
            self.seed = np.random.randint(0, 10000)
    def on_batch_begin(self, batch, logs=None):
        augmentation_functions = [self.rotate_image, self.flip_image]
        for i, func in enumerate(augmentation_functions):
            # i havent tested this, but i increase the seed by i to make sure that the random number generated is different for each function
            if(tf.random.uniform(1, seed=self.seed+i) < self.augmentation_prob): func(batch)
        
        # this list comprehension should make everything unreadable like real python programmers do
        #[func(batch) for i, func in enumerate(augmentation_functions) if tf.random.uniform(1, seed=self.seed+i) < self.augmentation_prob]

    # these should be inplace operations
    def rotate_image(self, batch):
        # rotate the image
        for i in range(len(batch)):
            batch[i] = tf.contrib.image.rotate(batch[i], tf.random.uniform(1, minval=-np.pi/4, maxval=np.pi/4))

    def flip_image(self, batch):
        # flip the image
        for i in range(len(batch)):
            batch[i] = tf.image.flip_left_right(batch[i])

    def zoom_image(self, batch):
        # zoom the image
        pass

    def color_jitter(self, batch):
        # apply color jitter
        pass