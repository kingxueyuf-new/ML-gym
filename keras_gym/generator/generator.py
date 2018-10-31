from skimage.io import imread
from skimage.transform import resize
from keras.utils import Sequence
import keras
import numpy as np


class MyGenerator(Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.

        ceil(109 / 10) = 11, so return 11
        """
        return np.ceil(len(self.image_filenames) / len(self.batch_size))

    def __getitem__(self, idx):
        """Gets batch at position `index`.

            # Arguments
                index: position of the batch in the Sequence.

            # Returns
                A batch

            ceil(109 / 10) = 11, so idx range from [0,10]
        """
        batch_x_img_names = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y_labels = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        # Convert label to one hot
        batch_y_labels = keras.utils.to_categorical(batch_y_labels, num_classes=10)

        batch_x = []
        for file_name in batch_x_img_names:
            batch_x.append(resize(imread(file_name), (200, 200)))

        return np.array(batch_x), np.array(batch_y_labels)
