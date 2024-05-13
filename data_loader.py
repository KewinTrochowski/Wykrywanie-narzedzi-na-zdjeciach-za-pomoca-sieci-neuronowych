import json
import numpy as np
from tensorflow.keras import Sequence
from PIL import Image
from config import IMAGE_DIMS, BATCH_SIZE

def load_data(json_file):
    with open(json_file, 'r') as file:
        return json.load(file)

class DataGenerator(Sequence):
    def __init__(self, dataset, batch_size=BATCH_SIZE, dim=IMAGE_DIMS, n_channels=3, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.dataset[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, 4), dtype=int)  # assuming only one object per image for simplicity

        for i, data in enumerate(list_IDs_temp):
            image = Image.open(data[0])
            image = image.resize(self.dim)
            X[i,] = np.array(image) / 255.0

            # Normalize bounding box coordinates
            bbox = data[1][0][:4]  # Assuming only one bounding box per image
            y[i,] = [
                bbox[0] / image.width,
                bbox[1] / image.height,
                bbox[2] / image.width,
                bbox[3] / image.height
            ]

        return X, y

train_generator = DataGenerator(load_data('train_data.json'))
val_generator = DataGenerator(load_data('val_data.json'))
