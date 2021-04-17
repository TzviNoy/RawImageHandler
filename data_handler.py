import os
import torch
import numpy as np

from skimage import exposure
from load_config import load_config_file


class DataHandler:
    def __init__(self, path_to_config_file):

        self.path = path_to_config_file
        self.configuration = load_config_file(path_to_config_file)
        self.raw_image = self.data_loading()
        self.switched_bytes = None
        self.formatted_array = None
        self.processed_image = None

        self.data_preprocessing()
        self.data_processing()

        self.labels = self.processed_image.flatten()
        self.torch_data = {"features": self.data_to_torch(self.formatted_array, np.int16, torch.float),
                           "labels": self.data_to_torch(self.labels, np.float32, torch.float)}

    def data_loading(self):

        path = os.path.join(self.configuration["files_location"], self.configuration["file_name"])

        data = np.fromfile(path, self.configuration["endian"], self.configuration["dimensions"][1] * self.configuration["dimensions"][0])

        image = data.reshape(self.configuration["dimensions"][0], self.configuration["dimensions"][1])

        cut_image = image[self.configuration["cut_values"][2]:self.configuration["cut_values"][3],
                          self.configuration["cut_values"][0]:self.configuration["cut_values"][1]]

        return cut_image

    def data_preprocessing(self):

        formatted_array = np.array([np.fromiter(format(element, '016b'), np.uint16) for element in self.raw_image.flatten()])
        formatted = np.array([format(element, '016b') for element in self.raw_image.flatten()])

        if np.sum(np.array([int(element[8:], 2) for element in formatted[1920 * 10:1920 * 11]])) == 40:
            switched_bytes = np.array([int(element[:8], 2) for element in formatted])
            switched_bytes = switched_bytes.reshape(self.raw_image.shape[0], self.raw_image.shape[1])
            print("8 bits image after DRC")

        else:
            switched_bytes = np.array([int(element[8:] + element[:8], 2) for element in formatted])
            switched_bytes = switched_bytes.reshape(self.raw_image.shape[0], self.raw_image.shape[1])
            print("16 bits raw image")

        self.switched_bytes = switched_bytes
        self.formatted_array = formatted_array

    @staticmethod
    def data_to_torch(data, input_type, output_type):

        return torch.tensor(data.astype(input_type), dtype=output_type)

    def data_processing(self):

        bottom, top = np.percentile(self.raw_image, (1, 99))
        clipped_image = np.clip(self.switched_bytes, bottom, top)
        normalized_image = exposure.rescale_intensity(clipped_image)
        equalized_image = exposure.equalize_hist(normalized_image)
        self.processed_image = exposure.equalize_adapthist(equalized_image)
