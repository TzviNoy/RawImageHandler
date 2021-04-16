import os
import torch
import numpy as np

from load_config import load_config_file


class DataHandler:
    def __init__(self, path_to_config_file):
        self.path = path_to_config_file
        self.configuration = load_config_file(path_to_config_file)
        self.data = self.data_loading()
        self.labels = self.data_processing(self.data)

    def data_loading(self):

        path = os.path.join(self.configuration["files_location"], self.configuration["file_name"])

        data = np.fromfile(path, self.configuration["endian"], self.configuration["dimensions"][1] * self.configuration["dimensions"][0])

        image = data.reshape(self.configuration["dimensions"][0], self.configuration["dimensions"][1])

        cut_image = image[self.configuration["cut_values"][2]:self.configuration["cut_values"][3],
                          self.configuration["cut_values"][0]:self.configuration["cut_values"][1]]

        return cut_image

    def data_preprocessing(self):

        formatted_array = np.array([np.fromiter(format(element, '016b'), np.uint16) for element in self.data.flatten()])
        formatted = np.array([format(element, '016b') for element in self.data.flatten()])
        if np.sum(np.array([int(element[8:], 2) for element in formatted[1920 * 10:1920 * 11]])) == 40:
            switched_bytes = np.array([int(element[:8], 2) for element in formatted])
            switched_bytes = switched_bytes.reshape(self.data.shape[0], self.data.shape[1])
            print("8 bits image after DRC")
        else:
            switched_bytes = np.array([int(element[8:] + element[:8], 2) for element in formatted])
            switched_bytes = switched_bytes.reshape(self.data.shape[0], self.data.shape[1])
            print("16 bits raw image")
        return switched_bytes, formatted_array

    def data_to_torch(self):
        self.data = torch.tensor(self.data.astype(np.int16), dtype=torch.float)
        self.labels = torch.tensor(self.labels.astype(np.float32), dtype=torch.float)

    def data_processing(self):

        switched_bytes, formatted = self.data_preprocessing(train_image)
        bottom, top = np.percentile(train_image, (1, 99))
        clipped_image = np.clip(switched_bytes, bottom, top)
        normalized_image = exposure.rescale_intensity(clipped_image)
        equalized_image = exposure.equalize_hist(normalized_image)
        adapt_equalized_image = exposure.equalize_adapthist(equalized_image)
