import os
import numpy as np


def data_loading(configuration):

    path = os.path.join(configuration["files_location"], configuration["file_name"])

    data = np.fromfile(path, configuration["endian"], configuration["dimensions"][1] * configuration["dimensions"][0])

    image = data.reshape(configuration["dimensions"][0], configuration["dimensions"][1])

    cut_image = image[configuration["cut_values"][2]:configuration["cut_values"][3],
                      configuration["cut_values"][0]:configuration["cut_values"][1]]

    return cut_image
