import os

import numpy as np
from skimage import exposure

from preprocessing import data_preprocessing
from load_config import load_config_file

if __name__ == "__main__":

    path = os.getcwd()

    load_config_file(os.path.join(path, "configuration.yaml"))
    switched_bytes, train_data = data_preprocessing(train_image)
    test_switched_bytes, test_data = data_preprocessing(test_image)
    bottom, top = np.percentile(train_image, (1, 99))
    clipped_image = np.clip(switched_bytes, bottom, top)
    normalized_image = exposure.rescale_intensity(clipped_image)
    equalized_image = exposure.equalize_hist(normalized_image)
    adapt_equalized_image = exposure.equalize_adapthist(equalized_image)


