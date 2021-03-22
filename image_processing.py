import numpy as np
from skimage import exposure

from data_loader import data_loading
from preprocessing import data_preprocessing

config1 = {"endian": "<u2",
           "dimensions": [1084, 1920],
           "cut_values": [0, 1280, 3, 1026],
           "files_location": r"C:\Users\Tzvi\Documents\programming\RawImages",
           "file_name": r"00040.raw"}

config2 = config1
config2["file_name"] = r"00039.raw"

train_image = data_loading(config1)
test_image = data_loading(config2)

switched_bytes, formatted = data_preprocessing(train_image)
bottom, top = np.percentile(train_image, (1, 99))
clipped_image = np.clip(switched_bytes, bottom, top)
normalized_image = exposure.rescale_intensity(clipped_image)
equalized_image = exposure.equalize_hist(normalized_image)
adapt_equalized_image = exposure.equalize_adapthist(equalized_image)
