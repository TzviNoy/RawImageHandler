import os
import numpy as np

path = os.path.join(files_location, file_name)
data = np.fromfile(path, endian, height * width)
image = data.reshape(height, width)
cut_image = image[y0:y1, x0:x1]