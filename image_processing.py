import numpy as np
from skimage import exposure


bottom, top = np.percentile(cut_image, (1, 99))
clipped_image = np.clip(switched_bytes, bottom, top)
normalized_image = exposure.rescale_intensity(clipped_image)
equalized_image = exposure.equalize_hist(normalized_image)
adapt_equalized_image = exposure.equalize_adapthist(equalized_image)