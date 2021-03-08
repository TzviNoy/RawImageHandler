import numpy as np


def data_preprocessing(img):
    formatted = np.array([format(element, '016b') for element in img.flatten()])
    if np.sum(np.array([int(element[8:], 2) for element in formatted[1920*10:1920*11]])) == 40:
        switched_bytes = np.array([int(element[:8], 2) for element in formatted])
        switched_bytes = switched_bytes.reshape(img.shape[0], img.shape[1])
        print("8 bits image after DRC")
    else:
        switched_bytes = np.array([int(element[8:] + element[:8], 2) for element in formatted])
        switched_bytes = switched_bytes.reshape(img.shape[0], img.shape[1])
        print("16 bits raw image")
    return switched_bytes