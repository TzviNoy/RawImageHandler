import yaml
import os


def load_config_file(path_to_config_file):

    path = [file for file in os.listdir() if ".yaml" in file][0]

    with open(path) as file:
        configuration = yaml.load(file, Loader=yaml.FullLoader)

    return configuration
