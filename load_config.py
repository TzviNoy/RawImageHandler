import yaml


def load_config_file(path_to_config_file):

    with open(path_to_config_file) as file:
        configuration = yaml.load(file, Loader=yaml.FullLoader)

    print(configuration)
