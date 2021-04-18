import os
import torch
import matplotlib.pyplot as plt
from training_pipeline import pipeline
from neural_network import Net
from data_handler import DataHandler
from display import display

if __name__ == "__main__":

    mode = "train"

    if mode == "train":

        path = os.path.join(os.getcwd(), "configuration.yaml")

        learning_rate = 9e-2
        model = Net()
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model = pipeline(path, model, criterion, optimizer, 200)

        torch.save(model, os.path.join(os.getcwd(), "model.pt"))

    elif mode == "display":

        config_path = "display_config.yaml"
        model_path = os.path.join(os.getcwd(), "model.pt")

        display(config_path, model_path)
