import os
import torch
from training_pipeline import pipeline
from neural_network import Net
from display import display

if __name__ == "__main__":

    mode = {"train": True,
            "save": False,
            "display": True}

    if mode["train"]:

        train_config_path = "train_config.yaml"

        learning_rate = 1e-2
        model = Net()
        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        trainer = pipeline(train_config_path, model, criterion, optimizer, 50)

        if mode["save"]:
            torch.save(trainer.model, os.path.join(os.getcwd(), "model.pt"))

    if mode["display"]:

        display_config_path = "display_config.yaml"
        model_path = os.path.join(os.getcwd(), "model.pt")

        display(display_config_path, model_path)
