import os
import torch
from training_pipeline import pipeline
from neural_network import Net

if __name__ == "__main__":

    path = os.path.join(os.getcwd(), "configuration.yaml")

    learning_rate = 9e-2
    model = Net()
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    pipeline(path, model, criterion, optimizer, 100)
