import torch

import numpy as np

from neural_network import Net
from trainer import Trainer


def run_model(x, y, num_of_epochs, learning_rate, net, train=True):

    model = net()
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(optimizer, criterion, net)

    saved_loss = trainer.train(num_of_epochs, x, y)

    plt.plot(saved_loss)
    plt.show()