import matplotlib.pyplot as plt

from data_handler import DataHandler
from trainer import Trainer


def pipeline(path, net, criterion, optimizer, num_of_epochs):

    data_handler = DataHandler(path)

    trainer = Trainer(optimizer, criterion, net)

    saved_loss = trainer.train(num_of_epochs, data_handler.torch_data["features"], data_handler.torch_data["labels"])

    plt.plot(saved_loss)
    plt.show()
