class Trainer:
    def __init__(self, optimizer, criterion, net):
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = net
        self.saved_loss = []

    def train(self, num_of_epochs, data, labels):

        for epoch in range(num_of_epochs):

            self.optimizer.zero_grad()
            y_pred = self.model(data).squeeze()
            loss = self.criterion(y_pred, labels)

            print(f"After {epoch} epochs, the loss is {loss.item()}")

            self.saved_loss.append(loss.item())

            loss.backward()
            self.optimizer.step()
