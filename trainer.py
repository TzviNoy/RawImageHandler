class Trainer:
    def __init__(self, optimizer, criterion, net):
        self.optimizer = optimizer
        self.criterion = criterion
        self.model = net()

    def train(self, num_of_epochs, data, labels):

        saved_loss = []

        for epoch in range(num_of_epochs):
            self.optimizer.zero_grad()
            y_pred = self.model(data)
            loss = self.criterion(y_pred, labels)

            print(f"After {epoch} epochs, the loss is {loss.item()}")

            saved_loss.append(loss.item())

        loss.backward()
        self.optimizer.step()

        return saved_loss
