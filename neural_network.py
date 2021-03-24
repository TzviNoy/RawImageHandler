from typing import Tuple

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 4)
        self.fc3 = nn.Linear(4, 2)
        self.fc4 = nn.Linear(2, 1)

    def forward(self, x):
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def show_predict(self, image_shape: Tuple, test):
        img = self(test)
        img = img.reshape(image_shape).detach().numpy()
        plt.imshow(img, cmap='gray')
        plt.show()
