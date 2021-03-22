import image_processing
import torch.nn
from neural_network import Net
import numpy as np
from skimage import exposure
from data_loader import data_loading
from preprocessing import data_preprocessing


if __name__ == "__main__":

    config1 = {"endian": "<u2",
               "dimensions": [1084, 1920],
               "cut_values": [0, 1280, 3, 1026],
               "files_location": r"C:\Users\Tzvi\Documents\programming\RawImages",
               "file_name": r"00040.raw"}

    config2 = config1.copy()
    config2["file_name"] = r"00039.raw"

    train_image = data_loading(config1)
    test_image = data_loading(config2)

    switched_bytes, train_data = data_preprocessing(train_image)
    bottom, top = np.percentile(train_image, (1, 99))
    clipped_image = np.clip(switched_bytes, bottom, top)
    normalized_image = exposure.rescale_intensity(clipped_image)
    equalized_image = exposure.equalize_hist(normalized_image)
    adapt_equalized_image = exposure.equalize_adapthist(equalized_image)

    net = Net()
    print(net)

    params = list(net.parameters())
    print(len(params))
    print(params[0].size())

    x = torch.tensor(train_data.astype(np.int16), dtype=torch.float)
    y = torch.tensor(equalized_image.flatten().astype(np.float32), dtype=torch.float).unsqueeze(0).T
    num_of_iteration = 1000
    learning_Rate = 1e-3
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_Rate)

    for iteration in range(num_of_iteration):

        # in your training loop:
        optimizer.zero_grad()  # zero the gradient buffers
        y_pred = net(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        if iteration % 25 == 0:
            print(f"After {iteration} iterations, the loss is {loss.item()}")
        loss.backward()
        optimizer.step()  # Does the update

    net.show_predict()
