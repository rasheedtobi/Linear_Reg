import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import torch.nn.functional as F


def getInputsTargets(filen, n_outputs):
    filen = np.genfromtxt(filen, dtype='float32', delimiter=',')
    k, l = filen.shape

    # n_outputs = int(input('How many columns are output: '))
    targets = filen[:, -1*n_outputs:]
    c = k - n_outputs
    inputs = filen[:, 0:(c)]

    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)
    return inputs, targets, n_outputs, c


inputs, targets, n_outputs, c = getInputsTargets('crop_yield.txt', 2)
batch_size = 5

train_ds = TensorDataset(inputs, targets)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

model = nn.Linear(c, n_outputs)
loss_fn = F.mse_loss
# Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)


# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):

    # Repeat for given number of epochs
    for epoch in range(num_epochs):

        # Train with batches of data
        for xb, yb in train_dl:

            # 1. Generate predictions
            pred = model(xb)

            # 2. Calculate loss
            loss = loss_fn(pred, yb)

            # 3. Compute gradients
            loss.backward()

            # 4. Update parameters using gradients
            opt.step()

            # 5. Reset the gradients to zero
            opt.zero_grad()

        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch +
                                                       1, num_epochs, loss.item()))

# For example, for 1000 epochs Uncomment below


# fit(1000, model, loss_fn, opt, train_dl)

# Uncommemnt To Observe the predictions if the trained data were th inputs. Not the best way  to eveluate but just observing
# print(model(inputs))
