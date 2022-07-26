import numpy as np

import torch


def linearReg(filen, epochs, n_outputs):

    filen = np.genfromtxt(filen, dtype='float32', delimiter=',')
    k, l = filen.shape

    # n_outputs = int(input('How many columns are output: '))
    targets = filen[:, -1*n_outputs:]
    inputs = filen[:, 0:(k-n_outputs)]

    p, q = inputs.shape
    r, s = targets.shape

    w = torch.randn(q, s, requires_grad=True)
    b = torch.randn(s, requires_grad=True)

    # Convert inputs and targets to tensors
    inputs = torch.from_numpy(inputs)
    targets = torch.from_numpy(targets)

    def model(x):
        return x@w + b

    def loss_fxn(t1, t2):
        diff = t1 - t2
        return torch.sum(diff * diff) / diff.numel()

    # Train for 100 epochs
    for i in range(epochs):
        preds = model(inputs)
        loss = loss_fxn(preds, targets)
        loss.backward()
        with torch.no_grad():
            w -= w.grad * 1e-5
            b -= b.grad * 1e-5
            w.grad.zero_()
            b.grad.zero_()
    print(preds)


# For example
linearReg('filname', 500, 2)
