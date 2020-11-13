import torch
import numpy as np
from torch.autograd import Variable

class linearRegression(torch.nn.Module):
    """
    Class Wrapper for Linear Regression Module
    """
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)
        self.linear.weight.data.fill_(0)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        """Inference"""
        out = self.linear(x)
        return out

    def predict(self, x):
        """ Numpy wrapper for computing prediction
            Arguments:
                x: 2-D numpy array for which predictions are to be computed
            Retuns:
                y: 1-D numpy predictions"""
        yhat = self(Variable(torch.from_numpy(x).float())).data.numpy().squeeze()
        return yhat

class polyRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, poly):
        """
        Class Wrapper for Polynomial Regression Module
        poly - degree of the polynomial feature
        """
        super(polyRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize*poly, outputSize)
        self.linear.weight.data.fill_(0)
        self.linear.bias.data.fill_(0)
        self.poly = poly

    def forward(self, x):
        """Inference"""
        out = torch.cat([1/i * x**i for i in range(1,self.poly+1)], 1)
        out = self.linear(out)
        return out

    def predict(self, x):
        """ Numpy wrapper for computing prediction
            Arguments:
                x: 2-D numpy array for which predictions are to be computed
            Retuns:
                y: 1-D numpy predictions"""
        yhat = self(Variable(torch.from_numpy(x).float())).data.numpy().squeeze()
        return yhat
