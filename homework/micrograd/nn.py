import numpy as np

from .engine import Tensor, Value


class Module:
    """
    Base class for every layer.
    """

    def forward(self, *args, **kwargs):
        """Depends on functionality"""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """For convenience we can use model(inp) to call forward pass"""
        return self.forward(*args, **kwargs)

    def parameters(self):
        """Return list of trainable parameters"""
        return []


class Linear(Module):
    def __init__(self, in_features, out_features, bias: bool = True):
        """Initializing model"""
        # Create Linear Module
        data = np.random.normal(
            scale=1, size=(out_features, in_features)) * \
            np.sqrt(2 / (in_features + out_features))
        data = data.tolist()
        for i in range(len(data)):
            for j in range(len(data[0])):
                data[i][j] = Value(data[i][j])
        self.W = Tensor(data)
        if bias:
            b = np.zeros(out_features)
            b = b.tolist()
            for k in range(len(b)):
                b[k] = Value(b[k])
            self.b = Tensor(b)
        else:
            self.b = None

    def forward(self, inp):
        """Y = W * x + b"""
        if self.b:
            return self.W.dot(inp) + self.b
        return self.W.dot(inp)

    def parameters(self):
        params = []
        params += self.W.parameters()
        if self.b:
            params += self.b.parameters()
        return params


class ReLU(Module):
    """The most simple and popular activation function"""

    def forward(self, inp: Tensor):
        # Create ReLU Module
        for value in inp.parameters():
            value.relu()
        return inp


class CrossEntropyLoss(Module):
    """Cross-entropy loss for multi-class classification"""

    def forward(self, inp: Tensor, label):
        # Create CrossEntropy Loss Module
        sum_ = 0
        exp_ = inp.exp()
        for i in exp_:
            sum_ += i
        loss = - inp[label] + sum_.log()
        loss.grad = loss.data
        return loss
