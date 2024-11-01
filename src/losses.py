import torch

class MSE():
    def __init__(self):
        self.loss = torch.nn.MSELoss()
    def __call__(self, y_pred, y_true):
        return self.loss(y_pred, y_true)
    
class MSEMean():
    def __init__(self):
        self.loss = torch.nn.MSELoss()
    def __call__(self, y_pred, y_true):
        loss1 = self.loss(y_pred, y_true)
        first = y_true[:3]
        second = y_true[3:]
        y_true_reversed = torch.cat((second, first))
        loss2 = self.loss(y_pred, y_true_reversed)
        return (loss1 + loss2) / 2
    
class MSEMin():
    def __init__(self):
        self.loss = torch.nn.MSELoss()
    def __call__(self, y_pred, y_true):
        loss1 = self.loss(y_pred, y_true)
        first = y_true[:3]
        second = y_true[3:]
        y_true_reversed = torch.cat((second, first))
        loss2 = self.loss(y_pred, y_true_reversed)
        return min(loss1, loss2)
class MSEMax():
    def __init__(self):
        self.loss = torch.nn.MSELoss()
    def __call__(self, y_pred, y_true):
        loss1 = self.loss(y_pred, y_true)
        first = y_true[:3]
        second = y_true[3:]
        y_true_reversed = torch.cat((second, first))
        loss2 = self.loss(y_pred, y_true_reversed)
        return max(loss1, loss2)