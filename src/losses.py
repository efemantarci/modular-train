import torch

class MSE():
    def __init__(self):
        self.loss = torch.nn.MSELoss()
    def __call__(self, y_pred, y_true):
        return self.loss(y_pred, y_true)
    
class MSEMean():
    def __init__(self):
        self.loss = torch.nn.MSELoss(reduction="none")
    def __call__(self, y_pred, y_true):
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(0)

        dir1_true = y_true[:, :3]   # normalized x1,y1,z1
        dir2_true = y_true[:, 3:6]
        dist1_true = y_true[:, 6]
        dist2_true = y_true[:, 7]
        dir1_pred = y_pred[:, :3]
        dir2_pred = y_pred[:, 3:6]
        dist1_pred = y_pred[:, 6]
        dist2_pred = y_pred[:, 7]
        loss1 = self.loss(torch.cat((dir1_pred, dir2_pred, dist1_pred.unsqueeze(1), dist2_pred.unsqueeze(1)), dim=1),
                  torch.cat((dir1_true, dir2_true, dist1_true.unsqueeze(1), dist2_true.unsqueeze(1)), dim=1))
        loss2 = self.loss(torch.cat((dir2_pred, dir1_pred, dist1_pred.unsqueeze(1), dist2_pred.unsqueeze(1)), dim=1),
                  torch.cat((dir2_true, dir1_true, dist1_true.unsqueeze(1), dist2_true.unsqueeze(1)), dim=1))
        # Add these two losses and take the mean
        return (loss1.mean(dim=1) + loss2.mean(dim=1)).mean()

class MSEMinRaw():
    def __init__(self):
        self.loss = torch.nn.MSELoss(reduction="none")
    def __call__(self, y_pred, y_true):
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(0)

        dir1_true = y_true[:, :3]   # normalized x1,y1,z1
        dir2_true = y_true[:, 3:6]
        dir1_pred = y_pred[:, :3]
        dir2_pred = y_pred[:, 3:6]
        loss1 = self.loss(torch.cat((dir1_pred, dir2_pred), dim=1),
                  torch.cat((dir1_true, dir2_true), dim=1))
        loss2 = self.loss(torch.cat((dir2_pred, dir1_pred), dim=1),
                  torch.cat((dir2_true, dir1_true), dim=1))
        return torch.min(loss1.mean(dim=1), loss2.mean(dim=1)).mean()
class MSEMin():
    def __init__(self):
        self.loss = torch.nn.MSELoss(reduction="none")
    def __call__(self, y_pred, y_true):
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(0)

        dir1_true = y_true[:, :3]   # normalized x1,y1,z1
        dir2_true = y_true[:, 3:6]
        dist1_true = y_true[:, 6]
        dist2_true = y_true[:, 7]
        dir1_pred = y_pred[:, :3]
        dir2_pred = y_pred[:, 3:6]
        dist1_pred = y_pred[:, 6]
        dist2_pred = y_pred[:, 7]
        loss1 = self.loss(torch.cat((dir1_pred, dir2_pred, dist1_pred.unsqueeze(1), dist2_pred.unsqueeze(1)), dim=1),
                  torch.cat((dir1_true, dir2_true, dist1_true.unsqueeze(1), dist2_true.unsqueeze(1)), dim=1))
        loss2 = self.loss(torch.cat((dir2_pred, dir1_pred, dist1_pred.unsqueeze(1), dist2_pred.unsqueeze(1)), dim=1),
                  torch.cat((dir2_true, dir1_true, dist1_true.unsqueeze(1), dist2_true.unsqueeze(1)), dim=1))
        return torch.min(loss1.mean(dim=1), loss2.mean(dim=1)).mean()

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
class DistanceAwareLoss():
    def __init__(self, ordering_penalty=0.1):
        self.ordering_penalty = ordering_penalty

    def __call__(self, y_pred, y_true):
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(0)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(0)

        # Split predictions
        dir1_true = y_true[:, :3]   # normalized x1,y1,z1
        dir2_true = y_true[:, 3:6]  # normalized x2,y2,z2
        dist1_true = y_true[:, 6]   # distance scalar 1
        dist2_true = y_true[:, 7]   # distance scalar 2

        dir1_pred = y_pred[:, :3]
        dir2_pred = y_pred[:, 3:6]
        dist1_pred = y_pred[:, 6]
        dist2_pred = y_pred[:, 7]

        # Normalize predicted directions
        dir1_pred = dir1_pred / dir1_pred.norm(dim=1, keepdim=True)
        dir2_pred = dir2_pred / dir2_pred.norm(dim=1, keepdim=True)

        # Direction loss using dot product between unit vectors
        dir_loss1 = 1 - torch.sum(dir1_true * dir1_pred, dim=1)
        dir_loss2 = 1 - torch.sum(dir2_true * dir2_pred, dim=1)

        # Distance loss
        dist_loss = torch.mean((dist1_true - dist1_pred)**2 + 
                               (dist2_true - dist2_pred)**2)

        # Ordering penalty
        ordering_violation = torch.clamp(dist1_pred - dist2_pred, min=0)

        # Final loss
        return (torch.mean(dir_loss1 + dir_loss2) + 
                dist_loss + 
                self.ordering_penalty * torch.mean(ordering_violation))