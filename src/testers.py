import torch

class Coordinate3DTesterWithIndexes():
    def set_criterion(self, criterion):
        self.criterion = criterion
        
    def calculate_batch_metrics(self, predictions, targets, indices=None):
        batch_size = predictions.size(0)
        batch_loss = torch.tensor(0.0)
        batch_results = []
        batch_euclidian_mse = 0.0
        # If indices are not provided, use range(batch_size)
        if indices is None:
            indices = range(batch_size)
            
        for i, idx in enumerate(indices):
            target = targets[i]
            prediction = predictions[i]
            
            # Calculate loss for the batch, considering all points
            loss = self.criterion(prediction, target)
            batch_loss += loss.item()
            real_center_1 = target[:3] * target[6]
            real_center_2 = target[3:6] * target[7]
            predicted_center_1 = prediction[:3] * prediction[6]
            predicted_center_2 = prediction[3:6] * prediction[7]
            euclidean_mse = torch.nn.MSELoss()(torch.cat((real_center_1, real_center_2)), torch.cat((predicted_center_1, predicted_center_2)))
            batch_euclidian_mse += euclidean_mse.item()
            # Log the results for each point in the batch
            data_log = {
                "index": idx,  # Add the dataset index
                "real_x_1": target[0].item(),
                "real_y_1": target[1].item(),
                "real_z_1": target[2].item(),
                "predicted_x_1": prediction[0].item(),
                "predicted_y_1": prediction[1].item(),
                "predicted_z_1": prediction[2].item(),
                "real_x_2": target[3].item(),
                "real_y_2": target[4].item(),
                "real_z_2": target[5].item(),
                "predicted_x_2": prediction[3].item(),
                "predicted_y_2": prediction[4].item(),
                "predicted_z_2": prediction[5].item(),
                "predicted_distance_1": prediction[6].item(),
                "predicted_distance_2": prediction[7].item(),
                "real_distance_1": target[6].item(),
                "real_distance_2": target[7].item(),
                "loss": loss.item(),
                "euclidean_mse": euclidean_mse.item()
            }
            batch_results.append(data_log)
            
        return batch_loss / batch_size, batch_euclidian_mse / batch_size, batch_results

class Coordinate3DTesterRawWithIndexes():
    # Tests the model with raw 3D coordinates. So distance info is in xyz coordinates.
    def set_criterion(self, criterion):
        self.criterion = criterion
        
    def calculate_batch_metrics(self, predictions, targets, indices=None):
        batch_size = predictions.size(0)
        batch_loss = torch.tensor(0.0)
        batch_results = []
        batch_euclidian_mse = 0.0
        # If indices are not provided, use range(batch_size)
        if indices is None:
            indices = range(batch_size)
            
        for i, idx in enumerate(indices):
            target = targets[i]
            prediction = predictions[i]
            
            # Calculate loss for the batch, considering all points
            loss = self.criterion(prediction, target)
            batch_loss += loss.item()
            real_center_1 = target[:3]
            real_center_2 = target[3:6]
            predicted_center_1 = prediction[:3]
            predicted_center_2 = prediction[3:6]
            euclidean_mse = torch.nn.MSELoss()(torch.cat((real_center_1, real_center_2)), torch.cat((predicted_center_1, predicted_center_2)))
            batch_euclidian_mse += euclidean_mse.item()
            # Log the results for each point in the batch
            data_log = {
                "index": idx,  # Add the dataset index
                "real_x_1": target[0].item(),
                "real_y_1": target[1].item(),
                "real_z_1": target[2].item(),
                "predicted_x_1": prediction[0].item(),
                "predicted_y_1": prediction[1].item(),
                "predicted_z_1": prediction[2].item(),
                "real_x_2": target[3].item(),
                "real_y_2": target[4].item(),
                "real_z_2": target[5].item(),
                "predicted_x_2": prediction[3].item(),
                "predicted_y_2": prediction[4].item(),
                "predicted_z_2": prediction[5].item(),
                "loss": loss.item(),
                "euclidean_mse": euclidean_mse.item()
            }
            batch_results.append(data_log)
            
        return batch_loss / batch_size, batch_euclidian_mse / batch_size, batch_results