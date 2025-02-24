import torch
from losses import MSEMinRaw, match_points
class Coordinate3DTesterWithIndexes():
    def set_criterion(self, criterion):
        self.criterion = criterion
        
    def calculate_batch_metrics(self, predictions, targets, indices=None):
        batch_size = predictions.size(0)
        batch_loss = torch.tensor(0.0)
        batch_results = []
        batch_euclidian_mse = 0.0
        batch_amortized_mse = 0.0
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
            other_mse = torch.nn.MSELoss()(torch.cat((real_center_2, real_center_1)), torch.cat((predicted_center_1, predicted_center_2)))
            amortized_mse = min(euclidean_mse, other_mse)
            batch_euclidian_mse += euclidean_mse.item()
            batch_amortized_mse += amortized_mse.item()
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
                "euclidean_mse": euclidean_mse.item(),
                "amortized_mse": amortized_mse.item()
            }
            batch_results.append(data_log)
        return batch_loss / batch_size, (batch_euclidian_mse / batch_size, batch_amortized_mse / batch_size), batch_results

class Coordinate3DTesterRawWithIndexes():
    # Tests the model with raw 3D coordinates. So distance info is in xyz coordinates.
    def __init__(self, normalization_factor=1):
        self.normalization_factor = normalization_factor
    def set_criterion(self, criterion):
        self.criterion = criterion
        
    def calculate_batch_metrics(self, predictions, targets, indices=None):
        batch_size = predictions.size(0)
        batch_loss = torch.tensor(0.0)
        batch_results = []
        batch_euclidian_mse = 0.0
        batch_amortized_mse = 0.0
        # If indices are not provided, use range(batch_size)
        if indices is None:
            indices = range(batch_size)
            
        for i, idx in enumerate(indices):
            target = targets[i]
            prediction = predictions[i]

            target = target * self.normalization_factor
            prediction = prediction * self.normalization_factor
            
            # Calculate loss for the batch, considering all points
            loss = self.criterion(prediction, target)
            batch_loss += loss.item()

            euclidean_mse = torch.nn.MSELoss()(target, prediction)
            amortized_mse = MSEMinRaw()(target, prediction)

            batch_euclidian_mse += euclidean_mse.item()
            batch_amortized_mse += amortized_mse.item()
            
            real_centers = target.view(-1, 3)
            predicted_centers = prediction.view(-1, 3)
            # Get the matching indices from Hungarian algorithm
            row_ind, col_ind = match_points(predicted_centers, real_centers)
            
            matched_predictions = predicted_centers[row_ind]
            matched_targets = real_centers[col_ind]
            
            # Log the results for each point in the batch with matched ordering
            data_log = {
                "index": idx,  # Add the dataset index
            }
            
            # Create real centers and predicted centers for each point
            data_log.update({
                "real_centers": matched_targets.cpu().numpy().tolist(),
                "predicted_centers": matched_predictions.cpu().numpy().tolist(),
            })
            
            # Add loss metrics
            data_log.update({
                "loss": loss.item(),
                "euclidean_mse": euclidean_mse.item(),
                "amortized_mse": amortized_mse.item()
            })
            
            batch_results.append(data_log)
            
            
            # real_center_1 = target[:3]
            # real_center_2 = target[3:6]
            # predicted_center_1 = prediction[:3]
            # predicted_center_2 = prediction[3:6]
            # euclidean_mse = torch.nn.MSELoss()(torch.cat((real_center_1, real_center_2)), torch.cat((predicted_center_1, predicted_center_2)))
            # other_mse = torch.nn.MSELoss()(torch.cat((real_center_2, real_center_1)), torch.cat((predicted_center_1, predicted_center_2)))
            # amortized_mse = min(euclidean_mse, other_mse)
            # batch_euclidian_mse += euclidean_mse.item()
            # batch_amortized_mse += amortized_mse.item()
            # # Log the results for each point in the batch
            # data_log = {
            #     "index": idx,  # Add the dataset index
            #     "real_x_1": target[0].item(),
            #     "real_y_1": target[1].item(),
            #     "real_z_1": target[2].item(),
            #     "predicted_x_1": prediction[0].item(),
            #     "predicted_y_1": prediction[1].item(),
            #     "predicted_z_1": prediction[2].item(),
            #     "real_x_2": target[3].item(),
            #     "real_y_2": target[4].item(),
            #     "real_z_2": target[5].item(),
            #     "predicted_x_2": prediction[3].item(),
            #     "predicted_y_2": prediction[4].item(),
            #     "predicted_z_2": prediction[5].item(),
            #     "loss": loss.item(),
            #     "euclidean_mse": euclidean_mse.item(),
            #     "amortized_mse": amortized_mse.item()
            # }
            # batch_results.append(data_log)
            
        return batch_loss / batch_size, (batch_euclidian_mse / batch_size, batch_amortized_mse / batch_size), batch_results