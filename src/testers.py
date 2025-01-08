import torch
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
            real_center_1 = target[:3]
            real_center_2 = target[3:6]
            predicted_center_1 = prediction[:3]
            predicted_center_2 = prediction[3:6]
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
                "loss": loss.item(),
                "euclidean_mse": euclidean_mse.item(),
                "amortized_mse": amortized_mse.item()
            }
            batch_results.append(data_log)
            
        return batch_loss / batch_size, (batch_euclidian_mse / batch_size, batch_amortized_mse / batch_size), batch_results
    
class Coordinate3DTesterDeltaDistanceWithIndexes():
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
            real_center_2 = target[3:6] * (target[6] + target[7])
            predicted_center_1 = prediction[:3] * prediction[6]
            predicted_center_2 = prediction[3:6] * (prediction[6] + prediction[7])
            euclidean_mse = torch.nn.MSELoss()(torch.cat((real_center_1, real_center_2)), torch.cat((predicted_center_1, predicted_center_2)))
            other_mse = torch.nn.MSELoss()(torch.cat((real_center_1, real_center_2)), torch.cat((predicted_center_1, predicted_center_2)))
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
                "predicted_distance_2": (prediction[6] + prediction[7]).item(),
                "real_distance_1": target[6].item(),
                "real_distance_2": (target[6] + target[7]).item(),
                "loss": loss.item(),
                "euclidean_mse": euclidean_mse.item(),
                "amortized_mse": amortized_mse.item()
            }
            batch_results.append(data_log)
            
        return batch_loss / batch_size, (batch_euclidian_mse / batch_size, batch_amortized_mse / batch_size), batch_results
    
class Probability3DTesterWithIndexes():
    def set_criterion(self, criterion):
        self.criterion = criterion
    
    def calculate_batch_metrics(self, outputs, targets, indices=None):
        # outputs: (batch_size, spatial_bins, spatial_bins, spatial_bins)
        # targets: (batch_size, 2, 3)
        
        device = outputs.device
        batch_size = outputs.size(0)
        spatial_bins = outputs.size(1)
        batch_loss = torch.tensor(0.0,device=device)
        batch_euclidean_distance = 0.0
        batch_results = []

        if indices is None:
            indices = range(batch_size)
        
        for i, idx in enumerate(indices):
            output = outputs[i]  # Shape: (spatial_bins, spatial_bins, spatial_bins)
            target_positions = targets[i]  # Shape: (2, 3)
            
            # Flatten the output probabilities
            output_flat = output.view(-1)

            # Get top 2 indices with highest probabilities
            topk = torch.topk(output_flat, k=2)
            top_indices = topk.indices  # Shape: (2,)

            # Convert flat indices to 3D coordinates
            predicted_positions = torch.zeros(2, 3, dtype=torch.long, device=device)
            predicted_positions[:, 0] = top_indices // (spatial_bins ** 2)
            predicted_positions[:, 1] = (top_indices % (spatial_bins ** 2)) // spatial_bins
            predicted_positions[:, 2] = top_indices % spatial_bins

            # Since the order doesn't matter, compute both permutations
            dist1 = torch.sum((predicted_positions[0] - target_positions[0]) ** 2) + \
                    torch.sum((predicted_positions[1] - target_positions[1]) ** 2)
            dist2 = torch.sum((predicted_positions[0] - target_positions[1]) ** 2) + \
                    torch.sum((predicted_positions[1] - target_positions[0]) ** 2)
            min_dist = torch.min(dist1, dist2).float()
            euclidean_distance = torch.sqrt(min_dist)
            batch_euclidean_distance += euclidean_distance.item()

            # Compute the loss for this sample
            loss = self.criterion(output.unsqueeze(0), target_positions.unsqueeze(0))
            batch_loss += loss.item()
            
            # Log the results
            data_log = {
                "index": idx,
                "true_positions": target_positions.tolist(),
                "predicted_positions": predicted_positions.tolist(),
                "loss": loss.item(),
                "euclidean_distance": euclidean_distance.item()
            }
            batch_results.append(data_log)
            
        avg_loss = batch_loss / batch_size
        avg_euclidean_distance = batch_euclidean_distance / batch_size
        
        return avg_loss, avg_euclidean_distance, batch_results
