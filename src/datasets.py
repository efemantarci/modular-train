import torch
from torch.utils.data import Dataset
import json
import os
import pandas as pd
import h5py
    
class VoxelDataset3DPointsAndDistance(Dataset):
    """
    Dataset class for loading the voxel grids and the 3D points and distance of the two transmitters.
    Voxel grids are stored in hdf5 files in sparse format. It is converted to dense format before returning.
    Args:
        path (str): Path to the folder containing the dataset.
        time_bins (int): Number of time bins in the dataset.
        spatial_bins (int): Number of spatial grids for every dimension in the dataset. So 20 means 20x20x20 grid.
    """
    def __init__(self, path, time_bins=10, spatial_bins=20):
        self.folder = path
        self.df = pd.read_csv(f"{path}/config.csv")
        self.len = len(os.listdir(path)) - 1
        self.time_bins = time_bins
        self.spatial_bins = spatial_bins
        print("Length of dataset: ", self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with h5py.File(os.path.join(self.folder, f"{idx}.h5"), 'r') as hf:
            indices = hf['indices'][:]
            values = hf['values'][:]
            shape = hf['shape'][:]

        # Converting dense tensor to sparse tensor
        indices = torch.tensor(indices, dtype=torch.int64)
        values = torch.tensor(values, dtype=torch.float32)
        voxel_grid = torch.sparse_coo_tensor(indices.T, values, size=tuple(shape), dtype=torch.float32)
        voxel_grid = voxel_grid.to_dense()
        
        row = self.df.iloc[idx]
        tx_centers = json.loads(row["tx_centers"])
        center1, center2 = torch.tensor(tx_centers[0]), torch.tensor(tx_centers[1])
        # Assuming first transmitter is closer to origin
        if torch.linalg.norm(center1) > torch.linalg.norm(center2):
            center1, center2 = center2, center1
        center1_distance = torch.linalg.norm(center1)
        center2_distance = torch.linalg.norm(center2)
        center1 /= center1_distance
        center2 /= center2_distance
        # Making distances one dimensional so we can concatenate them
        center1_distance = center1_distance.view(1)
        center2_distance = center2_distance.view(1)
        # Returning voxel grid and the two 3D points with their distances
        return voxel_grid, torch.cat((center1,center2,center1_distance,center2_distance))
    
class VoxelDataset3DPoints(Dataset):
    """
    Dataset class for loading the voxel grids and the 3D tx_centers.
    Voxel grids are stored in hdf5 files in sparse format. It is converted to dense format before returning.
    Args:
        path (str): Path to the folder containing the dataset.
        time_bins (int): Number of time bins in the dataset.
        spatial_bins (int): Number of spatial grids for every dimension in the dataset. So 20 means 20x20x20 grid.
    """
    def __init__(self, path, time_bins=10, spatial_bins=20):
        self.folder = path
        self.df = pd.read_csv(f"{path}/config.csv")
        self.len = len(os.listdir(path)) - 1
        self.time_bins = time_bins
        self.spatial_bins = spatial_bins
        print("Length of dataset: ", self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with h5py.File(os.path.join(self.folder, f"{idx}.h5"), 'r') as hf:
            indices = hf['indices'][:]
            values = hf['values'][:]
            shape = hf['shape'][:]

        # Converting dense tensor to sparse tensor
        indices = torch.tensor(indices, dtype=torch.int64)
        values = torch.tensor(values, dtype=torch.float32)
        voxel_grid = torch.sparse_coo_tensor(indices.T, values, size=tuple(shape), dtype=torch.float32)
        voxel_grid = voxel_grid.to_dense()
        
        row = self.df.iloc[idx]
        tx_centers = json.loads(row["tx_centers"])
        center1, center2 = torch.tensor(tx_centers[0]), torch.tensor(tx_centers[1])
        # Assuming first transmitter is closer to origin
        if torch.linalg.norm(center1) > torch.linalg.norm(center2):
            center1, center2 = center2, center1
        # Returning voxel grid and the two 3D points with their distances
        return voxel_grid, torch.cat((center1,center2))