import torch
from torch.utils.data import Dataset
import json
import os
import pandas as pd
import h5py
from torch_geometric.data import Dataset, Data
import numpy as np
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

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.dist = torch.sqrt(x**2 + y**2 + z**2)

    @staticmethod
    def should_swap(first : 'Point', second: 'Point', angle_threshold = 0.1):
        # Get angle of first and second point between z axis. It should be between 0 and pi
        angle1 = torch.acos(first.z / first.dist)
        angle2 = torch.acos(second.z / second.dist)
        if abs(angle1 - angle2) > angle_threshold:
            return angle1 > angle2
        return first.dist > second.dist
        
class VoxelDatasetPointMoreOrder(Dataset):
    """
    Dataset class for loading the voxel grids and the 3D points and distance of the two transmitters.
    Voxel grids are stored in hdf5 files in sparse format. It is converted to dense format before returning.
    Returns according to ordering of the unit vectors. Alphabetical ordering
    Args:
        path (str): Path to the folder containing the dataset.
        time_bins (int): Number of time bins in the dataset.
        spatial_bins (int): Number of spatial grids for every dimension in the dataset. So 20 means 20x20x20 grid.
    """
    def __init__(self, path, time_bins=10, spatial_bins=20, normalization_factor = 1):
        self.folder = path
        self.df = pd.read_csv(f"{path}/config.csv")
        self.len = len(os.listdir(path)) - 1
        self.time_bins = time_bins
        self.spatial_bins = spatial_bins
        self.normalization_factor = normalization_factor

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
        # tx_centers = json.loads(row["tx_centers"])
        points = [Point(*torch.tensor(center)) for center in json.loads(row["tx_centers"])]
        # Bubble sort :)
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if Point.should_swap(points[i], points[j]):
                    points[i], points[j] = points[j], points[i]
        sorted_centers = torch.cat([torch.tensor([point.x, point.y, point.z]) for point in points])
        sorted_centers = sorted_centers / self.normalization_factor
        return voxel_grid, sorted_centers


END_TOKEN = [0,0,0,1]
class VoxelDatasetSequence(Dataset):
    """
    Dataset class for loading the voxel grids and the 3D points and distance of the two transmitters.
    Voxel grids are stored in hdf5 files in sparse format. It is converted to dense format before returning.
    Returns a sequence with a points coordinate and end probability.
    End token is (0,0,0) with end probability 1.
    Args:
        path (str): Path to the folder containing the dataset.
        time_bins (int): Number of time bins in the dataset.
        spatial_bins (int): Number of spatial grids for every dimension in the dataset. So 20 means 20x20x20 grid.
    """
    def __init__(self, path, time_bins=10, spatial_bins=20, normalization_factor=1):
        self.folder = path
        self.df = pd.read_csv(f"{path}/config.csv")
        self.len = len(os.listdir(path)) - 1
        self.time_bins = time_bins
        self.spatial_bins = spatial_bins
        self.normalization_factor = normalization_factor

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
        n_tx = row["tx_count"]
        # tx_centers = json.loads(row["tx_centers"])
        points = [Point(*torch.tensor(center)) for center in json.loads(row["tx_centers"])]
        # Bubble sort :)
        n = len(points)
        for i in range(n):
            for j in range(i+1, n):
                if Point.should_swap(points[i], points[j]):
                    points[i], points[j] = points[j], points[i]
        sorted_centers = torch.cat([torch.tensor([point.x, point.y, point.z]) for point in points])
        sorted_centers = sorted_centers / self.normalization_factor
        return voxel_grid, sorted_centers