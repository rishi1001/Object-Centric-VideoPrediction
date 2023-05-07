import networkx as nx
import scipy.sparse as sp
import numpy as np
import torch
import pandas as pd
from torch_geometric.utils import dense_to_sparse
# from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import os


def get_numeric_value(file_name):
    if file_name.endswith(".npy"):
        return int(file_name.split('.')[0])
    else:
        return 0

def process_data_point(data,num_timesteps_in,num_timesteps_out,num_features):
    # Calculate the number of data points
    num_data_points = data.shape[1]//num_features - num_timesteps_in - num_timesteps_out + 1

    # Initialize x and y lists
    x = []
    y = []

    # Generate x and y data for each data point
    for i in range(num_data_points):
        # Extract the input data for this data point
        x_data = data[:, num_features*i:num_features*(i+num_timesteps_in)]

        # Extract the output data for this data point
        y_data = data[:, num_features*(i+num_timesteps_in):num_features*(i+num_timesteps_in+num_timesteps_out)]

        # Append the input and output data to the x and y lists
        x.append(x_data)
        y.append(y_data)

    

    return x, y

class VideoDataset(Dataset):
    def __init__(self, folder,num_timesteps_in: int = 2, num_timesteps_out: int = 1,num_features: int=4):
        super().__init__()
        self.folder = folder

        self.num_timesteps_in=num_timesteps_in
        self.num_timesteps_out=num_timesteps_out
        self.node_features = num_features

        file_names = [file for file in sorted(os.listdir(folder), key=get_numeric_value) if file.endswith(".npy")]
        # file_names = sorted(os.listdir(folder), key=get_numeric_value)
        self.data_points_x = []
        self.data_points_y = []
        self.tot_points = 0
        for data_name in file_names:
            data = np.load(os.path.join(folder, data_name))
            # skip data points that are too short
            if data.shape[1] < self.node_features*(self.num_timesteps_in+self.num_timesteps_out):
                continue
            # breakpoint()
            x, y = process_data_point(data,self.num_timesteps_in,self.num_timesteps_out,self.node_features)
            self.tot_points += len(x)
            self.data_points_x.extend(x)
            self.data_points_y.extend(y)
            
            if self.tot_points>10000:
                break
        
        print("Total data points: ", self.tot_points)
    
    def __len__(self):
        return len(self.data_points_x)
    
    def __getitem__(self, idx):


        x = self.data_points_x[idx]
        y = self.data_points_y[idx]
        num_nodes = x.shape[0]


        # TODO uncomment this
        # if num_features*(self.num_timesteps_in+self.num_timesteps_out) > all_features.shape[1]:
        #     raise ValueError("num_timesteps_in + num_timesteps_out > num_timesteps_total")
        
        # reshape x -> (num_nodes, num_features, num_timesteps_in)
        x = np.reshape(x, (num_nodes, self.node_features, self.num_timesteps_in))
        # y = all_features[:,num_features*self.num_timesteps_in:num_features*(self.num_timesteps_in+self.num_timesteps_out)]

        # reshape y -> (num_nodes,num_features, num_timesteps_out)
        y = np.reshape(y, (num_nodes,self.node_features, self.num_timesteps_out))

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        # normalise x
        # x_max = torch.max(x, dim=0)[0]
        # x_min = torch.min(x, dim=0)[0]
        # # Normalize to [-1, 1]
        # x = num_features * (x - x_min) / (x_max - x_min+0.01) - 1

        # # normalise y
        # y_max = torch.max(y, dim=0)[0]
        # y_min = torch.min(y, dim=0)[0]
        # # Normalize to [-1, 1]
        # y = num_features * (y - y_min) / (y_max - y_min+0.01) - 1  

        # adj matrix
        adj = torch.ones((num_nodes, num_nodes))
        

        # edge_index
        edge_index = torch.nonzero(adj).t().contiguous()        # TODO: check if this is correct



        return Data(x=x, edge_index=edge_index, num_nodes=num_nodes,y=y)



