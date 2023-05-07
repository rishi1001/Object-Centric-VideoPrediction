import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import os
import numpy as np


class dataset(Dataset):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder
        self.videos = sorted(os.listdir(self.folder))
    
    def __len__(self):
        return len(self.videos)
    
    def get_y(self,idx):        # TODO change this? should it also out num_nodes?
        video_folder = os.path.join(self.folder, self.videos[idx])
        frames = sorted(os.listdir(video_folder))
        # select last 5 frames for output
        frames = frames[-5:]
        y = []
        nodes_per_frame = []
        node_count = 0
        for frame in frames:
            frame_feature = np.load(os.path.join(video_folder,frame))
            y.append(frame_feature)
            nodes_per_frame.append(node_count)
            node_count += frame_feature.shape[0]

        y = np.concatenate(y, axis=0)
        y = torch.from_numpy(y).float()
        nodes_per_frame = torch.from_numpy(np.array(nodes_per_frame)).long()
        y = [y, nodes_per_frame]
        return y
    
    def __getitem__(self, idx):
        video_folder = os.path.join(self.folder, self.videos[idx])
        frames = sorted(os.listdir(video_folder))
        # select 5 frames for input(then we will generate remaining frames?)
        frames = frames[::5]            # TODO change this

        feature_video = []
        nodes_per_frame = []
        node_count = 0
        for frame in frames:
            frame_feature = np.load(os.path.join(video_folder,frame))
            # normalise frame_feature
            frame_feature_max = np.max(frame_feature, axis=0)
            frame_feature_min = np.min(frame_feature, axis=0)
            # Normalize to [-1, 1]
            frame_feature = 2 * (frame_feature - frame_feature_min) / (frame_feature_max - frame_feature_min+0.01) - 1

            feature_video.append(frame_feature)
            nodes_per_frame.append(node_count)
            node_count += frame_feature.shape[0]

        feature_video = np.concatenate(feature_video, axis=0)   
        # convert to torch
        feature_video = torch.from_numpy(feature_video).float()
        nodes_per_frame = torch.from_numpy(np.array(nodes_per_frame)).long()  

        # adj matrix
        adj = torch.zeros((feature_video.shape[0], feature_video.shape[0]))
        for i in range(len(nodes_per_frame)-1):
            adj[nodes_per_frame[i]:nodes_per_frame[i+1], nodes_per_frame[i]:nodes_per_frame[i+1]] = 1
        adj[nodes_per_frame[-1]:, nodes_per_frame[-1]:] = 1
        # TODO -> maybe remove diagonal zeros

        # edge_index
        edge_index = torch.nonzero(adj).t().contiguous()        # TODO: check if this is correct

        # y
        y = self.get_y(idx)

        return Data(x=feature_video, edge_index=edge_index, num_nodes=feature_video.shape[0], nodes_per_frame=nodes_per_frame,y=y)
    