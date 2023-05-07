import networkx as nx
import scipy.sparse as sp
import numpy as np
import torch
import pandas as pd
from torch_geometric.utils import dense_to_sparse
# from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import Dataset


device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
class TimeSeries(Dataset):
    def __init__(self,csv_file,graph_file,num_timesteps_in: int = 12, num_timesteps_out: int = 12) -> None:
        super().__init__()
        self.num_timesteps_in=num_timesteps_in
        self.num_timesteps_out=num_timesteps_out

        
        df = pd.read_csv(graph_file,index_col=0)
        cols=df.columns
        df.columns=[i for i in range(len(cols))]
        df=df.reset_index(drop=True)
        
        t=torch.tensor(df.values)
        self.edge_index , self.edge_weight = dense_to_sparse(t)
        self.edge_weight=self.edge_weight
        self.num_nodes = len(cols)
        self.edge_index=self.edge_index.to(device)
        self.edge_weight=self.edge_weight.to(device)

        #self.mapping ={i:cols[i] for i in range(len(cols))}                    
        self.mapping=cols
        
        df = pd.read_csv(csv_file)
        df=df.drop(['Unnamed: 0'], axis=1)
        # TODO normalise featuers
        self.dataset=torch.tensor(df.values).type(torch.DoubleTensor)
        
    def __len__(self):
        return len(self.dataset) - (self.num_timesteps_in+self.num_timesteps_out)+1
    
    def __getitem__(self, idx):
        # return {'x':torch.tensor(self.dataset[idx][0]).type(torch.DoubleTensor),'y': torch.tensor(self.dataset[idx][1]).type(torch.DoubleTensor),'edge_weight':self.edge_weight,'edge_index':self.edge_index} 
        # x=self.dataset[idx:idx+self.num_timesteps_in]
        # x=torch.reshape(x,(x.shape[0],x.shape[1],1))
        # x=torch.permute(x,(1,2,0))
        # assert(x.shape[1]==1 and x.shape[2]==12)
        # y=self.dataset[idx+self.num_timesteps_in:idx+self.num_timesteps_in+self.num_timesteps_out]
        # y=torch.permute(y,(1,0))
        # assert(y.shape[1]==12)
        # d= Data(x=x,edge_index=self.edge_index,edge_attr=self.edge_weight,y=y)
        # return d.to(device)

        ## Code for diff
        x=self.dataset[idx:idx+self.num_timesteps_in]
        y=self.dataset[idx+self.num_timesteps_in:idx+self.num_timesteps_in+self.num_timesteps_out]
        ## CODe
        y=y-x[self.num_timesteps_in-1]
        x=torch.reshape(x,(x.shape[0],x.shape[1],1))
        x=torch.permute(x,(1,2,0))
        assert(x.shape[1]==1 and x.shape[2]==12)
        y=torch.permute(y,(1,0))
        assert(y.shape[1]==12)
        d= Data(x=x,edge_index=self.edge_index,edge_attr=self.edge_weight,y=y)
        return d.to(device)



