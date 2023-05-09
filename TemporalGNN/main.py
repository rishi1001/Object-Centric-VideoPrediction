import torch
import os
from dataset import VideoDataset
from model import TemporalGNN
import pandas as pd
import numpy as np
from utils import *
from torch_geometric.loader import DataLoader
import sys
from tqdm import tqdm
import wandb
import random

def set_seed(seed: int = 1) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print("Random seed set as {}".format(seed))

set_seed()

wandb_save=True
## device setting
gpu=2
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
print(device)
### model-parameters
hidden_layers=16
lr=5e-5        # TODO varying this
weight_decay=5e-6
normalize=False     # just keep it False always

#dataset_X = "../a3_datasets/d2_small_X.csv"
#dataset_adj = "../a3_datasets/d2_adj_mx.csv"
#dataset_splits = "../a3_datasets/d2_graph_splits.npz"
#graph_name = "d2"

num_timestamps_in = 6
num_timestamps_out = 2
num_features=4


num_epochs=30
batch_size = 1
model_name = "A3TGCN"

folder_name = f"{model_name}_{num_timestamps_in}_{num_timestamps_out}"
model_save = folder_name

if wandb_save:
    wandb.init(project="vidgnn", name=folder_name,config={"lr":lr,"weight_decay":weight_decay,"hidden_layers":hidden_layers,"normalize":normalize,"num_timestamps_in":num_timestamps_in,"num_timestamps_out":num_timestamps_out,"batch_size":batch_size,"num_epochs":num_epochs},entity='rishishah')

folder_name = "Results/" + folder_name

root='/DATATWO/users/mincut/Object-Centric-VideoAnswering/data'
train_folder = os.path.join(root,'features','train')
val_folder = os.path.join(root,'features','validation')
test_folder = os.path.join(root,'features','test')

# print("Total Nodes in Dataset: ",dataset.num_nodes)
# dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=0)

dataset_train = VideoDataset(train_folder, num_timesteps_in=num_timestamps_in, num_timesteps_out=num_timestamps_out, num_features=num_features)
dataset_val = VideoDataset(val_folder, num_timesteps_in=num_timestamps_in, num_timesteps_out=num_timestamps_out,num_features=num_features)
dataset_test = VideoDataset(test_folder, num_timesteps_in=num_timestamps_in, num_timesteps_out=num_timestamps_out,num_features=num_features)


dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)


model = TemporalGNN(node_features=num_features, num_timestamps_in=num_timestamps_in, num_timestamps_out=num_timestamps_out).to(device)     # to device remains
model=model.double()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
criterion = torch.nn.MSELoss(reduction='sum')

best_loss = -1

train_loss=[]
val_loss=[]
train_mae=[]
val_mae=[]

def train(epoch,plot=False):
    model.train()

    running_loss = 0.0
    # batch wise training
    for data in tqdm(dataloader_train):
        optimizer.zero_grad()  # Clear gradients.
        #print(data.features)
        # breakpoint()
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        data.y = data.y.double().to(device)

        out = model(data.x, data.edge_index)  
        # print(out)
        # print(out.shape)
        loss = criterion(out, data.y)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # if i==100:
        #     break
    if wandb_save:
        wandb.log({"train_loss":running_loss /(len(dataloader_train))})
    print('epoch %d training loss: %.3f' % (epoch + 1, running_loss /(len(dataloader_train))))
    if plot:
        train_loss.append(running_loss /(len(dataloader_train)))
        # MAE, MAPE, RMSE, MAE2 = evaluate_metric(model, dataset,train_node_ids,diff=True)
        # train_mae.append(MAE)
    

def test(test=False,plot=False,dataloader=None):         # test=True for test set
    model.eval()
    global best_loss
    global bestmodel
    running_loss = 0.0
    with torch.no_grad():
        out0 = []
        y0 = []
        for data in dataloader:

            data.x = data.x.to(device)
            data.edge_index = data.edge_index.to(device)
            data.y = data.y.double().to(device)

            out = model(data.x, data.edge_index)
            if (len(out0)==0):
                out0=out
                y0=data.y

            loss = criterion(out, data.y)
            
            running_loss += loss.item()

        print('epoch %d Test/Val loss: %.3f' % (epoch + 1, running_loss / (len(dataloader))))
        if wandb_save:
            wandb.log({"val_loss":running_loss /(len(dataloader))})

        if ((epoch%10==0 or epoch==29) and not test and plot):
            x=[i for i in range(len(out0))]
            plt.plot(x,out0[:,0,0].cpu(),label="Pred-x")
            plt.plot(x,y0[:,0,0].cpu(),label="Actual-x") 
            plt.legend()
            plt.title("Objects vs X-Coordinate")
            plt.xlabel("Object ID")
            plt.ylabel("X-Coordinate")
            plt.savefig(f"{folder_name}/{epoch}_x.png")
            plt.clf()
            plt.plot(x,out0[:,1,0].cpu(),label="Pred-y")
            plt.plot(x,y0[:,1,0].cpu(),label="Actual-y")    
            plt.draw()
            plt.legend()
            plt.legend()
            plt.title("Objects vs Y-Coordinate")
            plt.xlabel("Object ID")
            plt.ylabel("Y-Coordinate")
            plt.savefig(f"{folder_name}/{epoch}_y.png")
            plt.clf()
        if (not test):
            val_loss.append(running_loss /(len(dataloader)))
            # MAE, MAPE, RMSE, MAE2 = evaluate_metric(model, dataset,val_node_ids,diff=True)
            # val_mae.append(MAE)
    
        
    
    if test==False and (best_loss==-1 or running_loss < best_loss):
        best_loss=running_loss
        # Saving our trained model
        torch.save(model.state_dict(), f'models/{model_save}.pt')     # TODO change this

if __name__ == '__main__':
    print('Start Training')
    os.makedirs('./models', exist_ok=True)
    os.makedirs(f'./{folder_name}', exist_ok=True)


    # TODO we can use scalar to fit transform the data, also pass that in evaluate metric
    plot=True
    for epoch in range(num_epochs): 
        train(epoch,plot=plot)
        test(plot=plot,dataloader=dataloader_val)      # on validation set    

    print('Finished Training')

    # print("For Training:  ")
    # MAE, MAPE, RMSE, MAE2 = evaluate_metric(model, dataset,train_node_ids,diff=True)
    # print("MAE: ", MAE, MAE2)

    # model.load_state_dict(torch.load('./cs1190382_task2.model'))
    # print("For Validation:  ")
    # MAE, MAPE, RMSE, MAE2 = evaluate_metric(model, dataset,val_node_ids,diff=True)
    # print("MAE: ", MAE, MAE2)
    
    
    # print("For Testing:  ")
    # MAE, MAPE, RMSE, MAE2 = evaluate_metric(model, dataset,test_node_ids,diff=True)
    # print("MAE: ", MAE, MAE2)


    ## ploting
    if (plot):
        epochs=[i for i in range(len(train_loss))]
        # print(epochs)
        plt.plot(epochs,train_loss,label="Train_loss")
        plt.plot(epochs,val_loss,label="Val_loss")    

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.draw()
        plt.savefig(f"{folder_name}/loss.png")
        plt.clf()    

        # plt.plot(epochs,train_mae,label="Train_mae")
        # plt.plot(epochs,val_mae,label="Val_mae")    

        # plt.xlabel("Epochs")
        # plt.ylabel("MAE")
        # plt.legend()
        # plt.draw()
        # plt.savefig(f"{plot_path}/MAE.png")
        # plt.clf()    

    test(test=True,plot=plot,dataloader=dataloader_test)      # on test set
