import torch
import os
from dataset import TimeSeries
from model import TemporalGNN
import pandas as pd
import numpy as np
from utils import *
from torch_geometric.loader import DataLoader
import sys
from tqdm import tqdm

def convert(l,mapping):
  m = {mapping[i]:i for i in range(len(mapping))}
  return [m[str(i)] for i in l]  

def get_train_node_ids(train_node_ids, batch_size):
    train_ids = []
    for i in range(batch_size):
        for x in train_node_ids:
            train_ids.append(x+i*dataset.num_nodes)
    # print(train_ids)
    return train_ids

## device setting
device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
print(device)
### model-parameters
hidden_layers=16
lr=0.005
weight_decay=5e-4
normalize=False     # just keep it False always

#dataset_X = "../a3_datasets/d2_small_X.csv"
#dataset_adj = "../a3_datasets/d2_adj_mx.csv"
#dataset_splits = "../a3_datasets/d2_graph_splits.npz"
#graph_name = "d2"

p = int(sys.argv[1])
f = int(sys.argv[2])
dataset_X = sys.argv[3]
dataset_adj = sys.argv[4]
dataset_splits = sys.argv[5]
graph_name = sys.argv[6]
num_epochs=int(sys.argv[7])
batch_size = 8
model_name = "A3TGCN"

model_path=f"./models/{model_name}/{num_epochs}/{graph_name}"
os.makedirs(model_path,exist_ok=True)
plot_path=f"./plot_losses/{model_name}/{num_epochs}/{graph_name}"
os.makedirs(plot_path,exist_ok=True)

dataset=TimeSeries(dataset_X,dataset_adj, num_timesteps_in=p, num_timesteps_out=f)
print("Total Nodes in Dataset: ",dataset.num_nodes)
dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=0)

splits = np.load(dataset_splits)
train_node_ids = convert(splits["train_node_ids"],dataset.mapping)
val_node_ids = convert(splits["val_node_ids"],dataset.mapping) 
test_node_ids = convert(splits["test_node_ids"],dataset.mapping)


model = TemporalGNN(node_features=1, p=p, f=f).to(device)     # to device remains
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
    for data in tqdm(dataloader):
        optimizer.zero_grad()  # Clear gradients.
        #print(data.features)
        out = model(data.x, data.edge_index,data.edge_weight)  
        # print(out)
        # print(out.shape)
        tt= get_train_node_ids(train_node_ids,data.y.shape[0]//dataset.num_nodes)
        loss = criterion(out[tt], data.y[tt])
        # print(loss)


        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        # if i==100:
        #     break
    print('epoch %d training loss: %.3f' % (epoch + 1, running_loss /(len(dataset)*len(train_node_ids)) ))
    if plot:
        train_loss.append(running_loss /(len(dataset)*len(train_node_ids)))
        # MAE, MAPE, RMSE, MAE2 = evaluate_metric(model, dataset,train_node_ids,diff=True)
        # train_mae.append(MAE)
    

def test(test=False,plot=False):         # test=True for test set
    model.eval()
    global best_loss
    global bestmodel
    running_loss = 0.0
    with torch.no_grad():
        out0 = []
        y0 = []
        for data in dataloader:
            out = model(data.x, data.edge_index, data.edge_weight)
            if (len(out0)==0):
                out0=out[val_node_ids[0]]
                y0=data.y[val_node_ids[0]]

            if test:
                tt= get_train_node_ids(test_node_ids,data.y.shape[0]//dataset.num_nodes)
                loss = criterion(out[tt], data.y[tt])
            else:
                tt= get_train_node_ids(val_node_ids,data.y.shape[0]//dataset.num_nodes)
                loss = criterion(out[tt], data.y[tt])
            running_loss += loss.item()
        ## TODO : waht factor to be multiplied
        if test:
            print('epoch %d Test loss: %.3f' % (epoch + 1, running_loss / (len(dataset)*len(test_node_ids))))
        else:
            print('epoch %d Test loss: %.3f' % (epoch + 1, running_loss / (len(dataset)*len(val_node_ids))))
        
        

        if (epoch%100==0 and not test and plot):
            x=[i for i in range(len(out0))]
            plt.plot(x,out0.cpu(),label="Model prediction")
            plt.plot(x,y0.cpu(),label="Actual")    
            plt.draw()
            plt.legend()
            plt.savefig(f"{plot_path}/{epoch}.png")
            plt.clf()
        if (not test):
            val_loss.append(running_loss /(len(dataset)*len(val_node_ids)))
            # MAE, MAPE, RMSE, MAE2 = evaluate_metric(model, dataset,val_node_ids,diff=True)
            # val_mae.append(MAE)
    
        
    
    if test==False and (best_loss==-1 or running_loss < best_loss):
        best_loss=running_loss
        # Saving our trained model
        torch.save(model.state_dict(), './cs1190382_task2.model')

if __name__ == '__main__':
    print('Start Training')
    os.makedirs('./models', exist_ok=True)

    # TODO we can use scalar to fit transform the data, also pass that in evaluate metric
    plot=False
    for epoch in range(num_epochs): 
        train(epoch,plot=plot)
        test(plot=plot)      # on validation set    

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
    if (False):
        epochs=[i for i in range(len(train_loss))]
        # print(epochs)
        plt.plot(epochs,train_loss,label="Train_loss")
        plt.plot(epochs,val_loss,label="Val_loss")    

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.draw()
        plt.savefig(f"{plot_path}/loss.png")
        plt.clf()    

        # plt.plot(epochs,train_mae,label="Train_mae")
        # plt.plot(epochs,val_mae,label="Val_mae")    

        # plt.xlabel("Epochs")
        # plt.ylabel("MAE")
        # plt.legend()
        # plt.draw()
        # plt.savefig(f"{plot_path}/MAE.png")
        # plt.clf()    
