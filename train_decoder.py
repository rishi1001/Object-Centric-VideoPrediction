import torch 
import torch.nn as nn 
from sg2im.model_modified import Sg2ImModel


def get_dataloader(batch_size = 32,mode='train'):
    fts_to_frames = None 
    resnet_ft = f"/DATATWO/users/mincut/Object-Centric-VideoAnswering/data/features_resnet/{mode}"
    bbox_ft = f"/DATATWO/users/mincut/Object-Centric-VideoAnswering/data/features/{mode}"

    
    return 
def train_decoder(data):
    


    for batch in data:
        obj_ft = batch['features']
        obj_pos=  batch['pos']
        obj_to_img = batch['obj_to_img']
    pass 



def main():
    model = Sg2ImModel()

    
