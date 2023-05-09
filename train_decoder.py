import torch 
import torch.nn as nn 
from sg2im.model_modified import Sg2ImModel
from torch.utils.data import Dataset , DataLoader 
import json
from PIL import Image
from torchvision import transforms 
import os 
import numpy as np 
import wandb 
from tqdm import tqdm 
import torch.nn.functional as F 
ROOT="/DATATWO/users/mincut/Object-Centric-VideoAnswering/data"
class DecodingDataset(Dataset):
    def __init__(self,mode):
        self.dtst_fldr = f"{ROOT}/extracted_frames/{mode}"
        self.fts_to_frames = json.load(open(f"{ROOT}/features_dummy/{mode}/mapping_dummy.json")) 
        self.resnet_ft = f"{ROOT}/features_resnet_dummy/{mode}"
        self.bbox_ft = f"{ROOT}/features_dummy/{mode}"
        self.mapping = []
        self.img_transform =transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
        
        num_frames = dict()
        for k,v in self.fts_to_frames.items():
            num_frames[v] = num_frames.get(v,-1)+1
            vid,img = k.split('_')
            filename = os.path.join(self.dtst_fldr,vid,img)
            
            bbox_ft = os.path.join(f"{self.bbox_ft}/{v}.npy")
            resnet_ft = os.path.join(f"{self.resnet_ft}/{v}.npy")
            assert os.path.isfile(bbox_ft)
            assert os.path.isfile(resnet_ft)

            if np.load(bbox_ft).shape[1] == 0:
                continue 
            self.mapping.append([filename,bbox_ft,resnet_ft,num_frames[v]])
        
        
    def __len__(self):
        return len(self.mapping) 
    
    def __getitem__(self,idx):
        img,bbox,resnet,i = self.mapping[idx]
        bbox,resnet = np.load(bbox),np.load(resnet)
        i = min(bbox.shape[1]//4 - 1,i)  ##change after fixing the mapping mess 
        # assert 4*(i+1) < bbox.shape[1]
        # assert 1000*(i+1) < resnet.shape[1]
        bbox = torch.tensor(bbox)[:,4*i:4*(i+1)]
        resnet = torch.tensor(resnet)[:,1000*i:1000*(i+1)]
        img = Image.open(img)
        img = self.img_transform(img)
        ft = torch.cat([resnet,bbox],dim=1)
        return ft.type(torch.FloatTensor),img.type(torch.FloatTensor) 
    

def get_dataloader(batch_size = 16,mode='train'):

    def collate_fn(datapoints):
        obj_to_img = []
        obj_ft = []
        imgs =[]

        for i,(ft,img) in enumerate(datapoints):
            obj_ft.append(ft)
            obj_to_img.extend([i,]*ft.size(0))
            imgs.append(img)
        obj_ft = torch.cat(obj_ft,dim=0)
        obj_to_img = torch.tensor(obj_to_img)
        imgs = torch.stack(imgs,dim=0)
        return  {
            "obj_ft" : obj_ft, 
            "obj_to_img" : obj_to_img,
            "imgs" : imgs 
        }
    
    dataset = DecodingDataset(mode)
    return DataLoader(dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)



def train_decoder(model,dataloader):
    
    from torch.optim import Adam 
    optimizer = Adam(model.parameters(),lr=1e-5)
    num_epochs = 50
    for j in range(num_epochs):
        for i,batch in tqdm(enumerate(iter(dataloader))):

            if i%5 == 0:
                torch.save(model,f"decoder_models/model_{j}.pth")
            obj_ft = batch["obj_ft"]
            obj_to_img = batch["obj_to_img"]
            img_pred,_ = model(obj_ft,obj_to_img)
            img  = batch["imgs"]
            if img_pred.isnan().any():
                print("#####NAN OUTPUT FOUND AND IGNORED!!")
                continue
            l1_pixel_loss = F.l1_loss(img_pred, img)
            wandb.log({'train_loss':l1_pixel_loss})
            # mask_loss = F.binary_cross_entropy(masks_pred, masks.float()) get image masks for this 
            print("TRAIN_loss",l1_pixel_loss)
            optimizer.zero_grad()
            l1_pixel_loss.backward()
            optimizer.step()



def main():
    model = Sg2ImModel()
    dataloader = get_dataloader()
    wandb.init(project="vidgnn",entity='rishishah')
    train_decoder(model,dataloader)

    

if __name__ == '__main__':
    main()
