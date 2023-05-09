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
import torchvision.transforms.functional as F_vision
import torchvision
import cv2

gpu=0
device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

ROOT="/DATATWO/users/mincut/Object-Centric-VideoAnswering/data"
class DecodingDataset(Dataset):
    def __init__(self,mode):
        self.dtst_fldr = f"{ROOT}/extracted_frames/{mode}"
        self.fts_to_frames = json.load(open(f"{ROOT}/features_trial/{mode}/mapping.json")) 
        self.resnet_ft = f"{ROOT}/features_resnet_trial/{mode}"
        self.mask_ft = f"{ROOT}/features_mask_trial/{mode}"
        self.bbox_ft = f"{ROOT}/features_trial/{mode}"
        self.mapping = []
        self.img_transform =transforms.Compose([
        transforms.Resize((256, 256)),
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
            mask_ft = os.path.join(f"{self.mask_ft}/{v}.npy")
            assert os.path.isfile(bbox_ft)
            assert os.path.isfile(resnet_ft)
            assert os.path.isfile(mask_ft)

            if np.load(bbox_ft).shape[1] == 0:
                continue 
            self.mapping.append([filename,bbox_ft,resnet_ft,mask_ft,num_frames[v]])
        
        ###resize the mask 
    def __len__(self):
        return len(self.mapping) 
    
    def __getitem__(self,idx):
        img,bbox,resnet,mask,i = self.mapping[idx]
        bbox,resnet,mask = np.load(bbox),np.load(resnet),np.load(mask)
        # i = min(bbox.shape[1]//4 - 1,i)  ##change after fixing the mapping mess 
        # assert 4*(i+1) < bbox.shape[1]
        # assert 1000*(i+1) < resnet.shape[1]
        bbox = torch.tensor(bbox)[:,4*i:4*(i+1)]
        resnet = torch.tensor(resnet)[:,1000*i:1000*(i+1)]
        mask = torch.tensor(mask)[:,320*i:320*(i+1)]
        img = Image.open(img)
        img = self.img_transform(img)
        ft = torch.cat([resnet,bbox],dim=1)
        return ft.type(torch.FloatTensor),img.type(torch.FloatTensor),mask.type(torch.FloatTensor)
    

def get_dataloader(batch_size = 16,mode='train'):

    def collate_fn(datapoints):
        obj_to_img = []
        obj_ft = []
        imgs =[]
        masks =[]
        for i,(ft,img,mask) in enumerate(datapoints):
            obj_ft.append(ft)
            obj_to_img.extend([i,]*ft.size(0))
            imgs.append(img)
            masks.append(mask)
        obj_ft = torch.cat(obj_ft,dim=0)
        obj_to_img = torch.tensor(obj_to_img)
        imgs = torch.stack(imgs,dim=0)
        masks = torch.cat(masks,dim=0)
        return  {
            "obj_ft" : obj_ft, 
            "obj_to_img" : obj_to_img,
            "imgs" : imgs ,
            "masks" : masks
        }
    
    dataset = DecodingDataset(mode)
    return DataLoader(dataset,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)

counter = 0
def write_img(predicted_image_tensor,original_img_tensor,batch_size=16):
    global counter
    folder_name = f"Results/exp{2}"
    os.makedirs(folder_name,exist_ok=True)
    # reverse_normalize = transforms.Normalize(
    #         mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    #         std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    #     )
    reverse_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.ToPILImage(),
        transforms.Resize((320, 480))
    ])

    predicted_image = predicted_image_tensor.cpu().detach()  # Move the tensor to CPU if it's on GPU
    actual_image = original_img_tensor.cpu().detach()
    for i in range(batch_size):
        pred_img_i = predicted_image[i]
        # pred_img_i = transforms.ToPILImage()(pred_img_i)
        pred_img_i = reverse_normalize(pred_img_i)
        pred_img_i = np.array(pred_img_i)
        pred_img_i = cv2.cvtColor(pred_img_i, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{folder_name}/predicted_img_{counter}.jpg", pred_img_i)

        actual_img_i = actual_image[i]
        # actual_img_i = transforms.ToPILImage()(actual_img_i)
        actual_img_i = reverse_normalize(actual_img_i)
        actual_img_i = np.array(actual_img_i)
        actual_img_i = cv2.cvtColor(actual_img_i, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{folder_name}/actual_img_{counter}.jpg", actual_img_i)

        counter +=1 

def inference_decoder(model,dataloader):
    global counter
    model = torch.load(f"decoder_models_new/model_{48}.pth")
    with torch.no_grad():
        model.eval()
        for i,batch in tqdm(enumerate(iter(dataloader))):
            obj_ft = batch["obj_ft"].to(device)
            obj_to_img = batch["obj_to_img"].to(device)
            img_pred,masks_pred,boxes_pred = model(obj_ft,obj_to_img)
            #@rishi: can take binary cross entropy with mask of the image anfd mse with predicted bounding boxes also. 
            #check scene-graph-decoder/sg2im/scripts/train.py loss functions for exact way. 
            #: Check for mask size that are output 
            img  = batch["imgs"].to(device)
            write_img(img_pred,img)
            masks_actual = batch["masks"].to(device)
            if img_pred.isnan().any():
                print("#####NAN OUTPUT FOUND AND IGNORED!!")
                continue
            l1_pixel_loss = F.l1_loss(img_pred, img)

            masks_actual = F_vision.resize(masks_actual, (256, 256), interpolation=torchvision.transforms.InterpolationMode.NEAREST)
            mask_loss = F.binary_cross_entropy(masks_pred, masks_actual.float())

            total_loss = l1_pixel_loss + 0.5*mask_loss
            # mask_loss = F.binary_cross_entropy(masks_pred, masks.float()) get image masks for this 
            print("TRAIN_loss",total_loss)
            if counter>10:
                break



def main():
    model = Sg2ImModel().to(device)
    dataloader = get_dataloader()
    # wandb.init(project="vidgnn",name="inference_decoder",entity='rishishah')
    inference_decoder(model,dataloader)


if __name__ == '__main__':
    main()
