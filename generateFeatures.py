import torch
import cv2
import os
import numpy as np
import torchvision
import torchvision.transforms as transforms
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from PIL import Image
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
import json


serial_num = 0
meta_info = []
mapping_srl_actual = {}

def is_connected(matrix):
    visited = set()
    stack = []
    # Find first "True" value
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            if matrix[row][col] == True:
                stack.append((row, col))
                break
        if stack:
            break

    while stack:
        row, col = stack.pop()
        if (row, col) not in visited and matrix[row][col] == True:
            visited.add((row, col))
            # Check all neighboring cells
            if row > 0:
                stack.append((row - 1, col))  # Check top
            if row < len(matrix) - 1:
                stack.append((row + 1, col))  # Check bottom
            if col > 0:
                stack.append((row, col - 1))  # Check left
            if col < len(matrix[0]) - 1:
                stack.append((row, col + 1))  # Check right
    # If all "True" values have been visited, they are connected
    return len(visited) == np.count_nonzero(matrix)



def isObject(mask):

    # check if bbox is too small
    # check if bbox is too big
    segment = mask['segmentation']
    area = mask['area']
    predicted_iou = mask['predicted_iou']

    # check if segmention is connected
    if not is_connected(segment):
        # print("Not connected")
        return False

    # bound on area
    # if area < 1000 or area > 10000:     # TODO tune this
    #     return False
    
    if predicted_iou < 0.992:     # TODO tune this
        return False

    return True


def getMask(mask_generator,img):        # tune threshold?
    masks = mask_generator.generate(img)
    bbox = []
    point_coords = []
    segmentations = []
    for i, mask in enumerate(masks):
        if i==0:
            continue
        if not isObject(mask):
            # print("Not an object")
            break
        # print("Object found")
        bbox.append(mask['bbox'])
        point_coords.append(mask['point_coords'][0])      # TODO check this(x,y are given reverse in mask['point_coords'])
        segmentations.append(mask['segmentation'])
    
    return bbox, point_coords, segmentations

def getFeaturesResnet(img, segmentations, model, device, transform):
    # transform2 = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    # ])
    # img = Image.fromarray(img)
    # img = transform(img)
    features = []
    for i,mask in enumerate(segmentations):
        mask = mask[:, :, np.newaxis].astype(np.uint8)
        # mask = np.repeat(mask, 3, axis=2)
        # mask = Image.fromarray(mask)
        # mask = transform2(mask)
        mask_img = img * mask
        # breakpoint()
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(f"check_{i}.png", mask_img.numpy())
        mask_img = Image.fromarray(mask_img)
        mask_img = transform(mask_img)
        mask_img = mask_img.unsqueeze(0)
        mask_img = mask_img.to(device)
        # get feature for the masked parts
        feature = model(mask_img).squeeze(0)
        # use mask to get feature
        feature = feature.detach().cpu().numpy()
        features.append(feature)
    # breakpoint()
    return features

def matchPointCoords(prev_point_coords, prev_bbox, prev_features_resnet, point_coords, bbox, features_resnet):
    if prev_point_coords is None:
        return np.array(bbox), np.array(point_coords), np.array(features_resnet)
    bbox = np.array(bbox)
    prev_bbox = np.array(prev_bbox)
    point_coords = np.array(point_coords)
    prev_point_coords = np.array(prev_point_coords)
    features_resnet = np.array(features_resnet)

    dists = np.zeros((len(bbox), len(prev_bbox)))
    for i in range(len(bbox)):
        for j in range(len(prev_bbox)):
            dists[i][j] = np.linalg.norm(features_resnet[i]-prev_features_resnet[j])
    

    # match bbox to prev_bbox
    # return arranged point_coords, bbox

    # dists2 = np.zeros((len(bbox), len(prev_bbox)))
    # for i in range(len(bbox)):
    #     for j in range(len(prev_bbox)):
    #         dists2[i][j] = np.linalg.norm(bbox[i]-prev_bbox[j])

    # dists = np.zeros((len(bbox), len(prev_bbox)))
    # for i in range(len(bbox)):
    #     for j in range(len(prev_bbox)):
    #         dists[i][j] = np.linalg.norm(point_coords[i]-prev_point_coords[j])

    # print(dists)
    
    # row_ind1, col_ind1 = linear_sum_assignment(dists1)
    # row_ind2, col_ind2 = linear_sum_assignment(dists2)
    row_ind, col_ind = linear_sum_assignment(dists)

    point_coords = point_coords[col_ind]
    bbox = bbox[col_ind]
    features_resnet = features_resnet[col_ind]
    
    return bbox, point_coords, features_resnet

def get_numeric_value(file_name):
    return int(file_name.split('.')[0].split('frame')[1])

def generateFeatures(model,mask_generator, device, transform, data_path, save_path, save_path_resnet):
    global serial_num
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path_resnet):
        os.mkdir(save_path_resnet)
    feature = np.empty((0,0))
    resnet_feature = np.empty((0,0))
    prev_point_coords = None
    prev_bbox = None
    prev_features_resnet = None
    prev_num_objects = -1
    file_names = sorted(os.listdir(data_path), key=get_numeric_value)
    for frame_file in tqdm(file_names):
        # data_path = '/DATATWO/users/mincut/Object-Centric-VideoAnswering/data/extracted_frames/train/00000'
        # frame_file = 'frame36.jpg'
        print("Processing frame: ",data_path, frame_file)
        mapping_srl_actual[f"{data_path.split('/')[-1]}_{frame_file}"]=serial_num
        img = cv2.imread(os.path.join(data_path, frame_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox, point_coords, segmentations = getMask(mask_generator, img)
        features_resnet = getFeaturesResnet(img,segmentations,model,device,transform)
        num_objects = len(bbox)
        meta_info.append([frame_file, num_objects])
        img_shape = img.shape
        # breakpoint()
        print("Number of objects: ", num_objects)
        print("Point coords: ", point_coords)
        if (prev_num_objects != -1 and prev_num_objects != num_objects) or num_objects==0:
            # save data point
            print("Saving data point")
            np.save(os.path.join(save_path, str(serial_num)), feature)
            np.save(os.path.join(save_path_resnet, str(serial_num)), resnet_feature)
            feature = np.empty((0,0))
            resnet_feature = np.empty((0,0))
            serial_num += 1
            prev_point_coords = None
            prev_bbox = None
            prev_features_resnet = None
        
        if num_objects==0:
            continue

        bbox, point_coords, features_resnet = matchPointCoords(prev_point_coords, prev_bbox, prev_features_resnet, point_coords, bbox, features_resnet)
        # breakpoint()
        positionFeature = bbox.copy().astype(float)

        # normalise positionFeature
        positionFeature[:,0] = positionFeature[:,0]/float(img_shape[1])
        positionFeature[:,1] = positionFeature[:,1]/float(img_shape[0])
        positionFeature[:,2] = positionFeature[:,2]/float(img_shape[1])
        positionFeature[:,3] = positionFeature[:,3]/float(img_shape[0])

        positionFeature = positionFeature[:,[1,0,3,2]]      # to make sure x is first?

        if feature.shape[0] == 0:
            feature = positionFeature
            resnet_feature = features_resnet.copy()
        else:
            feature = np.concatenate((feature, positionFeature), axis=1)
            resnet_feature = np.concatenate((resnet_feature, features_resnet.copy()), axis=1)

        prev_num_objects = num_objects
        prev_point_coords = point_coords
        prev_bbox = bbox
        prev_features_resnet = features_resnet
    
    # TODO check the data path where saving
    if feature.shape[0] != 0:
        print("Saving data point")
        np.save(os.path.join(save_path, str(serial_num)), feature)
        np.save(os.path.join(save_path_resnet, str(serial_num)), resnet_feature)
        serial_num += 1
    
        
if __name__ == '__main__':

    root='/DATATWO/users/mincut/Object-Centric-VideoAnswering/data'
    mode='train'


    folder = os.path.join(root,'extracted_frames',mode)

    save_folder = os.path.join(root,'features_dummy',mode)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_folder_resnet = os.path.join(root,'features_resnet_dummy',mode)
    if not os.path.exists(save_folder_resnet):
        os.makedirs(save_folder_resnet)


    # load model
    # model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
    # Load the pre-trained ResNet model
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    gpu=0
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # # load transform
    # transform = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl_transforms')
    # Define the pre-processing transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load sam
    ckpt='/DATATWO/users/mincut/segment-anything/sam_vit_h_4b8939.pth'
    model_type = "vit_h"
    sam = sam_model_registry["default"](checkpoint=ckpt)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    # iterate over all videos in "folder"
    for videos in tqdm(sorted(os.listdir(folder))):
        print("Processing video: ", videos)
        data_path = os.path.join(folder, videos)
        generateFeatures(model,mask_generator, device, transform, data_path, save_folder, save_folder_resnet)
        if serial_num>100:
            break

    # save meta info in json
    with open(os.path.join(save_folder, 'meta_info.json'), 'w') as f:
        json.dump(meta_info, f)
    
    with open(os.path.join(save_folder, 'mapping_dummy.json'), 'w') as f:
        json.dump(mapping_srl_actual, f)

    