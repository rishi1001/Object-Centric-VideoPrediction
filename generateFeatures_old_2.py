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

import sys


sys.setrecursionlimit(10**6)  # Set the recursion limit to a higher value



serial_num = 0
meta_info = []



def isObject(bbox):
    tot_rows = bbox[2] - bbox[0]
    tot_cols = bbox[3] - bbox[1]
    if tot_rows<10 or tot_cols<10:
        # print("Too small")
        return False
    return True

component = []

def dfs(matrix, visited, row, col):
    global component
    # Check if the current cell is within bounds and is 'True'
    if (
        row >= 0
        and row < len(matrix)
        and col >= 0
        and col < len(matrix[0])
        and matrix[row][col]==False
        and not visited[row][col]
    ):
        component.append((row, col))
        visited[row][col] = True
        size = 1  # Initialize the size of the connected component to 1

        # Recursively check neighboring cells
        size += dfs(matrix, visited, row - 1, col)  # Top
        size += dfs(matrix, visited, row + 1, col)  # Bottom
        size += dfs(matrix, visited, row, col - 1)  # Left
        size += dfs(matrix, visited, row, col + 1)  # Right

        return size
    else:
        return 0

def find_connected_components(matrix, threshold):
    rows, cols = len(matrix), len(matrix[0])
    visited = [[False] * cols for _ in range(rows)]
    components = []

    for i in range(rows):
        for j in range(cols):
            if matrix[i][j]==False and not visited[i][j]:
                global component
                component = []
                size = dfs(matrix, visited, i, j)
                if size > threshold:
                    components.append(component)

    return components

def getComponentInfo(component, size_x, size_y):
    min_row = min(component, key=lambda x: x[0])[0]
    max_row = max(component, key=lambda x: x[0])[0]
    min_col = min(component, key=lambda x: x[1])[1]
    max_col = max(component, key=lambda x: x[1])[1]
    bbox = [min_row, min_col, max_row, max_col]
    point_coords = [(min_row + max_row) // 2, (min_col + max_col) // 2]
    segment = np.zeros((size_x, size_y))
    for row, col in component:
        segment[row][col] = 1
    return bbox, point_coords, segment

def fixBackground(background):
    first_row = any(not cell for cell in background[0])
    last_row = any(not cell for cell in background[-1])
    first_col = any(not cell for cell in background[:, 0])
    last_col = any(not cell for cell in background[:, -1])
    if first_row:
        background[0]=[True]*len(background[0])
    if last_row:
        background[-1]=[True]*len(background[-1])
    if first_col:
        background[:, 0]=[True]*len(background[:, 0])
    if last_col:
        background[:, -1]=[True]*len(background[:, -1])
    return background


def getMask(mask_generator,img):        # tune threshold?
    masks = mask_generator.generate(img)
    max_area = 0
    background = None
    for mask in masks:
        if mask['area'] > max_area:
            max_area = mask['area']
            background = mask['segmentation']
    

    size_x, size_y = background.shape


    background = fixBackground(background)
    segmentations_ids = find_connected_components(background, 100)     # TODO tune this
    bbox = []
    point_coords = []
    segmentations = []
    for segments_ids in segmentations_ids:
        bbox_, point_coords_, segment = getComponentInfo(segments_ids, size_x, size_y)
        if not isObject(bbox_):
            # print("Not an object")
            continue
        bbox.append(bbox_)
        point_coords.append(point_coords_)
        segmentations.append(segment)
    return bbox, point_coords, segmentations

def getFeaturesResnet(img, segmentations, model, device, transform):
    features = []
    for mask in segmentations:
        mask = mask[:, :, np.newaxis].astype(np.uint8)
        mask_img = img * mask
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR)
        mask_img = Image.fromarray(mask_img)
        mask_img = transform(mask_img)
        mask_img = mask_img.unsqueeze(0)
        mask_img = mask_img.to(device)
        # get feature for the masked parts
        feature = model(mask_img).squeeze(0)
        # use mask to get feature
        feature = feature.detach().cpu().numpy()
        features.append(feature)
    return features

def matchPointCoords(prev_point_coords, prev_bbox, prev_features_resnet, point_coords, bbox, features_resnet):
    if prev_point_coords is None:
        return np.array(bbox), np.array(point_coords), np.array(features_resnet)
    
    bbox = np.array(bbox)
    prev_bbox = np.array(prev_bbox)
    point_coords = np.array(point_coords)
    prev_point_coords = np.array(prev_point_coords)
    features_resnet = np.array(features_resnet)

    # dists = np.zeros((len(bbox), len(prev_bbox)))
    # for i in range(len(bbox)):
    #     for j in range(len(prev_bbox)):
    #         dists[i][j] = np.linalg.norm(features_resnet[i]-prev_features_resnet[j])

    # match bbox to prev_bbox
    # return arranged point_coords, bbox

    # dists = np.zeros((len(bbox), len(prev_bbox)))
    # for i in range(len(bbox)):
    #     for j in range(len(prev_bbox)):
    #         dists[i][j] = np.linalg.norm(bbox[i]-prev_bbox[j])

    dists = np.zeros((len(bbox), len(prev_bbox)))
    for i in range(len(bbox)):
        for j in range(len(prev_bbox)):
            dists[i][j] = np.linalg.norm(prev_point_coords[i]-point_coords[j])

    # print(dists)
    # breakpoint()
    
    row_ind, col_ind = linear_sum_assignment(dists)
    point_coords = point_coords[col_ind]
    bbox = bbox[col_ind]
    features_resnet = features_resnet[col_ind]
    
    return bbox, point_coords, features_resnet

def get_numeric_value(file_name):
    return int(file_name.split('.')[0].split('frame')[1])

def generateFeatures(model,mask_generator, device, transform, data_path, save_path):
    global serial_num
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    feature = np.empty((0,0))
    prev_point_coords = None
    prev_bbox = None
    prev_features_resnet = None
    prev_num_objects = -1
    file_names = sorted(os.listdir(data_path), key=get_numeric_value)
    for frame_file in tqdm(file_names):
        print("Processing frame: ",data_path, frame_file)
        img = cv2.imread(os.path.join(data_path, frame_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bbox, point_coords, segmentations = getMask(mask_generator, img)
        features_resnet = getFeaturesResnet(img,segmentations,model,device,transform)
        num_objects = len(bbox)
        meta_info.append([frame_file, num_objects])
        img_shape = img.shape
        print("Number of objects: ", num_objects)
        print("Point coords: ", point_coords)
        if prev_num_objects != -1 and prev_num_objects != num_objects:
            # save data point
            print("Saving data point")
            np.save(os.path.join(save_path, str(serial_num)), feature)
            feature = np.empty((0,0))
            serial_num += 1
            prev_point_coords = None
            prev_bbox = None
            prev_features_resnet = None

        bbox, point_coords, features_resnet = matchPointCoords(prev_point_coords, prev_bbox, prev_features_resnet, point_coords, bbox, features_resnet)
        # breakpoint()
        positionFeature = point_coords.copy()
        # normalise positionFeature
        positionFeature[:,0] = positionFeature[:,0]/img_shape[0]
        positionFeature[:,1] = positionFeature[:,1]/img_shape[1]

        if feature.shape[0] == 0:
            feature = positionFeature
        else:
            feature = np.concatenate((feature, positionFeature), axis=1)
        prev_num_objects = num_objects
        prev_point_coords = point_coords
        prev_bbox = bbox
        prev_features_resnet = features_resnet
    
    # TODO check the data path where saving
    if feature.shape[0] != 0:
        print("Saving data point")
        np.save(os.path.join(save_path, str(serial_num)), feature)
        serial_num += 1
    
    if serial_num>10:
        exit(0)

if __name__ == '__main__':

    root='/DATATWO/users/mincut/Object-Centric-VideoAnswering/data'
    mode='train'


    folder = os.path.join(root,'extracted_frames',mode)

    save_folder = os.path.join(root,'features_trial',mode)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)


    # load model
    # model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
    # Load the pre-trained ResNet model
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    gpu=3
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
        save_path = os.path.join(save_folder)
        generateFeatures(model,mask_generator, device, transform, data_path, save_path)

    # save meta info in json
    with open(os.path.join(save_folder, 'meta_info.json'), 'w') as f:
        json.dump(meta_info, f)