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



serial_num = 0
info = {}


def get_numeric_value(file_name):
    return int(file_name.split('.')[0].split('frame')[1])

def generateFeatures(mask_generator, data_path):
    global serial_num, info
    file_names = sorted(os.listdir(data_path), key=get_numeric_value)
    for frame_file in tqdm(file_names):
        # print("Processing frame: ",data_path, frame_file)
        # data_path = '/DATATWO/users/mincut/Object-Centric-VideoAnswering/data/extracted_frames/train/00000'
        # frame_file = 'frame20.jpg'
        image_name = data_path.split('/')[-1] + '_' + frame_file
        img = cv2.imread(os.path.join(data_path, frame_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        masks = mask_generator.generate(img)
        count = 0
        for i, mask in enumerate(masks):
            if i==0:
                continue
            if isObject(mask):
                count += 1
                # print(i)
            else:
                break

        print(image_name, count)
        # exit(0)

        info[image_name] = count

        

if __name__ == '__main__':

    root='/DATATWO/users/mincut/Object-Centric-VideoAnswering/data'
    mode='train'


    folder = os.path.join(root,'extracted_frames',mode)


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
        # print("Processing video: ", videos)
        data_path = os.path.join(folder, videos)
        generateFeatures(mask_generator, data_path)

    # save info
    with open('info.json', 'w') as fp:
        json.dump(info, fp)
