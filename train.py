import os
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# import your class
from MOT16 import MOT16, detection_collate   # or: from your_file import MOT16 as MOT16Dataset

class Siamese_Network(nn.Module):
    def __init__(self):
        super(Siamese_Network, self).__init__()
        # CNN layers for feature extraction
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 22 * 22, 256)
        self.fc2 = nn.Linear(256, 256)
        
    def forward_one(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 22 * 22)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def forward(self, input1, input2):
        output1 = self.forward_one(input1)
        output2 = self.forward_one(input2)
        return output1, output2
    


def train_fasterCNN():

    # transforms
    tf = transforms.ToTensor()

    # get_ground_truth=True for train, False for test
    # get_gt = "train" in os.path.abspath(args.root)
    # ds = MOT16(root=args.root, transform=tf, getGroundTruth=get_gt)

    trainMOT16data = MOT16(root = 'MOT16/train', transform=tf, getGroundTruth=True) 
    # testMOT16data = MOT16(root = 'MOT16/test', transform=tf, getGroundTruth=False) 

    print(f"[INFO] train dataset size: {len( trainMOT16data)} frames | split has GT: {get_gt}")

    # do augmentation here
    #color_aug = transforms.Compose([
    #transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    #])
    # augmented_image = color_aug(original_image)

    # pretrained model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Freeze backbone layers
    for param in model.backbone.parameters():
        param.requires_grad = False
    # Only fine-tune the heads for classification and mask prediction
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]


def train_siamese():
    return




def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--root", required=True, help="Path to MOT16 split folder, e.g. MOT16/train or MOT16/test")
    #ap.add_argument("--batch", type=int, default=2)
    #ap.add_argument("--num", type=int, default=1, help="number of batches to inspect")
    args = ap.parse_args()
    



if __name__ == "__main__":
    main()
