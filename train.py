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

    # transforms. dont augment test data
    #print(torch.cuda.device_count())      # Should show number of NVIDIA GPUs available
    #print(torch.cuda.get_device_name(0))
    tf = transforms.ToTensor()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # do augmentation here
    augment_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        #transforms.RandomGrayscale(p=0.1),
        #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
        transforms.ToTensor(),
        # random rectangular occlusions might help training
        #transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])


    # get_ground_truth=True for train, False for test
    # get_gt = "train" in os.path.abspath(args.root)
    # ds = MOT16(root=args.root, transform=tf, getGroundTruth=get_gt)

    trainMOT16data = MOT16(root = 'MOT16/train', transform=augment_transform, getGroundTruth=True) 
    testMOT16data = MOT16(root = 'MOT16/test', transform=tf, getGroundTruth=False)


    print(f"[INFO] train dataset size: {len(trainMOT16data)} frames")


    train_loader = DataLoader(
        trainMOT16data, batch_size=16, shuffle=True,
        collate_fn=detection_collate, num_workers=4
    )

    test_loader = DataLoader(
        testMOT16data, batch_size=16, shuffle=False,
        collate_fn=detection_collate, num_workers=4
    )


    # pretrained model
    model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # Freeze backbone layers
    for param in model.backbone.parameters():
        param.requires_grad = False
    # Only fine-tune the heads for classification and mask prediction
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    # print(device)
    model.to(device)
    model.train()

    opt = torch.optim.Adam(params_to_optimize, lr=0.001)
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        imagesDone = 0 
        for images, targets, _ in train_loader:
            images = list(img.to(device) for img in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            opt.zero_grad()
            losses.backward()
            opt.step()
            imagesDone += len(images)
            print(f"Loss: {losses.item():.4f} | ({imagesDone}/{len(train_loader.dataset)})")
            del images, targets, loss_dict, losses
            torch.cuda.empty_cache()

    torch.save(model.state_dict(), "finetunedfasterrcnn.pth")

def train_siamese():
    return




def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--root", required=True, help="Path to MOT16 split folder, e.g. MOT16/train or MOT16/test")
    #ap.add_argument("--batch", type=int, default=2)
    #ap.add_argument("--num", type=int, default=1, help="number of batches to inspect")
    args = ap.parse_args()
    



if __name__ == "__main__":
    # main()
    train_fasterCNN()
