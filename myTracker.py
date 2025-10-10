import os
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



class MOT16(Dataset):

    # example init from https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html

    # def __init__(self, csv_file, root_dir, transform=None):
    #     """
    #     Arguments:
    #         csv_file (string): Path to the csv file with annotations.
    #         root_dir (string): Directory with all the images.
    #         transform (callable, optional): Optional transform to be applied
    #             on a sample.
    #     """
    #     self.landmarks_frame = pd.read_csv(csv_file)
    #     self.root_dir = root_dir
    #     self.transform = transform

    # we cant use that because we need multiple paths and have to keep track of the frames


    def __init__(self, root, transform=None, getGroundTruth = True):
        # setting stuff up like a torch dataset
        # need this stuff to work with torch dataloader

        self.root        = root # directory
        self.transform   = transform # using torch transforms to resize
        self.groundTruth = getGroundTruth # this is True for the training set and False for testing

        self.instances = []
        self.labels = {} 


        # problem is this has to be done for every sequence of images in the folder
        sequences = sorted(os.listdir(self.root)) # "MOT16-01", "MOT16-03", and so on
    
        for name in sequences:
            path = os.path.join(self.root, name) # /MOT16/train/MOT16-01, for example
            images_folder = os.path.join(path, "img1") # folder that holds all the jpegs
            groundtruth_txt = os.path.join(path, "gt", "gt.txt") # this doesnt exist in the test set


            imageFiles = os.listdir(images_folder) 

            for imageFile in imageFiles:
                image = os.path.join(images_folder, imageFile) # the jpg
                imageFileArray = imageFile.split('.') # ["number", "jpg"]
                frame_number = imageFileArray[0]
                frame_number = int(frame_number)
                instance_tuple = (image, name, frame_number) # (the jpg, the string for "MOT16-##", the frame id)
                self.instances.append(instance_tuple)

            # if we're training we want ground truth txt
            if getGroundTruth:
                if not os.path.exists(groundtruth_txt):
                    print(f"training set couldn't find ground truth file for {name}")
                    self.labels[name] = None
                else:
                    groundTruth = pd.read_csv(groundtruth_txt, header=None)
                    # columns according to https://github.com/khalidw/MOT16_Annotator:
                    groundTruth.columns = ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "flag", "class", "visibility"]
                    self.labels[name] = groundTruth
            else:
                # if not training then we dont care
                self.labels[name] = None


    def __len__(self):
        # the number of image files
        # torch's dataLoader needs this function
        # otherwise "TypeError: object of type 'MOT16' has no len()"
        return len(self.instances)
    
    # example getitem from pytorch tutorial https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html
    # def __getitem__(self, idx):
    #     if torch.is_tensor(idx):
    #         idx = idx.tolist()

    #     img_name = os.path.join(self.root_dir,
    #                             self.landmarks_frame.iloc[idx, 0])
    #     image = io.imread(img_name)
    #     landmarks = self.landmarks_frame.iloc[idx, 1:]
    #     landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
    #     sample = {'image': image, 'landmarks': landmarks}

    #     if self.transform:
    #         sample = self.transform(sample)

    #     return sample
    

    def __getitem__(self, index):
        # loads and returns the image and annotations
        # torch needs this too. otherwise "TypeError: 'MOT16' object is not subscriptable"
        if torch.is_tensor(index):
            index = index.tolist()

        (image, name, frame_number) = self.instances[index]

        image = Image.open(image)

        # apply transform
        if self.transform is not None:
            image = self.transform(image)

        groundTruth = self.labels[name] # this is None if its testing
    
        if groundTruth is not None: # ok so we're training
            boundingBoxes = groundTruth[groundTruth["frame"] == frame_number]

            # gotta put this stuff into torch tesnors
            theBoxes  = torch.tensor(boundingBoxes[["bb_left", "bb_top", "bb_width", "bb_height"]].values, dtype=torch.float32)
            objectIDs = torch.tensor(boundingBoxes['id'].values, dtype=torch.int32)

            values = {'sequence': name, 'frameID': frame_number, 'object IDs': objectIDs, 'bounding boxes': theBoxes}
        else: 
            # no ground truth so don't return anything with it
            values = {'sequence': name, 'frameID': frame_number}

        return image, values
    


# def parse_gt_file(file_path):
#     data = []
#     with open(file_path, 'r') as f:
#         pass # do the preprocessing here
#     return data


# color_aug = transforms.Compose([
#     transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
#     transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
# ])

# augmented_image = color_aug(original_image)

# Freeze backbone layers
# def getPretrained():
#     model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
#     for param in model.backbone.parameters():
#     param.requires_grad = False
#     # Only fine-tune the heads for classification and mask prediction
#     params_to_optimize = [p for p in model.parameters() if p.requires_grad]


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
    




if __name__ == "__main__":
    # test the functions real quick
    trainMOT16data = MOT16(root = 'MOT16/train', getGroundTruth=True) 
    testMOT16data = MOT16(root = 'MOT16/test', getGroundTruth=False) 

    print(len(trainMOT16data))
    print(len(testMOT16data))



