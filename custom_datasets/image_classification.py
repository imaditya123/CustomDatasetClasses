import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt


class ImageClassificationDataset(Dataset):
    def __init__(self, data_dir, annotations_file, transforms=None):
        """
        Dataset for Image Data.

        Args:
            data_dir (str): Directory containing the image files.
            annotations_file (str): Path to the annotations file for supervised tasks.
            transforms (callable, optional): Optional transformations to be applied on images.
        """
        self.data_dir = data_dir
        self.annotations_file = annotations_file
        self.transforms = transforms

        self.annotation_classes={}
        self.annotations = pd.read_csv(annotations_file,)
        # optional as we generally require the labels in the int datatype
        self.annotation_classes={k:i for i,k in enumerate(self.annotations['label'].unique())}


    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):

        img_path=os.path.join(self.data_dir,self.annotations.iloc[idx]['filename'],)
        img = Image.open(img_path)
        if self.transforms:
            img = self.transforms(img)

        label_str=self.annotations.iloc[idx]['label']
        label = self.annotation_classes[label_str] 
        return img, label

    def convert_itoc(self,x):
      # optional convert int values in classes
      itoc={i:k for k,i in self.annotation_classes.items()}
      x=x.int().numpy()
      return itoc[x[0]] if len(x)==1 else [itoc[xi] for xi in x]