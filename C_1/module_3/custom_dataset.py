"""
In pytorch a custom datasets need to answer 3 questions: 
1) Where to find data? 
2) The length of data 
3) Find the data with the help of index
"""
import torch
from torch.utils.data  import Dataset
import os 
import scipy
from PIL import Image

class CustomDataset(Dataset):
    # Find data
    def __init__(self, root):
        self.root = root
        self.data_path = os.path.join(root,"jpg")
        self.labels_mat = scipy.io.loadmat(os.path.join(root,"image_labels.mat"))
        self.labels = self.labels_mat['labels'][0]-1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        #Build the image 
        #Load the image
        image_name = f"image_{idx+1:05d}.jpg"
        image_path = os.path.join(self.root, image_name)
        load_image = Image.open(image_path)
        labels = self.labels[idx]

        return load_image, labels

    