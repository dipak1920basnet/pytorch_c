import torch
from torch.utils.data import Dataset
import os 
import scipy
from PIL import Image

class OxfordFlowersDataset(Dataset):

    # setting info to find images later
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "jpg")

        # Load Matlab labels
        labels_mat = scipy.io.loadmat(os.path.join(root_dir,"imagelabels.mat"))
        self.labels = labels_mat["labels"][0] - 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """
        idx: index of data
        """
        # :05d pads the index with zeros to match the file names eg: 00042.jpg
        img_name = f"image_{idx+1:05d}.jpg"
        img_path = os.path.join(self.img_dir, img_name)


        # Load the image
        image = Image.open(img_path)
        label = self.labels[idx]

        return image, label
    