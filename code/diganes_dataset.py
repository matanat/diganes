import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms

class DiganesDataset(torch.utils.data.Dataset):
    """Diganses seals dataset."""

    def __init__(self, csv_file, root_dir, category=None, min_label_count=None, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            root_dir (string): Directory with all the images.
            category (string, optional): Optional ilustration type, can be 'drawing' or 'photo'.
            min_label_count (int, optional): Minimum number of instances per label.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()

        labels = pd.read_csv(csv_file)

        #filter only specifc type of ilusration
        if category:
            labels = labels[labels.category == category]
            labels = labels.reset_index(drop=True)

        labels = labels.drop('category', axis=1)

        if min_label_count:
            #remove labels with insufficent number of samples
            count_per_label = labels.iloc[:,2:].sum()
            columns_to_remove = count_per_label[count_per_label < min_label_count].axes[0]
            labels = labels.drop(columns_to_remove, axis=1)

            #check if any row is left without a label
            labels_per_image = labels.iloc[:,2:].sum(axis=1)
            zero_label_rows = np.nonzero(labels_per_image.to_numpy() == 0)[0]
            labels = labels.drop(zero_label_rows)

            labels = labels.reset_index(drop=True)

        self.classes = labels.columns[2:]
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels.loc[idx, "img_name"])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        labels = torch.Tensor(self.labels.iloc[idx, 2:].astype(int))

        return image, labels

    def show_image(self, image, labels, groud_truth=None):
        """Show image with labels"""

        plt.figure()
        if torch.is_tensor(image):
            plt.imshow(image.permute(1, 2, 0))
        else:
            plt.imshow(image)

        img_label_names = self.classes[(labels > 0)].tolist()
        if groud_truth != None:
            groud_truth_names = self.classes[(groud_truth > 0)].tolist()
            plt.title(str(img_label_names) + '\nGround Truth: ' + str(groud_truth_names))
        else:
            plt.title(img_label_names)
