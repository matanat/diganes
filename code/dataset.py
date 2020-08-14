import torch
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

class H5Dataset(torch.utils.data.Dataset):
    # from https://medium.com/@oribarel/getting-the-most-out-of-your-google-colab-2b0585f82403
    def __init__(self, in_file, transform=None):
        super().__init__()

        self.file = h5py.File(in_file, 'r')
        self.transform = transform
        self.nof_classes = len(self.file['Y'][0, ...][0])
        self.pos_count = self.file['Y'][:].sum(axis=0).squeeze()

    def __getitem__(self, index):
        x = self.file['X'][index, ...]
        y = self.file['Y'][index, ...]

        y = y.astype(np.float32).squeeze()

        # Preprocessing each image
        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.file['X'].shape[0]

class DiganesDataset(torch.utils.data.Dataset):
    """Diganses seals dataset."""

    def __init__(self, csv_file, root_dir, category=None, min_label_count=None, transform=None, reverse_transform=None):
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
        self.labels_frame = labels
        self.root_dir = root_dir
        self.transform = transform
        self.reverse_transform = reverse_transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.labels_frame.loc[idx, "img_name"])
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)

        labels = torch.Tensor(self.labels_frame.iloc[idx, 2:].astype(int))

        return image, labels

    def show_image(self, image, labels, predictions=None):
        """Show image with labels"""

        plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if torch.is_tensor(image) and self.reverse_transform:
            plt.imshow(self.reverse_transform(image))
        else:
            plt.imshow(image)

        img_label_names = self.classes[(labels > 0)].tolist()
        if predictions != None:
            predictions_names = self.classes[(predictions > 0)].tolist()
            plt.title('P: ' + str(predictions_names) + '\nL: ' + str(img_label_names))
        else:
            plt.title(img_label_names)

    def label_count(self):
        return self.labels_frame.iloc[:, 2:].to_numpy().sum(axis=0)
