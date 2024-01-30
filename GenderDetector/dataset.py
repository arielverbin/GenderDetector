import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as t


class GenderDataset(Dataset):
    def __init__(self, root_dir, csv_path, transform=None, train=False, val=False, test=False):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.val = val
        self.test = test
        self.csv_path = csv_path

        self.image_paths, self.labels = self.load_dataset()

    def load_dataset(self):
        image_paths = []
        labels = []

        # Iterate through the files in the root directory
        for filename in os.listdir(self.root_dir):
            if filename.endswith(".jpg"):
                image_id = filename  # Assuming the filename is the image_id
                image_path = os.path.join(self.root_dir, filename)

                male_attribute = "Female" in image_id
                label = 0 if male_attribute == 1 else 1

                image_paths.append(image_path)
                labels.append(label)

        return image_paths[0:1000], labels[0:1000]

    def __len__(self):
        return 1000 #len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        # Convert the image to RGB format if it has more than 3 channels
        if image.mode != 'RGB':
            image = image.convert('RGB')

        transformed_image = self.transform(image)

        return transformed_image, label


# Initializing the 3 datasets: train, val, test.
def init_datasets():
    # Define transformations for image resizing
    transform = t.Compose([
        t.Resize(64),  # Resize the image to a consistent size
        t.CenterCrop(64),  # Take only the 64x64 pixels in the center
        t.ToTensor(),  # Convert the image to a PyTorch tensor
    ])

    train_dataset = GenderDataset(root_dir="./dataset/train",
                                  csv_path="./dataset/list_attr_celeba.csv",
                                  transform=transform,
                                  train=True)

    validation_dataset = GenderDataset(root_dir="./dataset/validation",
                                       csv_path="./dataset/list_attr_celeba.csv",
                                       transform=transform,
                                       train=True)

    test_dataset = GenderDataset(root_dir="./dataset/test",
                                 csv_path="./dataset/list_attr_celeba.csv",
                                 transform=transform,
                                 train=True)

    return train_dataset, validation_dataset, test_dataset
