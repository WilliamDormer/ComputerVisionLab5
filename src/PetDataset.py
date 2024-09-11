from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor, Normalize
from PIL import Image
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

class PetDataset(Dataset):
    def __init__(self, image_names, labels, image_directory,  resize_height, image_transform=None,):
        super().__init__()
        self.image_directory = image_directory
        self.image_names = image_names
        self.labels = labels
        self.image_transform = image_transform
        self.resize_height = resize_height

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        # get the image from the correct file
        image_path = os.path.join(self.image_directory, self.image_names[item])
        image = Image.open(image_path)

        if image.mode != "RGB":
            # convert to 24 bit colour
            image = image.convert("RGB")

        scale_factor_x = self.resize_height / image.width
        scale_factor_y = self.resize_height / image.height

        # get the right label
        label = self.labels[item]

        # # this is definitely giving the right label at this point
        # #code for printing out the image for test purposes.
        # # Convert the image to a NumPy array for easier manipulation
        # image_array = np.array(image)
        #
        # # Display the image using matplotlib
        # plt.imshow(image_array)
        #
        # # Highlight the specified pixel with a red cross
        # plt.scatter(*label, color='red', marker='x', s=100)
        #
        # # Show the plot
        # plt.show()

        # scale the label in the same way as the image
        scaled_label = [round(label[0].item() * scale_factor_x), round(label[1].item() * scale_factor_y)]
        scaled_label = torch.FloatTensor(scaled_label)
        label = scaled_label
        # TODO somehow check if these labels are correct. they are correct.

        # # TODO Abyssinian_5.jpg has a 32 bit colour instead of the 24 bit colour that the others have.
        # try:
        #     image = self.image_transform(image)
        # except Exception as e :
        #     print(e)
        #     print("the image name with the issue was: ", self.image_names[item])

        # doing the transform step by step for visibility
        scaled_image = Resize((self.resize_height, self.resize_height))(image)

        # # #code for printing out the scaled image for test purposes.
        # # Convert the image to a NumPy array for easier manipulation
        # image_array = np.array(scaled_image)
        #
        # # Display the image using matplotlib
        # plt.imshow(image_array)
        #
        # # Highlight the specified pixel with a red cross
        # plt.scatter(*scaled_label, color='blue', marker='x', s=100)
        #
        # # Show the plot
        # plt.show()

        tensor_image = ToTensor()(scaled_image)
        normalized_image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_image)

        return normalized_image, label
