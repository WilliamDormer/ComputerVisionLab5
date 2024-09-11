'''
The purpose of this is to validate the model using the test data partition.

It has two modes, one in which it takes in a single image and shows the label prediction
And a second where it takes the entire test data partition and calculates summary statistics on performance.
'''

from transformer_model import get_transformer_model
import argparse
import torch
from torchinfo import summary
import pandas as pd
import ast
from PetDataset import PetDataset
from torch.utils.data import DataLoader, Dataset
import os
from PIL import Image
from torchvision.transforms import Resize, ToTensor, Normalize, ToPILImage
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
import random
import time

from helper import get_euclidean_distance_info

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", help="increase output verbosity")
    parser.add_argument("-id", "--image_directory", type=str, default="../data/images/", help="The directory to load the images from")
    parser.add_argument("-ttlp", "--test_label_path", default="../data/test_labels.txt")
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-rh', '--resize_height', type=int, default=224)
    parser.add_argument("-mp", "--model_path", required=True, type=str, help="The model path")
    parser.add_argument('-s', '--single', action="store_true", help="determines if you want a single image with visuals or the whole training test with stats")
    parser.add_argument("-ti", "--test_images", nargs="+", default=["Abyssinian_3.jpg", "american_pit_bull_terrier_72.jpg", "beagle_17.jpg", "Birman_184.jpg", "chihuahua_93.jpg", "Egyptian_Mau_178.jpg", "american_pit_bull_terrier_163.jpg", "beagle_204.jpg", "german_shorthaired_183.jpg", "Ragdoll_13.jpg", "Bengal_52.jpg", "american_bulldog_188.jpg"], help="The list of test images to use if you're using single mode")
    args = parser.parse_args()

    device = torch.device("cpu")
    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("failed to load cuda, using cpu as backup")
    print("using device: ", device)

    # model = get_transformer_model(resized_image_height=args.resize_height, saved_path=args.model_path, verbose=args.verbose)

    model = models.resnet18(pretrained=True)

    # Modify the last fully connected layer for regression with 2 outputs
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)

    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)

    model.to(device)

    # print out the structure of the model
    if args.verbose:
        summary(model, input_size=(1, 3, 224, 224))
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(f"{name}: {module}")
                # TODO determine the structure of the network so I can figure out how to chop off the final linear layers and replace them.
                # I think it has to do with heads.head, which is a Linear(in_features=768, out_features=1000, bias=True)
                # we don't want to mess with the MLP layers because those are within the encoder, but we do want to mess with the MLP head.

    #single output dataset, needed because we have to have multiple values for the normalization operation.
    class SingleImageTestDataset(Dataset):
        def __init__(self, primary_images, test_labels_df, image_directory, resize_height, desired_batch_size ):
            self.image_directory = image_directory
            self.resize_height = resize_height

            num_images_to_get = desired_batch_size - len(primary_images)
            test_image_names = test_labels_df['image_name'].tolist()
            sampled_names = random.sample(test_image_names, num_images_to_get)

            self.image_names = primary_images + sampled_names

        def __len__(self):
            return len(self.image_names)

        def __getitem__(self, item):
            # get the image from the correct file
            image_path = os.path.join(self.image_directory, self.image_names[item])
            image = Image.open(image_path)

            if image.mode != "RGB":
                # convert to 24 bit colour
                image = image.convert("RGB")

            # doing the transform step by step for visibility
            scaled_image = Resize((self.resize_height, self.resize_height))(image)

            tensor_image = ToTensor()(scaled_image)
            normalized_image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_image)

            return normalized_image, tensor_image


    custom_label_dtypes = {
        'image_name': str,
        'position': str
    }
    test_labels_df = pd.read_csv(args.test_label_path, names=["image_name", "position"], dtype=custom_label_dtypes)

    if not args.single:

        # do the same for the test data
        test_labels_df["position"] = test_labels_df["position"].apply(lambda x: ast.literal_eval(x))
        test_labels = test_labels_df['position'].apply(lambda x: torch.tensor(x)).tolist()
        test_labels = torch.stack(test_labels)
        test_image_names = test_labels_df['image_name'].tolist()

        test_dataset = PetDataset(test_image_names, test_labels, args.image_directory, args.resize_height)

        print("length of test dataset: ", len(test_dataset))

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        correct = 0
        total = 0

        model.eval()

        with torch.no_grad():

            total_dist = torch.empty(0).to(device)

            start = time.time()

            for j, data in enumerate(test_loader):
                print("batch: ", j+1)
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)

                if j == 1:
                    print("shape of inputs: ", inputs.size())
                    print("shape of the outputs: ", outputs.size())
                    print("sample output: ", outputs[1])
                    print("sample label: ", labels[1])

                total += labels.size(0)
                labels = torch.round(labels)
                outputs = torch.round(outputs)
                correct += (outputs == labels).sum().item()



                dist = (labels - outputs).pow(2).sum(1).sqrt()

                #find really high values for inspection
                max_index = torch.argmax(dist)
                if dist[max_index] > 20:
                    # found a bad case
                    print("adversarial example at: ", test_image_names[j*args.batch_size + max_index])


                total_dist = torch.cat((total_dist, dist), dim=0)

            end = time.time()
            print("elapsed time per image: ", (end - start)/len(test_dataset))

            _ = get_euclidean_distance_info(total_dist, print_results=True)

            test_accuracy = correct / total
            print("ACC test {}".format(test_accuracy))
    else:

        #create our sample dataset:
        dataset = SingleImageTestDataset(args.test_images, test_labels_df, args.image_directory, args.resize_height, args.batch_size)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        with torch.no_grad():
            for j, data in enumerate(dataloader):
                normalized_images, tensor_images = data
                normalized_images = normalized_images.to(device)

                # get the output
                outputs = model(normalized_images)
                outputs = torch.round(outputs)

                for i in range(0, len(args.test_images)):
                    # these are the images we actually care about.

                    # now display the image and the output.

                    # Convert the image to a NumPy array for easier manipulation
                    image = tensor_images[i].cpu()
                    image = ToPILImage()(image)
                    image_array = np.array(image)

                    # Display the image using matplotlib
                    plt.imshow(image_array)

                    # Highlight the specified pixel with a red cross
                    plt.scatter(outputs[i][0].cpu(), outputs[i][1].cpu(), color='red', marker='x', s=100)

                    # Show the plot
                    plt.show()


            # for i in args.test_images:
            #     image_path = os.path.join(args.image_directory, i)
            #     image = Image.open(image_path)
            #
            #     if image.mode != "RGB":
            #         image = image.convert("RGB")
            #
            #     scaled_image = Resize((args.resize_height, args.resize_height))(image)
            #     tensor_image = ToTensor()(scaled_image)
            #
            #     # print(tensor_image)
            #
            #
            #
            #     normalized_image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_image)
            #
            #     #add a batch dimension to the tensor, and send to device
            #     input = normalized_image.unsqueeze(0).to(device)
            #     print("shape of input: ", input.size())
            #     #send the image to the model for prediction
            #     output = model(input)
            #     print("output before round: ", output)
            #     output = torch.round(output)
            #     output = output.cpu()
            #
            #     print("output: ", output)
            #     print("size of output: ", output.size())
            #
            #     # now display the image and the output.
            #
            #     # Convert the image to a NumPy array for easier manipulation
            #     image_array = np.array(scaled_image)
            #
            #     # Display the image using matplotlib
            #     plt.imshow(image_array)
            #
            #     # Highlight the specified pixel with a red cross
            #     plt.scatter(output[0][0], output[0][1], color='red', marker='x', s=100)
            #
            #     # Show the plot
            #     plt.show()
