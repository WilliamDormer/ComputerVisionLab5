
import torch
import os
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor, Normalize
import argparse
import pandas as pd
import ast
from torch.utils.data import Dataset, DataLoader, random_split, Subset, RandomSampler
from PIL import Image
from tqdm import tqdm
import torch.nn as nn

from transformer_model import get_transformer_model
from torchinfo import summary

from PetDataset import PetDataset

from helper import get_euclidean_distance_info

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action="store_true", help="increase output verbosity")
    parser.add_argument('--model_name', type=str, help="the base name you want to give the model for this training run.")
    parser.add_argument("-id", "--image_directory", type=str, default="../data/images/", help="The directory to load the images from")
    parser.add_argument("-ttlp", "--test_label_path", default="../data/test_labels.txt")
    parser.add_argument("-trlp", "--train_label_path", default="../data/train_labels.txt")
    parser.add_argument("-vp", "--validation_percentage", type=float, default=0.05)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-d", "--device", type=str, default="cuda")
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-rh', '--resize_height', type=int, default=224)
    parser.add_argument("-md", "--model_directory", default="../saved_models/", type=str,
                        help="The relative directory to save the model .pth files")
    parser.add_argument('-en', '--experiment_name', required=True, type=str,
                        help="The unique experiment name for this run.")
    parser.add_argument('-tl', "--train_limit", default=None, type=int, help="The limit for how many training samples to keep (used for overfit testing)")
    parser.add_argument('-lr', '--learning_rate', default=0.001, type=float, help="the learning rate for the optimizer")
    parser.add_argument("--train_entire_transformer", action="store_true", help="this option allows for the entire transformer to be trained, instead of just the final FC head.")

    # TODO add input for the training and testing directories
    # TODO add hyperparameter inputs
    args = parser.parse_args()

    if args.verbose:
        print("Verbosity turned on")
        print("Starting the script...")
    else:
        print("Starting the script...")

    device = torch.device("cpu")
    if args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("failed to load cuda, using cpu as backup")
    print("using device: ", device)

    model_path = args.model_directory + args.experiment_name + ".pth"

    # # get the base model and pretrained weights:
    # if args.train_entire_transformer == True:
    #     print("training entire transformer")
    # model = get_transformer_model(resized_image_height=args.resize_height, verbose=args.verbose, train_entire_transformer=args.train_entire_transformer)



    # TODO determine if resnet fixes it.
    # trying with resnet instead I guess, since this is not working
    # Load pre-trained ResNet-18 model
    model = models.resnet18(pretrained=True)

    # Modify the last fully connected layer for regression with 2 outputs
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)



    criterion = nn.MSELoss() # MSE is good for regression.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

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

    # get the data
    # get the labels
    custom_label_dtypes = {
        'image_name' : str,
        'position' : str
    }
    train_labels_df = pd.read_csv(args.train_label_path, names=["image_name","position"], dtype=custom_label_dtypes)
    test_labels_df = pd.read_csv(args.test_label_path, names=["image_name","position"], dtype=custom_label_dtypes)

    # TODO or perhaps the better way to solve this is by using some sort of colour channel conversion.
    #remove some of the problematic rows:
    # TODO Abyssinian_5.jpg

    def parse_tuple(tuple_string):
        return ast.literal_eval(tuple_string)

    # extract the labels and image name from the training data file.
    train_labels_df["position"] = train_labels_df["position"].apply(lambda x: ast.literal_eval(x))
    #extract the label into the tensor
    train_labels = train_labels_df['position'].apply(lambda x: torch.tensor(x)).tolist()
    train_labels = torch.stack(train_labels)
    # print(labels)

    #extract the image names into a list:
    train_image_names = train_labels_df['image_name'].tolist()

    # # do the same for the test data
    # test_labels_df["position"] = test_labels_df["position"].apply(lambda x: ast.literal_eval(x))
    # test_labels = test_labels_df['position'].apply(lambda x: torch.tensor(x)).tolist()
    # test_labels = torch.stack(test_labels)
    # test_image_names = test_labels_df['image_name'].tolist()


    # # define the DataSet class
    # class PetDataset(Dataset):
    #     def __init__(self, image_names, labels, image_directory, image_transform):
    #         super().__init__()
    #         self.image_directory = image_directory
    #         self.image_names = image_names
    #         self.labels = labels
    #         self.image_transform = image_transform
    #
    #     def __len__(self):
    #         return len(self.labels)
    #
    #     def __getitem__(self, item):
    #         # get the image from the correct file
    #         image_path = os.path.join(self.image_directory, self.image_names[item])
    #         image = Image.open(image_path)
    #
    #         if image.mode != "RGB":
    #             # convert to 24 bit colour
    #             image = image.convert("RGB")
    #
    #         scale_factor_x = args.resize_height / image.width
    #         scale_factor_y = args.resize_height / image.height
    #
    #         # get the right label
    #         label = self.labels[item]
    #
    #         # # this is definitely giving the right label at this point
    #         # #code for printing out the image for test purposes.
    #         # # Convert the image to a NumPy array for easier manipulation
    #         # image_array = np.array(image)
    #         #
    #         # # Display the image using matplotlib
    #         # plt.imshow(image_array)
    #         #
    #         # # Highlight the specified pixel with a red cross
    #         # plt.scatter(*label, color='red', marker='x', s=100)
    #         #
    #         # # Show the plot
    #         # plt.show()
    #
    #         # scale the label in the same way as the image
    #         scaled_label = [round(label[0].item() * scale_factor_x), round(label[1].item() * scale_factor_y)]
    #         scaled_label = torch.FloatTensor(scaled_label)
    #         label = scaled_label
    #         #TODO somehow check if these labels are correct. they are correct.
    #
    #         # # TODO Abyssinian_5.jpg has a 32 bit colour instead of the 24 bit colour that the others have.
    #         # try:
    #         #     image = self.image_transform(image)
    #         # except Exception as e :
    #         #     print(e)
    #         #     print("the image name with the issue was: ", self.image_names[item])
    #
    #         #doing the transform step by step for visibility
    #         scaled_image = Resize((args.resize_height, args.resize_height))(image)
    #
    #         # # #code for printing out the scaled image for test purposes.
    #         # # Convert the image to a NumPy array for easier manipulation
    #         # image_array = np.array(scaled_image)
    #         #
    #         # # Display the image using matplotlib
    #         # plt.imshow(image_array)
    #         #
    #         # # Highlight the specified pixel with a red cross
    #         # plt.scatter(*scaled_label, color='blue', marker='x', s=100)
    #         #
    #         # # Show the plot
    #         # plt.show()
    #
    #         tensor_image = ToTensor()(scaled_image)
    #         normalized_image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor_image)
    #
    #         return normalized_image, label


    # Define the transformation pipeline
    transform = transforms.Compose([
        #TODO could add padding before doing the resize to maintain the aspect ratio, probably should.
        transforms.Resize((args.resize_height, args.resize_height)),  # Adjust the size as per your model's input requirements
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # construct the overall dataset:
    dataset = PetDataset(train_image_names, train_labels, args.image_directory, args.resize_height)
    #set the test dataset
    # test_dataset = PetDataset(test_image_names, test_labels, args.image_directory, args.resize_height)

    #TODO the problem may be with the way I'm splitting the dataset, because how would it know which parts to assign to which?
    total_length = len(train_labels)
    validation_length = int(args.validation_percentage * total_length)
    train_length = total_length - validation_length

    # get the train and validation datasets.
    train_dataset, validation_dataset = random_split(
        dataset, [train_length, validation_length]
    )



    # #split the train dataset into train and validation
    # total_length = len(dataset)
    # train_length = int(args.train_percentage * total_length)
    #
    # # now split the dataset into the train, test, and validation partitions
    # total_length = len(dataset)
    # train_length = int(args.train_percentage * total_length)
    # validation_length = int(args.validation_percentage * total_length)
    # test_length = total_length - train_length - validation_length
    #
    # train_dataset, test_dataset, validation_dataset = random_split(
    #     dataset, [train_length, test_length, validation_length]
    # )

    print("length of train dataset: ", len(train_dataset))
    # print("length of test dataset: ", len(test_dataset))
    print("length of validation dataset: ", len(validation_dataset))

    # You can then use DataLoader to create loaders for each split
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    if args.train_limit:
        # keep only train limit items from the training dataset
        print("limiting train set")
        random_sampler = RandomSampler(train_dataset, replacement=False, num_samples=args.train_limit)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=random_sampler)

    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    # gonna make some test train loaders to check if the labeling is working for all of them
    ex_train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    ex_validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True)

    # print("trying example for train")
    # for j, data in enumerate(ex_train_loader):
    #
    #     break
    # print("trying example for validation")
    # for j, data in enumerate(ex_validation_loader):
    #     break
    # print("done checking exmaples")

    # store a record of the training and validation loss for plotting later.
    training_loss_record = torch.empty((args.epochs, 1), requires_grad=False)
    validation_loss_record = torch.empty((args.epochs, 1), requires_grad=False)

    # setting a high initial v_loss record so that it's easy to beat.
    best_vloss = 100000000.

    # def get_euclidean_distance_info(tensor, print_results=False):
    #     #min, max, mean, standard deviation
    #     tensor_min = torch.min(tensor).item()
    #     tensor_max = torch.max(tensor).item()
    #     tensor_mean = torch.mean(tensor).item()
    #     tensor_standard_dev = torch.std(tensor).item()
    #
    #     if print_results == True:
    #         print("MEAN ED: ", tensor_mean)
    #         print("MIN ED: ", tensor_min)
    #         print("MAX ED: ", tensor_max)
    #         print("STD ED: ", tensor_standard_dev)
    #
    #     return tensor_min, tensor_max, tensor_mean, tensor_standard_dev

    print("batches per epoch: ", len(train_loader))

    for i in tqdm(range(args.epochs)):
        model.train()
        running_loss = 0.
        running_dist = torch.empty((0)).to(device)

        t_total = 0
        t_correct = 0
        print("starting training for epoch")
        for j, data in enumerate(train_loader):
            print("batch: ", j+1)
            t_inputs, t_labels = data
            t_inputs = t_inputs.to(device)
            t_labels = t_labels.to(device)

            # print("t_labels")
            # print(t_labels)

            #zero the gradients
            optimizer.zero_grad()

            # compute the output of the transformer
            t_outputs = model(t_inputs)

            # print("shape of labels: ", t_labels.size())
            # print("shape of output: ", t_output.size())
            # print(output)

            # print("sample output and label: ", t_outputs[0], ", ", t_labels[0])

            loss = criterion(t_outputs, t_labels)

            loss.backward()

            optimizer.step()

            # todo need to convert the labels and outputs to rounded integers for compairison.
            t_total += t_labels.size(0)
            # TODO how do we determine what is close enough, is it the exact pixel? Or is it within some margin of pixels? fix this for valdation too
            #TODO could try testing on some input images to see where it is placing them (is it close or not really)
            # i think we need to bucket these or do some distance metric, because this is too stringent. I mean come on, even I couldn't pick out the exact pixel that someone chose.
            t_labels = torch.round(t_labels)
            t_outputs = torch.round(t_outputs)

            t_dist = (t_labels - t_outputs).pow(2).sum(1).sqrt()
            # print("t_dist shape: ", t_dist.size())
            # print(t_dist)
            running_dist = torch.cat((running_dist, t_dist),0)
            # _ = get_euclidean_distance_info(t_dist, print_results=True)

            t_correct += (t_outputs == t_labels).sum().item()

            running_loss += loss

        _ = get_euclidean_distance_info(running_dist, print_results=True)


        avg_train_loss = running_loss / len(train_dataset)

        running_valid_loss = 0.
        running_valid_dist = 0.

        model.eval()

        v_correct = 0
        v_total = 0


        print("starting validation for epoch")
        with torch.no_grad():
            for j, data in enumerate(validation_loader):
                print("batch: ", j+1)
                v_inputs, v_labels = data
                v_inputs = v_inputs.to(device)
                v_labels = v_labels.to(device)

                v_outputs = model(v_inputs)
                v_loss = criterion(v_outputs, v_labels)

                #todo convert these to integers for comparing.
                v_total += v_labels.size(0)
                v_labels = torch.round(v_labels)
                v_outputs = torch.round(v_outputs)



                v_correct += (v_outputs == v_labels).sum().item()

                running_valid_loss += v_loss

        avg_validation_loss = running_valid_loss / len(validation_dataset)

        training_accuracy = t_correct / t_total
        validation_accuracy = v_correct / v_total

        print('LOSS train {} valid {}'.format(avg_train_loss, avg_validation_loss))
        print("ACC train {} valid {}".format(training_accuracy, validation_accuracy))

        # update the loss records
        training_loss_record[i] = avg_train_loss
        validation_loss_record[i] = avg_validation_loss

        # determine if it's the best model (using validation loss) and save if it is.
        # Track the best performance, and save the model's state
        if avg_validation_loss < best_vloss:
            best_validation_loss = avg_validation_loss
            best_model_path = model_path
            torch.save(model.state_dict(), model_path)

        # scheduler.step() #step the optimizer after epoch.


    # get the loss records
    loss_record = training_loss_record.cpu().detach().numpy()
    v_loss_record = validation_loss_record.cpu().detach().numpy()

    # Generate and save the loss plots
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(loss_record) + 1))
    plt.plot(epochs, loss_record, label="Training Loss", marker='o')
    plt.plot(epochs, v_loss_record, label="Validation Loss", marker='o')

    plt.title('Loss Over Epochs')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.savefig(args.model_directory + args.experiment_name + ".jpg")

    # # perform testing on the best model version.
    # model.load_state_dict(torch.load(model_path))
    # model.eval()
    # total_acc = 0
    #
    # # keeping parameters for accuracy and confusion matrix calculations.
    # all_predictions = []
    # all_labels = []
    # total = 0
    # correct = 0
    #
    # # calculate the localization accuracy statistics
    # # TODO minimum ED
    # # TODO mean ED
    # # TODO maximum ED
    # # TODO standard deviation ED


    print("done")








