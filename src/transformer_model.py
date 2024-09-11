import torchvision.models
from torchvision.models import ViT_B_16_Weights, vit_b_16
import torch
import torch.nn as nn

class CustomOutputLayer(nn.Module):
    def __init__(self, resized_image_height):
        super(CustomOutputLayer, self).__init__()

        self.resized_image_height = resized_image_height

    def forward(self, x):
        # Map the outputs to the range 0 to 224
        mapped_output = (x + 1) / 2 * self.resized_image_height
        return mapped_output

def get_transformer_model(resized_image_height, saved_path=None, verbose=False, train_entire_transformer = False):

    '''
    :param saved_path:
    :return: a model with weights
    '''

    # todo figure out how how to speed up training (maybe freeze the weights of all except the final linear layer)

    # this is the adjusted FC head that will be used for regressing the pixel locations of the noses
    new_head_fc = torch.nn.Linear(768,2)
    # I don't think we need to initialize explicitly here.

    # num_params: 86567656
    # min_size : height=224, width=224
    model = None

    if saved_path == None:
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        # replace the fc to fit 2 classes
        setattr(model.heads, "head", new_head_fc)
        if verbose:
            print("head FC replaced to fit 2 classes")

        # Freeze all layers
        if train_entire_transformer == False:
            for param in model.parameters():
                param.requires_grad = False

        # Unfreeze last layer
        for param in model.heads.parameters():
            param.requires_grad = True

        # apply the model with the custom output
        model = nn.Sequential(
            model,
            CustomOutputLayer(resized_image_height)
        )

        if verbose:
            print("added custom output layer")

    if saved_path != None:
        model = vit_b_16()

        #replace the fc to fit 2 classes
        setattr(model.heads, "head", new_head_fc)
        if verbose:
            print("head FC replaced to fit 2 classes")

        # Freeze all layers
        if train_entire_transformer==False:
            for param in model.parameters():
                param.requires_grad = False

        # Unfreeze last layer
        for param in model.heads.parameters():
            param.requires_grad = True

        # apply the model with the custom output
        model = nn.Sequential(
            model,
            CustomOutputLayer(resized_image_height)
        )

        if verbose:
            print("added custom output layer")

        try:
            checkpoint = torch.load(saved_path, map_location="cpu")
        except Exception as e:
            print(f"Error loading model parameters: {e}")
            checkpoint = None

        if checkpoint is not None:

            model.load_state_dict(checkpoint)

            # if 'state_dict' in checkpoint:
            #     saved_state_dict = checkpoint['state_dict']
            #
            #     #check that the model's architectures match
            #     model_state_dict = model.state_dict()
            #     if all(k in model_state_dict for k in saved_state_dict):
            #         if verbose:
            #             print("Model architectures match.")
            #         model.load_state_dict(checkpoint)
            #     else:
            #         raise Exception("Model architectures do not match")
            # else:
            #     raise Exception("no state dict found in the saved parameters")
        else:
            raise Exception("failed to load the saved parameters")




    return model