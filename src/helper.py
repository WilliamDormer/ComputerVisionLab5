import torch

def get_euclidean_distance_info(tensor, print_results=False):
    # min, max, mean, standard deviation
    tensor_min = torch.min(tensor).item()
    tensor_max = torch.max(tensor).item()
    tensor_mean = torch.mean(tensor).item()
    tensor_standard_dev = torch.std(tensor).item()

    if print_results == True:
        print("MEAN ED: ", tensor_mean)
        print("MIN ED: ", tensor_min)
        print("MAX ED: ", tensor_max)
        print("STD ED: ", tensor_standard_dev)

    return tensor_min, tensor_max, tensor_mean, tensor_standard_dev