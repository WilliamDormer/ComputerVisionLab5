'''
The purpose of this file is to split the original labels file into a training and testing partition (validation will be taken from the training parition)
'''

def split_and_save_txt(input_file, output_file_1, output_file_2, split_percent):
    # Read the content of the input file
    with open(input_file, 'r') as file:
        content = file.readlines()

    # Calculate the split index based on the percentage
    total_length = len(content)
    split_index = int(total_length * (split_percent / 100))

    # Split the content into two parts
    part1 = content[:split_index]
    part2 = content[split_index:]

    # Save the first part into the first output file
    with open(output_file_1, 'w') as file1:
        file1.writelines(part1)

    # Save the second part into the second output file
    with open(output_file_2, 'w') as file2:
        file2.writelines(part2)


# Example usage:
input_file_path = '../data/train_noses.3.txt'
output_file_path_1 = '../data/train_labels.txt'
output_file_path_2 = '../data/test_labels.txt'
split_percentage = 95  # Adjust this value based on your needs

split_and_save_txt(input_file_path, output_file_path_1, output_file_path_2, split_percentage)