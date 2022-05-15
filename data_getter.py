import numpy as np
from mnist_reader import read_all_data


def prepare_dataset(dataset_path, digit: int, binary_label: bool = False, full_data: bool = False):
    """Returns the prepared dataset based on the input digit."""
    # dataset_path r'.\data\features_train.txt'
    
    features_num = 3 if not full_data else 257

    data = read_all_data(dataset_path, features_num=features_num)
    
    # Seperate features and labels into A and b
    A = data[:, 1:]
    # Adding ones for the intercepts to the feature matrix
    A_tilde = np.column_stack((np.ones(A.shape[0]), A))

    # Labels
    b = data[:, 0]
    # Replacing the target digit with +1 and all other digits with -1 (or 0 for logistic regression) in vector b
    b = np.where(b == digit, 1, -1)# if not binary_label else 0) 
    
    return A_tilde, b