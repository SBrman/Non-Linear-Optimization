#! python3

from typing import Callable
import numpy as np
from ast import literal_eval
from matplotlib import pyplot as plt


def read_all_data(file_path, features_num: int = 3):
    """Reads the data from the input file"""

    training_data = []

    with open(file_path) as file:
        for i, line in enumerate(file.readlines()):
            data_point = [literal_eval(feature) for feature in line.split('\n')[0].split(' ') 
                          if feature != '']
            
            if len(data_point) != features_num:
                print(f'Skipping {i}th datapoint')
                continue

            training_data.append(data_point)
    training_data = np.array(training_data)

    return training_data


def plotter(data, digit, with_line: Callable = None, file_name=None, algorithm=None, model=None):
    
    target_digit_data = np.array([data_point for data_point in data if data_point[0] == digit])
    non_target_digit_data = np.array([data_point for data_point in data if data_point[0] != digit])

    plt.figure(figsize=(16, 9), dpi=120)
    plt.scatter(target_digit_data[:, 1], target_digit_data[:, 2], c='g', s=1, label=f'{digit}')
    plt.scatter(non_target_digit_data[:, 1], non_target_digit_data[:, 2], c='r', s=1, label=f'not {digit}')
    
    if with_line:
        x = np.linspace(0, 0.7, 100)
        y = with_line(x)
        plt.plot(x, y)

    
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.suptitle(f"Algorithm: {algorithm}   Model: {model}")
    plt.savefig(fr"figs\{file_name}.png")


if __name__ == "__main__":
    data = read_all_data(r'.\data\features_train.txt')
##    data = read_all_data(r'.\data\training_data_raw.txt', 257)
##    plotter(data, 0)

