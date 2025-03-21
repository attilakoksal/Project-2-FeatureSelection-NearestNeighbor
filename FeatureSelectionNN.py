import numpy as np
from itertools import combinations
import random
import time

datasetName = ['CS170_Small_Data__15.txt', 'CS170_Large_Data__18.txt', 'CS170_Large_Data__99.txt', 'iris.data']
nFeatures = [10, 20, 80, 4]

# User Interface Implementation
print('Welcome to the Feature Selection algorithm')
nameInput = input('Type name of the file to test: ')
print('\n')


if('small' in nameInput):
    index_dataset = 0
elif('large' in nameInput):
    index_dataset = 1
else:
    index_dataset = 3


algorithmNum = input('Type the name of algorithm you want to run: \n1) Forward Selection\n2) Backward Elimination\n')
print('\n')

samplingAlgorithm = int(input('Do you want to apply sampling algorithm to reduce the time (Enter 1 or 2): \n1) Yes\n2) No\n'))
print('\n')

if samplingAlgorithm == 1:
    sampling_rate = int(input('Enter your desired sampling rate (Enter an integer number): \n'))
else:
    sampling_rate = 1


# Reading dataset
# Real world dataset
if (index_dataset == 3):
    with open(datasetName[index_dataset], "r") as file:
        file_content = file.readlines()
    for i in range(len(file_content)):
        file_content[i] = file_content[i][0:-1]
    file_content.remove('')

    # Get number of sanples
    num_samples = len(file_content)

    # Process data
    data = []

    # Split columns
    for i in range(num_samples):
        data.append(file_content[i].split(',')[0:5])

    # Encode target classes to integer values
    Encode_classes = [('Iris-setosa', 1), ('Iris-versicolor', 2), ('Iris-virginica', 3)]
    for i in range(num_samples):
        if (data[i][4] == 'Iris-setosa'):
            data[i][4] = 1
        elif (data[i][4] == 'Iris-versicolor'):
            data[i][4] = 2
        else:
            data[i][4] = 3
        # Convert strings to integers
        data[i][0] = float(data[i][0])
        data[i][1] = float(data[i][1])
        data[i][2] = float(data[i][2])
        data[i][3] = float(data[i][3])

    # Convert list data to numpy array, and put target class to first column to math dataset to previous format
    data = np.array(data)
    data = data[:,[4,0,1,2,3]]

    # Normalize the data using (0-1) normalization
    min_values = np.min(data, axis=0)
    max_values = np.max(data, axis=0)
    data = (data - min_values) / (max_values - min_values)

    print(f'This dataset has {nFeatures[index_dataset]} features (not including the class attribute), with {num_samples} instances\n')

# Synthetic dataset
else:
    # Open file in binary mode 
    with open(datasetName[index_dataset], 'rb') as file:
        ascii_text = file.read().decode('ascii') # Read the ASCII text data

    # Split the text into individual floating-point numbers:
    numbers = ascii_text.split()

    num_samples = int(len(numbers)/(nFeatures[index_dataset]+1))

    # convert the IEEE format to floating point numbers
    data = []
    for num in numbers:
        if (num[0] == '-'):
            data.append(float(num[0:10]) * 10**(int(num[11:])))
        else:
            data.append(float(num[0:9]) * 10**(int(num[10:])))
    data = np.array(data).reshape((num_samples, nFeatures[index_dataset]+1)) # This is the data we can work on it

    print(f'This dataset has {nFeatures[index_dataset]} features (not including the class attribute), with {num_samples} instances\n')

# Forward selection for feature selection
def getCombinationsWithLen(lst, length):
    return list(combinations(lst, length))

def forwardSelection(data, nFeatures, samplingAlgorithm, samp_rate = 100):
    num_test_org, _ = data.shape
    print('Beginning search:\n')
    best_feat_so_far = []
    best_feat = []
    best_acc = 0.0
    temp = []

    if samplingAlgorithm == 1:
        num_test = int(num_test_org/samp_rate)
    else:
        num_test = num_test_org

    for L in range(1, nFeatures+1):    
        temp  = list(range(1, nFeatures+1))
        if len(best_feat_so_far) != 0:
            for item in best_feat_so_far:
                temp.remove(item)
        else:
            temp = list(range(1, nFeatures+1))
        combinations_list = getCombinationsWithLen(temp, L-len(best_feat_so_far))
        acc = 0.0
        best_so = []
        for item in list(combinations_list):
            feat_ind = best_feat_so_far + list(item)
            label_pred = []
            ind_range = random.sample(range(0, num_test_org), num_test)
            for i in ind_range:
                label_pred.append(nearestNeighbor(data[:,[0]+feat_ind], data[i,feat_ind], i))
            correct = 0
            k = 0
            for i in ind_range:
                if (int(data[i,0]) == label_pred[k]):
                    correct += 1
                k += 1
            print(f'Accuracy for features {feat_ind} = {round(correct/num_test*100, 2)}%')
            if ( (correct/num_test*100) > acc):
                acc = (correct/num_test*100)
                best_so = feat_ind
                if(acc >= best_acc):
                    best_acc = acc
                    best_feat.append(feat_ind)
        best_feat_so_far = best_so
        print('\n')
    
    acc_list = []
    for  feat_ind in best_feat:
    
        label_pred = []
        for i in range(num_test_org):
            label_pred.append(nearestNeighbor(data[:,[0]+feat_ind], data[i,feat_ind], i))
        correct = 0

        for i in range(num_test_org):
            if (int(data[i,0]) == label_pred[i]):
                correct += 1
        
        acc_list.append(correct/num_test_org*100)
    
    max_ind = acc_list.index(max(acc_list))
    
    print(f'\n\nThe best accuracy achieved by features {best_feat[max_ind]} and accuracy = {round(acc_list[max_ind], 2)}%\n\n')


# Nearest Neighbor function

def distFunction(x, y):
    dist = np.linalg.norm(x-y)
    return dist
    
def nearestNeighbor(dataset, test_sample, index_data):
    num_samples, _ = dataset.shape
    Dist_list = []
    for i in range(num_samples):
        if (i != index_data):
            Dist_list.append(distFunction(dataset[i,1:], test_sample))
        else:
            Dist_list.append(10000000000.0)
    label = int(dataset[Dist_list.index(min(Dist_list)), 0])
    return label

# Backward Elimination for feature selection
def getCombinationsWithLen(lst, length):
    return list(combinations(lst, length))

def backwardElimination(data, nFeatures, samplingAlgorithm, samp_rate = 100):
    num_test_org, _ = data.shape
    print('Beginning search:\n')
    worst_feat_so_far = []
    best_feat = []
    best_acc = 0.0
    temp = list(range(1, nFeatures+1))

    if samplingAlgorithm == 1:
        num_test = int(num_test_org/samp_rate)
    else:
        num_test = num_test_org
    
    for L in range(0, nFeatures):    
        # Add availabels:
        temp  = list(range(1, nFeatures+1))
        if len(worst_feat_so_far) != 0:
            for item in worst_feat_so_far:
                temp.remove(item)
        else:
            temp = list(range(1, nFeatures+1))
        combinations_list = getCombinationsWithLen(temp, nFeatures-L)
        acc = 0.0
        worst_so = []
        for item in list(combinations_list):
            feat_ind = list(item)
            label_pred = []
            ind_range = random.sample(range(0, num_test_org), num_test)
            for i in ind_range:
                label_pred.append(nearestNeighbor(data[:,[0]+feat_ind], data[i,feat_ind], i))
            correct = 0
            k = 0
            for i in ind_range:
                if (int(data[i,0]) == label_pred[k]):
                    correct += 1
                k += 1
            print(f'Accuracy for features {feat_ind} = {round(correct/num_test*100, 2)}%')
                                  
            if ( (correct/num_test*100) > acc):
                acc = (correct/num_test*100)
                worst_so = list(range(1, nFeatures+1)) 
                if len(feat_ind) != 0:
                    for item in feat_ind:
                        worst_so.remove(item)
                if(acc >= best_acc):
                    best_acc = acc
                    best_feat.append(feat_ind)
        worst_feat_so_far = worst_so
        print('\n')

    acc_list = []
    for  feat_ind in best_feat:
        label_pred = []
        for i in range(num_test_org):
            label_pred.append(nearestNeighbor(data[:,[0]+feat_ind], data[i,feat_ind], i))
        correct = 0

        for i in range(num_test_org):
            if (int(data[i,0]) == label_pred[i]):
                correct += 1
        acc_list.append(correct/num_test_org*100)
    
    max_ind = acc_list.index(max(acc_list))

    print(f'\n\nThe best accuracy achieved by features {best_feat[max_ind]} and accuracy = {round(acc_list[max_ind], 2)}%\n\n')


# Run Feature selection algorithm on dataset
current_time = time.time()

if (algorithmNum == '1'):
    forwardSelection(data, nFeatures[index_dataset], samplingAlgorithm, sampling_rate)
else:
    backwardElimination(data, nFeatures[index_dataset], samplingAlgorithm, sampling_rate)

elapsed_time = time.time() - current_time

print("Elapsed time for this run: ", elapsed_time, "seconds\n\n")