
# Copy pasted parallel worker from bird_processing.py
import os
import sys
import numpy as np
import multiprocessing as mp
from subprocess import Popen, PIPE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

#--------------------Parameters---------------------#

# Change this to reduce cpu load
# Hey, if you have 32 cores, might as well use them :)
cores = mp.cpu_count()

#---------------------------------------------------#


# Given prepared datapoints, create a subset where categories have been removed
# This is to reduce redundant feature extraction work.
def subset_datapoints(X_train, X_val, y_train, y_val, n_species):
    classes = np.unique(np.concatenate((y_train, y_val)))
    if n_species >= len(classes):
        return X_train, X_val, y_train, y_val
    else:
        chosen_species = classes[:n_species]
        X_train_n = np.array([X_train[i] for i in range(len(y_train)) if (y_train[i] in chosen_species)])
        X_val_n = np.array([X_val[i] for i in range(len(y_val)) if (y_val[i] in chosen_species)])
        y_train_n = np.array([v for v in y_train if (v in chosen_species)])
        y_val_n = np.array([v for v in y_val if (v in chosen_species)])
        print(f'Set sizes train/validation: [{len(X_train_n)}, {len(X_val_n)}]')
        return X_train_n, X_val_n, y_train_n, y_val_n




# If a method is worse than this,
# then we have definitely failed
def random_guess_accuracy(y_train, y_val):
    n_classes = len(np.unique(np.concatenate((y_train, y_val))))
    return 1.0 / n_classes


# Logistic learner for one dataset
def logistic_learn(input):

    X_train = input[0]
    X_val = input[1]
    y_train = input[2]
    y_val = input[3]
    
    model = LogisticRegression(solver='liblinear', max_iter=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    rand_acc = random_guess_accuracy(y_train, y_val)

    # Use micro for multiclass scoring
    precision = precision_score(y_val, y_pred, average='micro')
    recall = recall_score(y_val, y_pred, average='micro')

    # Formula taken from scikit-learn.org
    f1 = 2 * (precision * recall) / (precision + recall)

    # Return metrics
    return np.array([
        accuracy,
        rand_acc,
        (accuracy-rand_acc),
        precision,
        recall,
        f1
    ])


# Train two logistic regressors
def double_trainer(input):
    return [logistic_learn(dataset) for dataset in input]



#-----------------Multiprocessing-------------------#

# Perform logistic learning accelerated by multiprocessing
# Call this function to spawn a child process that further
# distributes work among cpu cores
def parallel_learners(X_train, X_val, y_train, y_val, max_species, min_species, step):

    print('Setting up logistic trainers for parallel execution: ')
    with Popen('python bird_parallel_learn.py', cwd=os.getcwd(), stdin=PIPE, stdout=PIPE, universal_newlines=True) as p:

        # Send dataset
        with p.stdin as pipe:
            print('Transfering dataset to workers')
            pipe.write(f'{max_species}\n')
            pipe.write(f'{min_species}\n')
            pipe.write(f'{step}\n')
            pipe.write(f'{len(X_train)}\n')
            pipe.write(f'{len(X_val)}\n')
            pipe.write(f'{len(y_train)}\n')
            pipe.write(f'{len(y_val)}\n')
            pipe.flush()
            print(f'Set[{len(X_train)}, {len(X_val)}]')

            for vec in X_train:
                for v in vec:
                    pipe.write(f'{v} ')
                pipe.write('\n')
                pipe.flush()

            for vec in X_val:
                for v in vec:
                    pipe.write(f'{v} ')
                pipe.write('\n')
                pipe.flush()

            for v in y_train:
                pipe.write(f'{v}\n')
                pipe.flush()

            for v in y_val:
                pipe.write(f'{v}\n')
                pipe.flush()

        
        print(f'Initializing trainers: ')

        data = []

        # Receive progress updates
        for line in p.stdout:
            l = line.strip()
            print(l)
            if l == 'Done':
                break

        # Receive data
        for line in p.stdout:
            data.append(line.strip())

        # Reformat
        data_vectors = np.array([[float(v) for v in d.split(' ')] for d in data])
        return data_vectors



# Worker process main function

if __name__ == '__main__':

    # Receive data for training
    
    max_species = int(sys.stdin.readline().strip())
    min_species = int(sys.stdin.readline().strip())
    step = int(sys.stdin.readline().strip())
    X_train_len = int(sys.stdin.readline().strip())
    X_val_len = int(sys.stdin.readline().strip())
    y_train_len = int(sys.stdin.readline().strip())
    y_val_len = int(sys.stdin.readline().strip())

    X_train = np.array([[float(v) for v in (sys.stdin.readline().strip().split(' '))] for idx in range(X_train_len)])
    X_val = np.array([[float(v) for v in (sys.stdin.readline().strip().split(' '))] for idx in range(X_val_len)])
    y_train = np.array([int(sys.stdin.readline().strip()) for idx in range(y_train_len)])
    y_val = np.array([int(sys.stdin.readline().strip()) for idx in range(y_val_len)])

    # Create datasets for all species numbers
    datasets = [
        subset_datapoints(X_train, X_val, y_train, y_val, n_species)
        for n_species in range(min_species, max_species + 1, step)
    ]

    # These are ordered from shortest to longest, regroup based on time demand
    # [1, 2, 3, 4, 5, 6, 7] => [[1, 7], [2, 6], [3, 5], [4]]
    # [1, 2, 3, 4, 5, 6] => [[1, 6], [2, 5], [3, 4]]

    print('Partitioning data')

    N_datasets = len(datasets)

    doubled_sets = [[datasets[k], datasets[N_datasets - k - 1]] for k in range(N_datasets // 2)]
    if N_datasets % 2 == 1: doubled_sets.append([datasets[N_datasets // 2]])

    completed = []

    print('Processing...')
    sys.stdout.flush()
    
    with mp.Pool(cores) as p:
        completed = p.map(double_trainer, doubled_sets)
    print('Done')

    # Reorder back
    output = [completed[k][0] for k in range(N_datasets // 2)]
    if N_datasets % 2 == 1: output.append(completed[N_datasets // 2][0])
    output.extend([completed[N_datasets // 2 - k - 1][1] for k in range(N_datasets // 2)])

    # Send back data:
    for feature in output:
        vec = feature.flatten()
        for v in vec:
            print(v, end=' ')
        print()
        sys.stdout.flush()
