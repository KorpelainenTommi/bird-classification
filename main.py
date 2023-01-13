import random
import numpy as np
import pandas as pd

import bird_utility
import bird_processing

import time
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from bird_learn import logistic_learn, CNN_learn
from bird_parallel_learn import parallel_learners


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

#--------------------Parameters---------------------#

default_seed = 42
training_split = 0.2
csv_path = 'xeno-canto/xeno-canto_ca-nv_index.csv'

# Discard audio files shorter than this
# Also used as window duration
min_duration = 2

# Supported feature extraction methods
feature_extraction_methods = [

    # Logistic regression compatible
    'mfcc',
    'stft',
    'chroma',
    'melspec',
    'centroid',
    'bandwidth',
    'downsampling',

    # CNN only
    'spectral-tensor',
    'time-tensor'

]

#---------------------------------------------------#


#---------------------Methods-----------------------#

# Prepare data from index. Choose the scope of the classification problem
# by only considering n species from the dataset
def data_preparation(num_species, seed=default_seed):

    n_species = num_species

    # Open index and filter columns we don't need
    csv = pd.read_csv(csv_path)
    csv = csv[['english_cname', 'file_name', 'duration_seconds']]

    # Discard audio files that are too short
    print(np.max(csv['duration_seconds']))
    csv = csv[csv['duration_seconds'] >= min_duration]
    csv = csv.drop(columns=['duration_seconds'])
    
    csv.columns = ['name', 'file']
    print(f'Number of datapoints in index: {len(csv)}')
    
    # Optionally filter species to reduce scope
    if n_species > 0:
        species = csv['name'].drop_duplicates().values.tolist()
        random.Random(seed).shuffle(species)
        chosen_species = species[:n_species]
        csv = csv[csv['name'].isin(chosen_species)]
    else:
        # Set the value to the amount of unique species
        n_species = len(csv['name'].drop_duplicates().values.tolist())

    # Transform labels to numerical categories
    label_encoder = LabelEncoder().fit(csv['name'])
    labels = label_encoder.transform(csv['name'].to_numpy())

    # Binarizer for one hot encoding
    label_binarizer = LabelBinarizer().fit(labels)

    # Split dataset
    X_train_f, X_val_f, y_train, y_val = train_test_split(csv['file'].to_numpy(), labels, test_size=training_split, random_state=seed)
    print(f'Set sizes train/validation: [{len(X_train_f)}, {len(X_val_f)}]')
    return X_train_f, X_val_f, y_train, y_val, label_encoder, label_binarizer


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


# Test feature extration and confirm the shapes of processed data
def feature_extraction_showcase(file):
    windowed = bird_processing.preprocessing(file)
    sr = 22050
    bird_utility.audio_plot(windowed, sr, 'Original')
    bird_utility.audio_plot(bird_processing.audio_downsample(windowed, 256), 256 // 2, 'Downsampled')
    bird_utility.spectral_power_plot(bird_processing.audio_melspectrogram(windowed, sr), sr, 'Melspectrogram')
    bird_utility.mfcc_plot(bird_processing.audio_mfcc(windowed, sr), sr, 'MFCC')
    bird_utility.spectral_amplitude_plot(bird_processing.audio_stft(windowed), sr, 'STFT')
    bird_utility.spectral_amplitude_plot(bird_processing.audio_chroma(windowed, sr), sr, 'Chromagram')
    bird_utility.centroid_plot(bird_processing.audio_centroid(windowed, sr), sr, 'Spectral centroid')
    bird_utility.centroid_plot(bird_processing.audio_spectral_band(windowed, sr), sr, 'Spectral bandwidth')
    
    # Confirm shapes
    print('\nFeature shapes')
    print(f'original shape: {windowed.shape}')
    print(f'downsampled shape: {bird_processing.audio_downsample(windowed, 256).shape}')
    print(f'melspec shape: {bird_processing.audio_melspectrogram(windowed, sr).shape}')
    print(f'mfcc shape: {bird_processing.audio_mfcc(windowed, sr).shape}')
    print(f'stft shape: {bird_processing.audio_stft(windowed).shape}')
    print(f'chroma shape: {bird_processing.audio_chroma(windowed, sr).shape}')
    print(f'centroid shape: {bird_processing.audio_centroid(windowed, sr).shape}')
    print(f'spec band shape: {bird_processing.audio_spectral_band(windowed, sr).shape}')


# Given datapoints from the index, generate a feature matrix using some method
def generate_feature_matrix(X_train_f, X_val_f, chosen_method='mfcc'):
    print(f"Extracting feature vectors: method='{chosen_method}'")
    
    t_start = time.time()
    X_train = bird_processing.extract_features(X_train_f, chosen_method)
    print(f'Time elapsed: {"%.2f" % (time.time() - t_start)} s')
    
    t_start = time.time()
    X_val = bird_processing.extract_features(X_val_f, chosen_method)
    print(f'Time elapsed: {"%.2f" % (time.time() - t_start)} s')
    
    print('Feature extraction complete')
    print(f'Received flattened feature data: [X_train={X_train.shape}, X_val={X_val.shape}]')
    return X_train, X_val


# Train logistic regressors for multiple methods and compare the results
# Graph model performace compared to problem scope (n_species)
def logistic_training_montage(max_species, min_species=2, step=1, n_methods=2, seed=default_seed):

    X_train_files, X_val_files, y_train, y_val, label_encoder, label_binarizer = data_preparation(num_species=max_species, seed=seed)
    N = (max_species - min_species) // step + 1
    results = np.zeros(shape=(n_methods, N))
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange', 'purple']

    # Test all methods
    for m in range(n_methods):
        X_train, X_val = generate_feature_matrix(X_train_files, X_val_files, feature_extraction_methods[m])

        # Learn logistic regressors in parallel
        t_start = time.time()
        metrics_vector = parallel_learners(X_train, X_val, y_train, y_val, max_species, min_species, step)
        print(f'Time elapsed: {"%.2f" % (time.time() - t_start)} s')

        # Add results
        for s in range(N):
            results[m][s] = metrics_vector[s][0] # accuracy
        
        # Write metrics to file
        with open(f'metrics-{feature_extraction_methods[m]}.txt', mode = 'w') as out:
            out.write(f'Testing {feature_extraction_methods[m]} for categories [{min_species} => {max_species}]\n')
            out.write(f'Step size {step}\n')
            for metric in metrics_vector:
                for v in metric:
                    out.write(f'{v} ')
                out.write('\n')

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Logistic regression methods')
    plt.xlabel('Number of categories')
    plt.ylabel('Accuracy')

    guess = [1.0 / i for i in range(min_species, max_species + 1, step)]
    ax.plot(range(min_species, max_species + 1, step), guess, 'k', label='random quess')

    for m in range(n_methods):
        ax.plot(range(min_species, max_species + 1, step), results[m], colors[m], label=feature_extraction_methods[m])
    ax.legend()
    plt.savefig('multi-comp.pdf')
    plt.show()



# Compare logistic regression and CNN
def logistic_CNN_comparison(max_species, min_species=2, step=1,
                            logistic_method='mfcc', CNN_method='spectral-tensor',
                            CNN_architecture='CNN_S128', CNN_input_shape=(128, 128, 3), seed=default_seed):

    X_train_files, X_val_files, y_train, y_val, label_encoder, label_binarizer = data_preparation(num_species=max_species, seed=seed)
    N = (max_species - min_species) // step + 1
    results = np.zeros(shape=(2, N))
    colors = ['r', 'b']

    X_train_l, X_val_l = generate_feature_matrix(X_train_files, X_val_files, logistic_method)
    X_train_cnn, X_val_cnn = generate_feature_matrix(X_train_files, X_val_files, CNN_method)

    # Learn logistic regressors in parallel
    t_start = time.time()
    metrics_vector = parallel_learners(X_train_l, X_val_l, y_train, y_val, max_species, min_species, step)
    print(f'Time elapsed: {"%.2f" % (time.time() - t_start)} s')

    # Add results
    for s in range(N):
        results[0][s] = metrics_vector[s][0] # accuracy
    
    # Write metrics to file
    with open(f'metrics-{logistic_method}.txt', mode = 'w') as out:
        out.write(f'Testing {logistic_method} for categories [{min_species} => {max_species}]\n')
        out.write(f'Step size {step}\n')
        for metric in metrics_vector:
            for v in metric:
                out.write(f'{v} ')
            out.write('\n')

    # Learn CNNs (one at a time, but gpu boosted)
    for num_spec in range(min_species, max_species, step):

        X_train_c, X_val_c, y_train_c, y_val_c = subset_datapoints(X_train_cnn, X_val_cnn, y_train, y_val, num_spec)

        # Actually need to refit the binarizer so the y shape matches the generated CNN architecture
        print(f'Shape concat {np.concatenate((y_train_c, y_val_c)).shape}')
        print(f'num unique {len(np.unique(np.concatenate((y_train_c, y_val_c))))}')

        binarizer = LabelBinarizer().fit(np.concatenate((y_train_c, y_val_c)))

        X_train_c = np.array([np.reshape(feature, CNN_input_shape) for feature in X_train_c])
        X_val_c = np.array([np.reshape(feature, CNN_input_shape) for feature in X_val_c])

        y_train_c = binarizer.transform(y_train_c)
        y_val_c = binarizer.transform(y_val_c)
        print(y_train_c.shape)
        print(y_val_c.shape)
        
        metrics = CNN_learn(X_train_c, y_train_c, X_val_c, y_val_c, CNN_input_shape, num_spec, 50, CNN_architecture, plots=False)
        results[1][(num_spec - min_species) // step] = metrics['accuracy']

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title('Logistic regression vs CNN')
    plt.xlabel('Number of categories')
    plt.ylabel('Accuracy')

    guess = [1.0 / i for i in range(min_species, max_species + 1, step)]
    ax.plot(range(min_species, max_species + 1, step), guess, 'k', label='random quess')

    ax.plot(range(min_species, max_species + 1, step), results[0], colors[0], label=logistic_method)
    ax.plot(range(min_species, max_species + 1, step), results[1], colors[1], label=CNN_method)
    ax.legend()
    plt.savefig('log-cnn-comp.pdf')
    plt.show()
    



#----------------Execution starts here--------------#

#feature_extraction_showcase('')

# Logistic regression

X_train_f, X_val_f, y_train, y_val, label_encoder, label_binarizer = data_preparation(num_species=7, seed=103)
X_train, X_val = generate_feature_matrix(X_train_f, X_val_f, 'mfcc')

logistic_learn(X_train, y_train, X_val, y_val)


# Warning, reduce the parameters unless you want to leave running overnight
#logistic_training_montage(max_species=82, min_species=2, step=2, n_methods=7, seed=933)


#logistic_CNN_comparison(20, 3, 1, CNN_architecture='CNN_D128', seed=853)


# CNN testing

#num_spec = 8

#X_train_f, X_val_f, y_train_c, y_val_c, label_encoder, label_binarizer = data_preparation(num_species=num_spec, seed=392)
#X_train_c, X_val_c = generate_feature_matrix(X_train_f, X_val_f, 'stft')

#X_train_c = np.array([feature.reshape(128, 128, 1) for feature in X_train_c])
#X_val_c = np.array([feature.reshape(128, 128, 1) for feature in X_val_c])

#y_train_c = label_binarizer.transform(y_train_c)
#y_val_c = label_binarizer.transform(y_val_c)

#CNN_learn(X_train_c, y_train_c, X_val_c, y_val_c, (128, 128, 1), num_spec, 60, 'CNN_D64')



