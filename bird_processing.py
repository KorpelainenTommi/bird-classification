
# Multicore boosted signal processing
import os
import sys
import numpy as np
import librosa as lr
import audioread.ffdec
import multiprocessing as mp
from subprocess import Popen, PIPE
from scipy.signal import find_peaks


#--------------------Parameters---------------------#

# Change this to reduce cpu load
cores = mp.cpu_count()

# Resample audio to a common rate when opening
resampling = 22050

# Constant for peak finding algorithm
binary_filter_window = 150

# Window size to cut in seconds
window_size = 2

# Path to audio files
audio_path = 'xeno-canto/xeno-canto-ca-nv/'

# FFT window length
fft_window = 1024

# Targets for fitting the audio features to the shape of the 2D CNN
# Don't change these
component_target = 128
sample_target = 128

# Calculate hop_length needed to fit targets
hop_length_target = resampling * window_size // sample_target + 1

# Normalize outputs
# For each feature, center at 0 and divide by std deviation
normalize_features = True

#---------------------------------------------------#




#-----------------Signal processing-----------------#


# IMPORTANT requires access to ffmpeg in order to work
# Either add it to PATH or provide it in the root folder
def read_mp3(filepath):
    f = audioread.ffdec.FFmpegAudioFile(audio_path+filepath)
    return lr.load(f, sr=resampling if resampling > 0 else f.samplerate)


# Normalize audio to [-1, 1]
def audio_normalize(data):
    return lr.util.normalize(data, axis=0, fill=True)


# Downsample audio to N samples
def audio_downsample(data, samples):
    d = int(np.floor(len(data)/samples))
    vals = np.arange(0, d*samples, d)
    return np.array([data[i:(i+d)][0] for i in vals])



# Quantize audio to remove noise for peak finding
def audio_binary_filter(data, samples):
    d = int(np.floor(len(data)/samples))
    vals = np.arange(0, d*samples, samples)
    filtered = np.array([np.max(data[i:(i+samples)])*np.ones((samples)) for i in vals])    
    return filtered.flatten()



# Given a signal, identify the central area via analyzing peak amplitudes
def peak_finder(data):

    # Filter the signal before peak finding
    peaks = []
    sampling = 2*len(data) // binary_filter_window
    reduced = audio_binary_filter(data, sampling)

    # Initial treshold to consider peaks
    treshold = np.mean(reduced) + 1*np.std(reduced)

    # Find a minimum of 3 peaks by continually reducing the threshold
    while len(peaks) < 3:
        peaks, properties = find_peaks(reduced, height=(treshold, np.max(reduced)))
        treshold -= 10

    return peaks[len(peaks) // 2]



# Select an appropriate window from the recording
def audio_window(data, sampling_rate):
    
    window = window_size * sampling_rate

    # pad with mean value if we are too short
    # although this should never be the case
    if len(data) < window:
        mean = np.mean(data)
        return np.append(data, mean*np.ones((window-len(data))))
    elif len(data) == window:
        return data
    else:
        window_center = peak_finder(data)
        start = window_center - window // 2
        if start <= 0: return data[:window]
        
        end = window_center + window // 2
        if end >= len(data): return data[-window:]

        return data[start:end]




# Mel Frequency Cepstral Coefficients

def audio_mfcc(data, sampling_rate):
    return lr.feature.mfcc(y=data, sr=sampling_rate, n_fft=fft_window, n_mfcc=component_target, hop_length=hop_length_target)

def audio_melspectrogram(data, sampling_rate):
    return lr.feature.melspectrogram(y=data, sr=sampling_rate, n_fft=fft_window, n_mels=component_target, hop_length=hop_length_target)


# Short Time Fourier Transform
def audio_stft(data):
    return np.abs(lr.stft(y=data, n_fft=fft_window // 4 - 1, hop_length=hop_length_target))

# Chromagram
def audio_chroma(data, sampling_rate):
    return lr.feature.chroma_stft(y=data, sr=sampling_rate, n_fft=fft_window*4, n_chroma=component_target, hop_length=hop_length_target)

# Spectral centroid
def audio_centroid(data, sampling_rate):
    return lr.feature.spectral_centroid(y=data, sr=sampling_rate, hop_length=hop_length_target)

# Spectral bandwidth
def audio_spectral_band(data, sampling_rate):
    return lr.feature.spectral_bandwidth(y=data, sr=sampling_rate, hop_length=hop_length_target)

#---------------------------------------------------#




#----------------Feature extraction-----------------#

# Preprocessing before feature extraction
def preprocessing(filename):

    data, sr = read_mp3(filename)

    # Compress sound and select a constant size window
    norm_data = lr.util.normalize(data, axis=0, fill=True)
    return audio_window(norm_data, sr)

# Normalize output to 
def normalize(data):
    return (data - data.mean()) / (data.std() + 1e-8)


# Generate features by downsampling original audio vector
# Downsample to 256 => 16x16
def feature_downsample(filename):
    data = audio_downsample(preprocessing(filename), 256)
    return normalize(data) if normalize_features else data

# Generate features by taking different audio metrics

def feature_mfcc(filename):
    data = audio_mfcc(preprocessing(filename), resampling)
    return normalize(data) if normalize_features else data

def feature_chroma(filename):
    data = audio_chroma(preprocessing(filename), resampling)
    return normalize(data) if normalize_features else data

def feature_melspec(filename):
    data = audio_melspectrogram(preprocessing(filename), resampling)
    return normalize(data) if normalize_features else data

def feature_stft(filename):
    data = audio_stft(preprocessing(filename))
    return normalize(data) if normalize_features else data

def feature_centroid(filename):
    data = audio_centroid(preprocessing(filename), resampling)    
    return normalize(data) if normalize_features else data

def feature_bandwidth(filename):
    data = audio_spectral_band(preprocessing(filename), resampling)
    return normalize(data) if normalize_features else data

# Combine spectral components to form feature tensors of shape (N x N x 3)
def feature_spectral_tensor(filename):
    windowed = preprocessing(filename)
    c1 = audio_stft(windowed)
    c2 = audio_melspectrogram(windowed, resampling)
    c3 = audio_chroma(windowed, resampling)
    
    c1 = normalize(c1) if normalize_features else c1
    c2 = normalize(c2) if normalize_features else c2
    c3 = normalize(c3) if normalize_features else c3

    tensor = np.empty(shape=(component_target, sample_target, 3))
    for i in range(component_target):
        for j in range(sample_target):
            tensor[i][j][0] = c1[i][j]
            tensor[i][j][1] = c2[i][j]
            tensor[i][j][2] = c3[i][j]
    return tensor

# Combine time domain components to form tensors of shape (16, 8, 3)
def feature_time_tensor(filename):
    windowed = preprocessing(filename)
    c1 = audio_downsample(windowed, 128).reshape(16, 8)
    c2 = audio_centroid(windowed, resampling).reshape(16, 8)
    c3 = audio_spectral_band(windowed, resampling).reshape(16, 8)

    c1 = normalize(c1) if normalize_features else c1
    c2 = normalize(c2) if normalize_features else c2
    c3 = normalize(c3) if normalize_features else c3

    tensor = np.empty(shape=(16, 8, 3))
    for i in range(16):
        for j in range(8):
            tensor[i][j][0] = c1[i][j]
            tensor[i][j][1] = c2[i][j]
            tensor[i][j][2] = c3[i][j]
    return tensor


#---------------------------------------------------#





#-----------------Multiprocessing-------------------#

# Perform feature extraction accelerated by multiprocessing
# Call this function to spawn a child process that further
# distributes work among cpu cores
def extract_features(X_filenames, method):
    with Popen('python bird_processing.py', cwd=os.getcwd(), stdin=PIPE, stdout=PIPE, universal_newlines=True) as p:
        # Send data
        with p.stdin as pipe:
            pipe.write(f'{method}\n')
            for filename in X_filenames:
                pipe.write(f'{filename}\n')

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

    input = []
    output = []
    for line in sys.stdin:
        input.append(line.strip())

    method = input[0]
    input.pop(0)

    # Choose feature generation method
    generate_features = (
        {
            'mfcc': feature_mfcc,
            'stft': feature_stft,
            'chroma': feature_chroma,
            'melspec': feature_melspec,
            'centroid': feature_centroid,
            'bandwidth': feature_bandwidth,
            'downsampling': feature_downsample,
            'spectral-tensor': feature_spectral_tensor,
            'time-tensor': feature_time_tensor
        }[method]
    )

    print('Processing...')
    sys.stdout.flush()
    with mp.Pool(cores) as p:
        output = p.map(generate_features, input)
    print('Done')


    # Send back data:
    for feature in output:
        vec = feature.flatten()
        for v in vec:
            print(v, end=' ')
        print()
        sys.stdout.flush()
    
