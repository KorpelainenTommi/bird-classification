
import numpy as np
import librosa as lr
import matplotlib.pyplot as plt
from librosa.display import specshow
from matplotlib.ticker import MaxNLocator





# Plot audio in the time domain
def audio_plot(data, sampling_rate, title='Audio'):
    fig, ax = plt.subplots()
    ax.xaxis.set_major_formatter(lambda tval, tpos : '%.1f' % (1.0 * tval / sampling_rate))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    ax.set_title(title)
    ax.plot(data)
    plt.show()



# Plot spectrograms

def spectral_power_plot(data, sampling_rate, title='Audio'):
    fig, ax = plt.subplots()
    img = specshow(lr.power_to_db(data, ref=np.max), x_axis='time', y_axis='mel', sr=sampling_rate, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0F dB')
    ax.set_title(title)
    plt.show()

def spectral_amplitude_plot(data, sampling_rate, title='Audio'):
    fig, ax = plt.subplots()
    img = specshow(lr.amplitude_to_db(data, ref=np.max), x_axis='time', y_axis='mel', sr=sampling_rate, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0F dB')
    ax.set_title(title)
    plt.show()



# Special plots

def mfcc_plot(data, sampling_rate, title='Audio'):
    fig, ax = plt.subplots()
    img = specshow(data, x_axis='time', sr=sampling_rate, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0F dB')
    ax.set_title(title)
    plt.show()

def centroid_plot(data, sampling_rate, title='Audio'):
    fig, ax = plt.subplots()
    times = lr.times_like(data, sr=sampling_rate)
    ax.xaxis.set_major_formatter(lambda tval, tpos : '%.1f' % (1.0 * tval / sampling_rate))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_yscale('log')
    plt.xlabel("Time")
    plt.ylabel("Hz")
    ax.plot(times, data.T)
    ax.set_title(title)
    plt.show()
