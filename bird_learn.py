
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Flatten, Dense, LayerNormalization, Dropout
from keras.losses import CategoricalCrossentropy, CategoricalHinge

import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score


# If a method is worse than this,
# then we have definitely failed
def random_guess_accuracy(y_train, y_val):
    n_classes = len(np.unique(np.concatenate((y_train, y_val))))
    return 1.0 / n_classes



# Different CNN architectures

# Shallow 32
def CNN_S32(input_shape, n_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    return model

# Shallow 64
def CNN_S64(input_shape, n_classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    return model

# Shallow 128
def CNN_S128(input_shape, n_classes):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    return model

# Deep triple convolution 32
def CNN_D32(input_shape, n_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))    
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    return model

# Deep triple convolution 64
def CNN_D64(input_shape, n_classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    return model

# Deep triple convolution 128
def CNN_D128(input_shape, n_classes):
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    return model

# 128 => 32 convolvers, 7 => 3 kernel
def CNN_F128(input_shape, n_classes):
    model = Sequential()
    model.add(Conv2D(128, (7, 7), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), padding='same'))    
    model.add(LayerNormalization())
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    return model

# Deep 32, 5 convolutions
def CNN_T32(input_shape, n_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())

    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())

    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())

    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())

    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(LayerNormalization())
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    return model

# Deep 128, 5 convolutions
def CNN_T128(input_shape, n_classes):
    model = Sequential()

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())

    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())

    model.add(Dropout(0.5))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())

    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())

    model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(LayerNormalization())
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    return model

# Deep 128, 5x5 kernel, no dropout
def CNN_E128(input_shape, n_classes):
    model = Sequential()
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same', input_shape=input_shape))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(LayerNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(LayerNormalization())
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    return model

# 1D Convolution
def CNN_E1024(input_shape, n_classes):
    model = Sequential()
    model.add(Conv1D(1024, 3, activation='relu', padding='same', input_shape=input_shape))
    model.add(LayerNormalization())
    model.add(MaxPooling1D(2, padding='same'))
    model.add(LayerNormalization())
    model.add(Conv1D(1024, 3, activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling1D(2, padding='same'))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    model.add(Conv1D(1024, 3, activation='relu', padding='same'))
    model.add(LayerNormalization())
    model.add(MaxPooling1D(2, padding='same'))
    model.add(LayerNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    model.add(Dense(n_classes, activation='softmax'))
    model.summary()
    return model

def CNN_learn(X_train, y_train, X_val, y_val, shape, n_classes, n_epochs=20, model_name='CNN_S32', plots=True, name=''):
    fname = f'{model_name}_{n_classes}_{n_epochs}' if not name else name

    architectures = {
        'CNN_S32': CNN_S32,
        'CNN_S64': CNN_S64,
        'CNN_S128': CNN_S128,
        'CNN_D32': CNN_D32,
        'CNN_D64': CNN_D64,
        'CNN_D128': CNN_D128,
        'CNN_F128': CNN_F128,
        'CNN_T32': CNN_T32,
        'CNN_T128': CNN_T128,
        'CNN_E128': CNN_E128,
        'CNN_E1024': CNN_E1024
    }

    # Loss functions CategoricalHinge(), CategoricalCrossentropy(from_logits=False)

    model = architectures[model_name](shape, n_classes)
    model.compile(optimizer='adam', loss=CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
    training = model.fit(X_train, y_train, epochs=n_epochs, validation_data=(X_val, y_val))


    loss_values = training.history['loss']
    acc_values = training.history['accuracy']

    val_loss_values = training.history['val_loss']
    val_acc_values = training.history['val_accuracy']

    best_acc = 0.0
    best_epoch = 0
    for i in range(len(acc_values)):
        a = acc_values[i] * val_acc_values[i]
        if a > best_acc:
            best_acc = a
            best_epoch = i + 1

    accuracy = val_acc_values[best_epoch - 1]
    rand_acc = 1.0 / n_classes


    # Write metrics to file
    with open(f'metrics-{fname}.txt', mode = 'w') as out:
        out.write(f'Best achieved accuracy at epoch={best_epoch}:\n')
        out.write(f'{accuracy} {rand_acc} {accuracy - rand_acc}\n')
        out.write(f'Overview of training:\n')
        for i in range(len(acc_values)):
            out.write(f'{loss_values[i]} ')
            out.write(f'{acc_values[i]} ')
            out.write(f'{val_loss_values[i]} ')
            out.write(f'{val_acc_values[i]} ')
            out.write('\n')


    if plots:
        #val_loss_avg = np.convolve(val_loss_values, np.ones(25)/25, mode='same')
        #val_acc_avg = np.convolve(val_acc_values, np.ones(25)/25, mode='same')
        epochs = range(1,n_epochs+1)
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(15,5))
        ax1.plot(epochs,loss_values,'bo',label='Training Loss')
        ax1.plot(epochs,val_loss_values,'orange', label='Validation Loss')
        #ax1.plot(epochs,val_loss_avg,'red', label='Validation loss average')
        ax1.set_title('Training and validation loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax2.plot(epochs,acc_values,'bo', label='Training accuracy')
        ax2.plot(epochs,val_acc_values,'orange',label='Validation accuracy')
        #ax2.plot(epochs,val_acc_avg,'red',label='Validation accuracy average')
        ax2.set_title('Training and validation accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        plt.savefig(f'{fname}.pdf')
        plt.show()
    
    return {
        'accuracy': accuracy,
        'rand_accuracy': rand_acc,
        'improvement': (accuracy-rand_acc)
    }



def logistic_learn(X_train, y_train, X_val, y_val, prints=True):

    if prints: print(f'Training logistic regressor for X[{X_train.shape}] => y[{y_train.shape}]')

    # SAG solver suffered from convergence issues for some features
    # 
    model = LogisticRegression(solver='liblinear', max_iter=1000)
    model.fit(X_train, y_train)
    if prints: print(f'Validating on set X[{X_val.shape}] => y[{y_val.shape}]')
    y_pred = model.predict(X_val)

    accuracy = accuracy_score(y_val, y_pred)
    rand_acc = random_guess_accuracy(y_train, y_val)

    # Use micro for multiclass scoring
    precision = precision_score(y_val, y_pred, average='micro')
    recall = recall_score(y_val, y_pred, average='micro')

    # Formula taken from scikit-learn.org
    f1 = 2 * (precision * recall) / (precision + recall)

    if prints:
        print('--- Model statistics ---')
        print()    
        print(f'Model accuracy:  {"%.3f" % (100.0*accuracy)}%')
        print(f'== Random guess: {"%.3f" % (100.0*rand_acc)}%')
        print(f'== Improvement:  {"%.3f" % (100.0*(accuracy-rand_acc))}%')
        print()
        print(f'Precision:       {"%.3f" % precision}')
        print(f'Recall:          {"%.3f" % recall}')
        print(f'F1 Score:        {"%.3f" % f1}')
        print('------------------------\n')

    # Return metrics
    return {
        'accuracy': accuracy,
        'rand_accuracy': rand_acc,
        'improvement': (accuracy-rand_acc),
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
