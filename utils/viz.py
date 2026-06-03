import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from sklearn.metrics import confusion_matrix

SR = 22050


def plot_waveform(y, sr=SR, ax=None, title=''):
    if ax is None: _, ax = plt.subplots(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title(title)
    return ax


def plot_spectrogram(spec, sr=SR, hop=512, ax=None, y_axis='mel', title=''):
    if ax is None: _, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(spec, sr=sr, hop_length=hop,
                                    x_axis='time', y_axis=y_axis, ax=ax)
    ax.set_title(title)
    plt.colorbar(img, ax=ax, format='%+2.0f dB')
    return ax


def plot_features(y, sr=SR, title=''):
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    librosa.display.waveshow(y, sr=sr, ax=ax[0, 0]); ax[0, 0].set_title('Waveform')
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(S, sr=sr, x_axis='time', y_axis='log', ax=ax[0, 1])
    ax[0, 1].set_title('STFT (dB)')
    M = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
    librosa.display.specshow(M, sr=sr, x_axis='time', y_axis='mel', ax=ax[1, 0])
    ax[1, 0].set_title('Mel-spectrogram')
    mf = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    librosa.display.specshow(mf, x_axis='time', ax=ax[1, 1]); ax[1, 1].set_title('MFCC')
    plt.suptitle(title); plt.tight_layout()
    return fig


def plot_class_distribution(labels, ax=None, title='Class distribution'):
    if ax is None: _, ax = plt.subplots(figsize=(8, 5))
    s = pd_value_counts(labels)
    sns.barplot(x=s.values, y=s.index, ax=ax)
    ax.set_title(title); ax.set_xlabel('count')
    return ax


def pd_value_counts(labels):
    import pandas as pd
    return pd.Series(labels).value_counts()


def plot_confusion(y_true, y_pred, class_names=None, normalize=True,
                   ax=None, title='Confusion matrix'):
    cm = confusion_matrix(y_true, y_pred,
                          normalize='true' if normalize else None)
    if ax is None: _, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=class_names, yticklabels=class_names,
                cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Pred'); ax.set_ylabel('True'); ax.set_title(title)
    return ax


def plot_training_curves(history, title=''):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history['train_loss'], label='train')
    ax[0].plot(history['val_loss'], label='val')
    ax[0].set_title(f'{title} loss'); ax[0].legend(); ax[0].set_xlabel('epoch')
    ax[1].plot(history['train_acc'], label='train')
    ax[1].plot(history['val_acc'], label='val')
    ax[1].set_title(f'{title} acc'); ax[1].legend(); ax[1].set_xlabel('epoch')
    plt.tight_layout()
    return fig
