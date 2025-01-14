import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
import numpy as np
from torchaudio import load as load_audio
import librosa


class AudioClassificationDataset(Dataset):
    def __init__(self, data_dir, annotations_file=None, sample_rate=None, filepath_col="files", label_col="labels"):
        """
        Dataset for Sound Data.

        Args:
            data_dir (str): Directory containing the audio files.
            annotations_file (str, optional): Path to the annotations file for supervised tasks.
            sample_rate (int, optional): Desired sample rate for audio data.
            filepath_col (str, optional): Column name for file paths in the annotations file. Defaults to "files".
            label_col (str, optional): Column name for labels in the annotations file. Defaults to "labels".
        """
        # Initialize directory paths and configuration parameters
        self.data_dir = data_dir
        self.annotations_file = annotations_file
        self.sample_rate = sample_rate
        self.filepath_col = filepath_col
        self.label_col = label_col

        # Collect all audio file paths in the directory with valid extensions
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('wav', 'mp3'))]

        # Load annotations if provided, otherwise create a default annotations DataFrame
        if annotations_file:
            self.annotations = pd.read_csv(annotations_file)
        else:
            arr = []
            labels=os.listdir(self.data_dir)
            for label in labels:
                files = os.listdir(os.path.join(self.data_dir, label))
                for file in files:
                    arr.append((file, label))
            self.annotations = pd.DataFrame(arr, columns=["files", "labels"])

        # Encode labels into integers for model compatibility
        self.label_encoder = {k: i for i, k in enumerate(list(set(self.annotations[label_col])))}

    def __len__(self):
        """
        Returns the total number of annotated samples in the dataset.

        Returns:
            int: Number of annotated samples.
        """
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        Retrieves a single audio sample and its corresponding label by index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the audio tensor and the encoded label.
        """
        # Construct the full path to the audio file
        audio_path = os.path.join(self.data_dir, self.annotations.iloc[idx][self.label_col], self.annotations.iloc[idx][self.filepath_col])

        # Load the audio file and its sampling rate
        audio, sr = load_audio(audio_path)

        # Resample audio if a target sample rate is specified and differs from the original
        if self.sample_rate and self.sample_rate != sr:
            audio = librosa.resample(audio.numpy().squeeze(), sr, self.sample_rate)
            audio = torch.tensor(audio)

        # Retrieve and encode the label for the sample
        label = self.annotations.iloc[idx][self.label_col]
        return audio, self.label_encoder[label]

    def decode_labels(self, x):
        """
        Converts encoded labels back to their original string representations.

        Args:
            x (torch.Tensor): Tensor of encoded label(s).

        Returns:
            str or list: Decoded label(s) as a string or a list of strings.
        """
        # Create a decoder mapping from integers to label strings
        label_decoder = {i: k for k, i in self.label_encoder.items()}
        
        # Decode a single label or a batch of labels
        return label_decoder[int(x[0])] if len(x) == 1 else [label_decoder[int(xi)] for xi in x.squeeze(0)]
