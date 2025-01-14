import os
import argparse
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torchvision import transforms
from kaggle.api.kaggle_api_extended import KaggleApi

# Custom dataset imports
from custom_datasets.text_ner import TextNERDataset
from custom_datasets.text_classification import TextClassificationDataset
from custom_datasets.image_classification import ImageClassificationDataset
from custom_datasets.audio_classification import AudioClassificationDataset


def main():
    """
    Main function to handle the training or evaluation of different models.
    Depending on the provided arguments, it will download the dataset, 
    create the dataset objects, and display or print relevant information.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Script to train or evaluate a model")
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'evaluate'], 
        required=True,
        help="Specify the mode: 'train' for training the model, 'evaluate' for evaluating it"
    )
    parser.add_argument(
        '--dataset', 
        type=str, 
        choices=['ImageClassification', 'TextClassification', 'SoundClassification', 'TextNER'], 
        required=True,
        help="Specify the dataset to use: 'ImageClassification', 'TextClassification', 'SoundClassification', 'TextNER'"
    )

    args = parser.parse_args()

    if args.mode == 'train':
        # Handle training for the different datasets
        handle_training(args.dataset)
    elif args.mode == 'evaluate':
        # Evaluate mode can be implemented later based on specific requirements
        print("Evaluation mode is not yet implemented for the selected dataset.")
    else:
        print(f"Invalid mode: {args.mode}. Use 'train' or 'evaluate'.")


def handle_training(dataset_type):
    """
    Handle the training process for the selected dataset type.
    
    :param dataset_type: str, The dataset type to handle ('ImageClassification', 'TextClassification', etc.)
    """
    if dataset_type == 'ImageClassification':
        train_image_classification()
    elif dataset_type == 'TextClassification':
        train_text_classification()
    elif dataset_type == 'SoundClassification':
        train_sound_classification()
    elif dataset_type == 'TextNER':
        train_text_ner()
    else:
        print(f"Invalid dataset type: {dataset_type}.")


def train_image_classification():
    """
    Train the model using the Image Classification dataset.
    """
    path = download_kaggle_dataset("phucthaiv02/butterfly-image-classification", "data/")
    
    image_dataset = ImageClassificationDataset(
        data_dir=os.path.join(path, 'train'),
        annotations_file=os.path.join(path, 'Training_set.csv'),
        transforms=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    )

    train_dataloader = DataLoader(image_dataset, batch_size=1, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    npimg = train_features.squeeze(0).numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.title(image_dataset.convert_itoc(train_labels))
    plt.axis("off")
    plt.savefig("Image_Classification_1.png")


def train_text_classification():
    """
    Train the model using the Text Classification dataset.
    """
    path = download_kaggle_dataset("datatattle/covid-19-nlp-text-classification", "data/")
    
    dataset = TextClassificationDataset(
        os.path.join(path, 'Corona_NLP_train.csv'), 
        50, 
        'OriginalTweet', 
        'Sentiment',
        encoding='latin1',
        n_records=0.001  # Adjust the number of records for testing
    )

    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    input_ids, labels = next(iter(train_loader))
    print(f"{dataset.decode_tokens(input_ids)=}\n{dataset.decode_labels(labels)=}")


def train_sound_classification():
    """
    Train the model using the Sound Classification dataset.
    """
    path = download_kaggle_dataset("warcoder/cats-vs-dogs-vs-birds-audio-classification", "data/")
    
    sound_dataset = AudioClassificationDataset(
        data_dir=os.path.join(path, 'Animals'),
        sample_rate=16000
    )

    train_loader = DataLoader(sound_dataset, batch_size=1, shuffle=True)
    audio_data, labels = next(iter(train_loader))
    print(f"{audio_data.shape=}\n{sound_dataset.decode_labels(labels)}")


def train_text_ner():
    """
    Train the model using the Named Entity Recognition dataset.
    """
    texts = ["John lives in New York", "Alice works at Google"]
    annotations = [
        {'entities': [(0, 4, 1), (14, 22, 2)]},  # "John" -> 1, "New York" -> 2
        {'entities': [(0, 5, 1), (14, 20, 3)]}   # "Alice" -> 1, "Google" -> 3
    ]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text_dataset = TextNERDataset(texts, annotations, tokenizer)
    sample = text_dataset[0]
    print(f"Sample from NER Dataset: {sample}")


def download_kaggle_dataset(dataset_name, download_path):
    """
    Downloads a dataset from Kaggle given the dataset name.

    :param dataset_name: str, The Kaggle dataset name (e.g.,'phucthaiv02/butterfly-image-classification')
    :param download_path: str, The path where the dataset should be saved
    :return: str, The path to the downloaded dataset
    """
    api = KaggleApi()
    api.authenticate()

    downloaded_path = os.path.join(download_path, dataset_name.split("/")[-1])

    # Ensure the download path exists
    if  os.path.exists(downloaded_path):
        return downloaded_path
    os.makedirs(downloaded_path)
    
    print(f"Downloading {dataset_name} to {downloaded_path}...")
    api.dataset_download_files(dataset_name, path=downloaded_path, unzip=True)
    
    return downloaded_path


if __name__ == '__main__':
    main()
