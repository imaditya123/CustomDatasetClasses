import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from sklearn.model_selection import train_test_split
import spacy
import pandas as pd
from tqdm.notebook import tqdm

class TextClassificationDataset(Dataset):
    def __init__(self, csv_file, max_length, text_col, label_col,encoding="utf-8",n_records=1):
      """ 
        Initializes the TextDataset instance.

        Args:
            csv_file (str): Path to the CSV file containing the dataset.
            max_length (int): Maximum sequence length for tokenization.
            text_col (str): Name of the column containing text data.
            label_col (str): Name of the column containing labels.
            encoding (str, optional): File encoding. Defaults to "utf-8".
            n_records (float, optional): Fraction of the dataset to include. Defaults to 1.
      """
      self.nlp = spacy.load("en_core_web_sm")
      self.data = pd.read_csv(csv_file,encoding=encoding)
      self.len=int(len(self.data)*n_records)
      self.data=self.data.iloc[0:self.len]
      self.label_encoder={k:i for i,k in enumerate(list(set(self.data[label_col])))}
      self.data['new_labels']=[self.label_encoder[label] for label in self.data[label_col]]
      self.max_length = max_length
      self.text_col = text_col
      self.label_col = label_col
      self.vocab,self.vocab_decoder = self.build_vocab()

    def __len__(self):
      return len(self.data)

    def __getitem__(self, idx):
      text = self.data.iloc[idx][self.text_col]
      label = self.data.iloc[idx]['new_labels']

      # Tokenize the text using spaCy
      tokens = [token.text for token in self.nlp(text)]

      # Convert tokens to indices using the vocab
      # we are sending the constant sequence of tokens in the NN 
      # so that we are either padding it if the sequnce is small or truncate if the sequence is large
      token_indices = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens[:self.max_length]]
      padding_length = self.max_length - len(token_indices)
        
      # Pad or truncate to the required max length
      token_indices += [self.vocab["<PAD>"]] * padding_length

      # Convert to tensors
      input_ids = torch.tensor(token_indices, dtype=torch.long)
        
      return input_ids, torch.tensor(label, dtype=torch.long)

    def build_vocab(self,max_vocab_size=10000):

      texts=self.data[self.text_col]

      word_freq = {}
      for text in tqdm(texts):
          tokens = [token.text for token in self.nlp(text)]
          for token in tokens:
              word_freq[token] = word_freq.get(token, 0) + 1

      # Sort by frequency and take the most common words
      sorted_vocab = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_vocab_size]
      
      # Create vocab dict
      vocab = {word: idx for idx, (word, _) in enumerate(sorted_vocab, start=2)}
      vocab["<PAD>"] = 0  # Padding token
      vocab["<UNK>"] = 1  # Unknown token
      vocab_decoder={idx:word for word,idx in vocab.items()}
      return vocab,vocab_decoder
    
    def decode_labels(self,x):
      label_decoder={i:k for k,i in self.label_encoder.items()}
      return label_decoder[int(x[0])] if len(x)==1 else [label_decoder(int(xi)) for xi in x.squeeze(0)]

    def decode_tokens(self,x):
      print(x.squeeze(0).shape)
      return [self.vocab_decoder[int(xi)] for xi in x.squeeze(0)]
      

