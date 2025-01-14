import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class TextNERDataset(Dataset):
    def __init__(self, texts, annotations, tokenizer, max_length=512):
        """
        Args:
            texts (list of str): List of raw text samples.
            annotations (list of dict): List of annotations for each text, e.g., {'entities': [(start, end, label)]}.
            tokenizer: Tokenizer to convert text into tokens.
            max_length (int): Maximum sequence length for padding/truncation.
        """
        self.texts = texts
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        annotation = self.annotations[idx]

        # Tokenize the text
        tokens = self.tokenizer(
            text, 
            max_length=self.max_length, 
            truncation=True, 
            padding="max_length",
            return_offsets_mapping=True
        )
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        offsets = tokens['offset_mapping']

        # Create labels for tokens
        labels = [0] * len(input_ids)  # Default label: 0 (non-entity)
        for start, end, label in annotation['entities']:
            for i, (token_start, token_end) in enumerate(offsets):
                if token_start >= start and token_end <= end:
                    labels[i] = label  # Assign entity label

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }