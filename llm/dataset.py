import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class SynthDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_length=512):
        df = pd.read_parquet(parquet_path)
        df = df[df['language'] == 'en'].dropna(subset=['synthetic_answer'])
        self.texts = df['synthetic_answer'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encodings['input_ids'].squeeze(0)
        attention_mask = encodings['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100 
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def get_dataloader(parquet_path, tokenizer, batch_size=1, max_length=512):
    dataset = SynthDataset(parquet_path, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)