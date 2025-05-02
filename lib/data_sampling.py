import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, raw_text:str, tokenizer:tiktoken.Encoding, max_context_length:int, stride:int) -> None:
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(raw_text)
        for i in range(0, len(token_ids) - max_context_length, stride):
            input_chunk_ids = token_ids[i:i+max_context_length]
            output_chunk_ids = token_ids[i+1:i+max_context_length+1]

            self.input_ids.append(torch.tensor(input_chunk_ids))
            self.target_ids.append(torch.tensor(output_chunk_ids))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(raw_text:str, batch_size:int, max_context_length:int, stride:int, num_workers:int):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(raw_text, tokenizer, max_context_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    return dataloader