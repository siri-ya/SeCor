import torch
from torch.utils.data import Dataset, DataLoader
import json
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# lightgcn
class POI_dataset(Dataset):
    def __init__(self, json_file, tokenizer, cutoff_len=2048) -> None:
        super(POI_dataset, self).__init__()
        all_data = json.load(open(json_file,'r'))
        self.preprocess(all_data, tokenizer, cutoff_len)
        
    def preprocess(self, all_data, tokenizer, cutoff_len):
        self.data = []
        for data_point in all_data:
            full_prompt = f"""{data_point["instruction"]}

### Input:
{data_point["input"]}
"""
            result = tokenizer(
                full_prompt,
                truncation=True,
                max_length=cutoff_len,
                padding=False,
                return_tensors=None,
            )
            if (
                result["input_ids"][-1] != tokenizer.eos_token_id
                and len(result["input_ids"]) < cutoff_len
            ):
                result["input_ids"].append(tokenizer.eos_token_id)
                result["attention_mask"].append(1)

            data = {
                    "input_ids": result["input_ids"],
                    "attention_mask": result["attention_mask"],
                    "labels": data_point["labels"],
                    "uid": data_point["uid"],
                }
            if "known_labels" in data_point:
                data.update({"known_labels": data_point["known_labels"]})
                data.update({"labels":data_point["labels"]})

            self.data.append(data)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, index):
        return self.data[index]


# llm
class description_collator(object):
    def __init__(self, description_file, tokenizer, cutoff_len,  pad_to_multiple_of=8, return_tensors="pt", padding=True) -> None:
        super(description_collator, self).__init__()
        self.tokenizer = tokenizer
        self.cutoff_len = cutoff_len
        self.padding = padding
        self.max_length = 1024
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        description = np.load(description_file,allow_pickle=True).item()
        self.desription_data = []

        for poi_description in description.values():
            self.desription_data.append(self.get_token(poi_description))

    def get_token(self, trans_string):
        result = self.tokenizer(
            trans_string,
            truncation=True,
            max_length=self.cutoff_len,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.cutoff_len
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        return result

    def __call__(self, features):
        num = len(features)
        to_padding = [{'input_ids':f['input_ids'], 'attention_mask': f['attention_mask']} for f in features]
        batch = self.tokenizer.pad(
            to_padding,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch['uids'] = torch.tensor([f["uid"] for f in features])
        batch['labels'] = torch.tensor([f["labels"] for f in features]).unsqueeze(1)
        batch['neg_labels'] = torch.tensor([f["neg_labels"] for f in features])

        poi_ids = torch.concat([batch['labels'], batch['neg_labels']], dim=1).flatten()
        poi_padding = [self.desription_data[poi_id] for poi_id in poi_ids]
        poi_batch = self.tokenizer.pad(
            poi_padding,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch['poi_input_ids'] = poi_batch['input_ids']
        batch['poi_attention_mask'] = poi_batch['attention_mask']

        return batch


class Mycollator(object):
    def __init__(self, tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = 1024
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, features):
        to_padding = [{'input_ids':f['input_ids'], 'attention_mask': f['attention_mask']} for f in features]
        batch = self.tokenizer.pad(
            to_padding,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch['uids'] = torch.tensor([f["uid"] for f in features])
        batch['labels'] = torch.tensor([f["labels"] for f in features])
        return batch