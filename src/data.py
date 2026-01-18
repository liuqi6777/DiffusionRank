from dataclasses import dataclass
import json
import math
from typing import Sequence, Any

import torch
from torch import Tensor
from torch.utils.data import Dataset
import transformers


class SFTDataset(Dataset):
    listwise_prompt = """Given a query and {num} documents indicated by a character identifier, rank the documents from most relevant to least relevant to the query.

You should output a ranking using the document identifier, from most relevant to least relevant, separated by spaces.

Query: {query}

Documents:
{documents}
"""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        with open(data_path, "r") as f:
            self.raw_data = [json.loads(line) for line in f]
        print(f"Loaded {len(self.raw_data)} examples from {data_path}")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> dict[str, Any]:
        documents = self.raw_data[i]["document"]
        messages = [
            {
                "role": "user",
                "content": self.listwise_prompt.format(
                    num=len(documents),
                    query=self.raw_data[i]["query"],
                    documents="\n".join([f"Document {chr(ord('A') + i)}: {doc}" for i, doc in enumerate(documents)])
                )
            },
            {
                "role": "assistant",
                "content": "Ranking (most to least relevant):" + "".join([f" {chr(ord('A') + x - 1)}" for x in self.raw_data[i]["ranking"]])
            }
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            truncation=True,
            max_length=8192,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True
        ).input_ids[0]
        prompt_length = len(self.tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, return_dict=True).input_ids
        )
        ret = dict(
            input_ids=input_ids,
            prompt_lengths=prompt_length,
        )

        return ret


@dataclass
class SFTDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, Tensor]:
        input_ids, prompt_lengths = tuple([instance[key] for instance in instances] for key in ("input_ids", "prompt_lengths"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        prompt_lengths = torch.tensor(prompt_lengths, dtype=torch.long)
        batch = dict(
            input_ids=input_ids,
            prompt_lengths=prompt_lengths,
        )
        return batch


class MultiDocLogitsDataset(Dataset):
    listwise_prompt = """Given a query and {num} documents indicated by a numeric identifier, determine whether each document is relevant to the query. The relevance score should be either 0 (not relevant) or 1 (relevant).

Query: {query}

Documents:
{documents}
"""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        with open(data_path, "r") as f:
            self.raw_data = [json.loads(line) for line in f]
        print(f"Loaded {len(self.raw_data)} examples from {data_path}")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> dict[str, Any]:
        documents = self.raw_data[i]["document"]
        messages = [
            {
                "role": "user",
                "content": self.listwise_prompt.format(
                    num=len(documents),
                    query=self.raw_data[i]["query"],
                    documents="\n".join([f"Document {i}: {doc}" for i, doc in enumerate(documents)])
                )
            },
            {
                "role": "assistant",
                "content": "\n".join([f"Document {i}: <|mdm_mask|>" for i in range(len(documents))])
            }
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            truncation=True,
            max_length=8192,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True
        ).input_ids[0]
        ret = dict(
            input_ids=input_ids,
            ranking=self.raw_data[i]["ranking"],
        )

        return ret


@dataclass
class MultiDocLogitsDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, Tensor]:
        input_ids, ranking = tuple([instance[key] for instance in instances] for key in ("input_ids", "ranking"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        ranking = torch.tensor(ranking, dtype=torch.long) - 1  # make zero-indexed
        batch = dict(
            input_ids=input_ids,
            ranking=ranking,
        )
        return batch


class PointwiseDataset(Dataset):
    prompt = """Given a query and a document, determine whether the document is relevant to the query. The relevance score should be either 0 (not relevant) or 1 (relevant).

Query: {query}

Document:
{doc}
"""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        with open(data_path, "r") as f:
            self.raw_data = [json.loads(line) for line in f]
        print(f"Loaded {len(self.raw_data)} examples from {data_path}")

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> dict[str, Any]:
        documents = self.raw_data[i]["document"]
        messages = [
            [
                {"role": "user", "content": self.prompt.format(query=self.raw_data[i]["query"], doc=documents[j])},
                {"role": "assistant", "content": "<|mdm_mask|>"}
            ] for j in range(len(documents))
        ]
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            padding=True,
            truncation=True,
            max_length=1024,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True
        ).input_ids
        ret = dict(
            input_ids=input_ids,
            ranking=self.raw_data[i]["ranking"],
        )

        return ret


@dataclass
class PointwiseDataCollator:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[dict]) -> dict[str, Tensor]:
        input_ids, ranking = tuple([instance[key] for instance in instances] for key in ("input_ids", "ranking"))
        max_len = max(t.size(1) for t in input_ids)
        padded = [
            torch.nn.functional.pad(t, (0, max_len - t.size(1)), value=self.tokenizer.pad_token_id)
            for t in input_ids
        ]
        input_ids = torch.cat(padded, dim=0)
        ranking = torch.tensor(ranking, dtype=torch.long) - 1  # make zero-indexed
        batch = dict(
            input_ids=input_ids,
            ranking=ranking,
        )
        return batch


def make_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_path: str,
    method: str = "sft",
) -> dict:
    if method == "sft":
        train_dataset = SFTDataset(
            data_path,
            tokenizer=tokenizer,
        )
        data_collator = SFTDataCollator(tokenizer=tokenizer)
    elif method == "logits":
        train_dataset = MultiDocLogitsDataset(
            data_path,
            tokenizer=tokenizer,
        )
        data_collator = MultiDocLogitsDataCollator(tokenizer=tokenizer)
    elif method == "pointwise":
        train_dataset = PointwiseDataset(
            data_path,
            tokenizer=tokenizer,
        )
        data_collator = PointwiseDataCollator(tokenizer=tokenizer)
    else:
        raise ValueError(f"Unknown method: {method}")
    return dict(
        train_dataset=train_dataset,
        data_collator=data_collator
    )
