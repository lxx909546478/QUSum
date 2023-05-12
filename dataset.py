"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from dataclasses import replace
import datetime
import json
import math
from concurrent.futures.process import ProcessPoolExecutor
from os import remove, path, makedirs

import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizerBase
import numpy as np
from sentence_transformers import SentenceTransformer


def cos_sim(emb1, emb2):
    return torch.cosine_similarity(torch.Tensor(emb1), torch.Tensor(emb2))

class MultiEncoderDataset(Dataset):
    
    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizerBase,
        chunk_size: int,
        max_num_chunks: int,
        max_target_length: int,
        stride: bool = False,
        pad: bool = True,
        num_samples: int = None,
        verbose: bool = False,
        ignore_pad_token_for_loss: bool = True,
        max_workers=1,
        chunk_seq:bool = False
    ):

        self.tokenizer = tokenizer
        self.chunk_tokenizer = ChunkTokenizer(
            tokenizer,
            chunk_size,
            max_num_chunks,
            stride,
            pad,
            chunk_seq
        )
        model_name = "all-roberta-large-v1"
        self.model = SentenceTransformer(model_name, device="cuda:0")
        self.chunk_size = chunk_size
        self.max_num_chunks = max_num_chunks
        self.max_target_length = max_target_length
        self.num_samples = num_samples
        self.verbose = verbose
        self.stride=stride
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.dataset=None
        self.chunk_seq=chunk_seq
        self._encode_data(data_path, max_workers)
        self.model = None


    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, index):
        return self.encodings[index]

    def _encode_data(self, file_path, max_workers):
        _, file_name = path.split(file_path)
        # if self.stride:
        #     if self.chunk_seq:
        #         dir_name = "stride/seq"
        #     else:
        #         dir_name = "stride/noseq"
        # else:
        #     dir_name = "nostride"
        dir_name = "max"
        cache_dir = f"./data/qmsum/cache/{self.chunk_size}/{dir_name}"
        cached_file = path.join(cache_dir, file_name)
        if not path.exists(cached_file) or "preprocessed" not in file_path:
            with open(file_path) as f:
                if max_workers == 1:
                    encodings = list(map(self._process_line, enumerate(f)))
                else:
                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        encodings = executor.map(self._process_line, enumerate(f))
            self.encodings = [enc for enc in encodings if enc is not False]
            if self.num_samples is not None:
                assert self.num_samples == len(self.encodings)
            if not path.exists(cache_dir):
                makedirs(cache_dir)
            torch.save(self.encodings, path.join(cache_dir, file_name))
        else:
            self.encodings = torch.load(path.join(cache_dir, file_name))
            

    def _process_line(self, index_line):
        i, line = index_line
        if i % 100 == 0:
            print('Processed', i, 'records', datetime.datetime.now())
        if self.num_samples is not None and i >= self.num_samples:
            return False
        data = json.loads(line)
        turns = data['source']
        source = ' '.join(turns)
        query = data['query']
        target = data['target']

        encoding = self._encode_example(
            source,
            turns,
            target,
            query,
        )
        if self.verbose and i == 0:
            print('First record in dataset:')
            for token_ids in encoding['input_ids']:
                print()
                print(self.tokenizer.decode(token_ids))
        return encoding

    def _encode_example(self, source, turns, target, query=None):

        output = self.chunk_tokenizer(turns, query)
        source_ids = output['input_ids']
        source_attention_mask = output['attention_mask']
        turn_ids = output['turn_ids']

        tokenized_answer = self.tokenizer(
            target,
            pad_to_max_length=True,
            max_length=self.max_target_length,
            return_tensors="pt",
            truncation=True
        )
        target_ids = tokenized_answer['input_ids'].squeeze()
        if self.ignore_pad_token_for_loss:
            target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        
        tokenized_source = np.array(self.tokenizer.tokenize(source))
        query_len = len(self.tokenizer.tokenize(query))
        source_len = len(tokenized_source)
        suffix_chunk_len = self.chunk_size-query_len-2
        if self.stride:
            overlap = suffix_chunk_len//2
            pad_len = math.ceil(max(source_len-suffix_chunk_len, 0)/(suffix_chunk_len-overlap))*(suffix_chunk_len-overlap)+suffix_chunk_len-source_len
            tokenized_source = np.pad(tokenized_source, (0, pad_len), 'constant', constant_values=" ")
            starts = range(0, pad_len+source_len-suffix_chunk_len+1, suffix_chunk_len-overlap)
            tokenized_source = [tokenized_source[start: start+suffix_chunk_len] for start in starts]
            tokenized_source = np.array(tokenized_source)
            tokenized_source = tokenized_source.reshape(-1, suffix_chunk_len)

        else:
            pad_len = math.ceil(source_len/suffix_chunk_len)*suffix_chunk_len-source_len
            tokenized_source = np.pad(tokenized_source, (0, pad_len), 'constant', constant_values=" ")
            tokenized_source = tokenized_source.reshape(-1, suffix_chunk_len)
        tokenized_source = tokenized_source[:source_ids.size(0)]
        cat_source = ["".join([t.replace("Ġ", " ") if t.startswith("Ġ") else t for t in ts]) for ts in tokenized_source.tolist()]
        cat_source = [s.strip() for s in cat_source]
        
        
        importance = self.cal_sim(turns, [query])
        # turn_ids.clamp(min=0).int()
        # print(importance[turn_ids])
        for i, imp in enumerate(importance):
            turn_ids = torch.where(turn_ids==i, imp, turn_ids.float())
        turn_ids = torch.where(turn_ids==-1.0, torch.tensor(0.0, dtype=torch.float32), turn_ids)
        max_value = torch.max(turn_ids)
        importance = torch.where(turn_ids==-2.0, max_value, turn_ids)

        return {
            'input_ids': source_ids,
            'attention_mask': source_attention_mask,
            'labels': target_ids,
            'decoder_attention_mask': tokenized_answer['attention_mask'].squeeze(),
            'importance': importance.detach()
        }
    
    def cal_sim(self, sen1, sen2):
        self.model.eval()
        emb1 = self.model.encode(sen1, show_progress_bar=False)
        emb2 = self.model.encode(sen2, show_progress_bar=False)
        sim = cos_sim(emb1, emb2)
        return list(sim)


class ChunkTokenizer:
    """Chunks and tokenizes input text and optional query for input to multi-encoder model. Does both chunking and
    tokenizing because the chunking is based on tokenized text."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        chunk_size: int,
        max_num_chunks: int,
        stride: bool = False,
        pad: bool = False,
        chunk_seq:bool = False,
    ):
        """
        Args:
            tokenizer: tokenizer used to tokenize text
            chunk_size: chunk size in number of tokens
            max_num_chunks: maximum number of chunks in total (optional)
            stride: whether to use striding
            pad: whether to "pad" chunks with empty strings to attain max_num_chunks chunks
        """
        if pad and not max_num_chunks:
            raise ValueError("Cannot pad without specifying max_num_chunks")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.max_num_chunks = max_num_chunks
        self.stride = stride
        self.pad = pad
        self.chunk_seq = chunk_seq
        
        # print(self.tokenizer.tokenize("<T>hello world.</T>"))
        # print(self.tokenizer.tokenize("<s><T>hello world. </T></s>"))
        # print(self.tokenizer.tokenize("<T> hello world. </T>"))
        # print(self.tokenizer.encode("<T>hello world.</T>"))
        # print(self.tokenizer.encode("<T>hello world. </T>"))
        # print(self.tokenizer.encode("<T> hello world. </T>"))

    def __call__(
        self,
        turns: list,
        query: str = None
    ):
        """
        Args:
            source: source text
            query: optional query text
        Returns:
            dictionary with tokenized chunks
        """
        turns = [f"<T>{turn}</T>" for turn in turns]
        source = ''.join(turns)
        if query:
            prefix = f"<s>{query}</s>"
        else:
            prefix = f"<s>"
        prefix_token_ids = self.tokenizer(
            prefix,
            add_special_tokens=False,
            max_length=self.chunk_size,
            return_tensors="pt",
            truncation=True
        )['input_ids']

        prefix_len = prefix_token_ids.size(-1)
        suffix_chunk_len = self.chunk_size - prefix_len

        suffix = f"{source}</s>"
        suffix_total_size = self.max_num_chunks * suffix_chunk_len
        input_ids = self.tokenizer(
            suffix,
            add_special_tokens=False,
            truncation=True,
            max_length=suffix_total_size,
        )['input_ids']

            
        suffix_len = min(len(input_ids), suffix_total_size)
        suffix_token_ids = input_ids[:suffix_len]
        
        if len(suffix_token_ids)>2 and suffix_token_ids[-1] == 50266:
            suffix_token_ids.append(2)
            suffix_len+=1
        elif len(suffix_token_ids)>2 and suffix_token_ids[-2] != 50266:
            suffix_token_ids.append(50266)
            suffix_token_ids.append(2)
            suffix_len+=2

        turn_sizes = []
        tmp_num = 0
        for token_id in suffix_token_ids:
            if token_id == 50265:
                tmp_num = 0
            elif token_id == 50266:
                turn_sizes.append(tmp_num)
            else:
                tmp_num += 1
        turn_sizes[-1] += 1

        while 50265 in suffix_token_ids and 50266 in suffix_token_ids:
            suffix_len -= 2
            suffix_token_ids.remove(50265)
            suffix_token_ids.remove(50266)
            
        suffix_attention = [1] * len(suffix_token_ids)

        if not self.stride and self.max_num_chunks > 1:
            pad_len = max(suffix_total_size-suffix_len, 0)
            suffix_token_ids += [self.tokenizer.pad_token_id] * pad_len
            suffix_attention += [0] * pad_len

            suffix_token_ids = torch.tensor(suffix_token_ids)
            suffix_attention = torch.tensor(suffix_attention)

            suffix_chunks = suffix_token_ids.view(-1, suffix_chunk_len)
            suffix_attention = suffix_attention.view(-1, suffix_chunk_len)

            prefix_chunks = prefix_token_ids.expand(suffix_chunks.size(0), -1)
            prefix_attention = torch.ones_like(prefix_chunks)

            chunk_input_ids = torch.cat((prefix_chunks, suffix_chunks), dim=1)
            chunk_attention_mask = torch.cat((prefix_attention, suffix_attention), dim=1)

            turn_ids = torch.cat([torch.tensor([i for _ in range(turn)]) for i, turn in enumerate(turn_sizes)], dim=0)
            turn_ids = torch.cat((turn_ids, torch.tensor([-1 for _ in range(pad_len)])), dim=0)
            turn_ids = turn_ids.view(-1, suffix_chunk_len)
            query_turn_ids = torch.tensor([-2]).repeat((turn_ids.size(0), prefix_chunks.size(1)))
            turn_ids = torch.cat((query_turn_ids, turn_ids), dim=1)

        else:
            overlap = suffix_chunk_len//2
            pad_len = math.ceil((suffix_total_size-suffix_chunk_len)/(suffix_chunk_len-overlap))*(suffix_chunk_len-overlap)+suffix_chunk_len-suffix_len
            starts = range(0, pad_len+suffix_len-suffix_chunk_len+1, suffix_chunk_len-overlap)
            if self.chunk_seq:
                starts = [starts[i] for i in range(0, len(starts), 2)] + [starts[i] for i in range(1, len(starts), 2)]

            suffix_token_ids += [self.tokenizer.pad_token_id] * pad_len
            suffix_attention += [0] * pad_len
            suffix_chunks = [suffix_token_ids[start: start+suffix_chunk_len] for start in starts]
            suffix_attention = [suffix_attention[start: start+suffix_chunk_len] for start in starts]
            suffix_chunks = torch.tensor(suffix_chunks)
            suffix_attention = torch.tensor(suffix_attention)
            prefix_chunks = prefix_token_ids.expand(suffix_chunks.size(0), -1)
            prefix_attention = torch.ones_like(prefix_chunks)
            chunk_input_ids = torch.cat((prefix_chunks, suffix_chunks), dim=1)
            chunk_attention_mask = torch.cat((prefix_attention, suffix_attention), dim=1)

            no_stride_turn_ids = torch.cat([torch.tensor([i for _ in range(turn)]) for i, turn in enumerate(turn_sizes)], dim=0)
            no_stride_turn_ids = torch.cat((no_stride_turn_ids, torch.tensor([-1 for _ in range(pad_len)])), dim=0)
            turn_ids = [no_stride_turn_ids[start: start+suffix_chunk_len] for start in starts]
            turn_ids = torch.stack(turn_ids)
            query_turn_ids = torch.tensor([-2]).repeat(turn_ids.size(0), prefix_chunks.size(1))
            turn_ids = torch.cat((query_turn_ids, turn_ids), dim=1)
        

        return {
            "input_ids": chunk_input_ids,
            "attention_mask": chunk_attention_mask,
            "turn_ids": turn_ids
        }
