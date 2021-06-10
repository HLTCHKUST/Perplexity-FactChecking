import logging
import os
import pickle
import time
import numpy as np
import jsonlines

import torch
from torch.utils.data.dataset import Dataset

from ...tokenization_utils import PreTrainedTokenizer
from ...trainer import torch_distributed_zero_first

import re
from fuzzywuzzy import fuzz
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
        self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, overwrite_cache=False, local_rank=-1,
    ):
        assert os.path.isfile(file_path)

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, "cached_lm_{}_{}_{}".format(tokenizer.__class__.__name__, str(block_size), filename,),
        )

        with torch_distributed_zero_first(local_rank):
            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    text = f.read()

                tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    f"Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, local_rank=-1):
        assert os.path.isfile(file_path)
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        if 'txt' in file_path:
            with open(file_path, encoding="utf-8") as f:
                lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
        elif 'npy' in file_path:    
            lines = np.load(file_path, allow_pickle=True)
        batch_encoding = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class FactCheckDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, local_rank=-1):
        self.tokenizer = tokenizer
        self.block_size = block_size

        with jsonlines.open(file_path) as reader:
            objs = [obj for obj in reader]
            if len(objs) == 1:
                objs = objs[0]
            
            if 'fever' in file_path:
                claims = [self.fever_data_cleaning(obj['claim'].lower().strip()) for obj in objs]
            else:
                claims = [obj['claim'].lower().strip() for obj in objs]
            
            if 'fever' in file_path:
                evs_lines = [self.fever_data_cleaning(obj['evidences'][0][0]) if obj['evidences'] != []else '' for obj in objs]  # use top 1 evidence
            else:
                evs_lines = [obj['evidences'][0][0] for obj in objs if obj['evidences'] != []] # use top 1 evidence
            
            ev_claim_tuples = [(single_ev, claim) for claim, single_ev in zip(claims, evs_lines)]

            self.ev_claim_tuples = ev_claim_tuples

    def fever_data_cleaning(self, sent):
        sent = sent.replace('-LRB-', '(')
        sent = sent.replace('-RRB-', ')')
        sent = sent.replace('-LSB-', '[')
        sent = sent.replace('-RSB-', ']')
        return sent

    def convert_claim_ev(self, ev_claim_tuple):
        '''
            Mask out the "evidence tokens" to ignore them when calculating the final perplexity score 
        '''
        # TODO YEON CHECK
        # single tuple
    
        # batch_encoding = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=ev_claim_tuple, add_special_tokens=True, max_length=self.block_size)
        # evidence_token_len = len(batch_encoding["input_ids"][0]) - np.count_nonzero(batch_encoding['token_type_ids'][0])
        
        # labels = np.array(batch_encoding["input_ids"])
        # labels[0][:evidence_token_len] = -100

        # inputs = batch_encoding["input_ids"]

        ev_claim_tuple = ev_claim_tuple[0] # batch -> single
        batch_encoding = self.tokenizer.encode_plus(ev_claim_tuple[0], ev_claim_tuple[1], add_special_tokens=True, 
                                                    max_length=self.block_size, truncation='only_first', return_token_type_ids=True)
        
        evidence_token_len = len(batch_encoding["input_ids"]) - np.count_nonzero(batch_encoding['token_type_ids'])

        # print(evidence_token_len)
        # # token_type_ids = [0000,11111]
        # # use: input_ids, token_type_ids

        labels = np.array(batch_encoding["input_ids"])
        labels[:evidence_token_len] = -100

        inputs = batch_encoding["input_ids"]

        return inputs, labels

    def just_input(self, ev_claim_tuple):
        claims = [claim for (ev, claim) in ev_claim_tuple]

        single_encoding = self.tokenizer.encode_plus(claims[0])
        # batch_encoding = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=claims, add_special_tokens=True, max_length=self.block_size)
        
        inputs = single_encoding["input_ids"]

        return inputs


    def __len__(self):
        return len(self.ev_claim_tuples)

    def __getitem__(self, i) -> torch.Tensor:
        examples, labels = self.convert_claim_ev([self.ev_claim_tuples[i]])
        return (torch.tensor(examples, dtype=torch.long), torch.tensor(labels, dtype=torch.long))
