# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" GLUE processors and helpers """
import numpy as np
import jsonlines
import os
import warnings
from dataclasses import asdict
from enum import Enum
from typing import List, Optional, Union

from ...file_utils import is_tf_available
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
from .utils import DataProcessor, InputExample, InputFeatures
from torch.utils.data.dataset import Dataset
import random
import torch


if is_tf_available():
    import tensorflow as tf

logger = logging.get_logger(__name__)

DEPRECATION_WARNING = (
    "This {0} will be removed from the library soon, preprocessing should be handled with the 🤗 Datasets "
    "library. You can have a look at this example script for pointers: "
    "https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py"
)


def glue_convert_examples_to_features(
    examples: Union[List[InputExample], "tf.data.Dataset"],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = None,
    task=None,
    label_list=None,
    output_mode=None,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length. Defaults to the tokenizer's max_len
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset`` containing the
        task-specific features. If the input is a list of ``InputExamples``, will return a list of task-specific
        ``InputFeatures`` which can be fed to the model.

    """
    warnings.warn(DEPRECATION_WARNING.format("function"), FutureWarning)
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        if task is None:
            raise ValueError("When calling glue_convert_examples_to_features from TF, the task parameter is required.")
        return _tf_glue_convert_examples_to_features(examples, tokenizer, max_length=max_length, task=task)
    return _glue_convert_examples_to_features(
        examples, tokenizer, max_length=max_length, task=task, label_list=label_list, output_mode=output_mode
    )


def factcheck_convert_examples_to_features(
    examples: List[InputExample],
    tokenizer: PreTrainedTokenizer,
    max_length: Optional[int] = 512,
    task=None,
    label_list=None,
    output_mode=None,
):
    if max_length is None:
        max_length = tokenizer.max_len

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example: InputExample) -> Union[int, float, None]:
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]
        elif output_mode == "regression":
            return float(example.label)
        raise KeyError(output_mode)

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_length,
        # truncation=True,
        add_special_tokens=True,
        truncation='only_first', 
        return_token_type_ids=True
    )

        # pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
        # pad_token=tokenizer.pad_token_id,
        # pad_token_segment_id=tokenizer.pad_token_type_id,

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("features: %s" % features[i])

    return features


class OutputMode(Enum):
    classification = "classification"
    regression = "regression"


class PolifactProcessor(DataProcessor):

    def __init__(self, args, **kwargs):
        # super().__init__(*args, **kwargs)
        # print(args)
        warnings.warn(DEPRECATION_WARNING.format("processor"), FutureWarning)
        self.is_binary = args.is_binary # binary or multi
        self.has_evidence = args.has_evidence #False
        self.subtask = args.politifact_subtask #'liar' # liar, covid
        self.output_mode = args.output_mode
        self.filter_middle_classes = args.filter_middle_classes
        self.few_shot = args.few_shot
        self.myth = args.myth
        self.fever = args.fever
        self.liar = args.liar
        self.seed_ = args.seed
        self.covidpoli = args.covidpoli

        self.multi2binary = {
            "true" : "true",
            "mostly-true": "true",
            "half-true": "true",
            "barely-true": "false",
            "false": "false",
            "pants-fire": "false",
            "NOT ENOUGH INFO": "false",
            "REFUTES": "_",
            "SUPPORTS": "true"
        }
        if self.output_mode == 'regression':
            self.labels = [None]

        if self.is_binary:
            # classification binary
            self.labels = ["true", "false"]
        else:
            if self.fever:
                self.labels = ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]
            else:
                self.labels = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]

    # def get_example_from_tensor_dict(self, tensor_dict):
    #     """See base class."""
    #     return InputExample(
    #         tensor_dict["idx"].numpy(),
    #         tensor_dict["sentence1"].numpy().decode("utf-8"),
    #         tensor_dict["sentence2"].numpy().decode("utf-8"),
    #         str(tensor_dict["label"].numpy()),
    #     )

    def get_train_examples(self, data_dir):
        if self.has_evidence:
        #     if self.fever:
        #         path_ =  '/home/yejin/fever/data/fever_train_for_bert.jsonl'
        #         # path_ = "{}/naacl/fever_train_for_bert_w_ppl.jsonl".format(data_dir)
        #     elif self.myth:
        #         path_ = '/home/yejin/covid19_factcheck/data/covid_myth_test_v3.jsonl'
        #     elif self.liar:
        #         path_ = "{}/politifact/{}/liar-plus_train_v3.jsonl".format(data_dir, self.subtask)
        #         # path_ ='/home/nayeon/covid19_factcheck/data/liar-plus_train_v3_justification_top1_naacl.jsonl'
        #     elif self.covidpoli:
        #         path_='/home/yejin/covid19_factcheck/data/factcheck_data/politifact/liar/test_covid19_justification_naacl.jsonl'
        #     else:
        #         # using FEVER-based evidences
        #         if any([self.use_credit, self.use_metainfo, self.use_creditscore]):
        #             path_ = "{}/politifact/{}/train_evidence_meta_fever_v4a.jsonl".format(data_dir, self.subtask)
        #         else:
        #             print("reading data")
        #             path_ = "{}/politifact/{}/train_evidence_meta_fever_v4a.jsonl".format(data_dir, self.subtask)

        # # ============ PATH DONE ============
        #     print("loading from {}".format(path_))
        #     with jsonlines.open(path_) as reader:
        #         obj_list = [obj for obj in reader]

        #     if self.filter_middle_classes:
        #         obj_list = [obj for obj in obj_list if obj['label'] not in ['half-true','barely-true']]
            if self.few_shot:
                if self.fever:
                    path_ = '/home/yejin/fever/data/fever_train_for_bert_s.jsonl'
                    eval_file ='/home/nayeon/covid19_factcheck/ppl_results/naacl.gpt2.uni.fever_train_small.npy'
                elif self.liar:
                    path_ = "/home/nayeon/covid19_factcheck/data/liar-plus_train_v3_justification_top1_naacl.jsonl"
                    eval_file ='/home/nayeon/covid19_factcheck/ppl_results/naacl.gpt2.uni.liar_train_justification_top1.npy'
                elif self.covidpoli:
                    path_ = '/home/yejin/covid19_factcheck/data/factcheck_data/politifact/liar/test_covid19_justification_naacl.jsonl'
                    eval_file ='/home/nayeon/covid19_factcheck/ppl_results/naacl.gpt2.uni.naacl_covid_politifact_justification.npy'
                elif self.myth:
                    path_ = '/home/yejin/covid19_factcheck/data/covid_myth_test_v3.jsonl'
                    eval_file ='/home/nayeon/covid19_factcheck/ppl_results/naacl.gpt2.uni.naacl_covid_myth_v3.npy'

                all_objs = self.load_full_liar_with_ppl(path_, eval_file)
                combined_all_objs = all_objs['true'] + all_objs['false']
                random.seed(self.seed_)
                random.shuffle(combined_all_objs)
                obj_list = combined_all_objs[:self.few_shot]
                print("Looking from here {}".format(path_))
                print("Using few shot!!!! LEN: ", len(obj_list))

                return self._create_examples_with_evidences(obj_list, "train")
            else:
                if self.fever:
                    path_ = '/home/yejin/fever/data/fever_train_for_bert_s.jsonl'
                elif self.liar:
                    path_ = "/home/nayeon/covid19_factcheck/data/liar-plus_train_v3_justification_top1_naacl.jsonl"
                
                with jsonlines.open(path_) as reader:
                    obj_list = [obj for obj in reader if obj['label'] != 'REFUTES']
                print("Train from {}".format(path_))
                print("Train {} Samples".format(len(obj_list)))
                
                return self._create_examples_with_evidences(obj_list, "train")
        else:
            if self.fever:
                path_ = "{}/naacl/fever_train_for_bert_w_ppl.jsonl".format(data_dir)

                with jsonlines.open(path_) as reader:
                    obj_list = [obj for obj in reader if obj['evidences'] != [] and obj['evidences'][0][0] != 0]

                if self.few_shot:
                    new_obj_list = obj_list[:self.few_shot]
                    obj_list = new_obj_list
                    print("Using few shot!!!! LEN: ", len(obj_list))

                return self._create_fever_examples(obj_list, "train")

            else:
                path_ = "{}/politifact/{}/train{}.tsv".format(data_dir, self.subtask, data_source)
                print("loading from {}".format(path_))
                return self._create_examples(self._read_tsv(path_), "train")
            # return self._create_examples(self._read_tsv(os.path.join(data_dir, "train{}.tsv".format(self.data_source))), "train")
    
    def get_dev_examples(self, data_dir):
       if self.has_evidence:
            if self.few_shot:
                if self.fever:
                    path_ = "{}/naacl/fever_test_for_bert_w_ppl.jsonl".format(data_dir)
                    with jsonlines.open(path_) as reader:
                        obj_list = [obj for obj in reader if obj['label'] != 'REFUTES']
                elif self.liar:
                    path_ ='/home/nayeon/covid19_factcheck/data/liar-plus_test_v3_justification_top1_naacl.jsonl'
                    with jsonlines.open(path_) as reader:
                        obj_list = [obj for obj in reader if obj['label'] != 'REFUTES']
                elif self.myth:
                    path_ = '/home/yejin/covid19_factcheck/data/covid_myth_test_v3.jsonl'
                    eval_file = '/home/nayeon/covid19_factcheck/ppl_results/naacl.gpt2.uni.naacl_covid_myth_v3.npy'

                    all_objs = self.load_full_liar_with_ppl(path_, eval_file)
                    combined_all_objs = all_objs['true'] + all_objs['false']
                    random.seed(self.seed_)
                    random.shuffle(combined_all_objs)
                    obj_list = combined_all_objs[self.few_shot + 1:]
                elif self.covidpoli:
                    path_ = '/home/yejin/covid19_factcheck/data/factcheck_data/politifact/liar/test_covid19_justification_naacl.jsonl'
                    eval_file = '/home/nayeon/covid19_factcheck/ppl_results/naacl.gpt2.uni.naacl_covid_politifact_justification.npy'

                    all_objs = self.load_full_liar_with_ppl(path_, eval_file)
                    combined_all_objs = all_objs['true'] + all_objs['false']
                    random.seed(self.seed_)
                    random.shuffle(combined_all_objs)
                    print(len(combined_all_objs))
                    obj_list = combined_all_objs[self.few_shot+1:]

                    # random.seed(self.seed_)
                    # obj_list = obj_list[:self.few_shot]
                    print("Using few dev shot!!!! LEN: ", len(obj_list))
                print("loading from dev !! {}".format(path_))
                return self._create_examples_with_evidences(obj_list, "dev")
            else:
                if self.fever:
                    path_ = "{}/naacl/fever_test_for_bert_w_ppl.jsonl".format(data_dir)
                    with jsonlines.open(path_) as reader:
                        obj_list = [obj for obj in reader if obj['label'] != 'REFUTES']
                    
                elif self.liar:
                    path_ ='/home/nayeon/covid19_factcheck/data/liar-plus_test_v3_justification_top1_naacl.jsonl'
                    with jsonlines.open(path_) as reader:
                        obj_list = [obj for obj in reader if obj['label'] != 'REFUTES']
                    
                print("Evalutate from {}".format(path_))
                print("Evaluate {} samples".format(len(obj_list)))

                return self._create_examples_with_evidences(obj_list, "dev")
        # else:
        #     if self.fever:
        #         path_ = "{}/naacl/fever_valid_for_bert_w_ppl_s.jsonl".format(data_dir)
        #         # path_ = "{}/naacl/fever_test_for_bert_w_ppl_{}_test.jsonl".format(data_dir, self.cross_validation)
        #         with jsonlines.open(path_) as reader:
        #             obj_list = [obj for obj in reader if obj['evidences'] != [] and obj['evidences'][0][0] != 0]
        #         return self._create_fever_examples(obj_list, "dev")
        #     else:
        #         path_ = "{}/politifact/{}/valid{}.tsv".format(data_dir, self.subtask, data_source)
        #         print("loading from {}".format(path_))
        #         return self._create_examples(self._read_tsv(os.path.join(data_dir, path_)), "dev")
        #         # return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev{}.tsv".format(self.data_source))), "dev")


    def get_test_examples(self, data_dir):
        if self.has_evidence:
            if self.fever:
                path_ = "{}/naacl/fever_test_for_bert.jsonl".format(data_dir)
            elif self.liar:
                path_ = "{}/politifact/{}/liar-plus_test_v3.jsonl".format(data_dir, self.subtask)
                # path_ = '/home/nayeon/covid19_factcheck/data/liar-plus_test_v3_justification_top1_naacl.jsonl'
            else:
                if any([self.use_credit, self.use_metainfo, self.use_creditscore, self.use_ppl]):
                    path_ = "{}/politifact/{}/test_evidence_meta_fever_v4a.jsonl".format(data_dir, self.subtask)
                else:
                    path_ = "{}/politifact/{}/test_evidence_meta_fever_v4a.jsonl".format(data_dir, self.subtask)

            print("loading from {}".format(path_))
            if self.few_shot and self.fever:
                with jsonlines.open(path_) as reader:
                    obj_list = [obj for obj in reader if obj['label'] != 'REFUTES']
            else:
                with jsonlines.open(path_) as reader:
                    obj_list = [obj for obj in reader]

            return self._create_examples_with_evidences(obj_list, "test")
        else:
            if self.fever:
                path_ = "{}/naacl/fever_test_for_bert_w_ppl.jsonl".format(data_dir)
                # path_ = "{}/naacl/fever_test_for_bert_w_ppl_{}_test.jsonl".format(data_dir, self.cross_validation)
                with jsonlines.open(path_) as reader:
                    obj_list = [obj for obj in reader if obj['evidences'] != [] and obj['evidences'][0][0] != 0]
                return self._create_fever_examples(obj_list, "test")
            else:
                path_ = "{}/politifact/{}/test{}.tsv".format(data_dir, self.subtask, data_source)
                print("loading from {}".format(path_))
                return self._create_examples(self._read_tsv(os.path.join(data_dir, path_)), "test")

 
    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = None if set_type == "test" else line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


    def _create_examples_with_evidences(self, obj_list, set_type, evidence_option='concat'):
        examples = []
        for (i, obj) in enumerate(obj_list):
            try:
                guid = "%s-%s" % (set_type, obj['claim_id'])
            except:
                guid = "%s-%s" % (set_type, obj['id'])

            text_a = obj['claim']
        
            if evidence_option == 'concat':
                self.is_t3 = False
                if self.is_t3:
                    text_b = " ".join([e_tuple[0] for e_tuple in obj['evidences'][:3]])
                else:
                    text_b = obj['evidences'][0][0]
            elif evidence_option == 'use_body':
                raise NotImplementedError
            elif evidence_option == 'separate_evidences':
                raise NotImplementedError

            label = obj['label']

            if self.is_binary:
                # map to 6 label to binary label
                label = self.multi2binary[label]

            # print(text_a)
            # print(text_b)
            # print(label)
            # exit(0)

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def load_full_liar_with_ppl(self, data_path, ppl_result_path):
        with jsonlines.open(data_path) as reader:
            og_objs = [obj for obj in reader]

        ppl_results = np.load(ppl_result_path, allow_pickle=True)

        all_objs = {
            'true': [],
            'false': [],
            '_': []
        }

        for obj, ppl in zip(og_objs, ppl_results):
            label = self.multi2binary[obj['label']]
            if 'fever' in data_path:
                claim_id = obj['id']
            else:
                claim_id = obj['claim_id']
            claim = obj['claim']
            evs = obj['evidences'][:3]
            ppl = ppl['perplexity']

            new_objs = {'ppl': ppl, 'label': label, 'claim': claim, 'evidences': evs, 'claim_id': claim_id}
            all_objs[label].append(new_objs)
        return all_objs


factcheck_processors = {
    'gpt2baseline': PolifactProcessor,
    'politifact': PolifactProcessor,
    # 'fusion': FusionProcessor
}

class DatasetForClassification(Dataset):
    def __init__(self, args, tokenizer: PreTrainedTokenizer, phase: str, local_rank=-1):
        self.tokenizer = tokenizer

        self.labels = ["true", "false"]
        self.label_map = {label: i for i, label in enumerate(self.labels)}

        processor = PolifactProcessor(args)
        if phase == 'train':
            self.examples = processor.get_train_examples(args.data_dir)
        elif phase == 'dev' or 'test':
            self.examples = processor.get_train_examples(args.data_dir)
            # self.examples = processor.get_dev_examples(args.data_dir)
        

    # def fever_data_cleaning(self, sent):
    #     sent = sent.replace('-LRB-', '(')
    #     sent = sent.replace('-RRB-', ')')
    #     sent = sent.replace('-LSB-', '[')
    #     sent = sent.replace('-RSB-', ']')
    #     return sent

    def convert_claim_ev(self, example):
        
        example = example[0]
        single_encoding = self.tokenizer.encode_plus(
            (example.text_a, example.text_b),
            # max_length=args.max_seq_length,
            # truncation=True,
            add_special_tokens=True,
            pad_token=self.tokenizer.pad_token_id,
            return_token_type_ids=False
        )
        input_ids = torch.tensor(single_encoding['input_ids'], dtype=torch.long)
        labels = torch.tensor(self.label_map[example.label], dtype=torch.long)

        return input_ids, labels

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:

        # is_testing_with_claim_only = False
        # if is_testing_with_claim_only:
        #     examples = self.just_input([self.ev_claim_tuples[i]])
        #     return torch.tensor(examples, dtype=torch.long)

        input_ids, label = self.convert_claim_ev([self.examples[i]])
        return (torch.tensor(input_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long))
