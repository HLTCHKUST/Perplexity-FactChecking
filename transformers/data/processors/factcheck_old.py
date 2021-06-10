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
import logging
import os
import json
import jsonlines

from ...file_utils import is_tf_available
from .utils import DataProcessor, InputExample, InputFeatures
import random

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)
logger.disabled = True

CLASSIFICATION_TO_REGRESSION = {
    "true" : '0.0', 
    "mostly-true": '0.2', 
    "half-true": '0.4', 
    "barely-true": '0.6', 
    "false": '0.8', 
    "pants-fire": '1.0'
}

def factcheck_convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    task=None,
    label_list=None,
    output_mode=None,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = 0
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
            len_examples = tf.data.experimental.cardinality(examples)
        else:
            len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(
            example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True,
        )

        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    if is_tf_available() and is_tf_dataset:

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    },
                    ex.label,
                )

        return tf.data.Dataset.from_generator(
            gen,
            ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                tf.TensorShape([]),
            ),
        )

    return features

class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""
    def __init__(self, args):
        self.rte2misinfo_map = {
            'entailment': 'true',
            'not_entailment': 'false'
        }

        self.output_mode = args.output_mode
        if self.output_mode == 'regression':
            self.labels = [None]
        else:
            self.labels = ["true", "false"]

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, data_source=""):
        """See base class."""
        path = "{}/RTE/train{}.tsv".format(data_dir, data_source)
        print("loading from {}".format(path))
        return self._create_examples(self._read_tsv(path), "train")
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "train{}.tsv".format(self.data_source))), "train")

    def get_dev_examples(self, data_dir, data_source=""):
        """See base class."""
        path = "{}/RTE/dev{}.tsv".format(data_dir, data_source)
        print("loading from dev {}".format(path))

        return self._create_examples(self._read_tsv(path), "dev")
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev{}.tsv".format(self.data_source))), "dev")

    def get_test_examples(self, data_dir, data_source=""):
        """See base class."""
        path = "{}/RTE/test{}.tsv".format(data_dir, data_source)
        print("loading from {}".format(path))
        return self._create_examples(self._read_tsv(path), "test")


    def get_labels(self):
        """See base class."""
        # return ["entailment", "not_entailment"]
        return self.labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            
            label = self.rte2misinfo_map[line[-1]]
            if self.output_mode == 'regression':
                label = CLASSIFICATION_TO_REGRESSION[label]
            
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class SciTailProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, data_source=""):
        """See base class."""
        path = "{}/SciTail/train{}.tsv".format(data_dir, data_source)
        print("loading from {}".format(path))
        return self._create_examples(self._read_tsv(path), "train")
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "train{}.tsv".format(self.data_source))), "train")

    def get_dev_examples(self, data_dir, data_source=""):
        """See base class."""
        path = "{}/SciTail/dev{}.tsv".format(data_dir, data_source)
        print("loading from {}".format(path))
        return self._create_examples(self._read_tsv(path), "dev")
        # return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev{}.tsv".format(self.data_source))), "dev")

    def get_test_examples(self, data_dir, data_source=""):
        """See base class."""
        path = "{}/SciTail/test{}.tsv".format(data_dir, data_source)
        print("loading from {}".format(path))
        return self._create_examples(self._read_tsv(path), "test")


    def get_labels(self):
        """See base class."""
        return ["entails", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "" #"%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class PolifactProcessor(DataProcessor):
    def __init__(self, args):
        self.is_binary = args.is_binary # binary or multi
        self.has_evidence = args.has_evidence #False
        self.subtask = args.politifact_subtask #'liar' # liar, covid
        self.output_mode = args.output_mode
        self.filter_middle_classes = args.filter_middle_classes
        # self.use_credit = args.use_credit
        # self.use_metainfo = args.use_metainfo
        # self.use_creditscore = args.use_creditscore
        # self.use_ppl_vector = args.use_ppl_vector
        # self.use_ppl = args.use_ppl
        self.few_shot = args.few_shot
        # self.claim_only = args.claim_only
        self.myth = args.myth
        self.fever = args.fever
        self.liar = args.liar
        # self.cross_validation = args.cross_validation
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
        elif self.is_binary:
            # classification binary
            self.labels = ["true", "false"]
        else:
            # classification full
            if self.fever:
                self.labels = ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]
            else:
                self.labels = ["true", "mostly-true", "half-true", "barely-true", "false", "pants-fire"]



    def get_train_examples(self, data_dir, data_source=""):
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
                    path_ = "/home/nayeon/covid19_factcheck/data/liar-plus_train_v3_justification_top1_naacl.jsonl".format(data_dir, self.subtask)
                    eval_file ='/home/nayeon/covid19_factcheck/ppl_results/naacl.gpt2.uni.liar_train_justification_top1.npy'
                elif self.covidpoli:
                    path_ = '/home/yejin/covid19_factcheck/data/factcheck_data/politifact/liar/test_covid19_justification_naacl.jsonl'
                    eval_file ='/home/nayeon/covid19_factcheck/ppl_results/naacl.gpt2.uni.naacl_covid_politifact_justification.npy'
                elif self.myth:
                    path_ = '/home/yejin/covid19_factcheck/data/covid_myth_test_v3.jsonl'
                    eval_file ='/home/nayeon/covid19_factcheck/ppl_results/naacl.gpt2.uni.naacl_covid_myth_v3.npy'

                all_objs = self.load_full_liar_with_ppl(path_, eval_file)
                combined_all_objs = all_objs['true'] + all_objs['false']
                random.shuffle(combined_all_objs)
                random.seed(self.seed_)
                obj_list = combined_all_objs[:self.few_shot]

                print("Using few shot!!!! LEN: ", len(obj_list))

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

    def get_dev_examples(self, data_dir, data_source=""):
        if self.has_evidence:
            # if self.fever:
            #     path_ = "{}/naacl/fever_test_for_bert_w_ppl.jsonl".format(data_dir)
            # elif self.liar:
            #     path_ ='/home/nayeon/covid19_factcheck/data/liar-plus_test_v3_justification_top1_naacl.jsonl'

            # with jsonlines.open(path_) as reader:
            #     obj_list = [obj for obj in reader if obj['label'] != 'REFUTES']
            if self.few_shot:
                if self.fever:
                    path_ = "{}/naacl/fever_test_for_bert_w_ppl.jsonl".format(data_dir)
                    with jsonlines.open(path_) as reader:
                        obj_list = [obj for obj in reader if obj['label'] != 'REFUTES']
                elif self.liar:
                    path_ ='/home/nayeon/covid19_factcheck/data/liar-plus_test_v3_justification_top1_naacl.jsonl'
                    with jsonlines.open(path_) as reader:
                        obj_list = [obj for obj in reader if obj['label'] != 'REFUTES']
                if self.myth:
                    path_ = '/home/yejin/covid19_factcheck/data/covid_myth_test_v3.jsonl'
                    eval_file = '/home/nayeon/covid19_factcheck/ppl_results/naacl.gpt2.uni.naacl_covid_myth_v3.npy'

                    all_objs = self.load_full_liar_with_ppl(path_, eval_file)
                    combined_all_objs = all_objs['true'] + all_objs['false']
                    random.shuffle(combined_all_objs)
                    # random.seed(self.seed_)
                    obj_list = combined_all_objs[self.few_shot + 1:]
                elif self.covidpoli:
                    path_ = '/home/yejin/covid19_factcheck/data/factcheck_data/politifact/liar/test_covid19_justification_naacl.jsonl'
                    eval_file = '/home/nayeon/covid19_factcheck/ppl_results/naacl.gpt2.uni.naacl_covid_politifact_justification.npy'

                    all_objs = self.load_full_liar_with_ppl(path_, eval_file)
                    combined_all_objs = all_objs['true'] + all_objs['false']
                    random.shuffle(combined_all_objs)
                    # random.seed(self.seed_)
                    print(len(combined_all_objs))
                    obj_list = combined_all_objs[self.few_shot+1:]

                    # random.seed(self.seed_)
                    # obj_list = obj_list[:self.few_shot]
                    print("Using few dev shot!!!! LEN: ", len(obj_list))
            print("loading from dev !! {}".format(path_))
            return self._create_examples_with_evidences(obj_list, "dev")
        else:
            if self.fever:
                path_ = "{}/naacl/fever_valid_for_bert_w_ppl_s.jsonl".format(data_dir)
                # path_ = "{}/naacl/fever_test_for_bert_w_ppl_{}_test.jsonl".format(data_dir, self.cross_validation)
                with jsonlines.open(path_) as reader:
                    obj_list = [obj for obj in reader if obj['evidences'] != [] and obj['evidences'][0][0] != 0]
                return self._create_fever_examples(obj_list, "dev")
            else:
                path_ = "{}/politifact/{}/valid{}.tsv".format(data_dir, self.subtask, data_source)
                print("loading from {}".format(path_))
                return self._create_examples(self._read_tsv(os.path.join(data_dir, path_)), "dev")
                # return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev{}.tsv".format(self.data_source))), "dev")

    def get_test_examples(self, data_dir, data_source=""):
        """See base class."""

        if self.has_evidence:
            if self.cross_validation:
                if self.use_ppl:
                    if self.myth:
                        path_="{}/naacl/covid_myth_test_w_{}_test.jsonl".format(data_dir, self.cross_validation)
                    elif self.fever:
                        path_ = "{}/naacl/fever_test_for_bert_w_ppl_{}_test.jsonl".format(data_dir,
                                                                                          self.cross_validation)
                    else:
                        path_="{}/naacl/test_covid19_justification_w_{}_test.jsonl".format(data_dir, self.cross_validation)
  # path_="{}/naacl/covid_myth_test_w_{}_test.jsonl".format(data_dir, self.cross_validation)
                else:
                    if self.fever:
                        path_ = "{}/naacl/fever_test_for_bert_w_ppl_{}_test.jsonl".format(data_dir, self.cross_validation)
                    elif self.myth:
                        path_ = "{}/naacl/covid_myth_test_w_{}_test.jsonl".format(data_dir, self.cross_validation)
                    else:
                        path_ = "{}/naacl/test_covid19_justification_w_{}_test.jsonl".format(data_dir,
                                                                                              self.cross_validation)
                    # else:
                    #     path_ = "{}/politifact/{}/cross_validation/{}_test.jsonl".format(data_dir, self.subtask, self.cross_validation)
            else:
                if self.fever:
                    # if self.claim_only:
                    #     path_ = "{}/naacl/fever_test_for_bert_w_claimonly_ppl.jsonl".format(data_dir)
                    # else:
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
                    obj_list = [obj for obj in reader if obj['label'] != 'NOT ENOUGH INFO']
            else:
                with jsonlines.open(path_) as reader:
                    obj_list = [obj for obj in reader]

            # if self.fever and self.use_ppl:
            #     if self.claim_only:
            #         RESULT_PATH = '/home/nayeon/covid19_factcheck/ppl_results/naacl_fever_test_claim_only.npy'
            #     else:
            #         RESULT_PATH = '/home/nayeon/covid19_factcheck/ppl_results/naacl_fever_test_for_bert_cleaned.npy'
            #     new_obj_list = []
            #     ppl_results = np.load(RESULT_PATH, allow_pickle=True)
            #     ppls = [ppl['perplexity'] for ppl in ppl_results]
            #
            #     for obj, ppl in zip(obj_list, ppl_results):
            #         obj['ppl'] = ppl['perplexity']
            #         obj['ppl_avg'] = ppl['perplexity'] / np.mean(ppls)
            #         new_obj_list.append(obj)
            #     obj_list = new_obj_list
            #
            # if self.liar and self.use_ppl:
            #     new_obj_list = []
            #     RESULT_PATH = '/home/nayeon/covid19_factcheck/ppl_results/liar_test_justification_top1_ppl.npy'
            #     ppl_results = np.load(RESULT_PATH, allow_pickle=True)
            #     ppls = [ppl['perplexity'] for ppl in ppl_results]
            #
            #     for obj, ppl in zip(obj_list, ppl_results):
            #         obj['ppl'] = ppl['perplexity']
            #         obj['ppl_avg'] = ppl['perplexity'] / np.mean(ppls)
            #         new_obj_list.append(obj)
            #     obj_list = new_obj_list

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

    def _create_examples_with_evidences(self, obj_list, set_type, evidence_option='concat'):
        examples = []
        for (i, obj) in enumerate(obj_list):
            try:
                guid = "%s-%s" % (set_type, obj['claim_id'])
            except:
                guid = "%s-%s" % (set_type, obj['id'])

            text_a = obj['claim']
        
            if evidence_option == 'concat':
                # concat all evidence sentences into one "context"
                self.is_t3 = True
                if self.is_t3:
                    text_b = " ".join([e_tuple[0] for e_tuple in obj['evidences'][:3]])
                    # e_tuple e.g. = ['"building a wall" on the border "will take literally years.', 28.547153555348473]
                else:
                    text_b = obj['evidences'][0][0]
            elif evidence_option == 'use_body':
                raise NotImplementedError
            elif evidence_option == 'separate_evidences':
                # create multiple claim-evidence pair from one obj.
                # e.g. {claim, [evidence1, evidence2, evidence3]} => {claim, evidence1}, {claim, evidence2}, {claim, evidence3}
                # evidence_list = obj['evidences']
                raise NotImplementedError

            label = obj['label']

            if self.is_binary:
                # map to 6 label to binary label
                label = self.multi2binary[label]

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

        return examples

    def _create_fever_examples(self, obj_list, set_type, evidence_option='concat'):
        examples = []
        if self.use_ppl:
            print("using cap")
            ppls = [ppl['ppl'] for ppl in obj_list]
            cap = np.mean(ppls)+np.std(ppls)
        for (i, obj) in enumerate(obj_list):
            guid = "%s-%s" % (set_type, obj['id'])

            text_a = obj['claim']
            label = obj['label']
            text_b = None
            if self.use_ppl:
                # ppl = [obj['ppl']]
                # ppl = [obj['big_ppl']]
                # ppl = obj['ppl']
                # ppl = 1 if ppl >= 500 else (ppl / 500)
                # ppl = [float(ppl)]
                # if self.use_ppl == 'avg':
                #     ppl = [float(obj['ppl_avg'])]
                # elif self.use_ppl == 'cap':
                #     ppl = 1 if ppl >= cap else (ppl / cap)
                #     ppl = [ppl]
                # else:
                print("using cap")
                cap = 192
                ppl = 1 if ppl >= cap else (ppl / cap)
                ppl = [ppl]
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, ppl=ppl))

            else:
                examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[2]
            text_b = None

            label = line[1]
            if self.is_binary:
                # map to 6 label to binary label
                label = self.multi2binary[label]
            
            if self.output_mode == 'regression':
                label = CLASSIFICATION_TO_REGRESSION[label]

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


class FusionProcessor(DataProcessor):
    def __init__(self, args):
        self.politifact_dataset = PolifactProcessor(args)
        self.rte_dataset = RteProcessor(args)
        self.labels = self.politifact_dataset.get_labels()

    def get_train_examples(self, data_dir, data_source=""):
        politifact = self.politifact_dataset.get_train_examples(data_dir, data_source)
        rte = self.rte_dataset.get_train_examples(data_dir, data_source)
        fusion = politifact + rte

        return fusion

    def get_dev_examples(self, data_dir, data_source=""):
        politifact = self.politifact_dataset.get_dev_examples(data_dir, data_source)
        rte = self.rte_dataset.get_dev_examples(data_dir, data_source)
        fusion = politifact + rte

        return fusion

    def get_test_examples(self, data_dir, data_source=""):
        """Since we only care about politifact performance, only use politifact-test"""
        politifact = self.politifact_dataset.get_test_examples(data_dir, data_source)
        # rte = self.rte_dataset.get_test_examples(data_dir, data_source)
        # fusion = politifact + rte

        return politifact

    def get_labels(self):
        """See base class."""
        return self.labels


factcheck_processors = {
    "rte": RteProcessor,
    "scitail": SciTailProcessor,
    'gpt2baseline': PolifactProcessor,
    'politifact': PolifactProcessor,
    'fusion': FusionProcessor
}
