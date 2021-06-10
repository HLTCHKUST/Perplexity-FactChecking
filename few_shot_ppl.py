import matplotlib.pyplot as plt
import numpy as np
import jsonlines
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import datetime

from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, \
    confusion_matrix

multi2binary = {
    "true" : "true",
    "mostly-true": "true",
    "half-true": "true",
    "barely-true": "false",
    "false": "false",
    "pants-fire": "false",
    "NOT ENOUGH INFO": "false",
    "REFUTES": "false",
    "SUPPORTS": "true"
}


def load_full_context_with_ppl(data_path, ppl_result_path, isbert=False):
    with jsonlines.open(data_path) as reader:
        og_objs = [obj for obj in reader]

    ppl_results = np.load(ppl_result_path, allow_pickle=True)

    all_objs = {
        'true': [],
        'false': [],
        '_': []
    }

    if 'train' in data_path: # === fever train ===== #
        for obj, ppl in zip(og_objs, ppl_results):
            label = multi2binary[obj['label']]
            claim_id = obj['id']
            claim = obj['claim']
            evs = obj['evidences'][:3]
            ppl = ppl['perplexity']

            if isbert:
                ppl = abs(ppl)

            new_objs = {'ppl': ppl, 'label': label, 'claim': claim, 'evidences': evs, 'claim_id': claim_id}
            all_objs[label].append(new_objs)

    else: # === fever test ===== #
        for obj in og_objs:
            label = multi2binary[obj['label']]
            claim_id = obj['id']

            claim = obj['claim']
            evs = obj['evidences'][:3]

            row_id = obj['row_id']
            ppl = ppl_results[row_id]['perplexity']

            if isbert:
                ppl = abs(ppl)

            new_objs = {'ppl': ppl, 'label': label, 'claim': claim, 'evidences': evs, 'claim_id': claim_id}
            all_objs[label].append(new_objs)

    return all_objs


def print_stat(ppls):
    print("Mean: {:.2f}, Std: {:.2f}".format(np.mean(ppls), np.std(ppls)))
    print("Min: {:.2f}, Max: {:.2f}".format(min(ppls), max(ppls)))
    print("Median", np.median(ppls))
    print("10 percentile: {:.2f}, 75 percentile: {:.2f}".format(np.percentile(ppls, 10), np.percentile(ppls, 75)))


def get_metric(objs, ppl_threshold, is_print=True, for_excel=False):

    preds = ['true' if float(obj['ppl']) < ppl_threshold else 'false' for obj in objs]
    golds = [obj['label'] for obj in objs]
    acc = accuracy_score(golds, preds)
    #     tn, fp, fn, tp = confusion_matrix(golds, preds).ravel()
    false_positive_count = 0
    for i in range(len(golds)):
        if golds[i] != preds[i]:
            if golds[i] == 'false':
                false_positive_count += 1

    f1_binary = f1_score(golds, preds, pos_label='false', average='binary')
    f1_macro = f1_score(golds, preds, pos_label='false', average='macro')
    recall = recall_score(golds, preds, pos_label='false', average='binary')
    precision = precision_score(golds, preds, pos_label='false', average='binary')

    if is_print:
        print(
            "TH: {}, Acc: {:.4f}, F1-macro: {:.4f},  F1-binary: {:.4f}, Recall:{:.4f}, Precision: {:.4f}, FN Count: {}". \
            format(ppl_threshold, acc, f1_macro, f1_binary, recall, precision, false_positive_count))
    if for_excel:
        print("for excel sheet: \n {},{},{},{},{},{}" \
              .format(acc, f1_macro, f1_binary, recall, false_positive_count, ppl_threshold))
    return {'acc': acc, 'f1_macro': f1_macro, 'f1_binary': f1_binary, \
            'recall': recall, 'fp': false_positive_count, 'th': ppl_threshold}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--covid_data", action="store_true", help="for our own covid dataset")
    parser.add_argument("--k", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--exp_name", default=None, type=str, required=True, help=""
    )
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=False, help="" )
    parser.add_argument(
        "--train_result_path", default=None, type=str, required=False, help="")
    parser.add_argument(
        "--aug_test_data_file", default=None, type=str, required=False, help="" )
    parser.add_argument(
        "--aug_test_result_path", default=None, type=str, required=False, help="")
    parser.add_argument(
        "--test_data_file", default=None, type=str, required=True, help="")
    parser.add_argument(
        "--test_result_path", default=None, type=str, required=True, help="")
    args = parser.parse_args()

    k = args.k  # number of shots
    isbert = True if 'bert' in args.test_result_path else False

	random_seeds = [random.randint(0,10000) for _ in range(3)]
	print(random_seeds)
	# keep it across the dataset and shot settig (USE SAME FOR ALL MODELS!!!! FAIR COMPARISON)

    for seed_ in random_seeds:
        random.seed(seed_)
        thresholds = [i for i in range(1000)]

        test_data_path = args.test_data_file
        test_eval_file = args.test_result_path

        if args.covid_data:
            all_objs = load_full_context_with_ppl(test_data_path, test_eval_file, isbert)
            combined_all_objs = all_objs['true'] + all_objs['false']
        
            random.shuffle(combined_all_objs)

            train_set = combined_all_objs[:k]
            test_set = combined_all_objs[k+1:]

        else: #FEVER
            test_all_objs = load_full_context_with_ppl(test_data_path, test_eval_file, isbert)
            test_combined_all_objs = test_all_objs['true'] + test_all_objs['false']
            test_set = test_combined_all_objs

            # making train set 
            train_all_objs = load_full_context_with_ppl(args.train_data_file, args.train_result_path, isbert)
            true_train_objs = train_all_objs['true']
            false_train_objs = train_all_objs['false']

            random.shuffle(true_train_objs)
            random.shuffle(false_train_objs)
            train_set = true_train_objs[:int(k/2)] + false_train_objs[:int(k/2)]

            print("Train set: {} | Test set: {}".format(len(train_set), len(test_set)))

        results = []
        for ppl_th in thresholds:
            result = get_metric(train_set, ppl_th, False)
            results.append(result)

        f1_macros = [o['f1_macro'] for o in results]
        max_macro = max(f1_macros)

        optimal_ths = []
        for r in results:
            if r['f1_macro'] == max_macro:
                optimal_th = r['th']
                get_metric(train_set, optimal_th, True)
                optimal_ths.append(optimal_th)

        opt_th = np.mean(optimal_ths)

        result_on_test = get_metric(test_set, opt_th, True)

        exp_name = args.exp_name
        log_path = 'results/few_shot_logs/{}_{}.txt'.format(exp_name, k)

        with open(log_path , "a") as writer:
            writer.write("\n%s\t%s\t%s\t%s" % (args.test_result_path.split("/")[-1].split(".")[-4], k, seed_, result_on_test))
