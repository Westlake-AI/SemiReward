
import csv
import os
import json
import numpy as np
import logging
import torch

from typing import List
from semilearn.datasets.utils import split_ssl_data
from .datasetbase import BasicDataset


def get_alter_of_dna_sequence(sequence: str):
    """Get the reversed complement of the original DNA sequence."""
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])


def generate_kmer_str(sequence: str, k: int) -> str:
    """Generate k-mer string from DNA sequence."""
    return " ".join([sequence[i:i+k] for i in range(len(sequence) - k + 1)])


def load_or_generate_kmer(data_path: str, texts: List[str], k: int) -> List[str]:
    """
    Load or generate k-mer string for each DNA sequence.
    The generated k-mer string will be saved to the same directory as the original data with the same name but with a suffix of "_{k}mer".
    """
    kmer_path = data_path.replace(".csv", f"_{k}mer.json")
    if os.path.exists(kmer_path):
        logging.warning(f"Loading k-mer from {kmer_path}...")
        with open(kmer_path, "r") as f:
            kmer = json.load(f)
    else:        
        logging.warning(f"Generating k-mer...")
        kmer = [generate_kmer_str(text, k) for text in texts]
        with open(kmer_path, "w") as f:
            logging.warning(f"Saving k-mer to {kmer_path}...")
            json.dump(kmer, f)
        
    return kmer


def load_csv_data(data_path: str, kmer: int = -1):
    # load data from the disk
    with open(data_path, "r") as f:
        data = list(csv.reader(f))[1:]
    if len(data[0]) == 2:
        # data is in the format of [text, label]
        texts = [d[0] for d in data]
        labels = [int(d[1]) for d in data]
    elif len(data[0]) == 3:
        # data is in the format of [text1, text2, label]
        texts = [[d[0], d[1]] for d in data]
        labels = [int(d[2]) for d in data]
    else:
        raise ValueError("Data format not supported.")

    if kmer != -1:
        # only write file on the first process
        if torch.distributed.get_rank() not in [0, -1]:
            torch.distributed.barrier()
        logging.warning(f"Using {kmer}-mer as input...")
        texts = load_or_generate_kmer(data_path, texts, kmer)
        if torch.distributed.get_rank() == 0:
            torch.distributed.barrier()

    return texts, labels


def get_json_dset(args, alg='fixmatch', dataset='acmIb', num_labels=40, num_classes=2,
                  data_dir='./data', index=None, include_lb_to_ulb=True, onehot=False, kmer=-1):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
        
        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            onehot: If True, the target is converted into onehot vector.
        
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeled data)
        """

        train_data, train_label = load_csv_data(os.path.join(data_dir, dataset, "train.csv"), kmer=kmer)
        test_data, test_label = load_csv_data(os.path.join(data_dir, dataset, "test.csv"), kmer=kmer)
        dev_data, dev_label = load_csv_data(os.path.join(data_dir, dataset, "dev.csv"), kmer=kmer)

        json_dir = os.path.join(data_dir, dataset)
        # Supervised top line using all data as labeled data.
        with open(os.path.join(json_dir,'train.json'),'r') as json_data:
            train_data = json.load(json_data)
            train_sen_list = []
            train_label_list = []
            for idx in train_data:
                train_sen_list.append((train_data[idx]['ori'],train_data[idx]['aug_0'],train_data[idx]['aug_1']))
                train_label_list.append(int(train_data[idx]['label']))
        with open(os.path.join(json_dir,'dev.json'),'r') as json_data:
            dev_data = json.load(json_data)
            dev_sen_list = []
            dev_label_list = []
            for idx in dev_data:
                dev_sen_list.append((dev_data[idx]['ori'],'None','None'))
                dev_label_list.append(int(dev_data[idx]['label']))
        with open(os.path.join(json_dir,'test.json'),'r') as json_data:
            test_data = json.load(json_data)
            test_sen_list = []
            test_label_list = []
            for idx in test_data:
                test_sen_list.append((test_data[idx]['ori'],'None','None'))
                test_label_list.append(int(test_data[idx]['label']))

        dev_dset = BasicDataset(alg, dev_sen_list, dev_label_list, num_classes, False, onehot)
        test_dset = BasicDataset(alg, test_sen_list, test_label_list, num_classes, False, onehot)
        if alg == 'fullysupervised':
            lb_dset = BasicDataset(alg, train_sen_list, train_label_list, num_classes, False,onehot)
            return lb_dset, None, dev_dset, test_dset

        lb_sen_list, lb_label_list, ulb_sen_list, ulb_label_list = split_ssl_data(args, train_sen_list, train_label_list, num_classes, 
                                                                    lb_num_labels=num_labels,
                                                                    ulb_num_labels=args.ulb_num_labels,
                                                                    lb_imbalance_ratio=args.lb_imb_ratio,
                                                                    ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                    include_lb_to_ulb=include_lb_to_ulb)

        # output the distribution of labeled data for remixmatch
        count = [0 for _ in range(num_classes)]
        for c in train_label_list:
            count[c] += 1
        dist = np.array(count, dtype=float)
        dist = dist / dist.sum()
        dist = dist.tolist()
        out = {"distribution": dist}
        output_file = r"./data_statistics/"
        output_path = output_file + str(dataset) + '_' + str(num_labels) + '.json'
        if not os.path.exists(output_file):
            os.makedirs(output_file, exist_ok=True)
        with open(output_path, 'w') as w:
            json.dump(out, w)

        lb_dset = BasicDataset(alg, lb_sen_list, lb_label_list, num_classes, False, onehot)
        ulb_dset = BasicDataset(alg, ulb_sen_list, ulb_label_list, num_classes, True, onehot)
        return lb_dset, ulb_dset, dev_dset, test_dset
