from __future__ import unicode_literals
import json
import os
import jieba
from collections import Counter
import numpy as np
from tqdm import tqdm
import random
import tensorflow as tf


def wordseg(sentence, alter):
    """
    分词函数
    :return:
    """
    jieba.suggest_freq(alter, True)
    seg_list = jieba.cut(sentence, cut_all=False)
    return " ".join(seg_list).split(" ")


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False)


def if_opinion(alter):
    for word in alter[:2]:
        for char in word:
            if char == "不" or char == "没" or char == "否" or char == "无" or char == "假":
                return True
    # print(alter)
    return False


def mksegs(file, after_process_file, data_size, is_test=False):
    """
    分词
    :param file:
    :param after_process_file:
    :param data_size:
    :return:
    """
    examples = []
    dict=Counter()
    total = 0
    alter = []
    with open(file, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, total=data_size):
            sample = json.loads(line)
            alternatives = sample['alternatives'].split("|")
            for word in alternatives:
                dict[word] += 1
            if if_opinion(alternatives):
                total += 1
            else:
                alter.append(alternatives)
            # total += 1 if if_opinion(alternatives) else 0

    print(total)
    dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)

    with open("data/extract.txt", "w", encoding="utf-8") as fh:
        for line in alter:
            fh.write(" ".join(line)+"\n")
    with open(after_process_file, "w", encoding="utf-8") as fh:
        for tp in dict:
            fh.write(tp[0]+"\t"+str(tp[1])+"\n")


if __name__ == '__main__':
    datadir = os.path.join("F:\\", "data")
    train_file = os.path.join(datadir, "ai_challenger_oqmrc_train", "ai_challenger_oqmrc_trainingset.json")
    validation_file = os.path.join(datadir, "ai_challenger_oqmrc_validation", "ai_challenger_oqmrc_validationset.json")
    test_file = os.path.join(datadir, "ai_challenger_oqmrc_test", "ai_challenger_oqmrc_testa.json")

    # mksegs(train_file, "data/train_alter.txt", 250000)
    # mksegs(validation_file, "data/dev_alter.txt", 30000)
    mksegs(test_file, "data/test_alter.txt", 10000)