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


def mkdict(words, save_file):
    """
    候选答案里的都为单独的词
    :param dict:
    :return:
    """
    word_counter = Counter(words)
    with open(save_file, "w", encoding="utf-8") as fh:
        for k, v in word_counter.items():
            fh.write(k+" "+str(v)+"\n")


def backlabels(alter, label):
    labels = [0, 0, 0]
    cnt = 0
    for key in alter:
        if label == key:
            labels[cnt] = 1
        cnt += 1
    return labels


def sort_alter(alter):
    new_alter = ['', '', '']
    flag = {}
    # 是否丢弃样本
    cast = True
    for word in alter:
        flag[word] = 0
        if word == "无法确定":
            flag[word] = 2
            continue
        for char in word:
            if char == "不" or char == "没" or char == "否" or char == "无" or char == "假":
                flag[word] = 1
                cast = False
                break
    if ~cast:
        for key in flag:
            new_alter[flag[key]] = key

    return new_alter, cast


def mksegs(file, after_process_file, data_size, is_test=False):
    """
    分词
    :param file:
    :param after_process_file:
    :param data_size:
    :return:
    """
    examples = []
    dict=["能", "可以"]
    cast_total = 0
    with open(file, "r", encoding="utf-8") as fh:
        for line in tqdm(fh, total=data_size):
            sample = json.loads(line)
            if is_test:
                answer = ""
            else:
                answer = sample["answer"]

            alternatives = sample['alternatives'].split("|")
            # random.shuffle(alternatives)
            # 重新排序选项
            alternatives, cast = sort_alter(alternatives)
            if cast:
                cast_total += 1
                continue
            query = wordseg(sample["query"], alternatives)
            passage = wordseg(sample["passage"], alternatives)
            query_id = sample["query_id"]
            example = {"query_id": query_id, "passage": passage, "query": query, "alternatives": alternatives,
                       "answer": answer}
            examples.append(example)
        print("丢弃样本数：%d" % cast_total)
    save(after_process_file, examples, message="seg")


def preprosses_file(train_file, word_counter, data_size, is_test=False):
    examples = []
    eval_examples = {}
    para_limit = 0
    query_limit = 0
    total = 0
    with open(train_file, "r", encoding="utf-8") as fh:
        extra_words = []
        samples = json.load(fh)
        for sample in tqdm(samples, total=data_size):

            # print("test")
            query = sample["query"]
            if query_limit < len(query):
                query_limit = len(query)

            for word in query:
                word_counter[word] += 1
            passage = sample["passage"]
            if len(passage) > 1000:
                # print(passage)
                # print(sample["query_id"])
                continue
            if para_limit < len(passage):
                para_limit = len(passage)

            for word in passage:
                word_counter[word] += 1

            query_id = sample["query_id"]
            alternatives = sample['alternatives']
            if is_test:
                # 随机选第一个为候选答案
                answer = alternatives[0]
            else:
                answer = sample["answer"]

            for word in alternatives:
                word_counter[word] += 1
            # [001] [010] [100]
            labels = backlabels(alternatives, answer)
            if len(alternatives) == 2:
                print(query_id)
                print("answer size is 2!!")
                continue
            for word in alternatives:
                extra_words.append(word)
                word_counter[word] += 1
            example = {"passage": passage, "query": query, "alternatives": alternatives, "labels": labels, "id": total}
            examples.append(example)
            eval_examples[str(total)] = {"query_id": query_id, "alternatives": alternatives, "labels": labels}
            total += 1
            # if total>100:
            #     break
    random.shuffle(examples)

    return examples, eval_examples, para_limit, query_limit

    # mkdict(extra_words)


def get_embedding(counter, data_type, limit=-1, emb_file=None, token2idx_dict=None):
    print("Generating {} embedding...".format(data_type))
    dim = 0
    embedding_dict = {}
    # 获取词频为limit以上的词的向量
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        # 预存的词向量文件
        with open(emb_file, "r", encoding="utf-8",  errors='ignore') as fh:
            # 打开词向量文件
            line = fh.readline()
            dim = int(line.rstrip().split()[1])
            total = int(line.rstrip().split()[0])
            for line in tqdm(fh, total=total):
                # if first_line:
                #     first_line = False
                #     dim = int(line.rstrip().split()[1])
                #     print(line)
                #     continue
                array = line.split()
                word = "".join(array[0:-dim])
                vector = list(map(float, array[-dim:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert dim is not None
        for token in filtered_elements:
            # 若无词向量文件，则对于每个词都初始化一个vec_size长度的初始向量 {token1:[0.01,....,]}
            embedding_dict[token] = [np.random.normal(
                scale=0.01) for _ in range(dim)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"   # 未登录词
    OOV = "--OOV--"     #
    # 每个词对应的id token:id
    token2idx_dict = {token: idx for idx, token in enumerate(
        embedding_dict.keys(), 2)} if token2idx_dict is None else token2idx_dict
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(dim)]
    embedding_dict[OOV] = [0. for _ in range(dim)]
    # 每个词id对应的词向量 id:vec
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    # 词向量矩阵 每行索引对应这个词的id
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_features(para_limit, ques_limit, examples, data_type, out_file, word2idx_dict, is_test=False):
    """

    :param para_limit:
    :param ques_limit:
    :param examples:
    :param data_type:
    :param out_file:
    :param word2idx_dict:
    :param is_test:
    :return:
    """

    def filter_func(example, is_test=False):
        return len(example["passage"]) > para_limit or len(example["query"]) > ques_limit

    print("Processing {} examples...".format(data_type))
    writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example, is_test):
            continue

        total += 1
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        alternatives = np.zeros([3], dtype=np.int32)
        y = np.zeros([3], dtype=np.float32)

        def _get_word(word):
            if word in word2idx_dict:
                return word2idx_dict[word]
            else:
                return 1

        # 获取每个token的id [wid1,wid2,..]
        for i, token in enumerate(example["passage"]):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["query"]):
            ques_idxs[i] = _get_word(token)

        for i, token in enumerate(example["alternatives"]):
            alternatives[i] = _get_word(token)

        # 取最后一个答案作为y值
        for i, val in enumerate(example["labels"]):
            y[i] = val

        record = tf.train.Example(features=tf.train.Features(feature={
            "context_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[context_idxs.tostring()])),
            "ques_idxs": tf.train.Feature(bytes_list=tf.train.BytesList(value=[ques_idxs.tostring()])),
            "alternatives": tf.train.Feature(bytes_list=tf.train.BytesList(value=[alternatives.tostring()])),
            "y": tf.train.Feature(bytes_list=tf.train.BytesList(value=[y.tostring()])),
            "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[example["id"]])),
        }))
        writer.write(record.SerializeToString())
    print("Build {} / {} instances of features in total".format(total, total_))
    meta["total"] = total
    writer.close()
    return meta


def save(filename, obj, message=None):
    if message is not None:
        print("Saving {}...".format(message))
        with open(filename, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False)


def prepro(config):
    # 分词
    # TODO 添加额外的词库
    # jieba.load_userdict(save_dict_file)
    # mksegs(config.train_file, config.save_train_file, 250000)
    # mksegs(config.dev_file, config.save_validation_file, 30000)
    # mksegs(config.test_file, config.save_testa_file, 10000, is_test=True)
    # 获取样本
    word_counter = Counter()
    examples, eval_examples, para_limit, query_limit = preprosses_file(config.save_train_file, word_counter, 250000)
    examples_val, eval_examples_val, para_limit_val, query_limit_val = preprosses_file(config.save_validation_file,
                                                                                       word_counter, 3000)
    examples_testa, eval_examples_testa, para_limit_testa, query_limit_testa = preprosses_file(config.save_testa_file,
                                                                                       word_counter, 10000, is_test=True)

    # 获取词向量
    word2idx_dict = None
    if os.path.isfile(config.word2idx_file):
        with open(config.word2idx_file, "r") as fh:
            word2idx_dict = json.load(fh)
    word_emb_mat, word2idx_dict = get_embedding(word_counter, "word", emb_file=config.wd_file, token2idx_dict=word2idx_dict)

    # 构建tensorflow特征文件

    build_features(1000, 100, examples, "train", config.train_record_file, word2idx_dict)
    dev_meta = build_features(1000, 100, examples_val, "validation", config.dev_record_file, word2idx_dict)
    test_meta = build_features(1000, 100, examples_testa, "testa", config.test_record_file, word2idx_dict)

    # 保存
    save(config.word_emb_file, word_emb_mat, message="word embedding")
    save(config.train_eval_file, eval_examples, message="train eval")
    save(config.dev_eval_file, eval_examples_val, message="dev eval")
    save(config.test_eval_file, eval_examples_testa, message="test eval")
    save(config.word2idx_file, word2idx_dict, message="word2idx")
    save(config.dev_meta, dev_meta, message="dev_meta")
    save(config.test_meta, test_meta, message="test_meta")
    # print("param:", para_limit)
    # # print(query_limit)
    print("prepro success!")



