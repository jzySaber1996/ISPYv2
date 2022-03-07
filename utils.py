# coding: UTF-8
from random import shuffle
import json
import os
import time
from datetime import timedelta
import torch.nn.functional as F
import torch
import torch.nn as nn
from urllib import request, parse
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def samples_statistics(project):
    """
    统计基本信息
    :return:
    """
    all_sample = 0
    feature_request = 0
    expected = 0
    current = 0
    benefit = 0
    drawback = 0
    example = 0
    explanation = 0
    useless = 0

    with open('data/all.json', 'r') as f_obj:
        samples = json.load(f_obj)
        for sample in samples:
            sentence_samples = sample['sentence_samples']
            document_label = sample['document_label']
            project_name = sample['project_name']
            if project_name == project:
                all_sample += 1
            else:
                continue
            if document_label == 'fr':
                feature_request += 1
            for index, sentence_info in enumerate(sentence_samples):
                sentence_label = sentence_info['sentence_label']
                if sentence_label == 'expected':
                    expected += 1
                elif sentence_label == 'current':
                    current += 1
                elif sentence_label == 'benefit':
                    benefit += 1
                elif sentence_label == 'drawback':
                    drawback += 1
                elif sentence_label == 'example':
                    example += 1
                elif sentence_label == 'explanation':
                    explanation += 1
                elif sentence_label == 'useless':
                    useless += 1

    print('expected: ', expected)
    print('current: ', current)
    print('benefit: ', benefit)
    print('drawback: ', drawback)
    print('example: ', example)
    print('explanation: ', explanation)
    print('useless: ', useless)
    print('TOTAL', all_sample)
    print(feature_request)


# samples_statistics('hibernate')

def build_dataset(config):
    # 打开data/all.json数据
    with open('data/data_raw.json', 'r', encoding='utf8') as file_obj:
        # 装载json格式到all_samples所有样本中
        all_samples = json.load(file_obj)
    # 如果config中重新划分数据集为true（目前是false），或者data目录下缺少train,dev,test.json的任何一个，则划分数据集。
    if config.resplit_dataset \
            or (not os.path.exists('data/train.json') or not os.path.exists('data/dev.json') or not os.path.exists(
        'data/test.json')):
        split_dataset(all_samples)

    def load_dataset(path, sentence_pad_size, word_pad_size):
        # 定义变量来存放结果
        padding_samples = []
        # 打开训练/验证/测试集文件
        with open(path, 'r') as f_obj:
            # 把json装在samples变量中
            samples = json.load(f_obj)
            # 在samples中取出一个sample
            for sample in samples:
                # contents初始为空
                contents = []
                # sampled sentences
                sentence_samples = sample['sentence_samples']
                # relations
                relations = sample['relations']
                # padding word
                # 获取json中sentence_samples的每一个句子
                for index, sentence_info in enumerate(sentence_samples):
                    if index == config.sentence_pad_size:
                        break
                    speaker = sentence_info['speaker']
                    sentence = sentence_info['sentence']
                    issue_label = sentence_info['issue_label']
                    solution_label = sentence_info['solution_label']
                    feat_vec = feature_vec(index, speaker, sentence, sentence_samples, config)
                    tokens = config.tokenizer.tokenize(sentence)
                    mask = []
                    token_ids = config.tokenizer.convert_tokens_to_ids(tokens)
                    if word_pad_size:
                        if len(token_ids) < word_pad_size:
                            mask = [1] * len(token_ids) + [0] * (word_pad_size - len(token_ids))
                            token_ids += ([0] * (word_pad_size - len(token_ids)))
                        else:
                            mask = [1] * word_pad_size
                            token_ids = token_ids[:word_pad_size]
                    contents.append((token_ids, mask, issue_label, solution_label, feat_vec))
                    # padding sentence
                if len(sentence_samples) < sentence_pad_size:
                    for _ in range(0, sentence_pad_size - len(contents)):
                        contents.append(([0] * word_pad_size, [0] * word_pad_size, 0, 0, [0] * config.feat_size))
                if len(relations) > sentence_pad_size:
                    relations = [relations[i][:sentence_pad_size] for i in range(sentence_pad_size)]
                elif len(relations) < sentence_pad_size:
                    relation_matrix = [[1 if c_index == r_index else 0 for c_index in range(sentence_pad_size)]
                                       for r_index in range(sentence_pad_size)]
                    for i_raw in range(len(relations)):
                        for j_raw in range(len(relations)):
                            relation_matrix[i_raw][j_raw] = relations[i_raw][j_raw]
                    for i_raw in range(len(relations)):
                        for j_raw in range(len(relations)):
                            if relation_matrix[i_raw][j_raw] == 1 and relation_matrix[j_raw][i_raw] != 1:
                                relation_matrix[j_raw][i_raw] = relation_matrix[i_raw][j_raw]
                    relations = relation_matrix
                padding_samples.append((contents, relations))
        return padding_samples

    # 通过split_dataset函数，train_path,dev_path,test_path路径都有对应的数据文件了，之后装在变量中。
    train = load_dataset(config.train_path, config.sentence_pad_size, config.word_pad_size)
    dev = load_dataset(config.dev_path, config.sentence_pad_size, config.word_pad_size)
    test = load_dataset(config.test_path, config.sentence_pad_size, config.word_pad_size)
    return train, dev, test


def split_dataset(samples):
    """
    split the dataset into train, dev and test (evaluation)
    :return:
    """
    # 对样本随机排序
    shuffle(samples)
    # 前80%是训练集
    train = samples[: int(len(samples) * 0.6)]
    # 80%-90%是验证集
    dev = samples[int(len(samples) * 0.6): int(len(samples) * 0.8)]
    # 最后10%是测试集
    test = samples[int(len(samples) * 0.8):]
    # 打开train.json，写入数据
    with open('data/train.json', 'w') as file_obj:
        json.dump(train, file_obj)
    # 打开dev.json，写入数据
    with open('data/dev.json', 'w') as file_obj:
        json.dump(dev, file_obj)
    # 打开test.json，写入数据
    with open('data/test.json', 'w') as file_obj:
        json.dump(test, file_obj)


def build_iterator(dataset, config):
    """
    build batches for training, developing and evaluation
    :param dataset:
    :param config:
    :return:
    """
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        sentences_infos = [_[0] for _ in datas]
        x = []
        mask = []
        issue_label = []
        solution_label = []
        feats = []
        for row in sentences_infos:
            x.append([triple[0] for triple in row])
            mask.append([triple[1] for triple in row])
            issue_label.append([triple[2] for triple in row])
            solution_label.append([triple[3] for triple in row])
            feats.append([triple[4] for triple in row])
        graph_matrix = [_[1] for _ in datas]
        return (x, mask), (issue_label, solution_label), feats, graph_matrix

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def focal_loss(inputs, targets, alpha=1, gamma=2, logits=False, reduce=True):
    targets = torch.tensor(targets, dtype=torch.float).cuda()
    if logits:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
    else:
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
    # avg_bce = torch.sum(BCE_loss)
    # return avg_bce
    # BCE_loss = F.cross_entropy(inputs, targets)
    alpha_ones = alpha * torch.ones_like(targets).cuda()
    alpha_zeros = (1 - alpha) * torch.ones_like(targets).cuda()
    alpha_t = torch.where(targets > 0, alpha_ones, alpha_zeros)
    pt = torch.exp(-BCE_loss)
    F_loss = alpha_t * (1 - pt) ** gamma * BCE_loss
    # F_loss = alpha_t * BCE_loss
    if reduce:
        return torch.mean(F_loss)
    else:
        return torch.sum(F_loss)


# 有道翻译：中文→英文
def fy(i):
    req_url = 'http://fanyi.youdao.com/translate'  # 创建连接接口
    # 创建要提交的数据
    Form_Date = {}
    Form_Date['i'] = i
    Form_Date['doctype'] = 'json'
    Form_Date['form'] = 'AUTO'
    Form_Date['to'] = 'AUTO'
    Form_Date['smartresult'] = 'dict'
    Form_Date['client'] = 'fanyideskweb'
    Form_Date['salt'] = '1526995097962'
    Form_Date['sign'] = '8e4c4765b52229e1f3ad2e633af89c76'
    Form_Date['version'] = '2.1'
    Form_Date['keyform'] = 'fanyi.web'
    Form_Date['action'] = 'FY_BY_REALTIME'
    Form_Date['typoResult'] = 'false'

    data = parse.urlencode(Form_Date).encode('utf-8')  # 数据转换
    response = request.urlopen(req_url, data)  # 提交数据并解析
    html = response.read().decode('utf-8')  # 服务器返回结果读取
    # print(html)
    # 可以看出html是一个json格式
    translate_results = json.loads(html)  # 以json格式载入
    translate_results = translate_results['translateResult'][0][0]['tgt']  # json格式调取
    # print(translate_results)  # 输出结果
    return translate_results  # 返回结果


def feature_vec(pos, speaker, utterance, sentence_samples, config):
    feat_vec = list()
    # 5W1H
    wh = ['what', 'why', 'when', 'who', 'which', 'how']
    feat_vec += [1 if each_wh in utterance else 0 for each_wh in wh]
    # Punctuation
    punc = ['?', '!']
    feat_vec += [1 if each_punc in utterance else 0 for each_punc in punc]
    # Greetings
    greets = ['Hello', 'Hi', 'Morning', 'Hey']
    feat_vec += [1 if each_greets in utterance else 0 for each_greets in greets]
    # Disapproval
    disapps = ['no', 'n\'t', 'fail', 'break', 'error', 'wrong']
    feat_vec += [1 if each_disapps in utterance else 0 for each_disapps in disapps]
    # Mention
    mentions = ['sim', 'same']
    feat_vec += [1 if each_mentions in utterance else 0 for each_mentions in mentions]

    # NT
    nt_vec = [0] * config.word_pad_size
    nt_pos = len(utterance.split(' ')) if len(utterance.split(' ')) < config.word_pad_size else config.word_pad_size
    nt_vec[nt_pos - 1] = 1
    feat_vec += nt_vec
    # AP
    ap_vec = [0] * config.sentence_pad_size
    ap_pos = pos if pos < config.sentence_pad_size else config.sentence_pad_size
    ap_vec[ap_pos] = 1
    feat_vec += ap_vec
    # RP
    feat_vec += [(pos + 1) / len(sentence_samples)]

    # SS
    sid = SIA()
    ss_dict = sid.polarity_scores(utterance)
    ss = ['neg', 'neu', 'pos', 'compound']
    feat_vec += [ss_dict[each_ss] for each_ss in ss]

    # Role
    roles = [0] * 2
    if speaker == sentence_samples[0]['speaker']:
        roles[0] = 1
    else:
        roles[1] = 1
    feat_vec += roles
    return feat_vec
