# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from model.graphattnnet import GraphAttentionLayer, GAT
from model.textcnnnet import textCNN
from model.selftattnnet import SelfAttentionLayer, SAT
from transformers import *
import numpy as np


class Config(object):
    """配置参数"""

    def __init__(self):
        self.model_name = 'bert-base-uncased'
        self.all_sample_path = 'data/all.json'  # 所有样本路径
        self.resplit_dataset = False  # 是否重新划分数据
        self.train_path = 'data/train.json'  # 训练集
        self.dev_path = 'data/dev.json'  # 验证集
        self.test_path = 'data/test.json'  # 测试集
        self.save_path = 'saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # 设备
        self.require_improvement = 500  # 若超过1000batch效果还没提升，则提前结束训练
        # self.num_sentence_classes = 100                                 # 句子的最大长度
        self.num_epochs = 100  # epoch数
        self.batch_size = 2  # mini-batch大小
        self.word_pad_size = 20  # 每句话处理成的长度(短填长切)
        self.sentence_pad_size = 5  # 每个文档处理成的长度(短填长切)
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_hidden_size = 768
        self.dropout = 0.1
        self.rnn_hidden_size = 128
        self.hidden_size = 436
        self.gat_hidden_size = 128
        self.rnn_num_layers = 2
        self.knum = [512, 256, 128]
        self.ksize = [64]
        self.Cla = 128
        self.pooling_size = 10
        self.d_window = 6
        self.focal_alpha = 0.9
        self.focal_gamma = 2
        self.focal_logits = False
        self.focal_reduce = True
        self.feat_size = 52


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        """
        bert layer
        """
        # model config (Model,Tokenizer, pretrained weights shortcut
        self.model_config = (BertModel, BertTokenizer, "bert-base-uncased")
        self.bert_tokenizer = self.model_config[1].from_pretrained(self.model_config[2])
        self.bert = self.model_config[0].from_pretrained(self.model_config[2])
        # for param in self.bert.parameters():
        #     param.requires_grad = True

        "dropout layer"
        self.dropout = nn.Dropout(config.dropout)
        "textcnn layer"
        self.textcnn = textCNN(config.bert_hidden_size, config.knum, config.ksize, config.dropout, config.Cla,
                               config.pooling_size)
        "graph attention layer"
        self.gat = GAT(config.bert_hidden_size, config.bert_hidden_size, config.gat_hidden_size, config.dropout, 0.2, 4,
                       config.batch_size)
        self.sat = SAT(config.bert_hidden_size, config.bert_hidden_size, config.gat_hidden_size, config.dropout, 0.2, 4,
                       config.batch_size)

        "issue classifier layer"
        self.issue_classifier = nn.Linear(config.hidden_size, 1)
        "solution classifier layer"
        self.solution_classifier = nn.Linear(config.hidden_size, 1)
        # self.sentence_classifier = nn.Linear(config.rnn_hidden_size * 2, config.sentence_pad_size)
        self.batch_size = config.batch_size
        self.sentence_pad_size = config.sentence_pad_size

    def forward(self, samples, graphs, feat_vec):
        x_list = samples[0]
        mask_list = samples[1]
        bert_hidden_tensor = None
        for index, x in enumerate(x_list):
            x_tensor = torch.tensor(x).to(self.config.device)
            mask_tensor = torch.tensor(mask_list[index]).to(self.config.device)
            # pooled or average???
            poolled_output = self.bert(x_tensor, attention_mask=mask_tensor)[1].unsqueeze(0)  # Batch size 1
            if bert_hidden_tensor is None:
                bert_hidden_tensor = poolled_output
            else:
                bert_hidden_tensor = torch.cat((bert_hidden_tensor, poolled_output), 0)

        shape = bert_hidden_tensor.size()

        "Feature Layer"
        out_feat = torch.tensor(feat_vec).cuda()

        "TextCNN Layer"
        out_cnn = self.textcnn(bert_hidden_tensor)

        "SequenceAttention Layer"
        sequence_weight_tensor = self.init_weight_matrix(bert_hidden_tensor.size()[0], bert_hidden_tensor.size()[1],
                                                         self.config.d_window)
        out_sat = self.sat(bert_hidden_tensor, sequence_weight_tensor, self.config.d_window)

        "GraphAttention Layer"
        graph_tensor = torch.tensor(graphs).cuda()
        out_gat = self.gat(bert_hidden_tensor, graph_tensor, self.config.d_window)

        out = torch.cat([out_feat, out_cnn, out_sat, out_gat], dim=2)
        # TASK-SPECIFIC LAYER
        issue_out = self.issue_classifier(out)
        solution_out = self.solution_classifier(out)

        issue_out = issue_out.squeeze()
        solution_out = solution_out.squeeze()
        # issue_out = issue_out.view(-1, 3)
        # solution_out = solution_out.view(-1, 3)
        # issue_out = issue_out.reshape(self.batch_size * self.sentence_pad_size, 2)
        # solution_out = solution_out.reshape(self.batch_size * self.sentence_pad_size, 2)

        issue_out = torch.sigmoid(issue_out)
        solution_out = torch.sigmoid(solution_out)
        return issue_out, solution_out

    def init_weight_matrix(self, n_bat, n_utter, d_window):
        # print(n_utter)
        weight_matrix = [
            [[np.exp(-((i - j) ** 2) / (2 * (d_window / 2) ** 2)) for j in range(n_utter)] for i in range(n_utter)]]
        weight_matrix_tensor = torch.tensor(weight_matrix)
        weight_matrix_tensor = weight_matrix_tensor.repeat(n_bat, 1, 1)
        return weight_matrix_tensor.cuda()
