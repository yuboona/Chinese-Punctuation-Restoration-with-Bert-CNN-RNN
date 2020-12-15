import torch
from torch.autograd import Variable
from torch import nn
from transformers import BertForMaskedLM, DistilBertForMaskedLM, DistilBertModel, AutoModel
from transformers import AutoModelWithLMHead, AlbertModel, AlbertForMaskedLM
from transformers import AlbertForSequenceClassification

class BertPunc(nn.Module):  

    def __init__(self, segment_size, output_size, dropout):
        super(BertPunc, self).__init__()
        self.bert = BertForMaskedLM.from_pretrained('./models/')
        self.bert_vocab_size = 30522
        self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        print(x.shape)
        x = self.bert(x)[0]
        print('x', type(x))

        x = x.view(x.shape[0], -1)
        x = self.fc(self.dropout(self.bn(x)))
        return x


class DistillBertPunc(nn.Module):  
    
    def __init__(self, segment_size, output_size, dropout):
        super(DistillBertPunc, self).__init__()
        self.bert = DistilBertForMaskedLM.from_pretrained('./models/distillbert/')
        self.bert_vocab_size = 30522
        self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        print(x.shape)
        x = self.bert(x)[0]
        print('x', type(x))

        x = x.view(x.shape[0], -1)
        x = self.fc(self.dropout(self.bn(x)))
        return x


class ALBertPunc(nn.Module):  
    
    def __init__(self, segment_size, output_size, dropout):
        super(ALBertPunc, self).__init__()
        self.bert = AutoModelWithLMHead.from_pretrained('./models/albert_en/')
        self.bert_vocab_size = 30000
        # 去掉一些批标准化部分，减少计算量
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print(x.shape)
        x = self.bert(x)[0]
        # print('x', type(x))

        x = x.view(x.shape[0], -1)
        x = self.fc(self.dropout(x))
        return x


class BartPunc(nn.Module):  
    
    def __init__(self, segment_size, output_size, dropout):
        super(BartPunc, self).__init__()
        self.bert = AutoModelWithLMHead.from_pretrained('./models/bart_tiny/')
        self.bert_vocab_size = 50265
        self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print(x.shape)
        x = self.bert(x)[0]
        # print('x', type(x))

        x = x.view(x.shape[0], -1)
        x = self.fc(self.dropout(self.bn(x)))
        return x


class ALBertSmallPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ALBertSmallPunc, self).__init__()
        self.bert = AlbertModel.from_pretrained('./models/albert_chinese_small/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)

        # 使用bert中间层hidden_state 384
        self.fc = nn.Linear(segment_size*384, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 修改后
        x = self.bert(x)[0]
        # 原始版
        # x = self.bert(x)
        # l = len(x)
        # for i in range(l):
        #     print('{} shape:'.format(i), x[i].shape)
        # print('x', type(x))

        x = x.view(x.shape[0], -1)
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        return x


class BertDistillHiddenPunc(nn.Module): 
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertDistillHiddenPunc, self).__init__()
        self.bert = AutoModel.from_pretrained('./models/bert_distill_chinese')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)

        # 使用distill_bert中间层hidden_state 768
        self.fc = nn.Linear(segment_size*768, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 修改后
        x = self.bert(x)[0]
        # 原始版
        # x = self.bert(x)
        # l = len(x)
        # for i in range(l):
        #     print('{} shape:'.format(i), x[i].shape)
        # print('x', type(x))

        x = x.view(x.shape[0], -1)
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        return x


class ALBertSmallRNNPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ALBertSmallRNNPunc, self).__init__()
        self.bert = AutoModel.from_pretrained('./models/albert_chinese_small/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)

        # 批标准化
        self.bn = nn.BatchNorm1d(segment_size*384)
        self.gru = nn.GRU(384, 384, 2, bidirectional=True, batch_first=True)
        # NOTE rnn_hidden*2 使用bert中间层hidden_state 384
        self.fc = nn.Linear(segment_size*384*2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 修改后
        x = self.bert(x)[0]
        # 原始版
        # x = self.bert(x)
        shape1 = x.shape[1]
        shape2 = x.shape[2]
        x = x.view(x.shape[0], -1)
        x = self.bn(x)
        x = x.view(-1, shape1, shape2)

        # GRU*************************************
        # init_GRU_hidden
        def init_gru_hidden(batch_size):
            # when NOT bidirection (layer, B, H)
            h = Variable(torch.zeros(self.gru.num_layers*2, batch_size, self.gru.hidden_size))
            # h for storing hidden layer weight
            return h

        hidden = init_gru_hidden(x.shape[0])
        x, hidden = self.gru(x, hidden)
        x = x.contiguous()
        # ***************************************
        # l = len(x)
        # for i in range(l):
        #     print('{} shape:'.format(i), x[i].shape)
        # print('x', type(x))
        # print('shape', x.shape)
        x = x.view(x.shape[0], -1)
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        return x


class ALBertSmallRNNnewLinearPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ALBertSmallRNNnewLinearPunc, self).__init__()
        self.bert = AutoModel.from_pretrained('./models/albert_chinese_small/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)

        # 批标准化
        self.bn = nn.BatchNorm1d(segment_size*384)
        self.gru = nn.GRU(384, 384, 2, bidirectional=True, batch_first=True)
        # NOTE rnn_hidden*2 使用bert中间层hidden_state 384
        self.fc = nn.Linear(384*2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 修改后
        x = self.bert(x)[0]
        # 原始版
        # x = self.bert(x)
        shape1 = x.shape[1]
        shape2 = x.shape[2]
        x = x.view(x.shape[0], -1)
        x = self.bn(x)
        x = x.view(-1, shape1, shape2)

        # GRU*************************************
        # init_GRU_hidden
        def init_gru_hidden(batch_size):
            # when NOT bidirection (layer, B, H)
            h = Variable(torch.zeros(self.gru.num_layers*2, batch_size, self.gru.hidden_size))
            # h for storing hidden layer weight
            return h

        hidden = init_gru_hidden(x.shape[0])
        x, hidden = self.gru(x, hidden)
        x = x.contiguous()
        # ***************************************
        # l = len(x)
        # for i in range(l):
        #     print('{} shape:'.format(i), x[i].shape)
        # print('x', type(x))
        # print('shape', x.shape)
        shape1 = x.shape[0]
        shape2 = x.shape[1]
        x = x.view(-1, x.shape[2])
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # 变回batch_size第一维
        x = x.view(shape1, -1)
        return x



# NOTE
class ALBertSmallDenseHiddenPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ALBertSmallDenseHiddenPunc, self).__init__()
        self.bert = AutoModelWithLMHead.from_pretrained('./models/albert_chinese_small/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.albert = self.bert.albert
        self.dense = self.bert.predictions.dense
        # 批标准化
        self.bn = nn.BatchNorm1d(segment_size*128)
        # NOTE rnn_hidden*2 使用bert中间层hidden_state 384
        self.fc = nn.Linear(segment_size*128, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 修改后
        # [B, S, H]
        x = self.albert(x)[0]
        x = self.dense(x)
        # 原始版
        # x = self.bert(x)
        shape1 = x.shape[1]
        shape2 = x.shape[2]
        print('******', shape1, shape2)
        x = x.view(x.shape[0], -1)
        x = self.bn(x)
        x = x.view(-1, shape1, shape2)

        x = x.view(x.shape[0], -1)
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        return x


class ALBertSmallDenseRnnPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ALBertSmallDenseRnnPunc, self).__init__()
        self.bert = AutoModelWithLMHead.from_pretrained('./models/albert_chinese_small/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.albert = self.bert.albert
        self.dense = self.bert.predictions.dense
        # 批标准化
        self.bn = nn.BatchNorm1d(segment_size*128)
        # NOTE rnn_hidden*2 使用bert中间层hidden_state 384
        # self.gru = nn.GRU(128, 128, 2, bidirectional=True, batch_first=True)
        self.gru = nn.GRU(128, 128, 2, bidirectional=False, batch_first=True)
        self.fc = nn.Linear(segment_size*128, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 修改后
        # [B, S, H]
        x = self.albert(x)[0]
        x = self.dense(x)
        # 原始版
        # x = self.bert(x)
        shape1 = x.shape[1]
        shape2 = x.shape[2]
        print('******', shape1, shape2)
        x = x.view(x.shape[0], -1)
        x = self.bn(x)
        x = x.view(-1, shape1, shape2)

        # GRU*************************************
        # init_GRU_hidden
        def init_gru_hidden(batch_size):
            # when NOT bidirection (layer, B, H)
            # 1、双向
            # h = Variable(torch.zeros(self.gru.num_layers*2, batch_size, self.gru.hidden_size))
            # 2、单向
            h = Variable(torch.zeros(self.gru.num_layers, batch_size, self.gru.hidden_size))
            # h for storing hidden layer weight
            return h

        hidden = init_gru_hidden(x.shape[0])
        x, hidden = self.gru(x, hidden)
        x = x.contiguous()

        x = x.view(x.shape[0], -1)
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        return x
