import torch
from torch.autograd import Variable
from torch import nn
from transformers import BertModel, BertForMaskedLM, DistilBertForMaskedLM, DistilBertModel, AutoModel
from transformers import AutoModelWithLMHead, AlbertModel, AlbertForMaskedLM
from transformers import AlbertForSequenceClassification
from model_1_to_1_seg import (
    SegBertChineseLinearPunc,
    SegRobertaChineseLSTMLinearPunc,
    )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


class EncoderStackedCnnBnLSTMLSTM(nn.Module):
    """添加额外CNN向量
        1、cnn提取特征，后RNN；
        2、bert直连LSTM
        1+2
    """
    def __init__(self, bert, hidden_size, n_layers, cnn_kernel_size, cnn_filter_num):
        super(EncoderStackedCnnBnLSTMLSTM, self).__init__()
        self.bert = bert
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num

        # GRU输入为bert + cnn_out
        # self.gru = nn.GRU(bert_size+(bert_size-cnn_kernel_size[1]+1)*cnn_filter_num, hidden_size, n_layers, bidirectional=True, batch_first=False)

        # cnn_kernel_size包含高度（多少词一个卷）和宽度（每个词的多少位一个卷）
        self.conv = nn.ModuleList()
        cnn_layer_num = 4
        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
                'bn_w_{}'.format(i): nn.BatchNorm2d(cnn_filter_num),
                'conv_v_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
                'bn_v_{}'.format(i): nn.BatchNorm2d(cnn_filter_num),
            })
            self.conv.append(module_tmp)

        # LSTM输入为bert + cnn_out
        self.lstm = nn.ModuleDict({
            'cnn_lstm': nn.LSTM(
                # cnn的每个词对应的输入尺寸为 cnn_layer_num=4层CNN，每层cnn_filter_num=10 filter
                input_size=(bert.embedding_size-((cnn_kernel_size[1]-1) % 2 * cnn_layer_num))*cnn_filter_num,
                hidden_size=hidden_size, num_layers=n_layers, bidirectional=True, batch_first=False
                ),
            'emb_lstm': nn.LSTM(
                bert.embedding_size,
                hidden_size,
                n_layers,
                bidirectional=True,
                batch_first=False
                ),
        })

    # @torchsnooper.snoop()
    def forward(self, word_inputs):
        # Note: we run this all at once (over the whole input sequence)
        # [S, B, Emb]
        embedded = self.bert(word_inputs)[0]

        # [S, B, Emb] -> [B, S, Emb]
        embedded_cnn = embedded.transpose(0, 1)

        # # 补上padding，使得conv得到seq_len个向量
        # batch_size = embedded_cnn.shape[0]
        # padding_size = self.cnn_kernel_size[0] - 1
        # # [B, padding_size, Emb]
        # pad_start = torch.zeros(batch_size, padding_size//2, self.bert_size).to(device)
        # pad_end = torch.zeros(batch_size, padding_size//2, self.bert_size).to(device)
        # # [B, S, Emb] -> [B, S+p_s, Emb]
        # embedded_cnn = torch.cat([pad_start, embedded_cnn, pad_end], dim=1)

        # [B, S+p_s, Emb] -> [B, Channel, S+p_s, Emb]
        embedded_cnn = embedded_cnn.unsqueeze(1)
        # cnn_out: [B, Channel, S, x] 向量维度未知
        """ print('cnn_out size', embedded_cnn.shape) """
        cnn_out = []
        cnn_out = embedded_cnn
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            w = conv_dict['bn_w_{}'.format(i)](w)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            v = conv_dict['bn_v_{}'.format(i)](v)
            cnn_out = w * torch.sigmoid(v)

        # for conv in self.conv:
        #     cnn_out_temp = conv(embedded_cnn)
        #     # cat cnn_out的filter_num维度
        #     # [B, Channel, S, x] -> [B, S, x]
        #     # filter_size = 2
        #     cnn_out.append(torch.cat([cnn_out_temp[:, i, :, :] for i in range(2)], dim=-1))
        # cnn_out = torch.cat(cnn_out, dim=-1)

        # 通过拼接 [B, Channel, S, x] -> [B, S, x]
        # print(cnn_out.shape)
        cnn_out = torch.cat([cnn_out[:, i, :, :] for i in range(self.cnn_filter_num)], dim=-1)
        # cnn_out = cnn_out.squeeze(1)
        # print(cnn_out.shape)
        # [B, S, x] -> [S, B, x]
        cnn_out = cnn_out.transpose(0, 1)
        """ print('cnn_out size', cnn_out.shape) """

        # 打印嵌入的 词向量shape 和 隐状态shape。
        # print('enc_emb size', embedded.shape)
        # print('hidden size', hidden.shape)
        cnn_lstm_input = cnn_out
        emb_lstm_input = embedded
        # output1 = [S, B, 2H]      hidden1=[lyr*direct, B, H]
        output1, hidden1 = self.lstm['cnn_lstm'](cnn_lstm_input, None)
        output2, hidden2 = self.lstm['emb_lstm'](emb_lstm_input, None)

        # 在第三维cat起来
        output = torch.cat([output1, output2], dim=-1)

        # output size: [S, B, 4H];      hidden: ([lyr*direct, B, H], [lyr*direct, B, H])
        return output


class EncoderStackedNmlCnnLSTMLSTM(nn.Module):
    """添加额外CNN向量
        1、cnn提取特征，后RNN；
        2、bert直连LSTM
        1+2
    """
    def __init__(self, bert, hidden_size, n_layers, cnn_kernel_size, cnn_filter_num):
        super(EncoderStackedNmlCnnLSTMLSTM, self).__init__()
        self.bert = bert
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num

        # GRU输入为bert + cnn_out
        # self.gru = nn.GRU(bert_size+(bert_size-cnn_kernel_size[1]+1)*cnn_filter_num, hidden_size, n_layers, bidirectional=True, batch_first=False)

        # cnn_kernel_size包含高度（多少词一个卷）和宽度（每个词的多少位一个卷）
        self.conv = nn.ModuleList()
        cnn_layer_num = 2
        self.cnn_layer_num = cnn_layer_num
        for i in range(cnn_layer_num):
            module_tmp = nn.Conv2d(
                1 if i == 0 else cnn_filter_num,
                cnn_filter_num,
                cnn_kernel_size,
                padding=(
                    (cnn_kernel_size[0] - 1) // 2,
                    (cnn_kernel_size[1] - 1) // 2
                    )
            )
            self.conv.append(module_tmp)

        # LSTM输入为bert + cnn_out
        self.lstm = nn.ModuleDict({
            'cnn_lstm': nn.LSTM(
                # cnn的每个词对应的输入尺寸为 cnn_layer_num=4层CNN，每层cnn_filter_num=10 filter
                input_size=(bert.embedding_size-((cnn_kernel_size[1]-1) % 2 * cnn_layer_num))*cnn_filter_num,
                hidden_size=bert.embedding_size, num_layers=n_layers, bidirectional=True, batch_first=True
                ),
            'emb_lstm': nn.LSTM(
                bert.embedding_size,
                bert.embedding_size,
                n_layers,
                bidirectional=True,
                batch_first=False
                ),
        })

    # @torchsnooper.snoop()
    def forward(self, word_inputs):
        # Note: we run this all at once (over the whole input sequence)
        # [B, S, Emb]
        embedded = self.bert(word_inputs)[0]

        # # 补上padding，使得conv得到seq_len个向量
        # batch_size = embedded_cnn.shape[0]
        # padding_size = self.cnn_kernel_size[0] - 1
        # # [B, padding_size, Emb]
        # pad_start = torch.zeros(batch_size, padding_size//2, self.bert_size).to(device)
        # pad_end = torch.zeros(batch_size, padding_size//2, self.bert_size).to(device)
        # # [B, S, Emb] -> [B, S+p_s, Emb]
        # embedded_cnn = torch.cat([pad_start, embedded_cnn, pad_end], dim=1)

        # [B, S+p_s, Emb] -> [B, Channel, S+p_s, Emb]
        embedded_cnn = embedded.unsqueeze(1)
        # cnn_out: [B, Channel, S, x] 向量维度未知
        """ print('cnn_out size', embedded_cnn.shape) """
        cnn_out = []
        cnn_out = embedded_cnn
        for conv in self.conv:
            cnn_out = conv(cnn_out)

        # for conv in self.conv:
        #     cnn_out_temp = conv(embedded_cnn)
        #     # cat cnn_out的filter_num维度
        #     # [B, Channel, S, x] -> [B, S, x]
        #     # filter_size = 2
        #     cnn_out.append(torch.cat([cnn_out_temp[:, i, :, :] for i in range(2)], dim=-1))
        # cnn_out = torch.cat(cnn_out, dim=-1)

        # 通过拼接 [B, Channel, S, x] -> [B, S, x]
        # print(cnn_out.shape)
        cnn_out = torch.cat([cnn_out[:, i, :, :] for i in range(self.cnn_filter_num)], dim=-1)
        # cnn_out = cnn_out.squeeze(1)
        """ print('cnn_out size', cnn_out.shape) """

        # 打印嵌入的 词向量shape 和 隐状态shape。
        # print('enc_emb size', embedded.shape)
        # print('hidden size', hidden.shape)
        cnn_lstm_input = cnn_out
        emb_lstm_input = embedded
        # output1 = [B, S, 2H]      hidden1=[lyr*direct, B, H]
        output1, hidden1 = self.lstm['cnn_lstm'](cnn_lstm_input, None)
        output2, hidden2 = self.lstm['emb_lstm'](emb_lstm_input, None)

        # 在第三维cat起来
        output = torch.cat([output1, output2], dim=-1)

        # output size: [S, B, 4H];      hidden: ([lyr*direct, B, H], [lyr*direct, B, H])
        return output


class EncoderStackedCnnLSTMLSTM(nn.Module):
    """添加额外CNN向量
        1、cnn提取特征，后RNN；
        2、bert直连LSTM
        1+2
    """
    def __init__(self, bert, hidden_size, n_layers, cnn_kernel_size, cnn_filter_num):
        super(EncoderStackedCnnLSTMLSTM, self).__init__()
        self.bert = bert
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num

        # GRU输入为bert + cnn_out
        # self.gru = nn.GRU(bert_size+(bert_size-cnn_kernel_size[1]+1)*cnn_filter_num, hidden_size, n_layers, bidirectional=True, batch_first=False)

        # cnn_kernel_size包含高度（多少词一个卷）和宽度（每个词的多少位一个卷）
        self.conv = nn.ModuleList()
        cnn_layer_num = 4
        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
            })
            self.conv.append(module_tmp)

        # LSTM输入为bert + cnn_out
        self.lstm = nn.ModuleDict({
            'cnn_lstm': nn.LSTM(
                # cnn的每个词对应的输入尺寸为 cnn_layer_num=4层CNN，每层cnn_filter_num=10 filter
                input_size=(bert.embedding_size-((cnn_kernel_size[1]-1) % 2 * cnn_layer_num))*cnn_filter_num,
                hidden_size=bert.embedding_size, num_layers=n_layers, bidirectional=True, batch_first=True
                ),
            'emb_lstm': nn.LSTM(
                bert.embedding_size,
                bert.embedding_size,
                n_layers,
                bidirectional=True,
                batch_first=False
                ),
        })

    # @torchsnooper.snoop()
    def forward(self, word_inputs):
        # Note: we run this all at once (over the whole input sequence)
        # [B, S, Emb]
        embedded = self.bert(word_inputs)[0]

        # # 补上padding，使得conv得到seq_len个向量
        # batch_size = embedded_cnn.shape[0]
        # padding_size = self.cnn_kernel_size[0] - 1
        # # [B, padding_size, Emb]
        # pad_start = torch.zeros(batch_size, padding_size//2, self.bert_size).to(device)
        # pad_end = torch.zeros(batch_size, padding_size//2, self.bert_size).to(device)
        # # [B, S, Emb] -> [B, S+p_s, Emb]
        # embedded_cnn = torch.cat([pad_start, embedded_cnn, pad_end], dim=1)

        # [B, S+p_s, Emb] -> [B, Channel, S+p_s, Emb]
        embedded_cnn = embedded.unsqueeze(1)
        # cnn_out: [B, Channel, S, x] 向量维度未知
        """ print('cnn_out size', embedded_cnn.shape) """
        cnn_out = []
        cnn_out = embedded_cnn
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            cnn_out = w * torch.sigmoid(v)

        # for conv in self.conv:
        #     cnn_out_temp = conv(embedded_cnn)
        #     # cat cnn_out的filter_num维度
        #     # [B, Channel, S, x] -> [B, S, x]
        #     # filter_size = 2
        #     cnn_out.append(torch.cat([cnn_out_temp[:, i, :, :] for i in range(2)], dim=-1))
        # cnn_out = torch.cat(cnn_out, dim=-1)

        # 通过拼接 [B, Channel, S, x] -> [B, S, x]
        # print(cnn_out.shape)
        cnn_out = torch.cat([cnn_out[:, i, :, :] for i in range(self.cnn_filter_num)], dim=-1)
        # cnn_out = cnn_out.squeeze(1)
        """ print('cnn_out size', cnn_out.shape) """

        # 打印嵌入的 词向量shape 和 隐状态shape。
        # print('enc_emb size', embedded.shape)
        # print('hidden size', hidden.shape)
        cnn_lstm_input = cnn_out
        emb_lstm_input = embedded
        # output1 = [B, S, 2H]      hidden1=[lyr*direct, B, H]
        output1, hidden1 = self.lstm['cnn_lstm'](cnn_lstm_input, None)
        output2, hidden2 = self.lstm['emb_lstm'](emb_lstm_input, None)

        # 在第三维cat起来
        output = torch.cat([output1, output2], dim=-1)

        # output size: [S, B, 4H];      hidden: ([lyr*direct, B, H], [lyr*direct, B, H])
        return output


class EncoderStackedSlimCnn(nn.Module):
    """添加额外CNN向量
        1、cnn提取特征，后RNN；
        2、bert直连LSTM
        1+2
    """
    def __init__(self, bert, hidden_size, n_layers, cnn_kernel_size, cnn_filter_num):
        super(EncoderStackedCnndivided, self).__init__()
        self.bert = bert
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num

        # GRU输入为bert + cnn_out
        # self.gru = nn.GRU(bert_size+(bert_size-cnn_kernel_size[1]+1)*cnn_filter_num, hidden_size, n_layers, bidirectional=True, batch_first=False)

        # cnn_kernel_size包含高度（多少词一个卷）和宽度（每个词的多少位一个卷）
        self.conv = nn.ModuleList()
        cnn_layer_num = 4
        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size if i == 0 else (cnn_kernel_size[0], cnn_filter_num),
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size if i == 0 else (cnn_kernel_size[0], cnn_filter_num),
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
            })
            self.conv.append(module_tmp)

        # LSTM输入为bert + cnn_out
        self.lstm = nn.ModuleDict({
            'cnn_lstm': nn.LSTM(
                # cnn的每个词对应的输入尺寸为 cnn_layer_num=4层CNN，每层cnn_filter_num=10 filter
                input_size=(bert.embedding_size-((cnn_kernel_size[1]-1) % 2 * cnn_layer_num))*cnn_filter_num,
                hidden_size=bert.embedding_size, num_layers=n_layers, bidirectional=True, batch_first=True
                ),
            'emb_lstm': nn.LSTM(
                bert.embedding_size,
                bert.embedding_size,
                n_layers,
                bidirectional=True,
                batch_first=False
                ),
        })

    # @torchsnooper.snoop()
    def forward(self, word_inputs):
        # Note: we run this all at once (over the whole input sequence)
        # [B, S, Emb]
        embedded = self.bert(word_inputs)[0]

        # # 补上padding，使得conv得到seq_len个向量
        # batch_size = embedded_cnn.shape[0]
        # padding_size = self.cnn_kernel_size[0] - 1
        # # [B, padding_size, Emb]
        # pad_start = torch.zeros(batch_size, padding_size//2, self.bert_size).to(device)
        # pad_end = torch.zeros(batch_size, padding_size//2, self.bert_size).to(device)
        # # [B, S, Emb] -> [B, S+p_s, Emb]
        # embedded_cnn = torch.cat([pad_start, embedded_cnn, pad_end], dim=1)

        # [B, S+p_s, Emb] -> [B, Channel, S+p_s, Emb]
        embedded_cnn = embedded.unsqueeze(1)
        # cnn_out: [B, Channel, S, x] 向量维度未知
        """ print('cnn_out size', embedded_cnn.shape) """
        cnn_out = []
        cnn_out = embedded_cnn
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            cnn_out = w * torch.sigmoid(v)

        # for conv in self.conv:
        #     cnn_out_temp = conv(embedded_cnn)
        #     # cat cnn_out的filter_num维度
        #     # [B, Channel, S, x] -> [B, S, x]
        #     # filter_size = 2
        #     cnn_out.append(torch.cat([cnn_out_temp[:, i, :, :] for i in range(2)], dim=-1))
        # cnn_out = torch.cat(cnn_out, dim=-1)

        # 通过拼接 [B, Channel, S, x] -> [B, S, x]
        # print(cnn_out.shape)
        cnn_out = torch.cat([cnn_out[:, i, :, :] for i in range(self.cnn_filter_num)], dim=-1)
        # cnn_out = cnn_out.squeeze(1)
        """ print('cnn_out size', cnn_out.shape) """

        # 打印嵌入的 词向量shape 和 隐状态shape。
        # print('enc_emb size', embedded.shape)
        # print('hidden size', hidden.shape)
        cnn_lstm_input = cnn_out
        emb_lstm_input = embedded
        # output1 = [B, S, 2H]      hidden1=[lyr*direct, B, H]
        output1, hidden1 = self.lstm['cnn_lstm'](cnn_lstm_input, None)
        output2, hidden2 = self.lstm['emb_lstm'](emb_lstm_input, None)

        # 在第三维cat起来
        output = torch.cat([output1, output2], dim=-1)

        # output size: [S, B, 4H];      hidden: ([lyr*direct, B, H], [lyr*direct, B, H])
        return output


class EncoderStackedCnndivided(nn.Module):
    """添加额外CNN向量
        1、cnn提取特征，后RNN；
        2、bert直连LSTM
        1+2
    """
    def __init__(self, bert, hidden_size, n_layers, cnn_kernel_size, cnn_filter_num):
        super(EncoderStackedCnndivided, self).__init__()
        self.bert = bert
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num

        # GRU输入为bert + cnn_out
        # self.gru = nn.GRU(bert_size+(bert_size-cnn_kernel_size[1]+1)*cnn_filter_num, hidden_size, n_layers, bidirectional=True, batch_first=False)

        # cnn_kernel_size包含高度（多少词一个卷）和宽度（每个词的多少位一个卷）
        self.conv = nn.ModuleList()
        cnn_layer_num = 1
        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
            })
            self.conv.append(module_tmp)

        self.lstm = nn.LSTM(
                # cnn的每个词对应的输入尺寸为 cnn_layer_num=4层CNN，每层cnn_filter_num=10 filter
                input_size=(bert.embedding_size-((cnn_kernel_size[1]-1) % 2 * cnn_layer_num))*cnn_filter_num,
                hidden_size=bert.embedding_size, num_layers=n_layers, bidirectional=True, batch_first=True
                )

        # self.out_size = (bert.embedding_size-((cnn_kernel_size[1]-1) % 2 * cnn_layer_num))*cnn_filter_num
        self.out_size = bert.embedding_size*2

        # # LSTM输入为bert + cnn_out
        # self.lstm = nn.ModuleDict({
        #     'cnn_lstm': nn.LSTM(
        #         # cnn的每个词对应的输入尺寸为 cnn_layer_num=4层CNN，每层cnn_filter_num=10 filter
        #         input_size=(bert.embedding_size-((cnn_kernel_size[1]-1) % 2 * cnn_layer_num))*cnn_filter_num,
        #         hidden_size=bert.embedding_size, num_layers=n_layers, bidirectional=True, batch_first=True
        #         ),
        # })

    # @torchsnooper.snoop()
    def forward(self, word_inputs):
        # Note: we run this all at once (over the whole input sequence)
        # [B, S, Emb]
        embedded = self.bert(word_inputs)[0]

        # # 补上padding，使得conv得到seq_len个向量
        # batch_size = embedded_cnn.shape[0]
        # padding_size = self.cnn_kernel_size[0] - 1
        # # [B, padding_size, Emb]
        # pad_start = torch.zeros(batch_size, padding_size//2, self.bert_size).to(device)
        # pad_end = torch.zeros(batch_size, padding_size//2, self.bert_size).to(device)
        # # [B, S, Emb] -> [B, S+p_s, Emb]
        # embedded_cnn = torch.cat([pad_start, embedded_cnn, pad_end], dim=1)

        # [B, S+p_s, Emb] -> [B, Channel, S+p_s, Emb]
        embedded_cnn = embedded.unsqueeze(1)
        # cnn_out: [B, Channel, S, x] 向量维度未知
        """ print('cnn_out size', embedded_cnn.shape) """
        cnn_out = []
        cnn_out = embedded_cnn
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            cnn_out = w * torch.sigmoid(v)

        # for conv in self.conv:
        #     cnn_out_temp = conv(embedded_cnn)
        #     # cat cnn_out的filter_num维度
        #     # [B, Channel, S, x] -> [B, S, x]
        #     # filter_size = 2
        #     cnn_out.append(torch.cat([cnn_out_temp[:, i, :, :] for i in range(2)], dim=-1))
        # cnn_out = torch.cat(cnn_out, dim=-1)

        # 通过拼接 [B, Channel, S, x] -> [B, S, x]
        # print(cnn_out.shape)
        cnn_out = torch.cat([cnn_out[:, i, :, :] for i in range(self.cnn_filter_num)], dim=-1)
        # cnn_out = cnn_out.squeeze(1)
        cnn_out, hidden = self.lstm(cnn_out)
        """ print('cnn_out size', cnn_out.shape) """
        output = [cnn_out, embedded]
        # 打印嵌入的 词向量shape 和 隐状态shape。
        # print('enc_emb size', embedded.shape)
        # print('hidden size', hidden.shape)

        # output size: [S, B, 4H];      hidden: ([lyr*direct, B, H], [lyr*direct, B, H])
        return output


class EncoderStackedCnndividedBert(nn.Module):
    """添加额外CNN向量
        1、cnn提取特征，后RNN；
        2、bert直连LSTM
        1+2
    """
    def __init__(self, bert, hidden_size, n_layers, cnn_kernel_size, cnn_filter_num):
        super(EncoderStackedCnndividedBert, self).__init__()
        self.bert = bert
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num

        # GRU输入为bert + cnn_out
        # self.gru = nn.GRU(bert_size+(bert_size-cnn_kernel_size[1]+1)*cnn_filter_num, hidden_size, n_layers, bidirectional=True, batch_first=False)

        # cnn_kernel_size包含高度（多少词一个卷）和宽度（每个词的多少位一个卷）
        self.conv = nn.ModuleList()
        cnn_layer_num = 1
        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
            })
            self.conv.append(module_tmp)

        self.lstm = nn.LSTM(
                # cnn的每个词对应的输入尺寸为 cnn_layer_num=4层CNN，每层cnn_filter_num=10 filter
                input_size=(bert.embedding_size-((cnn_kernel_size[1]-1) % 2 * cnn_layer_num))*cnn_filter_num,
                hidden_size=bert.embedding_size, num_layers=n_layers, bidirectional=True, batch_first=True
                )

        # self.out_size = (bert.embedding_size-((cnn_kernel_size[1]-1) % 2 * cnn_layer_num))*cnn_filter_num
        self.out_size = bert.embedding_size*2

        # # LSTM输入为bert + cnn_out
        # self.lstm = nn.ModuleDict({
        #     'cnn_lstm': nn.LSTM(
        #         # cnn的每个词对应的输入尺寸为 cnn_layer_num=4层CNN，每层cnn_filter_num=10 filter
        #         input_size=(bert.embedding_size-((cnn_kernel_size[1]-1) % 2 * cnn_layer_num))*cnn_filter_num,
        #         hidden_size=bert.embedding_size, num_layers=n_layers, bidirectional=True, batch_first=True
        #         ),
        # })

    # @torchsnooper.snoop()
    def forward(self, word_inputs):
        # Note: we run this all at once (over the whole input sequence)
        # [B, S, Emb]
        embedded = self.bert(word_inputs)[0]

        # # 补上padding，使得conv得到seq_len个向量
        # batch_size = embedded_cnn.shape[0]
        # padding_size = self.cnn_kernel_size[0] - 1
        # # [B, padding_size, Emb]
        # pad_start = torch.zeros(batch_size, padding_size//2, self.bert_size).to(device)
        # pad_end = torch.zeros(batch_size, padding_size//2, self.bert_size).to(device)
        # # [B, S, Emb] -> [B, S+p_s, Emb]
        # embedded_cnn = torch.cat([pad_start, embedded_cnn, pad_end], dim=1)

        # [B, S+p_s, Emb] -> [B, Channel, S+p_s, Emb]
        embedded_cnn = embedded.unsqueeze(1)
        # cnn_out: [B, Channel, S, x] 向量维度未知
        """ print('cnn_out size', embedded_cnn.shape) """
        cnn_out = []
        cnn_out = embedded_cnn
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            cnn_out = w * torch.sigmoid(v)

        # for conv in self.conv:
        #     cnn_out_temp = conv(embedded_cnn)
        #     # cat cnn_out的filter_num维度
        #     # [B, Channel, S, x] -> [B, S, x]
        #     # filter_size = 2
        #     cnn_out.append(torch.cat([cnn_out_temp[:, i, :, :] for i in range(2)], dim=-1))
        # cnn_out = torch.cat(cnn_out, dim=-1)

        # 通过拼接 [B, Channel, S, x] -> [B, S, x]
        # print(cnn_out.shape)
        cnn_out = torch.cat([cnn_out[:, i, :, :] for i in range(self.cnn_filter_num)], dim=-1)
        # cnn_out = cnn_out.squeeze(1)
        cnn_out, hidden = self.lstm(cnn_out)
        """ print('cnn_out size', cnn_out.shape) """
        output = cnn_out
        # 打印嵌入的 词向量shape 和 隐状态shape。
        # print('enc_emb size', embedded.shape)
        # print('hidden size', hidden.shape)

        # output size: [S, B, 4H];      hidden: ([lyr*direct, B, H], [lyr*direct, B, H])
        return output


class EncoderStackedCnnDivideUniLSTM(nn.Module):
    """添加额外CNN向量
        1、cnn提取特征，后RNN；
        2、bert直连LSTM
        1+2
    """
    def __init__(self, bert, hidden_size, n_layers, cnn_kernel_size, cnn_filter_num):
        super(EncoderStackedCnnDivideUniLSTM, self).__init__()
        self.bert = bert
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num

        # GRU输入为bert + cnn_out
        # self.gru = nn.GRU(bert_size+(bert_size-cnn_kernel_size[1]+1)*cnn_filter_num, hidden_size, n_layers, bidirectional=True, batch_first=False)

        # cnn_kernel_size包含高度（多少词一个卷）和宽度（每个词的多少位一个卷）
        self.conv = nn.ModuleList()
        cnn_layer_num = 1
        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
            })
            self.conv.append(module_tmp)

        # self.lstm = nn.LSTM(
        #         # cnn的每个词对应的输入尺寸为 cnn_layer_num=4层CNN，每层cnn_filter_num=10 filter
        #         input_size=(bert.embedding_size-((cnn_kernel_size[1]-1) % 2 * cnn_layer_num))*cnn_filter_num,
        #         hidden_size=bert.embedding_size, num_layers=n_layers, bidirectional=False, batch_first=True
        #         )

        self.out_size = (bert.embedding_size-((cnn_kernel_size[1]-1) % 2 * cnn_layer_num))*cnn_filter_num
        # self.out_size = bert.embedding_size

        # # LSTM输入为bert + cnn_out
        # self.lstm = nn.ModuleDict({
        #     'cnn_lstm': nn.LSTM(
        #         # cnn的每个词对应的输入尺寸为 cnn_layer_num=4层CNN，每层cnn_filter_num=10 filter
        #         input_size=(bert.embedding_size-((cnn_kernel_size[1]-1) % 2 * cnn_layer_num))*cnn_filter_num,
        #         hidden_size=bert.embedding_size, num_layers=n_layers, bidirectional=True, batch_first=True
        #         ),
        # })

    # @torchsnooper.snoop()
    def forward(self, word_inputs):
        # Note: we run this all at once (over the whole input sequence)
        # [B, S, Emb]
        embedded = self.bert(word_inputs)[0]

        # # 补上padding，使得conv得到seq_len个向量
        # batch_size = embedded_cnn.shape[0]
        # padding_size = self.cnn_kernel_size[0] - 1
        # # [B, padding_size, Emb]
        # pad_start = torch.zeros(batch_size, padding_size//2, self.bert_size).to(device)
        # pad_end = torch.zeros(batch_size, padding_size//2, self.bert_size).to(device)
        # # [B, S, Emb] -> [B, S+p_s, Emb]
        # embedded_cnn = torch.cat([pad_start, embedded_cnn, pad_end], dim=1)

        # [B, S+p_s, Emb] -> [B, Channel, S+p_s, Emb]
        embedded_cnn = embedded.unsqueeze(1)
        # cnn_out: [B, Channel, S, x] 向量维度未知
        """ print('cnn_out size', embedded_cnn.shape) """
        cnn_out = []
        cnn_out = embedded_cnn
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            cnn_out = w * torch.sigmoid(v)

        # for conv in self.conv:
        #     cnn_out_temp = conv(embedded_cnn)
        #     # cat cnn_out的filter_num维度
        #     # [B, Channel, S, x] -> [B, S, x]
        #     # filter_size = 2
        #     cnn_out.append(torch.cat([cnn_out_temp[:, i, :, :] for i in range(2)], dim=-1))
        # cnn_out = torch.cat(cnn_out, dim=-1)

        # 通过拼接 [B, Channel, S, x] -> [B, S, x]
        # print(cnn_out.shape)
        cnn_out = torch.cat([cnn_out[:, i, :, :] for i in range(self.cnn_filter_num)], dim=-1)
        # cnn_out = cnn_out.squeeze(1)
        # cnn_out, hidden = self.lstm(cnn_out)
        """ print('cnn_out size', cnn_out.shape) """
        output = cnn_out
        # 打印嵌入的 词向量shape 和 隐状态shape。
        # print('enc_emb size', embedded.shape)
        # print('hidden size', hidden.shape)

        # output size: [S, B, 4H];      hidden: ([lyr*direct, B, H], [lyr*direct, B, H])
        return output


class EncoderStackedNoCnnLSTMLSTM(nn.Module):
    """添加额外CNN向量
        1、cnn提取特征，后RNN；
        2、bert直连LSTM
        1+2
    """
    def __init__(self, bert, hidden_size, n_layers, cnn_kernel_size, cnn_filter_num):
        super(EncoderStackedNoCnnLSTMLSTM, self).__init__()
        self.bert = bert
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num

        # GRU输入为bert + cnn_out
        # self.gru = nn.GRU(bert_size+(bert_size-cnn_kernel_size[1]+1)*cnn_filter_num, hidden_size, n_layers, bidirectional=True, batch_first=False)

        # cnn_kernel_size包含高度（多少词一个卷）和宽度（每个词的多少位一个卷）
        self.conv = nn.ModuleList()
        cnn_layer_num = 4
        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
            })
            self.conv.append(module_tmp)

        # LSTM输入为bert + cnn_out
        self.lstm = nn.ModuleDict({
            'cnn_lstm': nn.LSTM(
                # cnn的每个词对应的输入尺寸为 cnn_layer_num=4层CNN，每层cnn_filter_num=10 filter
                input_size=(bert.embedding_size-((cnn_kernel_size[1]-1) % 2 * cnn_layer_num))*cnn_filter_num,
                hidden_size=hidden_size, num_layers=n_layers, bidirectional=True, batch_first=True
                ),
            'emb_lstm': nn.LSTM(
                bert.embedding_size,
                hidden_size,
                n_layers,
                bidirectional=True,
                batch_first=False
                ),
        })

    # @torchsnooper.snoop()
    def forward(self, word_inputs):
        # Note: we run this all at once (over the whole input sequence)
        # [B, S, Emb]
        embedded = self.bert(word_inputs)[0]
        emb_lstm_input = embedded
        # output1 = [B, S, 2H]      hidden1=[lyr*direct, B, H]
        output2, hidden2 = self.lstm['emb_lstm'](emb_lstm_input, None)

        # 在第三维cat起来
        output = output2

        # output size: [S, B, 4H];      hidden: ([lyr*direct, B, H], [lyr*direct, B, H])
        return output


# TODO
class ALBertSmallCNNLSTMPunc(nn.Module):
    # NOTE bert hidden_size=384
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ALBertSmallCNNLSTMPunc, self).__init__()
        self.bert = AutoModel.from_pretrained('./models/albert_chinese_small/')
        self.bert.embedding_size = 384
        self.hidden_size = 200
        print(type(self.bert))
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.encoder = EncoderStackedCnnLSTMLSTM(
                bert=self.bert,
                hidden_size=self.hidden_size,
                n_layers=3,
                # 额外的参数
                # cnn -> rnn
                # NOTE 使用了args作为参数传递工具
                cnn_kernel_size=(5, 20),
                # rnn -> cnn
                # cnn_kernel_size=(3, hidden_size*2),
                cnn_filter_num=5
            )

        # 以rnn输出为输入
        self.fc = nn.Linear(
                # fc_input：late_fusion `f_t = a_t W_{fa} ◦ σ(a_t W_{fa} W_{ff} + h_t W_{fh} + b_f) + h_t `
                # size: [self.hidden_size*2] 双向的尺寸
                self.hidden_size*2*2,
                output_size
            )

        # 以单向的rnn输出和CNN输出 作为 输入
        # self.fc = nn.Linear(
        #         # rnn_out
        #         hidden_size * 2 +
        #         # cnn_out
        #         (hidden_size * 2 - self.encoder.cnn_kernel_size[1] + 1) *
        #         self.encoder.cnn_filter_num,
        #         num_class
        #     )

        # NOTE rnn_hidden*2 使用bert中间层hidden_state 384
        # self.fc = nn.Linear(384*2, output_size)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # src = src.transpose(0, 1).contiguous()
        x = x.transpose(0, 1)
        # print('enc_hidden size', enc_hidden.shape)
        # tuple类型([S, B, H], [S, B, H]) ,因为没有batch_first，所以S在B之前
        outputs = self.encoder(x)     # 输入pack，lstm默认输出pack

        # [S,B,H] -> [B, S, H]
        outputs = outputs.transpose(0, 1)
        # 记录尺寸，进行变换，以对每一时间步的输出进行late fuse
        out_size_S = outputs.size(0)
        out_size_B = outputs.size(1)
        out_size_H = outputs.size(2)

        # print(out_size_S, out_size_B)
        # print(outputs[1].shape)
        outputs = outputs.contiguous()
        # print(outputs.shape)
        outputs = outputs.view(out_size_S*out_size_B, out_size_H)

        outputs = outputs.contiguous()


        # for i in range(l):
        #     print('{} shape:'.format(i), x[i].shape)
        # print('x', type(x))
        # print('shape', x.shape)
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(outputs)
        return x


class ALBertSmallRNNPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ALBertSmallRNNPunc, self).__init__()
        self.bert = AutoModel.from_pretrained('./models/albert_chinese_small/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)

        # 批标准化
        # self.gru = nn.GRU(384, 384, 2, bidirectional=True, batch_first=True)
        self.lstm = nn.LSTM(384, 384, 2, bidirectional=True, batch_first=True)
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
        # x = self.bn(x)
        x = x.view(-1, shape1, shape2)

        # GRU*************************************
        # init_GRU_hidden
        def init_gru_hidden(batch_size):
            # when NOT bidirection (layer, B, H)
            h = Variable(torch.zeros(self.gru.num_layers*2, batch_size, self.gru.hidden_size)).to(device)
            # h for storing hidden layer weight
            return h

        # init_LSTM_hidden
        def init_lstm_hidden(batch_size):
            # when NOT bidirection (layer, B, H)
            h = Variable(torch.zeros(self.lstm.num_layers*2, batch_size, self.lstm.hidden_size)).to(device)
            c = Variable(torch.zeros(self.lstm.num_layers*2, batch_size, self.lstm.hidden_size)).to(device)
            # h for storing hidden layer weight
            return (h, c)

        # hidden = init_gru_hidden(x.shape[0])
        # x, hidden = self.gru(x, hidden)

        hidden = init_lstm_hidden(x.shape[0])
        x, hidden = self.lstm(x, hidden)
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
        return x


class ALBertSmallRNNnewLinearPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ALBertSmallRNNnewLinearPunc, self).__init__()
        print("fucking code**************")
        self.bert = AutoModelWithLMHead.from_pretrained('./models/albert_chinese_small/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)

        # 批标准化
        # NOTE dense*2 使用bert中间层 dense hidden_state 128
        self.bn = nn.BatchNorm1d(segment_size*128)
        # NOTE dense*2 使用bert中间层 dense hidden_state 128
        self.gru = nn.GRU(128, 128, 2, bidirectional=True, batch_first=True)
        # # rnn_hidden*2 使用bert中间层hidden_state 384
        # NOTE dense*2 使用bert中间层 dense hidden_state 128
        self.fc = nn.Linear(128*2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 修改后
        # print('input 的shape', x.shape)
        x = self.bert.albert(x)[0]
        # 384 -> 128
        x = self.bert.predictions.dense(x)
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
            h = Variable(torch.zeros(self.gru.num_layers*2, batch_size, self.gru.hidden_size)).to(device)
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
        # out [B*S, num_class]
        return x


class ChineseRobertaRNNnewLinearPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ChineseRobertaRNNnewLinearPunc, self).__init__()
        print("fucking code**************")
        self.bert = BertModel.from_pretrained('./models/chinese-roberta-wwm-ext/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768
        # 批标准化
        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.bn = nn.BatchNorm1d(segment_size*self.bert_size)
        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.gru = nn.GRU(self.bert_size, self.bert_size, 2, bidirectional=True, batch_first=True)
        # # rnn_hidden*2 使用bert中间层hidden_state 384
        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.fc = nn.Linear(self.bert_size*2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 修改后
        # print('input 的shape', x.shape)
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
            h = Variable(torch.zeros(self.gru.num_layers*2, batch_size, self.gru.hidden_size)).to(device)
            # h for storing hidden layer weight
            return h

        hidden = init_gru_hidden(x.shape[0])
        x, hidden = self.gru(x, None)
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
        # out [B*S, num_class]
        return x


class ChineseRobertaCNNbnLstmPunc(nn.Module):
    # NOTE bert hidden_size=384
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ChineseRobertaCNNbnLstmPunc, self).__init__()
        self.bert = BertModel.from_pretrained('./models/chinese-roberta-wwm-ext')
        self.bert.embedding_size = 768
        self.hidden_size = 200
        print(type(self.bert))
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.encoder = EncoderStackedCnnBnLSTMLSTM(
                bert=self.bert,
                hidden_size=self.hidden_size,
                n_layers=3,
                # 额外的参数
                # cnn -> rnn
                # NOTE 使用了args作为参数传递工具
                cnn_kernel_size=(5, 20),
                # rnn -> cnn
                # cnn_kernel_size=(3, hidden_size*2),
                cnn_filter_num=5
            )

        # 以rnn输出为输入
        self.fc = nn.Linear(
                # fc_input：late_fusion `f_t = a_t W_{fa} ◦ σ(a_t W_{fa} W_{ff} + h_t W_{fh} + b_f) + h_t `
                # size: [self.hidden_size*2] 双向的尺寸
                self.hidden_size*2*2,
                output_size
            )

        # 以单向的rnn输出和CNN输出 作为 输入
        # self.fc = nn.Linear(
        #         # rnn_out
        #         hidden_size * 2 +
        #         # cnn_out
        #         (hidden_size * 2 - self.encoder.cnn_kernel_size[1] + 1) *
        #         self.encoder.cnn_filter_num,
        #         num_class
        #     )

        # NOTE rnn_hidden*2 使用bert中间层hidden_state 384
        # self.fc = nn.Linear(384*2, output_size)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # src = src.transpose(0, 1).contiguous()
        x = x.transpose(0, 1)
        # print('enc_hidden size', enc_hidden.shape)
        # tuple类型([S, B, H], [S, B, H]) ,因为没有batch_first，所以S在B之前
        outputs = self.encoder(x)     # 输入pack，lstm默认输出pack

        # [S,B,H] -> [B, S, H]
        outputs = outputs.transpose(0, 1)
        # 记录尺寸，进行变换，以对每一时间步的输出进行late fuse
        out_size_S = outputs.size(0)
        out_size_B = outputs.size(1)
        out_size_H = outputs.size(2)

        # print(out_size_S, out_size_B)
        # print(outputs[1].shape)
        outputs = outputs.contiguous()
        # print(outputs.shape)
        outputs = outputs.view(out_size_S*out_size_B, out_size_H)

        outputs = outputs.contiguous()


        # for i in range(l):
        #     print('{} shape:'.format(i), x[i].shape)
        # print('x', type(x))
        # print('shape', x.shape)
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(outputs)
        return x

class ChineseRobertaCNNLstmPunc(nn.Module):
    # NOTE bert hidden_size=384
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ChineseRobertaCNNLstmPunc, self).__init__()
        self.bert = BertModel.from_pretrained('./models/chinese-roberta-wwm-ext')
        self.bert.embedding_size = 768
        self.hidden_size = 200
        print(type(self.bert))
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.encoder = EncoderStackedCnnLSTMLSTM(
                bert=self.bert,
                hidden_size=self.bert.embedding_size,
                n_layers=3,
                # 额外的参数
                # cnn -> rnn
                # NOTE 使用了args作为参数传递工具
                cnn_kernel_size=(5, 20),
                # rnn -> cnn
                # cnn_kernel_size=(3, hidden_size*2),
                cnn_filter_num=5
            )

        # 以rnn输出为输入
        self.fc = nn.Linear(
                # fc_input：late_fusion `f_t = a_t W_{fa} ◦ σ(a_t W_{fa} W_{ff} + h_t W_{fh} + b_f) + h_t `
                # size: [self.hidden_size*2] 双向的尺寸
                self.bert.embedding_size*2*2,
                output_size
            )

        # 以单向的rnn输出和CNN输出 作为 输入
        # self.fc = nn.Linear(
        #         # rnn_out
        #         hidden_size * 2 +
        #         # cnn_out
        #         (hidden_size * 2 - self.encoder.cnn_kernel_size[1] + 1) *
        #         self.encoder.cnn_filter_num,
        #         num_class
        #     )

        # NOTE rnn_hidden*2 使用bert中间层hidden_state 384
        # self.fc = nn.Linear(384*2, output_size)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print('enc_hidden size', enc_hidden.shape)
        # [B, S, H]
        outputs = self.encoder(x)     # 输入pack，lstm默认输出pack

        # 记录尺寸，进行变换，以对每一时间步的输出进行late fuse
        out_size_B = outputs.size(0)
        out_size_S = outputs.size(1)
        out_size_H = outputs.size(2)

        # print(out_size_S, out_size_B)
        # print(outputs[1].shape)
        outputs = outputs.contiguous()
        # print(outputs.shape)
        outputs = outputs.view(out_size_S*out_size_B, out_size_H)

        outputs = outputs.contiguous()


        # for i in range(l):
        #     print('{} shape:'.format(i), x[i].shape)
        # print('x', type(x))
        # print('shape', x.shape)
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(outputs)
        return x


class ChineseRobertaCNNlinearDivideConcatPunc(nn.Module):
    # NOTE bert hidden_size=384
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ChineseRobertaCNNlinearDivideConcatPunc, self).__init__()
        self.bert = BertModel.from_pretrained('./models/chinese-roberta-wwm-ext')
        self.bert.embedding_size = 768
        self.hidden_size = 200
        print(type(self.bert))
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.encoder = EncoderStackedCnndivided(
                bert=self.bert,
                hidden_size=self.bert.embedding_size,
                n_layers=3,
                # 额外的参数
                # cnn -> rnn
                # NOTE 使用了args作为参数传递工具
                cnn_kernel_size=(3, 20),
                # rnn -> cnn
                # cnn_kernel_size=(3, hidden_size*2),
                cnn_filter_num=5
            )

        self.dropout = nn.Dropout(dropout)
        # 1、concat
        self.fc = nn.Linear(
                # fc_input：late_fusion `f_t = a_t W_{fa} ◦ σ(a_t W_{fa} W_{ff} + h_t W_{fh} + b_f) + h_t `
                # size: [self.hidden_size*2] 双向的尺寸
                self.bert.embedding_size+self.encoder.out_size,
                output_size
            )

        # 多层线性层
        # self.fc = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(self.bert.embedding_size+self.encoder.out_size, self.bert.embedding_size+self.encoder.out_size),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(self.bert.embedding_size+self.encoder.out_size, self.bert.embedding_size+self.encoder.out_size),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(self.bert.embedding_size+self.encoder.out_size, self.bert.embedding_size+self.encoder.out_size),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(self.bert.embedding_size+self.encoder.out_size, output_size)
        # )


    def forward(self, x):
        # print('enc_hidden size', enc_hidden.shape)
        # [B, S, H]
        outputs = self.encoder(x)     # 输入pack，lstm默认输出pack

        # concat 最后一维
        outputs = torch.cat(outputs, dim=-1)

        # 记录尺寸，进行变换，以对每一时间步的输出进行late fuse
        out_size_B = outputs.size(0)
        out_size_S = outputs.size(1)
        out_size_H = outputs.size(2)

        # print(out_size_S, out_size_B)
        # print(outputs[1].shape)
        outputs = outputs.contiguous()
        # print(outputs.shape)
        outputs = outputs.view(out_size_S*out_size_B, out_size_H)

        outputs = outputs.contiguous()


        # for i in range(l):
        #     print('{} shape:'.format(i), x[i].shape)
        # print('x', type(x))
        # print('shape', x.shape)
        x = self.fc(self.dropout(outputs))
        return x


class BertChineseCNNlinearDivideConcatPunc(nn.Module):
    # NOTE bert hidden_size=384
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChineseCNNlinearDivideConcatPunc, self).__init__()
        self.bert = BertModel.from_pretrained('./models/bert_base_chinese')
        self.bert.embedding_size = 768
        self.hidden_size = 200
        print(type(self.bert))
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.encoder = EncoderStackedCnndivided(
                bert=self.bert,
                hidden_size=self.bert.embedding_size,
                n_layers=3,
                # 额外的参数
                # cnn -> rnn
                # NOTE 使用了args作为参数传递工具
                cnn_kernel_size=(3, 20),
                # rnn -> cnn
                # cnn_kernel_size=(3, hidden_size*2),
                cnn_filter_num=5
            )

        self.dropout = nn.Dropout(dropout)
        # 1、concat
        self.fc = nn.Linear(
                # fc_input：late_fusion `f_t = a_t W_{fa} ◦ σ(a_t W_{fa} W_{ff} + h_t W_{fh} + b_f) + h_t `
                # size: [self.hidden_size*2] 双向的尺寸
                self.bert.embedding_size+self.encoder.out_size,
                output_size
            )

        # 多层线性层
        # self.fc = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(self.bert.embedding_size+self.encoder.out_size, self.bert.embedding_size+self.encoder.out_size),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(self.bert.embedding_size+self.encoder.out_size, self.bert.embedding_size+self.encoder.out_size),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(self.bert.embedding_size+self.encoder.out_size, self.bert.embedding_size+self.encoder.out_size),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(self.bert.embedding_size+self.encoder.out_size, output_size)
        # )


    def forward(self, x):
        # print('enc_hidden size', enc_hidden.shape)
        # [B, S, H]
        outputs = self.encoder(x)     # 输入pack，lstm默认输出pack

        # concat 最后一维
        outputs = torch.cat(outputs, dim=-1)

        # 记录尺寸，进行变换，以对每一时间步的输出进行late fuse
        out_size_B = outputs.size(0)
        out_size_S = outputs.size(1)
        out_size_H = outputs.size(2)

        # print(out_size_S, out_size_B)
        # print(outputs[1].shape)
        outputs = outputs.contiguous()
        # print(outputs.shape)
        outputs = outputs.view(out_size_S*out_size_B, out_size_H)

        outputs = outputs.contiguous()


        # for i in range(l):
        #     print('{} shape:'.format(i), x[i].shape)
        # print('x', type(x))
        # print('shape', x.shape)
        x = self.fc(self.dropout(outputs))
        return x


class BertChineseCNNlinearDividedBertPunc(nn.Module):
    # NOTE bert hidden_size=768
    # 两个bert
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChineseCNNlinearDividedBertPunc, self).__init__()
        self.bert = BertModel.from_pretrained('./models/bert_base_chinese')
        self.bert_2 = BertModel.from_pretrained('./models/bert_base_chinese')
        self.bert.embedding_size = 768
        self.hidden_size = 200
        print(type(self.bert))
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.encoder = EncoderStackedCnndividedBert(
                bert=self.bert,
                hidden_size=self.bert.embedding_size,
                n_layers=3,
                # 额外的参数
                # cnn -> rnn
                # NOTE 使用了args作为参数传递工具
                cnn_kernel_size=(5, 20),
                # rnn -> cnn
                # cnn_kernel_size=(3, hidden_size*2),
                cnn_filter_num=5
            )


        # # 1、concat
        # self.fc = nn.Linear(
        #         # fc_input：late_fusion `f_t = a_t W_{fa} ◦ σ(a_t W_{fa} W_{ff} + h_t W_{fh} + b_f) + h_t `
        #         # size: [self.hidden_size*2] 双向的尺寸
        #         self.bert.embedding_size+self.encoder.out_size,
        #         output_size
        #     )

        # 2、各自作为输入**********************
        self.fc_cnn = nn.Linear(
            self.encoder.out_size,
            self.bert.embedding_size
        )
        self.fc_bert = nn.Linear(
            self.bert.embedding_size,
            self.bert.embedding_size
        )
        self.activate = nn.ReLU()
        self.fc = nn.Linear(
                # fc_input：late_fusion `f_t = a_t W_{fa} ◦ σ(a_t W_{fa} W_{ff} + h_t W_{fh} + b_f) + h_t `
                # size: [self.hidden_size*2] 双向的尺寸
                self.bert.embedding_size*2,
                output_size
            )
        self.dropout = nn.Dropout(dropout)


        # 以单向的rnn输出和CNN输出 作为 输入
        # self.fc = nn.Linear(
        #         # rnn_out
        #         hidden_size * 2 +
        #         # cnn_out
        #         (hidden_size * 2 - self.encoder.cnn_kernel_size[1] + 1) *
        #         self.encoder.cnn_filter_num,
        #         num_class
        #     )

        # NOTE rnn_hidden*2 使用bert中间层hidden_state 384
        # self.fc = nn.Linear(384*2, output_size)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print('enc_hidden size', enc_hidden.shape)
        # [B, S, H]
        outputs = self.encoder(x)     # 输入pack，lstm默认输出pack

        outputs1 = outputs.contiguous()
        outputs2 = self.bert_2(x)[0]

        # 记录尺寸，进行变换，以对每一时间步的输出进行late fuse
        out_size_B_1 = outputs1.size(0)
        out_size_S_1 = outputs1.size(1)
        out_size_H_1 = outputs1.size(2)

        out_size_B_2 = outputs2.size(0)
        out_size_S_2 = outputs2.size(1)
        out_size_H_2 = outputs2.size(2)

        # print(outputs.shape)
        # [X, H]
        outputs1 = outputs1.view(out_size_S_1*out_size_B_1, out_size_H_1)
        outputs2 = outputs2.view(out_size_S_2*out_size_B_2, out_size_H_2)

        outputs1 = self.fc_cnn(self.dropout(outputs1))
        outputs2 = self.fc_bert(self.dropout(outputs2))

        # [X, H] -> [X, 2*H]
        outputs = torch.cat([outputs1, outputs2], dim=-1)
        outputs = self.activate(outputs)
        x = self.fc(self.dropout(outputs))

        return x


class BertChinese2BertPunc(nn.Module):
    # NOTE bert hidden_size=768
    # 两个bert
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChinese2BertPunc, self).__init__()
        self.bert = BertModel.from_pretrained('./models/bert_base_chinese')
        self.bert_2 = BertModel.from_pretrained('./models/bert_base_chinese')
        self.bert.embedding_size = 768
        self.hidden_size = 200
        print(type(self.bert))

        # 2、各自作为输入**********************
        self.fc_cnn = nn.Linear(
            self.bert.embedding_size,
            self.bert.embedding_size
        )
        self.fc_bert = nn.Linear(
            self.bert.embedding_size,
            self.bert.embedding_size
        )
        self.activate = nn.Sigmoid()
        self.fc = nn.Linear(
                # fc_input：late_fusion `f_t = a_t W_{fa} ◦ σ(a_t W_{fa} W_{ff} + h_t W_{fh} + b_f) + h_t `
                # size: [self.hidden_size*2] 双向的尺寸
                self.bert.embedding_size*2,
                output_size
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print('enc_hidden size', enc_hidden.shape)
        # [B, S, H]

        outputs1 = self.bert(x)[0]
        outputs2 = self.bert_2(x)[0]

        # 记录尺寸，进行变换，以对每一时间步的输出进行late fuse
        out_size_B_1 = outputs1.size(0)
        out_size_S_1 = outputs1.size(1)
        out_size_H_1 = outputs1.size(2)

        out_size_B_2 = outputs2.size(0)
        out_size_S_2 = outputs2.size(1)
        out_size_H_2 = outputs2.size(2)

        # print(outputs.shape)
        # [X, H]
        outputs1 = outputs1.view(out_size_S_1*out_size_B_1, out_size_H_1)
        outputs2 = outputs2.view(out_size_S_2*out_size_B_2, out_size_H_2)

        outputs1 = self.fc_cnn(outputs1)
        outputs2 = self.fc_bert(outputs2)

        # [X, H] -> [X, 2*H]
        outputs = torch.cat([outputs1, outputs2], dim=-1)
        outputs = self.activate(outputs)
        x = self.fc(outputs)

        return x



class ChineseRobertaCNNlinearDividedPunc(nn.Module):
    # NOTE bert hidden_size=384
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ChineseRobertaCNNlinearDividedPunc, self).__init__()
        self.bert = BertModel.from_pretrained('./models/chinese-roberta-wwm-ext')
        self.bert.embedding_size = 768
        self.hidden_size = 200
        print(type(self.bert))
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.encoder = EncoderStackedCnndivided(
                bert=self.bert,
                hidden_size=self.bert.embedding_size,
                n_layers=3,
                # 额外的参数
                # cnn -> rnn
                # NOTE 使用了args作为参数传递工具
                cnn_kernel_size=(5, 20),
                # rnn -> cnn
                # cnn_kernel_size=(3, hidden_size*2),
                cnn_filter_num=5
            )


        # # 1、concat
        # self.fc = nn.Linear(
        #         # fc_input：late_fusion `f_t = a_t W_{fa} ◦ σ(a_t W_{fa} W_{ff} + h_t W_{fh} + b_f) + h_t `
        #         # size: [self.hidden_size*2] 双向的尺寸
        #         self.bert.embedding_size+self.encoder.out_size,
        #         output_size
        #     )

        # 2、各自作为输入**********************
        self.fc_cnn = nn.Linear(
            self.encoder.out_size,
            self.bert.embedding_size
        )
        self.fc_bert = nn.Linear(
            self.bert.embedding_size,
            self.bert.embedding_size
        )
        self.activate = nn.ReLU()
        self.fc = nn.Linear(
                # fc_input：late_fusion `f_t = a_t W_{fa} ◦ σ(a_t W_{fa} W_{ff} + h_t W_{fh} + b_f) + h_t `
                # size: [self.hidden_size*2] 双向的尺寸
                self.bert.embedding_size*2,
                output_size
            )
        self.dropout = nn.Dropout(dropout)


        # 以单向的rnn输出和CNN输出 作为 输入
        # self.fc = nn.Linear(
        #         # rnn_out
        #         hidden_size * 2 +
        #         # cnn_out
        #         (hidden_size * 2 - self.encoder.cnn_kernel_size[1] + 1) *
        #         self.encoder.cnn_filter_num,
        #         num_class
        #     )

        # NOTE rnn_hidden*2 使用bert中间层hidden_state 384
        # self.fc = nn.Linear(384*2, output_size)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print('enc_hidden size', enc_hidden.shape)
        # [B, S, H]
        outputs = self.encoder(x)     # 输入pack，lstm默认输出pack

        outputs1 = outputs[0].contiguous()
        outputs2 = outputs[1].contiguous()

        # 记录尺寸，进行变换，以对每一时间步的输出进行late fuse
        out_size_B_1 = outputs1.size(0)
        out_size_S_1 = outputs1.size(1)
        out_size_H_1 = outputs1.size(2)

        out_size_B_2 = outputs2.size(0)
        out_size_S_2 = outputs2.size(1)
        out_size_H_2 = outputs2.size(2)

        # print(outputs.shape)
        # [X, H]
        outputs1 = outputs1.view(out_size_S_1*out_size_B_1, out_size_H_1)
        outputs2 = outputs2.view(out_size_S_2*out_size_B_2, out_size_H_2)

        outputs1 = self.fc_cnn(self.dropout(outputs1))
        outputs2 = self.fc_bert(self.dropout(outputs2))

        # [X, H] -> [X, 2*H]
        outputs = torch.cat([outputs1, outputs2], dim=-1)
        outputs = self.activate(outputs)
        x = self.fc(self.dropout(outputs))

        return x


class ChineseRobertaCNNlinearDividedBertPunc(nn.Module):
    # NOTE bert hidden_size=768
    # 两个bert
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ChineseRobertaCNNlinearDividedBertPunc, self).__init__()
        self.bert = BertModel.from_pretrained('./models/chinese-roberta-wwm-ext')
        self.bert_2 = BertModel.from_pretrained('./models/chinese-roberta-wwm-ext')
        self.bert.embedding_size = 768
        self.hidden_size = 200
        print(type(self.bert))
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.encoder = EncoderStackedCnndividedBert(
                bert=self.bert,
                hidden_size=self.bert.embedding_size,
                n_layers=3,
                # 额外的参数
                # cnn -> rnn
                # NOTE 使用了args作为参数传递工具
                cnn_kernel_size=(5, 20),
                # rnn -> cnn
                # cnn_kernel_size=(3, hidden_size*2),
                cnn_filter_num=5
            )


        # # 1、concat
        # self.fc = nn.Linear(
        #         # fc_input：late_fusion `f_t = a_t W_{fa} ◦ σ(a_t W_{fa} W_{ff} + h_t W_{fh} + b_f) + h_t `
        #         # size: [self.hidden_size*2] 双向的尺寸
        #         self.bert.embedding_size+self.encoder.out_size,
        #         output_size
        #     )

        # 2、各自作为输入**********************
        self.fc_cnn = nn.Linear(
            self.encoder.out_size,
            self.bert.embedding_size
        )
        self.fc_bert = nn.Linear(
            self.bert.embedding_size,
            self.bert.embedding_size
        )
        self.activate = nn.ReLU()
        self.fc = nn.Linear(
                # fc_input：late_fusion `f_t = a_t W_{fa} ◦ σ(a_t W_{fa} W_{ff} + h_t W_{fh} + b_f) + h_t `
                # size: [self.hidden_size*2] 双向的尺寸
                self.bert.embedding_size*2,
                output_size
            )
        self.dropout = nn.Dropout(dropout)


        # 以单向的rnn输出和CNN输出 作为 输入
        # self.fc = nn.Linear(
        #         # rnn_out
        #         hidden_size * 2 +
        #         # cnn_out
        #         (hidden_size * 2 - self.encoder.cnn_kernel_size[1] + 1) *
        #         self.encoder.cnn_filter_num,
        #         num_class
        #     )

        # NOTE rnn_hidden*2 使用bert中间层hidden_state 384
        # self.fc = nn.Linear(384*2, output_size)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print('enc_hidden size', enc_hidden.shape)
        # [B, S, H]
        outputs = self.encoder(x)     # 输入pack，lstm默认输出pack

        outputs1 = outputs.contiguous()
        outputs2 = self.bert_2(x)[0]

        # 记录尺寸，进行变换，以对每一时间步的输出进行late fuse
        out_size_B_1 = outputs1.size(0)
        out_size_S_1 = outputs1.size(1)
        out_size_H_1 = outputs1.size(2)

        out_size_B_2 = outputs2.size(0)
        out_size_S_2 = outputs2.size(1)
        out_size_H_2 = outputs2.size(2)

        # print(outputs.shape)
        # [X, H]
        outputs1 = outputs1.view(out_size_S_1*out_size_B_1, out_size_H_1)
        outputs2 = outputs2.view(out_size_S_2*out_size_B_2, out_size_H_2)

        outputs1 = self.fc_cnn(self.dropout(outputs1))
        outputs2 = self.fc_bert(self.dropout(outputs2))

        # [X, H] -> [X, 2*H]
        outputs = torch.cat([outputs1, outputs2], dim=-1)
        outputs = self.activate(outputs)
        x = self.fc(self.dropout(outputs))

        return x




class ChineseRobertaNmlCNNLstmPunc(nn.Module):
    # NOTE bert hidden_size=384
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ChineseRobertaNmlCNNLstmPunc, self).__init__()
        self.bert = BertModel.from_pretrained('./models/chinese-roberta-wwm-ext')
        self.bert.embedding_size = 768
        self.hidden_size = 200
        print(type(self.bert))
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.encoder = EncoderStackedNmlCnnLSTMLSTM(
                bert=self.bert,
                hidden_size=self.hidden_size,
                n_layers=3,
                # 额外的参数
                # cnn -> rnn
                # NOTE 使用了args作为参数传递工具
                cnn_kernel_size=(5, 20),
                # rnn -> cnn
                # cnn_kernel_size=(3, hidden_size*2),
                cnn_filter_num=5
            )

        # 以rnn输出为输入
        self.fc = nn.Linear(
                # fc_input：late_fusion `f_t = a_t W_{fa} ◦ σ(a_t W_{fa} W_{ff} + h_t W_{fh} + b_f) + h_t `
                # size: [self.hidden_size*2] 双向的尺寸
                self.bert.embedding_size*2*2,
                output_size
            )

        # 以单向的rnn输出和CNN输出 作为 输入
        # self.fc = nn.Linear(
        #         # rnn_out
        #         hidden_size * 2 +
        #         # cnn_out
        #         (hidden_size * 2 - self.encoder.cnn_kernel_size[1] + 1) *
        #         self.encoder.cnn_filter_num,
        #         num_class
        #     )

        # NOTE rnn_hidden*2 使用bert中间层hidden_state 384
        # self.fc = nn.Linear(384*2, output_size)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print('enc_hidden size', enc_hidden.shape)
        # [B, S, H]
        outputs = self.encoder(x)     # 输入pack，lstm默认输出pack

        # 记录尺寸，进行变换，以对每一时间步的输出进行late fuse
        out_size_B = outputs.size(0)
        out_size_S = outputs.size(1)
        out_size_H = outputs.size(2)

        # print(out_size_S, out_size_B)
        # print(outputs[1].shape)
        outputs = outputs.contiguous()
        # print(outputs.shape)
        outputs = outputs.view(out_size_S*out_size_B, out_size_H)

        outputs = outputs.contiguous()


        # for i in range(l):
        #     print('{} shape:'.format(i), x[i].shape)
        # print('x', type(x))
        # print('shape', x.shape)
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(outputs)
        return x


class ChineseRobertaNoCNNLstmPunc(nn.Module):
    # NOTE bert hidden_size=384
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(ChineseRobertaNoCNNLstmPunc, self).__init__()
        self.bert = BertModel.from_pretrained('./models/chinese-roberta-wwm-ext')
        self.bert.embedding_size = 768
        self.hidden_size = 200
        print(type(self.bert))
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.encoder = EncoderStackedNoCnnLSTMLSTM(
                bert=self.bert,
                hidden_size=self.hidden_size,
                n_layers=3,
                # 额外的参数
                # cnn -> rnn
                # NOTE 使用了args作为参数传递工具
                cnn_kernel_size=(5, 20),
                # rnn -> cnn
                # cnn_kernel_size=(3, hidden_size*2),
                cnn_filter_num=5
            )

        # 以rnn输出为输入
        self.fc = nn.Linear(
                # fc_input：late_fusion `f_t = a_t W_{fa} ◦ σ(a_t W_{fa} W_{ff} + h_t W_{fh} + b_f) + h_t `
                # size: [self.hidden_size*2] 双向的尺寸
                self.hidden_size*2,
                output_size
            )

        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # print('enc_hidden size', enc_hidden.shape)
        # [B, S, H]
        outputs = self.encoder(x)     # 输入pack，lstm默认输出pack

        # 记录尺寸，进行变换，以对每一时间步的输出进行late fuse
        out_size_B = outputs.size(0)
        out_size_S = outputs.size(1)
        out_size_H = outputs.size(2)

        # print(out_size_S, out_size_B)
        # print(outputs[1].shape)
        outputs = outputs.contiguous()
        # print(outputs.shape)
        outputs = outputs.view(out_size_S*out_size_B, out_size_H)

        outputs = outputs.contiguous()


        # for i in range(l):
        #     print('{} shape:'.format(i), x[i].shape)
        # print('x', type(x))
        # print('shape', x.shape)
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(outputs)
        return x


class BertChineseRNNnewLinearPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChineseRNNnewLinearPunc, self).__init__()
        print("fucking code**************")
        self.bert = AutoModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768
        # 批标准化
        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.bn = nn.BatchNorm1d(segment_size*self.bert_size)
        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.gru = nn.GRU(self.bert_size, self.bert_size, 2, bidirectional=True, batch_first=True)
        # # rnn_hidden*2 使用bert中间层hidden_state 384
        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.fc = nn.Linear(self.bert_size*2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 修改后
        # print('input 的shape', x.shape)
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
            h = Variable(torch.zeros(self.gru.num_layers*2, batch_size, self.gru.hidden_size)).to(device)
            # h for storing hidden layer weight
            return h

        hidden = init_gru_hidden(x.shape[0])
        x, hidden = self.gru(x, None)
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
        # out [B*S, num_class]
        return x


class BertChineseRNNnoBnLinearPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChineseRNNnoBnLinearPunc, self).__init__()
        print("fucking code**************")
        self.name = 'BertChineseRNNnoBnLinearPunc'
        self.bert = AutoModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.gru = nn.GRU(self.bert_size, self.bert_size, 2, bidirectional=True, batch_first=True)
        # # rnn_hidden*2 使用bert中间层hidden_state 384
        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.fc = nn.Linear(self.bert_size*2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 修改后
        # print('input 的shape', x.shape)
        x = self.bert(x)[0]

        # GRU*************************************
        # init_GRU_hidden
        def init_gru_hidden(batch_size):
            # when NOT bidirection (layer, B, H)
            h = Variable(torch.zeros(self.gru.num_layers*2, batch_size, self.gru.hidden_size)).to(device)
            # h for storing hidden layer weight
            return h

        hidden = init_gru_hidden(x.shape[0])
        x, hidden = self.gru(x, None)
        x = x.contiguous()
        # ***************************************
        # l = len(x)
        # for i in range(l):
        #     print('{} shape:'.format(i), x[i].shape)
        # print('x', type(x))
        # print('shape', x.shape)

        x = x.view(-1, x.shape[2])
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class BertChineseLinearPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChineseLinearPunc, self).__init__()
        print("fucking code**************")
        self.bert = AutoModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert_size, output_size)

    def forward(self, x):
        # 修改后
        # print('input 的shape', x.shape)
        x = self.bert(x)[0]

        x = x.view(-1, x.shape[2])
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class RobertaChineseLinearPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(RobertaChineseLinearPunc, self).__init__()
        print("fucking code**************")
        self.bert = BertModel.from_pretrained('./models/chinese-roberta-wwm-ext/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert_size, output_size)

    def forward(self, x):
        # 修改后
        # print('input 的shape', x.shape)
        x = self.bert(x)[0]

        x = x.view(-1, x.shape[2])
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class BertChineseSegHiddenLinearPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChineseSegHiddenLinearPunc, self).__init__()
        print("fucking code**************")
        self.bert = AutoModel.from_pretrained('./models/bert_base_chinese/')
        self.bert_seg = SegBertChineseLinearPunc(segment_size, output_size, dropout, vocab_size)
        self.bert_seg.load_state_dict(torch.load('./models/3-SegBertChineseLinearPunc/model'))
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert_size+self.bert_size, output_size)

    def forward(self, x):
        # 修改后
        # print('input 的shape', x.shape)
        feature_bert = self.bert(x)[0]
        feature_seg = self.bert_seg.bert(x)[0]
        # for k, v in self.bert_seg.named_parameters():
        #     print(v.requires_grad)

        x = torch.cat([feature_bert, feature_seg], dim=-1)
        x = x.view(-1, x.shape[2])
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class RobertaLstmChineseSegBertLinearPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(RobertaLstmChineseSegBertLinearPunc, self).__init__()
        print("fucking code**************")
        self.bert = BertModel.from_pretrained('./models/chinese-roberta-wwm-ext/')
        self.bert_seg = SegRobertaChineseLSTMLinearPunc(segment_size, output_size, dropout, vocab_size)
        self.bert_seg.load_state_dict(torch.load('./models/5_SegRobertaChineseLSTMLinearPunc/model'))
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        # size = Bert + BiLstm(bert)
        self.fc = nn.Linear(self.bert_size+self.bert_size*2, output_size)

    def forward(self, x):
        # 修改后
        # print('input 的shape', x.shape)
        feature_bert = self.bert(x)[0]
        # 使用seg模型提取特征
        feature_seg = self.bert_seg.bert(x)[0]
        feature_seg, _ = self.bert_seg.lstm(feature_seg)
        # for k, v in self.bert_seg.named_parameters():
        #     print(v.requires_grad)

        x = torch.cat([feature_bert, feature_seg], dim=-1)
        x = x.view(-1, x.shape[2])
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class RobertaLstmChineseSegBertLstmLinearPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(RobertaLstmChineseSegBertLstmLinearPunc, self).__init__()
        print("fucking code**************")
        self.bert = BertModel.from_pretrained('./models/chinese-roberta-wwm-ext/')
        self.bert_seg = SegRobertaChineseLSTMLinearPunc(segment_size, output_size, dropout, vocab_size)
        self.bert_seg.load_state_dict(torch.load('./models/5_SegRobertaChineseLSTMLinearPunc/model'))
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768

        # 在两个bert之上，增加一个LSTM进行提取
        self.lstm = nn.LSTM(
            self.bert_size+self.bert_size*2,
            self.bert_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert_size*2, output_size)

    def forward(self, x):
        # 修改后
        # print('input 的shape', x.shape)
        feature_bert = self.bert(x)[0]
        # 使用seg模型提取特征
        feature_seg = self.bert_seg.bert(x)[0]
        feature_seg, _ = self.bert_seg.lstm(feature_seg)
        # for k, v in self.bert_seg.named_parameters():
        #     print(v.requires_grad)

        x = torch.cat([feature_bert, feature_seg], dim=-1)

        # 经过统一的LSTM进行学习
        x, _ = self.lstm(x)
        x = x.contiguous()

        # linear层
        x = x.view(-1, x.shape[2])
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class BertChineseCNNreplace(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChineseCNNreplace, self).__init__()
        print("fucking code**************")
        self.bert = BertModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768

        self.conv = nn.ModuleList()
        cnn_kernel_size = (3, 51)
        cnn_filter_num = 10
        cnn_layer_num = 1
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num
        self.cnn_layer_num = cnn_layer_num

        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
            })
            self.conv.append(module_tmp)

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert_size, output_size)

    def forward(self, x):
        input_ids = x
        attention_mask = None
        token_type_ids = None
        position_ids = None
        head_mask = None
        inputs_embeds = None
        encoder_hidden_states = None
        encoder_attention_mask = None
        output_attentions = None
        output_hidden_states = None

        output_attentions = output_attentions if output_attentions is not None else self.bert.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bert.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.bert.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.bert.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)

        embedding_output = self.bert.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # *************cnn*********
        cnn_out = embedding_output.unsqueeze(1)
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            cnn_out = w * torch.sigmoid(v)

        temp = cnn_out[:, 0, :, :] + cnn_out[:, 1, :, :]
        # print("cnn_size", temp.shape)
        for i in range(2, self.cnn_filter_num):
            temp = temp + cnn_out[:, i, :, :]
        embedding_output = temp
        # ******************cnn*****end***********
        # print("cnn_size", embedding_output.shape)
        # print("hidden_size", output_hidden_states)
        # print("hidden_size", output_attentions)

        encoder_outputs = self.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.bert.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here

        x = outputs[0]
        # 修改后
        # print('input 的shape', x.shape)

        x = x.view(-1, x.shape[2])
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class BertChineseCNNreplaceBert(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChineseCNNreplaceBert, self).__init__()
        print("fucking code**************")
        self.bert = BertModel.from_pretrained('./models/bert_base_chinese/')
        self.bert_2 = BertModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768

        self.conv = nn.ModuleList()
        cnn_kernel_size = (3, 51)
        cnn_filter_num = 10
        cnn_layer_num = 3
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num
        self.cnn_layer_num = cnn_layer_num

        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1 if i == 0 else cnn_filter_num,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   (cnn_kernel_size[1] - 1) // 2)),
            })
            self.conv.append(module_tmp)

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        self.fc_size = self.bert_size+self.bert_size
        self.fc = nn.Linear(self.fc_size, output_size)

    def forward(self, x):
        input_ids = x
        attention_mask = None
        token_type_ids = None
        position_ids = None
        head_mask = None
        inputs_embeds = None
        encoder_hidden_states = None
        encoder_attention_mask = None
        output_attentions = None
        output_hidden_states = None

        output_attentions = output_attentions if output_attentions is not None else self.bert.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bert.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.bert.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.bert.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)

        embedding_output = self.bert.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # *************cnn*********
        cnn_out = embedding_output.unsqueeze(1)
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            cnn_out = w * torch.sigmoid(v)

        temp = cnn_out[:, 0, :, :] + cnn_out[:, 1, :, :]
        # print("cnn_size", temp.shape)
        for i in range(2, self.cnn_filter_num):
            temp = temp + cnn_out[:, i, :, :]
        embedding_output = temp
        # ******************cnn*****end***********
        # print("cnn_size", embedding_output.shape)
        # print("hidden_size", output_hidden_states)
        # print("hidden_size", output_attentions)

        encoder_outputs = self.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.bert.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here

        output1 = outputs[0]

        # bert_2
        output2 = self.bert_2(x)[0]

        x = torch.cat([output1, output2], dim=-1)

        # 修改后
        # print('input 的shape', x.shape)

        x = x.view(-1, x.shape[2])

        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class BertChinese4lyrHiddenCNN(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChinese4lyrHiddenCNN, self).__init__()
        print("fucking code**************")
        self.bert = BertModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_2 = BertModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768
        # 选用倒数四层
        self.bert_hidden_num = 2

        self.conv = nn.ModuleList()
        cnn_kernel_size = (3, self.bert_size*self.bert_hidden_num)
        cnn_filter_num = self.bert_size*self.bert_hidden_num
        cnn_layer_num = 5
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num
        self.cnn_layer_num = cnn_layer_num

        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   0)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   0)),
            })
            self.conv.append(module_tmp)

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        self.fc_size = self.bert_size*self.bert_hidden_num
        self.fc = nn.Linear(self.fc_size, output_size)

    def forward(self, x):
        input_ids = x
        attention_mask = None
        token_type_ids = None
        position_ids = None
        head_mask = None
        inputs_embeds = None
        encoder_hidden_states = None
        encoder_attention_mask = None
        output_attentions = None
        output_hidden_states = None

        output_attentions = output_attentions if output_attentions is not None else self.bert.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bert.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.bert.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.bert.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)

        embedding_output = self.bert.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # encoder 分离*****************
        # 可以直接用 参数 output_hidden_states=True
        # (outputs, (hidden0,1,2,3.....16))
        output_hidden_states = True
        encoder_outputs = self.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        print(encoder_outputs[0].shape)
        sequence_output_4lyr = torch.cat(list(encoder_outputs[1][-2:]), dim=-1)
        print(sequence_output_4lyr.shape)
        # add hidden_states and attentions if they are here

        # *************cnn*********
        cnn_out = sequence_output_4lyr.unsqueeze(1)
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            cnn_out = w * torch.sigmoid(v)
            # 重新拼成768*4的纬度
            cnn_out = torch.cat([cnn_out[:, i_tmp, :, :] for i_tmp in range(self.cnn_filter_num)], dim=-1)
            cnn_out = cnn_out.unsqueeze(1)

        # ******************cnn*****end***********
        # print("cnn_size", embedding_output.shape)
        # print("hidden_size", output_hidden_states)
        # print("hidden_size", output_attentions)

        output1 = cnn_out.squeeze(1)

        x = output1

        # 修改后
        # print('input 的shape', x.shape)

        x = x.view(-1, x.shape[2])

        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class BertChineseBigStrideCNN(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChineseBigStrideCNN, self).__init__()
        print("fucking code**************")
        self.bert = BertModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_2 = BertModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768
        # 选用倒数四层
        self.bert_hidden_num = 1

        self.conv = nn.ModuleList()
        # cnn_kernel_size = (11, self.bert_size*self.bert_hidden_num)
        cnn_kernel_size = (11, 765)
        cnn_filter_num = self.bert_size*self.bert_hidden_num // 4
        cnn_layer_num = 2
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num
        self.cnn_layer_num = cnn_layer_num

        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((0, 0))),
                'conv_v_{}'.format(i):
                nn.Conv2d(1,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((0, 0))),
            })
            self.conv.append(module_tmp)

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        self.fc_size = self.bert_size*self.bert_hidden_num
        self.fc = nn.Linear(self.fc_size, output_size)

    def forward(self, x):
        input_ids = x
        attention_mask = None
        token_type_ids = None
        position_ids = None
        head_mask = None
        inputs_embeds = None
        encoder_hidden_states = None
        encoder_attention_mask = None
        output_attentions = None
        output_hidden_states = None

        output_attentions = output_attentions if output_attentions is not None else self.bert.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bert.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.bert.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.bert.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)

        embedding_output = self.bert.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # encoder 分离*****************
        # 可以直接用 参数 output_hidden_states=True
        # (outputs, (hidden0,1,2,3.....16))
        output_hidden_states = True
        encoder_outputs = self.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        print(encoder_outputs[0].shape)
        sequence_output_4lyr = torch.cat(list(encoder_outputs[1][-self.bert_hidden_num:]), dim=-1)
        print(sequence_output_4lyr.shape)
        # add hidden_states and attentions if they are here

        # *************cnn*********
        embeding_clone = torch.clone(sequence_output_4lyr)
        cnn_out = sequence_output_4lyr.unsqueeze(1)
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            cnn_out = w * torch.sigmoid(v)
            # 重新拼成768*1的纬度
            cnn_out = torch.cat([cnn_out[:, i_tmp, :, :] for i_tmp in range(self.cnn_filter_num)], dim=-1)
            cnn_out = torch.cat([embeding_clone[:,:(self.cnn_kernel_size[0] - 1) // 2, :], cnn_out, embeding_clone[:, -(self.cnn_kernel_size[0] - 1) // 2:, :]], dim=1)
            cnn_out = cnn_out.unsqueeze(1)

        # ******************cnn*****end***********
        # print("cnn_size", embedding_output.shape)
        # print("hidden_size", output_hidden_states)
        # print("hidden_size", output_attentions)

        output1 = cnn_out.squeeze(1)

        x = output1

        # 修改后
        # print('input 的shape', x.shape)

        x = x.view(-1, x.shape[2])

        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class BertChinese4lyrHiddenLinear(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChinese4lyrHiddenLinear, self).__init__()
        print("fucking code**************")
        self.bert = BertModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_2 = BertModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768
        # 选用倒数四层
        self.bert_hidden_num = 2

        self.conv = nn.ModuleList()
        cnn_kernel_size = (3, self.bert_size*self.bert_hidden_num)
        cnn_filter_num = self.bert_size*self.bert_hidden_num
        cnn_layer_num = 5
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num
        self.cnn_layer_num = cnn_layer_num

        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   0)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1,
                          cnn_filter_num,
                          cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   0)),
            })
            self.conv.append(module_tmp)

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        self.fc_size = self.bert_size*self.bert_hidden_num
        self.fc = nn.Linear(self.fc_size, output_size)

    def forward(self, x):
        input_ids = x
        attention_mask = None
        token_type_ids = None
        position_ids = None
        head_mask = None
        inputs_embeds = None
        encoder_hidden_states = None
        encoder_attention_mask = None
        output_attentions = None
        output_hidden_states = None

        output_attentions = output_attentions if output_attentions is not None else self.bert.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bert.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.bert.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.bert.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)

        embedding_output = self.bert.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # encoder 分离*****************
        # 可以直接用 参数 output_hidden_states=True
        # (outputs, (hidden0,1,2,3.....16))
        output_hidden_states = True
        encoder_outputs = self.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        print(encoder_outputs[0].shape)
        sequence_output_4lyr = torch.cat(list(encoder_outputs[1][-2:]), dim=-1)
        print(sequence_output_4lyr.shape)
        # add hidden_states and attentions if they are here

        # # *************cnn*********
        # cnn_out = sequence_output_4lyr.unsqueeze(1)
        # for i, conv_dict in enumerate(self.conv):
        #     w = conv_dict['conv_w_{}'.format(i)](cnn_out)
        #     v = conv_dict['conv_v_{}'.format(i)](cnn_out)
        #     cnn_out = w * torch.sigmoid(v)
        #     # 重新拼成768*4的纬度
        #     cnn_out = torch.cat([cnn_out[:, i_tmp, :, :] for i_tmp in range(self.cnn_filter_num)], dim=-1)
        #     cnn_out = cnn_out.unsqueeze(1)

        # # ******************cnn*****end***********
        # # print("cnn_size", embedding_output.shape)
        # # print("hidden_size", output_hidden_states)
        # # print("hidden_size", output_attentions)

        # output1 = cnn_out.squeeze(1)

        x = sequence_output_4lyr

        # 修改后
        # print('input 的shape', x.shape)

        x = x.view(-1, x.shape[2])

        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class BertChineseSlimCNNBert(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChineseSlimCNNBert, self).__init__()
        print("fucking code**************")
        self.bert = BertModel.from_pretrained('./models/bert_base_chinese/')
        self.bert_2 = BertModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768

        self.conv = nn.ModuleList()
        cnn_kernel_size = (3, self.bert_size)
        cnn_filter_num = self.bert_size
        cnn_layer_num = 5
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num
        self.cnn_layer_num = cnn_layer_num

        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1,
                          self.cnn_filter_num,
                          self.cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   0)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1,
                          self.cnn_filter_num,
                          self.cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   0))
            })
            self.conv.append(module_tmp)

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        self.fc_size = self.bert_size+self.bert_size
        self.fc = nn.Linear(self.fc_size, output_size)

    def forward(self, x):
        emb1 = self.bert(x)[0]
        emb2 = self.bert_2(x)[0]
        # *************cnn*********
        # 加入了skip_gram
        skip_connection = emb1.unsqueeze(1)
        cnn_out = emb1.unsqueeze(1)
        # print()
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            cnn_out = w * torch.sigmoid(v)
            # 有bert_size个filter，拼成原来的size
            cnn_out = torch.cat([cnn_out[:, i_tmp, :, :] for i_tmp in range(self.cnn_filter_num)], dim=-1)
            cnn_out = cnn_out.unsqueeze(1)
            cnn_out = skip_connection + cnn_out
            skip_connection = cnn_out

        # ******************cnn*****end***********
        output1 = cnn_out.squeeze(1)

        # bert_2
        output2 = emb2

        x = torch.cat([output1, output2], dim=-1)

        # 修改后
        # print('input 的shape', x.shape)

        x = x.view(-1, x.shape[2])

        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class BertChineseSlimCNNBertLSTM(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChineseSlimCNNBertLSTM, self).__init__()
        print("fucking code**************")
        self.bert = BertModel.from_pretrained('./models/bert_base_chinese/')
        self.bert_2 = BertModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768

        self.conv = nn.ModuleList()
        cnn_kernel_size = (3, self.bert_size)
        cnn_filter_num = self.bert_size
        cnn_layer_num = 5
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num
        self.cnn_layer_num = cnn_layer_num

        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1,
                          self.cnn_filter_num,
                          self.cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   0)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1,
                          self.cnn_filter_num,
                          self.cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   0))
            })
            self.conv.append(module_tmp)


        # self.lstm_cnn = nn.LSTM(
        #     self.bert_size,
        #     self.bert_size,
        #     num_layers=3,
        #     batch_first=True,
        #     bidirectional=True
        # )

        self.lstm_bert = nn.LSTM(
            self.bert_size,
            self.bert_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        # bert*1  lstm *2 
        self.fc_size = self.bert_size+self.bert_size*2
        self.fc = nn.Linear(self.fc_size, output_size)

    def forward(self, x):
        emb1 = self.bert(x)[0]
        emb2 = self.bert_2(x)[0]
        # *************cnn*********
        # 加入了skip_gram
        skip_connection = emb1.unsqueeze(1)
        cnn_out = emb1.unsqueeze(1)
        # print()
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            cnn_out = w * torch.sigmoid(v)
            # 有bert_size个filter，拼成原来的size
            cnn_out = torch.cat([cnn_out[:, i_tmp, :, :] for i_tmp in range(self.cnn_filter_num)], dim=-1)
            cnn_out = cnn_out.unsqueeze(1)
            cnn_out = skip_connection + cnn_out
            skip_connection = cnn_out

        # ******************cnn*****end***********
        output1 = cnn_out.squeeze(1)

        # bert_2
        output2 = emb2
        output2, _ = self.lstm_bert(output2)

        x = torch.cat([output1, output2], dim=-1)

        # 修改后
        # print('input 的shape', x.shape)

        x = x.view(-1, x.shape[2])

        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class BertChineseEmbSlimCNNBert(nn.Module):
    # 在bert的embedding后面直接加一个CNN进行，和bert的组合
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChineseEmbSlimCNNBert, self).__init__()
        print("fucking code**************")
        self.bert = BertModel.from_pretrained('./models/bert_base_chinese/')
        self.bert_2 = BertModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768

        self.conv = nn.ModuleList()
        cnn_kernel_size = (3, self.bert_size)
        cnn_filter_num = self.bert_size
        cnn_layer_num = 5
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num
        self.cnn_layer_num = cnn_layer_num

        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1,
                          self.cnn_filter_num,
                          self.cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   0)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1,
                          self.cnn_filter_num,
                          self.cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   0))
            })
            self.conv.append(module_tmp)

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        self.fc_size = self.bert_size+self.bert_size
        self.fc = nn.Linear(self.fc_size, output_size)

    def forward(self, x):
        # bert直出
        emb2 = self.bert_2(x)[0]

        input_ids = x
        attention_mask = None
        token_type_ids = None
        position_ids = None
        head_mask = None
        inputs_embeds = None
        encoder_hidden_states = None
        encoder_attention_mask = None
        output_attentions = None
        output_hidden_states = None

        output_attentions = output_attentions if output_attentions is not None else self.bert.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bert.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.bert.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.bert.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)

        # bert embedding
        embedding_output = self.bert.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # *************cnn*********
        # 加入了skip_gram
        skip_connection = embedding_output.unsqueeze(1)
        cnn_out = embedding_output.unsqueeze(1)
        # print()
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            cnn_out = w * torch.sigmoid(v)
            # 有bert_size个filter，拼成原来的size
            cnn_out = torch.cat([cnn_out[:, i_tmp, :, :] for i_tmp in range(self.cnn_filter_num)], dim=-1)
            cnn_out = cnn_out.unsqueeze(1)
            cnn_out = skip_connection + cnn_out
            skip_connection = cnn_out

        # ******************cnn*****end***********
        output1 = cnn_out.squeeze(1)

        # bert_2
        output2 = emb2

        x = torch.cat([output1, output2], dim=-1)

        # 修改后
        # print('input 的shape', x.shape)

        x = x.view(-1, x.shape[2])

        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class BertChineseEmbSlimCNNlstmBert(nn.Module):
    # 在bert的embedding后面直接加一个CNN进行，和bert的组合
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChineseEmbSlimCNNlstmBert, self).__init__()
        print("fucking code**************")
        self.bert = BertModel.from_pretrained('./models/bert_base_chinese/')
        self.bert_2 = BertModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768

        self.conv = nn.ModuleList()
        cnn_kernel_size = (3, self.bert_size)
        cnn_filter_num = self.bert_size
        cnn_layer_num = 5
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num
        self.cnn_layer_num = cnn_layer_num

        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1,
                          self.cnn_filter_num,
                          self.cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   0)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1,
                          self.cnn_filter_num,
                          self.cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   0))
            })
            self.conv.append(module_tmp)

        self.lstm_cnn = nn.LSTM(
            self.bert_size,
            self.bert_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        # lstm hidden*2
        self.fc_size = self.bert_size+self.bert_size*2
        self.fc = nn.Linear(self.fc_size, output_size)

    def forward(self, x):
        emb2 = self.bert_2(x)[0]

        input_ids = x
        attention_mask = None
        token_type_ids = None
        position_ids = None
        head_mask = None
        inputs_embeds = None
        encoder_hidden_states = None
        encoder_attention_mask = None
        output_attentions = None
        output_hidden_states = None

        output_attentions = output_attentions if output_attentions is not None else self.bert.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bert.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.bert.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.bert.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)

        embedding_output = self.bert.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # *************cnn*********
        # 加入了skip_gram
        skip_connection = embedding_output.unsqueeze(1)
        cnn_out = embedding_output.unsqueeze(1)
        # print()
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            cnn_out = w * torch.sigmoid(v)
            # 有bert_size个filter，拼成原来的size
            cnn_out = torch.cat([cnn_out[:, i_tmp, :, :] for i_tmp in range(self.cnn_filter_num)], dim=-1)
            cnn_out = cnn_out.unsqueeze(1)
            cnn_out = skip_connection + cnn_out
            skip_connection = cnn_out

        # ******************cnn*****end***********
        output1 = cnn_out.squeeze(1)

        output1, _ = self.lstm_cnn(output1)

        # bert_2
        output2 = emb2

        x = torch.cat([output1, output2], dim=-1)

        # 修改后
        # print('input 的shape', x.shape)

        x = x.view(-1, x.shape[2])

        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class RobertaChineseEmbSlimCNNlstmBert(nn.Module):
    # 在bert的embedding后面直接加一个CNN进行，和bert的组合
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(RobertaChineseEmbSlimCNNlstmBert, self).__init__()
        print("fucking code**************")
        self.bert = BertModel.from_pretrained('./models/chinese-roberta-wwm-ext/')
        self.bert_2 = BertModel.from_pretrained('./models/chinese-roberta-wwm-ext/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768

        self.conv = nn.ModuleList()
        cnn_kernel_size = (3, self.bert_size)
        cnn_filter_num = self.bert_size
        cnn_layer_num = 5
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num
        self.cnn_layer_num = cnn_layer_num

        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1,
                          self.cnn_filter_num,
                          self.cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   0)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1,
                          self.cnn_filter_num,
                          self.cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   0))
            })
            self.conv.append(module_tmp)

        self.lstm_cnn = nn.LSTM(
            self.bert_size,
            self.bert_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        # lstm hidden*2
        self.fc_size = self.bert_size+self.bert_size*2
        self.fc = nn.Linear(self.fc_size, output_size)

    def forward(self, x):
        emb2 = self.bert_2(x)[0]

        input_ids = x
        attention_mask = None
        token_type_ids = None
        position_ids = None
        head_mask = None
        inputs_embeds = None
        encoder_hidden_states = None
        encoder_attention_mask = None
        output_attentions = None
        output_hidden_states = None

        output_attentions = output_attentions if output_attentions is not None else self.bert.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bert.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.bert.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.bert.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)

        embedding_output = self.bert.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # *************cnn*********
        # 加入了skip_gram
        skip_connection = embedding_output.unsqueeze(1)
        cnn_out = embedding_output.unsqueeze(1)
        # print()
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            cnn_out = w * torch.sigmoid(v)
            # 有bert_size个filter，拼成原来的size
            cnn_out = torch.cat([cnn_out[:, i_tmp, :, :] for i_tmp in range(self.cnn_filter_num)], dim=-1)
            cnn_out = cnn_out.unsqueeze(1)
            cnn_out = skip_connection + cnn_out
            skip_connection = cnn_out

        # ******************cnn*****end***********
        output1 = cnn_out.squeeze(1)

        output1, _ = self.lstm_cnn(output1)

        # bert_2
        output2 = emb2

        x = torch.cat([output1, output2], dim=-1)

        # 修改后
        # print('input 的shape', x.shape)

        x = x.view(-1, x.shape[2])

        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x

class BertChineseEmbSlimCNNlstmBertLSTM(nn.Module):
    # 在bert的embedding后面直接加一个CNN进行，和bert的组合
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChineseEmbSlimCNNlstmBertLSTM, self).__init__()
        print("fucking code**************")
        self.bert = BertModel.from_pretrained('./models/bert_base_chinese/')
        self.bert_2 = BertModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768

        self.conv = nn.ModuleList()
        cnn_kernel_size = (3, self.bert_size)
        cnn_filter_num = self.bert_size
        cnn_layer_num = 5
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter_num = cnn_filter_num
        self.cnn_layer_num = cnn_layer_num

        for i in range(cnn_layer_num):
            module_tmp = nn.ModuleDict({
                'conv_w_{}'.format(i):
                nn.Conv2d(1,
                          self.cnn_filter_num,
                          self.cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   0)),
                'conv_v_{}'.format(i):
                nn.Conv2d(1,
                          self.cnn_filter_num,
                          self.cnn_kernel_size,
                          padding=((cnn_kernel_size[0] - 1) // 2,
                                   0))
            })
            self.conv.append(module_tmp)

        self.lstm_cnn = nn.LSTM(
            self.bert_size,
            self.bert_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )

        self.lstm_bert = nn.LSTM(
            self.bert_size,
            self.bert_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        # lstm hidden*2
        self.fc_size = self.bert_size*2+self.bert_size*2
        self.fc = nn.Linear(self.fc_size, output_size)

    def forward(self, x):
        emb2 = self.bert_2(x)[0]

        input_ids = x
        attention_mask = None
        token_type_ids = None
        position_ids = None
        head_mask = None
        inputs_embeds = None
        encoder_hidden_states = None
        encoder_attention_mask = None
        output_attentions = None
        output_hidden_states = None

        output_attentions = output_attentions if output_attentions is not None else self.bert.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.bert.config.output_hidden_states
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.bert.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.bert.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.bert.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.bert.get_head_mask(head_mask, self.bert.config.num_hidden_layers)

        embedding_output = self.bert.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # *************cnn*********
        # 加入了skip_gram
        skip_connection = embedding_output.unsqueeze(1)
        cnn_out = embedding_output.unsqueeze(1)
        # print()
        for i, conv_dict in enumerate(self.conv):
            w = conv_dict['conv_w_{}'.format(i)](cnn_out)
            v = conv_dict['conv_v_{}'.format(i)](cnn_out)
            cnn_out = w * torch.sigmoid(v)
            # 有bert_size个filter，拼成原来的size
            cnn_out = torch.cat([cnn_out[:, i_tmp, :, :] for i_tmp in range(self.cnn_filter_num)], dim=-1)
            cnn_out = cnn_out.unsqueeze(1)
            cnn_out = skip_connection + cnn_out
            skip_connection = cnn_out

        # ******************cnn*****end***********
        output1 = cnn_out.squeeze(1)

        output1, _ = self.lstm_cnn(output1)

        # bert_2
        output2 = emb2
        output2, _ = self.lstm_bert(output2)

        x = torch.cat([output1, output2], dim=-1)

        # 修改后
        # print('input 的shape', x.shape)

        x = x.view(-1, x.shape[2])

        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class BertChineseLSTMLinearPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChineseLSTMLinearPunc, self).__init__()
        print("fucking code**************")
        self.bert = AutoModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.lstm = nn.LSTM(
            self.bert_size,
            self.bert_size,
            num_layers=3,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert_size*2, output_size)

    def forward(self, x):
        # 修改后
        # print('input 的shape', x.shape)
        x = self.bert(x)[0]

        x, _ = self.lstm(x)

        x = x.contiguous()

        x = x.view(-1, x.shape[2])
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x


class BertChineseLinearCRFPunc(nn.Module):
    def __init__(self, segment_size, output_size, dropout, vocab_size):
        super(BertChineseLinearCRFPunc, self).__init__()
        print("fucking code**************")
        self.bert = AutoModel.from_pretrained('./models/bert_base_chinese/')
        # self.bert_vocab_size = vocab_size
        # self.bn = nn.BatchNorm1d(segment_size*self.bert_vocab_size)
        # self.fc = nn.Linear(segment_size*self.bert_vocab_size, output_size)
        self.bert_size = 768

        # NOTE dense*2 使用bert中间层 dense hidden_state self.bert_size
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert_size, output_size)

    def forward(self, x):
        # 修改后
        # print('input 的shape', x.shape)
        x = self.bert(x)[0]

        x = x.view(-1, x.shape[2])
        # x = self.fc(self.dropout(self.bn(x)))
        x = self.fc(self.dropout(x))
        # out [B*S, num_class]
        return x

    def loss(self, feats, mask, tags):
        """
        feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        :return:
        """
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value
