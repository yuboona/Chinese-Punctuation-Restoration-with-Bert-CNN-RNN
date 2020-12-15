# def load_vocab(vocab_path, extra_word_list=[], encoding='utf-8'):
#     n = len(extra_word_list)
#     with open(vocab_path, encoding='utf-8') as vf:
#         vocab = {word.strip(): i+n for i, word in enumerate(vf)}
#     for i, word in enumerate(extra_word_list):
#         vocab[word] = i
#     return vocab
import os

def proc(path, save_path):
    tmp_seqs = open(path, encoding='utf-8').readlines()
    txt_seqs = [i for seq in tmp_seqs for i in seq.split()]
    punc_list = ['，', '。', '？', '！']

    def preprocess():
        """将文本转为单词和应预测标点的id pair
        Parameters
        ----------
        txt : 文本
            文本每个单词跟随一个空格，符号也跟一个空格
        """
        input_r = []
        label_r = []
        # txt_seqs is a list like: ['char', 'char', 'char', '*，*', 'char', ......]
        count = 0
        length = len(txt_seqs)
        for token in txt_seqs:
            count += 1
            if count == length:
                break
            if token in punc_list:
                continue
            punc = txt_seqs[count]
            if punc not in punc_list:
                # print('标点{}：'.format(count), self.punc2id[" "])
                input_r.append(token)
                label_r.append('O')
            else:
                # print('标点{}：'.format(count), self.punc2id[punc])
                input_r.append(token)
                label_r.append(punc)
        with open(os.path.join(save_path, '{}'.format(path)), 'w', encoding='utf-8') as w:
            for i, j in zip(input_r, label_r):
                w.write(i)
                w.write('\t')
                w.write(j)
                w.write('\n')
    preprocess()


# proc('./train')
proc('valid', 'single_out')
# proc('test_valid', 'single_out')
proc('test', 'single_out')
proc('train', 'single_out')


