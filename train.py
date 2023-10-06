import re
import json
import sys
import tqdm
import os
import numpy as np
import random
import torch
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import precision_recall_fscore_support

import argparse

import Model

parser = argparse.ArgumentParser(description='Fact-checking')
parser.add_argument('--origin_data_path', type=str, default="./CHEF", help='original data path')
parser.add_argument('--bert_model_path', type=str, default="bert-base-chinese", help='bert model path')
parser.add_argument('--max_length', type=int, default=512, help='max input length')
parser.add_argument('--gpu', type=int, default=1, help='use gpu is or not?')
parser.add_argument('-device', type=int, default=0, help='gpu device')
parser.add_argument('--doc_embedding_path', type=str, default="data/doc_encoder", help='doc embedding path')
parser.add_argument('--doc_sen_num', type=int, default=181, help='max sentence num in doc')
parser.add_argument('--ebd_dim', type=int, default=768, help='word embedding dim')
parser.add_argument('--n_fea', type=int, default=768, help='input dim in Evidence extract')
parser.add_argument('--n_hid', type=int, default=768 // 8, help='hidden size in Evidence extract')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
parser.add_argument('--n_heads', type=int, default=8, help='attention head num in Evidence extract')
parser.add_argument('--n_layers', type=int, default=1, help='attention layer num in Evidence extract')
parser.add_argument('--out_channels', type=int, default=32, help='channel num in conv layer')
parser.add_argument('--alpha', type=float, default=0.2, help='leakyrelu function parm')
parser.add_argument('--gamma', type=float, default=0.5, help='mask thd')
parser.add_argument('--beta', type=float, default=0.1, help='neigative rate in evidence loss')
parser.add_argument('-batch_size', type=int, default=8, help='batch size')
parser.add_argument('--n_class', type=int, default=3, help='class num')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--bert_lr', type=float, default=1e-5, help='bert lr')
parser.add_argument('--eps', type=float, default=1e-8, help='optimizer parm')
parser.add_argument('--epoch', type=int, default=10, help='train epoch')
parser.add_argument('--warmup_ratio', type=float, default=0.06, help='optimizer parm')
parser.add_argument('--save_model', type=str, default="output", help='save model path')
args = parser.parse_args()


def print_params(model):
    print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters()]))

    print('total trainable parameters:', sum([np.prod(list(p.size())) for p in model.parameters() if p.requires_grad]))


class Reader():
    def __init__(self):

        self.arg = args
        self.domain2id = {
            '政治': 0, '社会': 1, '生活': 1, '文化': 2, '科学': 3, '公卫': 4
        }

        return

    def load(self):
        train_data = json.load(open(self.arg.origin_data_path + '/train.json', 'r', encoding='utf-8'))
        dev_data = json.load(open(self.arg.origin_data_path + '/dev.json', 'r', encoding='utf-8'))
        test_data = json.load(open(self.arg.origin_data_path + '/test.json', 'r', encoding='utf-8'))

        return train_data, dev_data, test_data


class Process():
    def __init__(self):

        self.arg = args

        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
        self.bert = BertModel.from_pretrained(args.bert_model_path)

        if args.gpu:
            self.bert.to(args.device)

        return

    def doc_encoder(self, data, data_label):
        save_path = self.arg.doc_embedding_path + '/' + data_label + '.npy'

        if os.path.exists(save_path):
            doc_encodes = np.load(save_path, allow_pickle=True)
            pad_vec = np.zeros((1, self.arg.ebd_dim), dtype=np.float32)
            doc_encodes = np.concatenate((doc_encodes, pad_vec), axis=0)

            new_data = json.load(open(self.arg.doc_embedding_path + '/' + data_label + '.json', 'r', encoding='utf-8'))

            print("load data!")

            return doc_encodes, new_data

        doc, new_data = [], []

        for d in tqdm.tqdm(data):
            evidence = d['evidence']
            doc_2_ids = []

            for e in evidence.keys():
                e_text = evidence[e]['text'].strip()
                try:
                    if e_text[len(e_text) - 1] not in ('。', '？', '！'):
                        e_text = e_text + '。'

                    e_sen = re.split(r'[。？！]', e_text)
                    e_syms = [match.group() for match in re.finditer(r'[。？！]', e_text)]
                    e_sen = [sen + s for sen, s in zip(e_sen[:len(e_sen)-1], e_syms)]

                    for e_t in e_sen:
                        if e_t != '':
                            if e_t not in doc:
                                doc_2_ids.append(len(doc))
                                doc.append(e_t)
                            else:
                                doc_2_ids.append(doc.index(e_t))
                except:
                    continue

            d['doc_ids'] = list(set(doc_2_ids))
            d['doc_ids'].sort(key=doc_2_ids.index)
            new_data.append(d)

        doc_encodes, input_xs, attention_masks = [], [], []
        for i, x in enumerate(tqdm.tqdm(doc)):
            input_x = self.tokenizer.encode_plus(
                x,
                max_length=self.arg.max_length,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True,
            )
            input_xs.append(input_x['input_ids'].data.cpu().numpy().tolist()[0])
            attention_masks.append(input_x['attention_mask'].data.cpu().numpy().tolist()[0])

            if (i%8 == 0 and i != 0) or (i == len(doc)-1):

                input_xs = torch.LongTensor(np.array(input_xs))
                attention_masks = torch.LongTensor(np.array(attention_masks))

                if self.arg.gpu:
                    input_xs = input_xs.to(self.arg.device)
                    attention_masks = attention_masks.to(self.arg.device)

                output = self.bert(input_ids=input_xs, attention_mask=attention_masks)

                doc_encode = output[1].data.cpu().numpy().tolist()
                doc_encodes.extend(doc_encode)

                input_xs, attention_masks = [], []

        print("finished to embedding! save data ... ")

        doc_encodes = np.array(doc_encodes)
        np.save(save_path, doc_encodes)

        with open(self.arg.doc_embedding_path + '/' + data_label + '.json', 'w', encoding='utf-8') as f_write:
            json.dump(new_data, f_write, ensure_ascii=False)
        f_write.close()

        pad_vec = np.zeros((1, self.arg.ebd_dim), dtype=np.float32)
        doc_encodes = np.concatenate((doc_encodes, pad_vec), axis=0)

        return doc_encodes, new_data

    def set_(self, data):
        max_sen_num = 0
        for d in data:
            max_sen_num = max(max_sen_num, len(d['doc_ids']))

        self.arg.doc_sen_num = max_sen_num

        return

    def processing(self, data, doc_encode, shuffle):

        batch_data = []  # doc_x, doc_adj,  claim_input_ids, label

        data_size = len(data)
        order = list(range(data_size))

        label_0, label_1, label_2 = 0, 0, 0

        if shuffle:
            random.shuffle(order)

        batch_num = data_size // self.arg.batch_size
        if data_size % self.arg.batch_size != 0:
            batch_num += 1

        batch_index = [i for i in range(batch_num)]
        for i in batch_index:
            s_index = i * self.arg.batch_size
            c_index = min(self.arg.batch_size, data_size - s_index)
            c_order = list(order[s_index : s_index + c_index])

            doc_x = np.zeros((c_index, self.arg.doc_sen_num, self.arg.ebd_dim), dtype=np.float32)
            doc_adj = np.zeros((c_index, self.arg.doc_sen_num, self.arg.doc_sen_num), dtype=np.int32)

            claim_input_ids = np.zeros((c_index, self.arg.max_length), dtype=np.int32)

            y = np.zeros((c_index), dtype=np.int32)

            max_doc_sen_num = 0
            max_claim_word_num = 0
            for k, index in enumerate(c_order):
                d = data[index]

                doc_ids = d['doc_ids']
                pad_doc_ids = doc_ids + [len(doc_encode)-1 for _ in range(self.arg.doc_sen_num - len(doc_ids))]

                c_doc_x = doc_encode[pad_doc_ids, :]
                doc_x[k, ] = c_doc_x

                c_adj = np.ones((len(doc_ids), len(doc_ids)), dtype=np.int32)
                doc_adj[k, :len(doc_ids), :len(doc_ids)] = c_adj

                max_doc_sen_num = max(max_doc_sen_num, len(doc_ids))

                claim = d['claim']
                claim_encode = self.tokenizer.encode_plus(
                    claim,
                    max_length=self.arg.max_length,
                    padding='max_length',
                    return_attention_mask=False,
                    return_tensors='pt',
                    truncation=True
                )

                claim_input_ids[k, ] = claim_encode['input_ids'].data.cpu().numpy()

                max_claim_word_num = max(max_claim_word_num, np.count_nonzero(claim_encode['input_ids'].data.cpu().numpy()))

                c_y = d['label']
                y[k] = c_y

                if c_y == 0:
                    label_0 += 1
                elif c_y == 1:
                    label_1 += 1
                elif c_y == 2:
                    label_2 += 1

            batch_data.append([
                torch.FloatTensor(doc_x[:, :max_doc_sen_num, :]),
                torch.LongTensor(doc_adj[:, :max_doc_sen_num, :max_doc_sen_num]),
                torch.LongTensor(claim_input_ids[:, :max_claim_word_num]),
                torch.LongTensor(y),
            ])

        print("label 0 num: %s; label 1 num: %s; label 2 num: %s"%(str(label_0), str(label_1), str(label_2)))

        return batch_data


class Trainer():
    def __init__(self, train_size):
        self.arg = args

        self.model = Model.FC(args)

        if self.arg.gpu:
            torch.cuda.set_device(self.arg.device)
            self.model.cuda()

        self.loss_func = CrossEntropyLoss()

        bert_layer = ['CE']
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in bert_layer)], "lr": self.arg.bert_lr},
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in bert_layer)], "lr": self.arg.lr},
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.arg.lr, eps=self.arg.eps)

        total_steps = train_size * self.arg.epoch
        warmup_steps = int(total_steps * self.arg.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        return

    def train(self, data):
        print("------------- Training -------------")
        self.model.zero_grad()
        self.model.train()

        train_order = list(range(len(data)))
        random.shuffle(train_order)

        step, total_loss = 0, 0
        all_outputs, all_y = np.array([]), np.array([])
        for i in train_order:
            batch_data = data[i]

            if args.gpu:
                batch_data = [d.cuda() for d in batch_data]

            doc_x = batch_data[0]
            doc_adj = batch_data[1]
            claim_input_ids = batch_data[2]
            y = batch_data[3]

            outputs = self.model(
                doc_x=doc_x,
                doc_adj=doc_adj,
                claim_input_ids=claim_input_ids,
            )

            loss = self.loss_func(outputs, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()
            self.scheduler.step()
            self.model.zero_grad()

            step += 1
            total_loss += loss.item()

            outputs = outputs.data.cpu().numpy()
            y = y.data.cpu().numpy()

            outputs = np.argmax(outputs, axis=1).flatten()
            y = y.flatten()

            all_outputs = np.concatenate((all_outputs, outputs), axis=None)
            all_y = np.concatenate((all_y, y), axis=None)

        total_loss /= step

        print("Loss: {:.5}".format(total_loss))

        precision, recall, f1, _ = precision_recall_fscore_support(all_outputs, all_y, average='micro')
        print("F1 (micro): {:.3%}".format(f1))
        precision, recall, f1, _ = precision_recall_fscore_support(all_outputs, all_y, average='macro')
        print("Precision (macro): {:.3%}".format(precision))
        print("Recall (macro): {:.3%}".format(recall))
        print("F1 (macro): {:.3%}".format(f1))

        return total_loss

    def devel(self, data, data_label):
        print("\n")
        print("------------- Devel on %s set -------------" % data_label)
        self.model.eval()

        total_loss, step = 0, 0
        all_outputs, all_y = np.array([]), np.array([])
        for batch_data in data:
            if self.arg.gpu:
                batch_data = [d.cuda() for d in batch_data]

            doc_x = batch_data[0]
            doc_adj = batch_data[1]
            claim_input_ids = batch_data[2]
            y = batch_data[3]

            outputs = self.model(
                doc_x=doc_x,
                doc_adj=doc_adj,
                claim_input_ids=claim_input_ids,
            )

            loss = self.loss_func(outputs, y)
            total_loss += loss.item()
            step += 1

            outputs = outputs.data.cpu().numpy()
            y = y.data.cpu().numpy()

            outputs = np.argmax(outputs, axis=1).flatten()
            y = y.flatten()

            all_outputs = np.concatenate((all_outputs, outputs), axis=None)
            all_y = np.concatenate((all_y, y), axis=None)

        total_loss /= step
        print("Loss: {:.5}".format(total_loss))

        precision, recall, f1, _ = precision_recall_fscore_support(all_outputs, all_y, average='micro')
        micro_f1 = f1
        print("F1 (micro): {:.3%}".format(f1))
        precision, recall, f1, _ = precision_recall_fscore_support(all_outputs, all_y, average='macro')
        print("Precision (macro): {:.3%}".format(precision))
        print("Recall (macro): {:.3%}".format(recall))
        print("F1 (macro): {:.3%}".format(f1))
        macro_f1 = f1

        return total_loss, micro_f1, macro_f1

def main():
    reader = Reader()
    train_data, dev_data, test_data = reader.load()

    process = Process()
    print("start process train data set...")
    train_encode, new_train_data = process.doc_encoder(train_data, 'train')

    print("start process dev data set...")
    dev_encode, new_dev_data = process.doc_encoder(dev_data, 'dev')

    print("start process test data set...")
    test_data, new_test_data = process.doc_encoder(test_data, 'test')

    process.set_(new_train_data + new_dev_data + new_test_data)

    batch_train_data = process.processing(new_train_data, train_encode, True)
    batch_dev_data = process.processing(new_dev_data, dev_encode, True)
    batch_test_data = process.processing(new_test_data, test_data, True)

    trainer = Trainer(len(batch_train_data))
    print_params(trainer.model)

    print("Training ... ")

    train_loss, dev_loss, test_loss = [], [], []
    best_micro_f1, best_macro_f1, best_epoch = 0, 0, 0
    for epoch in range(args.epoch):
        print("------------- Epoch: %s -------------" % str(epoch))
        train_ls = trainer.train(batch_train_data)
        train_loss.append(train_ls)

        dev_ls, _, _ = trainer.devel(batch_dev_data, "dev")
        dev_loss.append(dev_ls)

        test_ls, micro_f1, macro_f1 = trainer.devel(batch_test_data, "test")
        test_loss.append(test_ls)

        if best_micro_f1 < micro_f1:
            best_micro_f1 = micro_f1

        if best_macro_f1 < macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = epoch

            torch.save(trainer.model.state_dict(), args.save_model + '/model.pt')

        print("\n\n")

    print("Best Epoch: %s; Best macro_f1: %s"%(str(best_epoch), str(best_macro_f1)))

    with open(args.save_model + '/train_loss.csv', 'w', encoding='utf-8') as f_write:
        for i, ls in enumerate(train_loss):
            f_write.write(str(i+1))
            f_write.write('\t')
            f_write.write(str(ls))
            f_write.write('\n')
    f_write.close()

    with open(args.save_model + '/dev_loss.csv', 'w', encoding='utf-8') as f_write:
        for i, ls in enumerate(dev_loss):
            f_write.write(str(i+1))
            f_write.write('\t')
            f_write.write(str(ls))
            f_write.write('\n')
    f_write.close()

    with open(args.save_model + '/test_loss.csv', 'w', encoding='utf-8') as f_write:
        for i, ls in enumerate(test_loss):
            f_write.write(str(i+1))
            f_write.write('\t')
            f_write.write(str(ls))
            f_write.write('\n')
    f_write.close()

    return

if __name__ == '__main__':
    sys.exit(main())
