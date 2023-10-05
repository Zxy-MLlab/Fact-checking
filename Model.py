import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class MLP(nn.Module):
    def __init__(self, n_in, n_out, alpha, dropout):
        super(MLP, self).__init__()

        self.Linear = nn.Linear(n_in, n_out)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.Dropout = nn.Dropout(dropout)

        return

    def forward(self, x, activation=None):
        x = self.Dropout(x)
        x = self.Linear(x)
        if activation:
            x = x
        else:
            x = self.leakyrelu(x)

        return x


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_fea, out_fea, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.in_fea = in_fea
        self.out_fea = out_fea
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_fea, out_fea)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_fea, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.Dropout = nn.Dropout(p=dropout)

        return

    def forward(self, inp, adj):
        b, n, _ = inp.size()
        inp_h = torch.matmul(inp, self.W)
        inp_h_ = inp_h.clone()

        inp_in_chunks = inp_h.repeat(1, n, 1).view(b, n*n, -1)
        inp_alternating = inp_h.repeat(1, 1, n).view(b, n*n, -1)
        inp_combinations_matrix = torch.cat([inp_in_chunks, inp_alternating], dim=2)

        e = self.leakyrelu(torch.matmul(inp_combinations_matrix, self.a)).squeeze(-1).view(-1, n, n)

        zero_vec = -1e12 * torch.ones_like(e).to(inp.device)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(self.Dropout(attention), dim=2)
        out_h = torch.matmul(attention, inp_h)

        if self.concat:
            out_h = F.elu(out_h) + inp_h_

        return out_h


class MultiGAT(nn.Module):
    def __init__(self, n_fea, n_hid, dropout, alpha, n_heads):
        super(MultiGAT, self).__init__()

        self.attentions = [GraphAttentionLayer(n_fea, n_hid, dropout=dropout, alpha=alpha, concat=True) for _ in range(n_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.Dropout = nn.Dropout(p=dropout)

        return

    def forward(self, x, adj):
        c_x = x.clone()
        x = self.Dropout(x)

        x = torch.cat([att(x, adj) for att in self.attentions], dim=-1)

        x = x + c_x

        return x


class EvidenceExtract(nn.Module):
    def __init__(self, n_fea, n_hid, dropout, alpha, n_heads, n_layers, gamma):
        super(EvidenceExtract, self).__init__()

        self.graph_attns = [MultiGAT(n_fea, n_hid, dropout, alpha, n_heads) for _ in range(n_layers)]
        for i, graph_attn in enumerate(self.graph_attns):
            self.add_module('evidence_extract_gat_{}'.format(i), graph_attn)

        self.mlp = MLP(n_hid * n_heads, 1, alpha, dropout)

        self.Dropout = nn.Dropout(p=dropout)
        self.Sigmoid = nn.Sigmoid()

        return

    def forward(self, x, adj):
        b, n, _ = x.size()
        for i, graph_attn in enumerate(self.graph_attns):
            x = self.Dropout(x)
            x = graph_attn(x, adj)

        x_ = self.mlp(x).squeeze(2).argmax(dim=-1)
        evidence_fea = x[torch.arange(b), x_, :]

        return evidence_fea


class ClaimEncoder(nn.Module):
    def __init__(self, bert_model_path, dropout):
        super(ClaimEncoder, self).__init__()

        self.cl_encoder = BertModel.from_pretrained(bert_model_path)

        self.Dropout = nn.Dropout(p=dropout)

        return

    def forward(self, input_ids):
        n = input_ids.size()[1]
        attention_mask = input_ids.ne(0).to(input_ids.device)
        output = self.cl_encoder(input_ids=input_ids, attention_mask=attention_mask)
        output = self.Dropout(output[0])
        c_encode = output[:, 1:n - 1, :]

        return c_encode


class ECFeatureExtract(nn.Module):
    def __init__(self, n_fea, n_hid, out_channel, dropout):
        super(ECFeatureExtract, self).__init__()

        self.e_linear = nn.Linear(n_fea, n_hid)
        self.c_linear = nn.Linear(n_fea, n_hid)
        self.ec_linear = nn.Linear(2 * n_hid, 1)

        self.conv = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(3, out_channel, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.Dropout = nn.Dropout(p=dropout)

        return

    def forward(self, e_fea, c_fea):
        cs = self.CosineSimilarity(e_fea, c_fea).unsqueeze(3)
        es = self.ElementSimilarity(e_fea, c_fea).unsqueeze(3)
        cos = self.ContextSimilarity(e_fea, c_fea).unsqueeze(3)

        s = torch.cat([cs, es, cos], dim=-1).permute(0, 3, 1, 2)

        ec_fea = self.conv(s).permute(0, 2, 3, 1)

        return ec_fea

    def CosineSimilarity(self, x, y):
        n = y.size()[1]
        cs = []
        for y_ in range(n):
            cs_ = F.cosine_similarity(x, y[:, y_, :])
            cs.append(cs_)

        cs = torch.stack(cs, dim=1).unsqueeze(2)

        return cs

    def ElementSimilarity(self, x, y):

        es = torch.matmul(y, x.unsqueeze(1).transpose(2, 1))

        return es

    def ContextSimilarity(self, x, y):
        n = y.size()[1]
        x = x.unsqueeze(1)

        x = self.Dropout(self.e_linear(x)).repeat(1, n, 1)
        y = self.Dropout(self.c_linear(y))

        xy = torch.cat([x, y], dim=2)
        cos = self.ec_linear(xy)

        return cos


class FC(nn.Module):
    def __init__(self, arg):
        super(FC, self).__init__()

        self.arg = arg

        self.EE = EvidenceExtract(
            n_fea=arg.n_fea,
            n_hid=arg.n_hid,
            dropout=arg.dropout,
            alpha=arg.alpha,
            n_heads=arg.n_heads,
            n_layers=arg.n_layers,
            gamma=arg.gamma,
        )

        self.CE = ClaimEncoder(
             bert_model_path=arg.bert_model_path,
             dropout=arg.dropout,
        )

        self.EC = ECFeatureExtract(
            n_fea=arg.n_fea,
            n_hid=arg.n_hid,
            out_channel=arg.out_channels,
            dropout=arg.dropout,
        )

        self.check_linear = MLP(
            n_in=arg.out_channels,
            n_out=arg.n_class,
            alpha=arg.alpha,
            dropout=arg.dropout,
        )

        self.sigmoid = nn.Sigmoid()

        return

    def com_rewards(self, logits, y):

        reward = torch.log(torch.sum(logits * y, dim=-1))

        return reward

    def forward(self, doc_x, doc_adj, claim_input_ids):

        evidence_fea = self.EE(doc_x, doc_adj)
        claim_encode = self.CE(claim_input_ids)

        ec_fea = self.EC(evidence_fea, claim_encode).squeeze(2)

        ec_fea, _ = torch.max(ec_fea, dim=1)

        preds = self.check_linear(ec_fea, self.sigmoid)

        return preds