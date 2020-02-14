import torch
import torch.nn as nn
from torch.nn import init

import numpy as np
import torch.nn.functional as F
import sys

'''
WARNING: This is the test mode so the path decoder will return the cosine of two hashed embeddings!
And all the query will be turned into a hashed style by default!
'''

"""
A set of decoder modules.
Each decoder takes pairs of embeddings and predicts relationship scores given these embeddings.
"""

""" 
*Metapath decoders*
For all metapath encoders, the forward method returns a compositonal relationships score, 
i.e. the likelihood of compositonional relationship or metapath, between a pair of nodes.
"""

class BilinearMetapathDecoder(nn.Module):
    """
    Each edge type is represented by a matrix, and
    compositional relationships are a product matrices.
    """

    def __init__(self, relations, dims, beta=1):
        super(BilinearMetapathDecoder, self).__init__()
        self.relations = relations
        self.mats = {}
        self.beta = beta
        self.sigmoid = torch.nn.Sigmoid()
        self.cos = nn.CosineSimilarity(dim=0)
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.mats[rel] = nn.Parameter(torch.FloatTensor(dims[rel[0]], dims[rel[2]]))
                init.xavier_uniform_(self.mats[rel])
                self.register_parameter("_".join(rel), self.mats[rel])

    def tohash(self, x):
        return torch.sign(torch.sign(x).add(0.1))


    def forward(self, embeds1, embeds2, rels, sign=False):

        act = embeds1.t()
        for i_rel in rels:
            act = act.mm(self.mats[i_rel])
            act = torch.tanh(self.beta * act)
        if sign:
            act = self.tohash(act)
        act = self.cos(act.t(), embeds2)

        return act

    def project(self, embeds, rel, sign=False):
        a = self.mats[rel].mm(embeds)
        a = torch.tanh(self.beta * a)
        if sign:
            return self.tohash(a)
        return a


class TransEMetapathDecoder(nn.Module):
    """
    Decoder where the relationship score is given by translation of
    the embeddings, each relation type is represented by a vector, and
    compositional relationships are addition of these vectors
    """

    def __init__(self, relations, dims, beta):
        super(TransEMetapathDecoder, self).__init__()
        self.relations = relations
        self.beta = beta
        self.vecs = {}
        for r1 in relations:
            for r2 in relations[r1]:
                rel = (r1, r2[1], r2[0])
                self.vecs[rel] = nn.Parameter(torch.FloatTensor(dims[rel[0]]))
                init.uniform_(self.vecs[rel], a=-6.0/np.sqrt(dims[rel[0]]), b=6.0/np.sqrt(dims[rel[0]]))
                self.register_parameter("_".join(rel), self.vecs[rel])
        self.cos = nn.CosineSimilarity(dim=0)

    def tohash(self, x):
        return torch.sign(torch.sign(x).add(0.1))

    def forward(self, embeds1, embeds2, rels, sign=False):
        trans_embed = embeds1
        for i_rel in rels:
            trans_embed += self.vecs[i_rel].unsqueeze(1).expand(self.vecs[i_rel].size(0), embeds1.size(1))
            trans_embed = torch.tanh(self.beta * trans_embed)
        if sign:
            trans_embed = self.tohash(trans_embed)
        trans_dist = self.cos(embeds2, trans_embed)
        return trans_dist

    def project(self, embeds, rel, sign=False):
        trans_dist = embeds + self.vecs[rel].unsqueeze(1).expand(self.vecs[rel].size(0), embeds.size(1))
        if sign:
            trans_dist = self.tohash(trans_dist)
        return trans_dist


"""
Set intersection operator
"""

class SetIntersection(nn.Module):
    """
    Decoder that computes the implicit intersection between two state vectors
    Applies an MLP and takes elementwise mins.
    """
    def __init__(self, mode_dims, expand_dims, agg_func=torch.min, beta=1):
        super(SetIntersection, self).__init__()
        self.pre_mats = {}
        self.post_mats = {}
        self.beta = beta
        self.agg_func = agg_func
        for mode in mode_dims:
            self.pre_mats[mode] = nn.Parameter(torch.FloatTensor(expand_dims[mode], mode_dims[mode]))
            init.xavier_uniform_(self.pre_mats[mode])
            self.register_parameter(mode+"_premat", self.pre_mats[mode])
            self.post_mats[mode] = nn.Parameter(torch.FloatTensor(mode_dims[mode], expand_dims[mode]))
            init.xavier_uniform_(self.post_mats[mode])
            self.register_parameter(mode+"_postmat", self.post_mats[mode])

    def tohash(self, x):
        return torch.sign(torch.sign(x).add(0.1))

    def forward(self, embeds1, embeds2, mode, embeds3=[], sign=False):

        temp1 = F.relu(self.pre_mats[mode].mm(embeds1))
        temp2 = F.relu(self.pre_mats[mode].mm(embeds2))
        if len(embeds3) > 0:
            temp3 = F.relu(self.pre_mats[mode].mm(embeds3))
            combined = torch.stack([temp1, temp2, temp3])
        else:
            combined = torch.stack([temp1, temp2])
        combined = self.agg_func(combined,dim=0)
        if type(combined) == tuple:
            combined = combined[0]
        combined = self.post_mats[mode].mm(combined)
        ret = torch.tanh(self.beta * combined)
        if sign:
            ret = self.tohash(ret)
        return ret
