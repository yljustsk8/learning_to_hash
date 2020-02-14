import torch
import torch.nn as nn
import random
from netquery.graph import _reverse_relation

EPS = 10e-6

"""
End-to-end autoencoder models for representation learning on
heteregenous graphs/networks
"""

class QueryEncoderDecoder(nn.Module):
    """
    Encoder decoder model that reasons about edges, metapaths and intersections
    """

    def __init__(self, graph, enc, path_dec, inter_dec):
        super(QueryEncoderDecoder, self).__init__()
        self.enc = enc
        self.path_dec = path_dec
        self.inter_dec = inter_dec
        self.graph = graph
        self.cos = nn.CosineSimilarity(dim=0)

    def forward(self, formula, queries, source_nodes):
        if formula.query_type == "1-chain" or formula.query_type == "2-chain" or formula.query_type == "3-chain":
            # a chain is simply a call to the path decoder
            return self.path_dec.forward(
                    self.enc.forward(source_nodes, formula.target_mode), 
                    self.enc.forward([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0]),
                    formula.rels)
        elif formula.query_type == "2-inter" or formula.query_type == "3-inter" or formula.query_type == "3-inter_chain":
            target_embeds = self.enc(source_nodes, formula.target_mode)

            embeds1 = self.enc([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
            embeds1 = self.path_dec.project(embeds1, _reverse_relation(formula.rels[0]))

            embeds2 = self.enc([query.anchor_nodes[1] for query in queries], formula.anchor_modes[1])
            if len(formula.rels[1]) == 2:
                for i_rel in formula.rels[1][::-1]:
                    embeds2 = self.path_dec.project(embeds2, _reverse_relation(i_rel))
            else:
                    embeds2 = self.path_dec.project(embeds2, _reverse_relation(formula.rels[1]))

            if formula.query_type == "3-inter":
                embeds3 = self.enc([query.anchor_nodes[2] for query in queries], formula.anchor_modes[2])
                embeds3 = self.path_dec.project(embeds3, _reverse_relation(formula.rels[2]))
                query_intersection = self.inter_dec(embeds1, embeds2, formula.target_mode, embeds3)
            else:
                query_intersection = self.inter_dec(embeds1, embeds2, formula.target_mode)

            scores = self.cos(target_embeds, query_intersection)
            return scores
        elif formula.query_type == "3-chain_inter":
            target_embeds = self.enc(source_nodes, formula.target_mode)

            embeds1 = self.enc([query.anchor_nodes[0] for query in queries], formula.anchor_modes[0])
            embeds1 = self.path_dec.project(embeds1, _reverse_relation(formula.rels[1][0]))
            embeds2 = self.enc([query.anchor_nodes[1] for query in queries], formula.anchor_modes[1])
            embeds2 = self.path_dec.project(embeds2, _reverse_relation(formula.rels[1][1]))
            query_intersection = self.inter_dec(embeds1, embeds2, formula.rels[0][-1])
            query_intersection = self.path_dec.project(query_intersection, _reverse_relation(formula.rels[0]))

            scores = self.cos(target_embeds, query_intersection)
            return scores

    def margin_loss(self, formula, queries, hard_negatives=False, margin=1):
        if not "inter" in formula.query_type and hard_negatives:
            raise Exception("Hard negative examples can only be used with intersection queries")
        elif hard_negatives:
            neg_nodes = [random.choice(query.hard_neg_samples) for query in queries]
        elif formula.query_type == "1-chain":
            neg_nodes = [random.choice(self.graph.full_lists[formula.target_mode]) for _ in queries]
        else:
            neg_nodes = [random.choice(query.neg_samples) for query in queries]

        affs = self.forward(formula, queries, [query.target_node for query in queries])
        neg_affs = self.forward(formula, queries, neg_nodes)
        loss = margin - (affs - neg_affs)
        loss = torch.clamp(loss, min=0)
        loss = loss.mean()
        return loss

