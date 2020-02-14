import numpy as np
import scipy
import scipy.stats as stats
import torch
from sklearn.metrics import roc_auc_score
from netquery.decoders import BilinearMetapathDecoder, TransEMetapathDecoder, SetIntersection
from netquery.encoders import DirectEncoder
from netquery.aggregators import MeanAggregator
import pickle as pickle
import logging
import random

"""
Misc utility functions..
"""

def cudify(feature_modules, node_maps=None):
   if node_maps is None:
       features = lambda nodes, mode : feature_modules[mode](
            torch.autograd.Variable(torch.LongTensor(nodes)+1).cuda())
   else:
       features = lambda nodes, mode : feature_modules[mode](
            torch.autograd.Variable(torch.LongTensor([node_maps[mode][n] for n in nodes])+1).cuda())
   return features

def _get_perc_scores(scores, lengths):
    perc_scores = []
    cum_sum = 0
    neg_scores = scores[len(lengths):]
    for i, length in enumerate(lengths):
        perc_scores.append(stats.percentileofscore(neg_scores[cum_sum:cum_sum+length], scores[i]))
        cum_sum += length
    return perc_scores

def eval_auc_queries(test_queries, enc_dec, batch_size=1000, hard_negatives=False, seed=0):
    predictions = []
    labels = []
    formula_aucs = {}
    random.seed(seed)
    for formula in test_queries:
        formula_labels = []
        formula_predictions = []
        formula_queries = test_queries[formula]
        offset = 0
        while offset < len(formula_queries):
            max_index = min(offset+batch_size, len(formula_queries))
            batch_queries = formula_queries[offset:max_index]
            if hard_negatives:
                lengths = [1 for j in range(offset, max_index)]
                negatives = [random.choice(formula_queries[j].hard_neg_samples) for j in range(offset, max_index)]
            else:
                lengths = [1 for j in range(offset, max_index)]
                negatives = [random.choice(formula_queries[j].neg_samples) for j  in range(offset, max_index)]
            offset += batch_size

            formula_labels.extend([1 for _ in range(len(lengths))])
            formula_labels.extend([0 for _ in range(len(negatives))])
            batch_scores = enc_dec.forward(formula, 
                    batch_queries+[b for i, b in enumerate(batch_queries) for _ in range(lengths[i])], 
                    [q.target_node for q in batch_queries] + negatives)
            batch_scores = batch_scores.data.tolist()
            formula_predictions.extend(batch_scores)
        formula_aucs[formula] = roc_auc_score(formula_labels, np.nan_to_num(formula_predictions))
        labels.extend(formula_labels)
        predictions.extend(formula_predictions)
    overall_auc = roc_auc_score(labels, np.nan_to_num(predictions))
    return overall_auc, formula_aucs

    
def eval_perc_queries(test_queries, enc_dec, batch_size=1000, hard_negatives=False):
    perc_scores = []
    for formula in test_queries:
        formula_queries = test_queries[formula]
        offset = 0
        while offset < len(formula_queries):
            max_index = min(offset+batch_size, len(formula_queries))
            batch_queries = formula_queries[offset:max_index]
            if hard_negatives:
                lengths = [len(formula_queries[j].hard_neg_samples) for j in range(offset, max_index)]
                negatives = [n for j in range(offset, max_index) for n in formula_queries[j].hard_neg_samples]
            else:
                lengths = [len(formula_queries[j].neg_samples) for j in range(offset, max_index)]
                negatives = [n for j in range(offset, max_index) for n in formula_queries[j].neg_samples]
            offset += batch_size

            batch_scores = enc_dec.forward(formula, 
                    batch_queries+[b for i, b in enumerate(batch_queries) for _ in range(lengths[i])], 
                    [q.target_node for q in batch_queries] + negatives)
            batch_scores = batch_scores.data.tolist()
            perc_scores.extend(_get_perc_scores(batch_scores, lengths))
    return np.mean(perc_scores)

def get_encoder(depth, graph, out_dims, feature_modules, cuda, beta=0.1): 

    if depth == 0:
         enc = DirectEncoder(graph.features, feature_modules, beta)

    return enc

def get_metapath_decoder(graph, out_dims, decoder, beta=0.1):
    if decoder == "bilinear":
        dec = BilinearMetapathDecoder(graph.relations, out_dims, beta)
    elif decoder == "transe":
        dec = TransEMetapathDecoder(graph.relations, out_dims, beta)
    else:
        raise Exception("Metapath decoder not recognized.")
    return dec

def get_intersection_decoder(graph, out_dims, decoder, beta=0.1):
    if decoder == "mean":
        dec = SetIntersection(out_dims, out_dims, agg_func=torch.mean, beta=beta)
    else:
        raise Exception("Intersection decoder not recognized.")
    return dec

def setup_logging(log_file, console=True):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        filename=log_file,
                        filemode='w')
    if console:
        console = logging.StreamHandler()
        # optional, set the logging level
        console.setLevel(logging.INFO)
        # set a format which is the same for console use
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    return logging
