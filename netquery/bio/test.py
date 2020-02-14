import sys
sys.path.append('..')
sys.path.append('../..')

from argparse import ArgumentParser

from netquery.bio.data_utils import load_graph
from netquery.model import QueryEncoderDecoder
from netquery.hashed_model import HashedQueryEncoderDecoder
from netquery.utils import *
from netquery.train_helpers import run_eval
from netquery.data_utils import *
from netquery.graph import *
import torch


parser = ArgumentParser()
parser.add_argument("--embed_dim", type=int, default=128)
parser.add_argument("--data_dir", type=str, default="../../bio_data")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--depth", type=int, default=0)
parser.add_argument("--beta", type=int, default=20)
parser.add_argument("--log_dir", type=str, default="./log")
parser.add_argument("--model_dir", type=str, default="./model")
parser.add_argument("--decoder", type=str, default="bilinear")
parser.add_argument("--inter_decoder", type=str, default="mean")

args = parser.parse_args()

print("Loading graph data...")
graph, feature_modules, node_maps = load_graph(args.data_dir, args.embed_dim)
out_dims = {mode:args.embed_dim for mode in graph.relations}
print("Loading edge data..")
test_queries = load_test_queries_by_formula(args.data_dir + "/test_edges.pkl")
print("Loading query data..")
for i in range(2,4):
    i_test_queries = load_test_queries_by_formula(args.data_dir + "/test_queries_{:d}.pkl".format(i))
    test_queries["one_neg"].update(i_test_queries["one_neg"])
    test_queries["full_neg"].update(i_test_queries["full_neg"])

enc = get_encoder(args.depth, graph, out_dims, feature_modules, False, beta=args.beta)
dec = get_metapath_decoder(graph, out_dims, args.decoder, beta=args.beta)
inter_dec = get_intersection_decoder(graph, out_dims, args.inter_decoder, beta=args.beta)

print('loading model...')
model_dir = "./model" + "/{data:s}-{beta:d}-{depth:d}-{embed_dim:d}-{lr:f}-{decoder:s}-{inter_decoder:s}-edge_conv".format(
                data=args.data_dir.strip().split("/")[-1],
                beta=args.beta,
                depth=args.depth,
                embed_dim=args.embed_dim,
                lr=args.lr,
                decoder=args.decoder,
                inter_decoder=args.inter_decoder)

c = torch.load(model_dir)

enc_dec = QueryEncoderDecoder(graph, enc, dec, inter_dec)
enc_dec.load_state_dict(c)
hashed_enc_dec = HashedQueryEncoderDecoder(graph, enc, dec, inter_dec)
hashed_enc_dec.load_state_dict(c)

logger = setup_logging("./log/result.log")
logger.info("testing original model...")
run_eval(enc_dec, test_queries, 0, logger)
logger.info("testing hashed model...")
run_eval(hashed_enc_dec, test_queries, 0, logger)