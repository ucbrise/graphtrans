import os
import pickle

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


def make_adj_list(N, edge_index_transposed):
    A = np.eye(N)
    for edge in edge_index_transposed:
        A[edge[0], edge[1]] = 1
    adj_list = A != 0
    return adj_list


def make_adj_list_wrapper(x):
    return make_adj_list(x["num_nodes"], x["edge_index"].T)


def compute_adjacency_list(data):
    out = []
    for x in tqdm(data, "adjacency list", leave=False):
        out.append(make_adj_list_wrapper(x))
    return out


def combine_results(data, adj_list):
    out_data = []
    for x, l in tqdm(zip(data, adj_list), "assembling adj_list result", total=len(data), leave=False):
        x["adj_list"] = l
        out_data.append(x)
    return out_data


def compute_adjacency_list_cached(data, key, root="/data/zhwu/tmp"):
    cachefile = f"{root}/OGB_ADJLIST_{key}.pickle"
    if os.path.exists(cachefile):
        with open(cachefile, "rb") as cachehandle:
            logger.debug("using cached result from '%s'" % cachefile)
            result = pickle.load(cachehandle)
        return combine_results(data, result)
    result = compute_adjacency_list(data)
    with open(cachefile, "wb") as cachehandle:
        logger.debug("saving result to cache '%s'" % cachefile)
        pickle.dump(result, cachehandle)
    logger.info("Got adjacency list data for key %s" % key)
    return combine_results(data, result)
