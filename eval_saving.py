# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import random
random.seed(1234)
import os
import pickle
from utils.metrics import compute_metrics



def computeAverageMetrics(imfeats, recipefeats, ids, k, t=1, forceorder=False):
    """Computes retrieval metrics for two sets of features

    Parameters
    ----------
    imfeats : np.ndarray [n x d]
        The image features..
    recipefeats : np.ndarray [n x d]
        The recipe features.
    k : int
        Ranking size.
    t : int
        Number of evaluations to run (function returns the average).
    forceorder : bool
        Whether to force a particular order instead of picking random samples

    Returns
    -------
    dict
        Dictionary with metric values for all t runs.

    """

    glob_metrics = {}
    i = 0

    for tt in range(1):

        if forceorder:
            # pick the same samples in the same order for evaluation
            # forceorder is only True when the function is used during training
            sub_ids = np.array(range(i, i + k))
            i += k
        else:
            sub_ids = random.sample(range(0, len(imfeats)), k)
        ids_sub = ids[sub_ids]
        imfeats_sub = imfeats[sub_ids, :]
        recipefeats_sub = recipefeats[sub_ids, :]
        print(sub_ids[:5], sub_ids[-5:])

        metrics, intermediate_dict = compute_metrics(imfeats_sub, recipefeats_sub,
                                  recall_klist=(1, 5, 10), return_raw=True)

        for metric_name, metric_value in metrics.items():
            if metric_name not in glob_metrics:
                glob_metrics[metric_name] = []
            glob_metrics[metric_name].append(metric_value)

        id2results = {}
        for idx, one_id in enumerate(ids_sub):
            result = {}
            result['dist'] = intermediate_dict['dists'][idx]
            result['rank'] = intermediate_dict['medr'][idx]
            result['gt'] = idx
            id2results[one_id] = result

    return glob_metrics, id2results


def eval(args):

    # Load embeddings

    print('Loading embeddings', args.embeddings_file)
    with open(args.embeddings_file, 'rb') as f:
        imfeats = pickle.load(f)
        recipefeats = pickle.load(f)
        ids = pickle.load(f)
        ids = np.array(ids)

    # sort by name so that we always pick the same samples
    idxs = np.argsort(ids)
    ids = ids[idxs]
    recipefeats = recipefeats[idxs]
    imfeats = imfeats[idxs]

    if args.retrieval_mode == 'image2recipe':
        glob_metrics, id2results = computeAverageMetrics(imfeats, recipefeats, ids, args.medr_N, args.ntimes)
    else:
        glob_metrics = computeAverageMetrics(recipefeats, imfeats, ids, args.medr_N, args.ntimes)

    for k, v in glob_metrics.items():
        print (k + ':', np.mean(v))

    os.makedirs('./id2results/', exist_ok=True)
    with open('./id2results/{}_{}.pkl'.format(args.exp_choice, args.model_name), 'wb') as f:
        pickle.dump(id2results, f)

