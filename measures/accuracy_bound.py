""" Calculate how well we could predict adjectives on a given test set using any predictor based on scores assigned to adjective--noun pairs"""
import sys
import argparse
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.optim

DEFAULT_NUM_SAMPLES = 1000

def load_triples(filename):
    d = pd.read_csv(filename, sep="\t", header=None)
    d.columns = ['count', 'a1', 'a2', 'n']
    return d

def attested_pairs(d):
    """ Given a dataframe of triples and their counts, return a dataframe of all attested a-n pairs """
    result = pd.DataFrame(set(zip(d['a1'], d['n'])).union(zip(d['a2'], d['n'])))
    result.columns = ['a', 'n']
    return result

def random_accuracy(triples, pairs, num_samples=DEFAULT_NUM_SAMPLES):
    # Now for the random baseline, we assign random scores to pairs,
    # and calculate differences based on the superscript orders.
    random_scores = pd.DataFrame(np.random.random((pairs.shape[0], num_samples)))
    pairs = pd.concat([pairs, random_scores], axis=1)
    first_score = triples.merge(pairs, left_on=['a^1', 'n'], right_on=['a', 'n'])[range(num_samples)]
    second_score = triples.merge(pairs, left_on=['a^2', 'n'], right_on=['a', 'n'])[range(num_samples)]
    random_diff = first_score - second_score
    random_predictions = triples['match'][:, None] == (random_diff < 0)
    random_accuracy = (random_predictions * triples['count'][:, None]).sum(axis=0) / triples['count'].sum()
    random_accuracy = np.maximum(random_accuracy, 1 - random_accuracy)
    # TODO what are the upper and lower CIs on this? Can we calculate significance?    
    return random_accuracy.mean()

def triple_upper_bound(triples, pairs):
    # Assign to each triple the score that maximizes accuracy
    d = triples[['a^1', 'a^2', 'n', 'count', 'match']].copy()
    d['countmatch'] = d['count'] * d['match']
    aggregated = d[['a^1', 'a^2', 'n', 'countmatch']].groupby(['a^1', 'a^2', 'n']).sum().reset_index()
    aggregated['count'] = d[['a^1', 'a^2', 'n', 'count']].groupby(['a^1', 'a^2', 'n']).sum().reset_index()['count']
    aggregated['prop_matching'] = aggregated['countmatch'] / aggregated['count']
    prop_accurate = np.maximum(aggregated['prop_matching'], 1 - aggregated['prop_matching'])
    return prop_accurate.mean()

def pair_upper_bound(triples, pairs):
    # The score for each adjective-noun is just the proportion of times the adjective shows up first.
    # Conjecture: This is the MLE.
    count_first = Counter()
    total_count = Counter()
    for a, n in zip(pairs['a'], pairs['n']):
        count_first[a, n] += triples[(triples['a1'] == a) & (triples['n'] == n)]['count'].sum()
        total_count[a, n] = count_first[a, n] + triples[(triples['a2'] == a) & (triples['n'] == n)]['count'].sum()
    pair_props = pd.DataFrame([(a,n,c/total_count[a,n]) for (a, n), c in count_first.items()])
    pair_props.columns = ['a', 'n', 'prop_first']
    first_score = triples.merge(pair_props, left_on=['a^1', 'n'], right_on=['a', 'n'])['prop_first']
    second_score = triples.merge(pair_props, left_on=['a^2', 'n'], right_on=['a', 'n'])['prop_first']
    diff = first_score - second_score
    predictions = triples['match'] == (diff > 0)
    predictions[diff == 0.0] = .5
    return (predictions * triples['count']).sum(axis=0) / triples['count'].sum()

def torch_pair_upper_bound(triples, pairs, num_epochs):    
    # Represent each pair as a 1-hot X of dimension K.
    # Then V = W*X is the weight for adjective X, where M is a vector of dimension K, and * is elementwise product.
    # Now for each triple a^1, a^2, n, the weight difference is W*X^1 - W*X^2,
    # where X^1 is the embedding of (a^1, n).
    # Obviously this gives W*(X^1 - X^2).
    # Now we want to fit logit(W*(X^1 - X_2)) = p(a^1 before a^2).
    # All we have to find is the vector W.
    TODO
    weights = torch.rand(pairs.shape[0], required_grad=True) # start with random weights
    opt = torch.optim.Adam([weights])
    for _ in range(num_epochs):
        opt.zero_grad()
        logits = weights*diff
        logprobs = torch.log_softmax(logits, dim=0) # choose dim carefully
        loss = -(props * logprobs).sum()
        loss.backward()
        opt.step()
        

def main(filename, num_samples=DEFAULT_NUM_SAMPLES):
    triples = load_triples(filename)
    pairs = attested_pairs(triples)
    
    # Now we will need to format the data into an (X,y) for the logistic regression.
    # we have triples a_1, a_2, n where the subscript indicates the attested order.
    # X will be the direct sum of one-hot embeddings of a^1, a^2, and n,
    # where superscript a^i indicates alphabetical order.
    # 'match' means (a^1, a^2) == (a_1, a_2)
    alphabetical = map(sorted, zip(triples['a1'], triples['a2']))
    triples['a^1'], triples['a^2'] = zip(*alphabetical)
    # The dependent variable y is whether the superscript indices match the subscript indices.    
    triples['match'] = (triples['a1'] == triples['a^1']) & (triples['a2'] == triples['a^2'])

    print("Pair random accuracy, mean in %s samples:" % str(num_samples), random_accuracy(triples, pairs, num_samples))
    print("Triple upper bound:", triple_upper_bound(triples, pairs))
    print("Pair upper bound:", pair_upper_bound(triples, pairs))

    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate accuracy upper bound')
    parser.add_argument('inputfile', type=str, help='AAN triples, a tsv file with columns count, a1, a2, n, and no header')
    parser.add_argument('num_samples', type=int, default=DEFAULT_NUM_SAMPLES, help='Number of random samples for the random baseline')
    args = parser.parse_args()

    sys.exit(main(args.inputfile))


