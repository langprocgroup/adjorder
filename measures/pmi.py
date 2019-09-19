#!/usr/bin/env python3
import sys
from math import log
import operator
from collections import Counter

def read_dict(filename, **kwds):
    with open(filename, **kwds) as infile:
        d = {}
        for line in infile:
            k, v = line.strip().rsplit(maxsplit=1)
            d[k] = int(v)
    return d

def pmi(joint, m1, m2):
    Z_joint = sum(joint.values())
    Z_m1 = sum(m1.values())
    Z_m2 = sum(m2.values())
    A = log(Z_m1) + log(Z_m2) - log(Z_joint)
    def gen():
        for (one, two), c in joint.items():
            yield (one, two), (
                -log(m1[one]/Z_m1)
                + (-log(m2[two]/Z_m2))
                - (-log(joint[one, two]/Z_joint))
            )
    return dict(gen())

def marginalize(joint, k):
    result = Counter()
    for j, c in joint.items():
        key = j[k]
        result[key] += c
    return result

def main(joint, marginal1=None, marginal2=None):
    joint_counts = read_dict(joint)
    joint_counts = {tuple(k.split()) : v for k, v in joint_counts.items()}
    if marginal1:
        marginal1_counts = read_dict(marginal1)
    else:
        marginal1_counts = marginalize(joint_counts, 0)
    if marginal2:
        marginal2_counts = read_dict(marginal2)
    else:
        marginal2_counts = marginalize(joint_counts, 1)
    pmi_dict = pmi(joint_counts, marginal1_counts, marginal2_counts)
    for (one, two), v in pmi_dict.items():
        print("\t".join(map(str, [one, two, v])))

if __name__ == '__main__':
    main(*sys.argv[1:])
