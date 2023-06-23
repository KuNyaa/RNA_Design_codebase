import argparse
from collections import defaultdict

import copy
import numpy as np

from utils.structure import extract_pairs

MIN_LEN = 1

pairs = {'au', 'gc', 'gu'}
nucs = 'ACGU'
nuc2idx = {nuc:idx for idx, nuc in enumerate(nucs)}
pairs_index = [(0, 3), (3, 0), (1, 2), (2, 1), (2, 3), (3, 2)]  # A C G U
nuc_pairs = [nucs[i]+nucs[j] for i, j in pairs_index]
pair2score = {"CG": -3, "GC": -3, "AU": -2, "UA":-2, "GU": -1, "UG":-1}
unpair_index = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 3), (2, 0), (2, 2), (3, 1), (3, 3)]
unpair_score = 1.
mispair_score = 1.
score_stack = -2.
kT = 1.

sharpturn = 3
# score_stack = 0.

def match(x, y):
    return (x+y).lower() in pairs or (y+x).lower() in pairs


def set_sharpturn(sharpturn_val):
    global sharpturn
    sharpturn = sharpturn_val
    # print(f"sharpturn is: {sharpturn}") 
    

def inside_forward(s):
    assert len(s) > 1, "the length of rna should be at least 2!"
    n = len(s)
    counts = defaultdict(lambda: defaultdict(int)) # number of structures
    for k in range(n):
        counts[k][k]=1
        counts[k][k-1] = 1
    for j in range(1, n):
        for i in range(0, j):
            counts[i][j] += counts[i][j-1] # x.
            if match(s[i], s[j]) and j-i>sharpturn: # (x); x(x) 
                counts_right = counts[i+1][j-1] 
                for t in range(0, i):
                    counts_left = counts[t][i-1] 
                    counts[t][j] += counts_left*counts_right
                counts[i][j] += counts_right
    return counts


def inside_forward_pt(s):
    assert len(s) > 1, "the length of rna should be at least 2!"
    n = len(s)
    counts = defaultdict(lambda: defaultdict(lambda: -np.inf)) # number of structures
    for k in range(n):
        counts[k][k]=-unpair_score/kT
        counts[k][k-1] = 0
    for j in range(1, n):
        for i in range(0, j):
            counts[i][j] = np.logaddexp(counts[i][j], counts[i][j-1]+(-unpair_score/kT))  # x.
            if match(s[i], s[j]) and j-i>sharpturn: # (x); x(x) 
                counts_right = counts[i+1][j-1]
                pair_ij = s[i]+s[j]
                score_ij = -pair2score[pair_ij]/kT
                for t in range(0, i):
                    counts_left = counts[t][i-1] 
                    counts[t][j] =np.logaddexp(counts[t][j], counts_left+counts_right+score_ij) #x(x)
                counts[i][j] = np.logaddexp(counts[i][j], counts_right+score_ij) #(x)
    return counts


def outside_forward_pt(s, inside):
    assert len(s) > 1, "the length of rna should be at least 2!"
    n = len(s)
    outside = defaultdict(lambda: defaultdict(lambda: -np.inf)) # number of structures
    outside[0][n-1] = 0.
    marginal = defaultdict(lambda: defaultdict(lambda: -np.inf))
    for j in range(n-1, -1, -1):
        for i in range(j-1, -1, -1):
            # counts[i][j] = np.logaddexp(counts[i][j], counts[i][j-1]+(-unpair_score/kT))  # x.
            outside[i][j-1] = np.logaddexp(outside[i][j-1], outside[i][j]+(-unpair_score/kT))
            if match(s[i], s[j]) and j-i>sharpturn: # (x); x(x) 
                pair_ij = s[i]+s[j]
                score_ij = -pair2score[pair_ij]/kT
                for t in range(0, i+1):
                    outside[t][i-1] = np.logaddexp(outside[t][i-1], outside[t][j]+inside[i+1][j-1]+score_ij)
                    outside[i+1][j-1] = np.logaddexp(outside[i+1][j-1], outside[t][j]+inside[t][i-1]+score_ij)
                    marginal[i][j] = np.logaddexp(marginal[i][j], outside[t][j]+inside[t][i-1]+inside[i+1][j-1]+score_ij)
                # outside[i+1][j-1] = np.logaddexp(outside[i+1][j-1], outside[i][j]+score_ij) # (x)
    return outside, marginal


def inside_forward_pt_stack(s):
    assert len(s) > 1, "the length of rna should be at least 2!"
    # print(f"sharpturn: {sharpturn}")
    n = len(s)
    p = defaultdict(lambda: defaultdict(lambda: -np.inf)) # partition 
    a = defaultdict(lambda: defaultdict(lambda: -np.inf)) # left and right don't pair
    b = defaultdict(lambda: defaultdict(lambda: -np.inf)) # left and right pair
    for k in range(n):
        p[k][k]=-unpair_score/kT
        p[k][k-1] = 0
        a[k][k]=-unpair_score/kT
        a[k][k-1] = 0        
    for j in range(1, n):
        for i in range(0, j):
            a[i][j] = np.logaddexp(a[i][j], p[i][j-1]+(-unpair_score/kT))  # a <- s.
            p[i][j] = np.logaddexp(p[i][j], p[i][j-1]+(-unpair_score/kT))  # p <- s.
            if match(s[i], s[j]) and j-i>sharpturn:
                a_right = a[i+1][j-1]
                b_right = b[i+1][j-1]
                pair_ij = s[i]+s[j]
                score_ij = -pair2score[pair_ij]/kT
                for t in range(0, i):
                    p_left = p[t][i-1] 
                    a[t][j] = np.logaddexp(a[t][j], p_left+a_right+score_ij) # a <- s(a)
                    p[t][j] = np.logaddexp(p[t][j], p_left+a_right+score_ij) # p <- s(a)
                    a[t][j] = np.logaddexp(a[t][j], p_left+b_right+score_ij+(-score_stack/kT)) # a <- s(b)
                    p[t][j] = np.logaddexp(p[t][j], p_left+b_right+score_ij+(-score_stack/kT)) # p <- s(b)
                b[i][j] = np.logaddexp(b[i][j], a_right+score_ij) # b <- (a)
                p[i][j] = np.logaddexp(p[i][j], a_right+score_ij) # p <- (a)
                b[i][j] = np.logaddexp(b[i][j], b_right+score_ij+(-score_stack/kT)) # b <- (b)
                p[i][j] = np.logaddexp(p[i][j], b_right+score_ij+(-score_stack/kT)) # p <- (b)
    return p, a, b


def outside_forward_pt_stack(s, inside, a, b):
    assert len(s) > 1, "the length of rna should be at least 2!"
    # print(f"sharpturn: {sharpturn}")
    n = len(s)
    outside = defaultdict(lambda: defaultdict(lambda: -np.inf)) # partition 
    out_a = defaultdict(lambda: defaultdict(lambda: -np.inf)) # left and right don't pair
    out_b = defaultdict(lambda: defaultdict(lambda: -np.inf)) # left and right pair
    marginal = defaultdict(lambda: defaultdict(lambda: -np.inf))
    outside[0][n-1] = 0.        
    for j in range(n-1, 0, -1):
        for i in range(j-1, -1, -1):
            outside[i][j-1] = np.logaddexp(outside[i][j-1], outside[i][j]+(-unpair_score/kT))
            if match(s[i], s[j]) and j-i>sharpturn:
                pair_ij = s[i]+s[j]
                score_ij = -pair2score[pair_ij]/kT
                for t in range(0, i+1):
                    # a[t][j] = np.logaddexp(a[t][j], p_left+a_right+score_ij) # a <- s(a)
                    # p[t][j] = np.logaddexp(p[t][j], p_left+a_right+score_ij) # p <- s(a)
                    outside[t][i-1] = np.logaddexp(outside[t][i-1], outside[t][j]+a[i+1][j-1]+score_ij)
                    # a[t][j] = np.logaddexp(a[t][j], p_left+b_right+score_ij+(-score_stack/kT)) # a <- s(b)
                    # p[t][j] = np.logaddexp(p[t][j], p_left+b_right+score_ij+(-score_stack/kT)) # p <- s(b)
                    outside[t][i-1] = np.logaddexp(outside[t][i-1], outside[t][j]+b[i+1][j-1]+score_ij+(-score_stack/kT))
                    marginal[i][j] = np.logaddexp(marginal[i][j], outside[t][j]+inside[t][i-1]+a[i+1][j-1]+score_ij)
                    marginal[i][j] = np.logaddexp(marginal[i][j], outside[t][j]+inside[t][i-1]+b[i+1][j-1]+score_ij+(-score_stack)/kT)
                # marginal[i][j] = np.logaddexp(marginal[i][j], outside[i][j]+a[i+1][j-1]+score_ij)
                # marginal[i][j] = np.logaddexp(marginal[i][j], outside[i][j]+b[i+1][j-1]+score_ij+(-score_stack)/kT)
    return outside, out_a, out_b, marginal


# def outside_forward_pt_stack(s, inside, a, b):
#     assert len(s) > 1, "the length of rna should be at least 2!"
#     # print(f"sharpturn: {sharpturn}")
#     n = len(s)
#     outside = defaultdict(lambda: defaultdict(lambda: -np.inf)) # partition 
#     out_a = defaultdict(lambda: defaultdict(lambda: -np.inf)) # left and right don't pair
#     out_b = defaultdict(lambda: defaultdict(lambda: -np.inf)) # left and right pair
#     marginal = defaultdict(lambda: defaultdict(lambda: -np.inf))
#     outside[0][n-1] = 0.        
#     for j in range(n-1, 0, -1):
#         for i in range(j-1, -1, -1):
#             # a[i][j] = np.logaddexp(a[i][j], p[i][j-1]+(-unpair_score/kT))  # a <- s.
#             # p[i][j] = np.logaddexp(p[i][j], p[i][j-1]+(-unpair_score/kT))  # p <- s.
#             outside[i][j-1] = np.logaddexp(outside[i][j-1], outside[i][j]+(-unpair_score/kT))
#             # outside[i][j-1] = np.logaddexp(outside[i][j-1], out_a[i][j]+(-unpair_score/kT))
#             # outside[i][j-1] = np.logaddexp(outside[i][j-1], outside[i][j]+(-unpair_score/kT))
#             # print(f'i={i}, j-1={j-1}, outside={outside[i][j-1]: .4f}')
#             # print()
#             if match(s[i], s[j]) and j-i>sharpturn:
#                 pair_ij = s[i]+s[j]
#                 score_ij = -pair2score[pair_ij]/kT
#                 for t in range(0, i):
#                     # a[t][j] = np.logaddexp(a[t][j], p_left+a_right+score_ij) # a <- s(a)
#                     # outside[t][i-1] = np.logaddexp(outside[t][i-1], out_a[t][j]+a[i+1][j-1]+score_ij)
#                     # out_a[i+1][j-1] = np.logaddexp(out_a[i+1][j-1], out_a[t][j]+inside[t][i-1]+score_ij)
#                     # p[t][j] = np.logaddexp(p[t][j], p_left+a_right+score_ij) # p <- s(a)
#                     outside[t][i-1] = np.logaddexp(outside[t][i-1], outside[t][j]+a[i+1][j-1]+score_ij)
#                     # out_a[i+1][j-1] = np.logaddexp(out_a[i+1][j-1], outside[t][j]+inside[t][i-1]+score_ij)
#                     # a[t][j] = np.logaddexp(a[t][j], p_left+b_right+score_ij+(-score_stack/kT)) # a <- s(b)
#                     # outside[t][i-1] = np.logaddexp(outside[t][i-1], out_a[t][j]+b[i+1][j-1]+score_ij+(-score_stack/kT))
#                     # out_b[i+1][j-1] = np.logaddexp(out_b[i+1][j-1], out_a[t][j]+inside[t][i-1]+score_ij+(-score_stack)/kT)
#                     # p[t][j] = np.logaddexp(p[t][j], p_left+b_right+score_ij+(-score_stack/kT)) # p <- s(b)
#                     outside[t][i-1] = np.logaddexp(outside[t][i-1], outside[t][j]+b[i+1][j-1]+score_ij+(-score_stack/kT))
#                     # out_b[i+1][j-1] = np.logaddexp(out_b[i+1][j-1], outside[t][j]+inside[t][i-1]+score_ij+(-score_stack)/kT)
#                     # outside[i+1][j-1] = np.logaddexp(outside[i+1][j-1], outside[t][j]+inside[t][i-1]+score_ij+(-score_stack)/kT)
#                     marginal[i][j] = np.logaddexp(marginal[i][j], outside[t][j]+inside[t][i-1]+a[i+1][j-1]+score_ij)
#                     marginal[i][j] = np.logaddexp(marginal[i][j], outside[t][j]+inside[t][i-1]+b[i+1][j-1]+score_ij+(-score_stack)/kT)
#                     # marginal[i][j] = np.logaddexp(marginal[i][j], outside[t][j]+inside[t][i-1]+np.logaddexp(a[i+1][j-1], b[i+1][j-1])+score_ij)
#                     # print(f'marginal: i={i}, j={j}: {marginal[i][j]}')
#                     # if i==2 and j==3:
#                     #     print(f'{t}, inside_a: {a[i+1][j-1]}, inside_b: {b[i+1][j-1]}, plus: {np.logaddexp(a[i+1][j-1], b[i+1][j-1])}')
#                     #     print(f'{t}, outside: {outside[t][j]}')
#                     #     print(f'{t}, inside left: {inside[t][i-1]}' )
#                 # b[i][j] = np.logaddexp(b[i][j], a[i+1][j-1]+score_ij) # b <- (a)
#                 # out_a[i+1][j-1] = np.logaddexp(out_a[i+1][j-1], out_b[i][j]+score_ij)
#                 # p[i][j] = np.logaddexp(p[i][j], a[i+1][j-1]+score_ij) # p <- (a)
#                 # out_a[i+1][j-1] = np.logaddexp(out_a[i+1][j-1], outside[i][j]+score_ij)
#                 # b[i][j] = np.logaddexp(b[i][j], a[i+1][j-1]+score_ij+(-score_stack/kT)) # b <- (b)
#                 # out_b[i+1][j-1] = np.logaddexp(out_b[i+1][j-1], out_b[i][j]+score_ij+(-score_stack/kT)) 
#                 # p[i][j] = np.logaddexp(p[i][j], a[i+1][j-1]+score_ij+(-score_stack/kT)) # p <- (b)
#                 # out_b[i+1][j-1] = np.logaddexp(out_b[i+1][j-1], outside[i][j]+score_ij+(-score_stack/kT)) 
#                 marginal[i][j] = np.logaddexp(marginal[i][j], outside[i][j]+a[i+1][j-1]+score_ij)
#                 marginal[i][j] = np.logaddexp(marginal[i][j], outside[i][j]+b[i+1][j-1]+score_ij+(-score_stack)/kT)
#     return outside, out_a, out_b, marginal



def inside_outside_forward_pt_stack(s):
    inside, a, b = inside_forward_pt_stack(s)
    outside, out_a, out_b = outside_forward_pt_stack(s, inside, a, b)
    return outside, out_a, out_b


def outside_forward(s, inside, prnt=True): 
    assert len(s) > 1, "the length of rna should be at least 2!"
    assert len(s) == len(inside), "the length of rna should match counts matrix!"
    n = len(s)
    outside = defaultdict(lambda: defaultdict(int))
    p = defaultdict(lambda: defaultdict(int))
    outside[0][n-1] = 1
    for j in range(n-1, -1, -1):
        for i in range(j, -1, -1 ): # end with j-1, start with i
            outside[i][j-1] += outside[i][j] # x.
            if i>0 and match(s[i-1], s[j]) and j-(i-1)>sharpturn: # (x); x(x) 
                counts_right = inside[i][j-1] 
                for t in range(i-1, -1, -1): # end with i-2, start with t
                    counts_left = inside[t][i-2] 
                    counts_out = outside[t][j]
                    outside[t][i-2] += counts_out*counts_right  # pop left
                    outside[i][j-1] += counts_out*counts_left # pop right
                    p[i-1][j] += counts_out*counts_left*counts_right # count pairs
    if prnt:
        triples = []
        for i in p:
            for j in p[i]:
                if p[i][j] > 0:
                    triples.append((i+1, j+1, p[i][j]))
        for triple in sorted(triples):
            print(triple[0], triple[1], triple[2])
    return outside, p


def inside_forward_stoc(s): # s: n*4 matrix, row sum-to-one; A C G U
    assert len(s) > 1, "the length of rna should be at least 2!"
    n = len(s)
    counts = defaultdict(lambda: defaultdict(int)) # number of structures
    for k in range(n):
        counts[k][k]=1
        counts[k][k-1] = 1
    for j in range(1, n):
        for i in range(0, j):
            counts[i][j] += counts[i][j-1] # x.
            if j-i>sharpturn:
                prob = 0
                for il, jr in pairs_index: # (x); x(x) 
                    prob += s[i][il]*s[j][jr]
                counts_right = counts[i+1][j-1] 
                for t in range(0, i):
                    counts_left = counts[t][i-1] 
                    counts[t][j] += counts_left*counts_right*prob
                counts[i][j] += counts_right*prob
    return counts


def inside_forward_stoc_pt(s): # s: n*4 matrix, row sum-to-one; A C G U
    assert len(s) > 1, "the length of rna should be at least 2!"
    # print(f'sharpturn: {sharpturn}')
    n = len(s)
    counts = defaultdict(lambda: defaultdict(lambda: -np.inf)) # number of structures
    for k in range(n):
        counts[k][k] = -unpair_score/kT
        counts[k][k-1] = 0
    for j in range(1, n):
        for i in range(0, j):
            counts[i][j] = np.logaddexp(counts[i][j], counts[i][j-1]+(-unpair_score/kT))  # x.
            if j-i>sharpturn:
                score_ij = -np.inf
                prob_sum = 0.
                for il, jr in pairs_index: 
                    prob = s[i][il]*s[j][jr]
                    prob_sum += prob
                    if prob > 0:
                        pair_ij = nucs[il]+nucs[jr]
                        score_ij = np.logaddexp(score_ij, np.log(prob) + (-pair2score[pair_ij]/kT)) 
                if prob_sum > 0:
                    counts_right = counts[i+1][j-1] 
                    for t in range(0, i):
                        counts_left = counts[t][i-1] 
                        counts[t][j] = np.logaddexp(counts[t][j], counts_left+counts_right+score_ij) # x(x)
                    counts[i][j] = np.logaddexp(counts[i][j], counts_right+score_ij) # (x)
    return counts


def outside_forward_stoc_pt(s, inside): # s: n*4 matrix, row sum-to-one; A C G U
    assert len(s) > 1, "the length of rna should be at least 2!"
    # print(f'sharpturn: {sharpturn}')
    n = len(s)
    outside = defaultdict(lambda: defaultdict(lambda: -np.inf)) # number of structures
    outside[0][n-1] = 0.
    marginal = defaultdict(lambda: defaultdict(lambda: -np.inf))
    for j in range(n-1, -1, -1):
        for i in range(j-1, -1, -1):
            # counts[i][j] = np.logaddexp(counts[i][j], counts[i][j-1]+(-unpair_score/kT))  # x.
            outside[i][j-1] = np.logaddexp(outside[i][j-1], outside[i][j]+(-unpair_score/kT))
            if j-i>sharpturn:
                score_ij = -np.inf
                prob_sum = 0.
                for il, jr in pairs_index: 
                    prob = s[i][il]*s[j][jr]
                    prob_sum += prob
                    if prob > 0:
                        pair_ij = nucs[il]+nucs[jr]
                        score_ij = np.logaddexp(score_ij, np.log(prob) + (-pair2score[pair_ij]/kT)) 
                if prob_sum > 0:
                    # inside_right = inside[i+1][j-1] 
                    for t in range(0, i+1):
                        # counts_left = counts[t][i-1] 
                        # counts[t][j] = np.logaddexp(counts[t][j], counts_left+counts_right+score_ij) # x(x)
                        outside[t][i-1] = np.logaddexp(outside[t][i-1], outside[t][j]+inside[i+1][j-1]+score_ij)
                        outside[i+1][j-1] = np.logaddexp(outside[i+1][j-1], outside[t][j]+inside[t][i-1]+score_ij)
                        marginal[i][j] = np.logaddexp(marginal[i][j], outside[t][j]+inside[t][i-1]+inside[i+1][j-1]+score_ij)
                    # outside[i+1][j-1] = np.logaddexp(outside[i+1][j-1], outside[i][j]+score_ij) # (x)
    return outside, marginal


def inside_outside_forward_pt(s):
    inside = inside_forward_pt(s)
    outside, marginal = outside_forward_pt(s, inside)
    return outside, marginal


def inside_forward_stoc_pt_stack(s):
    assert len(s) > 1, "the length of rna should be at least 2!"
    # print(f"sharpturn: {sharpturn}")
    n = len(s)
    p = defaultdict(lambda: defaultdict(lambda: -np.inf)) # number of structures
    a = defaultdict(lambda: defaultdict(lambda: -np.inf)) 
    b = defaultdict(lambda: defaultdict(lambda: -np.inf))
    for k in range(n):
        p[k][k]=-unpair_score/kT
        p[k][k-1] = 0
        a[k][k]=-unpair_score/kT
        a[k][k-1] = 0        
    for j in range(1, n):
        for i in range(0, j):
            a[i][j] = np.logaddexp(a[i][j], p[i][j-1]+(-unpair_score/kT))  # a <- s.
            p[i][j] = np.logaddexp(p[i][j], p[i][j-1]+(-unpair_score/kT))  # p <- s.
            if j-i > sharpturn:
                score_ij = -np.inf
                score_ij_stack = -np.inf
                for il, jr in pairs_index: 
                    prob = s[i][il]*s[j][jr]
                    if prob > 0:
                        pair_ij = nucs[il]+nucs[jr]
                        score_ij = np.logaddexp(score_ij, np.log(prob) + (-pair2score[pair_ij]/kT))
                        score_ij_stack = np.logaddexp(score_ij_stack, np.log(prob) + (-pair2score[pair_ij]/kT) + (-score_stack/kT))
                a_right = a[i+1][j-1]
                b_right = b[i+1][j-1]
                for t in range(0, i):
                    p_left = p[t][i-1] 
                    a[t][j] = np.logaddexp(a[t][j], p_left+a_right+score_ij) # a <- s(a)
                    p[t][j] = np.logaddexp(p[t][j], p_left+a_right+score_ij) # p <- s(a)
                    a[t][j] = np.logaddexp(a[t][j], p_left+b_right+score_ij_stack) # a <- s(b)
                    p[t][j] = np.logaddexp(p[t][j], p_left+b_right+score_ij_stack) # p <- s(b)
                b[i][j] = np.logaddexp(b[i][j], a_right+score_ij) # b <- (a)
                p[i][j] = np.logaddexp(p[i][j], a_right+score_ij) # p <- (a)
                b[i][j] = np.logaddexp(b[i][j], b_right+score_ij_stack) # b <- (b)
                p[i][j] = np.logaddexp(p[i][j], b_right+score_ij_stack) # p <- (b)
    return p, a, b


def outside_forward_stoc_pt_stack(s, inside, a, b):
    assert len(s) > 1, "the length of rna should be at least 2!"
    # print(f"sharpturn: {sharpturn}")
    n = len(s)
    outside = defaultdict(lambda: defaultdict(lambda: -np.inf)) # number of structures
    out_a = defaultdict(lambda: defaultdict(lambda: -np.inf)) 
    out_b = defaultdict(lambda: defaultdict(lambda: -np.inf))
    outside[0][n-1] = 0.       
    for j in range(n-1, 0, -1):
        for i in range(j-1, -1, -1):
            outside[i][j-1] = np.logaddexp(outside[i][j-1], outside[i][j]+inside[i][j-1]+(-unpair_score/kT))
            # print(f'i={i}, j-1={j-1}, outside={outside[i][j-1]: .4f}')
            # out_a[i][j-1] = np.logaddexp(out_a[i][j-1], a[i][j]+inside[i][j-1]+(-unpair_score/kT))
            # a[i][j] = np.logaddexp(a[i][j], p[i][j-1]+(-unpair_score/kT))  # a <- s.
            # p[i][j] = np.logaddexp(p[i][j], p[i][j-1]+(-unpair_score/kT))  # p <- s.
            if j-i > sharpturn:
                score_ij = -np.inf
                score_ij_stack = -np.inf
                for il, jr in pairs_index: 
                    prob = s[i][il]*s[j][jr]
                    if prob > 0:
                        pair_ij = nucs[il]+nucs[jr]
                        score_ij = np.logaddexp(score_ij, np.log(prob) + (-pair2score[pair_ij]/kT))
                        score_ij_stack = np.logaddexp(score_ij_stack, np.log(prob) + (-pair2score[pair_ij]/kT) + (-score_stack/kT))
                a_right = a[i+1][j-1]
                b_right = b[i+1][j-1]
                for t in range(0, i):
                    inside_left = inside[t][i-1] 
                    # a[t][j] = np.logaddexp(a[t][j], p_left+a_right+score_ij) # a <- s(a)
                    # p[t][j] = np.logaddexp(p[t][j], p_left+a_right+score_ij) # p <- s(a)
                    outside[t][i-1] = np.logaddexp(outside[t][i-1], outside[t][j]+a_right+score_ij)
                    out_a[i+1][j-1] = np.logaddexp(out_a[i+1][j-1], outside[t][j]+inside_left+score_ij)
                    # outside[i+1][j-1] = np.logaddexp(outside[i+1][j-1], outside[t][j]+inside_left+score_ij)
                    # a[t][j] = np.logaddexp(a[t][j], p_left+b_right+score_ij_stack) # a <- s(b)
                    # p[t][j] = np.logaddexp(p[t][j], p_left+b_right+score_ij_stack) # p <- s(b)
                    outside[t][i-1] = np.logaddexp(outside[t][i-1], outside[t][j]+b_right+score_ij_stack)
                    out_b[i+1][j-1] = np.logaddexp(out_b[i+1][j-1], outside[t][j]+inside_left+score_ij_stack)
                    # outside[i+1][j-1] = np.logaddexp(outside[i+1][j-1], outside[t][j]+inside_left+score_ij_stack)
                # b[i][j] = np.logaddexp(b[i][j], a_right+score_ij) # b <- (a)
                # p[i][j] = np.logaddexp(p[i][j], a_right+score_ij) # p <- (a)
                out_a[i+1][j-1] = np.logaddexp(out_a[i+1][j-1], outside[i][j]+score_ij)
                # b[i][j] = np.logaddexp(b[i][j], b_right+score_ij_stack) # b <- (b)
                # p[i][j] = np.logaddexp(p[i][j], b_right+score_ij_stack) # p <- (b)
                out_b[i+1][j-1] = np.logaddexp(out_b[i+1][j-1], outside[i][j]+score_ij_stack)
    # print('outside[0][n-1]: ', outside[0][n-1])
    return outside, out_a, out_b


# def inside_outside_forward_stoc_pt_stack(s):
#     inside, a, b = inside_forwardptstack


def outside_forward_stoc(s, inside, prnt=True): 
    assert len(s) > 1, "the length of rna should be at least 2!"
    assert len(s) == len(inside), "the length of rna should match counts matrix!"
    n = len(s)
    outside = defaultdict(lambda: defaultdict(int))
    p = defaultdict(lambda: defaultdict(int))
    outside[0][n-1] = 1
    for j in range(n-1, -1, -1):
        for i in range(j, -1, -1 ): # end with j-1, start with i, start with at most j
            outside[i][j-1] += outside[i][j] # x.
            if i>0 and j-(i-1)>sharpturn:
                counts_right = inside[i][j-1] 
                prob = 0
                for il, jr in pairs_index:
                    prob += s[i-1][il]*s[j][jr]
                for t in range(i-1, -1, -1): # end with i-2, start with t, start with at most i-1
                    counts_left = inside[t][i-2] 
                    counts_out = outside[t][j]
                    outside[t][i-2] += counts_out*counts_right*prob  # pop left
                    outside[i][j-1] += counts_out*counts_left*prob # pop right
                    p[i-1][j] += counts_out*counts_left*counts_right*prob # count pairs
    if prnt:
        triples = []
        for i in p:
            for j in p[i]:
                if p[i][j] > 0:
                    triples.append((i+1, j+1, p[i][j]))
        for triple in sorted(triples):
            print(triple[0], triple[1], triple[2])
    return outside, p


def seq2arr(s):
    n = len(s)
    arr = np.zeros((n, 4))
    for i, c in enumerate(s):
        arr[i][nuc2idx[c.upper()]] = 1
    return arr


def arr2seq(arr):
    s= ""
    for idx in arr:
        s += nucs[idx]
    return s


def rand_arr(l): # return a nd array
    arr_rand = np.random.rand(l, 4)
    for i in range(len(arr_rand)):
        arr_rand[i] /= sum(arr_rand[i])
    return arr_rand


def rand_simplex(n, dim, seed=None):
    if seed is not None:
        np.random.seed(seed)
    mat = np.random.rand(n, dim)
    for i in range(n):
        mat[i] /= sum(mat[i])
    return mat


def all_rna(l):
    assert l > MIN_LEN
    num_rna = 4**l # total number of all sequences of length l
    for i in range(num_rna): # i is the index of one sequence
        digits = [0 for _ in range(l)] # Quaternary numeral system
        remain = i # generate each digit using modulo of remain
        for j in range(l):
            digits[l-1-j] = remain%4 # digits at position l-1-j
            remain = remain>>2  # update remain
        yield digits


def all_inside(l, f=inside_forward):
    rna_all = all_rna(l)
    counts_all = []
    for i, arr in enumerate(rna_all):
        count = f(arr2seq(arr))
        counts_all.append(count)
        # print(arr2seq(arr), count[0][l-1])
    return counts_all


def all_outside(l, inside_all):
    rna_all = all_rna(l)
    outside_all = []
    for i, arr in enumerate(rna_all):
        seq = arr2seq(arr)
        inside = inside_all[i]
        outside = outside_forward(seq, inside, prnt=False)
        outside_all.append(outside)
    return outside_all


def all_probs(rna):
    l = len(rna)
    num_rna = 4**l
    probs = []
    for i in range(num_rna):
        digits = []
        start = i
        for j in range(l):
            digits.append(start%4)
            start = start>>2
        digits =reversed(digits)
        pr = 1
        for i, d in enumerate(digits):
            pr *= rna[i][d]
        probs.append(pr)
    return probs


def enum_upair(probs):
    if len(probs) == 0:
        yield "", 1
    else:
        for seq, prob in enum_upair(probs[:-1]):
            for i, rate in enumerate(probs[-1]):
                yield seq+nucs[i], prob*rate

                
def enum_pair(probs):
    if len(probs) == 0:
        yield [], 1
    else:
        for seq, prob in enum_pair(probs[:-1]):
            for i, rate in enumerate(probs[-1]):
                seq_new = copy.deepcopy(seq)
                seq_new.append(nuc_pairs[i])
                yield seq_new, prob*rate
                
                
def stitch(probs_unpair_cmpt, probs_pair, idx_unpair, idx_pair):
    l = len(idx_unpair) + 2*len(idx_pair)
    seq_prob_list = []
    for unpair, p_u in enum_upair(probs_unpair_cmpt):
        for pair, p_p in enum_pair(probs_pair):
            seq = ["A"]*l
            for i_u, c in enumerate(unpair):
                seq[idx_unpair[i_u]] = c
            for i_p, c in enumerate(pair):
                seq[idx_pair[i_p][0]] = c[0]
                seq[idx_pair[i_p][1]] = c[1]
            seq_prob_list.append(("".join(seq), p_u*p_p)) # or to use yield to enumerate all seqs and probs
    return seq_prob_list


def probs_join(probs_unpair, probs_pair, idx_pair):
    assert len(probs_pair) == len(idx_pair)
    for prob_six, (i, j) in zip(probs_pair, idx_pair):
        simplex_i = np.zeros(4)
        simplex_j = np.zeros(4)
        for prob, (idx_nuc_l, idx_nuc_r) in zip(prob_six, pairs_index):
            simplex_i[idx_nuc_l] += prob
            simplex_j[idx_nuc_r] += prob
        assert abs(sum(simplex_i) - 1) < 0.0001, sum(simplex_i)
        assert abs(sum(simplex_j) - 1) < 0.0001, sum(simplex_j)
        probs_unpair[i] = simplex_i
        probs_unpair[j] = simplex_j
    return probs_unpair


def pt_with_constraints(ss, probs_unpair=None, probs_pair=None):
    from utils.structure import extract_pairs_list
    print(f'ss: {ss}')
    idx_unpair = [i for i, s in enumerate(ss) if s=='.']
    idx_pair = sorted(extract_pairs_list(ss))
    print('idx_unpair:', idx_unpair)
    print('idx_pair:', idx_pair)
    if probs_unpair is None:
        probs_unpair = rand_simplex(len(ss), len(nucs), seed=2022)
    if probs_pair is None:
        probs_pair = rand_simplex(len(idx_pair), len(nuc_pairs), seed=2022)
    probs_unpair_cmpt = probs_unpair[idx_unpair]
    assert len(probs_unpair_cmpt) + 2*len(probs_pair) == len(ss)
    seq_prob_list = stitch(probs_unpair_cmpt, probs_pair, idx_unpair, idx_pair)
    print(f'count of possible seq_upair: {len(nucs)**len(idx_unpair)}')
    print(f'count of possible seq_pair : {len(nuc_pairs)**len(idx_pair)}')
    print(f'seq_unpair * seq_pair : {len(nucs)**len(idx_unpair)*len(nuc_pairs)**len(idx_pair)}')
    print(f'count of possible seqs: {len(seq_prob_list)}')
    print(f'count of all seqs: {len(nucs)**len(ss)}')
    pt_list = [inside_forward_pt(seq)[0][len(ss)-1] for seq, _ in seq_prob_list]
    prob_list = [prob for _, prob in seq_prob_list]
    pt_enum = np.log(np.dot(np.exp(pt_list), prob_list))
    print(f"pt_enum: {pt_enum}")
    probs_seq = probs_join(probs_unpair, probs_pair, idx_pair)
    pt_dp = inside_forward_stoc_pt(probs_seq)[0][len(probs_seq)-1]
    print(f"pt_dp  : {pt_dp}")
    pt_all = all_inside(len(ss), inside_forward_pt)
    probs_all = all_probs(probs_seq)
    pt_expect= np.log(np.dot([np.exp(pt[0][len(ss)-1]) for pt in pt_all], probs_all))
    print(f'pt_enum_2: {pt_expect}')
    
                    
def test(length=5):
    counts_all = all_inside(length)
    arr_rand = rand_arr(length)
    print('stochastic sequence:')
    print(arr_rand)
    count_stoc = inside_forward_stoc(arr_rand)
    print(f'count by inside: {count_stoc[0][length-1]}')
    probs_all = all_probs(arr_rand)
    count_expect= np.dot([count[0][length-1] for count in counts_all], probs_all)
    print(f'count by expectation: {count_expect}')
    outside_stoc, p_stoc = outside_forward_stoc(arr_rand, count_stoc, prnt=False)
    print('stochastic outside: ')
    for i in range(length):
        for j in range(i+1, length):
            print(i+1, j+1, f"{outside_stoc[i][j]: .6e}")
    print('expected outside: ')
    outside_all = all_outside(args.length, counts_all)
    for i in range(length):
        for j in range(i+1, length):
            outside_expect = np.dot([outside[0][i][j] for outside in outside_all], probs_all)
            print(i+1, j+1, f"{outside_expect: .6e}")
    print('stochastic pairs: ')
    for i in range(length):
        for j in range(i+1, length):
            print(i+1, j+1, f"{p_stoc[i][j]: .6e}")
    print('expected pairs: ')
    outside_all = all_outside(args.length, counts_all)
    for i in range(length):
        for j in range(i+1, length):
            outside_expect = np.dot([outside[1][i][j] for outside in outside_all], probs_all)
            print(i+1, j+1, f"{outside_expect: .6e}")

            
def energy_simple(seq, ss):
    e_simple = 0
    pair_map = extract_pairs(ss)
    for i, nuc in enumerate(seq):
        if pair_map[i] == i:
            e_simple += unpair_score
        else:
            j = pair_map[i]
            if i < j:
                pair = seq[i]+seq[j]
                e_simple += pair2score[pair]
    return e_simple

            
def energy_stack(seq, ss):
    e_stack = 0
    from utils.structure import extract_pairs
    pair_map = extract_pairs(ss)
    for i, nuc in enumerate(seq):
        if pair_map[i] == i:
            e_stack += unpair_score
        else:
            j = pair_map[i]
            if i < j:
                pair = seq[i]+seq[j]
                e_stack += pair2score[pair]
                if (i-1) >= 0 and (j+1) < len(seq) and pair_map[i-1] == j+1:
                    e_stack += score_stack
    return e_stack
            

def test_pt(length):
    print(f'score_stack: {score_stack}')
    print(f'sharpturn: {sharpturn}')
    import random
    from scipy.special import logsumexp
    rna = ""
    for i in range(length):
        rna += random.choice("ACGU")
    # rna = "ACCGAG"
    # rna = "ACAGAG"
    rna = "ACCGGA"
    print(f'seq: {rna}')
    from k_best import algo_all
    best_1, num_struct, ss_all = algo_all(rna, sharpturn=sharpturn)
    e_list = []
    for score, ss in ss_all:
        e_list.append(-energy_stack(rna, ss)/kT)
    pt_gold = logsumexp(e_list)
    inside= inside_forward_pt(rna)
    outside, marginal = outside_forward_pt(rna, inside)
    print(f'gold: {pt_gold}, inside: {inside[0][len(rna)-1]}')
    for i in range(length):
        for j in range(i+1, length):
            if match(rna[i], rna[j]):
                print('in * out:', i, j, marginal[i][j]) # f'inside: {inside[i][j]}, outside: {outside[i][j]}'
                e_list_ij = []
                for _, ss in ss_all:
                    pair_map = extract_pairs(ss)
                    # print(i, j, pair_map)
                    if pair_map[i] == j:
                        # print(i, j, ss)
                        e_list_ij.append(-energy_simple(rna, ss)/kT)
                print('marginal:', i, j, logsumexp(e_list_ij))
                print()
                
                
def test_pt_stack(length):
    print(f'score_stack: {score_stack}')
    print(f'sharpturn: {sharpturn}')
    import random
    from scipy.special import logsumexp
    rna = ""
    for i in range(length):
        rna += random.choice("ACGU")
    # rna = "ACCGAG"
    # rna = "ACAGAG"
    rna = "ACCGGA"
    print(f'seq: {rna}')
    from k_best import algo_all
    best_1, num_struct, ss_all = algo_all(rna, sharpturn=sharpturn)
    e_list = []
    for score, ss in ss_all:
        e_list.append(-energy_stack(rna, ss)/kT)
    pt_gold = logsumexp(e_list)
    inside, a, b= inside_forward_pt_stack(rna)
    outside, out_a, out_b, marginal = outside_forward_pt_stack(rna, inside, a, b)
    print(f'gold: {pt_gold}, inside: {inside[0][len(rna)-1]}')
    for i in range(length):
        for j in range(i+1, length):
            if match(rna[i], rna[j]):
                print('in * out:', i, j, marginal[i][j]) # f'inside: {inside[i][j]}, outside: {outside[i][j]}'
                e_list_ij = []
                for _, ss in ss_all:
                    pair_map = extract_pairs(ss)
                    # print(i, j, pair_map)
                    if pair_map[i] == j:
                        # print(i, j, ss)
                        e_list_ij.append(-energy_stack(rna, ss)/kT)
                print('marginal:', i, j, logsumexp(e_list_ij))
                print()
            

def test_pt_stoc(length=5):
    import random
    pt_all = all_inside(length, inside_forward_pt)
    arr_rand = rand_arr(length)
    # for i in range(length):
    #     if i%2 == 0:
    #         one_hot = np.zeros(4)
    #         one_hot[random.choice([0, 1, 2, 3])] = 1.
    #         arr_rand[i] = one_hot
    print('stochastic sequence:')
    print(arr_rand)
    pt_stoc = inside_forward_stoc_pt(arr_rand)
    print(f'partition by inside: {pt_stoc[0][length-1]}, {np.exp(pt_stoc[0][length-1])}')
    probs_all = all_probs(arr_rand)
    pt_expect= np.dot([np.exp(pt[0][length-1]) for pt in pt_all], probs_all)
    print(f'partition by expectation: {np.log(pt_expect)}, {pt_expect}')
    
    out, marginal = outside_forward_stoc_pt(arr_rand, pt_stoc)
    out_all = all_inside(length, inside_outside_forward_pt)
    for i in range(len(arr_rand)):
        for j in range(i+1, len(arr_rand)):
            # print(f'i={i}, j={j}, outside by     dp: {out[i][j]}, {np.exp(out[i][j])}')
            # out_ij_exp = np.dot([np.exp(o[i][j]) for o, m in out_all], probs_all)
            # print(f'i={i}, j={j}, outside by expect: {np.log(out_ij_exp)}, {out_ij_exp}')
            
            print(f'i={i}, j={j}, marginal by    dp: {marginal[i][j]}, {np.exp(marginal[i][j])}')
            m_ij_exp = np.dot([np.exp(m[i][j]) for o, m in out_all], probs_all)
            print(f'i={i}, j={j}, outside by expect: {np.log(m_ij_exp)}, {m_ij_exp}')
    
    

def test_pt_stoc_stack(length=5):
    pt_all = all_inside(length, inside_forward_pt_stack)
    arr_rand = rand_arr(length)
    print('stochastic sequence:')
    print(arr_rand)
    pt_stoc, a, b = inside_forward_stoc_pt_stack(arr_rand)
    print(f'partition by inside: {pt_stoc[0][length-1]}, {np.exp(pt_stoc[0][length-1])}')
    probs_all = all_probs(arr_rand)
    pt_expect= np.dot([np.exp(pt[0][length-1]) for pt, a, b in pt_all], probs_all)
    print(f'partition by expectation: {np.log(pt_expect)}, {pt_expect}')
    for i in range(len(arr_rand)):
        for j in range(i, len(arr_rand)):
            print(f'inside by     dp, i={i}, j={j}: {pt_stoc[i][j]}, {np.exp(pt_stoc[i][j])}')
            pt_expect= np.dot([np.exp(pt[i][j]) for pt, a, b in pt_all], probs_all)
            print(f'inside by expect, i={i}, j={j}: {np.log(pt_expect)}, {pt_expect}')
            # print(f'i={i}, j={j}, outside by inside: {out_a[i][j]}, {np.exp(out_a[i][j])}')
            # out_a_ij_exp = np.dot([np.exp(o_a[i][j]) for o, o_a, o_b in out_a_all], probs_all)
            # print(f'i={i}, j={j}, outside by expect: {np.log(out_a_ij_exp)}, {out_a_ij_exp}')
    print('-------------------------------------------------------------------')
    out, out_a, out_b = outside_forward_stoc_pt_stack(arr_rand, pt_stoc, a, b)
    out_all = all_inside(length, inside_outside_forward_pt_stack)
    for i in range(len(arr_rand)):
        for j in range(i, len(arr_rand)):
            # print(f'i={i}, j={j}, outside by inside: {out_a[i][j]}, {np.exp(out_a[i][j])}')
            # out_a_ij_exp = np.dot([np.exp(o_a[i][j]) for o, o_a, o_b in out_a_all], probs_all)
            # print(f'i={i}, j={j}, outside by expect: {np.log(out_a_ij_exp)}, {out_a_ij_exp}')
            
            print(f'i={i}, j={j}, outside by     dp: {out_b[i][j]}, {np.exp(out_b[i][j])}')
            out_b_ij_exp = np.dot([np.exp(o_b[i][j]) for o, o_a, o_b in out_all], probs_all)
            print(f'i={i}, j={j}, outside by expect: {np.log(out_b_ij_exp)}, {out_b_ij_exp}')
    seq = "CCUACGAUAAAGC"
    seq = "GGGCGGCCCAAAUG"
    arr = seq2arr(seq)
    pt_seq, a_seq, b_seq = inside_forward_pt_stack(seq)
    pt_arr, a_arr, b_arr = inside_forward_stoc_pt_stack(arr)
    print(f'pt_seq: {pt_seq[0][len(seq)-1]}')
    print(f'pt_arr: {pt_arr[0][len(arr)-1]}')
    # out_seq, out_a_seq, out_b_seq = outside_forward_pt_stack(seq, pt_seq, a_seq, b_seq)
    # print('-------------------------------------------------------------------------------')
    # out_arr, out_a_arr, out_b_arr = outside_forward_stoc_pt_stack(arr, pt_arr, a_arr, b_arr)
    # for i in range(len(seq)):
    #     for j in range(len(seq)):
    #         print(f'i={i}, j={j}, {out_seq[i][j]}, {out_a_seq[i][j]}, {out_b_seq[i][j]}')
    #         print(f'i={i}, j={j}, {out_arr[i][j]}, {out_a_arr[i][j]}, {out_b_arr[i][j]}')
    
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", type=str, default="GCACG")
    parser.add_argument("--sharpturn", type=int, default=0)
    parser.add_argument("-l", "--length", type=int, default=5)
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--testpt", action='store_true')
    parser.add_argument("--testptout", action='store_true')
    parser.add_argument("--teststock", action='store_true')
    parser.add_argument("--teststack", action='store_true')
    parser.add_argument("--ptss", action='store_true')
    args = parser.parse_args()
    print('args:')
    print(args)
    set_sharpturn(args.sharpturn)
    if args.testpt:
        test_pt(args.length)
        exit(0)
    if args.testptout:
        test_pt_stack(args.length)
        exit(0)
    if args.teststock:
        test_pt_stoc(args.length)
        exit(0)
    if args.teststack:
        test_pt_stoc_stack(args.length)
        exit(0)
    if args.ptss:
        ss = '((....))'
        pt_with_constraints(ss)
        print()
        print()
        ss = '(.(..).)'
        pt_with_constraints(ss)
        exit(0)
    if args.test:
        test(args.length)
    else:
        arr = seq2arr(args.rna)
        count = inside_forward_stoc(arr)
        print(f"count: ", count[0][len(arr)-1])
        print("pairs:")
        outside_forward_stoc(arr, count)