import argparse
from collections import defaultdict


pairs = {'au', 'gc', 'gu'}
sharpturn = 0

def match(x, y):
    return (x+y).lower() in pairs or (y+x).lower() in pairs


def set_sharpturn(sharpturn_val):
    global sharpturn
    sharpturn = sharpturn_val
    print(f"sharpturn is: {sharpturn}")


def inside_backward(s): 
    assert len(s) > 1, "the length of rna should be at least 2!"
    n = len(s)
    counts = defaultdict(lambda: defaultdict(int)) # number of structures
    for k in range(n):
        counts[k][k]=1
        counts[k][k-1] = 1
    for l in range(1, n):
        for i in range(0, n-l):
            j = i + l
            counts[i][j] += counts[i][j-1] # x.
            for t in range(i, j):
                if match(s[t],  s[j]) and j-t>sharpturn: # (x); x(x)    
                    count_left = counts[i][t-1] 
                    count_right = counts[t+1][j-1] 
                    counts[i][j] += count_left*count_right     
    return counts


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


def outside_forward(s, inside): 
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
    triples = []
    for i in p:
        for j in p[i]:
            if p[i][j] > 0:
                triples.append((i+1, j+1, p[i][j]))
    for triple in sorted(triples):
        print(triple[0], triple[1], triple[2])
    return outside, p


def outside_backward(s, inside):
    assert len(s) > 1, "the length of rna should be at least 2!"
    assert len(s) == len(inside), "the length of rna should match counts matrix!"
    n = len(s)
    outside = defaultdict(lambda: defaultdict(int))
    p = defaultdict(lambda: defaultdict(int))
    outside[0][n-1] = 1
    for l in range(n, 0, -1):
        for j in range(n-1, l-2, -1): # end: from n-1 to l-1
            i = j-l+1  # start: j-(l-1)
            outside[i][j-1] += outside[i][j] # # x.
            for k in range(j-1, i-1, -1): # split: for j-1 to i
                if match(s[k], s[j]) and j-k>sharpturn: # (x); x(x) 
                    right = inside[k+1][j-1] 
                    left = inside[i][k-1] 
                    outside[i][k-1] += outside[i][j]*right # pop left
                    outside[k+1][j-1] += outside[i][j]*left # pop right
                    p[k][j] += outside[i][j]*left*right
    triples = []
    for i in p:
        for j in p[i]:
            if p[i][j] > 0:
                triples.append((i+1, j+1, p[i][j]))
    for triple in sorted(triples):
        print(triple[0], triple[1], triple[2])
    return outside, p


def inside_outside(algo_in, algo_out, rna):
    inside = algo_in(rna)
    outside, pair = algo_out(rna, inside)
    return inside, outside, pair

                    
def test(algo_in, algo_out):
    rnas = [
        "ACAGU",
        "AC",
        "GUAC",
        "GCACG",
        "CCGG",
        "CCCGGG",
        "UUCAGGA",
        "AUAACCUA",
        "UUGGACUUG",
        "UUUGGCACUA",
        "GAUGCCGUGUAGUCCAAAGACUUC",
        "AGGCAUCAAACCCUGCAUGGGAGCG"
    ]
    print()
    for i, rna in enumerate(rnas):
        inside_outside(algo_in, algo_out, rna)

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", type=str, default="GCACG")
    parser.add_argument("--sharpturn", type=int, default=0)
    parser.add_argument("--inside",  type=str, default='forward')
    parser.add_argument("--outside", type=str, default='backward')
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    print('args:')
    print(args)
    set_sharpturn(args.sharpturn)
    if args.inside == "forward":
        partition_in = inside_forward
    else:
        partition_in = inside_backward
    if args.outside == "forward":
        partition_out = outside_forward
    else:
        partition_out = outside_backward
        
    inside_outside(partition_in, partition_out, args.rna)
    
    if args.test:
        test(partition_in, partition_out)