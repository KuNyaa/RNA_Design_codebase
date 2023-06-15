import network
import sys
from collections import defaultdict

match = {'CG': 3, 'GC': 3, 'AU': 2, 'UA': 2, 'GU': 1, 'UG': 1}
sharpturn = 3

import numpy as np

class Float(float): # log-space
    def __iadd__(self, other):
        self = Float(np.logaddexp(self, other)) # must return Float not float
        return self

class Float2(tuple): # viterbi, including backpointer
    def __iadd__(self, other):
        if other[0] > self[0]:
            self = Float2(other)
        return self

float_class = Float2 # Float2 much slower

def nussinov_mfe(graph):

    def backtraceS(i_node, j_node):
        if i_node == j_node:
            return "", "" # (xxx) case: left side is empty

        try:
            backpointer = bestS[i_node, j_node][1] # together with best

            if type(backpointer) is str: # singleton
                return backpointer, "."
            elif len(backpointer) == 2 and type(backpointer[1]) is str: # xxx .
                jminus1_node, j_nuc = backpointer
                seq, struct = backtraceS(i_node, jminus1_node)
                return seq + j_nuc, struct + "."
            else: # xxx-xxx
                k_node = backpointer                     
                seq1, struct1 = backtraceS(i_node, k_node)
                seq2, struct2 = backtraceP(k_node, j_node)
                return seq1 + seq2, struct1 + struct2  
        except TypeError:
            print("S", i_node, j_node, bestS[i_node, j_node][0], backpointer)
            raise TypeError

    def backtraceP(i_node, j_node): # only one case: P -> ( S )
        try:
            i_nuc, iplus1_node, jminus1_node, j_nuc = bestP[i_node, j_node][1]
            seq, struct = backtraceS(iplus1_node, jminus1_node)
            return i_nuc + seq + j_nuc, "(" + struct + ")"
        except TypeError:
            print("P", i_node, j_node, bestP[i_node, j_node][0], bestP[i_node, j_node][1])
            raise TypeError

                
    protein = graph.aa

    print(protein)
    m = len(protein) # protein lengthh
    n = m * 3 # mRNA length
    bestS = defaultdict(lambda : float_class((-np.inf, None))) # not -1
    bestP = defaultdict(lambda : float_class((-np.inf, None)))
    for i in range(n):  # between-nuc indices
        for i_node in graph.nodes[i]:
            bestS[i_node, i_node] = float_class((0, ""))
            for iplus1_node, nuc in graph.right_edges[i_node]:
                bestS[i_node, iplus1_node] += (0, nuc) # two choices => e^0 + e^0
            
    for span in range(2, n+1):
        for i in range(n-span+1):
            j = i + span
            for i_node in graph.nodes[i]:
                for j_node in graph.nodes[j]:
                    for jminus1_node, j_nuc in graph.left_edges[j_node]:
                        # S -> S N
                        bestS[i_node, j_node] += (bestS[i_node, jminus1_node][0],
                                                  (jminus1_node, j_nuc))

                        # P -> ( S )
                        if j - i > sharpturn+1:
                            for iplus1_node, i_nuc in graph.right_edges[i_node]:
                                if i_nuc + j_nuc in match:
                                    bestP[i_node, j_node] += (bestS[iplus1_node, jminus1_node][0] + match[i_nuc + j_nuc],
                                                              (i_nuc, iplus1_node, jminus1_node, j_nuc))
                            
                    # S -> S P
                    for k in range(i, j): # i..j-1; left S could be epsilon; k==i, if (i,0)->(i,1): -inf
                        for k_node in graph.nodes[k]:
                            bestS[i_node, j_node] += (bestS[i_node, k_node][0] + bestP[k_node, j_node][0],
                                                      k_node)
                                        
    bestv = bestS[(0,0), (n,0)][0]
    #print(bestv)
    bestseq, beststruct = backtraceS((0,0), (n,0))
    return bestv, (bestseq, beststruct)
    
if __name__ == "__main__":
    aa_graphs, _ = network.read_wheel("coding_wheel.txt")
    for line in sys.stdin:
        protein = line.split()
        graph = network.Lattice.protein_graph(protein, aa_graphs)
        print(nussinov_mfe(graph))
