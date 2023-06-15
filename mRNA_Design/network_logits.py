import dynet as dy
import numpy as np
from collections import defaultdict


class Lattice:
    def __init__(self, aa=None):
        if aa is None: # empty lattice
            self.aa = []
            self.nodes = {0:[(0,0)]}
        else:
            self.aa = [aa]
            self.nodes = {0:[(0,0)], 1:[], 2:[], 3:[(3,0)]}
        self.left_edges = defaultdict(list)
        self.right_edges = defaultdict(list)

    @staticmethod
    def protein_graph(protein, aa_graphs):
        graph = Lattice()
        for aa in protein:
            graph += aa_graphs[aa]
        return graph

    def add_edge(self, n1, n2, nuc):
        self.right_edges[n1].append((n2, nuc))
        self.left_edges[n2].append((n1, nuc))

    def add_node(self, n1):
        pos, num = n1
        self.nodes[pos].append(n1)

    def pp(self):
        print("\nAmino Acid:", self.aa)
        for pos in self.nodes:
            for node in self.nodes[pos]:
                print("node", node)
                for n2, nuc in self.right_edges[node]:
                    print(" " * 14, node, "-" + nuc + "->", n2)
                for n1, nuc in self.left_edges[node]:
                    print("  ", n1, "<-" + nuc + "-", node)

    def __iadd__(self, other):
        def edges_add_n(edge_list):
            return [((pos+n, num), nuc) for (pos, num), nuc in edge_list]

        def nodes_add_n(node_list):
            return [(pos+n, num) for (pos, num) in node_list]

        n = max(self.nodes) # max node position
        self.aa += other.aa
        for pos, node_list in other.nodes.items():
            if pos > 0: 
                self.nodes[pos + n] = nodes_add_n(node_list)
            for pos, num in node_list:
                self.right_edges[pos + n, num] += edges_add_n(other.right_edges[pos, num])                
                self.left_edges [pos + n, num] += edges_add_n(other.left_edges [pos, num]) # safe for 0
        return self

class Prob_Lattice(Lattice):

    def __init__(self, lattice): # assign random vars to each 3rd position (3rd letter in each codon)
        self.aa = lattice.aa
        self.nodes = lattice.nodes
        self.left_edges = defaultdict(list)
        self.right_edges = defaultdict(list)

        self.model = dy.ParameterCollection()
        self.parameters = []        
        
        for i in self.nodes:            
            if i % 3 > 0: # (i-1) -- (i) is not a 3rd letter
                for i_node in self.nodes[i]:
                    for (iminus1_node, nuc) in lattice.left_edges[i_node]:
                        self.add_edge(iminus1_node, i_node, nuc, dy.scalarInput(1.)) # no random
            
            elif i > 0: # (i-1) -- (i) is a 3rd letter
                for i_node in self.nodes[i]: # TODO: just nodes[0]
                    num_codons = len (lattice.left_edges[i_node])
                    curr_para = self.model.add_parameters(num_codons)
                    curr_para.set_value([0] * num_codons)
                    #curr_para.set_value(np.random.randn(num_codons).tolist())
                    self.parameters.append(curr_para)
                    porbs = dy.softmax(curr_para)

                    for p, (iminus1_node, nuc) in enumerate(lattice.left_edges[i_node]):
                        self.add_edge(iminus1_node, i_node, nuc, porbs[p])

    def add_edge(self, n1, n2, nuc, prob):
        self.right_edges[n1].append((n2, nuc, prob))
        self.left_edges[n2].append((n1, nuc, prob))

    def pp(self):
        print("\nAmino Acid:", self.aa)
        for pos in self.nodes:
            for node in self.nodes[pos]:
                print("node", node)
                for n2, nuc, prob in self.right_edges[node]:
                    print(" " * 14, node, "-" + nuc + "->", n2, "%.3f" % prob.value())
                for n1, nuc, prob in self.left_edges[node]:
                    print("  ", n1, "<-" + nuc + "-", node, "%.3f" % prob.value())

    
def read_wheel(filename, old=False):
    aa_graphs = {}
    codon_table = {}
    for line in open(filename):
        stuff = line.strip().split("\t") # Leu	U U AG	C U UCAG
        aa = stuff[0]
        graph = Lattice(aa)
        last_first = None
        codons = []
        for i, option in enumerate(stuff[1:]):
            first, second, thirds = option.split(" ")
            for third in thirds:
                codons.append(first+second+third)
            n2 = (2, i)
            graph.add_node(n2)
            if first != last_first:
                n1 = (1, i)
                graph.add_node(n1)
                graph.add_edge((0, 0), n1,     first)
            else:
                n1 = (1, i-1) # STOP shares the first U

            last_first = first
            graph.add_edge(n1,     n2,     second)
            for third in thirds: # third nuc is often ambiguous
                graph.add_edge(n2,     (0,0) if old else (3, 0), third)
        aa_graphs[aa] = graph
        codon_table[aa] = codons

    return aa_graphs, codon_table
            
if __name__ == "__main__":
    read_wheel("coding_wheel.txt")
