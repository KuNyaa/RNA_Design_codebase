import sys
import time
import json
import network_logits as network
import nussinov
import numpy as np
import dynet as dy
from collections import defaultdict
from inside_outside import inside_forward_log, inside_viterbi

energy_model = {'CG': 3, 'GC': 3, 'AU': 2, 'UA': 2, 'GU': 1, 'UG': 1} # a Nussinov–Jacobson Energy Model
sharpturn = 3
INF = 1e32
lr = 1
n_iters = 100

class Float(float): # using log-space for preventing arithmetic underflow
    def __iadd__(self, other):
        self = Float(np.logaddexp(self, other)) # must return Float not float
        return self

float_class = Float

def train(graph):
    def update(type, q_i, q_j, value):
        best[type, q_i, q_j] = dy.logsumexp([best[type, q_i, q_j], value])

    def get_solution(graph):
        seq = ""
        for i, para in enumerate(graph.parameters):
            aa = graph.aa[i]
            seq += codon_table[aa][np.argmax(para.value())]
        return seq

    best = defaultdict(lambda : dy.scalarInput(-INF)) 

    protein = graph.aa
    print(protein)
    m = len(protein) # protein lengthh
    n = m * 3 # mRNA length
    
    dummy = dy.scalarInput(0.)
    graph.parameters[0] += dummy

    # build forward computational graph
    for i in range(n):  # between-nuc indices
        for i_node in graph.nodes[i]:
            best["S", i_node, i_node] = dy.scalarInput(0)
            for iplus1_node, nuc, prob in graph.right_edges[i_node]:
                update("S", i_node, iplus1_node, 0 + dy.log(prob)) # two choices => e^0 + e^0

    for span in range(2, n+1):
        for i in range(n-span+1):
            j = i + span
            for i_node in graph.nodes[i]:
                for j_node in graph.nodes[j]:
                    for jminus1_node, j_nuc, j_prob in graph.left_edges[j_node]:
                        # S -> S N
                        update("S", i_node, j_node, 
                                best["S", i_node, jminus1_node] + dy.log(j_prob))
                        # P -> ( S )
                        if j - i > sharpturn+1:
                            for iplus1_node, i_nuc, i_prob in graph.right_edges[i_node]:
                                if i_nuc + j_nuc in energy_model:
                                    update("P", i_node, j_node,
                                           best["S", iplus1_node, jminus1_node] + energy_model[i_nuc + j_nuc] + dy.log(i_prob) + dy.log(j_prob))
                    # S -> S P
                    for k in range(i, j): # i..j-1; left S could be epsilon
                        for k_node in graph.nodes[k]:
                            update("S", i_node, j_node,
                                   best["S", i_node, k_node] + best["P", k_node, j_node])                       

    # get the final objective
    obj = best["S", (0,0), (n,0)]

    logs = []
    trainer = dy.RMSPropTrainer(graph.model, learning_rate=lr)
    start_time = time.time()
    for it in range(n_iters):
        v = obj.value()
        seq = get_solution(graph)
        realv = inside_forward_log(seq) # more accurate
        viterbiv = inside_viterbi(seq)
        print("[Iteration %d]\tobj value %.5f\tseq value %.5f seq mfe %d ========================" % (it,
            v, realv, viterbiv))
        print("[Current argmax sequence]:", seq)
        logs.append((it, time.time() - start_time, v, realv))

        (-obj).backward() # calculate gradients
        # update parameters
        for i, para in enumerate(graph.parameters):
            aa = protein[i]
            if i > 0 and len(codon_table[aa]) > 1:
                print(i, "%4s" % aa,
                      " | ".join("[%s]:%.3f, grad:%.3f" % (c,v,g) for (c,v,g) in zip(codon_table[aa], para.value(), para.gradient()) \
                         if v > -0.01))
        trainer.update()
        dummy.set(0.)
    print("[Final Result]: (obj): %.5f\t(seq): %s"  % (realv, get_solution(graph)))
    # save logs to a json file
    json_object = json.dumps(logs, indent=4)
    with open("GD_RMSProp_len{}_lr{}_n_iters{}.json".format(m, lr, n_iters), "w") as outfile:
        outfile.write(json_object)



    
if __name__ == "__main__":
    aa_graphs, codon_table = network.read_wheel("coding_wheel.txt")

    for line in sys.stdin:
        protein = line.split()
        graph = network.Lattice()

        for aa in protein:
            graph += aa_graphs[aa]

        mfe, (seq, struct) = nussinov.nussinov_mfe(graph)
        v = inside_forward_log(seq)
        print("iteration -1\tobj value %.5f\tseq value %.5f seq mfe %d ====" % (v, v, mfe))
        print(seq)

        prob_graph = network.Prob_Lattice(graph)
        sys.stdout.flush()
        train(prob_graph)
