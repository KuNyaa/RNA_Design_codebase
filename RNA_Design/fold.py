import argparse

from k_best import algo_0, algo_1, algo_2, algo_all
from k_best_lazy import algo_3

def test(algo, k):
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
    for rna in rnas:
        best_1, num_struct, best_k = algo(rna, k)
        print(rna)
        print(best_1)
        print(num_struct)
        print(best_k)
        print()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rna", type=str, default="GCACG")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--algo", type=int, default=0)
    parser.add_argument("--test", action='store_true')
    args = parser.parse_args()
    print(f"args: {args}")
    algo = algo_0
    if args.algo == 1:
        algo = algo_1
    if args.algo == 2:
        algo = algo_2
    if args.algo == 3:
        algo = algo_3
    if args.algo == 4:
        algo = algo_all
    best_1, num_struct, best_k = algo(args.rna, args.k)
    print()
    print(args.rna)
    print(best_1)
    print(num_struct)
    print(best_k)
    
    if args.test:
        test(algo, args.k)
    