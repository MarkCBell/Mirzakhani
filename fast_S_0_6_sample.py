
from itertools import count
from random import randrange
import multiprocessing as mp
import numpy as np

import curver

from processor import process
from polyhedron import Polyhedron

T = curver.load(0, 6).triangulation
embedding = np.array([
    [ 1, 0, 1,-1, 0, 0],
    [ 1, 0, 1, 0, 0, 0],
    [ 1, 1, 0, 0, 0, 0],
    [ 1, 1, 0, 0, 1,-1],
    [ 1, 0, 1,-2, 0, 0],
    [-1, 1, 0, 2, 0, 0],
    [-1, 1, 0, 2, 1, 1],
    [ 0, 1, 0, 1, 1, 0],
    [ 0, 0, 0, 1, 0, 0],
    [ 0, 1, 1, 0, 0, 0],
    [ 0, 0, 0, 0, 1, 1],
    [-1, 0, 0, 1, 0, 1],
    ], dtype=object)

def from_point(T, P, index, point):
    if point not in P:
        return 'not {}'.format(point)
    geometric = [int(w) for w in embedding.dot(point)]  # Matrix multiply to embed into RR^18 weight space.
    c = T(geometric)
    return '{}: {}, {}, {}'.format(index, geometric, c.num_components(), c.topological_type())

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='sample curves on S_2 of at most a given weight')
    parser.add_argument('--weight', '-w', type=int, default=1000000, help='max weight of curve allowed')
    parser.add_argument('--lower', '-l', type=int, default=0, help='min weight of curve allowed')
    parser.add_argument('--systematic', '-s', action='store_true', help='evaluate all')
    parser.add_argument('--cores', '-c', type=int, default=1, help='number of cores to use')
    parser.add_argument('--output', '-o', help='path to output to if not stdout')
    args = parser.parse_args()
    
    if args.cores <= 0: args.cores = mp.cpu_count()
    
    ieqs = [
        [0, 1, 0, 1, -2, 0, 0],  # 1 + 3 - 4 - 4
        [0, -1, 1, 0, 2, 0, 0],  # -1 + 2 + 4 + 4
        [0, 1, 1, 0, 0, 0, -1],  # 1 + 2 - 6
        [0, -1, 0, 0, 1, 0, 1],  # 4 + 6 - 1
        [args.weight, -4, -8, -5, -4, -5, -2],
        [-args.lower, 4, 8, 5, 4, 5, 2],
        ] + [[-1] + [0] * i + [1] + [0] * (6-i-1) for i in range(6)]
    
    P = Polyhedron(eqns=[], ieqs=ieqs)
    B = [(int(lower), int(upper)+1) for lower, upper in zip(*P.basic_bounding_box())]
    
    setup = dict(T=T, P=P)
    points = ([randrange(lower, upper+1) for lower, upper in B] for _ in count())
    datum = (dict(index=-1, point=point) for point in points if point in P)
    
    process(setup, from_point, datum, cores=args.cores, path=args.output)

