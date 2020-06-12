
from itertools import count, product
from random import randrange
import multiprocessing as mp

import numpy as np
import curver

from processor import process
try:
    from sage.all import Polyhedron
except ImportError:
    from polyhedron import Polyhedron

T = curver.load(3, 1).triangulation
embedding = np.array([
    #a1, b1, c1, a2, b2, c2, a3, b3, c3,  x,  y,  z
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0, +1, +1],  # 0
    [ 0, -1, -1,  0,  0,  0,  0,  0,  0,  0, +1, +1],  # 1
    [ 0, +1, +1,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 2
    [+1,  0, +1,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 3
    [+1, +1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # 4
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0, +1,  0, +1],  # 5
    [ 0,  0,  0,  0, -1, -1,  0,  0,  0, +1,  0, +1],  # 6
    [ 0,  0,  0,  0, +1, +1,  0,  0,  0,  0,  0,  0],  # 7
    [ 0,  0,  0, +1,  0, +1,  0,  0,  0,  0,  0,  0],  # 8
    [ 0,  0,  0, +1, +1,  0,  0,  0,  0,  0,  0,  0],  # 9
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0, +1, +1,  0],  # 10
    [ 0,  0,  0,  0,  0,  0,  0, -1, -1, +1, +1,  0],  # 11
    [ 0,  0,  0,  0,  0,  0,  0, +1, +1,  0,  0,  0],  # 12
    [ 0,  0,  0,  0,  0,  0, +1,  0, +1,  0,  0,  0],  # 13
    [ 0,  0,  0,  0,  0,  0, +1, +1,  0,  0,  0,  0],  # 14
    ])

def from_point(T, P, index, point):
    geometric = list(embedding.dot(point))  # Matrix multiply to embed into RR^15 weight space.
    multicurve = T(geometric)
    return '{}: {}, {}, {}'.format(index, geometric, multicurve.num_components(), multicurve.topological_type(closed=True))

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
        #    a1, b1, c1, a2, b2, c2, a3, b3, c3,  x,  y,  z
        [-1,  0, -1, -1,  0,  0,  0,  0,  0,  0,  0, +1, +1],
        [-2,  0, -2,  0,  0,  0,  0,  0,  0,  0,  0, +1, +1],
        [-2,  0,  0, -2,  0,  0,  0,  0,  0,  0,  0, +1, +1],
        [-2, +2, +2, +2,  0,  0,  0,  0,  0,  0,  0, -1, -1],
        
        [-1,  0,  0,  0,  0, -1, -1,  0,  0,  0, +1,  0, +1],
        [-2,  0,  0,  0,  0, -2,  0,  0,  0,  0, +1,  0, +1],
        [-2,  0,  0,  0,  0,  0, -2,  0,  0,  0, +1,  0, +1],
        [-2,  0,  0,  0, +2, +2, +2,  0,  0,  0, -1,  0, -1],
        
        [-1,  0,  0,  0,  0,  0,  0,  0, -1, -1, +1, +1,  0],
        [-2,  0,  0,  0,  0,  0,  0,  0, -2,  0, +1, +1,  0],
        [-2,  0,  0,  0,  0,  0,  0,  0,  0, -2, +1, +1,  0],
        [-2,  0,  0,  0,  0,  0,  0, +2, +2, +2, -1, -1,  0],
        
        [args.weight, -4, -2, -2, -4, -2, -2, -4, -2, -2, -6, -6, -6],
        [-args.lower, +4, +2, +2, +4, +2, +2, +4, +2, +2, +6, +6, +6],
        ] + [[-1] + [0] * i + [1] + [0] * (12-i-1) for i in range(12)]
    
    P = Polyhedron(eqns=[], ieqs=ieqs)
    
    def points():
        while True:
            while True:
                x, y, z = randrange(1, args.weight // 6 + 1), randrange(1, args.weight // 6 + 1), randrange(1, args.weight // 6 + 1)
                if (x + y) % 2 == 0 and (y + z) % 2 == 0 and (z + x) % 2 == 0: break
            
            a1, a2, a3 = randrange(1, args.weight // 4 + 1), randrange(1, args.weight // 4 + 1), randrange(1, args.weight // 4 + 1)
            b1, c1 = randrange(1, (y + z) // 2 + 1), randrange(1, (y + z) // 2 + 1)
            b2, c2 = randrange(1, (x + z) // 2 + 1), randrange(1, (x + z) // 2 + 1)
            b3, c3 = randrange(1, (x + y) // 2 + 1), randrange(1, (x + y) // 2 + 1)
            point = (a1, b1, c1, a2, b2, c2, a3, b3, c3, x, y, z)
            if point in P:
                yield point
    
    setup = dict(T=T, P=P)
    datum = (dict(index=-1, point=point) for point in points())
    
    process(setup, from_point, datum, cores=args.cores, path=args.output)

