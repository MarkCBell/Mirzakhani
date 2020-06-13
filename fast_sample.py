
from itertools import count
from random import randrange
import multiprocessing as mp
import numpy as np

import curver

from processor import process
from polyhedron import Polyhedron


def from_point(T, embedding, closed, point):
    geometric = [int(w) for w in embedding.dot(point)]  # Matrix multiply to embed into weight space.
    c = T(geometric)
    return '-1: {}, {}, {}'.format(geometric, c.num_components(), c.topological_type(closed=closed))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='sample curves on S_{0,6} of at most a given weight')
    parser.add_argument('--genus', '-g', type=int, default=2, help='genus of surface to work over')
    parser.add_argument('--punctures', '-p', type=int, default=0, help='num punctures of surface to work over')
    parser.add_argument('--weight', '-w', type=int, default=1000000, help='max weight of curve allowed')
    parser.add_argument('--lower', '-l', type=int, default=0, help='min weight of curve allowed')
    parser.add_argument('--cores', '-c', type=int, default=1, help='number of cores to use')
    parser.add_argument('--output', '-o', help='path to output to if not stdout')
    args = parser.parse_args()
    
    if args.genus == 0 and args.punctures == 6:
        setup = dict(
            T=curver.load(0, 6).triangulation,
            embedding=np.array([
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
                ], dtype=object),
            closed=False,
            )
        P = Polyhedron(eqns=[], ieqs=[
            [0, 1, 0, 1, -2, 0, 0],  # 1 + 3 - 4 - 4
            [0, -1, 1, 0, 2, 0, 0],  # -1 + 2 + 4 + 4
            [0, 1, 1, 0, 0, 0, -1],  # 1 + 2 - 6
            [0, -1, 0, 0, 1, 0, 1],  # 4 + 6 - 1
            [args.weight, -4, -8, -5, -4, -5, -2],
            [-args.lower, 4, 8, 5, 4, 5, 2],
            ] + [[-1] + [0] * i + [1] + [0] * (6-i-1) for i in range(6)]
            )
        B = [(int(lower), int(upper)+1) for lower, upper in zip(*P.basic_bounding_box())]
        points = ([randrange(lower, upper+1) for lower, upper in B] for _ in count())
        datum = (dict(point=point) for point in points if point in P)
    elif args.genus == 2 and args.punctures == 0:
        setup = dict(
            T=curver.load(2, 1).triangulation,
            embedding=np.array([
                [2, 2, 2, 0, 0, 0],
                [0, 0, 0, 1, 1, 2],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 1, 1],
                [0, 0, 0, 1, 0, 1],
                [1, 2, 1, 0, 0, 0],
                [1, 0, 1, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [1, 1, 0, 0, 0, 0],
                ]),
            closed=True,
            )
        P = Polyhedron(eqns=[], ieqs=[
            [-1, 1, 1, 1, 0, 0, -1],
            [-1, -1, -1, -1, 1, 1, 1],
            [args.weight, -7, -8, -7, -5, -5, -6],
            [-args.lower, 7, 8, 7, 5, 5, 6],
            ] + [[-1] + [0] * i + [1] + [0] * (6-i-1) for i in range(6)])
        B = [(int(lower), int(upper)+1) for lower, upper in zip(*P.basic_bounding_box())]
        points = ([randrange(lower, upper+1) for lower, upper in B] for _ in count())
        datum = (dict(point=point) for point in points if point in P)
    elif args.genus == 3 and args.punctures == 0:
        setup = dict(
            T=curver.load(3, 1).triangulation,
            embedding=np.array([
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
                ]),
            closed=True,
            )
        P = Polyhedron(eqns=[], ieqs=[
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
            )
        
        def points():
            while True:
                while True:
                    x, y, z = randrange(1, args.weight // 6 + 1), randrange(1, args.weight // 6 + 1), randrange(1, args.weight // 6 + 1)
                    if (x + y) % 2 == 0 and (y + z) % 2 == 0 and (z + x) % 2 == 0: break
                
                a1, a2, a3 = randrange(1, args.weight // 4 + 1), randrange(1, args.weight // 4 + 1), randrange(1, args.weight // 4 + 1)
                b1, c1 = randrange(1, (y + z) // 2 + 1), randrange(1, (y + z) // 2 + 1)
                b2, c2 = randrange(1, (x + z) // 2 + 1), randrange(1, (x + z) // 2 + 1)
                b3, c3 = randrange(1, (x + y) // 2 + 1), randrange(1, (x + y) // 2 + 1)
                yield (a1, b1, c1, a2, b2, c2, a3, b3, c3, x, y, z)
        
        datum = (dict(point=point) for point in points() if point in P)
    
    process(setup, from_point, datum, cores=args.cores if args.cores > 0 else mp.cpu_count(), path=args.output)

