
from itertools import count
from random import randrange
import multiprocessing as mp
import numpy as np

import curver

from processor import process
from polyhedron import Polyhedron

def func(T, P, embedding, closed):
    point = P.random_point()
    geometric = embedding.dot(point).tolist()
    curve = T(geometric)
    assert isinstance(curve, curver.kernel.MultiCurve)
    return '-1: {}, {}, {}'.format(geometric, curve.num_components(), curve.topological_type(closed=closed))

def main(args):
    if args.genus == 0 and args.punctures == 6:
        common = dict(
            T=curver.load(0, 6).triangulation,
            P = Polyhedron(eqns=[], ieqs=[
                [0, 1, 0, 1, -2, 0, 0],  # 1 + 3 - 4 - 4
                [0, -1, 1, 0, 2, 0, 0],  # -1 + 2 + 4 + 4
                [0, 1, 1, 0, 0, 0, -1],  # 1 + 2 - 6
                [0, -1, 0, 0, 1, 0, 1],  # 4 + 6 - 1
                [args.upper, -4, -8, -5, -4, -5, -2],
                [-args.lower, 4, 8, 5, 4, 5, 2],
                ] + [[-1] + [0] * i + [1] + [0] * (6-i-1) for i in range(6)]
                ),
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
    elif args.genus == 2 and args.punctures == 0:
        common = dict(
            T=curver.load(2, 1).triangulation,
            P = Polyhedron(eqns=[], ieqs=[
                [-1, 1, 1, 1, 0, 0, -1],
                [-1, -1, -1, -1, 1, 1, 1],
                [args.upper, -7, -8, -7, -5, -5, -6],
                [-args.lower, 7, 8, 7, 5, 5, 6],
                ] + [[-1] + [0] * i + [1] + [0] * (6-i-1) for i in range(6)]
                ),
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
    elif args.genus == 2 and args.punctures == 1:
        common = dict(
            T=curver.load(2, 1).triangulation,
            P = Polyhedron(eqns=[], ieqs=[
                [0, 1, 1, 0,-1, 0, 0, 0, 0],  # 1 + 2 - 4
                [0, 0,-1, 1, 1, 0, 0, 0, 0],  # 3 + 4 - 2
                [0, 0, 0, 0, 0, 1, 1, 0,-1],  # 5 + 6 - 8
                [0, 0, 0, 0, 0, 0,-1, 1, 1],  # 7 + 8 - 6
                [0, 0, 0,-1,-1, 0, 0, 1, 1],  # 7 + 8 - 3 - 4
                [0, 0, 0, 1, 1, 0,-1, 0, 1],  # 3 + 4 + 8 - 6
                [0, 0, 0, 1, 1, 0, 1, 0,-1],  # 3 + 4 + 6 - 8
                [args.upper, -11, -9, -9, -2, -9, -6, -6, -2],
                [-args.lower, 11, 9, 9, 2, 9, 6, 6, 2],
                ] + [[-1] + [0] * i + [1] + [0] * (8-i-1) for i in range(8)]
                ),
            embedding=np.array([
                [ 0, 0, 2, 2, 0, 0, 0, 0],  # 3 + 3 + 4 + 4
                [ 0, 1, 1, 0, 0, 0, 0, 0],  # 2 + 3
                [ 0,-1, 1, 2, 0, 0, 0, 0],  # 3 + 4 + 4 - 2
                [ 1, 0, 1, 0, 0, 0, 0, 0],  # 1 + 3
                [ 1, 1, 0, 0, 0, 0, 0, 0],  # 1 + 2
                [ 0, 0, 0, 0, 0, 1, 1, 0],  # 6 + 7
                [ 0, 0, 0, 0, 0,-1, 1, 2],  # 7 + 8 + 8 - 6
                [ 0, 0, 0, 0, 1, 0, 1, 0],  # 5 + 7
                [ 0, 0, 0, 0, 1, 1, 0, 0],  # 5 + 6
                ], dtype=object),
            closed=False,
            )
    elif args.genus == 3 and args.punctures == 0:
        common = dict(
            T=curver.load(3, 1).triangulation,
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
                [args.upper, -4, -2, -2, -4, -2, -2, -4, -2, -2, -6, -6, -6],
                [-args.lower, +4, +2, +2, +4, +2, +2, +4, +2, +2, +6, +6, +6],
                ] + [[-1] + [0] * i + [1] + [0] * (12-i-1) for i in range(12)]
                ),
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
    else:
        raise ValueError(f'Intrinsic coordinates not known for S_{args.genus},{args.punctures}')

    iterable = (dict() for _ in count())
    process(func, common, iterable, cores=args.cores, path=args.output)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='sample curves on S_{0,6} of at most a given weight')
    parser.add_argument('--genus', '-g', type=int, default=2, help='genus of surface to work over')
    parser.add_argument('--punctures', '-p', type=int, default=0, help='num punctures of surface to work over')
    parser.add_argument('--upper', '-u', type=int, default=1000000, help='max weight of curve allowed')
    parser.add_argument('--lower', '-l', type=int, default=0, help='min weight of curve allowed')
    parser.add_argument('--cores', '-c', type=int, default=1, help='number of cores to use')
    parser.add_argument('--output', '-o', help='path to output to if not stdout')
    args = parser.parse_args()

    main(args)
