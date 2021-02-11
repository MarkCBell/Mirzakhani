
from random import randrange
import multiprocessing as mp

import curver

from polyhedron import Polyhedron
from processor import process

def from_index(T, P, closed, index):
    branch_weights = [int(x) for x in P.get_integral_point(index, triangulation='cddlib')]
    geometric = [sum(branch_weights[label] for label in T.corner_lookup[i].labels[1:]) for i in range(T.zeta)]
    multicurve = T(geometric)
    return '{}: {}, {}, {}'.format(index, multicurve.geometric, multicurve.num_components(), multicurve.topological_type(closed=closed))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='sample curves of at most a given weight')
    parser.add_argument('--num', '-n', type=int, default=1000, help='number of samples to take')
    parser.add_argument('--sig', '-s', type=str, help='signature of triangulation to use')
    parser.add_argument('--genus', '-g', type=int, default=2, help='genus of surface to work over')
    parser.add_argument('--punctures', '-p', type=int, default=0, help='num punctures of surface to work over')
    parser.add_argument('--weight', '-w', type=int, default=1000000, help='max weight of a curve')
    parser.add_argument('--zeros', '-z', type=int, default=35, help='which normal arcs to set to zero')
    parser.add_argument('--cores', '-c', type=int, default=1, help='number of cores to use')
    parser.add_argument('--output', '-o', help='path to output to if not stdout')
    args = parser.parse_args()
    
    if args.cores <= 0: args.cores = mp.cpu_count()
    
    if args.sig is not None:
        T = curver.triangulation_from_sig(args.sig)
    else:
        T = curver.load(args.genus, max(args.punctures, 1)).triangulation
    
    P = Polyhedron.from_triangulation(T, args.weight, zeros=args.zeros)
    num_integral_points = P.integral_points_count(triangulation='cddlib')
    print(P)
    try:
        print('Polytope dimension: {}'.format(P.as_sage().dimension()))
    except AttributeError:
        print('Polytope dimension: Unknown')
    print('Drawing from [0, {})'.format(num_integral_points))
    
    common = dict(T=T, P=P, closed=args.punctures == 0)
    iterable = (dict(index=randrange(0, num_integral_points)) for _ in range(args.num))
    
    process(from_index, common, iterable, cores=args.cores, path=args.output)

# python sample.py --genus=2 --punctures=0 --zeros=35
# python sample.py --genus=1 --punctures=2 --zeros=18
# python sample.py --genus=0 --punctures=6 --zeros=197187

