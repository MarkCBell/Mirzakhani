
from random import randrange
import multiprocessing as mp

import curver

from polyhedron import get_polyhedron
from processor import process

def from_index(T, P, closed, index):
    branch_weights = [int(x) for x in P.get_integral_point(index, triangulation='cddlib')]
    geometric = [sum(branch_weights[label] for label in T.corner_lookup[i].labels[1:]) for i in range(T.zeta)]
    multicurve = T(geometric)
    return '{}: {}, {}, {}'.format(index, multicurve.geometric, multicurve.num_components(), multicurve.topological_type(closed=closed))

def get_polyhedron(T, max_weight, zeroed=None, zeros=None):
    if zeros is not None: zeroed = [(zeros >> i) & 1 for i in range(2 * T.zeta)][::-1]
    # Build the polytope.
    eqns, ieqs = [], []
    # Edge equations.
    for i in range(T.zeta):
        eqn = [0] * 2*T.zeta
        if T.is_flippable(i):
            A, B = T.corner_lookup[i], T.corner_lookup[~i]
            x, y = A.labels[1], A.labels[2]
            z, w = B.labels[1], B.labels[2]
            eqn[x], eqn[y], eqn[z], eqn[w] = +1, +1, -1, -1
        else:
            A = T.corner_lookup[i]
            x = A.labels[1] if A[1] == ~A[0] else A.labels[2]
            z = A.labels[0]
            eqn[x], eqn[z] = +1, -1
        eqns.append([0] + eqn)  # V_x + X_y == V_z + V_w.
    # Zeroed (in)equalities
    for i in range(2*T.zeta):
        if not zeroed[i]:
            ieq = [0] * 2*T.zeta
            ieq[i] = +1
            ieqs.append([-1] + ieq)  # V_i >= 1.
        else:  # Zeroed equation.
            eqn = [0] * 2*T.zeta
            eqn[i] = +1
            eqns.append([0] + eqn)  # V_i == 0.
    # Max weight inequality.
    ieqs.append([max_weight] + [-1] * 2*T.zeta)  # sum V_i <= max_weight.
    return Polyhedron(eqns=eqns, ieqs=ieqs)

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
    
    P = get_polyhedron(T, args.weight, zeros=args.zeros)
    num_integral_points = P.integral_points_count(triangulation='cddlib')
    print(P)
    try:
        print('Polytope dimension: {}'.format(P.as_sage().dimension()))
    except AttributeError:
        print('Polytope dimension: Unknown')
    print('Drawing from [0, {})'.format(num_integral_points))
    
    setup = dict(T=T, P=P, closed=args.punctures == 0)
    datum = (dict(index=randrange(0, num_integral_points)) for _ in range(args.num))
    
    process(setup, from_index, datum, cores=args.cores, path=args.output)

# python sample.py --genus=2 --punctures=0 --zeros=35
# python sample.py --genus=1 --punctures=2 --zeros=18
# python sample.py --genus=0 --punctures=6 --zeros=197187

