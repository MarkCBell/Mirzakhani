
from sample import get_polyhedron

from processor import process

import curver

def from_geometric(T, P, closed, labels, geometric):
    multicurve = T(geometric, promote=False)  # Save the promotion until we know this is a multicurve in this polytope.
    branch_weights = [multicurve.dual_weight(label) for label in labels]
    if branch_weights not in P:
        return '{}: {}, {}, {}'.format('?', multicurve.geometric, '?', '?')
    
    index = P.get_index(branch_weights)
    multicurve = multicurve.promote()
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
    parser.add_argument('path', type=str, help='path to file to process')
    args = parser.parse_args()
    
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
    
    common = {'T': T, 'P': P, 'closed': args.punctures == 0, 'labels': [label for label in [i for i in range(T.zeta)] + [~i for i in range(T.zeta)][::-1]]}
    with open(args.path) as F:
        iterable = ({'geometric': eval(line)} for line in F)
        process(from_geometric, common, iterable, cores=args.cores)

# grep -oh "\[.*\]" ./* > data.all
# python resample --genus=2 --punctures=0 --zeros=35 --weight=1000000 data.all
# nice stdbuf -i0 -o0 -e0 python resample.py --genus=2 --punctures=0 --weight=1000000 --zeros=35 data.all --cores=20 | tee "S_2_0-$(date).log"

