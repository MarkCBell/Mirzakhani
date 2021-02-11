
import os
from subprocess import Popen, PIPE
try:
    from sage.all import Polyhedron as PP
except ImportError:
    PP = None
from tempfile import TemporaryDirectory
from fractions import Fraction
import inspect
from decorator import decorator

@decorator
def memoize(function, *args, **kwargs):
    ''' A decorator that memoizes a function. '''
    
    inputs = inspect.getcallargs(function, *args, **kwargs)  # pylint: disable=deprecated-method
    self = inputs.pop('self', function)  # We test whether function is a method by looking for a `self` argument. If not we store the cache in the function itself.
    
    if not hasattr(self, '_cache'):
        self._cache = dict()
    key = (function.__name__, frozenset(inputs.items()))
    if key not in self._cache:
        self._cache[key] = function(*args, **kwargs)
    
    return self._cache[key]

class Polyhedron:
    def __init__(self, eqns, ieqs):
        self.eqns = eqns
        self.ieqs = ieqs
        self.ambient_dimension = len((self.eqns + self.ieqs)[0]) - 1
    
    @classmethod
    def from_triangulation(cls, T, max_weight, zeroed=None, zeros=None):
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
        return cls(eqns=eqns, ieqs=ieqs)
    
    def __str__(self):
        return 'EQN:[\n{}\n]\nIEQS:[\n{}\n]'.format(',\n'.join(str(eqn) for eqn in self.eqns), ',\n'.join(str(ieq) for ieq in self.ieqs))
    
    def split(self, inequality):
        return Polyhedron(self.eqns, self.ieqs + [inequality])
    def restrict(self, equality):
        return Polyhedron(self.eqns + [equality], self.ieqs)
    def as_sage(self):
        if PP is None:
            raise AttributeError('Not running in Sage')
        return PP(eqns=self.eqns, ieqs=self.ieqs)
    
    def __contains__(self, coordinate):
        assert len(coordinate) == self.ambient_dimension
        homogenised = [1] + list(coordinate)
        def dot(X, Y): return sum(x * y for x, y in zip(X, Y))
        return all(dot(homogenised, eqn) == 0 for eqn in self.eqns) and all(dot(homogenised, ieq) >= 0 for ieq in self.ieqs)
    
    def get_index(self, coordinate, **kwds):
        assert coordinate in self
        D = self.ambient_dimension
        P = self
        index = 0
        for i in range(D):
            neg_axis = [0] * i + [-1] + [0] * (D - i - 1)
            P_lt_comp = P.split([coordinate[i]-1] + neg_axis)
            index += P_lt_comp.integral_points_count(**kwds)
            P = P.restrict([coordinate[i]] + neg_axis)
        
        return index
    
    def get_integral_point(self, index, **kwds):
        D = self.ambient_dimension
        axes = [[0] *i + [1] + [0] * (D - i - 1) for i in range(D)]
        coordinate = []
        P = self
        P_count = P.integral_points_count(**kwds)  # Record the number of integral points in P_{lower <= x_i < upper}.
        bounding_box = zip(*self.bounding_box())
        
        for axis, bounds in zip(axes, bounding_box):  # Now compute x_i, the ith component of coordinate.
            neg_axis = [-x for x in axis]
            lower, upper = int(bounds[0]), int(bounds[1]) + 1  # So lower <= x_i < upper.
            while lower < upper-1:
                # print(coordinate, lower, upper, P_count)
                guess = (lower + upper) // 2  # > lower.
                # Build new polyhedron by intersecting P with the halfspace {x_i < guess}.
                P_lt_guess = P.split([-lower] + axis).split([guess-1] + neg_axis)
                P_lt_guess_count = P_lt_guess.integral_points_count(**kwds)
                
                if P_lt_guess_count > index:  # Move upper down to guess.
                    upper = guess
                    index -= 0
                    P_count = P_lt_guess_count
                else:  # P_lt_guess_count <= index:  # Move lower up to guess.
                    lower = guess
                    index -= P_lt_guess_count
                    P_count -= P_lt_guess_count
                if P_count == 1:
                    Q = P.split([-lower] + axis).split([upper-1] + neg_axis)
                    vertices = Q.vertices()
                    if len(vertices) == 1:  # Polytope is 0-dimensional.
                        return [int(x) for x in vertices[0]]  # Remove any Fractions.
            coordinate.append(lower)  # Record the new component that we have found.
            P = P.restrict([lower] + neg_axis)
        # assert self.get_index(coordinate) == orig_index
        return coordinate
    
    def integral_points_count(self, **kwds):
        # latte_input = 'H-representation\n{} {} rational\n{}\nlinearity {} {}'.format(
        latte_input = '{} {}\n{}\nlinearity {} {}'.format(
            len(self.eqns) + len(self.ieqs), self.ambient_dimension+1,
            '\n'.join(' '.join(str(x) for x in X) for X in self.eqns + self.ieqs),
            len(self.eqns), ' '.join(str(i+1) for i in range(len(self.eqns))),
            )
        
        args = [os.path.abspath(os.path.join('bin', 'count'))]

        for key, value in kwds.items():
            if value is None or value is False:
                continue

            key = key.replace('_','-')
            if value is True:
                args.append('--{}'.format(key))
            else:
                args.append('--{}={}'.format(key, value))
        # args += ['/dev/stdin']

        # The cwd argument is needed because latte
        # always produces diagnostic output files.
        with TemporaryDirectory() as tmpdir:
            # with TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, 'test.pol'), 'w') as x:
                x.write(latte_input)
            args += [os.path.join(tmpdir, 'test.pol')]
            # X = subprocess.run(args, capture_output=True)
            # ans, err = X.stdout, X.stderr
            # ret_code = X.returncode
            latte_proc = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=str(tmpdir))
            ans, err = latte_proc.communicate()
            ret_code = latte_proc.poll()
            
            if ans: # Sometimes (when LattE's preproc does the work), no output appears on stdout.
                ans = ans.splitlines()[-1].decode()
            else:
                # opening a file is slow (30e-6s), so we read the file
                # numOfLatticePoints only in case of a IndexError above
                with open(os.path.join(tmpdir, 'numOfLatticePoints'), 'r') as f:
                    ans = f.read()

        try:
            return int(ans)
        except ValueError:
            return 0
    
    @memoize
    def vertices(self):
        cddlib_input = 'H-representation\nlinearity {} {}\nbegin\n{} {} rational\n{}\nend'.format(
            len(self.eqns), ' '.join(str(i+1) for i in range(len(self.eqns))),
            len(self.eqns) + len(self.ieqs), self.ambient_dimension+1,
            '\n'.join(' '.join(str(x) for x in X) for X in self.eqns + self.ieqs),
            )
        
        args = [os.path.abspath(os.path.join('bin', 'cddexec_gmp')), '--rep']
        cddlib_proc = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE)
        ans, err = cddlib_proc.communicate(input=cddlib_input.encode())
        ret_code = cddlib_proc.poll()
        
        def parse(x):
            n, _, d = x.partition('/')
            return Fraction(int(n), int(d) if d else 1)
        
        return [[parse(item) for item in line.split()[1:]] for line in ans.decode().splitlines()[4:-1]]
    
    def bounding_box(self):
        return list(zip(*[(min(coords), max(coords)) for coords in zip(*self.vertices())]))
    
    def basic_bounding_box(self):
        non_negative = [ieq for ieq in self.ieqs if all(entry >= 0 for entry in ieq[1:])]
        non_positive = [ieq for ieq in self.ieqs if all(entry <= 0 for entry in ieq[1:])]
        return [max(-ieq[0] // ieq[i+1] for ieq in non_negative if ieq[i+1]) for i in range(self.ambient_dimension)], \
            [min(-ieq[0] // ieq[i+1] + 1 for ieq in non_positive if ieq[i+1]) for i in range(self.ambient_dimension)]

if __name__ == '__main__':
    P = Polyhedron([], [[6243, 55, 108], [310, -12, -143], [16, -43, 35]])
    print(P.integral_points_count())
    print(P.get_integral_point(100))

