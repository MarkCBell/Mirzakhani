
import sys
import re
import pandas as pd
from glob import glob
from math import sqrt

EXTRACT = r'(?P<index>-?\d+): (?P<coordinate>\[[0-9, ]+\]), (?P<num_components>\d+), (?P<topo>.+)'
IS_PRIMITIVE = r'\(\[([0-9]*(, |\]))*, \[((\[([01]?(, |\]))*)(, |\]))*(, \[\[\](, \[\])*\])?\)'
F = re.compile(EXTRACT)
G = re.compile(IS_PRIMITIVE)

df = pd.DataFrame(
    [F.match(row).groups() for match in sys.argv[1:] for path in glob(match) for row in open(path, 'r') if F.match(row)],
    columns=['index', 'coordinate', 'num_components', 'topo']
    )

primitive = df[df.topo.str.match(G)]
curves = df[df.num_components == '1']  # Curve ==> primitive.

for ds in [primitive, curves]:
    histogram = ds.topo.value_counts()
    print(histogram)
    print(1.0 * histogram.max() / histogram.min())
    print('-' * 30)


print('For curves, the fraction that are non-separating are:')
# Not doing FPC.
# pT = pTs[0]  # Assume all the same for now.
# assert all(item == pT for item in pTs)
sT = 1.0 * len(curves)
sY = 1.0 * len(curves[curves.topo.apply(lambda x: x.split(']')[0].count(',') == 0)])
sN = sT - sY
sM = sY / sT
sSD = sqrt((sN * sM**2 + sY * (1 - sM)**2) / (sT - 1))

print('Mean: {}'.format(sM))
print('Mean reciprocal: {}'.format(1.0 / (1.0 - sM)))
print('SD: {}'.format(sSD))

for confidence, Z_alpha in [('95', 1.96), ('99', 2.576), ('99.9', 3.291)]:
    d = Z_alpha * sSD / sqrt(sT)  # * sqrt((pT - sT) / (pT - 1))  # Ignore FPC.

    print('\t{}% Range: {} -- {}'.format(confidence, sM - d, sM + d))
    print('\t{}% Range reciprocal: {} -- {}'.format(confidence, 1.0 / (1.0 - sM + d), 1.0 / (1.0 - sM - d)))

