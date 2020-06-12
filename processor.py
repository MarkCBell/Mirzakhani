
import sys
from itertools import chain, cycle
import multiprocessing as mp

def helper(input_queue, output_queue, setup, function):
    while True:
        data = input_queue.get()
        # setup.update(data)
        # result = function(**setup)
        result = function(**setup, **data)
        output_queue.put(result)

def process(setup, function, datum, cores=1, path=None):
    if cores <= 0: cores = mp.cpu_count()
    datum = iter(datum)
    
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    
    with (open(path, 'w') if path is not None else sys.stdout) as F:
        with mp.Pool(cores, initializer=helper, initargs=(input_queue, output_queue, setup, function)) as pool:
            for data, do_output in zip(chain(datum, (None for _ in range(cores))), chain((False for _ in range(cores)), cycle([True]))):
                if data is not None:
                    input_queue.put(data)
                if do_output:
                    result = output_queue.get()
                    print(result, file=F)

