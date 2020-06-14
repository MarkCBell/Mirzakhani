
import sys
from itertools import chain, cycle
import multiprocessing as mp

TOKEN = object()

def helper(input_queue, output_queue, common, function):
    while True:
        data = input_queue.get()
        # common.update(data)
        # result = function(**common)
        result = function(**common, **data)
        output_queue.put(result)

def process(function, common, iterable, cores=1, path=None):
    if cores <= 0: cores = mp.cpu_count()
    iterable = iter(iterable)
    
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    
    with (open(path, 'w') if path is not None else sys.stdout) as F:
        with mp.Pool(cores, initializer=helper, initargs=(input_queue, output_queue, common, function)):
            for data, do_output in zip(chain(iterable, (TOKEN for _ in range(cores))), chain((False for _ in range(cores)), cycle([True]))):
                if data is not TOKEN:
                    input_queue.put(data)
                if do_output:
                    print(output_queue.get(), file=F)

