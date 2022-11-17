import multiprocessing
from functools import partial

import numpy as np
from tqdm.auto import tqdm


def _chunk_run(chunks, runner):
    # print(chunks)
    res = [runner(**c) if type(c) == dict else runner(c)
           for c in chunks]
    return res


def parallel_runner(runner, items, *, max_cpu=0, parallel=True, max_threads=1000000, silent=False):
    generetor = _parallel_runner(runner, items, max_cpu=max_cpu, parallel=parallel, max_threads=max_threads,)
    return tqdm(generetor, total=len(items), disable=silent)


def _parallel_runner(runner, items, *, max_cpu=0, parallel=True, max_threads=1000000):
    max_cpu = multiprocessing.cpu_count() if max_cpu <= 0 else max_cpu
    if parallel and max_cpu > 1:

        spls = np.array_split(range(len(items)), max_threads)
        chunks = [items[spl[0]: spl[-1] + 1] for spl in spls if len(spl)]

        # pool = multiprocessing.Pool(max_cpu, maxtasksperchild=20)  # TODO: maxchunk
        with NoDaemonPool()(max_cpu, maxtasksperchild=20) as pool:
            result = pool.imap(partial(_chunk_run, runner=runner), chunks)
            try:
                # print(result)
                for c in chunks:
                    # print(c)
                    for i, r in enumerate(result.next()):
                        yield c[i], r
            except KeyboardInterrupt:
                # pool.terminate()
                # pool.join()
                # pool.close()
                print('Error')
                raise

    else:
        for item in items:
            res = _chunk_run([item], runner)
            yield item, res[0]
            # yield item, res


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.

def NoDaemonPool():
    class NoDaemonPool(multiprocessing.Pool().__class__):
        # Process = NoDaemonProcess

        @staticmethod
        def Process(ctx, *args, **kwds):
            return NoDaemonProcess(*args, **kwds)

    return NoDaemonPool
