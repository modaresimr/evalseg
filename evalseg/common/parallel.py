import multiprocessing
import ray.util.multiprocessing as raymlp
from functools import partial

import numpy as np
from tqdm.auto import tqdm
import traceback


def _chunk_run(chunks, runner):
    # print(chunks)
    res = [__run(c, runner) for c in chunks]
    return res


def __run(params, runner):
    # print('__run', params)
    try:
        if type(params) == dict:
            # print('dict')
            return runner(**params)
        if type(params) == list:
            # print('list')
            return runner(*params)
        # print('other    ')
        return runner(params)
    except Exception as e:
        # if isinstance(e, MemoryError):
        raise e
        # print(e)
        # traceback.print_exc()
        # return e


def parallel_runner(runner, items, max_cpu=0, maxtasksperchild=10, parallel=True, max_threads=1000000, silent=False):
    generetor = _parallel_runner(runner, items, max_cpu=max_cpu, maxtasksperchild=maxtasksperchild, parallel=parallel, max_threads=max_threads,)
    pbar = tqdm(generetor, total=len(items), disable=silent)
    # return pbar
    for i, x in pbar:
        pbar.set_postfix_str(f'{i}'[0:100])
        yield i, x


def _parallel_runner(runner, items, *, max_cpu=0, parallel=True, maxtasksperchild=10, max_threads=1000000):
    max_cpu = multiprocessing.cpu_count() if max_cpu <= 0 else max_cpu
    if parallel:

        # spls = np.array_split(range(len(items)), max_threads)
        # chunks = [items[spl[0]: spl[-1] + 1] for spl in spls if len(spl)]
        maxchunks = max((len(items)-1)//max_threads + 1, 1)

        # pool = multiprocessing.Pool(max_cpu, maxtasksperchild=20)  # TODO: maxchunk
        with NoDaemonPool(max_cpu, maxtasksperchild=maxtasksperchild) as pool:
            # with multiprocessing.Pool(max_cpu, maxtasksperchild=maxtasksperchild) as pool:
            # result = pool.imap(partial(_chunk_run, runner=runner), chunks)
            result = pool.imap(partial(__run, runner=runner), items, chunksize=maxchunks)
            try:
                # print(result)
                for c in items:
                    # print('c', c)
                    yield c, result.next()
                # for c in chunks:
                #     # print(c)
                #     for i, r in enumerate(result.next()):
                #         yield c[i], r
            except KeyboardInterrupt:
                # pool.terminate()
                # pool.join()
                # pool.close()
                print('Error')
                raise

    else:
        for item in items:
            res = __run(item, runner)
            yield item, res
        # for item in items:
        #     res = _chunk_run([item], runner)
        #     yield item, res[0]
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

def NoDaemonPool(*args, **kwargs):

    class NoDaemonPoolCLS(multiprocessing.Pool().__class__):
        # class NoDaemonPoolCLS(raymlp.Pool().__class__):
        # Process = NoDaemonProcess

        @staticmethod
        def Process(ctx, *args, **kwds):
            return NoDaemonProcess(*args, **kwds)

    return NoDaemonPoolCLS(*args, **kwargs)


def __test(arg1, arg2):
    import time
    time.sleep(arg1)
    print(f'arg1={arg1} ,arg2={arg2}')
    return f'arg1={arg1} ,arg2={arg2}'
