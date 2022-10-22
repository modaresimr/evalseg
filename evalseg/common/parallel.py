import multiprocessing
from functools import partial

import numpy as np
from tqdm.auto import tqdm


def _parallel_run_in_chunk(chunks, runner):
    return [runner(c) for c in chunks]


def parallel_runner(
    runner,
    items,
    *,
    max_cpu=0,
    parallel=True,
    max_threads=1000000,
    silent=False
):
    generetor = _parallel_runner(
        runner,
        items,
        max_cpu=max_cpu,
        parallel=parallel,
        max_threads=max_threads,
    )
    return tqdm(generetor, total=len(items), disable=silent)


def _parallel_runner(
    runner, items, *, max_cpu=0, parallel=True, max_threads=1000000
):
    max_cpu = multiprocessing.cpu_count() if max_cpu <= 0 else max_cpu
    if parallel and max_cpu > 1:

        spls = np.array_split(range(len(items)), max_threads)
        chunks = [items[spl[0] : spl[-1] + 1] for spl in spls if len(spl)]
        pool = multiprocessing.Pool(max_cpu, maxtasksperchild=20)

        result = pool.imap(
            partial(_parallel_run_in_chunk, runner=runner), chunks
        )
        try:
            for c in chunks:
                for i, r in enumerate(result.next()):
                    yield c[i], r
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            pool.close()
            raise
    else:
        for item in items:
            res = runner(item)
            yield item, res
