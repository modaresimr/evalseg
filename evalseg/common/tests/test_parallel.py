import random
import time

import eval_seg


def _nothing(item):
    time.sleep(1 + random.random() * 2)
    return item


def test_parallel():
    items = range(100)
    res = list(
        eval_seg.utils.parallel_runner(
            _nothing, items, max_chunks=10, silent=False
        )
    )

    for i in range(len(items)):
        assert items[i] == res[i]
