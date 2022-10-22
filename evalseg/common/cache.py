import functools
import hashlib
from ast import Constant

import compress_pickle
import diskcache
import numpy as np

ENOVAL = Constant("ENOVAL")


class Cache:
    _instance = None
    directory = ".cache"
    expire = 7 * 24 * 60 * 60

    def __new__(cls, *args, **kwargs):
        if cls._instance == None:
            # print('Creating the cache')
            cls._instance = super(Cache, cls).__new__(cls, *args, **kwargs)
            cls._instance._cache = diskcache.Cache(directory=cls.directory)
        return cls._instance

    @classmethod
    def clear(cls):
        cls()._cache.clear()

    @classmethod
    def memoize(cls, name=None, typed=False, expire=None, tag=None, ignore=()):
        def decorator(func):
            base = diskcache.core.full_name(func) if name is None else name

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                key = wrapper.__cache_key__(*args, **kwargs)

                result = cls()._cache.get(key, default=ENOVAL, retry=True)

                if result is ENOVAL:
                    result = func(*args, **kwargs)
                    if expire is None or expire > 0:
                        cls()._cache.set(
                            key, result, expire, tag=tag, retry=True
                        )

                return result

            def __cache_key__(*args, **kwargs):
                return diskcache.core.args_to_key(
                    (base,), args, kwargs, typed, ignore
                )

            def clear_method_cache():
                keys = [c for c in cls()._cache if base in c[0]]
                for k in keys:
                    del cls()._cache[k]

            wrapper.__cache_key__ = __cache_key__
            wrapper.clear_method_cache = clear_method_cache

            return wrapper

        return decorator


def args_to_key(base, args, kwargs, typed, ignore, dohash=True):
    """Create cache key out of function arguments.
    :param tuple base: base of key
    :param tuple args: function arguments
    :param dict kwargs: function keyword arguments
    :param bool typed: include types in cache key
    :param set ignore: positional or keyword args to ignore
    :return: cache key tuple
    """
    args = tuple(arg for index, arg in enumerate(args) if index not in ignore)
    key = args + (None,)

    if kwargs:
        kwargs = {key: val for key, val in kwargs.items() if key not in ignore}
        sorted_items = sorted(kwargs.items())

        for item in sorted_items:
            key += item

    if typed:
        key += tuple(type(arg) for arg in args)

        if kwargs:
            key += tuple(type(value) for _, value in sorted_items)
    if dohash:
        return (base, hash(key))
    return (base,) + key
