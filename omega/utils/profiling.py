import time


_ENABLED = False


def enable_profiling():
    global _ENABLED
    _ENABLED = True


def disable_profiling():
    global _ENABLED
    _ENABLED = False


def timeit(method):
    def timed(*args, **kw):
        method_name = method.__name__

        if _ENABLED:
            print(f'{method_name} started')

        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if _ENABLED:
            print(f'{method_name} took {(te - ts) * 1000} ms')

        return result

    return timed
