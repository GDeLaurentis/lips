#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Giuseppe

from __future__ import print_function
from __future__ import unicode_literals

import sys
import math
import time
import functools
import multiprocessing
import multiprocessing.pool


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class Progress(object):

    def __init__(self, maximum):
        self.manager = multiprocessing.Manager()
        self.time = self.manager.Value('f', 0)
        self.average_time = self.manager.Value('f', 0)
        self.current = self.manager.Value('i', -1)
        self.maximum = maximum
        self.hours = self.manager.Value('i', 0)
        self.minutes = self.manager.Value('i', 0)
        self.seconds = self.manager.Value('i', 0)

    def increment(self):
        self.current.value += 1

    def decrement(self):
        self.current.value -= 1

    def write(self):
        if self.current.value >= 1:
            previous_percentage = float(math.ceil(float(self.current.value - 1) / self.maximum * 10000)) / 100
        current_percentage = float(math.ceil(float(self.current.value) / self.maximum * 10000)) / 100
        previous_time = self.time.value
        self.time.value = time.time()
        if current_percentage != 0:
            msg = "{0:.2f}% completed.".format(current_percentage)
            left_percentage = 100 - current_percentage
            step_percentage = current_percentage - previous_percentage
            if step_percentage == 0:   # not pretty but avoids division by zero
                step_percentage = 0.01
            nbr_left_steps = left_percentage / step_percentage
            self.average_time.value = (
                (self.average_time.value * (self.current.value - 1) +
                 (self.time.value - previous_time)) / self.current.value)
            seconds = int((self.average_time.value) * nbr_left_steps)
            minutes, seconds = divmod(seconds, 60)
            hours, minutes = divmod(minutes, 60)
            delta_time = ((hours - self.hours.value) * 3600 +
                          (minutes - self.minutes.value) * 60 +
                          (seconds - self.seconds.value))
            if delta_time > 5 or delta_time < 0:
                ETA = "%d:%02d:%02d" % (hours, minutes, seconds)
                self.hours.value = hours
                self.minutes.value = minutes
                self.seconds.value = seconds
            else:
                ETA = "%d:%02d:%02d" % (self.hours.value, self.minutes.value,
                                        self.seconds.value)
            msg += " ETA: {}.".format(ETA)
        else:
            msg = "{0:.2f}% completed.".format(current_percentage)
        return msg


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


try:
    from pytest_cov.embed import cleanup_on_sigterm
except ImportError:
    pass
else:
    cleanup_on_sigterm()


class NoDaemonProcess(multiprocessing.Process):

    def _get_daemon(self):  # make 'daemon' attribute always return False
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class NoDaemonProcessPool(multiprocessing.pool.Pool):
    # see https://stackoverflow.com/questions/52948447/error-group-argument-must-be-none-for-now-in-multiprocessing-pool
    def Process(self, *args, **kwargs):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwargs)
        proc.__class__ = NoDaemonProcess
        return proc


class MyPool(object):      # context manager pool

    def __init__(self, processes=1, initializer=None, initargs=None):
        self.processes = processes
        self.initializer = initializer
        self.initargs = initargs

    def __enter__(self):
        self.obj = NoDaemonProcessPool(self.processes, self.initializer, self.initargs)
        return self.obj

    def __exit__(self, exc_type, exc_value, traceback):
        self.obj.close()
        self.obj.join()
        self.obj.terminate()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def progress_wrapper(func, *args, **kwargs):
    global lock, prog
    with lock if lock is not None else nullcontext():
        prog.increment()
        if type(args[-1]) in [list, tuple] and isinstance(args[-1][0], str) and len(", ".join(args[-1])) < 40:
            print("\r{} {} working on {}.                                   ".format(func.func.__name__ if hasattr(func.func, "__name__") else "function",
                                                                                     prog.write(), ", ".join(args[-1])), end="\r")
        elif len(str(args[-1]).replace("\n", "")) < 40:
            print("\r{} {} working on {}.                                   ".format(func.func.__name__ if hasattr(func.func, "__name__") else "function",
                                                                                     prog.write(), str(args[-1]).replace("\n", "")), end="\r")
        else:
            print("\r{} {}.                                                 ".format(func.func.__name__ if hasattr(func.func, "__name__") else "function",
                                                                                     prog.write()), end="\r")
        sys.stdout.flush()
    return func(*args, **kwargs)


_lambda_compatible_func = None


def worker(x):
    return _lambda_compatible_func(x)


def mapThreads(func, *args, **kwargs):
    UseParallelisation, Cores, verbose = kwargs.pop("UseParallelisation", True), kwargs.pop("Cores", 6), kwargs.pop("verbose", True)
    _func_partial = functools.partial(func, *args[:-1], **kwargs)

    def _init(l, p, func,):
        global lock, prog, _lambda_compatible_func
        lock = l
        prog = p
        _lambda_compatible_func = func

    if verbose:
        func_partial = functools.partial(progress_wrapper, _func_partial)
    else:
        func_partial = _func_partial

    if UseParallelisation is True:
        l = multiprocessing.Lock()
        p = Progress(len(args[-1]))
        with MyPool(Cores, initializer=_init, initargs=(l, p, func_partial,)) as pool:
            results = pool.map(worker, args[-1])
    else:
        global prog, lock
        lock = None
        prog = Progress(len(args[-1]))
        results = list(map(func_partial, args[-1]))

    if verbose:
        print("\r                                                                                                                   ", end="\r")
        sys.stdout.flush()
    return results


def filterThreads(lambda_func, iterable):
    lambda_func.__name__ = str("filterThreads")
    TrueOrFalseList = mapThreads(lambda_func, iterable)
    iterable = [entry for i, entry in enumerate(iterable) if TrueOrFalseList[i] is True]
    return iterable


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


class nullcontext(object):

    def __init__(self, enter_result=None):
        self.enter_result = enter_result

    def __enter__(self):
        return self.enter_result

    def __exit__(self, *excinfo):
        pass


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def retry(ExceptionsToCheck, max_tries=2, silent=False):
    def deco_retry(func):
        @functools.wraps(func)
        def f_retry(*args, **kwargs):
            current_try = 0
            while current_try < max_tries:
                try:
                    return_value = func(*args, **kwargs)
                except ExceptionsToCheck as e:
                    last_exception = e
                    if silent is False:
                        print("\r{}: {}...  ".format(type(e).__name__, str(e)[:75]), end="")
                        if type(args[-1]) in [list, tuple]:
                            print(", ".join(args[-1]), end="")
                        else:
                            print("Last arg: {}".format(args[-1]), end="")
                    current_try += 1
                else:
                    if current_try != 0 and silent is False:
                        print("\r{}: {}... ".format(type(last_exception).__name__, str(last_exception)[:75]), end="")
                        if type(args[-1]) in [list, tuple]:
                            print(", ".join(args[-1]), end="")
                        else:
                            print("Last arg: {}".format(args[-1]), end="")
                        print(" ~ Corrected on try {}!                                                               ".format(current_try + 1))
                    return return_value
            if silent is False:
                print("\r{}: {}... ".format(type(last_exception).__name__, str(last_exception)[:75]), end="")
                if type(args[-1]) in [list, tuple]:
                    print(", ".join(args[-1]), end="")
                else:
                    print("Last arg: {}".format(args[-1]), end="")
                print(" ~ Uncorrected after {} tries. :(                                                           ".format(max_tries))
            return None
        return f_retry
    return deco_retry
