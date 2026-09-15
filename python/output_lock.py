"""Fail closed if two local/Slurm processes try to write the same result stream."""
from pathlib import Path
from functools import wraps
import inspect
import fcntl

def exclusive_output(filename):
    def decorate(function):
        signature=inspect.signature(function)
        @wraps(function)
        def wrapped(*args,**kwargs):
            arguments=signature.bind_partial(*args,**kwargs).arguments
            out=Path(arguments['out']);out.mkdir(parents=True,exist_ok=True)
            with (out/filename).open('a') as lock:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(f'Another process owns {out}/{filename}; refusing concurrent output writes') from error
                return function(*args,**kwargs)
        return wrapped
    return decorate
