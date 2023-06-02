import inspect
import os
import sys
import torch
from typing import Callable
from make_tensor import map_dtype_to_str


def _instantiate_opinfo_test_template(
    template: Callable, scope, *, opinfo, dtype: torch.dtype
) -> Callable:
    """Instantiates a test template for an operator."""

    test_name = "_".join((template.__name__, opinfo.name, map_dtype_to_str[dtype]))

    def test():
        return template(opinfo, dtype)

    test.__name__ = test_name
    test.__module__ = test.__module__
    return test


class ops:
    def __init__(self, opinfos, *, scope=None):
        self.opinfos = opinfos

        # Acquires the caller's global scope
        if scope is None:
            previous_frame = inspect.currentframe().f_back
            scope = previous_frame.f_globals
        self.scope = scope

    def __call__(self, test_template):
        # NOTE Unlike a typical decorator, this __call__ does not return a function, because it may
        #   (and typically does) instantiate multiple functions from the template it consumes.
        #   Since Python doesn't natively support one-to-many function decorators, the produced
        #   functions are directly assigned to the requested scope (the caller's global scope by default)
        for opinfo in self.opinfos:
            for dtype in sorted(opinfo._dtypes, key=lambda t: repr(t)):
                test = _instantiate_opinfo_test_template(
                    test_template,
                    self.scope,
                    opinfo=opinfo,
                    dtype=dtype,
                )
                # Adds the instantiated test to the requested scope
                self.scope[test.__name__] = test


def run_snippet(snippet, opinfo, dtype, *args, **kwargs):
    try:
        snippet(*args, **kwargs)
    except Exception as e:
        exc_info = sys.exc_info()

        # Raises exceptions that occur with pytest, and returns debug information when
        # called otherwise
        # NOTE: PYTEST_CURRENT_TEST is set by pytest
        if "PYTEST_CURRENT_TEST" in os.environ:
            raise e
        return e, exc_info, snippet, opinfo, dtype, args, kwargs

    return None
