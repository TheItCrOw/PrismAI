from typing import Callable, Concatenate


def docs[**P, T](wrapper: Callable[P, T]):
    def decorator(
        func: Callable[..., T],
    ) -> Callable[Concatenate[..., P], T]:
        func.__doc__ = wrapper.__doc__
        func.__annotations__ |= wrapper.__annotations__
        return func

    return decorator
