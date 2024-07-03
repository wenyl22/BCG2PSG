import importlib
import typing as t


_T = t.TypeVar("_T")


def get_class(base: t.Type[_T], name: str) -> t.Type[_T]:
    module, cls_name = name.rsplit(".", 1)
    cls = getattr(importlib.import_module(module), cls_name)
    assert issubclass(cls, base), f"{cls} is not a subclass of {base}"
    return cls


def create_instance(
    base: t.Type[_T], config: dict[str, t.Any], *args: t.Any, **default_kwargs: t.Any
) -> _T:
    config = config.copy()
    cls = get_class(base, config.pop("class"))
    kwargs = {**default_kwargs, **config}
    return cls(*args, **kwargs)
