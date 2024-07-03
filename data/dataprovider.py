from typing import Generic, Iterable, TypeVar

_T = TypeVar("_T")


class DataProvider(Generic[_T]):
    @property
    def train(self) -> Iterable[_T]:
        raise NotImplementedError

    @property
    def val(self) -> Iterable[_T]:
        raise NotImplementedError

    @property
    def test(self) -> Iterable[_T]:
        raise NotImplementedError
