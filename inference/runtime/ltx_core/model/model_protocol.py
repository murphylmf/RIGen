from typing import Protocol, TypeVar

ModelType = TypeVar("ModelType")


class ModelConfigurator(Protocol[ModelType]):
    ''

    @classmethod
    def from_config(cls, config: dict) -> ModelType: ...
