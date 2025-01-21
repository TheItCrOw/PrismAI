class ModelIntializationError(Exception):
    pass


class DataClassMappingMixin:
    def keys(self):
        return self.__dataclass_fields__.keys()

    def __getitem__(self, key):
        if hasattr(self, key):
            value = getattr(self, key)
            if isinstance(value, DataClassMappingMixin):
                return dict(**value)
            return value
        raise KeyError(f"Key {key} not found in {type(self).__name__}")
