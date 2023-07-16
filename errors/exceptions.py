__all__ = [
    "DimensionException",
    "ShapeException",
    "DTypeException",
    "DeviceException",
    "InconsistentException"
]


class DimensionException(ValueError):
    def __init__(self, *args: object):
        super().__init__(*args)


class ShapeException(ValueError):
    def __init__(self, *args: object):
        super().__init__(*args)


class DTypeException(ValueError):
    def __init__(self, *args: object):
        super().__init__(*args)


class DeviceException(ValueError):
    def __init__(self, *args: object):
        super().__init__(*args)


class InconsistentException(AssertionError):
    def __init__(self, *args: object):
        super().__init__(*args)
