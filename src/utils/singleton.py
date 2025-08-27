class Singleton(type):
    """
    Metaclass for implementing the Singleton pattern.
    Ensures only one instance of a class is created.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        # If an instance doesn't exist, create one
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]