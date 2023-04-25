from threading import Lock, Thread
from numpy.random import Generator, PCG64
from numpy import zeros

class SingletonMeta(type):
    """
    This is a thread-safe implementation of Singleton.
    https://refactoring.guru/design-patterns/singleton/python/example#example-1
    """

    _instances = {}

    _lock: Lock = Lock()
    """
    We now have a lock object that will be used to synchronize threads during
    first access to the Singleton.
    """

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        # Now, imagine that the program has just been launched. Since there's no
        # Singleton instance yet, multiple threads can simultaneously pass the
        # previous conditional and reach this point almost at the same time. The
        # first of them will acquire lock and will proceed further, while the
        # rest will wait here.
        with cls._lock:
            # The first thread to acquire the lock, reaches this conditional,
            # goes inside and creates the Singleton instance. Once it leaves the
            # lock block, a thread that might have been waiting for the lock
            # release may then enter this section. But since the Singleton field
            # is already initialized, the thread won't create a new object.
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class NoiseGenerator(metaclass=SingletonMeta):
    num_features: int = None
    generators: list = None

    mean: float = 5
    std: float = 1
    rho: float = 0.1

    def __init__(self, num_features: int, mean: float = None, std: float = None, rho: float = None) -> None:
        self.num_features = num_features
        self.generators = []

        if mean is not None:
            self.mean = mean
        if std is not None:
            self.std = std
        if rho is not None:
            self.rho = rho

        for i in range(num_features):
            self.generators.append(Generator(PCG64(12345+i)))

    def getWhiteNoise(self) -> list:
        values = zeros(self.num_features)
        for i in range(self.num_features):
            values[i] = self.generators[i].normal(loc=0.0, scale=self.std)
        
        return values
    
    def getGaussianMixture(self) -> list:
        values = zeros(self.num_features)
        for i in range(self.num_features):
            values[i] = (1 - self.rho) * self.generators[i].normal(loc=0.0, scale=self.std) + self.rho * self.generators[i + self.num_features].normal(loc=self.mean, scale=self.std)
        
        return values
