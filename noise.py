from threading import Lock, Thread
from numpy.random import Generator, PCG64, default_rng
from numpy import zeros
from enum import Enum

class NoiseType(Enum):
    WHITE_NOISE = 1
    GAUSSIAN_MIXTURE = 2
    GAUSSIAN_BIMODAL = 3
    ALPHA_STABLE = 4

    @classmethod
    def numberOfGenerators(cls, type) -> int:
        num = 0
        if type == cls.WHITE_NOISE:
            num = 1
        elif type == cls.GAUSSIAN_MIXTURE:
            num = 2
        elif type == cls.GAUSSIAN_BIMODAL:
            num = 3
        return num

class NoiseProfiler():

    def __init__(self, num_features: int, noise_type: enumerate, seed: int = None, **noise_params) -> None:
        self.num_features = num_features
        self.generators = []
        
        self.noise_type = noise_type
        
        if noise_type == NoiseType.WHITE_NOISE or noise_type == NoiseType.GAUSSIAN_MIXTURE or noise_type == NoiseType.GAUSSIAN_BIMODAL:
            noise_params = noise_params["noise_params"]
            self.std = noise_params['std'] if "std" in noise_params else 0.25
        
            if noise_type == NoiseType.GAUSSIAN_MIXTURE or noise_type == NoiseType.GAUSSIAN_BIMODAL:
                self.mean = noise_params['mean'] if "mean" in noise_params else 5
                self.rho = noise_params['rho'] if "rho" in noise_params else 0.2

                self.rhoGenerators = []

                for i in range(num_features):
                    if seed is None:
                        self.rhoGenerators.append(default_rng())    
                    else:
                        self.rhoGenerators.append(Generator(PCG64(2*seed+i)))

        for i in range(NoiseType.numberOfGenerators(noise_type) * num_features):
            if seed is None:
                self.generators.append(default_rng())    
            else:
                self.generators.append(Generator(PCG64(seed+i)))


    def getNoise(self) -> list :

        if self.noise_type == NoiseType.WHITE_NOISE:
            values = self.getWhiteNoise()
        elif self.noise_type == NoiseType.GAUSSIAN_MIXTURE:
            values = self.getGaussianMixture()
        elif self.noise_type == NoiseType.GAUSSIAN_BIMODAL:
            values = self.getBimodalGaussianMixture()
        elif self.noise_type == NoiseType.ALPHA_STABLE:
            values = self.getAlphaStable()

        return values

    def getWhiteNoise(self) -> list:
        values = zeros(self.num_features)
        for i in range(self.num_features):
            values[i] = self.generators[i].normal(loc=0.0, scale=self.std)
        
        return values
    
    def getGaussianMixture(self) -> list:
        values = zeros(self.num_features)
        for i in range(self.num_features):
            rho = self.rhoGenerators[i].uniform(low=0, high=1)
            if rho > self.rho:
                values[i] = self.generators[i].normal(loc=0.0, scale=self.std) # Use white noise gaussian
            else:
                values[i] = self.generators[i + self.num_features].normal(loc=self.mean, scale=self.std) # use displaced gaussian
        
        return values
    
    def getBimodalGaussianMixture(self) -> list:
        values = zeros(self.num_features)
        for i in range(self.num_features):
            rho = self.rhoGenerators[i].uniform(low=0, high=1)
            if rho > self.rho:
                values[i] = self.generators[i].normal(loc=0.0, scale=self.std) # Use white noise gaussian
            elif rho > self.rho/2:
                values[i] = self.generators[i + self.num_features].normal(loc=self.mean, scale=self.std) # use displaced gaussian
            else:
                values[i] = self.generators[i + 2*self.num_features].normal(loc=-self.mean, scale=self.std) # use negative mean displaced gaussian

        return values
    
    def getAlphaStable(self) -> list:
        values = zeros(self.num_features)
        return values
