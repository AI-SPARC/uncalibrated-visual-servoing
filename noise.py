from threading import Lock, Thread
from numpy.random import Generator, PCG64, default_rng
from numpy import zeros, sin, cos, tan, arctan, sqrt, log, pi
from enum import Enum
import logging

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
        elif type == cls.ALPHA_STABLE:
            num = 1
        return num

class NoiseProfiler():

    def __init__(self, num_features: int, noise_type: enumerate, seed: int = None, logger: object = None, **noise_params) -> None:
        self.num_features = num_features
        self.generators = []
        
        self.logger = logging.getLogger(__name__)
        if logger is not None:
            self.logger.setLevel(logger.level)
            for handler in logger.handlers:
                self.logger.addHandler(handler)

        self.noise_type = noise_type

        if "noise_params" in noise_params:
            noise_params = noise_params["noise_params"]
        
        if noise_type == NoiseType.WHITE_NOISE or noise_type == NoiseType.GAUSSIAN_MIXTURE or noise_type == NoiseType.GAUSSIAN_BIMODAL:
            self.std = noise_params['std']
        
            if noise_type == NoiseType.GAUSSIAN_MIXTURE or noise_type == NoiseType.GAUSSIAN_BIMODAL:
                self.mean = noise_params['mean']
                self.rho = noise_params['rho']

                self.rhoGenerators = []

                for i in range(num_features):
                    if seed is None:
                        self.rhoGenerators.append(default_rng())
                    else:
                        self.rhoGenerators.append(Generator(PCG64(2*seed+i)))
        elif noise_type == NoiseType.ALPHA_STABLE:
            self.alpha = noise_params['alpha']
            self.beta = noise_params['beta']
            self.gamma = noise_params['gamma']
            self.delta = noise_params['delta']

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
        '''
        STBLRND alpha-stable random number generator.
        R = getAlphaStable(ALPHA,BETA,GAMMA,DELTA) draws a sample from the Levy 
        alpha-stable distribution with characteristic exponent ALPHA, 
        skewness BETA, scale parameter GAMMA and location parameter DELTA.
        ALPHA,BETA,GAMMA and DELTA must be scalars which fall in the following 
        ranges :
           0 < ALPHA <= 2
           -1 <= BETA <= 1  
           0 < GAMMA < inf 
           -inf < DELTA < inf
        
        
        
        References:
        [1] J.M. Chambers, C.L. Mallows and B.W. Stuck (1976) 
            "A Method for Simulating Stable Random Variables"  
            JASA, Vol. 71, No. 354. pages 340-344  
        
        [2] Aleksander Weron and Rafal Weron (1995)
            "Computer Simulation of Levy alpha-Stable Variables and Processes" 
            Lec. Notes in Physics, 457, pages 379-392
        '''

        values = zeros(self.num_features)

        for i in range(self.num_features):
            # Checking special cases
            if self.alpha == 2: # Gaussian distribution
                values[i] = self.generators[i].normal(loc=0.0, scale=sqrt(2))
            elif self.alpha == 1 and self.beta == 0: # Cauchy distribution
                values[i] = tan(self.generators[i].uniform(low=-pi/2, high=pi/2)) # https://en.wikipedia.org/wiki/Cauchy_distribution#Generating_values_from_Cauchy_distribution
            elif self.alpha == 0.5 and abs(self.beta) == 1: # Levy distribution (a.k.a. Pearson V)
                values[i] = self.beta / (self.generators[i].normal(loc=0.0, scale=1.0)**2)
            else:
                # Alpha-stable cases
                V = self.generators[i].uniform(low=-pi/2, high=pi/2)
                W = -log(self.generators[i].uniform(low=0.0, high=1.0)) # You can test it with histogram(-log(rand(1000, 1))) and mean(log(rand(1000, 1)))
                if self.beta == 0: # Symmetric alpha-stable
                    values[i] = (sin(self.alpha * V) / (cos(V)**(1/self.alpha))) * (cos(V*(1-self.alpha))/W)**((1-self.alpha)/self.alpha)
                elif self.alpha != 1: # General case, alpha not 1
                    constant = self.beta * tan(pi*self.alpha/2)
                    B = arctan(constant)
                    S = (1 + constant**2)**(1/(2*self.alpha))
                    values[i] = S * sin(self.alpha*V + B) / (cos(V) ** (1/self.alpha)) * (cos((1 - self.alpha) * V - B) / W)**((1 - self.alpha) / self.alpha) # TODO: Check if it should be B * self.alpha
                else: # General case, alpha is 1
                    sclshftV = pi/2 + self.beta * V
                    values[i] = 2/pi * (sclshftV * tan(V) - self.beta * log((W * cos(V)) / sclshftV)) # WARNING: Check if that pi/2 inside log is correct 

            # Scale and shift
            if self.alpha == 1:
                values[i] = self.gamma * values[i] + (2/pi) * self.beta * self.gamma * log(self.gamma) + self.delta
            else:
                values[i] = self.gamma * values[i] + self.delta

        return values
