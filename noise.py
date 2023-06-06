from threading import Lock, Thread
from numpy.random import Generator, PCG64, default_rng
from numpy import zeros, sin, cos, tan, arctan, sqrt, log, pi, any
from enum import Enum
import logging

class NoiseType(Enum):
    WHITE_NOISE = 1
    GAUSSIAN_MIXTURE = 2
    GAUSSIAN_BIMODAL = 3
    ALPHA_STABLE = 4
    UNIFORM = 5

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
        elif type == cls.UNIFORM:
            num = 1
        return num

class NoiseProfiler():

    def __init__(self, num_features: int, noise_type: enumerate, seed: int = None, logger: object = None, noise_hold: bool = False, noise_hold_cnt: int = 0, **noise_params) -> None:
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
                self.generators.append(Generator(PCG64(seed+10*i)))
        
        if noise_hold:
            self.noise_hold_cnt_max_predefined = noise_hold_cnt
        else:
            self.noise_hold_cnt_max_predefined = 0
        self.noise_hold_cnt = zeros(int(self.num_features/2))
        self.noise_hold_cnt_max = zeros(int(self.num_features/2))

        self.noise = zeros(self.num_features)

    def getNoise(self) -> list:
        for i in range(int(self.num_features/2)):
            if self.noise_hold_cnt[i] >= self.noise_hold_cnt_max[i]:
                self.noise_hold_cnt[i] = 0
            
                # Treat features as pairs
                noise_pair = zeros(2)
                outlier_detect = False
                for j in range(2):
                    # Get new noise values
                    if self.noise_type == NoiseType.WHITE_NOISE:
                        noise_pair[j] = self.__getWhiteNoise(2*i + j)
                    elif self.noise_type == NoiseType.GAUSSIAN_MIXTURE:
                        noise_pair[j] = self.__getGaussianMixture(2*i + j)
                    elif self.noise_type == NoiseType.GAUSSIAN_BIMODAL:
                        noise_pair[j] = self.__getBimodalGaussianMixture(2*i + j)
                    elif self.noise_type == NoiseType.ALPHA_STABLE:
                        noise_pair[j] = self.__getAlphaStable(2*i + j)
                    elif self.noise_type == NoiseType.UNIFORM:
                        noise_pair[j] = self.__getUniform(2*i + j)
                    
                    # Hold only when one of the features presents impulsive noise
                    if abs(noise_pair[j]) > 20:
                        outlier_detect = True

                if outlier_detect:
                    self.noise_hold_cnt_max[i] = self.noise_hold_cnt_max_predefined
                else:
                    self.noise_hold_cnt_max[i] = 0

                # Update noise value for feature pair
                self.noise[2*i] = noise_pair[0]
                self.noise[2*i + 1] = noise_pair[1]
            else:
                # Hold noise for a moment to simulate realistic computer vision problems, only increment counter
                self.noise_hold_cnt[i] += 1

        return self.noise
    
    def __getUniform(self, idx) -> float:
        value = 0
        value = self.generators[idx].uniform()
        return value
    
    def __getWhiteNoise(self, idx) -> float:
        value = 0
        value = self.generators[idx].normal(loc=0.0, scale=self.std)
        return value
    
    def __getGaussianMixture(self, idx):
        value = 0
        rho = self.rhoGenerators[idx].uniform(low=0, high=1)
        if rho > self.rho:
            value = self.generators[idx].normal(loc=0.0, scale=self.std) # Use white noise gaussian
        else:
            value = self.generators[idx + self.num_features].normal(loc=self.mean, scale=self.std) # use displaced gaussian
        
        return value
    
    def __getBimodalGaussianMixture(self, idx) -> float:
        value = 0
        rho = self.rhoGenerators[idx].uniform(low=0, high=1)
        if rho > self.rho:
            value = self.generators[idx].normal(loc=0.0, scale=self.std) # Use white noise gaussian
        elif rho > self.rho/2:
            value = self.generators[idx + self.num_features].normal(loc=self.mean, scale=self.std) # use displaced gaussian
        else:
            value = self.generators[idx + 2*self.num_features].normal(loc=-self.mean, scale=self.std) # use negative mean displaced gaussian

        return value
    
    def __getAlphaStable(self, idx) -> float:
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

        value = 0

        # Checking special cases
        if self.alpha == 2: # Gaussian distribution
            value = self.generators[idx].normal(loc=0.0, scale=sqrt(2))
        elif self.alpha == 1 and self.beta == 0: # Cauchy distribution
            value = tan(self.generators[idx].uniform(low=-pi/2, high=pi/2)) # https://en.wikipedia.org/wiki/Cauchy_distribution#Generating_valueom_Cauchy_distribution
        elif self.alpha == 0.5 and abs(self.beta) == 1: # Levy distribution (a.k.a. Pearson V)
            value = self.beta / (self.generators[idx].normal(loc=0.0, scale=1.0)**2)
        else:
            # Alpha-stable cases
            V = self.generators[idx].uniform(low=-pi/2, high=pi/2)
            W = -log(self.generators[idx].uniform(low=0.0, high=1.0)) # You can test it with histogram(-log(rand(1000, 1))) and mean(log(rand(1000, 1)))
            if self.beta == 0: # Symmetric alpha-stable
                value = (sin(self.alpha * V) / (cos(V)**(1/self.alpha))) * (cos(V*(1-self.alpha))/W)**((1-self.alpha)/self.alpha)
            elif self.alpha != 1: # General case, alpha not 1
                constant = self.beta * tan(pi*self.alpha/2)
                B = arctan(constant)
                S = (1 + constant**2)**(1/(2*self.alpha))
                value = S * sin(self.alpha*V + B) / (cos(V) ** (1/self.alpha)) * (cos((1 - self.alpha) * V - B) / W)**((1 - self.alpha) / self.alpha) # TODO: Check if it should be B * self.alpha
            else: # General case, alpha is 1
                sclshftV = pi/2 + self.beta * V
                value = 2/pi * (sclshftV * tan(V) - self.beta * log((W * cos(V)) / sclshftV)) # WARNING: Check if that pi/2 inside log is correct 

        # Scale and shift
        if self.alpha == 1:
            value = self.gamma * value + (2/pi) * self.beta * self.gamma * log(self.gamma) + self.delta
        else:
            value = self.gamma * value + self.delta

        return value
