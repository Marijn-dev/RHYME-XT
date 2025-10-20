import numpy as np


class InitialStateGenerator:

    def __init__(self, rng: np.random.Generator = None):
        self._rng = rng if rng else np.random.default_rng()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def sample(self):
        return self._sample_impl()


class GaussianInitialState(InitialStateGenerator):

    def __init__(self, n, rng: np.random.Generator = None):
        super().__init__(rng)
        self.n = n

    def _sample_impl(self):
        return self._rng.standard_normal(size=self.n)


class UniformInitialState(InitialStateGenerator):

    def __init__(self, n, rng: np.random.Generator = None):
        super().__init__(rng)
        self.n = n

    def _sample_impl(self):
        return self._rng.uniform(size=self.n)


class HHFSInitialState(InitialStateGenerator):

    def __init__(self, rng: np.random.Generator = None):
        super().__init__(rng)

    def _sample_impl(self):
        x0 = self._rng.uniform(size=(4, ))
        x0[0] = 2. * x0[0] - 1.

        return x0


class HHRSAInitialState(InitialStateGenerator):

    def __init__(self, rng: np.random.Generator = None):
        super().__init__(rng)

    def _sample_impl(self):
        x0 = self._rng.uniform(size=(5, ))
        x0[0] = 2. * x0[0] - 1.

        return x0


class HHFFEInitialState(InitialStateGenerator):

    def __init__(self, rng: np.random.Generator = None):
        super().__init__(rng)

    def _sample_impl(self):
        x0 = self._rng.uniform(size=(10, ))
        x0[0] = 2. * x0[0] - 1.
        x0[5] = 2. * x0[5] - 1.

        return x0


class HHFBEInitialState(InitialStateGenerator):

    def __init__(self, rng: np.random.Generator = None):
        super().__init__(rng)

    def _sample_impl(self):
        x0 = self._rng.uniform(size=(11, ))
        x0[0] = 2. * x0[0] - 1.
        x0[5] = 2. * x0[5] - 1.

        return x0


class HHIBInitialState(InitialStateGenerator):

    def __init__(self, rng: np.random.Generator = None):
        super().__init__(rng)

    def _sample_impl(self):
        x0 = self._rng.uniform(size=(7, ))
        x0[0] = 2. * x0[0] - 1.

        return x0


class GreenshieldsInitialState(InitialStateGenerator):

    def __init__(self, n_cells, n_sections, rng: np.random.Generator = None):
        super().__init__(rng)

        self.n_cells = n_cells
        self.n_sec = n_sections
        self.sec_size = self.n_cells // self.n_sec

    def _sample_impl(self):
        x0_vals = self._rng.uniform(0., 0.5, size=(self.n_sec, ))
        x0 = np.empty((self.n_cells, ))
        x0[0:self.sec_size * self.n_sec] = np.repeat(x0_vals, self.sec_size)
        x0[self.sec_size * self.n_sec:-1] = x0[self.sec_size * self.n_sec - 1]

        return x0

class HeatInitialState(InitialStateGenerator):
    def __init__(self, n, L, rng: np.random.Generator = None):
        super().__init__(rng)
        self.size = n
        self.L = L
        self.x_points = np.linspace(0,L,n)
        self.sigma = L/15

    def _sample_impl(self):
        firstpeak = np.random.uniform(0.1*self.L, 0.9*self.L)
        secondpeak = np.random.uniform(0.1*self.L, 0.9*self.L)
        x0 = 5*np.exp(-(self.x_points - firstpeak)**2 / (2 * self.sigma**2))
        x0 += 5*np.exp(-(self.x_points - secondpeak+2)**2 / (2 * self.sigma**2))
        
        return x0

class AmariInitialState(InitialStateGenerator):
    def __init__(self,x_lim,dx,rng: np.random.Generator = None):
        '''returns empty array'''
        super().__init__(rng)
        self.n = int(np.round(x_lim/dx)) * 2+1
        self.x = np.arange(-x_lim,x_lim+dx,dx)

    def _sample_impl(self):
        # x0 = np.zeros(self.n)
        x0 = 0.3 * np.exp(-(self.x**2) / (0.05**2))  # Small bump in center
        x0 = 0.05 * np.random.randn(self.n)  # Small random fluctuations
        x0 = 1.5 * np.exp(-(self.x**2) / (1**2))  # above threshold
        x0 += 1.5*np.exp(-(self.x - self.x[-1])**2 / 0.01)  # bump near edge
        x0 += 1.5*np.exp(-(self.x - self.x[0])**2 / 0.01)  # bump near edge

        return x0
        
class AmariCoupledInitialState(InitialStateGenerator):
    def __init__(self,x_lim,dx,theta,rng: np.random.Generator = None):
        '''returns empty array'''
        super().__init__(rng)
        self.n = int(np.round(x_lim/dx)) + 1
        self.theta = theta

    def _sample_impl(self):
        u0 = -self.theta*np.ones(self.n)
        K = 0
        v0 = K - u0
        return np.stack((u0,v0),axis=-1)
    
class AmariCoupledFHNInitialState(InitialStateGenerator):
    def __init__(self,x_lim,dx,theta,rng: np.random.Generator = None):
        '''returns empty array'''
        super().__init__(rng)
        self.n = int(np.round(x_lim/dx)) + 1
        self.theta = theta
        
    def _sample_impl(self):
        u0 = np.zeros(self.n)
        K = 0
        v0 = K - u0
        return np.stack((u0,v0),axis=-1)

class LifInitialState(InitialStateGenerator):
    def __init__(self,bumps,N,rng:np.random.Generator = None):
        '''returns empty array'''
        super().__init__(rng)
        self.N = N
        self.bumps = bumps


    def _sample_impl(self):
        x = np.arange(self.N)
        v0 = np.zeros(self.N)

        for _ in range(self.bumps):
            width = self._rng.uniform(5, 15)                    # random std dev with fixed intervals
            center = self._rng.integers(0, self.N)              # random center
            distance = np.minimum(
                np.abs(x - center),
                self.N - np.abs(x - center)
                )

            v0 += 0.45 * np.exp(-0.5 * (distance / width) ** 2)

        return v0
    
_initstategen_names = {
    "GaussianInitialState": GaussianInitialState,
    "UniformInitialState": UniformInitialState,
    "HHFSInitialState": HHFSInitialState,
    "HHRSAInitialState": HHRSAInitialState,
    "HHIBInitialState": HHIBInitialState,
    "HHFFEInitialState": HHFFEInitialState,
    "HHFBEInitialState": HHFBEInitialState,
    "GreenshieldsInitialState": GreenshieldsInitialState,
    "HeatInitialState": HeatInitialState,
    "AmariInitialState": AmariInitialState,
    "AmariCoupledInitialState":AmariCoupledInitialState,
    "AmariCoupledFHNInitialState":AmariCoupledFHNInitialState,
    "LifInitialState": LifInitialState,
}


def get_initial_state_generator(name, args):
    return _initstategen_names[name](**args)
