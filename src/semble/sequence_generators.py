import numpy as np
from brian2 import *

class SequenceGenerator:

    def __init__(self, dim, rng: np.random.Generator = None):
        self.dim = dim
        self._rng = rng if rng else np.random.default_rng()

    @property
    def name(self) -> str:
        return self.__class__.__name__

    def sample(self, time_range, delta):
        return self._sample_impl(time_range, delta)


class Product(SequenceGenerator):

    def __init__(self, seq_gens, rng: np.random.Generator = None):
        super().__init__(len(seq_gens), rng)
        self._seq_gens = seq_gens

    def _sample_impl(self, time_range, delta):
        samples = tuple(g.sample(time_range, delta) for g in self._seq_gens)

        return np.hstack(samples)
    
class Spatial(SequenceGenerator):

    def __init__(self, seq_gens, rng: np.random.Generator = None):
        super().__init__(len(seq_gens), rng)
        self._seq_gens = seq_gens

    def _sample_impl(self, time_range, delta):
        samples = tuple(g.sample(time_range, delta) for g in self._seq_gens)

        return np.hstack(samples)


class GaussianSequence(SequenceGenerator):

    def __init__(self, mean=0., std=1., dim=1, rng=None):
        super().__init__(dim, rng)

        self._mean = mean
        self._std = std

    def _sample_impl(self, time_range, delta):
        n_control_vals = int(1 +
                             np.floor((time_range[1] - time_range[0]) / delta))
        control_seq = self._rng.normal(loc=self._mean,
                                       scale=self._std,
                                       size=(n_control_vals, self.dim))

        return control_seq


class GaussianSqWave(SequenceGenerator):

    def __init__(self, period, mean=0., std=1., dim=1, rng=None):
        super().__init__(dim, rng)

        self._period = period
        self._mean = mean
        self._std = std

    def _sample_impl(self, time_range, delta):
        n_control_vals = int(1 +
                             np.floor((time_range[1] - time_range[0]) / delta))

        n_amplitude_vals = int(np.ceil(n_control_vals / self._period))

        amp_seq = self._rng.normal(loc=self._mean,
                                   scale=self._std,
                                   size=(n_amplitude_vals, self.dim))

        control_seq = np.repeat(amp_seq, self._period, axis=0)[:n_control_vals]
        return control_seq


class LogNormalSqWave(SequenceGenerator):

    def __init__(self, period, mean=0., std=1., dim=1, rng=None):
        super().__init__(dim, rng)

        self._period = period
        self._mean = mean
        self._std = std

    def _sample_impl(self, time_range, delta):
        n_control_vals = int(1 +
                             np.floor((time_range[1] - time_range[0]) / delta))

        n_amplitude_vals = int(np.ceil(n_control_vals / self._period))

        amp_seq = self._rng.lognormal(mean=self._mean,
                                      sigma=self._std,
                                      size=(n_amplitude_vals, self.dim))

        control_seq = np.repeat(amp_seq, self._period, axis=0)[:n_control_vals]

        return control_seq


class UniformSqWave(SequenceGenerator):

    def __init__(self, period, min=0., max=1., dim=1, rng=None):
        super().__init__(dim, rng)

        self._period = period
        self._min = min
        self._max = max

    def _sample_impl(self, time_range, delta):
        n_control_vals = int(1 +
                             np.floor((time_range[1] - time_range[0]) / delta))

        n_amplitude_vals = int(np.ceil(n_control_vals / self._period))

        amp_seq = self._rng.uniform(low=self._min,
                                    high=self._max,
                                    size=(n_amplitude_vals, self.dim))

        control_seq = np.repeat(amp_seq, self._period, axis=0)[:n_control_vals]

        return control_seq


class RandomWalkSequence(SequenceGenerator):

    def __init__(self, mean=0., std=1., dim=1, rng=None):
        super().__init__(dim, rng)

        self._mean = mean
        self._std = std

    def _sample_impl(self, time_range, delta):
        n_control_vals = int(1 +
                             np.floor((time_range[1] - time_range[0]) / delta))

        control_seq = np.cumsum(self._rng.normal(loc=self._mean,
                                                 scale=self._std,
                                                 size=(n_control_vals,
                                                       self.dim)),
                                axis=1)

        return control_seq


class SinusoidalSequence(SequenceGenerator):

    def __init__(self, max_freq=1.0, rng=None):
        super().__init__(1, rng)

        self._amp_mean = 1.0
        self._amp_std = 1.0
        self._mf = max_freq

    def _sample_impl(self, time_range, delta):
        amplitude = self._rng.lognormal(mean=self._amp_mean,
                                        sigma=self._amp_std)
        frequency = self._rng.uniform(0, self._mf)

        n_control_vals = int(1 +
                             np.floor((time_range[1] - time_range[0]) / delta))
        time = np.linspace(time_range[0], time_range[1], n_control_vals)

        return (amplitude * np.sin(np.pi * frequency / delta * time)).reshape(
            (-1, 1))

class Gaussian1D(SequenceGenerator):
    '''input used for the amari model, contains several guassians'''
    def __init__(self,x_lim,dx,duration,amplitude,std,difference):
        self.x_lim = x_lim
        self.dx = dx
        self.x = np.arange(-x_lim,x_lim+dx,dx)
        self.duration = duration # measured in deltas
        self.amplitude = amplitude
        self.std = std
        self.difference = difference
        
    def _sample_impl(self,time_range,delta):
        n_control_vals = int(1+np.floor((time_range[1] - time_range[0]) / delta))
        control_seq = np.zeros([n_control_vals, len(self.x)])

        durations = []
        onsets = []
        i = 0
        onsets.append(np.random.randint(1,5))
        
        # while in time
        while ((onsets[i]+self.duration[1])*delta) < time_range[1]:
            durations.append(np.random.randint(int(self.duration[0]),int(self.duration[1])))
            difference = np.random.randint(self.difference[0],self.difference[1]) # delta difference between inputs
            onsets.append(onsets[i]+durations[i]+difference)
            i += 1
        
        onsets.pop() # remove last value since new signal doesnt fit anymore
        onsets = np.array(onsets)
        durations = np.array(durations)

        amplitudes = np.random.uniform(self.amplitude[0], self.amplitude[1], len(onsets)) 
        sigmas = np.random.uniform(self.std[0], self.std[1], len(onsets)) 
        positions = np.random.uniform(self.x[0]*0.9,self.x[-1]*0.9, len(onsets)) 
        
        # combine n guassians
        for i in range(0,len(onsets)):
            control_seq[onsets[i]:onsets[i]+int(durations[i]),:] = amplitudes[i] * np.exp(-0.5 * (self.x - positions[i]) ** 2 / sigmas[i] ** 2)
            # control_seq
        # np.zeros([n_control_vals, len(self.x)])
        # return control_seq
        return np.ones([n_control_vals, len(self.x)]) * 0

class LIF_input(SequenceGenerator):

    def __init__(self,step,magnitudes,period,dim,amplitude,std,rng=None):
        super().__init__(dim, rng)
        self.step = step
        self.period = period
        self._period = period
        self._min, self._max = magnitudes
        self.amplitude = amplitude
        self.std = std

    def _sample_impl(self,time_range,delta):
        n_control_vals = int(1+np.floor((time_range[1] - time_range[0]) / delta))
        n_amplitude_vals = int(np.ceil(n_control_vals / self._period))
        amp_seq = np.zeros(shape=(n_amplitude_vals, self.dim))

        if self.step:
            mask_amp_seq = self._rng.uniform(low=self._min,
                                high=self._max,
                                size=(n_amplitude_vals, 1)) 

            for i in range(n_amplitude_vals):
                # Randomly choose a starting index for the std-neuron segment
                start_idx = numpy.random.randint(0, self.dim - self.std)
                # Set the std consecutive neurons to the original random value
                amp_seq[i, 0:] = mask_amp_seq[i, 0]  # Use the random value from t

        else: # guassian over space
            neuron_indices = np.arange(self.dim)
            mu = numpy.random.randint(low=neuron_indices[0], high=neuron_indices[-1], size=n_amplitude_vals) # random neuron locations
            magnitude = numpy.random.uniform(low=self._min, high=self._max, size=n_amplitude_vals) # random magnitudes
            for i,mean in enumerate(mu):
                distance = np.minimum(
                    np.abs(neuron_indices - mean),
                    self.dim - np.abs(neuron_indices - mean)
                    )
                amp_seq[i] = magnitude[i] * np.exp(-0.5 * (distance ** 2) / self.std ** 2)

        control_seq = np.repeat(amp_seq, self._period, axis=0)[:n_control_vals]
        return control_seq

    
_seqgen_names = {
    "GaussianSequence": GaussianSequence,
    "UniformSqWave": UniformSqWave,
    "GaussianSqWave": GaussianSqWave,
    "LogNormalSqWave": LogNormalSqWave,
    "RandomWalkSequence": RandomWalkSequence,
    "SinusoidalSequence": SinusoidalSequence,
    "Gaussian1D": Gaussian1D,
    "LIF_input": LIF_input,
}


def get_sequence_generator(name, args):
    if name == "Product":
        # args should be a list of tuples (name, args) for each coordinate.
        components = [
            get_sequence_generator(el["name"], el["args"]) for el in args
        ]
        return Product(components)        
    else:
        return _seqgen_names[name](**args)
