# Adapted from https://github.com/arvoelke/phd/blob/master/code/poisson-performance.py

import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame

from collections import defaultdict

import nengo
from nengo.neurons import LIFRate

figname = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "../figures", "poisson-performance.pdf")


# https://github.com/nengo/nengo/issues/1487
class PoissonLIF(LIFRate):
    """Poisson-spiking leaky integrate-and-fire (LIF) neuron model.

    Parameters
    ----------
    tau_rc : float
        Membrane RC time constant, in seconds. Affects how quickly the membrane
        voltage decays to zero in the absence of input (larger = slower decay).
    tau_ref : float
        Absolute refractory period, in seconds. This is how long the
        membrane voltage is held at zero after a spike.
    amplitude : float
        Scaling factor on the neuron output. Corresponds to the relative
        amplitude of the output spikes of the neuron.
    """

    probeable = ('spikes',)

    def __init__(self, tau_rc=0.02, tau_ref=0.002, amplitude=1, seed=None):
        super(PoissonLIF, self).__init__(
            tau_rc=tau_rc, tau_ref=tau_ref, amplitude=amplitude)

        # TODO(arvoelke): the simulator should pass in the rng
        self.rng = np.random.RandomState(seed=seed)

    def _sample_exponential(self, rates):
        # generate an exponential random variable (time between
        # poisson events) using its inverse CDF. note that this
        # distribution is "memoryless", which is why we don't
        # need to save this state or worry about what happens
        # outside of this time-step.
        return -np.log1p(-self.rng.rand(len(rates))) / rates

    def _poisson_step_math(self, dt, rates, spiked):
        spiked[...] = 0
        next_spikes = np.zeros_like(spiked)
        to_process = np.ones_like(spiked, dtype=bool)

        while np.any(to_process):
            next_spikes[to_process] += self._sample_exponential(
                rates[to_process])
            to_process &= next_spikes < dt
            spiked[to_process] += self.amplitude / dt

    def step_math(self, dt, J, spiked):
        rates = np.zeros_like(J)
        LIFRate.step_math(self, dt=1, J=J, output=rates)
        self._poisson_step_math(dt, rates, spiked)


def go(freq,
       neuron_type,
       n_neurons_over_freq=50,  # scale-invariant
       n_steps=1000,
       tau_times_freq=0.01,  # dimensionless
       dt_times_freq=0.001,  # dimensionless
       max_rates=nengo.dists.Uniform(20, 40),
       seed=0,
      ):

    n_neurons = int(n_neurons_over_freq * freq)
    tau = tau_times_freq / freq
    dt = dt_times_freq / freq

    print(n_neurons)

    with nengo.Network(seed=seed) as model:
        u = nengo.Node(lambda t: np.sin(freq*2*np.pi*t))
        x = nengo.Ensemble(n_neurons, 1, neuron_type=neuron_type,
                           max_rates=max_rates)
        nengo.Connection(u, x, synapse=None)    
        p_actual = nengo.Probe(x, synapse=tau)
        p_ideal = nengo.Probe(u, synapse=tau)

    with nengo.Simulator(model, dt=dt) as sim:
        rng = np.random.RandomState(seed=seed)
        if isinstance(neuron_type, nengo.LIF):
            from nengolib.neurons import init_lif
            init_lif(sim, x, rng=rng)
        elif isinstance(neuron_type, nengo.SpikingRectifiedLinear):
            # https://github.com/nengo/nengo/issues/1415
            sim.signals[sim.model.sig[x.neurons]['voltage']] = (
                rng.rand(x.n_neurons))

        sim.run_steps(n_steps)

    return nengo.utils.numpy.rmse(
        sim.data[p_actual], sim.data[p_ideal])


n_trials = 5
xlabel = "$m$ (# Neurons)"

def get_models(seed):
    return (
        ('Poisson', PoissonLIF(seed=seed)),
    )


data = defaultdict(list)
for seed in range(n_trials):
    for name, neuron_type in get_models(seed):
        for n_neurons_over_freq in np.geomspace(10, 2001, 11):
            freq = 10
            data['Model'].append(name)
            data[xlabel].append(n_neurons_over_freq * freq)
            data['Seed'].append(seed)
            data['RMSE'].append(go(
                freq, neuron_type, n_neurons_over_freq=n_neurons_over_freq,
                seed=seed))

n_neurons = np.unique(data[xlabel])
c = 5  # fudge-factor to shift the line vertically
lw = 3

with sns.plotting_context('paper', font_scale=4):
    plt.figure(figsize=(14, 7))
    plt.plot(n_neurons, c / np.sqrt(n_neurons), lw=lw, linestyle='--',
         c='black', label=r"$\frac{d}{\sqrt{m}}$")
    sns.lineplot(data=DataFrame(data),
                 x=xlabel, y="RMSE", hue="Model", lw=lw)
    plt.xscale('log')
    plt.yscale('log')
    sns.despine(offset=15)

    plt.savefig(figname, bbox_inches='tight')
