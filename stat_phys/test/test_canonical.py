import numpy as np, pytest
from stat_phys.system import System
from stat_phys.ensembles.canonical import CanonicalMC

N = 10
def gen(idx):
    return np.random.choice([0,1])
def H(state):
    return np.sum(state)
def energy_update(state, state_energy, idx, s):
    return state_energy-state[idx]+s
def mean_energy(state):
    return np.mean(state)

@pytest.fixture
def canonical():
    system = System(
        hamiltonian = H,
        ndf = N,
        state_generator = gen,
        observables = [mean_energy],
        energy_update = energy_update
    )
    return CanonicalMC(system = system)

def test_init(canonical):
    assert canonical.T is None
    assert canonical.kb == 1
def test_set_T(canonical):
    with pytest.raises(ValueError):
        canonical.set_temperature(-1)
def test_beta(canonical):
    with pytest.raises(ValueError):
        beta = canonical.beta
def test_metropolis(canonical):
    canonical.set_temperature(295.15)
    observables, autocorr_time, acceptance_rate = canonical.metropolis(
        steps = 100, 
        thermal_steps = 10, 
        thinning = 10, 
        progress = False
    )
    assert observables.shape[0] == 100
    assert observables.shape[1] == 1
    assert autocorr_time is not None
    assert acceptance_rate is not None
def test_sweep(canonical):
    observables, autocorr_times, acceptance_rates = canonical.sweep(
        Tstart = 10.1,
        Tend = 0.1,
        n_T = 100,
        n_sample = 100,
        thermal_steps = 10,
        thinning = 10,
        progress = False
    )
    assert observables.shape[0] == 100
    assert observables.shape[1] == 100
    assert observables.shape[2] == 1
    assert autocorr_times.shape[0] == 100
    assert autocorr_times.shape[1] == 1
    assert acceptance_rates.shape[0] == 100