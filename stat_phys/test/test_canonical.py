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
        steps = 10, 
        thermal_steps = 5, 
        thinning = 2, 
        progress = False
    )
    assert observables.shape[0] == 10
    assert observables.shape[1] == 1
    assert autocorr_time is not None
    assert acceptance_rate is not None
def test_sweep(canonical):
    observables, autocorr_times, acceptance_rates = canonical.sweep(
        Tstart = 2.0,
        Tend = 1.0,
        n_T = 5,
        n_sample = 10,
        thermal_steps = 5,
        thinning = 2,
        progress = False
    )
    assert observables.shape[0] == 5
    assert observables.shape[1] == 10
    assert observables.shape[2] == 1
    assert autocorr_times.shape[0] == 5
    assert autocorr_times.shape[1] == 1
    assert acceptance_rates.shape[0] == 5
def test_std_error(canonical):
    observables, autocorr_times, acceptance_rates = canonical.sweep(
        Tstart = 2.0,
        Tend = 1.0,
        n_T = 5,
        n_sample = 10,
        thermal_steps = 5,
        thinning = 2,
        progress = False
    )
    O = observables[:,:,0]
    autocorr_time = autocorr_times[:, 0]
    sigma = canonical.std_error(O, autocorr_time)
    assert sigma.shape == (5,)
    assert np.all(sigma >= 0)
def test_chi2(canonical):
    observables, autocorr_times, acceptance_rates = canonical.sweep(
        Tstart = 2.0,
        Tend = 1.0,
        n_T = 5,
        n_sample = 10,
        thermal_steps = 5,
        thinning = 2,
        progress = False
    )
    O = observables[:,:,0]
    autocorr_time = autocorr_times[:, 0]
    sigma = canonical.std_error(O, autocorr_time)
    T = np.linspace(2.0, 1.0, 5)
    p1_analytical = np.exp(-1/T) / (1 + np.exp(-1/T))
    chi2, dof, excluded = canonical.chi2(O, autocorr_time, p1_analytical)
    assert chi2 >= 0
    assert dof >= 0
    assert excluded >= 0