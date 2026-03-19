import numpy as np, pytest
from stat_phys.system import System

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
def system():
    return System(
        hamiltonian = H,
        ndf = N,
        state_generator = gen,
        observables = [mean_energy],
        energy_update = energy_update
    )

def test_init_n_obs(system):
    assert system.n_obs == 1
def test_init_ndf(system):
    assert system.ndf == N
def test_init_energy(system):
    assert system.state_energy == H(system.state)
def test_init_energy_update(system):
    assert system.energy_update == energy_update
def test_state_reset(system):
    system.state_reset()
    assert system.state_energy == H(system.state)
def test_propose_move(system):
    state, energy = system._propose_move()
    assert H(state) == energy
def test_accept_move(system):
    state = system.state
    energy = system.state_energy
    system.state_reset()
    system._accept_move(state, energy)
    assert np.array_equal(system.state, state)
    assert system.state_energy == energy
def test_observables(system):
    state = system.state
    obs = mean_energy(state)
    assert system.observables[0] == obs
    