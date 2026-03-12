import numpy as np

class System:
    def __init__(self, hamiltonian, observables, ndf, state_generator, energy_update = None):
        self.hamiltonian = hamiltonian #hamiltonian(state) = energy
        self.obs = observables #observable(state) = observable
        self.n_obs = len(self.obs)

        self.ndf = ndf
        self.gen_state = state_generator
        self.state = np.zeros(self.ndf)

        for i in range(self.ndf):
            self.state[i] = self.gen_state(idx = i)

        self.state_energy = self.hamiltonian(self.state)
        self.energy_update = energy_update #energy_update(state, state_energy, idx, value) = energy(new_state)
        
    def state_reset(self):
        for i in range(self.ndf):
            self.state[i] = self.gen_state(idx = i)
        self.state_energy = self.hamiltonian(self.state)

    def _propose_move(self):
        idx = np.random.randint(self.ndf)
        new_state = self.state.copy()
        new_state_value = self.gen_state(idx = idx)
        new_state[idx] = new_state_value

        if self.energy_update is None:
            new_energy = self.hamiltonian(new_state)
        else:
            new_energy = self.energy_update(self.state, self.state_energy, idx, new_state_value)
        
        return new_state, new_energy

    def _accept_move(self, new_state, new_energy):
        self.state = new_state
        self.state_energy = new_energy

    @property
    def observables(self):
        measures = np.zeros(self.n_obs)
        for i in range(self.n_obs):
            measures[i] = self.obs[i](self.state)
        return measures












