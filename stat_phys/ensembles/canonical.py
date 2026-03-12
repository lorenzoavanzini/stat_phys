import numpy as np
from tqdm import tqdm

class CanonicalMC:
    def __init__(self, system, kbred = True, T = None):
        self.system = system
        self.T = T
        
        if kbred == True:
            self.kb_red()
        else:
            self.reset_kb()
            
    def kb_red(self):
        self.kb = 1
        
    def reset_kb(self):
        self.kb = 1.380649*10**-23
        
    def set_temperature(self, T):
        if T <= 0:
            raise ValueError('T<=0. Temperature must be positive!')
        self.T = T

    @property
    def beta(self):
        if self.T is None:
            raise ValueError("Temperature not set!")
        return 1/(self.kb*self.T)

    def metropolis(self, steps, thermal_steps, progress = True):
        observables = np.zeros([thermal_steps+steps, self.system.n_obs])
        iterator = range(thermal_steps+steps)
        if progress:
            iterator = tqdm(iterator, desc = 'MC steps', leave = False)

        accepted = 0
        for step in iterator:
            new_state, E_new = self.system._propose_move()
            dE = E_new-self.system.state_energy

            if dE <= 0 or np.random.rand() < np.exp(-self.beta*dE):
                accepted += 1
                self.system._accept_move(new_state = new_state, new_energy = E_new)
 
            observables[step, :] = self.system.observables
        if steps!=0:
            acceptance_rate = accepted/(steps)
        else:
            acceptance_rate = None
        return observables[thermal_steps:,:], acceptance_rate

    def run(self, Tstart, Tend, n_T, n_sample, thermal_steps, reset = False, description = 'Temperature sweep', thermal_each_T = False):
        T = np.linspace(Tstart, Tend, n_T)
        self.set_temperature(Tstart)
        dimensions = [T.shape[0], n_sample, self.system.n_obs]
        observables = np.zeros(dimensions)
        acceptance_rates = np.zeros(n_T)

        if reset:
            self.system.state_reset()
        _, _ = self.metropolis(steps = 0, thermal_steps = thermal_steps)

        for i in tqdm(range(T.shape[0]), desc = description):
            self.set_temperature(T[i])
            if not thermal_each_T:
                loop_thermal_steps = 0
            else:
                loop_thermal_steps = thermal_steps
            observables[i,:,:], acceptance_rates[i] = self.metropolis(steps = n_sample, thermal_steps = loop_thermal_steps)

        return observables, acceptance_rates #n_T, n_sample, n_obs/ n_T









































