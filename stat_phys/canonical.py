import numpy as np, matplotlib.pyplot as plt, time
from tqdm import tqdm

class CanonicalMC:
    def __init__(self, Nbodies, hamiltonian, state_generator, flipper, kbred = True, T = None):
        if kbred == True:
            self.kb_red()
        else:
            self.reset_kb()
        self.N = Nbodies #integer
        self.T = T
        self.energy = hamiltonian #hamiltonian(state) = float
        self.gen_state = state_generator #state_generator(Nbodies) = state (size Nbodies numpy array)
        self.flip = flipper #flipper(state1, idx) = state2
        self.state = self.gen_state(self.N)
        self.state_energy = self.energy(self.state)
            
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

    def state_reset(self):
        self.state = self.gen_state(self.N)
        self.state_energy = self.energy(self.state)

    def metropolis(self, steps, thermal_steps):
        dimensions = [thermal_steps+steps]
        for i in self.state.shape:
            dimensions.append(i)
        states = np.zeros(dimensions) #step, {parameter1, parameter2, ...} (=state)
        
        for step in range(thermal_steps+steps):
            idx = np.random.randint(self.N)
            new_state = self.flip(self.state, idx)
            E_new = self.energy(new_state)
            dE = E_new-self.state_energy

            if dE <= 0 or np.random.rand() < np.exp(-self.beta*dE):
                self.state = new_state
                self.state_energy = E_new
 
            states[step] = self.state
        return states[thermal_steps:]
            
    def sample(self, N_sample, thermal_steps):
        return self.metropolis(N_sample, thermal_steps)

    def run(self, Tstart, Tend, dT, n_sample, thermal_steps, reset = False, 
            verbose = False, progressbar = True, perc = 0.1, description = "Running MC"):
        t0 = time.time()
        last = t0

        T = np.arange(Tstart, Tend, dT)
        self.set_temperature(Tstart)
        dimensions = [T.shape[0], n_sample]
        for i in self.state.shape:
            dimensions.append(i)
        STATES = np.zeros(dimensions) #step, {parameter1, parameter2, ...} (=state)
        PERC = perc

        if reset:
            self.state_reset()
        _ = self.sample(N_sample = 0, thermal_steps = thermal_steps)

        for i in tqdm(range(T.shape[0]), desc = description):
            self.set_temperature(T[i])
            STATES[i,:] = self.sample(N_sample = n_sample, thermal_steps = 0)
            if i/T.shape[0] >= PERC and verbose == True:
                print('perc:', round(100*PERC, 0), '%, partial time:', time.time()-last, 'seconds.')
                PERC += perc
                last = time.time()
                
        dt = time.time()-t0
        if verbose == True:
            print('perc: 100%, partial time:,', time.time()-last, 'seconds. \n')
            print('time taken:', dt, 'seconds')

        return STATES
