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

    def _C(self, tau, x):
        N = x.shape[0]
        if tau >= N:
            return 0.0
        mean = x.mean()
        xy = np.mean(x[:N-tau]*x[tau:])
        return xy-mean**2

    def _rho(self, tau, x, cutoff = 1e-4):
        Ctau = self._C(tau, x)
        C0 = self._C(0, x)
        if abs(C0) < cutoff:
            return 0.0
        return Ctau/C0
        
    def _autocorr_time(self, chain, c = 5, cutoff = 1e-5):
        t = 0
        tau = self._rho(t, chain, cutoff)
        while t < c*tau and t < len(chain)//2:
            t += 1
            rho_t = self._rho(t, chain, cutoff)
            tau += rho_t
        return tau

    def metropolis(self, steps, thermal_steps, thinning = 1, correlation = True, c = 5, cutoff = 1e-5, progress = True):
        observables = np.zeros([steps, self.system.n_obs])
        total_steps = thermal_steps+steps*thinning
        iterator = range(total_steps)
        if progress:
            iterator = tqdm(iterator, desc = 'MCMC steps', leave = False)

        accepted = 0
        for step in iterator:
            new_state, E_new = self.system._propose_move()
            dE = E_new-self.system.state_energy

            if dE <= 0 or np.random.rand() < np.exp(-self.beta*dE):
                accepted += 1
                self.system._accept_move(new_state = new_state, new_energy = E_new)

            if step >= thermal_steps and (step-thermal_steps)%thinning == 0:
                observables[(step-thermal_steps)//thinning, :] = self.system.observables
        if steps != 0:
            acceptance_rate = accepted/(steps*thinning)
        else:
            acceptance_rate = None

        if correlation == True:
            autocorr_time = np.zeros(self.system.n_obs)
            for i in range(self.system.n_obs):
                autocorr_time[i] = self._autocorr_time(observables[:, i], c, cutoff)
        else:
            autocorr_time = None
        return observables, autocorr_time, acceptance_rate

    def sweep(self, Tstart, Tend, n_T, n_sample, thermal_steps, thinning = 1, correlation = True, c = 5, cutoff = 1e-5, 
            reset = False, progress = True, description = 'Temperature sweep', thermal_each_T = False):
        
        T = np.linspace(Tstart, Tend, n_T)
        self.set_temperature(Tstart)
        dimensions = [T.shape[0], n_sample, self.system.n_obs]
        DIMENSIONS = [T.shape[0], self.system.n_obs]
        observables = np.zeros(dimensions)
        autocorr_times = np.zeros(DIMENSIONS)
        acceptance_rates = np.zeros(n_T)

        if reset:
            self.system.state_reset()
        _, _, _ = self.metropolis(steps = 0, thermal_steps = thermal_steps, correlation = False)

        iterator = range(T.shape[0])
        if progress:
            iterator = tqdm(iterator, desc = description, leave = False)
            
        for i in iterator:
            self.set_temperature(T[i])
            if not thermal_each_T:
                loop_thermal_steps = 0
            else:
                loop_thermal_steps = thermal_steps
            observables[i,:,:], autocorr_times[i,:], acceptance_rates[i] = self.metropolis(steps = n_sample, thermal_steps = loop_thermal_steps, 
                                                                                           thinning = thinning, correlation = correlation, c = c, 
                                                                                           cutoff = cutoff, progress = progress)

        return observables, autocorr_times, acceptance_rates #n_T, n_sample, n_obs/ n_T, n_obs / n_T


