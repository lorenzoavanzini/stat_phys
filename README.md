# stat_phys

`stat_phys` is an early-stage Python library for performing statistical physics simulations.

## Structure

Currently at version 0.3.0, the library has the following structure:

- `stat_phys/`
  - `__init__.py`
  - `system.py`
  - `ensembles/`
    - `canonical.py`

## Modules

### system.py — `System`

The `System` class is used to describe a generic physical system. A system is defined by:

- **hamiltonian** — a function `H(state)` that returns the state's energy
- **observables** — a list of functions, each taking the state and returning one observable
- **ndf** — the number of degrees of freedom
- **state_generator** — a function `gen(idx)` that returns a random value for the degree of freedom at index `idx`
- **energy_update** *(optional)* — a function `energy_update(state, state_energy, idx, new_value)` that computes the new state energy after updating one degree of freedom, without recalculating from scratch

The system's state is represented as a 1D NumPy array of length `ndf`.

### ensembles/canonical.py — `CanonicalMC`

The `CanonicalMC` class is used to perform simulations in the canonical ensemble via Monte Carlo methods. It implements the Metropolis algorithm and supports temperature sweeps.

## Example: CanonicalMC simulation of a 2-level system
The following example shows how to simulate a 2-level energy system with N particles calculating the mean energy as a function of the temperature. 
```python
import numpy as np, matplotlib.pyplot as plt
from stat_phys.system import System
from stat_phys.ensembles.canonical import CanonicalMC

# Two-level system: N independent particles with energy 0 or 1
N = 100

# each degree of freedom can take value 0 or 1
def gen(idx):
    return np.random.choice([0, 1])

# total energy is the sum of all degrees of freedom
def hamiltonian(state):
    return np.sum(state)

# when degree of freedom idx changes from state[idx] to S,
# the energy changes by (S - state[idx])
def energy_update(state, state_energy, idx, S):
    return state_energy + (S - state[idx])

# observable: mean energy per particle
def mean_energy(state):
    return np.mean(state)

Tmax = 10.1
Tmin = 0.1
n_T = 50
T = np.linspace(Tmax, Tmin, n_T)  # temperature array for plotting

# define the system
system = System(
    hamiltonian = hamiltonian,
    observables = [mean_energy],
    ndf = N,
    state_generator = gen,
    energy_update = energy_update
)

# initialize the canonical ensemble
ensemble = CanonicalMC(system = system)

# run a temperature sweep from Tmax to Tmin (simulated annealing)
results, autocorr_times, acceptance_rates = ensemble.sweep(
    Tstart = Tmax,
    Tend = Tmin,
    n_T = n_T,
    n_sample = 1000,
    thermal_steps = 200,
    thinning = 300,
    progress = True
)
```

Notice that the results of the sweep method is a NumPy array of shape (n_T,n_samples,n_obs). In order to access the first observable (the only one in this case) and to consider its mean value at fixed temperature, we have to write as follows

```python

# average over samples to get mean energy at each temperature
m = np.mean(results[:,:,0], axis = 1)
```

Then we can plot the mean energy of the system and the acceptance rate of the Monte Carlo engine over the temperature. 

```python
#get autocorrelation time of the energy chain
autocorr_time = autocorr_times[:, 0]

# analytical solution: probability of occupying the level with energy 1
p1_analytical = np.exp(-1/T) / (1 + np.exp(-1/T))

# plot mean energy: MC vs analytical
fig, axes = plt.subplots(3, 1, figsize = (8, 10))
axes[0].plot(T, m, color = 'red', label = 'MC simulation', lw = 1, alpha = 0.75)
axes[0].plot(T, p1_analytical, color = 'black', label = 'analytical', lw = 1.0, linestyle = '--')
axes[0].set_xlabel('T')
axes[0].set_ylabel(r'$\langle E \rangle / N$')
axes[0].set_title(f'Two-level system Energy (N={N})')
axes[0].legend(loc = 'best')
axes[0].grid()

# plot acceptance rate as a function of temperature
axes[1].plot(T, acceptance_rates, color = 'blue', label = 'acceptance rate', lw = 0.7, alpha = 0.75)
axes[1].set_xlabel('T')
axes[1].set_ylabel('acceptance rate')
axes[1].set_title('Acceptance rate')
axes[1].legend(loc = 'best')
axes[1].grid()

#plot autocorrelation time as a function of temperature
axes[2].plot(T, autocorr_time, color = 'purple', label = 'autocorrelation time', lw = 0.7, alpha = 0.75)
axes[2].set_xlabel('T')
axes[2].set_ylabel('autocorrelation time')
axes[2].set_title('Energy autocorrelation time')
axes[2].legend(loc = 'best')
axes[2].grid()

plt.tight_layout()
plt.savefig('two_levels.png')
```


## Changelog

### v0.3.0
- Added `thinning` parameter to `metropolis` for subsampling the chain
- Added autocorrelation time calculation (`_autocorr_time`, `_C`, `_rho`)
- Added autocorrelation time output to `metropolis` and `sweep`
- Renamed `run` to `sweep` for clarity

### v0.2.0
- Added `System` class in `system` module to describe a generic physical system
- Added support for `energy_update` function to optimize local energy calculation
- Added acceptance rate calculation
- Added `thermal_each_T` option to control thermalization at each temperature step
- Restructured the library with `ensembles/` subfolder

### v0.1.0
- First release
