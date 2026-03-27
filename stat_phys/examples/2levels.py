import numpy as np, matplotlib.pyplot as plt
from stat_phys.system import System
from stat_phys.ensembles.canonical import CanonicalMC

#set the random seed
np.random.seed(42)

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

#simulation properties
np.random.seed(42)
Tmax = 10.1
Tmin = 0.1
n_T = 50
T = np.linspace(Tmax, Tmin, n_T)  # temperature array for plotting
n_sample = 1000
thermal_steps = 200
thinning = 200

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
    n_sample = n_sample,
    thermal_steps = thermal_steps,
    thinning = thinning,
    progress = True
)

# average over samples to get mean energy at each temperature
m = np.mean(results[:,:,0], axis = 1)

#get autocorrelation time of the energy chain
autocorr_time = autocorr_times[:, 0]

#get statistical errors
sigma = ensemble.std_error(results[:,:,0], autocorr_time)

# analytical solution: probability of occupying the level with energy 1
p1_analytical = np.exp(-1/T) / (1 + np.exp(-1/T))

#get chi2 and dof
chi2, dof, excluded = ensemble.chi2(results[:,:,0], autocorr_time, p1_analytical)

# plot mean energy: MC vs analytical
fig, axes = plt.subplots(3, 1, figsize = (12, 10))
axes[0].errorbar(T, m, yerr = sigma, color = 'red', label = f'MC simulation \n $\\chi^2/dof$ = {chi2/dof:.3f} \n excluded = {excluded}', 
                 lw = 0.7, alpha = 0.75, capsize = 5)

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

#plot autucorrelation time as a function of temperature
axes[2].plot(T, autocorr_time, color = 'purple', label = 'autocorrelation time', lw = 0.7, alpha = 0.75)
axes[2].set_xlabel('T')
axes[2].set_ylabel('autocorrelation time')
axes[2].set_title('Energy autocorrelation time')
axes[2].legend(loc = 'best')
axes[2].grid()

plt.tight_layout()
plt.savefig('two_levels.png')

ensemble.sweep_save(file = 'save.npz', kb = ensemble.kb, chain = results[:,:,0], autocorr_times = autocorr_times, acceptance_rates = acceptance_rates, 
                    Tstart = Tmax, Tend = Tmin, n_T = n_T, n_sample = n_sample, thermal_steps = thermal_steps, thinning = thinning)

data = ensemble.sweep_load('save.npz')
print( 'kb: \n', data['kb'], '\n\n',
       'chain: \n', data['chain'], '\n\n', 
       'autocorr_times: \n', data['autocorr_times'], '\n\n',
       'acceptance_rates: \n', data['acceptance_rates'], '\n\n',
       'Tstart: \n', data['Tstart'], '\n\n',
       'Tend: \n', data['Tend'], '\n\n',
       'n_T: \n', data['n_T'], '\n\n',
       'n_sample: \n', data['n_sample'], '\n\n',
       'thermal_steps: \n', data['thermal_steps'], '\n\n',
       'thinning: \n', data['thinning'], '\n\n',
       'c: \n', data['c'], '\n\n',
       'cutoff: \n', data['cutoff'], '\n\n',
       'thermal_each_T: \n', data['thermal_each_T']
     )
