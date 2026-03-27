# stat_phys

`stat_phys` is an early-stage Python library for performing statistical physics simulations.

## Structure

Currently at version 0.3.3, the library has the following structure:

- `stat_phys/`
  - `__init__.py`
  - `system.py`
  - `ensembles/`
    - `canonical.py`
  - `test`
    - `test_system.py`
    - `test_canonical.py`
  - `examples`
    - `2levels.py`
    - `two_levels.png`

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

The `CanonicalMC` class is used to perform simulations in the canonical ensemble via Monte Carlo methods. It implements the Metropolis algorithm and supports temperature sweeps. You can find an example that shows how to simulate a two-levels system in the `examples` subfolder.

## Testing with Pytest
To test the library with `pytest`, navigate to the `stat_phys` folder in your terminal and run:
```bash
pytest -v
```

## Changelog

### v0.3.3
- Added `sweep_save` method in `CanonicalMC` class to save simulation results
- Added `sweep_load` method in `CanonicalMC` class to load previous simulation results

### v0.3.2
- Added `std_error` method for statistical error estimation accounting for autocorrelation
- Added `chi2` method for chi-squared goodness-of-fit test against theoretical predictions
- Fixed numerical warnings in `std_error`

### v0.3.1
- Added `test` sub-folder with `test_system.py` and `test_canonical.py` for `pytest` testing
- Added `conftest.py` for `pytest` support

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
