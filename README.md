# Fourier-based Observation of aXions
an Axion Haloscope Analysis Pipeline 
Goal: Develop a Python package that simulates and analyzes haloscope data inspired by the full chain described in Ed Daw's thesis and Chapter 7 of the HAYSTAC thesis [1]. 

The pipeline starts from raw power spectra (simulated Gaussian noise with optional axion signals) and produces a final exclusion limit plot.

The design is modular – each step of the analysis is encapsulated in separate functions/modules with clear interfaces and documentation.


### Setting up the code
1. Downlowad the directory in your local area
>git clone https://github.com/HALO-DM/FOX

2. Make sure you  install all dependencies by
going into the FOX directory
>pip install -e ".[dev]"


3. Test things work
- run unit test
> pytest --basetemp=./test_output

- run a simulation
> python scripts/simulation/simulate_run_yaml.py configs/simulate_run.yaml

---
### Repository structure
```text

FOX/                       <-- Project root (where pyproject.toml lives)
├── .gitignore             <-- Extensions that are NOT uploaded to the github
├── README.md              <-- Here we are :) 
├── generate_tree.sh       <-- bash script to generate a printout of the repository structure
├── pyproject.toml         <-- project configuration file for the "axion_haloscope" Python package
├── axion_haloscope/       <-- the Python package (imported in code/tests). This is where the classes live
│   ├── __init__.py
│   ├── baseline.py
│   ├── bootstrapping.py
│   ├── cli.py
│   ├── cli_example.sh
│   ├── combine.py
│   ├── config.py
│   ├── data_quality.py
│   ├── detection.py
│   ├── external_noise.py
│   ├── how_to_replace_lineshape.py
│   ├── io.py
│   ├── limit.py
│   ├── lineshape.py
│   ├── rebin.py
│   ├── run_pipeline.py
│   ├── simulation.py
│   └── width_fq.py
├── configs/                             <-- put your yaml configuration files here
│   ├── simulate_run.yaml
│   └── simulate_spectra_only.yaml
├── scripts/                             <-- put run scripts here
│   ├── full_chain_template.py
│   ├── read_npz_pipeline.py
│   ├── read_spectra_npz.py
│   └── simulation/
│   └── simulate_run.py
├── output/                              <-- generated results from the scripts folder go here (gitignored)
├── tests/                               <-- unit & integration tests
    ├── test_combine.py
    ├── test_data_quality.py
    ├── test_detection.py
    ├── test_io_hdf5.py
    ├── test_lineshape.py
    ├── test_simulation_plot.py
    └── test_simulation_smoke.py
└── test_output/                         <-- generated results of the unit tests (gitignored)

```


### A word on pyproject.toml
pyproject.toml is the main project configuration file to setup and build Python package. Here we define:

    1. How to build/install the package:
    
    2. The package metadata
    
    3. Python and dependency requirements
        When you run the "pip install -e ".[dev]" command, pip reads this file and installs those dependencies
        
    4. Optional development tools
    
    5. Command-line tools 
        e.g. create the shell command "axion-haloscope" that creates that runs axion_haloscope/cli.py
        
    6. Pytest configuration 
        e.g. tells pytest where to look for tests
    

# Analysis structure
Below we outline each stage of the analysis and how it is implemented:

## Simulated Power Spectrum Input
We begin by generating synthetic power spectra that mimic haloscope data.
Each spectrum represents the measured receiver power vs. frequency for a fixed tuning of the cavity, with:

- Gaussian noise corresponding to the thermal and amplifier noise (zero-mean fluctuations around a baseline power level).

- Optional injected axion signals as narrow peaks on top of the noise (with known position and amplitude).

Each spectrum is typically discretized into bins of width Δν (e.g. 100 Hz) and spans a small frequency range [given by the number of bins -- configurable].  In a real run, many such spectra are taken at adjacent cavity tunings to cover a broader frequency range. 

Our simulation allows configuring parameters such as:
```
number of spectra, 
noise level, 
bin width, 
injected signal properties: frequency
injected signal properties: strength
```

In `simulation.py`, the injected signal is currently a **Gaussian axion-like line added directly in RF frequency space**.

The injection flow is:

1. `AxionParams` defines the fake signal:

```python
f_axion_hz   # center frequency
sigma_hz     # Gaussian width
total_power  # total injected power
```

2. A **global RF grid** is built for the whole scan, including all overlapping spectra. Each individual spectrum is mapped onto this common RF grid with `rf_index_map`. 

3. If `axion` is provided, the code builds `axion_power_global` over the full RF grid:

```python
axion_power_global = inject_axion_power(
    rf_grid,
    axion.f_axion_hz,
    axion.sigma_hz,
    axion.total_power
)
```

That uses `axion_lineshape_gaussian()`, which computes
$L(f) \propto \exp\left[-\frac{1}{2}\left(\frac{f-f_a}{\sigma_f}\right)^2\right]$
and normalizes it so the discrete bin sum is 1. Then `total_power * L` gives the power deposited per RF bin.

4. Each simulated raw spectrum is generated as baseline × noise/external background:

```python
raw = baseline * (external + noise)
```

5. The relevant slice of the global axion signal is then added to each spectrum:

```python
raw = raw + axion_power_global[idx]
```

where `idx = rf_index_map[i]`, so each tuning only receives the part of the signal that lies inside its frequency coverage. 

So the injection is **additive power injection after the receiver/noise baseline is generated**.

Notes and to-dos:

`lineshape.py` defines the **SHM/Maxwellian template** used later for matched filtering in the grand spectrum. It maps the Standard Halo Model speed distribution into a one-sided frequency-space profile and normalizes it as a template.  But the current `simulation.py` injection itself still uses the simpler **Gaussian** lineshape, not the SHM template.

We assume the spectra are already calibrated to units of excess power (i.e. dimensionless “power above noise” units) for analysis convenience. We are working to change that. An initial normalization by the average noise power can be applied to each raw spectrum. 


## Baseline Removal with Savitzky–Golay Filter
Each raw spectrum contains a baseline – a smooth background from the receiver noise that may have a frequency dependence (e.g. the cavity’s Lorentzian response or gain variations). To reliably detect tiny axion peaks, we must remove this baseline so that the remaining spectrum is flattened to fluctuations around zero.

`baseline.py` does two main things:

First, `remove_baseline()` estimates and removes the smooth background shape of a spectrum. If no external baseline is supplied, it builds one with a Savitzky–Golay filter. Then it either removes it multiplicatively,

```python
processed = spectrum / baseline
```

or additively,

```python
processed = spectrum - baseline
```

depending on `mode`. It can also subtract 1 afterward, and it supports a diagnostic plot with two panels: raw + SG baseline on top, processed spectrum below. 

Second, it has helper utilities: `align_and_average_spectra()` aligns several spectra onto a common x-axis and averages them, while `mask_bins()` masks selected bins either by explicit indices or by frequency ranges --> allows to cut known bad bins (e.g. if certain frequencies are contaminated by instrumental interference). 


Notes: the Savitzky–Golay Filtering fits a low-order polynomial to a moving window of the spectrum and subtracts the fitted “smooth” curve, removing variations broader than the window width. It preserves narrow spectral features (like potential axion peaks) while filtering out slower baseline drifts. In practice, we choose a window length W (in bins) larger than the expected width of any axion signal, and a polynomial order d (low order to capture baseline shape without fitting peaks). This approach requires no specific model of the baseline. To-do: implement safe-guards to SGF number of bins, quantify the power attenuation given due to the filter.



##  Combining Spectra Vertically (Maximum-Likelihood Weighting)
This step combines many baseline-removed spectra into one common RF spectrum. Each measured spectrum covers a small frequency range. Neighboring spectra overlap because the cavity is tuned in small steps.

`combine.py` takes all processed spectra and places them onto a shared RF frequency grid.

Where more than one spectrum contributes to the same RF bin, the code averages them using inverse-variance weighting. Main function:

```python
from axion_haloscope.combine import combine_ml

combined, sigma_c, counts = combine_ml(
    processed_spectra,
    rf_index_map,
    total_rf_bins=len(rf_grid),
)

```

Note. We need to add a rescaling functionality to the code. Before combining, rescale each spectrum so that a potential axion signal of a given coupling would produce the same expected amplitude in all spectra. In a real experiment, an axion converted in one spectrum might produce slightly different power than in another (due to varying expected signal power from B field, system noise temperature, integration time)



## Rebinning Horizontally into the Grand Spectrum (Axion Lineshape Integration)

After baseline removal and vertical combination, the analysis has one combined spectrum on a common RF grid which is still sampled at the original frequency bin width, for example 100 Hz. A real axion signal is expected to be broader than one bin -->  at GHz frequencies, the expected linewidth of the order of a few kHz. Therefore, the analysis combines neighboring bins using an expected axion signal template.

This is done in two steps:

1. rebin.py rebins and constructs the grand spectrum.
   groups neighboring frequency bins by a factor C + slides an axion signal template Lq across the rebinned spectrum and combines neighboring rebinned bins using maximum-likelihood weighting
   
2. lineshape.py provides the axion signal template used in the matched filter --> this is where we need to get the shape right from physics parameters. The code will allow different lineshape models (making it easy to test sensitivity to e.g. alternate halo models).


# Grand Spectrum Construction and Calculation of the z-score
When creating the Grand Specturm we have

| Input    | Description                        |
| -------- | ---------------------------------- |
| `Dr`     | Rebinned spectrum.                 |
| `sr`     | Uncertainty of each rebinned bin.  |
| `Lq`     | Expected axion lineshape template. |

| Output | Description                             |
| ------ | --------------------------------------- |
| `Dg`   | Grand spectrum.                         |
| `sg`   | Uncertainty of each grand-spectrum bin. |

The detection statistic is the grand-spectrum z-score: `z = Dg / sg`
In noise-only data, this should be approximately Gaussian with mean 0 and standard deviation 1.
Candidate axion signals are searched for as positive excesses in this z-score spectrum. This is the spectrum in which we apply threshold cuts to look for candidates.

## SNR Estimation and Candidate Selection (Thresholding & Rescan)
With the grand spectrum in hand, we next determine which (if any) bins are significant outliers that could indicate an axion. We also quantify the sensitivity in terms of SNR and set thresholds to achieve a desired confidence level for exclusion.

Signal-to-Noise Ratio (SNR): In haloscope analysis, the SNR of a potential axion is defined as the ratio of the expected signal power to the uncertainty (noise).


Candidate Flagging: The package function find_candidates(grand_spec, sigma, threshold) (in detection.py) will return a list of candidate frequencies where the grand spectrum exceeds the threshold.


## Determining the 95% Exclusion Limit and Outputting Results
No Detection Case (default): when no candidate survives, we proceed to set an exclusion limit. But first, we should apply a small correction for the SG filter’s impact on SNR. As noted, the SG baseline removal can attenuate signal power. HAYSTAC analytically and empirically estimated this attenuation (denoted η) and incorporated it into the final limit calculation. Our code can include a factor for “SNR efficiency”  as a user-set parameter (snr_efficiency). This will linearly affect the coupling limit. Finally, after analyzing the data and finding no persistent candidates, we set a 95% confidence exclusion limit on the axion-photon coupling $|g_{a\gamma\gamma}|$ as a function of frequency (or axion mass).







## More nerdy bits

### Unit tests
Unit tests to check code issues: they protect the analysis pipeline from silent code mistakes.
They do not prove the physics result!!!

Examples:
    
    simulation.py → correct spectra and frequency grids
    
    io.py → saved HDF5/NPZ data reloads unchanged
    
    baseline.py → baseline removal returns finite spectra
    
    combine.py → overlapping bins combine correctly
    
    lineshape.py → SHM template is normalized
    
    detection.py → thresholds behave as expected
    
    data_quality.py → bad spectra are removed


### I/O
The current axion_haloscope/io.py supports a common I/O interface for simulated and real spectra through a SpectrumSet container.

```
SpectrumSet(
    spectra=[...],                 # list of raw power spectra
    freqs_per_spec=[...],   # frequency axis for each spectrum
    rf_grid=...,                    # common global RF grid
    rf_index_map=[...]      # maps each spectrum onto rf_grid
)
```


The I/O module lets the rest of the analysis pipeline ignore where the data came from. 
Whether the spectra are simulated, loaded from CSV, loaded from .npz, or loaded from HDF5, they all become the same object:  SpectrumSet

