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



4. Rebinning Horizontally into the Grand Spectrum (Axion Lineshape Integration)
The grand spectrum is the final spectrum in which we search for axion peaks. It is constructed by combining adjacent bins of the combined spectrum in a way that optimizes sensitivity to the expected axion lineshape[22][23].
Axion Signal Lineshape: Galactic axions have a characteristic frequency distribution (from their velocity dispersion in the Milky Way halo). In the HAYSTAC analysis, a standard virialized halo model was assumed, yielding a roughly Maxwellian broadened line on the order of Δν_a ~ a few kHz wide at ~5 GHz frequencies[24][23]. This width is tens of times larger than the 100 Hz bin size (Δν_a >> Δν_b). As a result, an axion signal would appear not in one bin but spread over many adjacent combined-spectrum bins with a specific shape (peaking at the axion’s rest frequency and tapering off).
Grand Spectrum Construction: We choose a segment length K (number of combined bins to sum) comparable to the axion linewidth (K ≈ Δν_a/Δν_b)[25]. For each possible K-bin segment in the combined spectrum, we compute a weighted sum of those bins to produce one grand spectrum bin. As with vertical combination, we use ML weighting: we assign weights proportional to the expected signal contribution in each combined bin and inversely to its variance[23]. In practice, this means:
First, rescale the combined bins in a segment so that a putative axion signal would contribute equally to each (i.e. divide by the expected lineshape value at each bin)[26]. This is analogous to the vertical rescaling: it “flattens” a true axion signal across the bins.
Then, weight each bin by $1/\sigma^2_c$ (its noise variance) and sum to get the grand spectrum value for that segment[26]. This ML approach maximizes the SNR for a distributed signal of known shape[27], essentially performing a matched filter for the axion line.
Our implementation in rebin.py might define construct_grand_spectrum(combined_spec, axion_lineshape) which outputs arrays for the grand spectrum values and their uncertainties. The axion_lineshape could be a function or template array giving the relative power fraction an axion signal would deposit in each of the K bins (normalized to unit total). For example, if we assume a simple Gaussian or Maxwellian profile for the axion signal in velocity space, this can be converted to a corresponding power spectral shape[28][29]. The code will allow different lineshape models (making it easy to test sensitivity to e.g. alternate halo models).
Grand Spectrum Details: One must decide how to slide the K-bin window across the spectrum. HAYSTAC’s approach effectively yielded a grand spectrum bin for every combined bin, by considering overlapping segments (each grand bin offset by one from the previous)[25]. This ensures no sensitivity loss if an axion’s peak lies between segment boundaries. Our package can either produce a non-overlapping rebinned spectrum (faster, but one might miss signals on segment edges) or an overlapping one. By default, we adopt the overlapping approach: the grand spectrum will have nearly the same number of points as the combined spectrum (each grand bin centered on each combined bin frequency). The weighting takes into account partial signal contributions when an axion is offset from segment center[25]. (In practice, overlapping means the grand spectrum is correlated between points, but HAYSTAC found this correlation effect to be minor and accounted for it separately[30][31].)
After this step, we have the grand spectrum δ^g(ν) with its noise σ^g(ν). If an axion of the assumed lineshape is present at frequency ν_0 with coupling corresponding to our normalization, the grand spectrum should contain a peak of height ~1 (in units of σ) at the bin corresponding to ν_0[24][27]. All bins of δ^g should be approximately standard normal noise (mean ~0, σ~1) in the absence of signal. This is the spectrum in which we apply threshold cuts to look for candidates.
5. SNR Estimation and Candidate Selection (Thresholding & Rescan)
With the grand spectrum in hand, we next determine which (if any) bins are significant outliers that could indicate an axion. We also quantify the sensitivity in terms of SNR and set thresholds to achieve a desired confidence level for exclusion.
Signal-to-Noise Ratio (SNR): In haloscope analysis, the SNR of a potential axion is defined as the ratio of the expected signal power to the uncertainty (noise). Our construction normalized a KSVZ-model axion to have SNR = 1 in the grand spectrum if it were exactly at threshold coupling. However, typically experiments aim for a higher SNR target to claim detection. HAYSTAC chose a target such that a candidate would be noticeable with high probability. In the first run, they set a 5σ target SNR at 95% detection confidence[32][33]. In our package, we define target_snr (e.g. 5.0) and detection_confidence (e.g. 0.95) as parameters.
Threshold Setting: We compute a threshold Θ on the grand spectrum (in units of σ) such that an axion with the target SNR would exceed Θ with the given probability (95%). For example, if we aim for 95% detection of a 5σ signal, the threshold can be derived from the cumulative distribution of a normal with mean 5 and σ 1[34][35]. In HAYSTAC’s case, the threshold was Θ ≈ 3.455σ, because a 5.1σ signal has about a 95% chance to produce a grand-spectrum excess above 3.455σ[36]. In general, our code can compute Θ by solving Φ((Θ - SNR_target)/1) = confidence (where Φ is the Gaussian CDF). We will document this so users can adjust confidence levels. All grand spectrum bins with δ^g > Θ are flagged as rescan candidates[37].
Candidate Flagging: The package function find_candidates(grand_spec, sigma, threshold) (in detection.py) will return a list of candidate frequencies where the grand spectrum exceeds the threshold. We also apply clustering: if multiple adjacent bins are all above threshold, they likely represent one candidate (since a true axion would affect a cluster of bins). We merge such clusters and take the highest bin as the candidate frequency. We also might ignore very closely spaced multiple candidates, as HAYSTAC did (they had rules to handle adjacent candidates within a certain spacing) – these details can be configured.
Rescan Simulation: If any candidates are found, the analysis would normally proceed to rescan those frequencies with additional data to confirm or refute the signal. Our package includes an optional rescan simulation mode. If enabled, for each candidate frequency we simulate an extra set of spectra with increased sensitivity (for example, by assuming longer integration time or improved SNR). We then run the same analysis steps on this subset. Essentially, we treat each candidate’s rescan as a mini-analysis: combine the new spectra (and possibly the old ones) and check the grand spectrum again at that frequency. Because more data is collected, a real axion would appear with higher SNR on rescan, whereas a statistical fluctuation likely will not repeat. The threshold for rescan confirmation can be set (often higher confidence is required). In HAYSTAC’s actual run, 28 initial candidates were rescanned, and none produced a significant excess in the follow-up data[38]. In our simulation, we can mark a candidate as “confirmed” if it exceeds the rescan threshold in the rescan grand spectrum, otherwise it is dismissed as noise.
The rescan simulation is handled by a function rescan_candidate(freq, additional_time_factor) which can take the original data at the candidate frequency and simulate more data (e.g. by reducing noise variance by $\sqrt{\text{time factor}}$). The pipeline then recomputes the grand spectrum for that region. Finally, detection.py will produce an output list of surviving candidates after rescans. (If any survived, that would indicate a potential discovery; but usually none do, leading to setting a limit.)
No Detection Case: If no candidate survives (or none were above threshold to begin with), we proceed to set an exclusion limit. But first, we optionally apply a small correction for the SG filter’s impact on SNR. As noted, the SG baseline removal can attenuate signal power. HAYSTAC analytically and empirically estimated this attenuation (denoted η) and incorporated it into the final limit calculation[39][29]. Our code can include a factor for “SNR efficiency” – for example, if the SG filter yields 85% efficiency, we would effectively require a slightly higher coupling to reach the same SNR. For simplicity, we may assume this correction is ~10–15% and include it as a user-set parameter (snr_efficiency). This will linearly affect the coupling limit.
6. Determining the 95% Exclusion Limit and Outputting Results
Finally, after analyzing the data and finding no persistent candidates, we set a 95% confidence exclusion limit on the axion-photon coupling $|g_{a\gamma\gamma}|$ as a function of frequency (or axion mass). This is the ultimate science result: it tells us the strongest coupling that could still be hidden in the data at each frequency with 95% confidence.
Coupling Limit Calculation: In the simplest terms, if our analysis could have detected an axion of coupling $g_{\text{min}}$ at frequency ν with 95% probability, then $g_{\text{min}}(ν)$ is the limit. We derive this from the threshold SNR. Since SNR ∝ $g_{a\gamma\gamma}^2$ (axion signal power scales with the square of the coupling), the limit coupling is basically the coupling that would produce SNR = SNR_target at that frequency. For example, HAYSTAC normalized their spectra to the KSVZ model (which has a specific coupling $g_{\rm KSVZ}$)[40][41]. They then found that on average $|g_{a\gamma\gamma}|{\min} \approx 2.3 \times g ≈ 5$ times the KSVZ signal (hence requiring a coupling 2.3× larger than KSVZ to be detected at 95% CL)[38]. }$ in the range analyzed[38]. This corresponds to the fact that their SNR target (~5σ) was $\sim\sqrt{2.3^2
In our code, after determining no detections, we compute for each grand spectrum bin ℓ a coupling limit: $$ g_{\min}(ν_ℓ) = g_0 \times \sqrt{\frac{\text{SNR}\text{target}}{\text{observed SNR}\ ,$$ where $g_0$ is the reference coupling used in normalization (e.g. $g_{\rm KSVZ}$), and “observed SNR” could be 1 for a baseline KSVZ-normalized spectrum. In practice, this simplifies: $g_{\min}(ν) = g_{\rm KSVZ} \times R_T^{1/2}$ uniformly if the sensitivity is roughly flat, where $R_T$ is the target SNR in power (e.g. 5.1)[36][38]. If certain frequencies had worse sensitivity (e.g. data missing or higher noise), the code will yield a higher $g_{\min}$ there. For instance, at the intruder-mode frequency 5.704 GHz where HAYSTAC had to cut data, the limit is much weaker (no coverage at that spot)[42]. Our output will reflect such notches.}}
Output Data and Plot: The final output includes the array of $g_{\min}$ vs frequency. The package can save this to a CSV file and also produce a plot similar to Fig. 7.9 of the HAYSTAC thesis. This plot shows the exclusion line over the scanned frequency range, with any gaps or degraded regions indicated, along with a band indicating systematic uncertainty.
Example output plot: 95% confidence exclusion limit on $|g_{a\gamma\gamma}|$ vs frequency (simulated). The green line is the upper limit achieved by the analysis – any axion with coupling above this line at those frequencies is excluded at 95% CL. The light green shaded band is the $\pm1σ$ uncertainty region (here ~4% uncertainty from calibration of noise power, similar to HAYSTAC[38]). The large notch at ~5.704 GHz reflects a frequency range where the experiment had no sensitivity (e.g. a parasitic cavity mode caused data to be cut[42]). Narrow notches show frequencies where synthetic axion signals were injected for calibration and subsequently removed from the data[43][44]. The inset (if included) would compare this result (green) with previous experiments (magenta, blue, cyan) and the theoretical axion model band (yellow)[42], as in the HAYSTAC publication.
From this plot, one can quote the limit. For example, in this simulated first run, we exclude $|g_{a\gamma\gamma}| \gtrsim 2\times10^{-14}$ GeV$^{-1}$ (about 2–3 times the KSVZ benchmark) over the 5.6–5.8 GHz range[45][38]. Any would-be axion in that range with stronger coupling is ruled out with 95% confidence by our analysis.
Modular Package Design and Usage
We emphasize that each part of this analysis is modular and reusable. The code is structured as a Python package (e.g. axion_haloscope_analysis) with submodules corresponding to each major step:
simulation.py – Functions to generate synthetic spectra or to ingest real spectral data. Allows configuration of noise level, bin width, number of spectra, injected signals, etc.
baseline.py – Implements baseline removal (Savitzky–Golay filter). e.g. remove_baseline(spec, window, poly) returns a baseline-flattened spectrum. This module also contains utilities for identifying and masking bad bins (like known RFI spikes) before combination.
combine.py – Functions for vertical combination of spectra. For example, combine_spectra(spec_list) takes a list of processed spectra and produces the combined spectrum with ML weighting[14][18]. It handles frequency alignment and weight calculations (assuming each spectrum object carries information about its noise or uses internal estimates).
rebin.py – Tools for horizontal rebinning. It might define axion_lineshape = compute_lineshape(v_dispersion, ... ) to get the template shape, and construct_grand_spectrum(combined_spec, lineshape) to perform the matched filtering (ML horizontal sum)[23]. This yields the grand spectrum ready for thresholding.
detection.py – Contains the logic for SNR calculation, threshold setting, and candidate identification. For instance, threshold = determine_threshold(confidence=0.95, target_snr=5) computes the cut Θ[46]. find_candidates(grand_spec, threshold) returns candidate indices. If rescan is enabled, rescan_and_confirm(candidates, data) will simulate additional data for each and re-check the grand spectrum (possibly using functions in the above modules).
limit.py – Functions to compute the coupling exclusion limit once the analysis is complete. For example, compute_limit(grand_spec, noise, target_snr) applies the relationship $g_{\min}(ν) \propto \sqrt{\text{target SNR}}$ (accounting for any frequency-dependent noise or missing data). Also includes plot_limit(frequencies, g_min, uncertainty) to create a publication-quality plot of the exclusion line. We utilize matplotlib for plotting.
Each module is documented with clear docstrings and usage examples, facilitating understanding. For instance, the docstring of remove_baseline will explain the SG filter method and note the impact on signal amplitude (so users can adjust poly_order/window_length or apply an SNR correction factor as needed).
The package is designed such that one can easily swap in real data for the simulation. Real spectra (after appropriate calibration) can be fed into the combine_spectra function directly. The modular design also means improvements can be made in one part (say, using a more sophisticated filter or a Bayesian weights method) without altering the rest of the chain, as long as the interface (input/output formats) remains consistent.
Conclusion
We have outlined a comprehensive Python package for axion haloscope data analysis, faithfully implementing the steps from HAYSTAC’s thesis Chapter 7:
We simulate power spectra with noise and possible axion signals,
remove noise baselines with Savitzky–Golay filtering[7],
combine multiple spectra with maximum-likelihood weighting[1],
rebin and filter the combined spectrum according to the axion signal lineshape[23],
identify candidates with a threshold that guarantees 95% detection efficiency[46],
and finally set a 95% CL exclusion limit on the axion coupling across the scanned frequencies[38].
Throughout, we provide flexibility (rescan handling, tunable filter parameters) and clear documentation. This modular toolkit not only serves simulated studies (e.g. to validate analysis or optimize parameters) but can be adapted to real haloscope data, helping the community apply a proven analysis framework (HAYSTAC/ADMX-style) to new axion search experiments[1]. The end result – as visualized in our example plot – is a scientifically informative limit on axion presence, which can be directly compared against theoretical models and other experiments’ results, all obtained with a well-structured, reproducible analysis pipeline.
Sources: This design is based on the HAYSTAC collaboration’s described procedure[47][1] and subsequent refinements in the literature.

[1] [2] [3] [4] [6] [8] [9] [11] [12] [13] [14] [15] [16] [17] [18] [19] [20] [21] [22] [23] [24] [25] [26] [27] [28] [29] [30] [31] [32] [33] [34] [35] [36] [37] [39] [40] [41] [46] link.aps.org
https://link.aps.org/accepted/10.1103/PhysRevD.96.123008
[5] [7] arxiv.org
https://arxiv.org/pdf/2503.04288
[10] (PDF) First Results from an Axion Haloscope at CAPP around 10.7 μ eV
https://www.researchgate.net/publication/351536043_First_Results_from_an_Axion_Haloscope_at_CAPP_around_107_m_eV
[38] [43] [44] Our exclusion limit at 90% confidence. The light green shaded region is... | Download Scientific Diagram
https://www.researchgate.net/figure/Our-exclusion-limit-at-90-confidence-The-light-green-shaded-region-is-a-1s-error-band_fig3_386726876
[42] (PDF) First results from a microwave cavity axion search at 24 micro-eV
https://www.researchgate.net/publication/386726876_First_results_from_a_microwave_cavity_axion_search_at_24_micro-eV
[45] [1801.00835] First results from the HAYSTAC axion search
https://arxiv.org/abs/1801.00835
[47] HAYSTAC axion search analysis procedure | Phys. Rev. D
https://journals.aps.org/prd/abstract/10.1103/PhysRevD.96.123008


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

