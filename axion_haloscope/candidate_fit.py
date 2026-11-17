from __future__ import annotations

"""
Some extra functions for fitting Maxwell-Boltzmann distributions to each candidate peak, picked out by find_candidates()
All candidates are fitted, and then filtered based on chi^2 and r^2
For QSHS it checks the found candidates and fits against known injection frequencies.
"""

from typing import Tuple
import numpy as np
from typing import Optional
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def max_bolt(f, a, mu, sigma):
    """
    Maxwell-Boltzmann distribution function.

    Parameters
    ----------
    f : array of frequencies from grand spec
    a : float, amplitude of distribution (based on z score)
    mu : float, mean frequency
    sigma : float, standard deviation

    Returns
    -------
    array of amplitudes related to the 
        Maxwell-Boltzmann distribution evaluated at `f`.
    """
    return a * (f**2) * np.exp(-0.5 * ((f - mu) / sigma) ** 2)

def max_bolt_band(f, params, pcov, z=1.96):

    """
    Calculate the uncertainty band for the MB fit using the covariance matrix.

    Parameters
    ---------
    f: array of frequencies from grand spec
    params: parameters for MB distribution function
    pcov: covariance of MB fit
    z: offset with z score values
    """

    a, mu, sigma = params
    delta = f - mu
    exp_term = np.exp(-0.5 * (delta / sigma) ** 2)
    model = a * f**2 * exp_term

    # gradient of max_bolt with respect to (a, mu, sigma)
    jacobian = np.column_stack((
        f**2 * exp_term,
        model * delta / sigma**2,
        model * delta**2 / sigma**3,
    ))

    variance = np.einsum("ij,jk,ik->i", jacobian, pcov, jacobian)
    # works out variance using Einstein sum of generalised derivative (jacobian), with the covariance parameters
    uncertainty = np.sqrt(np.maximum(variance, 0))
    return model, model - z * uncertainty, model + z * uncertainty #one sd above and below

def red_chisq(data_y,model_y,dof): # RSS divided by degrees of freedom
    residuals = data_y - model_y 
    rss = np.sum(residuals**2)
    return rss/dof; # dof is no. of data minus paramters to find

def adj_r_squared(data_x,data_y,model_y,mean):
    # Calculate the R^2 for the fit
    R2 = 1 - np.sum((data_y-model_y)**2)/np.sum((data_y-mean)**2)      # The standard definition of R^2
    R2_adjusted = 1-(1-R2)*(len(data_x)-1)/(len(data_x)-2)             # The "adjusted" R^2
    return R2_adjusted;


def fit_to_candidates(
    grand_freqs: np.ndarray,
    grand_psd: np.ndarray,
    candidates: np.ndarray,
    main_plots_dir: str,
    colour_vals: Optional[np.ndarray]=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit a Maxwell-Boltzmann distribution to each candidate in the grand spectrum.
    Use the candidate frequency and z value as the initial guess for the maximum
    Assume the candidates are already distinct due to find_candidates()

    Parameters
    ----------
    grand_freqs : np.ndarray
        The grand frequency data from the combined spectra.
    grand_psd : np.ndarray
        This is actually the z score.
    candidates : np.ndarray
        Indices of candidate peaks in the grand spectrum, found with find_candidates().
    main_plots_dir: str
        where the output plots get saved to. At the moment it is just the main_plots directory
    colour_vals: np.ndarray
        frequency shifted values for injection frequencies
    

    Returns
    -------
    fit_params : np.ndarray
        Fitted parameters for each candidate (amplitude, mean, stddev).
    fit_errors : np.ndarray
        Errors associated with the fitted parameters.
    """

    colours_agree = []
    cands_disagree_index = []
    cand_index=[]
    plt.figure()
    if colour_vals is not None:
        for i in range(len(candidates)):
                cand_freq=1e9*grand_freqs[candidates[i]] # need to convert to common frequency scale
                for m in range (len(colour_vals)):
                    if np.abs(cand_freq/1e6-colour_vals[m]*1e3) < 0.01: # anything between 1 and 10 kHz seems reasonable compared to axion linewidth
                        cand_index.append(candidates[i])
                        colours_agree.append(colour_vals[m])
                    else:
                        cands_disagree_index.append(candidates[i])
    fit_params = []
    fit_errors = []
    plt.plot(grand_freqs, grand_psd, label='Grand Spectrum', color='cornflowerblue',linewidth=-0.5)
    if colour_vals is not None:
        plt.scatter(grand_freqs[cands_disagree_index], grand_psd[cands_disagree_index], color='red', label='Candidates not near injections')
        plt.scatter(grand_freqs[cand_index], grand_psd[cand_index], color='green',label='Candidates within 10 kHz of injections')
    positive=0
    negative=0
    for i in range(len(candidates)):
        idx = candidates[i]
        m=2
        # Define a window around the candidate for fitting (might need configuring tbh)
        # QSHS injected axions are all very close together
        start_idx = max(0, idx - m)  # 5 bins to the left
        end_idx = min(len(grand_freqs), idx + m + 1)  # 5 bins to the right
        dof=len(grand_freqs[start_idx:end_idx])-3 # degrees of freedom for reduced chi-squared
        freq_guess= grand_freqs[idx]
        psd_guess= grand_psd[idx]
        x_data = grand_freqs[start_idx:end_idx]
        
        # initial guess for Gaussian parameters: amplitude, mean, sigma
        initial_guess = [1/freq_guess**2, freq_guess, np.std(grand_freqs[start_idx:end_idx])] 
        # assuming the candidate frequency is the mean, and candidate amplitude gives estimate for amplitude
        # standard deviation is just a guess atm, based off of number of bins etc.

        try:
            popt, pcov = curve_fit( max_bolt, x_data, grand_psd[start_idx:end_idx], p0=initial_guess) 
            fit_params.append(popt)
            fit_errors.append(np.sqrt(np.diag(pcov)))
        except RuntimeError:
            # If the fit fails, append NaNs
            fit_params.append([np.nan, np.nan, np.nan])
            fit_errors.append([np.nan, np.nan, np.nan])

        fitted_x=x_data
        fitted_y=max_bolt(fitted_x, *popt)
        adjr=adj_r_squared(fitted_x,grand_psd[start_idx:end_idx],fitted_y,np.mean(grand_psd[start_idx:end_idx]))
     
        fit_freqs = np.linspace(x_data.min(), x_data.max(), 200)  
        fitted_y, lower, upper = max_bolt_band(fitted_x, popt, pcov)
        chisq=red_chisq(grand_psd[start_idx:end_idx],fitted_y,dof)
        fitted_y, lower, upper = max_bolt_band(fit_freqs, popt, pcov)
        
        if chisq<0.1 and np.max(fitted_y)<1.5*np.max(grand_psd[start_idx:end_idx]) and adjr>0.9:
            plt.plot(fit_freqs, fitted_y, linewidth=1,linestyle="--", color='black')
            plt.fill_between(fit_freqs, lower, upper, alpha=0.5,color='gray')
            if  idx in cand_index:
                positive=positive+1
            else:
                negative=negative+1
            #plt.plot(fitted_x, fitted_y,   label=f"R^2={adjr:.2f}",ls='--')
    print(f'Correct identifications: {positive}, false identifications: {negative}')
        
    plt.title(f'MB fit with χ²<0.1, R²>0.9')
    plt.xlabel('Frequency [GHz]')
    plt.ylim(-1,6)
    plt.xlim(0,0.0015)
    plt.ylabel('Z-score')
    plt.grid()
    plt.legend()
    plt.savefig(f'{main_plots_dir}/candidate_fit{m}.png',dpi=150)
    plt.close()

    return np.array(fit_params), np.array(fit_errors)