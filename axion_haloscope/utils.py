from pathlib import Path
import yaml

def find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / ".git").exists():      # or pyproject.toml
            return path
    raise RuntimeError("Could not find project root")

def _get(d, key, default):
    v = d.get(key, default)
    return default if v is None else v

def load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    inp  = raw.get("input",      {}) or {}
    sim  = raw.get("simulation", {}) or {}
    inj  = raw.get("injection",  {}) or {}
    qc   = raw.get("quality",    {}) or {}
    alg  = raw.get("alignment",  {}) or {}
    base = raw.get("baseline",   {}) or {}
    rb   = raw.get("rebin",      {}) or {}
    det  = raw.get("detection",  {}) or {}
    out  = raw.get("output",     {}) or {}
    diag = raw.get("diagnostic", {}) or {}

    cfg = {
        "input": {
            "input_mode":             str(_get(inp, "input_mode", "simulation")),
            "directory":              str(_get(inp, "directory", "scripts/qshs")),
            "input_file_name":        str(_get(inp, "input_file_name", "converted_spectra.h5")),
        },
        "simulation": {
            "n_spectra":      int(_get(sim, "n_spectra", 80)),
            "n_bins":         int(_get(sim, "n_bins", 8000)),
            "bin_width_hz":   float(_get(sim, "bin_width_hz", 100.0)),
            "f_start_hz":     float(_get(sim, "f_start_hz", 5.70e9)),
            "tune_step_bins": int(_get(sim, "tune_step_bins", 100)),
            "rng_seed":       int(_get(sim, "rng_seed", 1234)),
            "noise_sigma":    float(_get(sim, "noise_sigma", 1.0)),
        },
        "injection": {
            "enabled":     bool(_get(inj, "enabled", False)),
            "f_axion_hz":  inj.get("f_axion_hz", None),  # optional
            "total_power": float(_get(inj, "total_power", 20.0)),
        },
        "quality": {
            "max_power_filter":         bool(_get(qc, "max_power_filter", True)),
            "p_max":                    float(_get(qc, "p_max", 1e-8)),
            "noise_filter":             bool(_get(qc, "noise_filter", True)),
            "rms_max":                  float(_get(qc, "rms_max", 1e-10)),
            "nan_fail":                 bool(_get(qc, "nan_fail", True)),
            "robust":                   bool(_get(qc, "robust", True)), 
            "small_bandwidth_filter":   bool(_get(qc, "small_bandwidth_filter", True)),
            "bw_min":                   float(_get(qc, "bw_min", 0.00027)),
            "bandwidth_zeros_filter":   bool(_get(qc, "bandwidth_zeros_filter", True)),
            "res_freq_zeros_filter":    bool(_get(qc, "res_freq_zeros_filter", True)),
            "cw_freq_zeros_filter":     bool(_get(qc, "cw_freq_zeros_filter", True)),
            "bad_time_filter":          bool(_get(qc, "bad_time_filter", True)),
            "start_time":               _get(qc, "start_time", None),
            "end_time":                 _get(qc, "end_time", None),
            "data_cleaning":            bool(_get(qc, "data_cleaning", False)),
        },
        "alignment": {
            "gain_alignment": bool(_get(alg, "gain_alignment", True))
        },
        "baseline": {
            "sg_window_warm": int(_get(base, "sg_window_warm", 251)),
            "sg_poly_warm":   int(_get(base, "sg_poly_warm", 2)),
            "sg_window_cold": int(_get(base, "sg_window_cold", 401)),
            "sg_poly_cold":   int(_get(base, "sg_poly_cold", 4)),
            "spacing_minutes":float(_get(base, "spacing_minutes", 30)),
            "sigma_cut":      float(_get(base, "sigma_cut", 3.5)),
            "clipping_mode":  str(_get(base, "clipping_mode", "Claude")),
            "n_iterations":   int(_get(base, "n_iterations", 3)),
        },
        "rebin": {
            "C": int(_get(rb, "C", 10)),
            "K": int(_get(rb, "K", 9)),
        },
        "detection": {
            "target_snr": float(_get(det, "target_snr", 5.0)),
            "confidence": float(_get(det, "confidence", 0.95)),
            "snr_eff":    float(_get(det, "snr_eff", 0.90)),
            "g0":         float(_get(det, "g0", 1.0)),
        },
        "output": {
            "save_data":                bool(_get(out, "save_data", False)),
            "plots_step":               int(_get(out, "plots_step", 1)),   # plot every Nth spectrum
            "max_plots":                out.get("max_plots", None),        # optional int
            "root":                     _get(out, "root", "output"),
            "subdir_prefix":            _get(out, "subdir_prefix", "run"),
        },
        "diagnostic": {
            "run_diagnostics":          bool(_get(diag, "run_diagnostics", False)),
            "save_raw_plots":           bool(_get(diag, "save_raw_plots", False)),
            "combined_plot":            bool(_get(diag, "combined_plot", False)),
            "offset_combined_plot":     bool(_get(diag, "offset_combined_plot", False)),
            "injection_distribution":   bool(_get(diag, "injection_distribution", False)),
            "set_average_diagnostics":  bool(_get(diag, "set_average_diagnostics", False)),
            "clipping_residuals":       bool(_get(diag, "clipping_residuals", False)),
            "varying_set_size":         bool(_get(diag, "varying_set_size", False)),
                        
        }
    }
    return cfg