import numpy as np
from axion_haloscope.io_working import SpectrumSet
from datetime import datetime
from axion_haloscope.io_working import SpectrumSet

def cut_by_datetime(data, start, end, key="date"):
    '''
    Filters data by a predetermined time range
    '''

    specs, fper, rf, rf_map, metadata = data.spectra, data.freqs_per_spec, data.rf_grid, data.rf_index_map, data.metadata

    dt = np.array([
        datetime.strptime(str(x), "%Y-%m-%d %H:%M:%S") if x is not None else None
        for x in metadata[key]
    ])

    start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d %H:%M:%S")

    mask = np.array([(d is not None) and (start_dt <= d <= end_dt) for d in dt])

    spectra  = [b for a, b in zip(mask, specs) if a]
    freqs_per_spec  = [b for a, b in zip(mask, fper) if a]
    rf_index_map  = [b for a, b in zip(mask, rf_map) if a]

    removed = [[metadata["file_name"][i], "not in good time range", metadata[key][i]]
        for i, keep in enumerate(mask) if not keep]

    invalid = list(metadata.get("invalid_files", []))
    invalid_all = invalid + removed

    spec_metadata = {
        k: (invalid_all if k == "invalid_files" else [val for keep, val in zip(mask, v) if keep])
        for k, v in metadata.items()
    }

    return SpectrumSet(
        spectra=spectra,
        freqs_per_spec=freqs_per_spec,
        rf_grid=rf,
        rf_index_map=rf_index_map,
        metadata=spec_metadata
    ) 

def cut_by_values(sset, cut_min_val, cut_max_val):

    specs, fper, rf, rf_map, metadata = sset.spectra, sset.freqs_per_spec, sset.rf_grid, sset.rf_index_map, sset.metadata

    cut_min_val = -0.3e6
    cut_max_val = 2.3e6
    cut_min_idx = np.abs(fper[0] - cut_min_val).argmin()
    cut_max_idx = np.abs(fper[0] - cut_max_val).argmin()

    new_specs = []
    new_freqs = []
    new_rf_map = []
    for spec, freq, rf_vals in zip(specs, fper, rf_map):
        x = np.where(freq == 0)[0]
        for j in x:
            for i in range(2, -1, -1):
                spec[j+i] = spec[j+i+1]
                spec[j-i-1] = spec[j-i-2]


        spec = spec[cut_min_idx:cut_max_idx]
        freq = freq[cut_min_idx:cut_max_idx]
        rf_vals = rf_vals[cut_min_idx:cut_max_idx]
        

        new_specs.append(spec)
        new_freqs.append(freq)
        new_rf_map.append(rf_vals)

    specs = new_specs
    fper = new_freqs
    rf = rf[cut_min_idx:cut_max_idx]
    rf_map = new_rf_map

    return SpectrumSet(specs, fper, rf, rf_map, metadata)