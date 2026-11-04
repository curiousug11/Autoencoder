# Spectrum autoencoder — datasets

Data-only branch. Contains just the `data/` files used by the code on the main branches.

## Data (`data/`)

| File | Shape | |
|---|---|---|
| `pos_spec_binned.npy` | (98725, 52) | binned spectra (transit depth, fraction) |
| `pos_params_clean.npy` | (98725, 9) | planet parameters per spectrum |
| `prt_spec_binned.npy` | (98725, 52) | binned spectra from petitRADTRANS |
| `prt_params_clean.npy` | (98725, 9) | parameters for the petitRADTRANS spectra |
| `taurex_spec_binned.npy` | (98725, 52) | binned spectra from TauREx |
| `taurex_params_clean.npy` | (98725, 9) | parameters for the TauREx spectra |
| `wavelength_binned.npy` | (52,) | wavelength grid (µm, descending) |

All three parameter files are identical: row *i* is the same planet in POS, petitRADTRANS and TauREx.
