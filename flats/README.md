# Flat fields

Instrument calibration files used by the scan program and the analysis (not in git: they are large binaries and specific
to the microscope, camera and lamp):

| file | use |
|---|---|
| `flat_10x_3ms.npz` | 10x tiles (`--flat`, scan setting `flat10`) |
| `flat_100x_285nm.npz` | 100x images on ~285 nm SiO2 (`--flat100 285=...`, scan setting `flat100`) |
| `flat_100x_90nm.npz` | 100x images on ~90 nm SiO2 |

Put copies here to build the microscope PC kit (`sh tools/win7/make_kit.sh` copies them into the kit). New ones are taken
with the scan program's flat mode (`python -m flakeid.scope --mode flat`). Keep the current set on the lab share.
