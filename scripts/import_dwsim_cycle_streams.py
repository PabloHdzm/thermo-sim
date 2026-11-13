"""Convert a DWSIM material-stream export of the propane refrigeration cycle into
reference_data/dwsim/propane_cycle_streams.csv.

    python scripts/import_dwsim_cycle_streams.py material_streams_ref_cycle.csv

Only rows whose number of fields matches the header are used (DWSIM does not quote names that
contain a comma). Pressure is read in bar (a Pa export is converted) and the molar flow in mol/s
(a kmol/s export is converted), following the Units column. The displayed "Molar Enthalpy" of DWSIM
leaves out the formation enthalpy; "Energy Flow" includes it, so energy_flow_per_kmol_kJ =
Energy Flow / molar flow is also stored. Values of an absent phase are stored as empty.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reference_data' / 'dwsim' / 'propane_cycle_streams.csv'


def read_export(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    header = rows[0]
    table, units = {}, {}
    for r in rows[1:]:
        if len(r) == len(header) and r[0]:
            table[r[0]] = dict(zip(header[2:], r[2:]))
            units[r[0]] = r[1].strip()
    return header[2:], table, units


def build(names, table, units):
    p_fac = {'bar': 1.0, 'Pa': 1e-5}[units['Pressure']]
    n_fac = {'mol/s': 1.0, 'kmol/s': 1e3}[units['Molar Flow (Mixture) / Propane']]
    def val(row, col):
        s = table[row][col].strip()
        return float(s) if s != '' else np.nan

    out = []
    for col in names:
        vf = val('Molar Fraction (Vapor)', col)
        has_v, has_l = vf > 0, vf < 1
        n = val('Molar Flow (Mixture) / Propane', col) * n_fac         # mol/s
        rec = {'stream': col, 'T_K': val('Temperature', col), 'P_bar': val('Pressure', col) * p_fac,
               'n_mol_s': n, 'mass_flow_kg_s': val('Mass Flow', col), 'vapor_fraction': vf,
               'Z_vapor': val('Compressibility Factor (Vapor)', col) if has_v else np.nan,
               'Z_liquid': val('Compressibility Factor (Overall Liquid)', col) if has_l else np.nan,
               'rho_mix_kg_m3': val('Density (Mixture)', col),
               'rho_vapor_kg_m3': val('Density (Vapor)', col) if has_v else np.nan,
               'rho_liquid_kg_m3': val('Density (Overall Liquid)', col) if has_l else np.nan,
               'H_kJ_kmol': val('Molar Enthalpy (Mixture)', col),
               'H_vapor_kJ_kmol': val('Molar Enthalpy (Vapor)', col) if has_v else np.nan,
               'H_liquid_kJ_kmol': val('Molar Enthalpy (Overall Liquid)', col) if has_l else np.nan,
               'S_kJ_kmol_K': val('Molar Entropy (Mixture)', col),
               'energy_flow_kW': val('Energy Flow', col),
               'Cp_ideal_vapor_kJ_kmol_K': (val('Ideal Gas Heat Capacity Cp (Vapor)', col)
                                           * val('Molecular Weight (Vapor)', col)) if has_v else np.nan,
               'T_bubble_K': val('Bubble Temperature at Stream Pressure', col),
               'T_dew_K': val('Dew Temperature at Stream Pressure', col),
               'energy_flow_per_kmol_kJ': val('Energy Flow', col) / (n * 1e-3) }
        out.append(rec)
    return pd.DataFrame(out)


if __name__ == '__main__':
    df = build(*read_export(sys.argv[1]))
    df.to_csv(OUT, index=False)
    print(f'wrote {OUT}')
    print(df.set_index('stream').T.to_string())