"""Convert a DWSIM material-stream export (Master Property Table saved as CSV) of the ammonia
synthesis flowsheet into reference_data/dwsim/ammonia_loop_streams.csv.

    python scripts/import_dwsim_loop_streams.py material_streams_nh3_synthesis.csv

DWSIM column names are mapped to the names used by the tests (feed, comp_out, hot, rx_out, cold,
purge, nh3_liquid). Only rows that do not contain a comma in their name are used (DWSIM does not
quote the others, e.g. "Fugacity Coefficient, Vapor Phase / Hydrogen").
The displayed "Molar Enthalpy" of DWSIM leaves out the formation enthalpy; "Energy Flow" includes it,
so the CSV also stores energy_flow_per_kmol_kJ = Energy Flow / molar flow.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reference_data' / 'dwsim' / 'ammonia_loop_streams.csv'

COLUMNS = {'feed': 'feed', 'out': 'comp_out', 'Hot': 'hot', 'Rx-Vapor': 'rx_out',
           'Cold': 'cold', 'Purge-Gas': 'purge', 'NH3-Liquid': 'nh3_liquid'}
SPECIES = {'H2': 'Hydrogen', 'N2': 'Nitrogen', 'NH3': 'Ammonia'}


def read_export(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        rows = list(csv.reader(f))
    header = rows[0]
    table = {}
    for r in rows[1:]:
        if len(r) == len(header) and r[0]:
            table[r[0]] = dict(zip(header[2:], r[2:]))
    return table


def build(table):
    def val(row, col):
        s = table[row][col].strip()
        return float(s) if s != '' else np.nan

    out = []
    for col, name in COLUMNS.items():
        vf = val('Molar Fraction (Vapor)', col)
        has_v, has_l = vf > 0, vf < 1
        n = {k: val(f'Molar Flow (Mixture) / {v}', col) for k, v in SPECIES.items()}
        ntot = sum(n.values())
        rec = {'stream': name, 'T_K': val('Temperature', col), 'P_bar': val('Pressure', col),
               **{f'n_{k}_mol_s': n[k] for k in SPECIES}, 'vapor_fraction': vf,
               'Z_vapor': val('Compressibility Factor (Vapor)', col) if has_v else np.nan,
               'Z_liquid': val('Compressibility Factor (Overall Liquid)', col) if has_l else np.nan,
               'rho_mix_kg_m3': val('Density (Mixture)', col),
               'rho_vapor_kg_m3': val('Density (Vapor)', col) if has_v else np.nan,
               'rho_liquid_kg_m3': val('Density (Overall Liquid)', col) if has_l else np.nan,
               'H_kJ_kmol': val('Molar Enthalpy (Mixture)', col),
               'S_kJ_kmol_K': val('Molar Entropy (Mixture)', col),
               'H_vapor_kJ_kmol': val('Molar Enthalpy (Vapor)', col) if has_v else np.nan,
               'H_liquid_kJ_kmol': val('Molar Enthalpy (Overall Liquid)', col) if has_l else np.nan,
               'energy_flow_kW': val('Energy Flow', col)}
        for k, v in SPECIES.items():
            rec[f'y_{k}'] = val(f'Molar Fraction (Vapor) / {v}', col) if has_v else np.nan
            rec[f'x_{k}'] = val(f'Molar Fraction (Overall Liquid) / {v}', col) if has_l else np.nan
        # the file lists y_H2, x_H2, y_N2, x_N2, y_NH3, x_NH3 in this order
        rec['Cp_ideal_vapor_kJ_kmol_K'] = (val('Ideal Gas Heat Capacity Cp (Vapor)', col)
                                           * val('Molecular Weight (Vapor)', col)) if has_v else np.nan
        rec['energy_flow_per_kmol_kJ'] = rec['energy_flow_kW'] / ntot * 1000.0
        out.append(rec)
    df = pd.DataFrame(out)
    order = ['stream', 'T_K', 'P_bar', 'n_H2_mol_s', 'n_N2_mol_s', 'n_NH3_mol_s', 'vapor_fraction',
             'Z_vapor', 'Z_liquid', 'rho_mix_kg_m3', 'rho_vapor_kg_m3', 'rho_liquid_kg_m3',
             'H_kJ_kmol', 'S_kJ_kmol_K', 'H_vapor_kJ_kmol', 'H_liquid_kJ_kmol', 'energy_flow_kW',
             'y_H2', 'x_H2', 'y_N2', 'x_N2', 'y_NH3', 'x_NH3', 'Cp_ideal_vapor_kJ_kmol_K',
             'energy_flow_per_kmol_kJ']
    return df[order]


if __name__ == '__main__':
    df = build(read_export(sys.argv[1]))
    df.to_csv(OUT, index=False)
    print(f'wrote {OUT}')
    print(df.set_index('stream').T.to_string())
