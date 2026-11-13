import warnings
import numpy as np
import pandas as pd
import CoolProp
import CoolProp.CoolProp as CP
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit, least_squares
from pathlib import Path
warnings.filterwarnings("ignore", category=RuntimeWarning)
from thermo_sim.constants import R_GAS

ROOT     = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "src" / "thermo_sim" / "data"
RAW_DIR  = DATA_DIR / "raw"
DOCS_DIR = ROOT / "docs"

COMPONENTS = {"Water": "Water", "Methanol": "Methanol", "Ethanol": "Ethanol",
              "Benzene": "Benzene", "Toluene": "Toluene", "Acetone": "Acetone",
              "CarbonDioxide": "CarbonDioxide", "Nitrogen": "Nitrogen",
              "Oxygen": "Oxygen", "Methane": "Methane", "Ammonia": "Ammonia",
              "Hydrogen": "Hydrogen", "Propane": "Propane",}

GAS_T_LOW = 250.0     # K, lower limit of the gas transport fits (viscosity, k)
GAS_CP_T_LOW = 200.0  # K, lower limit of the ideal-gas Cp fit (clipped to the EOS Tmin) 
GAS_CP_DEG = 6        # degree of the ideal-gas Cp polynomial (columns A_Cp_Gas ... G_Cp_Gas)
GAS_T_HIGH = 1000.0   # K, clipped to the EOS Tmax
LIQ_TR_MIN = 0.42     # minimum Tr for the liquid (raised to Ttriple if that is higher)
LIQ_TR_MAX = 0.90     # maximum Tr for liquid fits (Cp, viscosity, k, rho)
HVAP_TR_MAX = 0.98    # maximum Tr for the Hvap fit
BAR_TO_MMHG = 750.01   # same factor used by _psat_antoine
ANT_TR_MAX = 0.98      # maximum Tr for the Antoine fit
N_PTS = 60

COL = {
    "Tc": "Tc (K)", "Pc": "Pc (bar)", "Vc": "Vc (cm3/mol)", "w": "w",
    "MW": "MM", "hf": "hf_298_k", "gf": "gf_298_k",
}

# moles of gaseous elements (H2, N2, O2) in the formation reaction of one mole of compound;
# graphite is a solid and does not count (propane: 3 C + 4 H2). Used to refer the formation entropy to Pref.
N_GAS_EL = {'water': 1.5, 'methanol': 2.5, 'ethanol': 3.5, 'benzene': 3.0, 'toluene': 4.0,
            'acetone': 3.5, 'carbondioxide': 1.0, 'nitrogen': 1.0, 'oxygen': 1.0,
            'methane': 2.0, 'ammonia': 2.0, 'hydrogen': 1.0, 'propane': 4.0}

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def state(fluid):
    return CP.AbstractState("HEOS", fluid)

def poly_fit(T, y, deg):
    """Polynomial fit A + B T + C T^2 ... with coefficients in raw T."""
    p = Polynomial.fit(T, y, deg, w=1.0 / np.abs(y))   # minimizes relative error
    c = p.convert().coef
    return np.pad(c, (0, deg + 1 - len(c)))

def sample(fn, Ts):
    out_T, out_y = [], []
    for T in Ts:
        try:
            v = fn(T)
            if np.isfinite(v):
                out_T.append(T); out_y.append(v)
        except Exception:
            pass
    return np.array(out_T), np.array(out_y)

def errs(y_fit, y_ref):
    e = 100 * np.abs(y_fit - y_ref) / np.abs(y_ref)
    return e.max(), e.mean()

# ----------------------------------------------------------------------
# CoolProp read functions
# ----------------------------------------------------------------------
def cp0_gas(fl, T):          # J/mol/K
    s = state(fl); s.update(CoolProp.DmolarT_INPUTS, 1e-3, T)
    return s.cp0molar()

def dilute(fl, T, what):     # dilute-gas limit
    s = state(fl); s.update(CoolProp.DmolarT_INPUTS, 1e-3, T)
    return s.viscosity() if what == "mu" else s.conductivity()

def sat_liq(fl, T, what):
    s = state(fl); s.update(CoolProp.QT_INPUTS, 0, T)
    return {"cp": s.cpmolar, "rho": s.rhomass, "mu": s.viscosity,
            "k": s.conductivity}[what]()

def hvap(fl, T):             # kJ/mol
    s = state(fl); s.update(CoolProp.QT_INPUTS, 1, T); hg = s.hmolar()
    s.update(CoolProp.QT_INPUTS, 0, T); hl = s.hmolar()
    return (hg - hl) / 1000.0
def psat_bar(fl, T):
    s = state(fl); s.update(CoolProp.QT_INPUTS, 0, T)
    return s.p() / 1e5

# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------
def build_component(name, fl):
    s = state(fl)
    Tc, Pc = s.T_critical(), s.p_critical()
    Mw = s.molar_mass() * 1000.0                      # g/mol
    Vc = 1.0 / s.rhomolar_critical() * 1e6            # cm3/mol
    Tt = s.Ttriple()
    Tmin_eos, Tmax_eos = s.Tmin(), s.Tmax()

    row = {"Component": name, "CoolProp_fluid": fl, COL["MW"]: Mw,
           COL["Tc"]: Tc, COL["Pc"]: Pc / 1e5, COL["Vc"]: Vc,
           COL["w"]: s.acentric_factor()}
    q = []   # fit-quality rows

    def log(prop, T, y_fit, y_ref):
        mx, mn = errs(y_fit, y_ref)
        q.append({"Component": name, "property": prop, "T_min": T.min(),
                  "T_max": T.max(), "max_err_%": mx, "mean_err_%": mn})

    # Ranges
    Tg = np.linspace(max(Tmin_eos, GAS_T_LOW), min(Tmax_eos, GAS_T_HIGH), N_PTS)
    T_lo = max(Tt, Tmin_eos, LIQ_TR_MIN * Tc) + 0.5
    Tl = np.linspace(T_lo, LIQ_TR_MAX * Tc, N_PTS)
    Th = np.linspace(T_lo, HVAP_TR_MAX * Tc, N_PTS)

    # --- Ideal-gas Cp (polynomial of degree GAS_CP_DEG, Chebyshev nodes) ---
    # Chebyshev nodes keep the fit accurate over the whole range, including the low-T end
    # (a uniform grid leaves it worse fitted).
    T_lo_cp, T_hi_cp = max(Tmin_eos, GAS_CP_T_LOW), GAS_T_HIGH   # cp0 is analytic above Tmax_eos
    k_nodes = np.arange(N_PTS)
    Tg_cp = (0.5 * (T_lo_cp + T_hi_cp)
             - 0.5 * (T_hi_cp - T_lo_cp) * np.cos(np.pi * (k_nodes + 0.5) / N_PTS))
    T, y = sample(lambda t: cp0_gas(fl, t), Tg_cp)
    c = poly_fit(T, y, GAS_CP_DEG)
    for k, L in enumerate("ABCDEFGH"[:GAS_CP_DEG + 1]): row[f"{L}_Cp_Gas"] = c[k]
    row["Tmin_Cp_Gas"], row["Tmax_Cp_Gas"] = T_lo_cp, T_hi_cp     # fitted interval (nodes do not reach the ends)
    log("Cp_gas", T, np.polynomial.polynomial.polyval(T, c), y)

    # --- Saturated-liquid Cp (cubic) ---
    T, y = sample(lambda t: sat_liq(fl, t, "cp"), Tl)
    c = poly_fit(T, y, 3)
    for k, L in enumerate("ABCD"): row[f"{L}_Cp_Liquid"] = c[k]
    row["Tmin_Cp_Liquid"], row["Tmax_Cp_Liquid"] = T.min(), T.max()
    log("Cp_liq", T, np.polynomial.polynomial.polyval(T, c), y)

    # --- Hvap = A * tau^n  (kJ/mol) ---
    T, y = sample(lambda t: hvap(fl, t), Th)
    tau = 1 - T / Tc
    n0, lnA = np.polyfit(np.log(tau), np.log(y), 1)
    f = lambda tt, A, n: A * (1 - tt / Tc) ** n
    (A, n), _ = curve_fit(f, T, y, p0=[np.exp(lnA), n0], sigma=y)
    row["A_HVav"], row["n_HVav"] = A, n
    row["Tmin_HVav"], row["Tmax_HVav"] = T.min(), T.max()
    log("Hvap", T, f(T, A, n), y)

    # --- Extended Antoine: log10(P[mmHg]) = A + B/T + C log10(T) + D T + E T^2 ---
    T_ant = np.linspace(T_lo, ANT_TR_MAX * Tc, 80)
    T, y = sample(lambda t: np.log10(psat_bar(fl, t) * BAR_TO_MMHG), T_ant)
    M = np.column_stack([np.ones_like(T), 1 / T, np.log10(T), T, T**2])
    sc = np.abs(M).max(axis=0)                      # scaling: the columns are strongly collinear
    coef = np.linalg.lstsq(M / sc, y, rcond=None)[0] / sc
    for k, L in enumerate("ABCDE"):
        row[f"{L}_Antoine_K"] = coef[k]
    row["Tmin_Antoine"], row["Tmax_Antoine"] = T.min(), T.max()
    log("Psat (Antoine)", T, 10 ** (M @ coef) / BAR_TO_MMHG, 10 ** y / BAR_TO_MMHG)

    # --- Liquid density: Yaws-type form  rho = A * B^-(1-T/Tc)^n  (kg/m3) ---
    T, y = sample(lambda t: sat_liq(fl, t, "rho"), Tl)
    rho_c = Mw / Vc * 1000.0
    f = lambda tt, A, B, n: A * B ** (-(1 - tt / Tc) ** n)
    (A, B, n), _ = curve_fit(f, T, y, p0=[rho_c, 0.27, 0.28], sigma=y,
                             bounds=([0, 0.01, 0.01], [1e5, 5, 2]), maxfev=20000)
    row["A_Rho_liq"], row["B_Rho_liq"], row["n_Rho_liq"] = A, B, n
    row["Tmin_Rho_liq"], row["Tmax_Rho_liq"] = T.min(), T.max()
    log("Rho_liq (A,B,n)", T, f(T, A, B, n), y)

    # --- Liquid density: Rackett  rho = M Pc / (R Tc Z^alpha) ---
    def rackett(tt, Z):
        al = 1 + (1 - tt / Tc) ** (2 / 7)
        return (Mw / 1000.0) * Pc / (R_GAS * Tc * Z ** al)
    r = least_squares(lambda z: (rackett(T, z[0]) - y) / y, x0=[0.27])
    row["Z_RA"] = r.x[0]
    log("Rho_liq (Rackett)", T, rackett(T, r.x[0]), y)

    # --- Gas viscosity (uP, cubic) ---
    T, y = sample(lambda t: dilute(fl, t, "mu") * 1e7, Tg)
    if len(T) < 5:   # CoolProp has no transport model (e.g. acetone)
        for L in "ABCD": row[f"{L}_Gas_Viscos_uP"] = np.nan
        for L in "ABCD": row[f"{L}_Liq_Viscos_cP"] = np.nan
        for L in "ABC": row[f"{L}_K_gas-Thermal_Cond"] = np.nan
        for L in "ABC": row[f"{L}_K_liq-Thermal_Cond"] = np.nan
        row["transport_note"] = "Not available in CoolProp: must be supplied by the user"
        return row, q
    c = poly_fit(T, y, 3)
    for k, L in enumerate("ABCD"): row[f"{L}_Gas_Viscos_uP"] = c[k]
    row["Tmin_Gas_Viscos"], row["Tmax_Gas_Viscos"] = T.min(), T.max()
    log("Visc_gas", T, np.polynomial.polynomial.polyval(T, c), y)

    # --- Liquid viscosity: log10(cP) = A + B/T + C T + D T^2 ---
    T, y = sample(lambda t: sat_liq(fl, t, "mu") * 1e3, Tl)
    M = np.column_stack([np.ones_like(T), 1 / T, T, T**2])
    sc = np.abs(M).max(axis=0)
    coef = np.linalg.lstsq(M / sc, np.log10(y), rcond=None)[0] / sc
    for k, L in enumerate("ABCD"): row[f"{L}_Liq_Viscos_cP"] = coef[k]
    row["Tmin_Liq_Viscos"], row["Tmax_Liq_Viscos"] = T.min(), T.max()
    log("Visc_liq", T, 10 ** (M @ coef), y)

    # --- Gas and liquid k (quadratic, W/m/K) ---
    T, y = sample(lambda t: dilute(fl, t, "k"), Tg)
    c = poly_fit(T, y, 2)
    for k, L in enumerate("ABC"): row[f"{L}_K_gas-Thermal_Cond"] = c[k]
    row["Tmin_K_gas"], row["Tmax_K_gas"] = T.min(), T.max()
    log("k_gas", T, np.polynomial.polynomial.polyval(T, c), y)

    T, y = sample(lambda t: sat_liq(fl, t, "k"), Tl)
    c = poly_fit(T, y, 2)
    for k, L in enumerate("ABC"): row[f"{L}_K_liq-Thermal_Cond"] = c[k]
    row["Tmin_K_liq"], row["Tmax_K_liq"] = T.min(), T.max()
    log("k_liq", T, np.polynomial.polynomial.polyval(T, c), y)

    return row, q


def main():
    fdata = pd.read_csv(RAW_DIR / "formation_data.csv", encoding="utf-8").set_index("Component")
    rows, quality = [], []
    for name, fl in COMPONENTS.items():
        try:
            r, q = build_component(name, fl)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            continue
        r[COL["hf"]] = fdata.loc[name, "hf_298_k"]
        r[COL["gf"]] = fdata.loc[name, "gf_298_k"]
        rows.append(r); quality += q
        print(f"[OK] {name}")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df['n_gas_el'] = df['Component'].map(lambda c: N_GAS_EL[str(c).lower().replace(' ', '').replace('_', '')])
    df.to_csv(DATA_DIR / "properties_db.csv", index=False, encoding="utf-8")
    pd.DataFrame(quality).round(4).to_csv(DOCS_DIR / "fit_quality.csv", index=False, encoding="utf-8")
    print(f"\nCoolProp {CoolProp.__version__} | {len(rows)} components")

if __name__ == "__main__":
    main()