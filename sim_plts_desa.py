"""
Simulasi per jam: Indeks Ketersediaan Terkoreksi Cuaca (I_w) vs tiga skema pembanding
untuk PLTS desa off-grid berbaterai.

Data masukan:
- Cuaca : tanpa --weather, skrip memakai cuaca SINTETIS (hanya untuk menguji kode).
          Seluruh hasil di esai memakai data NASA POWER (--weather nasa.csv).
- Beban : tanpa --load, skrip memakai profil beban ilustratif bawaan (asumsi penulis,
          +-700 kWh/hari, puncak malam). Profil inilah yang dipakai di esai.

Pemakaian:
    python sim_plts_desa.py                                  # uji kode, data sintetis
    python sim_plts_desa.py --weather nasa.csv --month 11 --stress 200   # hasil di esai

Format --weather : CSV NASA POWER hourly (kolom YEAR,MO,DY,HR,ALLSKY_SFC_SW_DWN,T2M),
                   unduh dengan time standard LST. Nilai -999 dianggap hilang.
Format --load    : CSV 24 baris, satu kolom 'kw' (profil beban kontrak per jam, kW).
"""
import argparse
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------------
# 1. PARAMETER (semua ASUMSI yang harus disebut di P13; ganti dengan data Pulau Sembur Laut)
# ----------------------------------------------------------------------------------
@dataclass
class Config:
    kwp: float = 400.0            # kWp PV (dari blueprint)
    batt_kwh: float = 600.0       # kWh nominal baterai (dari blueprint)
    inv_kw: float = 300.0         # [ASUMSI] rating inverter hybrid AC
    pv_derate: float = 0.86       # [ASUMSI] soiling, mismatch, kabel
    gamma: float = -0.0035        # [ASUMSI] koef. suhu modul Si (/degC)
    dT_per_G: float = 0.03125     # [ASUMSI] Tcell = Tamb + dT_per_G * G (NOCT 45C)
    eta_inv: float = 0.96         # [ASUMSI]
    rte: float = 0.92             # [ASUMSI] round-trip efficiency baterai
    soc_min: float = 0.10         # [ASUMSI]
    c_rate: float = 0.5           # [ASUMSI] batas daya baterai = c_rate * kapasitas
    soh_ref: float = 0.80         # SoH acuan kontrak = ambang pemicu penggantian
    soh_baseline: float = 0.90    # SoH baterai saat operasi normal
    soh_degraded: float = 0.65    # SoH skenario 4 (di bawah ambang)
    daily_load_kwh: float = 700.0 # [ASUMSI] energi harian beban (placeholder)
    on_frac: float = 0.50         # jam dihitung "nyala" jika energi tersalur >= on_frac * beban
    warmup_days: int = 3
    eval_days: int = 30           # periode pembayaran bulanan


@dataclass
class Events:
    weather_start_day: int = 10
    weather_days: int = 5
    weather_factor: float = 0.35
    inv_start_day: int = 15
    inv_hours: int = 72
    low_demand_factor: float = 0.5
    irr_bias: float = 0.0       # bias sensor iradiasi terukur (mis. 0.05 = +5%)
    twin_pv_err: float = 0.0    # galat parameter PV pada twin sehat (derate)
    twin_rte_err: float = 0.0   # galat efisiensi baterai pada twin sehat


SCENARIOS = {
    1: "Operasi normal",
    2: "Cuaca buruk, sistem sehat",
    3: "Kerusakan inverter",
    4: "Degradasi baterai",
    5: "Permintaan rendah",
    6: "Cuaca buruk + kerusakan",
}

# Kriteria lulus: KUNCI SEBELUM simulasi dijalankan pada data sungguhan (P13).
CRITERIA = {
    "mae_indeks_maks_pp": 5.0,          # galat pembayaran rata-rata indeks <= 5 poin persen
    "skenario2_min_pct_ideal": 98.0,    # cuaca buruk tidak boleh dipotong (>= 98% pembayaran ideal)
    "skenario5_min_pct_ideal": 98.0,    # permintaan rendah tidak boleh dipotong
}


# ----------------------------------------------------------------------------------
# 2. DATA MASUKAN
# ----------------------------------------------------------------------------------
def synthetic_weather(total_days, seed=0):
    rng = np.random.default_rng(seed)
    h = np.arange(total_days * 24)
    hod = h % 24
    day = h // 24
    clear = np.where((hod >= 6) & (hod <= 18), np.sin(np.pi * (hod - 6) / 12), 0.0) * 900.0
    cloud = rng.beta(5, 2, size=total_days + 1)[day]
    G = np.clip(clear * cloud * (1 + 0.05 * rng.standard_normal(len(h))), 0, None)
    T = 27 + 4 * np.sin(2 * np.pi * (hod - 9) / 24) + 0.5 * rng.standard_normal(len(h))
    return G, T


def load_nasa_power(path, month, total_days, warmup_days=3):
    """Mengembalikan [(tahun, G, T), ...]; tiap jendela = warm-up + periode evaluasi,
    dimulai warmup_days sebelum tanggal 1 bulan yang dipilih (boleh lintas bulan)."""
    with open(path) as f:   # CSV tanpa header (API header=false) juga bisa dibaca
        skip = next((i for i, line in enumerate(f) if "-END HEADER-" in line), -1) + 1
    df = pd.read_csv(path, skiprows=skip).replace(-999, np.nan)
    cols = ["ALLSKY_SFC_SW_DWN", "T2M"]
    df[cols] = df[cols].interpolate(limit=3)
    df["ts"] = pd.to_datetime(dict(year=df.YEAR, month=df.MO, day=df.DY, hour=df.HR))
    out = []
    for y in sorted(df.YEAR.unique()):
        start = pd.Timestamp(year=int(y), month=month, day=1) - pd.Timedelta(days=warmup_days)
        w = df[(df.ts >= start) & (df.ts < start + pd.Timedelta(days=total_days))]
        if len(w) == total_days * 24 and not w[cols].isna().any().any():
            out.append((int(y), w["ALLSKY_SFC_SW_DWN"].to_numpy(), w["T2M"].to_numpy()))
    if not out:
        raise ValueError("Tidak ada tahun dengan data lengkap untuk bulan itu")
    return out


def contract_load_profile(cfg, path=None):
    if path:
        shape = pd.read_csv(path)["kw"].to_numpy(dtype=float)
        assert len(shape) == 24, "profil beban harus 24 baris"
        return shape
    shape = np.array([.35, .33, .32, .32, .34, .45, .60, .55, .45, .42, .42, .45,
                      .48, .45, .42, .42, .50, .75, 1.0, 1.0, .95, .80, .60, .45])
    return shape / shape.sum() * cfg.daily_load_kwh


# ----------------------------------------------------------------------------------
# 3. MODEL SISTEM (time-stepped: SoC bergantung riwayat)
# ----------------------------------------------------------------------------------
def pv_ac_potential(G, T, cfg, inv_cap):
    tcell = T + cfg.dT_per_G * G
    dc = cfg.kwp * G / 1000.0 * (1 + cfg.gamma * (tcell - 25.0)) * cfg.pv_derate
    return np.minimum(np.clip(dc, 0, None) * cfg.eta_inv, inv_cap)


def dispatch(G, T, load, cfg, inv_cap, soh, soc0=0.5):
    """Load-following: PV -> beban, surplus -> baterai, defisit <- baterai.
    Mengembalikan energi tersalur, tak terlayani, dan E_cap (kemampuan maks. per jam)."""
    n = len(G)
    pv_ac = pv_ac_potential(G, T, cfg, inv_cap)
    cap = cfg.batt_kwh * soh
    e_min, e_max = cfg.soc_min * cap, cap
    e = soc0 * cap
    eta1 = np.sqrt(cfg.rte)
    pmax = cfg.c_rate * cap
    deliv, ecap = np.zeros(n), np.zeros(n)
    for t in range(n):
        avail = min(pmax, max(0.0, e - e_min) * eta1)
        ecap[t] = min(inv_cap[t], pv_ac[t] + avail)
        direct = min(load[t], pv_ac[t])
        dis = min(load[t] - direct, avail, max(0.0, inv_cap[t] - direct))
        e -= dis / eta1
        chg = min(pv_ac[t] - direct, pmax, (e_max - e) / eta1)
        e += max(chg, 0.0) * eta1
        deliv[t] = direct + dis
    return deliv, load - deliv, ecap, pv_ac


# ----------------------------------------------------------------------------------
# 4. SKENARIO, INDEKS, DAN SKEMA PEMBAYARAN
# ----------------------------------------------------------------------------------
def dhat_rolling(d_a, shed, contract, days=7):
    out = np.tile(contract, len(d_a) // 24).astype(float)
    for t in range(len(d_a)):
        idx = [t - 24*k for k in range(1, days+1) if t - 24*k >= 0 and not shed[t - 24*k]]
        if idx:
            out[t] = np.median(d_a[idx])
    return out


def build_scenario(k, G, T, cfg, ev, contract):
    n = len(G)
    w0 = cfg.warmup_days
    day = lambda d: (w0 + d) * 24
    G = G.copy()
    load = np.tile(contract, n // 24)
    inv_full = np.full(n, cfg.inv_kw)
    inv = inv_full.copy()
    soh = cfg.soh_baseline
    if k in (2, 6):
        s = day(ev.weather_start_day)
        G[s: s + ev.weather_days * 24] *= ev.weather_factor
    if k in (3, 6):
        s = day(ev.weather_start_day + 1) if k == 6 else day(ev.inv_start_day)
        inv[s: s + ev.inv_hours] = 0.0
    if k == 4:
        soh = cfg.soh_degraded
    if k == 5:
        load = load * ev.low_demand_factor
    return G, load, inv, inv_full, soh


def evaluate(k, G0, T, cfg, ev, contract, eba_baseline=None):
    G, load, inv, inv_full, soh = build_scenario(k, G0, T, cfg, ev, contract)
    sl = slice(cfg.warmup_days * 24, (cfg.warmup_days + cfg.eval_days) * 24)
    tile_contract = np.tile(contract, len(G) // 24)

    # sistem nyata
    d_a, u_a, _, _ = dispatch(G, T, load, cfg, inv, soh)
    # ground truth: sistem sehat (SoH acuan, inverter normal) pada cuaca dan beban SEBENARNYA
    _, u_h, _, pv_pot = dispatch(G, T, load, cfg, inv_full, cfg.soh_ref)
    u_op = max(0.0, u_a[sl].sum() - u_h[sl].sum())
    ideal = 1.0 - u_op / load[sl].sum()

    # twin memakai iradiasi TERUKUR (bisa bias) dan parameter yang tidak persis sama dengan sistem nyata
    G_meas = G * (1 + ev.irr_bias)
    cfg_tw = replace(cfg, pv_derate=cfg.pv_derate * (1 + ev.twin_pv_err),
                     rte=min(0.99, cfg.rte * (1 + ev.twin_rte_err)))
    # D_eff dari data yang TERUKUR: tersalur saat menyala, profil kontrak saat padam/pelepasan beban
    shed = u_a > 1e-6
    d_eff = np.where(shed, dhat_rolling(d_a, shed, contract), d_a)
    # E_cap,sehat dari twin sehat yang dijalankan menerus dengan D_eff, cuaca terukur
    _, _, ecap_h, _ = dispatch(G_meas, T, d_eff, cfg_tw, inv_full, cfg.soh_ref)
    e_ref = np.minimum(d_eff, ecap_h)                                   # Pers. (1)
    i_w = 1.0 - np.maximum(0.0, e_ref[sl] - d_a[sl]).sum() / e_ref[sl].sum()   # Pers. (2)

    pay = {
        "Bayar per instalasi": 1.0,
        "Bayar per jam nyala": float(np.mean(d_a[sl] >= cfg.on_frac * load[sl])),
        "Indeks usulan": i_w,
    }
    eba_ratio = d_a[sl].sum() / pv_pot[sl].sum()   # EBA versi grid: energi aktual / energi potensial PV
    pay["EBA versi grid"] = None if eba_baseline is None else min(1.0, eba_ratio / eba_baseline)
    return {"ideal": ideal, "pay": pay, "eba_ratio": eba_ratio, "u_op": u_op,
            "eba_raw": min(1.0, eba_ratio),   # EBA tanpa kalibrasi (jaminan = 100% energi potensial PV)
            "series": dict(G=G, load=load, d_a=d_a, e_ref=e_ref, d_eff=d_eff, u_a=u_a, sl=sl)}


def run_all(G, T, cfg, ev, contract):
    base = evaluate(1, G, T, cfg, ev, contract)           # kalibrasi EBA: jaminan = rasio kondisi normal
    results = {k: evaluate(k, G, T, cfg, ev, contract, base["eba_ratio"]) for k in SCENARIOS}
    return results


def table2(results):
    schemes = ["Bayar per instalasi", "Bayar per jam nyala", "EBA versi grid", "Indeks usulan"]
    rows, errs = [], {s: [] for s in schemes}
    for k, name in SCENARIOS.items():
        r = results[k]
        row = {"Skenario": f"{k}. {name}", "Ideal (%)": 100 * r["ideal"]}
        for s in schemes:
            row[s] = 100 * r["pay"][s]
            errs[s].append(abs(100 * (r["pay"][s] - r["ideal"])))
        rows.append(row)
    rows.append({"Skenario": "Galat rata-rata (poin persen)", "Ideal (%)": np.nan,
                 **{s: float(np.mean(errs[s])) for s in schemes}})
    return pd.DataFrame(rows).set_index("Skenario")


def check_criteria(results):
    t = table2(results)
    out = {
        "MAE indeks": t.loc["Galat rata-rata (poin persen)", "Indeks usulan"] <= CRITERIA["mae_indeks_maks_pp"],
        "Skenario 2 tidak dipotong": 100 * results[2]["pay"]["Indeks usulan"] >=
            CRITERIA["skenario2_min_pct_ideal"] * results[2]["ideal"],
        "Skenario 5 tidak dipotong": 100 * results[5]["pay"]["Indeks usulan"] >=
            CRITERIA["skenario5_min_pct_ideal"] * results[5]["ideal"],
    }
    return out


# ----------------------------------------------------------------------------------
# 5. GAMBAR 1 dan UJI TEKANAN
# ----------------------------------------------------------------------------------
def figure1(results, cfg, path="gambar2.png", k=6, day0=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = results[k]["series"]
    a = (cfg.warmup_days + day0) * 24
    sl = slice(a, a + 7 * 24)
    x = np.arange(7 * 24)
    fig, ax = plt.subplots(figsize=(10, 4))
    load, d_a, e_ref = s["load"][sl], s["d_a"][sl], s["e_ref"][sl]
    ax.plot(x, load, label="Permintaan warga", color="0.5", lw=1)
    ax.plot(x, d_a, label="Energi tersalur", color="tab:blue")
    ax.plot(x, e_ref, label="Energi acuan sistem sehat", color="tab:green", ls="--")
    short = np.maximum(0, e_ref - d_a)
    ax.fill_between(x, d_a, d_a + short, color="tab:red", alpha=0.35,
                    label="Tak terlayani akibat operator (dipotong)")
    top = np.maximum(e_ref, d_a)
    ax.fill_between(x, top, load, where=load > top, color="0.6", alpha=0.25, hatch="//",
                    edgecolor="0.4", lw=0, label="Tak terlayani akibat cuaca (tidak dipotong)")
    ax.set_xticks(range(0, 7 * 24 + 1, 24))
    ax.set_xlim(0, 7 * 24 - 1)
    ax.set_xlabel("Jam ke- (7 hari, mulai 11 November 2024)", fontsize=10)
    ax.set_ylabel("Energi (kWh per jam)", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9, frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def stress(weather_list, cfg, contract, n, seed=1, err=0.05):
    """Uji tekanan: acak waktu/durasi gangguan dan cuaca buruk, atau tahun cuaca lain."""
    rng = np.random.default_rng(seed)
    maes = {s: [] for s in ["Bayar per instalasi", "Bayar per jam nyala", "EBA versi grid", "Indeks usulan"]}
    worst = None
    for i in range(n):
        G, T = weather_list[i % len(weather_list)]
        ev = Events(weather_start_day=int(rng.integers(2, 22)), weather_days=int(rng.integers(2, 8)),
                    weather_factor=float(rng.uniform(0.2, 0.6)),
                    inv_start_day=int(rng.integers(2, 24)), inv_hours=int(rng.integers(12, 120)),
                    irr_bias=float(rng.uniform(-err, err)), twin_pv_err=float(rng.uniform(-err, err)),
                    twin_rte_err=float(rng.uniform(-err, err)))
        res = run_all(G, T, cfg, ev, contract)
        t = table2(res).loc["Galat rata-rata (poin persen)"]
        for s in maes:
            maes[s].append(t[s])
        if worst is None or t["Indeks usulan"] > worst[0]:
            worst = (t["Indeks usulan"], ev)
    summ = pd.DataFrame({s: [np.mean(v), np.std(v), np.max(v)] for s, v in maes.items()},
                        index=["rata-rata", "std", "maks"]).T
    return summ, worst


def sensitivity(G, T, cfg, contract, errs=(0.0, 0.03, 0.05, 0.10)):
    """Galat indeks vs galat sensor iradiasi dan parameter twin (P16). Tanda +/- keduanya diuji."""
    rows = {}
    for e in errs:
        vals = []
        for sign in (+1, -1):
            ev = Events(irr_bias=sign * e, twin_pv_err=sign * e, twin_rte_err=sign * e)
            t = table2(run_all(G, T, cfg, ev, contract))
            vals.append(t.loc["Galat rata-rata (poin persen)", "Indeks usulan"])
            s2 = 100 * t.loc["2. Cuaca buruk, sistem sehat", "Indeks usulan"] / max(1e-9, t.loc["2. Cuaca buruk, sistem sehat", "Ideal (%)"])
            vals.append(s2)
        rows[f"galat {int(100*e)}%"] = vals
    return pd.DataFrame(rows, index=["MAE indeks (+)", "Skenario 2 % ideal (+)",
                                     "MAE indeks (-)", "Skenario 2 % ideal (-)"]).T


# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather"); ap.add_argument("--load")
    ap.add_argument("--month", type=int, default=11)
    ap.add_argument("--stress", type=int, default=0)
    ap.add_argument("--eval-days", type=int, default=None, help="mis. 28 untuk Februari")
    a = ap.parse_args()

    cfg, ev = Config(), Events()
    if a.eval_days:
        cfg = replace(cfg, eval_days=a.eval_days)
    total_days = cfg.warmup_days + cfg.eval_days
    contract = contract_load_profile(cfg, a.load)
    if a.weather:
        wl = [(G, T) for _, G, T in load_nasa_power(a.weather, a.month, total_days, cfg.warmup_days)]
        print(f"Cuaca NASA POWER: {len(wl)} tahun untuk bulan {a.month}")
    else:
        print("!! DATA SINTETIS: hanya untuk uji kode, bukan untuk esai !!")
        wl = [synthetic_weather(total_days, seed=s) for s in range(5)]
    G, T = wl[0]

    res = run_all(G, T, cfg, ev, contract)
    pd.set_option("display.float_format", lambda v: f"{v:6.1f}")
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(table2(res)); print()
    print("EBA tanpa kalibrasi (% pembayaran):",
          {k: round(100 * r["eba_raw"], 1) for k, r in res.items()})
    print("Kriteria lulus:", check_criteria(res))
    figure1(res, cfg)
    print("Gambar 2 disimpan: gambar2.png")
    print("\nSensitivitas galat sensor/parameter twin:"); print(sensitivity(G, T, cfg, contract))
    if a.stress:
        summ, worst = stress(wl, cfg, contract, a.stress)
        print("\nUji tekanan (galat pembayaran, poin persen):"); print(summ)
        print("Kasus terburuk untuk indeks:", worst)