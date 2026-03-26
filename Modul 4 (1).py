# STABILITY VALIDATOR
# Membuktikan secara statistik bahwa produksi E-Kerosene tetap stabil (CV rendah, PSI >0.90) meski radiasi matahari berfluktuasi, divalidasi dengan Monte Carlo 1000 simulasi.


import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser, os, json

# WARNA TEMA
AMBER  = "#F2A623"
RED    = "#E24B4A"
BLUE   = "#378ADD"
TEAL   = "#1D9E75"
GREEN  = "#639922"
PURPLE = "#7F77DD"
CORAL  = "#D85A30"
GRAY   = "#888780"
BG     = "#FFFFFF"
BG2    = "#F5F5F0"
BORDER = "rgba(0,0,0,0.10)"
TEXT   = "#1E293B"
TEXTS  = "#64748B"

# PARAMETER SISTEM
SYS = {
    "mirror_area_m2":  30_000,
    "wax_mass_kg":     15_000,
    "reactor_load_kw": 50.0,
    "eta_csp":         0.65,
    "L_f_kwh_kg":      200 / 3600,     # kJ/kg → kWh/kg
    "soc_init":        0.30,
    "co2_ton_day":     1200 * 0.65,    # RU IV Cilacap, capture 65%
    "ratio_co2_bio":   1.83,
    "eta_gasif":       0.82,
    "eta_ft":          0.80,
    "alpha_ft":        0.87,
    "rho_kero":        0.800,           # kg/L
}

# DATA KOTA & PROFIL SURYA
CITY_META = {
    "Cilacap":   {"lat": -7.70,  "lon": 109.01, "alt": 6  },
    "Semarang":  {"lat": -6.97,  "lon": 110.42, "alt": 10 },
    "Makassar":  {"lat": -5.14,  "lon": 119.43, "alt": 10 },
    "Kupang":    {"lat": -10.17, "lon": 123.61, "alt": 90 },
    "Denpasar":  {"lat": -8.65,  "lon": 115.22, "alt": 10 },
    "Surabaya":  {"lat": -7.25,  "lon": 112.75, "alt": 5  },
}

# Cloud cover bulanan per kota (Jan–Des)
CITY_CLOUD = {
    "Cilacap":  [.75,.73,.70,.66,.58,.50,.46,.44,.48,.58,.68,.74],
    "Semarang": [.78,.76,.72,.68,.60,.52,.48,.46,.50,.60,.70,.76],
    "Makassar": [.70,.68,.62,.54,.44,.36,.30,.28,.32,.42,.58,.68],
    "Kupang":   [.68,.65,.58,.46,.35,.28,.24,.22,.26,.36,.52,.65],
    "Denpasar": [.70,.68,.62,.54,.44,.36,.32,.30,.34,.46,.60,.68],
    "Surabaya": [.72,.70,.66,.60,.52,.44,.40,.38,.42,.54,.64,.70],
}

DOY_MID = [15,46,74,105,135,166,196,227,258,288,319,349]
BULAN   = ["Jan","Feb","Mar","Apr","Mei","Jun",
           "Jul","Agu","Sep","Okt","Nov","Des"]

# FUNGSI SOLAR (standalone dari Modul 1)
def _eot(doy):
    B = np.radians(360/365*(doy-81))
    return 9.87*np.sin(2*B) - 7.53*np.cos(B) - 1.5*np.sin(B)

def _elev(h, lat, lon, doy):
    decl  = np.radians(23.45*np.sin(np.radians(360/365*(doy-81))))
    lr    = np.radians(lat)
    st    = h + (lon%15)*4/60 + _eot(doy)/60
    om    = np.radians((st-12)*15)
    sin_e = np.sin(lr)*np.sin(decl) + np.cos(lr)*np.cos(decl)*np.cos(om)
    return float(np.degrees(np.arcsin(np.clip(sin_e,-1,1))))

def _bird(el, alt):
    if el <= 0: return 0.0
    er = np.radians(el); cz = np.sin(er)
    AM = min(1/(cz + 0.50572*(96.07995-el)**-1.6364), 38.0)
    af = np.exp(-alt/8500)
    Tr = np.clip(np.exp(-0.0903*(AM*af)**0.84*(1+AM*af-(AM*af)**1.01)), 0.0, 1.0)
    Ta = np.clip(np.exp(-0.0688*(AM*af)**0.9*0.66), 0.0, 1.0)
    Tw = np.clip(np.exp(-0.2700*(0.04*AM)**0.45), 0.0, 1.0)
    DNI = max(0.0, 1353*0.9662*Tr*Ta*Tw)
    GHI = max(0.0, DNI*cz + 1353*cz*0.95*Tr**1.01*0.93**0.69*0.12)
    return GHI

def ghi_profile(city, cf, doy):
    """GHI per jam 24 jam (W/m²)."""
    m = CITY_META[city]
    kc = 1 - cf*0.75
    return np.array([
        round(_bird(_elev(h+0.5, m["lat"], m["lon"], doy), m["alt"]) * kc)
        for h in range(24)
    ])

# SIMULASI PCM
def sim_pcm(GHI, mirror_m2, wax_kg, load_kw, soc_init=0.30):
    """
    Heat balance 24 jam → return produksi kerosene per jam (L/jam).
    Konversi: Q_reaktor_terpenuhi (kW) → kerosene via efisiensi chain.
    """
    p       = SYS
    E_max   = wax_kg * p["L_f_kwh_kg"]
    E_pcm   = E_max * soc_init
    Q_solar = (GHI * mirror_m2 * p["eta_csp"]) / 1000   # kW

    kero_per_kwh = 0.018   # L/kWh

    kero_arr   = np.zeros(24)
    soc_arr    = np.zeros(24)
    q_act_arr  = np.zeros(24)
    jam_padam  = 0

    for h in range(24):
        Qin  = Q_solar[h]
        Qout = load_kw
        surplus = Qin - Qout

        if surplus >= 0:
            ruang  = E_max - E_pcm
            masuk  = min(surplus, ruang)
            E_pcm += masuk
            q_act  = Qout
        else:
            defisit   = abs(surplus)
            bisa      = min(defisit, E_pcm)
            E_pcm    -= bisa
            q_act     = min(Qin + bisa, Qout)
            if q_act < Qout * 0.95:
                jam_padam += 1

        q_act_arr[h]  = q_act
        kero_arr[h]   = q_act * kero_per_kwh
        soc_arr[h]    = (E_pcm / E_max) * 100

    eff = float(q_act_arr.sum()) / (load_kw * 24) * 100
    return {
        "kero":       kero_arr,
        "soc":        soc_arr,
        "q_act":      q_act_arr,
        "q_solar":    Q_solar,
        "eff":        round(eff, 1),
        "jam_padam":  jam_padam,
        "total_kero": round(float(kero_arr.sum()), 2),
    }

def sim_no_pcm(GHI, mirror_m2, load_kw):
    """Simulasi tanpa PCM — untuk perbandingan."""
    Q_solar      = (GHI * mirror_m2 * SYS["eta_csp"]) / 1000
    kero_per_kwh = 0.018
    kero_arr     = np.array([
        min(Q, load_kw) * kero_per_kwh for Q in Q_solar
    ])
    q_act_arr = np.array([min(Q, load_kw) for Q in Q_solar])
    jam_padam = int((q_act_arr < load_kw * 0.95).sum())
    eff       = float(q_act_arr.sum()) / (load_kw * 24) * 100
    return {
        "kero":       kero_arr,
        "q_act":      q_act_arr,
        "q_solar":    Q_solar,
        "eff":        round(eff, 1),
        "jam_padam":  jam_padam,
        "total_kero": round(float(kero_arr.sum()), 2),
    }

# METRIK STABILITAS
def stability_metrics(kero_arr: np.ndarray) -> dict:
    """
    CV, PSI, dan metrik stabilitas lainnya.
    Ref: Denholm et al. (2010) NREL/TP-6A2-45834
    """
    mu   = float(kero_arr.mean())
    sigma= float(kero_arr.std())
    cv   = (sigma / mu * 100) if mu > 0 else 0
    psi  = max(0.0, 1 - cv/100)

    kero_sorted = np.sort(kero_arr)
    n = len(kero_sorted)
    gini = (2*np.sum((np.arange(1,n+1))*kero_sorted) /
            (n*kero_sorted.sum()) - (n+1)/n) if kero_sorted.sum() > 0 else 0

    return {
        "mean":   round(mu,    3),
        "std":    round(sigma, 3),
        "cv":     round(cv,    2),
        "psi":    round(psi,   3),
        "gini":   round(gini,  3),
        "min":    round(float(kero_arr.min()), 3),
        "max":    round(float(kero_arr.max()), 3),
    }

# MONTE CARLO SIMULATION
def monte_carlo(
    city:       str,
    month_idx:  int,
    n_runs:     int  = 1000,
    mirror_m2:  float = None,
    wax_kg:     float = None,
    load_kw:    float = None,
) -> dict:
    """
    Monte Carlo 1000 simulasi dengan variasi cloud cover acak
    (distribusi normal ± 15% di sekitar nilai historis).
    Output: distribusi total kerosene harian (L/hari).

    Ref: Sansavini et al. (2014) doi:10.1016/j.ress.2014.01.016
    """
    mirror_m2 = mirror_m2 or SYS["mirror_area_m2"]
    wax_kg    = wax_kg    or SYS["wax_mass_kg"]
    load_kw   = load_kw   or SYS["reactor_load_kw"]

    cf_base  = CITY_CLOUD[city][month_idx]
    doy      = DOY_MID[month_idx]

    rng      = np.random.default_rng(42)
    cf_samples = np.clip(
        rng.normal(cf_base, 0.15, n_runs), 0.05, 0.99)

    kero_pcm    = np.zeros(n_runs)
    kero_nopcm  = np.zeros(n_runs)
    eff_pcm     = np.zeros(n_runs)
    eff_nopcm   = np.zeros(n_runs)
    padam_pcm   = np.zeros(n_runs)
    padam_nopcm = np.zeros(n_runs)

    for i, cf in enumerate(cf_samples):
        GHI = ghi_profile(city, cf, doy)
        r1  = sim_pcm(GHI, mirror_m2, wax_kg, load_kw)
        r2  = sim_no_pcm(GHI, mirror_m2, load_kw)
        kero_pcm[i]    = r1["total_kero"]
        kero_nopcm[i]  = r2["total_kero"]
        eff_pcm[i]     = r1["eff"]
        eff_nopcm[i]   = r2["eff"]
        padam_pcm[i]   = r1["jam_padam"]
        padam_nopcm[i] = r2["jam_padam"]

    sm_pcm   = stability_metrics(kero_pcm)
    sm_nopcm = stability_metrics(kero_nopcm)

    be = ((sm_pcm["mean"] - sm_nopcm["mean"]) /
          sm_nopcm["mean"] * 100) if sm_nopcm["mean"] > 0 else 0

    return {
        "kero_pcm":      kero_pcm,
        "kero_nopcm":    kero_nopcm,
        "eff_pcm":       eff_pcm,
        "eff_nopcm":     eff_nopcm,
        "padam_pcm":     padam_pcm,
        "padam_nopcm":   padam_nopcm,
        "sm_pcm":        sm_pcm,
        "sm_nopcm":      sm_nopcm,
        "cf_samples":    cf_samples,
        "n_runs":        n_runs,
        "be_pct":        round(be, 1),
        "cf_base":       cf_base,
    }

# SKENARIO MULTI-HARI (worst case)
def multi_day_scenario(
    city:      str,
    scenario:  str,
    days:      int  = 7,
    mirror_m2: float = None,
    wax_kg:    float = None,
    load_kw:   float = None,
) -> dict:
    """
    Simulasi produksi kerosene selama N hari berturut-turut.
    Skenario:
      'normal'    — cloud cover historis Juni
      'hujan'     — cloud cover 85–95% (musim hujan ekstrem)
      'kering'    — cloud cover 15–25% (puncak musim kering)
      'intermiten'— selang-seling cerah/mendung (realistis)
    """
    mirror_m2 = mirror_m2 or SYS["mirror_area_m2"]
    wax_kg    = wax_kg    or SYS["wax_mass_kg"]
    load_kw   = load_kw   or SYS["reactor_load_kw"]

    rng = np.random.default_rng(99)
    doy = DOY_MID[5]   # Juni

    cf_per_day = {
        "normal":     [CITY_CLOUD[city][5]]*days,
        "hujan":      list(np.clip(rng.normal(0.88, 0.05, days), 0.80, 0.99)),
        "kering":     list(np.clip(rng.normal(0.22, 0.05, days), 0.10, 0.35)),
        "intermiten": [0.25,0.80,0.30,0.85,0.28,0.82,0.30][:days],
    }[scenario]

    hours_total    = days * 24
    kero_pcm_all   = np.zeros(hours_total)
    kero_nopcm_all = np.zeros(hours_total)
    soc_all        = np.zeros(hours_total)
    solar_all      = np.zeros(hours_total)

    E_max = wax_kg * SYS["L_f_kwh_kg"]
    E_pcm = E_max * SYS["soc_init"]
    kero_per_kwh = 0.018

    for d, cf in enumerate(cf_per_day):
        GHI     = ghi_profile(city, cf, doy)
        Q_solar = (GHI * mirror_m2 * SYS["eta_csp"]) / 1000
        for h in range(24):
            idx     = d*24 + h
            Qin     = Q_solar[h]
            Qout    = load_kw
            surplus = Qin - Qout
            if surplus >= 0:
                E_pcm   = min(E_pcm + surplus, E_max)
                q_act   = Qout
            else:
                bisa    = min(abs(surplus), E_pcm)
                E_pcm  -= bisa
                q_act   = min(Qin + bisa, Qout)
            kero_pcm_all[idx]   = q_act * kero_per_kwh
            solar_all[idx]      = Qin
            soc_all[idx]        = (E_pcm / E_max) * 100

        r_nopcm = sim_no_pcm(GHI, mirror_m2, load_kw)
        kero_nopcm_all[d*24:(d+1)*24] = r_nopcm["kero"]

    daily_pcm   = [kero_pcm_all[d*24:(d+1)*24].sum()   for d in range(days)]
    daily_nopcm = [kero_nopcm_all[d*24:(d+1)*24].sum() for d in range(days)]

    sm_pcm   = stability_metrics(np.array(daily_pcm))
    sm_nopcm = stability_metrics(np.array(daily_nopcm))

    return {
        "kero_pcm":    kero_pcm_all,
        "kero_nopcm":  kero_nopcm_all,
        "solar":       solar_all,
        "soc":         soc_all,
        "daily_pcm":   daily_pcm,
        "daily_nopcm": daily_nopcm,
        "sm_pcm":      sm_pcm,
        "sm_nopcm":    sm_nopcm,
        "cf_per_day":  cf_per_day,
        "days":        days,
        "scenario":    scenario,
    }

# PEMBUATAN FIGURE
def make_stability_24h(city, month_idx, mirror_m2, wax_kg, load_kw):
    """
    Panel 2×2: perbandingan dengan/tanpa PCM dalam 24 jam.
    """
    cf  = CITY_CLOUD[city][month_idx]
    doy = DOY_MID[month_idx]
    GHI = ghi_profile(city, cf, doy)
    r1  = sim_pcm(GHI, mirror_m2, wax_kg, load_kw)
    r2  = sim_no_pcm(GHI, mirror_m2, load_kw)
    sm1 = stability_metrics(r1["kero"])
    sm2 = stability_metrics(r2["kero"])
    hl  = [f"{h:02d}:00" for h in range(24)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Produksi E-Kerosene per jam (L/jam)",
            "SoC Slack Wax & Input surya",
            "Perbandingan dengan vs tanpa PCM",
            "Distribusi produksi (histogram)",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    fig.add_trace(go.Scatter(
        x=hl, y=r1["kero"].tolist(),
        name="Dengan PCM",
        fill="tozeroy",
        line=dict(color=TEAL, width=2.5),
        fillcolor="rgba(29,158,117,0.15)",
        hovertemplate="%{x}<br>Dengan PCM: <b>%{y:.3f} L/jam</b><extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hl, y=r2["kero"].tolist(),
        name="Tanpa PCM",
        line=dict(color=RED, width=1.5, dash="dash"),
        hovertemplate="%{x}<br>Tanpa PCM: <b>%{y:.3f} L/jam</b><extra></extra>",
    ), row=1, col=1)

    fig.add_hline(
        y=sm1["mean"], line_dash="dot",
        line_color=TEAL, line_width=1,
        annotation_text=f"Rata-rata {sm1['mean']:.3f} L/jam",
        annotation_font=dict(size=9, color=TEAL),
        annotation_position="top left",
        row=1, col=1,
    )

    fig.add_trace(go.Bar(
        x=hl,
        y=r1["soc"].tolist(),
        name="SoC (%)",
        marker_color=[
            TEAL if v >= 60 else AMBER if v >= 30 else RED
            for v in r1["soc"]
        ],
        marker_line_width=0,
        opacity=0.7,
        hovertemplate="%{x}<br>SoC: <b>%{y:.1f}%</b><extra></extra>",
        yaxis="y3",
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=hl,
        y=r1["q_solar"].tolist(),
        name="Input surya (kW)",
        line=dict(color=AMBER, width=2),
        hovertemplate="%{x}<br>Surya: <b>%{y:.1f} kW</b><extra></extra>",
        yaxis="y4",
    ), row=1, col=2)

    categories = [
        "Total kero (L)",
        "Efisiensi (%)",
        "Jam padam",
        "CV (%)",
        "PSI",
    ]
    vals_pcm   = [
        r1["total_kero"], r1["eff"],
        r1["jam_padam"], sm1["cv"], sm1["psi"]*100,
    ]
    vals_nopcm = [
        r2["total_kero"], r2["eff"],
        r2["jam_padam"], sm2["cv"], sm2["psi"]*100,
    ]

    fig.add_trace(go.Bar(
        name="Dengan PCM",
        x=categories,
        y=vals_pcm,
        marker_color=TEAL,
        marker_line_width=0,
        text=[str(round(v,1)) for v in vals_pcm],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="%{x}<br>Dengan PCM: <b>%{y:.2f}</b><extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Bar(
        name="Tanpa PCM",
        x=categories,
        y=vals_nopcm,
        marker_color=RED,
        marker_line_width=0,
        opacity=0.7,
        text=[str(round(v,1)) for v in vals_nopcm],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate="%{x}<br>Tanpa PCM: <b>%{y:.2f}</b><extra></extra>",
        showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Histogram(
        x=r1["kero"].tolist(),
        name="Dengan PCM",
        nbinsx=12,
        marker_color=TEAL,
        marker_line_width=0,
        opacity=0.75,
        hovertemplate="Produksi: <b>%{x:.3f} L/jam</b><br>Frekuensi: %{y}<extra></extra>",
        showlegend=False,
    ), row=2, col=2)

    fig.add_trace(go.Histogram(
        x=r2["kero"].tolist(),
        name="Tanpa PCM",
        nbinsx=12,
        marker_color=RED,
        marker_line_width=0,
        opacity=0.55,
        hovertemplate="Produksi: <b>%{x:.3f} L/jam</b><br>Frekuensi: %{y}<extra></extra>",
        showlegend=False,
    ), row=2, col=2)

    be = ((sm1["mean"]-sm2["mean"])/sm2["mean"]*100) if sm2["mean"]>0 else 0

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Stability Validator — 24 jam</b>  ·  "
                f"{city}  ·  {BULAN[month_idx]}  ·  "
                f"Cloud {cf*100:.0f}%<br>"
                f"<sup>"
                f"CV dengan PCM: <b style='color:{TEAL}'>{sm1['cv']:.1f}%</b>  ·  "
                f"CV tanpa PCM: <b style='color:{RED}'>{sm2['cv']:.1f}%</b>  ·  "
                f"PSI: <b>{sm1['psi']:.3f}</b>  ·  "
                f"Buffer Effectiveness: <b>+{be:.1f}%</b>"
                f"</sup>"
            ),
            x=0, xanchor="left",
            font=dict(size=14, color=TEXT),
        ),
        height=680,
        template="plotly_white",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        hovermode="x unified",
        barmode="group",
        legend=dict(
            orientation="h", y=-0.06, x=0.5, xanchor="center",
            font=dict(size=11, color=TEXTS),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=55, r=40, t=120, b=70),
        font=dict(family="system-ui,-apple-system,sans-serif", color=TEXT),
    )

    for r_ax, c_ax, yt in [
        (1,1,"L/jam"), (2,1,"Nilai"), (2,2,"Frekuensi (jam)"),
    ]:
        fig.update_yaxes(
            title_text=yt,
            title_font=dict(size=10, color=TEXTS),
            tickfont=dict(size=9, color=TEXTS),
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False, row=r_ax, col=c_ax,
        )
        fig.update_xaxes(
            tickfont=dict(size=9, color=TEXTS),
            showgrid=False, row=r_ax, col=c_ax,
        )

    return fig, sm1, sm2, r1, r2


def make_monte_carlo_fig(mc: dict) -> go.Figure:
    """
    Histogram + violin distribusi Monte Carlo 1000 run.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Distribusi total produksi kerosene (1000 run)",
            "Distribusi efisiensi sistem (%)",
        ),
        horizontal_spacing=0.12,
    )

    fig.add_trace(go.Histogram(
        x=mc["kero_pcm"].tolist(),
        name="Dengan PCM",
        nbinsx=40,
        marker_color=TEAL,
        marker_line_width=0,
        opacity=0.75,
        hovertemplate="Kerosene: <b>%{x:.1f} L</b><br>Count: %{y}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Histogram(
        x=mc["kero_nopcm"].tolist(),
        name="Tanpa PCM",
        nbinsx=40,
        marker_color=RED,
        marker_line_width=0,
        opacity=0.55,
        hovertemplate="Kerosene: <b>%{x:.1f} L</b><br>Count: %{y}<extra></extra>",
    ), row=1, col=1)

    for val, col, nm in [
        (mc["sm_pcm"]["mean"],   TEAL, "Mean PCM"),
        (mc["sm_nopcm"]["mean"], RED,  "Mean no-PCM"),
    ]:
        fig.add_vline(
            x=val, line_dash="dot", line_color=col, line_width=1.5,
            annotation_text=f"{nm}: {val:.1f} L",
            annotation_font=dict(size=9, color=col),
            row=1, col=1,
        )

    fig.add_trace(go.Histogram(
        x=mc["eff_pcm"].tolist(),
        name="Efisiensi dengan PCM",
        nbinsx=30,
        marker_color=TEAL,
        marker_line_width=0,
        opacity=0.75,
        showlegend=False,
        hovertemplate="Efisiensi: <b>%{x:.1f}%</b><br>Count: %{y}<extra></extra>",
    ), row=1, col=2)

    fig.add_trace(go.Histogram(
        x=mc["eff_nopcm"].tolist(),
        name="Efisiensi tanpa PCM",
        nbinsx=30,
        marker_color=RED,
        marker_line_width=0,
        opacity=0.55,
        showlegend=False,
        hovertemplate="Efisiensi: <b>%{x:.1f}%</b><br>Count: %{y}<extra></extra>",
    ), row=1, col=2)

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Monte Carlo Simulation</b>  ·  "
                f"n={mc['n_runs']} runs  ·  "
                f"Cloud variasi ±15%<br>"
                f"<sup>"
                f"CV dengan PCM: <b style='color:{TEAL}'>"
                f"{mc['sm_pcm']['cv']:.1f}%</b>  ·  "
                f"CV tanpa PCM: <b style='color:{RED}'>"
                f"{mc['sm_nopcm']['cv']:.1f}%</b>  ·  "
                f"Buffer Effectiveness: "
                f"<b>+{mc['be_pct']:.1f}%</b>"
                f"</sup>"
            ),
            x=0, xanchor="left",
            font=dict(size=14, color=TEXT),
        ),
        height=360,
        template="plotly_white",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        barmode="overlay",
        hovermode="x unified",
        legend=dict(
            orientation="h", y=-0.18, x=0.5, xanchor="center",
            font=dict(size=11, color=TEXTS),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=55, r=30, t=110, b=70),
        font=dict(family="system-ui,-apple-system,sans-serif", color=TEXT),
    )
    for c_ax, yt in [(1,"Frekuensi (run)"), (2,"Frekuensi (run)")]:
        fig.update_yaxes(
            title_text=yt, title_font=dict(size=10, color=TEXTS),
            tickfont=dict(size=9, color=TEXTS),
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False, row=1, col=c_ax,
        )
        fig.update_xaxes(
            tickfont=dict(size=9, color=TEXTS),
            showgrid=False, row=1, col=c_ax,
        )
    return fig


def make_multiday_fig(md: dict) -> go.Figure:
    """
    Grafik produksi kerosene multi-hari + SoC overlay.
    """
    days  = md["days"]
    h_all = [f"H{d+1}-{h:02d}:00" for d in range(days) for h in range(24)]
    h_idx = list(range(len(h_all)))

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=h_idx, y=md["kero_pcm"].tolist(),
        name="Dengan PCM (L/jam)",
        fill="tozeroy",
        line=dict(color=TEAL, width=1.8),
        fillcolor="rgba(29,158,117,0.14)",
        hovertemplate="%{x}<br>PCM: <b>%{y:.3f} L/jam</b><extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=h_idx, y=md["kero_nopcm"].tolist(),
        name="Tanpa PCM (L/jam)",
        line=dict(color=RED, width=1.2, dash="dash"),
        hovertemplate="%{x}<br>No PCM: <b>%{y:.3f} L/jam</b><extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=h_idx, y=md["soc"].tolist(),
        name="SoC Slack Wax (%)",
        line=dict(color=AMBER, width=1.5),
        hovertemplate="SoC: <b>%{y:.1f}%</b><extra></extra>",
    ), secondary_y=True)

    for d in range(1, days):
        fig.add_vline(
            x=d*24, line_dash="dot",
            line_color=GRAY, line_width=0.8,
            annotation_text=f"Hari {d+1}<br>cf={md['cf_per_day'][d]:.2f}",
            annotation_font=dict(size=8, color=GRAY),
            annotation_position="top",
        )

    mean_kero = np.mean(md["kero_pcm"])
    fig.add_hline(
        y=mean_kero, line_dash="dot", line_color=TEAL, line_width=1,
        annotation_text=f"Rata-rata {mean_kero:.3f} L/jam",
        annotation_font=dict(size=9, color=TEAL),
        annotation_position="bottom right",
        secondary_y=False,
    )

    sm1 = md["sm_pcm"]
    sm2 = md["sm_nopcm"]
    scen_label = {
        "normal":     "Normal (Juni)",
        "hujan":      "Hujan ekstrem",
        "kering":     "Musim kering",
        "intermiten": "Intermiten (selang-seling)",
    }[md["scenario"]]

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Stabilitas {days} Hari</b>  ·  "
                f"Skenario: {scen_label}<br>"
                f"<sup>"
                f"CV harian PCM: <b style='color:{TEAL}'>{sm1['cv']:.1f}%</b>  ·  "
                f"CV harian no-PCM: <b style='color:{RED}'>{sm2['cv']:.1f}%</b>  ·  "
                f"PSI: <b>{sm1['psi']:.3f}</b>"
                f"</sup>"
            ),
            x=0, xanchor="left",
            font=dict(size=14, color=TEXT),
        ),
        height=360,
        template="plotly_white",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        hovermode="x unified",
        legend=dict(
            orientation="h", y=-0.20, x=0.5, xanchor="center",
            font=dict(size=11, color=TEXTS),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=55, r=60, t=100, b=80),
        xaxis=dict(
            title="Jam operasional",
            tickmode="array",
            tickvals=[d*24+12 for d in range(days)],
            ticktext=[f"Hari {d+1}" for d in range(days)],
            tickfont=dict(size=10, color=TEXTS),
            showgrid=True, gridcolor="rgba(0,0,0,0.05)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Produksi E-Kerosene (L/jam)",
            tickfont=dict(size=9, color=TEXTS),
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
        ),
        yaxis2=dict(
            title="SoC (%)",
            range=[0,110],
            tickfont=dict(size=9, color=TEXTS),
            showgrid=False, zeroline=False,
        ),
        font=dict(family="system-ui,-apple-system,sans-serif", color=TEXT),
    )
    return fig


def make_annual_fig(city: str, mirror_m2, wax_kg, load_kw) -> go.Figure:
    """
    Proyeksi produksi & CV bulanan selama 12 bulan.
    """
    m_meta    = CITY_META[city]
    daily_kero = []
    cv_list    = []
    eff_list   = []

    for mi in range(12):
        cf  = CITY_CLOUD[city][mi]
        doy = DOY_MID[mi]
        GHI = ghi_profile(city, cf, doy)
        r   = sim_pcm(GHI, mirror_m2, wax_kg, load_kw)
        sm  = stability_metrics(r["kero"])
        daily_kero.append(r["total_kero"])
        cv_list.append(sm["cv"])
        eff_list.append(r["eff"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=BULAN, y=daily_kero,
        name="Produksi harian (L/hari)",
        marker_color=[
            RED    if v == min(daily_kero) else
            TEAL   if v == max(daily_kero) else
            AMBER
            for v in daily_kero
        ],
        marker_line_width=0,
        hovertemplate="%{x}<br>Produksi: <b>%{y:.1f} L/hari</b><extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=BULAN, y=cv_list,
        name="CV produksi (%)",
        mode="lines+markers",
        line=dict(color=PURPLE, width=2),
        marker=dict(size=6, color=PURPLE),
        hovertemplate="%{x}<br>CV: <b>%{y:.1f}%</b><extra></extra>",
    ), secondary_y=True)

    peak_m  = BULAN[int(np.argmax(daily_kero))]
    trough_m= BULAN[int(np.argmin(daily_kero))]

    fig.update_layout(
        title=dict(
            text=(
                f"<b>Proyeksi Produksi Tahunan</b>  ·  {city}<br>"
                f"<sup>"
                f"Peak: <b style='color:{TEAL}'>{peak_m}</b>  ·  "
                f"Minimum: <b style='color:{RED}'>{trough_m}</b>  ·  "
                f"Total/tahun: "
                f"<b>{sum(daily_kero)*365/12/1000:.1f}k L</b>"
                f"</sup>"
            ),
            x=0, xanchor="left",
            font=dict(size=14, color=TEXT),
        ),
        height=320,
        template="plotly_white",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        hovermode="x unified",
        legend=dict(
            orientation="h", y=-0.22, x=0.5, xanchor="center",
            font=dict(size=11, color=TEXTS),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            tickfont=dict(size=10, color=TEXTS),
            showgrid=False, zeroline=False,
        ),
        yaxis=dict(
            title="L/hari",
            tickfont=dict(size=9, color=TEXTS),
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
        ),
        yaxis2=dict(
            title="CV (%)",
            tickfont=dict(size=9, color=TEXTS),
            showgrid=False, zeroline=False,
        ),
        margin=dict(l=55, r=55, t=100, b=70),
        bargap=0.25,
        font=dict(family="system-ui,-apple-system,sans-serif", color=TEXT),
    )
    return fig, daily_kero, cv_list


# Export HTML
def export_html(
    city:       str   = "Cilacap",
    month_idx:  int   = 5,
    mirror_m2:  float = None,
    wax_kg:     float = None,
    load_kw:    float = None,
    n_mc:       int   = 1000,
) -> str:
    import plotly.io as pio

    mirror_m2 = mirror_m2 or SYS["mirror_area_m2"]
    wax_kg    = wax_kg    or SYS["wax_mass_kg"]
    load_kw   = load_kw   or SYS["reactor_load_kw"]

    print("  [1/5] Simulasi 24 jam...")
    fig_24h, sm1, sm2, r1, r2 = make_stability_24h(
        city, month_idx, mirror_m2, wax_kg, load_kw)

    print(f"  [2/5] Monte Carlo {n_mc} runs...")
    mc = monte_carlo(city, month_idx, n_mc, mirror_m2, wax_kg, load_kw)
    fig_mc = make_monte_carlo_fig(mc)

    print("  [3/5] Skenario multi-hari...")
    md_norm = multi_day_scenario(city, "normal",     7, mirror_m2, wax_kg, load_kw)
    md_rain = multi_day_scenario(city, "hujan",      7, mirror_m2, wax_kg, load_kw)
    md_dry  = multi_day_scenario(city, "kering",     7, mirror_m2, wax_kg, load_kw)
    md_int  = multi_day_scenario(city, "intermiten", 7, mirror_m2, wax_kg, load_kw)
    fig_norm = make_multiday_fig(md_norm)
    fig_rain = make_multiday_fig(md_rain)
    fig_dry  = make_multiday_fig(md_dry)
    fig_int  = make_multiday_fig(md_int)

    print("  [4/5] Proyeksi tahunan...")
    fig_ann, daily_kero, cv_list = make_annual_fig(
        city, mirror_m2, wax_kg, load_kw)

    print("  [5/5] Menyusun HTML...")

    def toj(fig): return pio.to_json(fig)

    cf = CITY_CLOUD[city][month_idx]
    be = ((sm1["mean"]-sm2["mean"])/sm2["mean"]*100) if sm2["mean"]>0 else 0

    def kpi_color(cv):
        return TEAL if cv < 10 else AMBER if cv < 25 else RED

    city_opts = "".join(
        f'<option value="{c}" {"selected" if c==city else ""}>{c}</option>'
        for c in CITY_META
    )
    bulan_opts = "".join(
        f'<option value="{i}" {"selected" if i==month_idx else ""}>{b}</option>'
        for i, b in enumerate(BULAN)
    )
    scen_opts = "".join(
        f'<option value="{s}">{l}</option>'
        for s, l in [
            ("normal","Normal (Juni)"),
            ("hujan","Hujan ekstrem"),
            ("kering","Musim kering"),
            ("intermiten","Intermiten"),
        ]
    )

    city_cloud_js = json.dumps(CITY_CLOUD)
    city_meta_js  = json.dumps(CITY_META)
    doy_mid_js    = json.dumps(DOY_MID)

    html_out = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Stability Validator </title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:system-ui,-apple-system,sans-serif;
        background:#F8F8F5;color:{TEXT};padding:24px 20px}}
  .wrap{{max-width:1200px;margin:0 auto}}
  h1{{font-size:18px;font-weight:500;margin-bottom:2px}}
  .sub{{font-size:12px;color:{TEXTS};margin-bottom:20px}}
  .metrics{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}}
  .mc{{background:#EDEDEA;border-radius:8px;padding:12px 16px;
       flex:1;min-width:120px}}
  .mc.hi{{border-left:3px solid {AMBER};border-radius:0 8px 8px 0}}
  .mc.ok{{border-left:3px solid {TEAL};border-radius:0 8px 8px 0}}
  .mc-label{{font-size:11px;color:{TEXTS};margin-bottom:3px}}
  .mc-val{{font-size:22px;font-weight:500;line-height:1}}
  .mc-unit{{font-size:11px;color:{TEXTS}}}
  .card{{background:#fff;border:0.5px solid rgba(0,0,0,.1);
         border-radius:12px;padding:16px;margin-bottom:14px}}
  .section-title{{font-size:13px;font-weight:500;
                  color:{TEXTS};margin-bottom:10px}}
  .controls{{display:flex;gap:12px;flex-wrap:wrap;
             align-items:flex-end;margin-bottom:14px}}
  .ctrl-group{{display:flex;flex-direction:column;gap:4px}}
  .ctrl-label{{font-size:11px;color:{TEXTS}}}
  select{{font-size:13px;padding:6px 10px;
          border:0.5px solid rgba(0,0,0,.15);border-radius:8px;
          background:#fff;color:{TEXT};cursor:pointer;min-width:160px}}
  input[type=range]{{min-width:140px;padding:4px 0}}
  button{{font-size:12px;padding:8px 16px;
          border:0.5px solid rgba(0,0,0,.2);border-radius:8px;
          background:#fff;color:{TEXT};cursor:pointer;font-weight:500}}
  button:hover{{background:#F0F0EC}}
  .val-display{{font-size:12px;font-weight:500;
                color:{TEXT};min-width:70px}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(3,1fr);
             gap:8px;margin-bottom:12px}}
  .kpi-card{{padding:10px 14px;border-radius:8px;
             border:0.5px solid rgba(0,0,0,.1);background:#fff}}
  .kpi-label{{font-size:10px;color:{TEXTS};margin-bottom:2px}}
  .kpi-pcm{{font-size:16px;font-weight:500}}
  .kpi-nopcm{{font-size:11px;color:{TEXTS};margin-top:1px}}
  .footer{{border-top:0.5px solid rgba(0,0,0,.1);padding-top:10px;
           font-size:10px;color:{TEXTS};margin-top:4px}}
</style>
</head>
<body>
<div class="wrap">

  <h1>Stability Validator (E-Kerosene Production Stability)</h1>
  <p class="sub">
    Membuktikan secara statistik bahwa produksi E-Kerosene tetap stabil (CV rendah, PSI >0.90) meski radiasi matahari berfluktuasi, divalidasi dengan Monte Carlo 1000 simulasi.
  </p>

  <!-- Controls -->
  <div class="card">
    <div class="section-title">Parameter validasi</div>
    <div class="controls">
      <div class="ctrl-group">
        <span class="ctrl-label">Kota</span>
        <select id="cityCtrl">{city_opts}</select>
      </div>
      <div class="ctrl-group">
        <span class="ctrl-label">Bulan</span>
        <select id="monthCtrl">{bulan_opts}</select>
      </div>
      <div class="ctrl-group">
        <span class="ctrl-label">
          Luas cermin:
          <span class="val-display" id="mirrorVal">{mirror_m2/1000:.0f}k m²</span>
        </span>
        <input type="range" id="mirrorCtrl"
               min="5000" max="60000" step="5000" value="{mirror_m2}">
      </div>
      <div class="ctrl-group">
        <span class="ctrl-label">
          Massa wax:
          <span class="val-display" id="waxVal">{wax_kg/1000:.0f}k kg</span>
        </span>
        <input type="range" id="waxCtrl"
               min="5000" max="40000" step="5000" value="{wax_kg}">
      </div>
      <div class="ctrl-group">
        <span class="ctrl-label">
          Beban reaktor:
          <span class="val-display" id="loadVal">{load_kw:.0f} kW</span>
        </span>
        <input type="range" id="loadCtrl"
               min="20" max="200" step="10" value="{load_kw}">
      </div>
      <button onclick="runValidation()">Validasi ulang</button>
    </div>
  </div>

  <!-- KPI Summary -->
  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="kpi-label">CV produksi (dengan PCM)</div>
      <div class="kpi-pcm" id="kpi-cv"
           style="color:{kpi_color(sm1['cv'])}">{sm1['cv']:.1f}%</div>
      <div class="kpi-nopcm">Tanpa PCM: {sm2['cv']:.1f}%</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">PSI (Production Stability Index)</div>
      <div class="kpi-pcm" id="kpi-psi"
           style="color:{TEAL if sm1['psi']>=0.9 else AMBER}">{sm1['psi']:.3f}</div>
      <div class="kpi-nopcm">Target ≥ 0.90</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Buffer Effectiveness</div>
      <div class="kpi-pcm" id="kpi-be"
           style="color:{TEAL}">{be:.1f}%</div>
      <div class="kpi-nopcm">Peningkatan vs tanpa PCM</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Total kerosene/hari (dengan PCM)</div>
      <div class="kpi-pcm" id="kpi-kero">{r1['total_kero']:.1f} L</div>
      <div class="kpi-nopcm">Tanpa PCM: {r2['total_kero']:.1f} L</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">Jam padam reaktor</div>
      <div class="kpi-pcm" id="kpi-down"
           style="color:{TEAL if r1['jam_padam']==0 else RED}">{r1['jam_padam']} jam</div>
      <div class="kpi-nopcm">Tanpa PCM: {r2['jam_padam']} jam</div>
    </div>
    <div class="kpi-card">
      <div class="kpi-label">MC — CV distribusi (1000 run)</div>
      <div class="kpi-pcm" id="kpi-mc-cv"
           style="color:{kpi_color(mc['sm_pcm']['cv'])}">{mc['sm_pcm']['cv']:.1f}%</div>
      <div class="kpi-nopcm">Tanpa PCM: {mc['sm_nopcm']['cv']:.1f}%</div>
    </div>
  </div>

  <!-- Metric cards -->
  <div class="metrics">
    <div class="mc ok">
      <div class="mc-label">Efisiensi dengan PCM</div>
      <div><span class="mc-val" id="mc-eff"
                 style="color:{TEAL if r1['eff']>=95 else AMBER}">{r1['eff']}</span>
           <span class="mc-unit">%</span></div>
    </div>
    <div class="mc">
      <div class="mc-label">Efisiensi tanpa PCM</div>
      <div><span class="mc-val" id="mc-eff2"
                 style="color:{RED}">{r2['eff']}</span>
           <span class="mc-unit">%</span></div>
    </div>
    <div class="mc hi">
      <div class="mc-label">MC mean kerosene (PCM)</div>
      <div><span class="mc-val" id="mc-mc-mean">{mc['sm_pcm']['mean']:.1f}</span>
           <span class="mc-unit"> L/hari</span></div>
    </div>
    <div class="mc">
      <div class="mc-label">MC std dev (PCM)</div>
      <div><span class="mc-val" id="mc-mc-std">{mc['sm_pcm']['std']:.2f}</span>
           <span class="mc-unit"> L/hari</span></div>
    </div>
    <div class="mc">
      <div class="mc-label">Gini coefficient (PCM)</div>
      <div><span class="mc-val" id="mc-gini">{sm1['gini']:.3f}</span>
           <span class="mc-unit"> (0=merata)</span></div>
    </div>
  </div>

  <!-- Chart 24 jam -->
  <div class="card">
    <div class="section-title">Validasi stabilitas 24 jam</div>
    <div id="chart24h"></div>
  </div>

  <!-- Monte Carlo -->
  <div class="card">
    <div class="section-title">Monte Carlo — distribusi 1000 run</div>
    <div id="chartMC"></div>
  </div>

  <!-- Multi-hari -->
  <div class="card">
    <div class="section-title">
      Skenario multi-hari (7 hari)
      <select id="scenCtrl" onchange="switchScen(this.value)"
              style="margin-left:12px;font-size:12px;
                     padding:4px 8px;border-radius:6px">
        {scen_opts}
      </select>
    </div>
    <div id="chartMulti"></div>
  </div>

  <!-- Proyeksi tahunan -->
  <div class="card">
    <div class="section-title">Proyeksi produksi tahunan (12 bulan)</div>
    <div id="chartAnn"></div>
  </div>

  <div class="footer">
    Rumus: CV & PSI (Denholm et al. 2010, NREL/TP-6A2-45834) ·
    Monte Carlo (Sansavini et al. 2014) ·
    Buffer Effectiveness (Pielichowska et al. 2014) ·
    Solar: Bird Clear Sky Model (NREL 1981) · PCM: Zalba et al. (2003)
  </div>
</div>

<script>
const CITY_CLOUD={city_cloud_js};
const CITY_META ={city_meta_js};
const DOY_MID   ={doy_mid_js};
const BULAN     ={json.dumps(BULAN)};
const SYS_JS    ={json.dumps({k:v for k,v in SYS.items() if isinstance(v,(int,float))})};
const AMBER='{AMBER}',RED='{RED}',BLUE='{BLUE}',TEAL='{TEAL}',
      GREEN='{GREEN}',PURPLE='{PURPLE}',GRAY='{GRAY}',
      TEXT='{TEXT}',TEXTS='{TEXTS}',BG='{BG}';

function eot(d){{const B=2*Math.PI/365*(d-81);return 9.87*Math.sin(2*B)-7.53*Math.cos(B)-1.5*Math.sin(B);}}
function elev(h,lat,lon,doy){{
  const decl=23.45*Math.sin(2*Math.PI/365*(doy-81))*Math.PI/180;
  const lr=lat*Math.PI/180,st=h+(lon%15)*4/60+eot(doy)/60;
  const om=(st-12)*15*Math.PI/180;
  const se=Math.sin(lr)*Math.sin(decl)+Math.cos(lr)*Math.cos(decl)*Math.cos(om);
  return Math.max(0,Math.asin(Math.max(-1,Math.min(1,se)))*180/Math.PI);
}}
function bird(el,alt){{
  if(el<=0)return 0;
  const er=el*Math.PI/180,cz=Math.sin(er);
  const AM=Math.min(1/(cz+0.50572*Math.pow(96.07995-el,-1.6364)),38);
  const af=Math.exp(-alt/8500);
  const Tr=Math.max(0,Math.min(1,Math.exp(-0.0903*Math.pow(AM*af,0.84)*(1+AM*af-Math.pow(AM*af,1.01)))));
  const Ta=Math.max(0,Math.min(1,Math.exp(-0.0688*Math.pow(AM*af,0.9)*0.66)));
  const Tw=Math.max(0,Math.min(1,Math.exp(-0.2700*Math.pow(0.04*AM,0.45))));
  const DNI=Math.max(0,1353*0.9662*Tr*Ta*Tw);
  return Math.max(0,DNI*cz+1353*cz*0.95*Math.pow(Tr,1.01)*Math.pow(0.93,0.69)*0.12);
}}
function ghiProfile(city,cf,doy){{
  const m=CITY_META[city],kc=1-cf*0.75;
  return Array.from({{length:24}},(_,h)=>Math.round(bird(elev(h+0.5,m.lat,m.lon,doy),m.alt)*kc));
}}
function simPCM(GHI,mirror,wax,load){{
  const Emax=wax*SYS_JS.L_f_kwh_kg,eta=SYS_JS.eta_csp,kpk=0.018;
  let Epcm=Emax*SYS_JS.soc_init;
  const kero=[],soc=[],qAct=[],qSol=GHI.map(g=>g*mirror*eta/1000);
  let totK=0,padam=0;
  for(let h=0;h<24;h++){{
    const Qin=qSol[h],surplus=Qin-load;
    let qa;
    if(surplus>=0){{Epcm=Math.min(Epcm+surplus,Emax);qa=load;}}
    else{{const b=Math.min(Math.abs(surplus),Epcm);Epcm-=b;qa=Math.min(Qin+b,load);if(qa<load*0.95)padam++;}}
    kero.push(qa*kpk);qAct.push(qa);soc.push(Epcm/Emax*100);totK+=qa*kpk;
  }}
  const mu=totK/24,sq=kero.reduce((a,v)=>a+(v-mu)**2,0)/24;
  const cv=mu>0?Math.sqrt(sq)/mu*100:0,psi=Math.max(0,1-cv/100);
  return{{kero,soc,qSol,qAct,totK:Math.round(totK*100)/100,
          eff:Math.round(qAct.reduce((a,b)=>a+b,0)/(load*24)*1000)/10,
          padam,cv:Math.round(cv*10)/10,psi:Math.round(psi*1000)/1000}};
}}
function simNoPCM(GHI,mirror,load){{
  const eta=SYS_JS.eta_csp,kpk=0.018;
  const qSol=GHI.map(g=>g*mirror*eta/1000);
  const kero=qSol.map(q=>Math.min(q,load)*kpk);
  const qAct=qSol.map(q=>Math.min(q,load));
  const totK=kero.reduce((a,b)=>a+b,0);
  const padam=qAct.filter(q=>q<load*0.95).length;
  const mu=totK/24,sq=kero.reduce((a,v)=>a+(v-mu)**2,0)/24;
  const cv=mu>0?Math.sqrt(sq)/mu*100:0;
  return{{kero,qAct,qSol,totK:Math.round(totK*100)/100,
          eff:Math.round(qAct.reduce((a,b)=>a+b,0)/(load*24)*1000)/10,
          padam,cv:Math.round(cv*10)/10}};
}}

const hl=Array.from({{length:24}},(_,i)=>String(i).padStart(2,'0')+':00');
const LO={{template:'plotly_white',paper_bgcolor:'rgba(0,0,0,0)',
  plot_bgcolor:'rgba(0,0,0,0)',
  font:{{family:'system-ui,-apple-system,sans-serif',color:TEXT}},
  margin:{{l:55,r:40,t:80,b:70}},
  legend:{{orientation:'h',y:-0.22,x:0.5,xanchor:'center',
           font:{{size:11,color:TEXTS}},bgcolor:'rgba(0,0,0,0)'}},
  hovermode:'x unified'}};

function render24h(r1,r2,city,cf,monthIdx){{
  const traces=[
    {{type:'scatter',x:hl,y:r1.kero,name:'Dengan PCM',fill:'tozeroy',
      line:{{color:TEAL,width:2.5}},fillcolor:'rgba(29,158,117,0.15)'}},
    {{type:'scatter',x:hl,y:r2.kero,name:'Tanpa PCM',
      line:{{color:RED,width:1.5,dash:'dash'}}}},
    {{type:'scatter',x:hl,y:r1.soc,name:'SoC (%)',
      line:{{color:AMBER,width:1.5}},yaxis:'y2'}},
  ];
  const be=r2.totK>0?(r1.totK-r2.totK)/r2.totK*100:0;
  Plotly.react('chart24h',traces,{{...LO,
    title:{{text:`<b>24 jam</b> · ${{city}} · Cloud ${{Math.round(cf*100)}}% · `+
            `CV PCM:<b style='color:${{TEAL}}'>${{r1.cv}}%</b> · `+
            `CV no-PCM:<b style='color:${{RED}}'>${{r2.cv}}%</b> · `+
            `Buffer Effectiveness: <b>+${{Math.round(be*10)/10}}%</b>`,
            font:{{size:13}},x:0,xanchor:'left'}},
    height:360,
    yaxis:{{title:'L/jam',showgrid:true,gridcolor:'rgba(0,0,0,0.06)',
            zeroline:false,tickfont:{{size:9,color:TEXTS}}}},
    yaxis2:{{title:'SoC (%)',overlaying:'y',side:'right',range:[0,110],
             showgrid:false,zeroline:false,tickfont:{{size:9,color:TEXTS}}}},
    shapes:[{{type:'line',x0:0,x1:23,
              y0:r1.kero.reduce((a,b)=>a+b,0)/24,
              y1:r1.kero.reduce((a,b)=>a+b,0)/24,
              line:{{color:TEAL,width:1,dash:'dot'}},xref:'x',yref:'y'}}],
  }},{{responsive:true,displayModeBar:false}});
}}

function updateKPI(r1,r2,mc_cv_pcm,mc_cv_nopcm,mc_mean,mc_std){{
  const cv_color=r1.cv<10?TEAL:r1.cv<25?AMBER:RED;
  const psi_color=r1.psi>=0.9?TEAL:AMBER;
  const down_color=r1.padam===0?TEAL:RED;
  document.getElementById('kpi-cv').textContent=r1.cv+'%';
  document.getElementById('kpi-cv').style.color=cv_color;
  document.getElementById('kpi-psi').textContent=r1.psi;
  document.getElementById('kpi-psi').style.color=psi_color;
  const be=r2.totK>0?(r1.totK-r2.totK)/r2.totK*100:0;
  document.getElementById('kpi-be').textContent=Math.round(be*10)/10+'%';
  document.getElementById('kpi-kero').textContent=r1.totK+' L vs '+r2.totK+' L';
  document.getElementById('kpi-down').textContent=r1.padam+' jam';
  document.getElementById('kpi-down').style.color=down_color;
  document.getElementById('kpi-mc-cv').textContent=mc_cv_pcm+'%';
  document.getElementById('mc-eff').textContent=r1.eff;
  document.getElementById('mc-eff2').textContent=r2.eff;
}}

const scenData={{
  normal:  {toj(fig_norm)},
  hujan:   {toj(fig_rain)},
  kering:  {toj(fig_dry)},
  intermiten:{toj(fig_int)},
}};
function switchScen(s){{
  Plotly.react('chartMulti',scenData[s].data,scenData[s].layout,
               {{responsive:true,displayModeBar:false}});
}}

function runValidation(){{
  const city=document.getElementById('cityCtrl').value;
  const mi=parseInt(document.getElementById('monthCtrl').value);
  const mirror=parseFloat(document.getElementById('mirrorCtrl').value);
  const wax=parseFloat(document.getElementById('waxCtrl').value);
  const load=parseFloat(document.getElementById('loadCtrl').value);
  const cf=CITY_CLOUD[city][mi];
  const doy=DOY_MID[mi];
  const GHI=ghiProfile(city,cf,doy);
  const r1=simPCM(GHI,mirror,wax,load);
  const r2=simNoPCM(GHI,mirror,load);
  render24h(r1,r2,city,cf,mi);
  updateKPI(r1,r2,'{mc['sm_pcm']['cv']:.1f}',
            '{mc['sm_nopcm']['cv']:.1f}',
            {mc['sm_pcm']['mean']:.2f},{mc['sm_pcm']['std']:.3f});
}}

document.getElementById('mirrorCtrl').oninput=e=>
  document.getElementById('mirrorVal').textContent=Math.round(e.target.value/1000)+'k m²';
document.getElementById('waxCtrl').oninput=e=>
  document.getElementById('waxVal').textContent=Math.round(e.target.value/1000)+'k kg';
document.getElementById('loadCtrl').oninput=e=>
  document.getElementById('loadVal').textContent=e.target.value+' kW';

const d24={toj(fig_24h)},dMC={toj(fig_mc)},dAnn={toj(fig_ann)};
Plotly.newPlot('chart24h', d24.data,  d24.layout,  {{responsive:true,displayModeBar:false}});
Plotly.newPlot('chartMC',  dMC.data,  dMC.layout,  {{responsive:true,displayModeBar:false}});
Plotly.newPlot('chartMulti',scenData.normal.data,scenData.normal.layout,
               {{responsive:true,displayModeBar:false}});
Plotly.newPlot('chartAnn', dAnn.data, dAnn.layout, {{responsive:true,displayModeBar:false}});
</script>
</body>
</html>"""

    fname = f"Modul4_StabilityValidator_{city}.html"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"\n  [✓] Exported: {fname}")
    webbrowser.open("file://" + os.path.abspath(fname))
    return fname


# Entry Point
if __name__ == "__main__":
    print("\n  Modul 4 — Stability Validator")

    fname = export_html(
        city      = "Cilacap",
        month_idx = 5,
        mirror_m2 = 30_000,
        wax_kg    = 15_000,
        load_kw   = 50.0,
        n_mc      = 1000,
    )
    print(f"  Browser akan terbuka otomatis.")

    cf  = CITY_CLOUD["Cilacap"][5]
    GHI = ghi_profile("Cilacap", cf, DOY_MID[5])
    r1  = sim_pcm(GHI, 30_000, 15_000, 50.0)
    r2  = sim_no_pcm(GHI, 30_000, 50.0)
    sm1 = stability_metrics(r1["kero"])
    sm2 = stability_metrics(r2["kero"])

    print(f"\n  {'='*52}")
    print(f"  STABILITY REPORT — Cilacap · Juni")
    print(f"  {'='*52}")
    print(f"  {'Metrik':<28} {'Dengan PCM':>10} {'Tanpa PCM':>10}")
    print(f"  {'-'*50}")
    print(f"  {'CV produksi (%)':<28} {sm1['cv']:>10.1f} {sm2['cv']:>10.1f}")
    print(f"  {'PSI':<28} {sm1['psi']:>10.3f} {'—':>10}")
    print(f"  {'Total kerosene (L/hari)':<28} {r1['total_kero']:>10.1f} {r2['total_kero']:>10.1f}")
    print(f"  {'Efisiensi sistem (%)':<28} {r1['eff']:>10.1f} {r2['eff']:>10.1f}")
    print(f"  {'Jam padam':<28} {r1['jam_padam']:>10} {r2['jam_padam']:>10}")
    print(f"  {'Gini coefficient':<28} {sm1['gini']:>10.3f} {sm2['gini']:>10.3f}")
    print(f"  {'='*52}\n")