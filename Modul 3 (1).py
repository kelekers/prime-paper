# MASS BALANCE SOLVER
# Menghitung konversi massa end-to-end dari CO₂ emisi kilang menjadi biomassa alga, syngas, hingga E-Kerosene menggunakan stoikiometri gasifikasi dan distribusi Fischer-Tropsch.
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import webbrowser, os, json, math

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

# PARAMETER FISIK & BIOLOGIS
# Ref: Wang et al. (2008)  doi:10.1007/S00253-008-1518-Y
#      Bhola et al. (2014) doi:10.1007/S13762-013-0487-6
ALGA_PARAMS = {
    "ratio_co2_biomass":  1.83,
    "mu_max":             1.80,
    "I_sat":              200.0,
    "Kc":                 0.02,
    "X0":                 0.50,
    "lipid_fraction":     0.30,
    "ash_fraction":       0.05,
    "moisture_harvest":   0.10,
}

# Ref: Ptasinski (2016) doi:10.1002/9781118702116
GASIF_PARAMS = {
    "H_biomass":    1.8,
    "O_biomass":    0.5,
    "N_biomass":    0.2,
    "steam_ratio":  0.5,
    "CO_yield":     0.95,
    "H2_yield":     0.92,
    "T_gasif":      850.0,
    "eta_gasif":    0.82,
}

# Ref: Dry (2002)    doi:10.1016/S0920-5861(01)00453-9
#      Colelli (2023) doi:10.1016/j.enconman.2023.117165
FT_PARAMS = {
    "alpha":           0.87,
    "H2_CO_ratio":     2.05,
    "eta_ft":          0.80,
    "C_kerosene_low":  10,
    "C_kerosene_high": 14,
    "rho_kerosene":    0.800,
    "LHV_kerosene":    43.1,
}

# PARAMETER LOKASI KILANG
# Data emisi CO₂ representatif kilang minyak Indonesia
# Sumber: Pertamina Sustainability Report 2023
KILANG_PROFILES = {
    "RU IV Cilacap (studi kasus)": {
        "co2_ton_hari": 1200,
        "capture_rate": 0.65,
    },
    "IT Semarang (studi kasus)": {
        "co2_ton_hari": 450,
        "capture_rate": 0.70,
    },
    "Kilang skala kecil (50 ktpa)": {
        "co2_ton_hari": 140,
        "capture_rate": 0.80,
    },
    "Kilang menengah (300 ktpa)": {
        "co2_ton_hari": 820,
        "capture_rate": 0.65,
    },
    "Kilang besar (500 ktpa)": {
        "co2_ton_hari": 1370,
        "capture_rate": 0.60,
    },
}

# FUNGSI MASS BALANCE INTI

def co2_capture(co2_available_ton_day: float,
                capture_rate: float) -> float:
    """
    CO₂ yang berhasil ditangkap oleh sistem bioreaktor (ton/hari).
    Ref: Viswanaathan & Sudhakar (2019)
         doi:10.1016/B978-0-12-818258-1.00008-X
    """
    return co2_available_ton_day * capture_rate


def alga_growth_monod(
    co2_captured_ton_day: float,
    irradiance_umol: float,
    reactor_volume_m3: float,
    X0: float = None,
    days: int = 7,
) -> dict:
    """
    Simulasi pertumbuhan alga — Monod kinetics + light limitation.

    dX/dt = μ_max × (I / (I + I_sat)) × (S / (S + Kc)) × X
    S = konsentrasi CO₂ terlarut (g/L)

    Ref: Bhola et al. (2014) doi:10.1007/S13762-013-0487-6
         Hoque et al. (2024) doi:10.3390/plants13172400
    """
    p  = ALGA_PARAMS
    X0 = X0 or p["X0"]

    co2_g_per_liter = min(
        (co2_captured_ton_day * 1e6) / (reactor_volume_m3 * 1000),
        2.0
    )

    f_light = irradiance_umol / (irradiance_umol + p["I_sat"])
    f_co2   = co2_g_per_liter / (co2_g_per_liter + p["Kc"])
    mu_eff  = p["mu_max"] * f_light * f_co2

    X_max   = 8.0
    dt      = 0.1
    t_arr   = np.arange(0, days + dt, dt)
    X_arr   = np.zeros(len(t_arr))
    X_arr[0]= X0

    for i in range(1, len(t_arr)):
        dX = mu_eff * X_arr[i-1] * (1 - X_arr[i-1] / X_max) * dt
        X_arr[i] = X_arr[i-1] + dX

    X_harvest   = X_arr[-1] - X0
    biomass_ton = (X_harvest * reactor_volume_m3 * 1000) / 1e6

    return {
        "t":            t_arr,
        "X":            X_arr,
        "mu_eff":       round(mu_eff, 3),
        "f_light":      round(f_light, 3),
        "f_co2":        round(f_co2, 3),
        "X_final":      round(float(X_arr[-1]), 3),
        "biomass_ton":  round(biomass_ton, 4),
        "co2_g_L":      round(co2_g_per_liter, 4),
    }


def biomass_to_syngas(biomass_dry_ton_day: float) -> dict:
    """
    Konversi biomassa → Syngas via steam gasification.

    Stoikiometri (per mol C):
      CH₁.₈O₀.₅ + 0.5H₂O → CO + 1.4H₂
      MW_biomass_unit ≈ 12 + 1.8 + 8 = 21.8 g/mol (per C-unit)

    Output per ton biomassa kering:
      n_C = 1e6 / 21.8 mol
      n_CO  = n_C × CO_yield
      n_H2  = n_C × 1.4 × H2_yield + n_H2O_steam × H2_yield

    Ref: Ptasinski (2016) doi:10.1002/9781118702116
    """
    p = GASIF_PARAMS
    MW_unit = 12 + p["H_biomass"]*1 + p["O_biomass"]*16 + p["N_biomass"]*14
    n_C_per_ton = 1e6 / MW_unit
    n_C_total   = biomass_dry_ton_day * n_C_per_ton

    n_CO  = n_C_total * p["CO_yield"]
    n_H2  = n_C_total * (p["H_biomass"] / 2 + p["steam_ratio"]) * p["H2_yield"]

    mass_CO_ton  = (n_CO * 28)  / 1e6
    mass_H2_ton  = (n_H2 * 2)   / 1e6
    H2_CO_ratio  = n_H2 / n_CO if n_CO > 0 else 0

    energy_CO_GJ  = mass_CO_ton  * 12.6
    energy_H2_GJ  = mass_H2_ton  * 120.0
    total_energy_GJ = energy_CO_GJ + energy_H2_GJ

    return {
        "n_CO_kmol":       round(n_CO / 1e3, 2),
        "n_H2_kmol":       round(n_H2 / 1e3, 2),
        "mass_CO_ton":     round(mass_CO_ton, 4),
        "mass_H2_ton":     round(mass_H2_ton, 4),
        "H2_CO_ratio":     round(H2_CO_ratio, 3),
        "total_energy_GJ": round(total_energy_GJ, 2),
        "T_gasif":         p["T_gasif"],
        "eta_gasif":       p["eta_gasif"],
    }


def asf_distribution(alpha: float, n_max: int = 30) -> dict:
    """
    Anderson-Schulz-Flory (ASF) chain length distribution.
    Wn = n × (1-α)² × α^(n-1)   — fraksi massa produk rantai-n

    Ref: Dry (2002) doi:10.1016/S0920-5861(01)00453-9
    """
    n_arr  = np.arange(1, n_max + 1)
    Wn     = n_arr * (1 - alpha)**2 * alpha**(n_arr - 1)
    Wn     = Wn / Wn.sum()

    C1_C4   = float(Wn[0:4].sum())
    C5_C9   = float(Wn[4:9].sum())
    C10_C14 = float(Wn[9:14].sum())
    C15_C19 = float(Wn[14:19].sum())
    C20plus = float(Wn[19:].sum())

    return {
        "n":        n_arr,
        "Wn":       Wn,
        "C1_C4":    round(C1_C4   * 100, 2),
        "C5_C9":    round(C5_C9   * 100, 2),
        "C10_C14":  round(C10_C14 * 100, 2),
        "C15_C19":  round(C15_C19 * 100, 2),
        "C20plus":  round(C20plus * 100, 2),
    }


def syngas_to_kerosene(
    mass_CO_ton_day:  float,
    mass_H2_ton_day:  float,
) -> dict:
    """
    Konversi Syngas → E-Kerosene via Fischer-Tropsch (ASF model).

    Langkah:
      1. Hitung mol CO & H₂
      2. Tentukan limiting reactant berdasarkan rasio H₂/CO = 2.05
      3. Konversi CO → total hydrocarbon (CH₂)ₙ
         n·CO + 2n·H₂ → (CH₂)ₙ + n·H₂O
         MW_CH2 = 14 g/mol
      4. Distribusi ASF → fraksi C10–C14
      5. Konversi massa → volume (rho = 0.800 kg/L)

    Ref: Dry (2002)    doi:10.1016/S0920-5861(01)00453-9
         Colelli (2023) doi:10.1016/j.enconman.2023.117165
    """
    p = FT_PARAMS

    mol_CO = (mass_CO_ton_day * 1e6) / 28
    mol_H2 = (mass_H2_ton_day * 1e6) / 2

    H2_needed = mol_CO * p["H2_CO_ratio"]
    if mol_H2 < H2_needed:
        mol_CO_react = mol_H2 / p["H2_CO_ratio"]
    else:
        mol_CO_react = mol_CO

    mass_HC_ton = (mol_CO_react * 14 / 1e6) * p["eta_ft"]

    asf = asf_distribution(p["alpha"])

    mass_kerosene_ton = mass_HC_ton * (asf["C10_C14"] / 100)
    mass_naphtha_ton  = mass_HC_ton * (asf["C5_C9"]   / 100)
    mass_diesel_ton   = mass_HC_ton * (asf["C15_C19"] / 100)
    mass_gas_ton      = mass_HC_ton * (asf["C1_C4"]   / 100)

    vol_kerosene_liter = (mass_kerosene_ton * 1000) / p["rho_kerosene"]
    vol_kerosene_kL    = vol_kerosene_liter / 1000

    energy_GJ = mass_kerosene_ton * p["LHV_kerosene"]

    H2_CO_actual = (mol_H2 / mol_CO) if mol_CO > 0 else 0

    return {
        "mass_HC_ton":        round(mass_HC_ton, 4),
        "mass_kerosene_ton":  round(mass_kerosene_ton, 4),
        "mass_naphtha_ton":   round(mass_naphtha_ton, 4),
        "mass_diesel_ton":    round(mass_diesel_ton, 4),
        "mass_gas_ton":       round(mass_gas_ton, 4),
        "vol_kerosene_liter": round(vol_kerosene_liter, 1),
        "vol_kerosene_kL":    round(vol_kerosene_kL, 3),
        "energy_GJ":          round(energy_GJ, 2),
        "H2_CO_actual":       round(H2_CO_actual, 3),
        "asf":                asf,
    }


def full_mass_balance(
    co2_ton_day:       float,
    capture_rate:      float,
    irradiance_umol:   float,
    reactor_volume_m3: float,
    growth_days:       int   = 7,
    alpha_ft:          float = None,
) -> dict:
    """
    Pipeline lengkap: CO₂ → Alga → Syngas → E-Kerosene.
    Mengembalikan semua metrik per tahap beserta
    carbon intensity (kg CO₂ per liter E-Kerosene).
    """
    if alpha_ft:
        FT_PARAMS["alpha"] = alpha_ft

    co2_cap = co2_capture(co2_ton_day, capture_rate)

    alga = alga_growth_monod(
        co2_cap, irradiance_umol, reactor_volume_m3,
        days=growth_days,
    )
    biomass_day = alga["biomass_ton"] / growth_days
    biomass_dry = biomass_day * (1 - ALGA_PARAMS["moisture_harvest"])

    syngas = biomass_to_syngas(biomass_dry)

    ft = syngas_to_kerosene(syngas["mass_CO_ton"], syngas["mass_H2_ton"])

    co2_combustion = ft["mass_kerosene_ton"] * 3.16
    co2_net        = co2_combustion - co2_cap
    ci_per_liter   = (co2_net * 1e6) / ft["vol_kerosene_liter"] \
                     if ft["vol_kerosene_liter"] > 0 else 0

    co2_to_kerosene_eff = (ft["mass_kerosene_ton"] * 12/170 /
                           (co2_cap / 44 * 12)) * 100 \
                          if co2_cap > 0 else 0

    return {
        "co2_available":     round(co2_ton_day,   2),
        "co2_captured":      round(co2_cap,        2),
        "capture_rate_pct":  round(capture_rate*100, 0),
        "biomass_day_ton":   round(biomass_day,    4),
        "biomass_dry_ton":   round(biomass_dry,    4),
        "alga":              alga,
        "syngas":            syngas,
        "ft":                ft,
        "co2_net_ton":       round(co2_net,        4),
        "ci_per_liter":      round(ci_per_liter,   2),
        "carbon_eff_pct":    round(co2_to_kerosene_eff, 2),
        "vol_kerosene_liter":ft["vol_kerosene_liter"],
        "mass_kerosene_ton": ft["mass_kerosene_ton"],
    }


# PEMBUATAN FIGURE

def make_sankey(mb: dict) -> go.Figure:
    """
    Sankey diagram aliran massa:
    CO₂ Emisi → CO₂ Captured → Biomassa → Syngas → Produk FT
    """
    co2_av  = mb["co2_available"]
    co2_cap = mb["co2_captured"]
    co2_los = co2_av - co2_cap
    bio     = mb["biomass_day_ton"] * 1000
    syn_CO  = mb["syngas"]["mass_CO_ton"]    * 1000
    syn_H2  = mb["syngas"]["mass_H2_ton"]    * 1000
    ker     = mb["ft"]["mass_kerosene_ton"]  * 1000
    nap     = mb["ft"]["mass_naphtha_ton"]   * 1000
    die     = mb["ft"]["mass_diesel_ton"]    * 1000
    gas_ft  = mb["ft"]["mass_gas_ton"]       * 1000

    labels = [
        "CO₂ emisi kilang",
        "CO₂ tidak ditangkap",
        "CO₂ captured",
        "Biomassa alga",
        "CO (syngas)",
        "H₂ (syngas)",
        "E-Kerosene",
        "Naphtha",
        "Diesel sintetis",
        "Gas ringan (C1-C4)",
    ]
    colors_node = [
        GRAY, "#B4B2A9", TEAL, GREEN,
        AMBER, BLUE, RED, CORAL, PURPLE, GRAY,
    ]

    source = [0, 0, 2, 3, 3, 4, 4, 4, 4]
    target = [2, 1, 3, 4, 5, 6, 7, 8, 9]
    value  = [
        co2_cap,
        co2_los,
        bio,
        syn_CO,
        syn_H2,
        ker,
        nap,
        die,
        gas_ft,
    ]
    link_colors = [
        "rgba(29,158,117,0.3)",
        "rgba(136,135,128,0.2)",
        "rgba(99,153,34,0.3)",
        "rgba(242,166,35,0.3)",
        "rgba(55,138,221,0.3)",
        "rgba(226,75,74,0.35)",
        "rgba(216,90,48,0.3)",
        "rgba(127,119,221,0.3)",
        "rgba(136,135,128,0.25)",
    ]

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=18,
            line=dict(color="white", width=0.5),
            label=labels,
            color=colors_node,
            hovertemplate="%{label}<br>Total: <b>%{value:.2f} kg/hari</b><extra></extra>",
        ),
        link=dict(
            source=source, target=target, value=value,
            color=link_colors,
            hovertemplate=(
                "%{source.label} → %{target.label}<br>"
                "<b>%{value:.2f} kg/hari</b><extra></extra>"
            ),
        ),
    ))
    fig.update_layout(
        title=dict(
            text="<b>Aliran massa sistem</b> · CO₂ → Alga → Syngas → E-Kerosene (kg/hari)",
            x=0, xanchor="left", font=dict(size=14, color=TEXT),
        ),
        height=380,
        paper_bgcolor=BG,
        font=dict(family="system-ui,-apple-system,sans-serif",
                  size=11, color=TEXT),
        margin=dict(l=20, r=20, t=70, b=20),
    )
    return fig


def make_asf_fig(alpha: float) -> go.Figure:
    """
    Grafik distribusi ASF — fraksi massa vs panjang rantai karbon.
    Highlight fraksi E-Kerosene (C10–C14).
    """
    asf = asf_distribution(alpha, n_max=25)
    n   = asf["n"]
    Wn  = asf["Wn"] * 100

    colors = [
        RED    if 10 <= ni <= 14 else
        AMBER  if 5  <= ni <= 9  else
        BLUE   if 15 <= ni <= 19 else
        GRAY
        for ni in n
    ]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[f"C{ni}" for ni in n],
        y=Wn.tolist(),
        marker_color=colors,
        marker_line_width=0,
        hovertemplate="C%{x}<br>Fraksi: <b>%{y:.2f}%</b><extra></extra>",
        name="Fraksi massa (%)",
    ))

    fig.add_vrect(
        x0="C9", x1="C15",
        fillcolor=RED, opacity=0.08,
        line_width=0,
        annotation_text="E-Kerosene (C10–C14)",
        annotation_position="top left",
        annotation_font=dict(size=10, color=RED),
    )

    total_kero = asf["C10_C14"]
    fig.update_layout(
        title=dict(
            text=(f"<b>Distribusi ASF</b> · α={alpha:.2f} · "
                  f"Yield E-Kerosene: <b style='color:{RED}'>{total_kero:.1f}%</b><br>"
                  f"<sup>Anderson-Schulz-Flory chain length distribution "
                  f"— Fischer-Tropsch synthesis</sup>"),
            x=0, xanchor="left", font=dict(size=14, color=TEXT),
        ),
        height=320,
        template="plotly_white",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        xaxis=dict(
            title="Panjang rantai karbon",
            tickfont=dict(size=10, color=TEXTS),
            showgrid=False,
        ),
        yaxis=dict(
            title="Fraksi massa (%)",
            tickfont=dict(size=10, color=TEXTS),
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
        ),
        showlegend=False,
        margin=dict(l=55, r=30, t=100, b=55),
        font=dict(family="system-ui,-apple-system,sans-serif"),
    )
    return fig


def make_alga_growth_fig(alga: dict) -> go.Figure:
    """Kurva pertumbuhan alga — densitas sel vs waktu (hari)."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=alga["t"].tolist(),
        y=alga["X"].tolist(),
        name="Densitas sel (g/L)",
        fill="tozeroy",
        line=dict(color=GREEN, width=2.5),
        fillcolor="rgba(99,153,34,0.14)",
        hovertemplate="Hari %{x:.1f}<br>Densitas: <b>%{y:.3f} g/L</b><extra></extra>",
    ))

    fig.add_vline(
        x=alga["t"][-1], line_dash="dot", line_color=TEAL, line_width=1.5,
        annotation_text=f"Harvest  X={alga['X_final']} g/L",
        annotation_font=dict(size=10, color=TEAL),
        annotation_position="top left",
    )

    fig.update_layout(
        title=dict(
            text=(f"<b>Pertumbuhan mikroalga</b> · "
                  f"μ_eff={alga['mu_eff']}/hari · "
                  f"f_light={alga['f_light']} · "
                  f"f_CO₂={alga['f_co2']}<br>"
                  f"<sup>Monod kinetics + light limitation "
                  f"— Bhola et al. (2014)</sup>"),
            x=0, xanchor="left", font=dict(size=14, color=TEXT),
        ),
        height=280,
        template="plotly_white",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        xaxis=dict(
            title="Waktu (hari)",
            tickfont=dict(size=10, color=TEXTS),
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Densitas sel (g/L)",
            tickfont=dict(size=10, color=TEXTS),
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
        ),
        showlegend=False,
        margin=dict(l=55, r=30, t=100, b=55),
        font=dict(family="system-ui,-apple-system,sans-serif"),
    )
    return fig


def make_sensitivity_co2_fig(
    capture_rate:    float,
    irradiance_umol: float,
    reactor_volume:  float,
) -> go.Figure:
    """
    Grafik sensitivitas: produksi E-Kerosene (L/hari)
    vs laju CO₂ masuk, untuk beberapa skenario irradiasi.
    """
    co2_range  = np.linspace(50, 2000, 40)
    irrad_list = [100, 200, 350, 500]
    irrad_colors = [BLUE, TEAL, AMBER, RED]

    fig = go.Figure()
    for irr, col in zip(irrad_list, irrad_colors):
        vols = []
        for co2 in co2_range:
            mb = full_mass_balance(
                co2, capture_rate, irr, reactor_volume)
            vols.append(mb["vol_kerosene_liter"])
        fig.add_trace(go.Scatter(
            x=co2_range.tolist(),
            y=vols,
            name=f"Irradiasi {irr} μmol/m²/s",
            line=dict(color=col, width=2),
            hovertemplate=(
                f"Irradiasi {irr} μmol<br>"
                "CO₂: <b>%{x:.0f} ton/hari</b><br>"
                "E-Kerosene: <b>%{y:.0f} L/hari</b><extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text=("<b>Sensitivitas produksi E-Kerosene</b> · "
                  "Liter/hari vs CO₂ input (ton/hari)<br>"
                  f"<sup>Capture rate {capture_rate*100:.0f}% · "
                  f"Volume bioreaktor {reactor_volume:.0f} m³</sup>"),
            x=0, xanchor="left", font=dict(size=14, color=TEXT),
        ),
        height=300,
        template="plotly_white",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        hovermode="x unified",
        xaxis=dict(
            title="CO₂ input (ton/hari)",
            tickfont=dict(size=10, color=TEXTS),
            showgrid=True, gridcolor="rgba(0,0,0,0.1)",
            zeroline=False,
        ),
        yaxis=dict(
            title="E-Kerosene (L/hari)",
            tickfont=dict(size=10, color=TEXTS),
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
        ),
        legend=dict(
            orientation="h", y=-0.28, x=0.5, xanchor="center",
            font=dict(size=10, color=TEXTS),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=30, t=100, b=80),
        font=dict(family="system-ui,-apple-system,sans-serif"),
    )
    return fig


# Export HTML
def export_html(
    kilang_key:      str   = "RU IV Cilacap (studi kasus)",
    irradiance_umol: float = 350.0,
    reactor_volume:  float = 5000.0,
    growth_days:     int   = 7,
    alpha_ft:        float = 0.87,
) -> str:
    import plotly.io as pio

    kp  = KILANG_PROFILES[kilang_key]
    mb  = full_mass_balance(
        kp["co2_ton_hari"], kp["capture_rate"],
        irradiance_umol, reactor_volume,
        growth_days, alpha_ft,
    )

    fig_sankey  = make_sankey(mb)
    fig_asf     = make_asf_fig(alpha_ft)
    fig_alga    = make_alga_growth_fig(mb["alga"])
    fig_sens    = make_sensitivity_co2_fig(
        kp["capture_rate"], irradiance_umol, reactor_volume)

    sankey_json = pio.to_json(fig_sankey)
    asf_json    = pio.to_json(fig_asf)
    alga_json   = pio.to_json(fig_alga)
    sens_json   = pio.to_json(fig_sens)

    rows = [
        ("CO₂ tersedia (kilang)",    f"{mb['co2_available']:.1f}",  "ton/hari", ""),
        ("CO₂ ditangkap",            f"{mb['co2_captured']:.1f}",   "ton/hari",
         f"capture rate {mb['capture_rate_pct']:.0f}%"),
        ("Biomassa alga (basah)",    f"{mb['biomass_day_ton']*1000:.1f}", "kg/hari",
         f"CO₂/biomassa = {ALGA_PARAMS['ratio_co2_biomass']}"),
        ("Biomassa kering (gasif.)", f"{mb['biomass_dry_ton']*1000:.1f}", "kg/hari",
         f"moisture {ALGA_PARAMS['moisture_harvest']*100:.0f}%"),
        ("CO (syngas)",              f"{mb['syngas']['mass_CO_ton']*1000:.2f}", "kg/hari",
         f"H₂/CO = {mb['syngas']['H2_CO_ratio']:.2f}"),
        ("H₂ (syngas)",              f"{mb['syngas']['mass_H2_ton']*1000:.2f}", "kg/hari",
         f"T gasif = {mb['syngas']['T_gasif']:.0f}°C"),
        ("E-Kerosene",               f"{mb['ft']['mass_kerosene_ton']*1000:.2f}", "kg/hari",
         f"α = {alpha_ft:.2f}, fraksi C10–C14 = {mb['ft']['asf']['C10_C14']:.1f}%"),
        ("Volume E-Kerosene",        f"{mb['vol_kerosene_liter']:.0f}", "L/hari",
         f"{mb['vol_kerosene_liter']*365/1e6:.3f} juta L/tahun"),
        ("Energi kandungan",         f"{mb['ft']['energy_GJ']:.2f}", "GJ/hari",
         f"LHV = {FT_PARAMS['LHV_kerosene']} MJ/kg"),
        ("Carbon intensity",         f"{mb['ci_per_liter']:.1f}", "g CO₂/L",
         "negatif = net sink"),
    ]

    table_rows_html = ""
    for name, val, unit, note in rows:
        hl = ' style="background:#FFF8ED;font-weight:500"' \
             if "Kerosene" in name or "E-Kero" in name else ""
        table_rows_html += (
            f"<tr{hl}>"
            f"<td style='padding:7px 12px;color:{TEXTS}'>{name}</td>"
            f"<td style='padding:7px 12px;text-align:right;"
            f"font-weight:500;color:{TEXT}'>{val}</td>"
            f"<td style='padding:7px 12px;color:{TEXTS}'>{unit}</td>"
            f"<td style='padding:7px 12px;font-size:11px;"
            f"color:{TEXTS}'>{note}</td>"
            f"</tr>\n"
        )

    ci_color = TEAL if mb["ci_per_liter"] < 0 else \
               AMBER if mb["ci_per_liter"] < 50 else RED

    html_out = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mass Balance Solver </title>
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
       flex:1;min-width:140px}}
  .mc.hi{{border-left:3px solid {AMBER};border-radius:0 8px 8px 0}}
  .mc-label{{font-size:11px;color:{TEXTS};margin-bottom:3px}}
  .mc-val{{font-size:22px;font-weight:500;line-height:1}}
  .mc-unit{{font-size:11px;color:{TEXTS}}}
  .card{{background:#fff;border:0.5px solid rgba(0,0,0,.1);
         border-radius:12px;padding:16px;margin-bottom:14px}}
  .section-title{{font-size:13px;font-weight:500;
                  color:{TEXTS};margin-bottom:10px}}
  .controls{{display:flex;gap:12px;flex-wrap:wrap;
             align-items:flex-end;margin-bottom:16px}}
  .ctrl-group{{display:flex;flex-direction:column;gap:4px}}
  .ctrl-label{{font-size:11px;color:{TEXTS}}}
  select{{font-size:13px;padding:6px 10px;
          border:0.5px solid rgba(0,0,0,.15);border-radius:8px;
          background:#fff;color:{TEXT};cursor:pointer;min-width:220px}}
  input[type=range]{{min-width:150px;padding:4px 0}}
  button{{font-size:12px;padding:8px 16px;
          border:0.5px solid rgba(0,0,0,.2);border-radius:8px;
          background:#fff;color:{TEXT};cursor:pointer;font-weight:500}}
  button:hover{{background:#F0F0EC}}
  .val-display{{font-size:12px;font-weight:500;color:{TEXT};min-width:70px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{text-align:left;padding:8px 12px;font-size:11px;
      font-weight:500;color:{TEXTS};border-bottom:1px solid rgba(0,0,0,.08)}}
  tr:not(:last-child) td{{border-bottom:0.5px solid rgba(0,0,0,.05)}}
  .footer{{border-top:0.5px solid rgba(0,0,0,.1);padding-top:10px;
           font-size:10px;color:{TEXTS};margin-top:4px}}
  .badge{{display:inline-block;padding:3px 10px;border-radius:99px;
          font-size:11px;font-weight:500}}
</style>
</head>
<body>
<div class="wrap">

  <h1>Mass Balance Solver (CO₂ → Alga → Syngas → E-Kerosene) </h1>
  <p class="sub">
    Menghitung konversi massa end-to-end dari CO₂ emisi kilang menjadi biomassa alga, syngas, hingga E-Kerosene menggunakan stoikiometri gasifikasi dan distribusi Fischer-Tropsch.
  </p>

  <!-- Controls -->
  <div class="card">
    <div class="section-title">Parameter simulasi</div>
    <div class="controls">
      <div class="ctrl-group">
        <span class="ctrl-label">Profil kilang</span>
        <select id="kilangCtrl">
          {"".join(f'<option value="{k}" {"selected" if k==kilang_key else ""}>'
                   f'{k} ({v["co2_ton_hari"]} ton CO₂/hari)</option>'
                   for k,v in KILANG_PROFILES.items())}
        </select>
      </div>
      <div class="ctrl-group">
        <span class="ctrl-label">
          Irradiasi: <span class="val-display" id="irrVal">{irradiance_umol:.0f} μmol/m²/s</span>
        </span>
        <input type="range" id="irrCtrl"
               min="50" max="600" step="50" value="{irradiance_umol}">
      </div>
      <div class="ctrl-group">
        <span class="ctrl-label">
          Volume bioreaktor: <span class="val-display" id="volVal">{reactor_volume:.0f} m³</span>
        </span>
        <input type="range" id="volCtrl"
               min="500" max="20000" step="500" value="{reactor_volume}">
      </div>
      <div class="ctrl-group">
        <span class="ctrl-label">
          α FT (chain growth): <span class="val-display" id="alphaVal">{alpha_ft:.2f}</span>
        </span>
        <input type="range" id="alphaCtrl"
               min="0.70" max="0.95" step="0.01" value="{alpha_ft}">
      </div>
      <button onclick="runSim()">Jalankan simulasi</button>
    </div>
  </div>

  <!-- Metric Cards -->
  <div class="metrics">
    <div class="mc hi">
      <div class="mc-label">E-Kerosene / hari</div>
      <div><span class="mc-val" id="mc-vol">{mb['vol_kerosene_liter']:.0f}</span>
           <span class="mc-unit"> L/hari</span></div>
    </div>
    <div class="mc">
      <div class="mc-label">E-Kerosene / tahun</div>
      <div><span class="mc-val" id="mc-vol-yr">{mb['vol_kerosene_liter']*365/1e6:.3f}</span>
           <span class="mc-unit"> juta L/tahun</span></div>
    </div>
    <div class="mc">
      <div class="mc-label">CO₂ ditangkap</div>
      <div><span class="mc-val" id="mc-co2">{mb['co2_captured']:.0f}</span>
           <span class="mc-unit"> ton/hari</span></div>
    </div>
    <div class="mc">
      <div class="mc-label">Biomassa alga</div>
      <div><span class="mc-val" id="mc-bio">{mb['biomass_day_ton']*1000:.0f}</span>
           <span class="mc-unit"> kg/hari</span></div>
    </div>
    <div class="mc">
      <div class="mc-label">H₂/CO syngas</div>
      <div><span class="mc-val" id="mc-h2co">{mb['syngas']['H2_CO_ratio']:.2f}</span>
           <span class="mc-unit"> (target 2.0–2.1)</span></div>
    </div>
    <div class="mc">
      <div class="mc-label">Carbon intensity</div>
      <div><span class="mc-val" id="mc-ci"
                 style="color:{ci_color}">{mb['ci_per_liter']:.1f}</span>
           <span class="mc-unit"> g CO₂/L</span></div>
    </div>
  </div>

  <!-- Sankey -->
  <div class="card">
    <div class="section-title">Aliran massa sistem (Sankey diagram)</div>
    <div id="sankeyChart"></div>
  </div>

  <!-- Tabel mass balance -->
  <div class="card">
    <div class="section-title">Tabel mass balance lengkap</div>
    <table id="mbTable">
      <thead>
        <tr>
          <th>Parameter</th><th style="text-align:right">Nilai</th>
          <th>Satuan</th><th>Keterangan</th>
        </tr>
      </thead>
      <tbody id="mbBody">{table_rows_html}</tbody>
    </table>
  </div>

  <!-- ASF + Alga growth -->
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:14px">
    <div class="card">
      <div class="section-title">Distribusi ASF — Fischer-Tropsch</div>
      <div id="asfChart"></div>
    </div>
    <div class="card">
      <div class="section-title">Pertumbuhan mikroalga (Monod)</div>
      <div id="algaChart"></div>
    </div>
  </div>

  <!-- Sensitivitas -->
  <div class="card">
    <div class="section-title">Sensitivitas produksi vs CO₂ input & irradiasi</div>
    <div id="sensChart"></div>
  </div>

  <div class="footer">
    Rumus: Monod kinetics (Bhola et al. 2014) · CO₂/biomassa rasio (Wang et al. 2008) ·
    Steam gasification stoikiometri (Ptasinski 2016) ·
    ASF distribution (Dry 2002) · FT yield (Colelli et al. 2023) ·
    Carbon intensity (Micheli et al. 2022)
  </div>
</div>

<script>
const KILANG = {json.dumps(KILANG_PROFILES)};
const ALGA_P = {json.dumps({k:v for k,v in ALGA_PARAMS.items()
                             if isinstance(v,(int,float))})};
const GASIF_P= {json.dumps({k:v for k,v in GASIF_PARAMS.items()
                             if isinstance(v,(int,float))})};
const FT_P   = {json.dumps({k:v for k,v in FT_PARAMS.items()
                             if isinstance(v,(int,float))})};
const AMBER='{AMBER}',RED='{RED}',BLUE='{BLUE}',TEAL='{TEAL}',
      GREEN='{GREEN}',PURPLE='{PURPLE}',CORAL='{CORAL}',
      GRAY='{GRAY}',TEXT='{TEXT}',TEXTS='{TEXTS}',BG='{BG}';

function asfDist(alpha,nMax=25){{
  const n=Array.from({{length:nMax}},(_,i)=>i+1);
  let Wn=n.map(ni=>ni*(1-alpha)**2*alpha**(ni-1));
  const s=Wn.reduce((a,b)=>a+b,0);
  Wn=Wn.map(w=>w/s);
  const C10_14=Wn.slice(9,14).reduce((a,b)=>a+b,0)*100;
  return{{n,Wn,C10_14:Math.round(C10_14*10)/10}};
}}

function simMB(co2Avail,captureRate,irr,vol,alpha){{
  const co2Cap=co2Avail*captureRate;
  const Isat=ALGA_P.I_sat||200,Kc=ALGA_P.Kc||0.02;
  const muMax=ALGA_P.mu_max||1.8;
  const co2gL=Math.min((co2Cap*1e6)/(vol*1000),2.0);
  const fL=irr/(irr+Isat),fC=co2gL/(co2gL+Kc);
  const muEff=muMax*fL*fC;
  const Xmax=8.0,dt=0.1,days=7;
  let X=ALGA_P.X0||0.5,X0=X;
  for(let t=0;t<days;t+=dt){{
    X+=muEff*X*(1-X/Xmax)*dt;
  }}
  const bioTon=(X-X0)*vol*1000/1e6;
  const bioDayDry=bioTon/days*(1-0.10);

  const MWunit=12+GASIF_P.H_biomass*1+GASIF_P.O_biomass*16+GASIF_P.N_biomass*14;
  const nC=bioDayDry*1e6/MWunit;
  const nCO=nC*GASIF_P.CO_yield;
  const nH2=nC*(GASIF_P.H_biomass/2+GASIF_P.steam_ratio)*GASIF_P.H2_yield;
  const mCO=nCO*28/1e6,mH2=nH2*2/1e6;
  const H2CO=mH2>0&&mCO>0?nH2/nCO:0;

  const asf=asfDist(alpha);
  const molCO=mCO*1e6/28,molH2=mH2*1e6/2;
  const H2need=molCO*FT_P.H2_CO_ratio;
  const molCOr=molH2<H2need?molH2/FT_P.H2_CO_ratio:molCO;
  const mHC=molCOr*14/1e6*FT_P.eta_ft;
  const mKer=mHC*asf.C10_14/100;
  const volLiter=mKer*1000/FT_P.rho_kerosene;

  const co2comb=mKer*3.16;
  const ci=(co2comb-co2Cap)*1e6/volLiter;

  return{{co2Cap:Math.round(co2Cap*10)/10,
          bioKg:Math.round(bioDayDry*1000),
          H2CO:Math.round(H2CO*100)/100,
          mCO:Math.round(mCO*1000*100)/100,
          mH2:Math.round(mH2*1000*100)/100,
          mKer:Math.round(mKer*1000*100)/100,
          volL:Math.round(volLiter),
          volYr:Math.round(volLiter*365/1e6*1000)/1000,
          ci:Math.round(ci*10)/10,
          C10_14:asf.C10_14,
          muEff:Math.round(muEff*1000)/1000,
          fL:Math.round(fL*1000)/1000,
          fC:Math.round(fC*1000)/1000}};
}}

function renderASF(alpha){{
  const asf=asfDist(alpha);
  const colors=asf.n.map(ni=>ni>=10&&ni<=14?RED:ni>=5&&ni<=9?AMBER:ni>=15&&ni<=19?BLUE:GRAY);
  Plotly.react('asfChart',[{{
    type:'bar',x:asf.n.map(ni=>'C'+ni),
    y:asf.Wn.map(w=>Math.round(w*10000)/100),
    marker:{{color:colors,line:{{width:0}}}},
    hovertemplate:'C%{{x}}<br><b>%{{y:.2f}}%</b><extra></extra>',
  }}],{{
    template:'plotly_white',paper_bgcolor:'rgba(0,0,0,0)',
    plot_bgcolor:'rgba(0,0,0,0)',height:260,
    title:{{text:`α=${{alpha}} · Yield E-Kerosene: <b style='color:${{RED}}'>${{asf.C10_14}}%</b>`,
            font:{{size:12,color:TEXT}},x:0,xanchor:'left'}},
    xaxis:{{tickfont:{{size:9,color:TEXTS}},showgrid:false}},
    yaxis:{{title:'%',tickfont:{{size:9,color:TEXTS}},showgrid:true,
            gridcolor:'rgba(0,0,0,0.06)',zeroline:false}},
    margin:{{l:40,r:10,t:50,b:40}},showlegend:false,
    font:{{family:'system-ui,-apple-system,sans-serif'}},
    shapes:[{{type:'rect',x0:'C9',x1:'C15',y0:0,y1:1,
              xref:'x',yref:'paper',fillcolor:RED,opacity:0.07,line:{{width:0}}}}],
  }},{{responsive:true,displayModeBar:false}});
}}

function updateMetrics(r){{
  const ci_color=r.ci<0?TEAL:r.ci<50?'{AMBER}':RED;
  document.getElementById('mc-vol').textContent=r.volL;
  document.getElementById('mc-vol-yr').textContent=r.volYr;
  document.getElementById('mc-co2').textContent=r.co2Cap;
  document.getElementById('mc-bio').textContent=r.bioKg;
  document.getElementById('mc-h2co').textContent=r.H2CO;
  document.getElementById('mc-ci').textContent=r.ci;
  document.getElementById('mc-ci').style.color=ci_color;
}}

function updateTable(r,kilangKey,irr,vol,alpha){{
  const kp=KILANG[kilangKey];
  const rows=[
    ["CO₂ tersedia (kilang)",    kp.co2_ton_hari.toFixed(1), "ton/hari",""],
    ["CO₂ ditangkap",            r.co2Cap,   "ton/hari",`capture rate ${{Math.round(kp.capture_rate*100)}}%`],
    ["Biomassa kering (gasif.)", r.bioKg,    "kg/hari", "moisture 10%"],
    ["CO (syngas)",              r.mCO,      "kg/hari", `H₂/CO = ${{r.H2CO}}`],
    ["H₂ (syngas)",              r.mH2,      "kg/hari", ""],
    ["E-Kerosene (massa)",       r.mKer,     "kg/hari", `C10–C14 = ${{r.C10_14}}%`],
    ["E-Kerosene (volume)",      r.volL,     "L/hari",  `${{r.volYr}} juta L/tahun`],
    ["Carbon intensity",         r.ci,       "g CO₂/L", "negatif = net sink"],
  ];
  document.getElementById('mbBody').innerHTML=rows.map(([n,v,u,note],i)=>{{
    const hl=n.includes('E-Kerosene')?' style="background:#FFF8ED;font-weight:500"':'';
    return `<tr${{hl}}><td style="padding:7px 12px;color:${{TEXTS}}">${{n}}</td>`+
           `<td style="padding:7px 12px;text-align:right;font-weight:500">${{v}}</td>`+
           `<td style="padding:7px 12px;color:${{TEXTS}}">${{u}}</td>`+
           `<td style="padding:7px 12px;font-size:11px;color:${{TEXTS}}">${{note}}</td></tr>`;
  }}).join('');
}}

document.getElementById('irrCtrl').oninput=function(){{
  document.getElementById('irrVal').textContent=this.value+' μmol/m²/s';
}};
document.getElementById('volCtrl').oninput=function(){{
  document.getElementById('volVal').textContent=this.value+' m³';
}};
document.getElementById('alphaCtrl').oninput=function(){{
  document.getElementById('alphaVal').textContent=parseFloat(this.value).toFixed(2);
  renderASF(parseFloat(this.value));
}};

function runSim(){{
  const kilangKey=document.getElementById('kilangCtrl').value;
  const irr=parseFloat(document.getElementById('irrCtrl').value);
  const vol=parseFloat(document.getElementById('volCtrl').value);
  const alpha=parseFloat(document.getElementById('alphaCtrl').value);
  const kp=KILANG[kilangKey];
  const r=simMB(kp.co2_ton_hari,kp.capture_rate,irr,vol,alpha);
  updateMetrics(r);
  updateTable(r,kilangKey,irr,vol,alpha);
  renderASF(alpha);
}}

const sankeyData={sankey_json};
const asfData   ={asf_json};
const algaData  ={alga_json};
const sensData  ={sens_json};
Plotly.newPlot('sankeyChart', sankeyData.data, sankeyData.layout,
               {{responsive:true,displayModeBar:false}});
Plotly.newPlot('asfChart',    asfData.data,    asfData.layout,
               {{responsive:true,displayModeBar:false}});
Plotly.newPlot('algaChart',   algaData.data,   algaData.layout,
               {{responsive:true,displayModeBar:false}});
Plotly.newPlot('sensChart',   sensData.data,   sensData.layout,
               {{responsive:true,displayModeBar:false}});
</script>
</body>
</html>"""

    fname = f"Modul3_MassBalance_{kilang_key.split()[0]}.html"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"[✓] Exported: {fname}")
    webbrowser.open("file://" + os.path.abspath(fname))
    return fname


# Entry Point
if __name__ == "__main__":
    print("\n  Modul 3 — Mass Balance Solver")

    fname = export_html(
        kilang_key      = "RU IV Cilacap (studi kasus)",
        irradiance_umol = 350.0,
        reactor_volume  = 5000.0,
        growth_days     = 7,
        alpha_ft        = 0.87,
    )
    print(f"\n  [✓] Dashboard siap: {fname}")
    print("  Browser akan terbuka otomatis.\n")

    kp = KILANG_PROFILES["RU IV Cilacap (studi kasus)"]
    mb = full_mass_balance(
        kp["co2_ton_hari"], kp["capture_rate"], 350.0, 5000.0)
    print(f"  {'='*50}")
    print(f"  MASS BALANCE RINGKASAN — RU IV Cilacap")
    print(f"  {'='*50}")
    print(f"  CO₂ tersedia      : {mb['co2_available']:.1f} ton/hari")
    print(f"  CO₂ ditangkap     : {mb['co2_captured']:.1f} ton/hari")
    print(f"  Biomassa alga     : {mb['biomass_day_ton']*1000:.1f} kg/hari")
    print(f"  H₂/CO syngas      : {mb['syngas']['H2_CO_ratio']:.3f}")
    print(f"  E-Kerosene        : {mb['vol_kerosene_liter']:.0f} L/hari")
    print(f"  E-Kerosene/tahun  : {mb['vol_kerosene_liter']*365/1e6:.3f} juta L")
    print(f"  Carbon intensity  : {mb['ci_per_liter']:.1f} g CO₂/L")
    print(f"  {'='*50}\n")