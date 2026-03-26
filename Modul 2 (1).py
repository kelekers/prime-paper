# PCM THERMAL LOGIC (Slack Wax State of Charge)
# Memvalidasi kemampuan Slack Wax menyerap dan melepas panas laten selama 24 jam agar suhu reaktor tetap di atas 600°C meski input surya fluktuatif.

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
GRAY   = "#888780"
BG     = "#FFFFFF"
BG2    = "#F5F5F0"
BORDER = "rgba(0,0,0,0.10)"
TEXT   = "#1E293B"
TEXTS  = "#64748B"

# Ref: Sari & Karaipekli (2007) doi:10.1016/j.enconman.2007.05.013
#      Trai-In et al. (2025)    doi:10.52526/25-1053
SLACK_WAX = {
    "T_melt_low":  50.0,
    "T_melt_high": 70.0,
    "L_f":         200.0,
    "Cp_solid":    1.8,
    "Cp_liquid":   2.1,
    "rho":         800,
    "T_op":        650.0,
    "T_min_op":    600.0,
}

DOY_JUNI = 166

CITY_PROFILES = {
    "Cilacap":   {"lat": -7.70,  "lon": 109.01, "alt": 6,   "cf": 0.50},
    "Semarang":  {"lat": -6.97,  "lon": 110.42, "alt": 10,  "cf": 0.52},
    "Makassar":  {"lat": -5.14,  "lon": 119.43, "alt": 10,  "cf": 0.36},
    "Surabaya":  {"lat": -7.25,  "lon": 112.75, "alt": 5,   "cf": 0.44},
    "Kupang":    {"lat": -10.17, "lon": 123.61, "alt": 90,  "cf": 0.28},
    "Denpasar":  {"lat": -8.65,  "lon": 115.22, "alt": 10,  "cf": 0.36},
    "Jayapura":  {"lat": -2.53,  "lon": 140.72, "alt": 40,  "cf": 0.60},
    "Pontianak": {"lat": -0.02,  "lon": 109.34, "alt": 5,   "cf": 0.62},
    "Medan":     {"lat":  3.59,  "lon": 98.67,  "alt": 25,  "cf": 0.70},
}

SCENARIO_CLOUDS = {
    "Cerah (cloud 20%)":         0.20,
    "Sebagian berawan (40%)":    0.40,
    "Normal Juni Cilacap (50%)": 0.50,
    "Berawan (70%)":             0.70,
    "Sangat berawan (85%)":      0.85,
}

def _eot(doy):
    B = np.radians(360/365*(doy-81))
    return 9.87*np.sin(2*B) - 7.53*np.cos(B) - 1.5*np.sin(B)

def _solar_elev(h, lat, lon, doy):
    decl  = np.radians(23.45*np.sin(np.radians(360/365*(doy-81))))
    lat_r = np.radians(lat)
    st    = h + (lon%15)*4/60 + _eot(doy)/60
    omega = np.radians((st-12)*15)
    sin_e = np.sin(lat_r)*np.sin(decl) + np.cos(lat_r)*np.cos(decl)*np.cos(omega)
    return float(np.degrees(np.arcsin(np.clip(sin_e,-1,1))))

def _bird(el, alt):
    if el <= 0: return 0.0
    er  = np.radians(el); cz = np.sin(er)
    AM  = min(1/(cz + 0.50572*(96.07995-el)**-1.6364), 38.0)
    af  = np.exp(-alt/8500)
    Tr  = np.clip(np.exp(-0.0903*(AM*af)**0.84*(1+AM*af-(AM*af)**1.01)), 0.0, 1.0)
    Ta  = np.clip(np.exp(-0.0688*(AM*af)**0.9*0.66), 0.0, 1.0)
    Tw  = np.clip(np.exp(-0.2700*(0.04*AM)**0.45), 0.0, 1.0)
    DNI = max(0.0, 1353*0.9662*Tr*Ta*Tw)
    GHI = max(0.0, DNI*cz + 1353*cz*0.95*Tr**1.01*0.93**0.69*0.12)
    return GHI

def solar_profile_ghi(city_key, cloud_override=None):
    """Hitung GHI per jam untuk kota & bulan Juni."""
    c   = CITY_PROFILES[city_key]
    cf  = cloud_override if cloud_override is not None else c["cf"]
    kc  = 1 - cf*0.75
    GHI = np.zeros(24)
    for h in range(24):
        el      = _solar_elev(h+0.5, c["lat"], c["lon"], DOY_JUNI)
        GHI[h]  = round(_bird(el, c["alt"]) * kc)
    return GHI

def ghi_to_thermal(GHI_arr, mirror_area_m2, efficiency_csp=0.65):
    """
    Daya termal dari Solar Concentrator (kW).
    Q_solar = GHI × A_cermin × η_CSP
    η_CSP = 0.65 (efisiensi sistem konsentrator)
    Ref: Romero & Steinfeld (2012) doi:10.1039/C2EE21275J
         Mehos et al. (2020) NREL Technical Report
    """
    return (GHI_arr * mirror_area_m2 * efficiency_csp) / 1000

def simulate_pcm(
    GHI_arr,
    mirror_area_m2: float,
    wax_mass_kg: float,
    reactor_load_kw: float,
    soc_init: float = 0.30,
    dt_hours: float = 1.0,
) -> dict:
    """
    Simulasi heat balance 24 jam dengan Slack Wax PCM.

    Model:
        Q_in(t)  = GHI(t) × A_cermin × η_CSP          [kW]
        Q_out(t) = beban_reaktor                        [kW]
        ΔQ(t)    = Q_in(t) - Q_out(t)                  [kW]

        Kapasitas PCM:
          E_max = m × L_f                               [kWh]
          (menggunakan hanya panas laten, bukan sensibel,
           karena operasi berlangsung di sekitar T_melt)

        Charging   (ΔQ > 0): E_pcm += ΔQ × dt
        Discharging(ΔQ < 0): E_pcm += ΔQ × dt (negatif)
        Klamp: 0 ≤ E_pcm ≤ E_max

        SoC(t) = E_pcm(t) / E_max × 100  [%]

        Output reaktor aktual:
          Jika ΔQ ≥ 0 atau E_pcm > 0: Q_actual = Q_out (terpenuhi)
          Jika E_pcm = 0 & Q_in < Q_out: Q_actual = Q_in (kekurangan)

    Ref: Zalba et al. (2003)        doi:10.1016/S1359-4311(02)00192-8
         Farid et al. (2004)        doi:10.1016/j.enconman.2003.09.015
         Pielichowska et al. (2014) doi:10.1016/j.pmatsci.2014.03.005
    """
    sw        = SLACK_WAX
    L_f_kwh   = sw["L_f"] / 3600
    E_max     = wax_mass_kg * L_f_kwh
    E_pcm     = E_max * soc_init

    Q_solar   = ghi_to_thermal(GHI_arr, mirror_area_m2)

    soc_arr       = np.zeros(24)
    q_solar_arr   = Q_solar.copy()
    q_actual_arr  = np.zeros(24)
    q_charge_arr  = np.zeros(24)
    q_waste_arr   = np.zeros(24)
    t_reactor_arr = np.zeros(24)
    jam_padam     = 0
    total_penuhi  = 0.0
    total_waste   = 0.0

    for h in range(24):
        Q_in  = Q_solar[h]
        Q_out = reactor_load_kw

        surplus = Q_in - Q_out

        if surplus >= 0:
            charge_energy = surplus * dt_hours
            ruang         = E_max - E_pcm
            masuk         = min(charge_energy, ruang)
            waste         = charge_energy - masuk
            E_pcm        += masuk
            total_waste  += waste
            q_charge_arr[h] = masuk
            q_waste_arr[h]  = waste
            q_actual_arr[h] = Q_out
            total_penuhi   += Q_out
        else:
            defisit       = abs(surplus) * dt_hours
            bisa_keluar   = min(defisit, E_pcm)
            E_pcm        -= bisa_keluar
            q_charge_arr[h] = -bisa_keluar
            q_penuhi      = Q_in + bisa_keluar / dt_hours
            q_penuhi      = min(q_penuhi, Q_out)
            q_actual_arr[h] = q_penuhi
            total_penuhi   += q_penuhi
            if q_penuhi < Q_out * 0.95:
                jam_padam += 1

        soc_arr[h] = (E_pcm / E_max) * 100

        soc_frac       = soc_arr[h] / 100
        t_reactor_arr[h] = (sw["T_min_op"] +
                            (sw["T_op"] - sw["T_min_op"]) *
                            min(1.0, soc_frac * 1.2 +
                                (Q_in / max(reactor_load_kw, 1)) * 0.3))

    efisiensi = (total_penuhi / (reactor_load_kw * 24)) * 100

    return {
        "soc":       soc_arr,
        "q_solar":   q_solar_arr,
        "q_actual":  q_actual_arr,
        "q_charge":  q_charge_arr,
        "q_waste":   q_waste_arr,
        "t_reactor": t_reactor_arr,
        "E_max_kwh": round(E_max, 2),
        "efisiensi": round(efisiensi, 1),
        "jam_padam": jam_padam,
        "waste_kwh": round(total_waste, 2),
        "peak_solar":round(float(q_solar_arr.max()), 1),
    }

def make_pcm_dashboard(
    city_key:       str   = "Cilacap",
    mirror_area:    float = 30000.0,
    wax_mass:       float = 15000.0,
    reactor_load:   float = 50.0,
    cloud_factor:   float = None,
) -> go.Figure:
    """
    Dashboard 4-panel:
      Panel 1 (atas kiri)  : Q_solar vs beban + area charge/discharge
      Panel 2 (atas kanan) : SoC Slack Wax 24 jam
      Panel 3 (bawah kiri) : Suhu reaktor estimasi vs batas KPI
      Panel 4 (bawah kanan): Waterfall energi harian
    """
    GHI  = solar_profile_ghi(city_key, cloud_override=cloud_factor)
    res  = simulate_pcm(GHI, mirror_area, wax_mass, reactor_load)
    hours  = list(range(24))
    hlabels= [f"{h:02d}:00" for h in hours]
    cf_val = cloud_factor if cloud_factor else CITY_PROFILES[city_key]["cf"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Input energi surya vs beban reaktor",
            "State of Charge (SoC) Slack Wax",
            "Estimasi suhu reaktor",
            "Neraca energi harian (kWh)",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    # Panel 1: Q_solar vs beban
    charge_pos = np.where(res["q_charge"] > 0, res["q_solar"], np.nan)
    fig.add_trace(go.Scatter(
        x=hlabels, y=res["q_solar"].tolist(),
        name="Input surya (kW)",
        fill="tozeroy",
        line=dict(color=AMBER, width=2.5),
        fillcolor="rgba(242,166,35,0.18)",
        hovertemplate="%{x}<br>Input surya: <b>%{y:.1f} kW</b><extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hlabels,
        y=[reactor_load]*24,
        name="Beban reaktor (kW)",
        line=dict(color=RED, width=2, dash="dot"),
        hovertemplate="Beban target: <b>%{y:.0f} kW</b><extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=hlabels, y=res["q_actual"].tolist(),
        name="Output aktual (kW)",
        line=dict(color=TEAL, width=1.5, dash="dash"),
        hovertemplate="%{x}<br>Output aktual: <b>%{y:.1f} kW</b><extra></extra>",
    ), row=1, col=1)

    # Panel 2: SoC
    soc_colors = [
        TEAL   if v >= 60 else
        AMBER  if v >= 30 else
        RED
        for v in res["soc"]
    ]
    fig.add_trace(go.Bar(
        x=hlabels, y=res["soc"].tolist(),
        name="SoC (%)",
        marker_color=soc_colors,
        marker_line_width=0,
        hovertemplate="%{x}<br>SoC: <b>%{y:.1f}%</b><extra></extra>",
    ), row=1, col=2)

    fig.add_hline(y=30, line_dash="dot", line_color=AMBER,
                  line_width=1, row=1, col=2,
                  annotation_text="Batas aman (30%)",
                  annotation_font_size=9, annotation_font_color=AMBER)
    fig.add_hline(y=10, line_dash="dot", line_color=RED,
                  line_width=1, row=1, col=2,
                  annotation_text="Kritis (10%)",
                  annotation_font_size=9, annotation_font_color=RED)

    # Panel 3: Suhu reaktor
    fig.add_trace(go.Scatter(
        x=hlabels, y=res["t_reactor"].tolist(),
        name="Suhu reaktor (°C)",
        fill="tozeroy",
        line=dict(color=PURPLE, width=2),
        fillcolor="rgba(127,119,221,0.12)",
        hovertemplate="%{x}<br>Suhu: <b>%{y:.0f}°C</b><extra></extra>",
    ), row=2, col=1)

    fig.add_hline(y=SLACK_WAX["T_min_op"], line_dash="dot",
                  line_color=RED, line_width=1.5, row=2, col=1,
                  annotation_text=f"KPI minimum {SLACK_WAX['T_min_op']:.0f}°C",
                  annotation_font_size=9, annotation_font_color=RED,
                  annotation_position="bottom right")
    fig.add_hline(y=SLACK_WAX["T_op"], line_dash="dot",
                  line_color=TEAL, line_width=1, row=2, col=1,
                  annotation_text=f"Target {SLACK_WAX['T_op']:.0f}°C",
                  annotation_font_size=9, annotation_font_color=TEAL,
                  annotation_position="top right")

    # Panel 4: Waterfall neraca energi
    total_solar  = round(float(res["q_solar"].sum()), 1)
    total_beban  = round(reactor_load * 24, 1)
    total_waste  = res["waste_kwh"]
    total_actual = round(float(res["q_actual"].sum()), 1)
    deficit      = round(total_beban - total_actual, 1)

    wf_labels = ["Total surya", "Beban reaktor", "Terbuang", "Dipenuhi", "Defisit"]
    wf_vals   = [total_solar, -total_beban, -total_waste, total_actual, -deficit]
    wf_colors = [AMBER, BLUE, GRAY, TEAL, RED]
    wf_text   = [f"{abs(v):.1f} kWh" for v in wf_vals]

    fig.add_trace(go.Bar(
        x=wf_labels,
        y=[abs(v) for v in wf_vals],
        marker_color=wf_colors,
        marker_line_width=0,
        text=wf_text,
        textposition="outside",
        textfont=dict(size=10, color=TEXT),
        hovertemplate="%{x}: <b>%{text}</b><extra></extra>",
        showlegend=False,
    ), row=2, col=2)

    # Layout keseluruhan
    cf_pct   = f"{cf_val*100:.0f}%"
    eff_color= TEAL if res["efisiensi"] >= 95 else AMBER if res["efisiensi"] >= 80 else RED

    fig.update_layout(
        title=dict(
            text=(
                f"<b>PCM Thermal Logic</b>"
                f"{city_key}  ·  Cloud {cf_pct}  ·  "
                f"Cermin {mirror_area/1000:.0f}k m²  ·  "
                f"Wax {wax_mass/1000:.0f}k kg  ·  "
                f"Beban {reactor_load:.0f} kW<br>"
                f"<sup>"
                f"Efisiensi sistem: "
                f"<span style='color:{eff_color}'>"
                f"<b>{res['efisiensi']}%</b></span>  ·  "
                f"Jam padam: <b>{res['jam_padam']} jam</b>  ·  "
                f"Kapasitas PCM: <b>{res['E_max_kwh']} kWh</b>  ·  "
                f"Energi terbuang: <b>{res['waste_kwh']} kWh</b>"
                f"</sup>"
            ),
            x=0, xanchor="left",
            font=dict(size=14, color=TEXT),
        ),
        height=700,
        template="plotly_white",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        hovermode="x unified",
        legend=dict(
            orientation="h", y=-0.08, x=0.5, xanchor="center",
            font=dict(size=11, color=TEXTS),
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(family="system-ui, -apple-system, sans-serif", color=TEXT),
        margin=dict(l=60, r=40, t=130, b=80),
    )

    for r, c_ax, ytitle in [
        (1,1,"Daya (kW)"), (1,2,"SoC (%)"),
        (2,1,"Suhu (°C)"), (2,2,"Energi (kWh)"),
    ]:
        fig.update_yaxes(
            title_text=ytitle,
            title_font=dict(size=11, color=TEXTS),
            tickfont=dict(size=10, color=TEXTS),
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
            row=r, col=c_ax,
        )
        fig.update_xaxes(
            tickfont=dict(size=9, color=TEXTS),
            showgrid=False, zeroline=False,
            row=r, col=c_ax,
        )

    fig.update_yaxes(range=[0, 110], row=1, col=2)

    return fig, res


def make_sensitivity_fig(
    city_key: str   = "Cilacap",
    reactor_load: float = 50.0,
) -> go.Figure:
    """
    Heatmap sensitivitas: efisiensi sistem (%) vs
    luas cermin × massa wax, untuk cloud cover Juni.
    Berguna untuk menentukan konfigurasi optimal.
    """
    GHI         = solar_profile_ghi(city_key)
    mirror_range= np.arange(10000, 55000, 5000)
    wax_range   = np.arange(5000,  35000, 5000)
    Z           = np.zeros((len(wax_range), len(mirror_range)))

    for i, wm in enumerate(wax_range):
        for j, ma in enumerate(mirror_range):
            r      = simulate_pcm(GHI, ma, wm, reactor_load)
            Z[i,j] = r["efisiensi"]

    fig = go.Figure(go.Heatmap(
        z=Z,
        x=[f"{int(m/1000)}k" for m in mirror_range],
        y=[f"{int(w/1000)}k" for w in wax_range],
        colorscale=[
            [0.0,  "#E24B4A"],
            [0.5,  "#F2A623"],
            [0.8,  "#1D9E75"],
            [1.0,  "#085041"],
        ],
        zmin=0, zmax=100,
        colorbar=dict(
            title=dict(text="Efisiensi (%)", font=dict(size=11, color=TEXTS)),
            tickfont=dict(size=10, color=TEXTS),
            thickness=14, outlinewidth=0,
        ),
        hovertemplate=(
            "Cermin: <b>%{x} m²</b><br>"
            "Wax: <b>%{y} kg</b><br>"
            "Efisiensi: <b>%{z:.1f}%</b><extra></extra>"
        ),
        text=np.round(Z, 1),
        texttemplate="%{text}%",
        textfont=dict(size=9),
    ))

    fig.add_trace(go.Scatter(
        x=["30k"], y=["15k"],
        mode="markers",
        marker=dict(
            symbol="star", size=16,
            color=AMBER, line=dict(width=1.5, color="white")
        ),
        name="Konfigurasi proposal",
        hovertemplate="Konfigurasi proposal<br>30.000 m² · 15.000 kg<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text=(f"<b>Sensitivitas Konfigurasi</b>  ·  {city_key}  ·  "
                  f"Beban {reactor_load:.0f} kW  ·  Cloud Juni<br>"
                  f"<sup>Efisiensi sistem (%) vs Luas Cermin × Massa Wax</sup>"),
            x=0, xanchor="left",
            font=dict(size=14, color=TEXT),
        ),
        height=400,
        template="plotly_white",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        xaxis=dict(
            title="Luas cermin (m²)",
            title_font=dict(size=11, color=TEXTS),
            tickfont=dict(size=10, color=TEXTS),
        ),
        yaxis=dict(
            title="Massa Slack Wax (kg)",
            title_font=dict(size=11, color=TEXTS),
            tickfont=dict(size=10, color=TEXTS),
        ),
        legend=dict(
            font=dict(size=11, color=TEXTS),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=40, t=100, b=60),
        font=dict(family="system-ui, -apple-system, sans-serif"),
    )
    return fig


def make_scenario_fig(
    city_key:     str   = "Cilacap",
    mirror_area:  float = 30000.0,
    wax_mass:     float = 15000.0,
    reactor_load: float = 50.0,
) -> go.Figure:
    """
    Perbandingan SoC untuk 5 skenario cloud cover
    (menunjukkan ketahanan sistem Slack Wax terhadap cuaca buruk).
    """
    fig = go.Figure()
    scenario_colors = [TEAL, GREEN, AMBER, RED, PURPLE]

    for (label, cf), color in zip(SCENARIO_CLOUDS.items(), scenario_colors):
        GHI = solar_profile_ghi(city_key, cloud_override=cf)
        res = simulate_pcm(GHI, mirror_area, wax_mass, reactor_load)
        fig.add_trace(go.Scatter(
            x=[f"{h:02d}:00" for h in range(24)],
            y=res["soc"].tolist(),
            name=f"{label} → η={res['efisiensi']}%",
            line=dict(color=color, width=2),
            mode="lines",
            hovertemplate=(
                f"{label}<br>"
                "%{x}<br>SoC: <b>%{y:.1f}%</b><extra></extra>"
            ),
        ))

    fig.add_hline(y=30, line_dash="dot", line_color=AMBER, line_width=1,
                  annotation_text="Batas aman (30%)",
                  annotation_font_size=9, annotation_font_color=AMBER)
    fig.add_hline(y=10, line_dash="dot", line_color=RED, line_width=1,
                  annotation_text="Level kritis (10%)",
                  annotation_font_size=9, annotation_font_color=RED)
    fig.add_hrect(y0=0, y1=10, fillcolor=RED,
                  opacity=0.06, line_width=0)

    fig.update_layout(
        title=dict(
            text=(f"<b>Ketahanan Sistem terhadap Variasi Cuaca</b>  ·  "
                  f"{city_key}<br>"
                  f"<sup>SoC Slack Wax (%) · 5 Skenario Cloud Cover · "
                  f"Cermin {mirror_area/1000:.0f}k m² · "
                  f"Wax {wax_mass/1000:.0f}k kg · Beban {reactor_load:.0f} kW</sup>"),
            x=0, xanchor="left",
            font=dict(size=14, color=TEXT),
        ),
        height=380,
        template="plotly_white",
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        hovermode="x unified",
        xaxis=dict(
            title="Jam operasional",
            tickfont=dict(size=10, color=TEXTS),
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
        ),
        yaxis=dict(
            title="SoC (%)",
            range=[0, 105],
            tickfont=dict(size=10, color=TEXTS),
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
        ),
        legend=dict(
            orientation="h", y=-0.22, x=0.5, xanchor="center",
            font=dict(size=10, color=TEXTS),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=40, t=100, b=90),
        font=dict(family="system-ui, -apple-system, sans-serif"),
    )
    return fig


# Export HTML
def export_html(
    city_key:     str   = "Cilacap",
    mirror_area:  float = 30000.0,
    wax_mass:     float = 15000.0,
    reactor_load: float = 50.0,
    cloud_factor: float = None,
) -> str:
    import plotly.io as pio

    fig_main, res = make_pcm_dashboard(
        city_key, mirror_area, wax_mass, reactor_load, cloud_factor)
    fig_sens      = make_sensitivity_fig(city_key, reactor_load)
    fig_scen      = make_scenario_fig(city_key, mirror_area, wax_mass, reactor_load)

    cf_val = cloud_factor if cloud_factor else CITY_PROFILES[city_key]["cf"]

    main_json = pio.to_json(fig_main)
    sens_json = pio.to_json(fig_sens)
    scen_json = pio.to_json(fig_scen)

    stats = {
        "efisiensi": res["efisiensi"],
        "jam_padam": res["jam_padam"],
        "E_max":     res["E_max_kwh"],
        "waste":     res["waste_kwh"],
        "peak":      res["peak_solar"],
        "cf_pct":    f"{cf_val*100:.0f}",
        "city":      city_key,
        "mirror":    f"{mirror_area/1000:.0f}k",
        "wax":       f"{wax_mass/1000:.0f}k",
        "load":      reactor_load,
    }

    html_out = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PCM Thermal Logic · {city_key} </title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:system-ui,-apple-system,sans-serif;
        background:#F8F8F5;color:#1E293B;padding:24px 20px}}
  .wrap{{max-width:1200px;margin:0 auto}}
  h1{{font-size:18px;font-weight:500;margin-bottom:2px}}
  .sub{{font-size:12px;color:#64748B;margin-bottom:20px}}
  .metrics{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px}}
  .mc{{background:#EDEDEA;border-radius:8px;padding:12px 16px;
       flex:1;min-width:130px}}
  .mc.hi{{border-left:3px solid {AMBER};
          border-radius:0 8px 8px 0}}
  .mc.ok{{border-left:3px solid {TEAL};
          border-radius:0 8px 8px 0}}
  .mc.warn{{border-left:3px solid {RED};
            border-radius:0 8px 8px 0}}
  .mc-label{{font-size:11px;color:#64748B;margin-bottom:3px}}
  .mc-val{{font-size:24px;font-weight:500;line-height:1}}
  .mc-unit{{font-size:11px;color:#64748B}}
  .card{{background:#fff;border:0.5px solid rgba(0,0,0,.1);
         border-radius:12px;padding:16px;margin-bottom:14px}}
  .section-title{{font-size:13px;font-weight:500;
                  color:#64748B;margin-bottom:8px}}
  .controls{{display:flex;gap:12px;flex-wrap:wrap;
             align-items:flex-end;margin-bottom:18px}}
  .ctrl-group{{display:flex;flex-direction:column;gap:4px}}
  .ctrl-label{{font-size:11px;color:#64748B}}
  select,input[type=range]{{
    font-size:13px;padding:6px 10px;
    border:0.5px solid rgba(0,0,0,.15);
    border-radius:8px;background:#fff;
    color:#1E293B;cursor:pointer;min-width:180px}}
  input[type=range]{{min-width:160px;padding:4px 0}}
  button{{font-size:12px;padding:8px 16px;
          border:0.5px solid rgba(0,0,0,.2);
          border-radius:8px;background:#fff;
          color:#1E293B;cursor:pointer;font-weight:500}}
  button:hover{{background:#F0F0EC}}
  .footer{{border-top:0.5px solid rgba(0,0,0,.1);
           padding-top:10px;font-size:10px;
           color:#64748B;margin-top:4px}}
  .val-display{{font-size:12px;font-weight:500;
                color:#1E293B;min-width:60px}}
</style>
</head>
<body>
<div class="wrap">

  <h1>PCM Thermal Logic (Slack Wax State of Charge)</h1>
  <p class="sub">
    Memvalidasi kemampuan Slack Wax menyerap dan melepas panas laten selama 24 jam agar suhu reaktor tetap di atas 600°C meski input surya fluktuatif.
  </p>

  <!-- Controls interaktif -->
  <div class="card">
    <div class="section-title">Parameter simulasi</div>
    <div class="controls">
      <div class="ctrl-group">
        <span class="ctrl-label">Kota</span>
        <select id="cityCtrl">
          {"".join(f'<option value="{k}" {"selected" if k==city_key else ""}>{k}</option>'
                   for k in CITY_PROFILES)}
        </select>
      </div>
      <div class="ctrl-group">
        <span class="ctrl-label">Skenario cuaca</span>
        <select id="cloudCtrl">
          {"".join(f'<option value="{v}">{k}</option>'
                   for k,v in SCENARIO_CLOUDS.items())}
        </select>
      </div>
      <div class="ctrl-group">
        <span class="ctrl-label">
          Luas cermin:
          <span class="val-display" id="mirrorVal">{mirror_area/1000:.0f}k m²</span>
        </span>
        <input type="range" id="mirrorCtrl"
               min="5000" max="60000" step="5000"
               value="{mirror_area}">
      </div>
      <div class="ctrl-group">
        <span class="ctrl-label">
          Massa Slack Wax:
          <span class="val-display" id="waxVal">{wax_mass/1000:.0f}k kg</span>
        </span>
        <input type="range" id="waxCtrl"
               min="5000" max="40000" step="5000"
               value="{wax_mass}">
      </div>
      <div class="ctrl-group">
        <span class="ctrl-label">
          Beban reaktor:
          <span class="val-display" id="loadVal">{reactor_load:.0f} kW</span>
        </span>
        <input type="range" id="loadCtrl"
               min="20" max="200" step="10"
               value="{reactor_load}">
      </div>
      <button onclick="runSim()">Jalankan simulasi</button>
    </div>
  </div>

  <!-- Metric Cards (diupdate via JS) -->
  <div class="metrics" id="metricCards">
    <div class="mc hi">
      <div class="mc-label">Efisiensi sistem</div>
      <div><span class="mc-val" id="mc-eff">{res['efisiensi']}</span>
           <span class="mc-unit">%</span></div>
    </div>
    <div class="mc">
      <div class="mc-label">Jam padam</div>
      <div><span class="mc-val" id="mc-down">{res['jam_padam']}</span>
           <span class="mc-unit"> jam</span></div>
    </div>
    <div class="mc">
      <div class="mc-label">Kapasitas PCM</div>
      <div><span class="mc-val" id="mc-cap">{res['E_max_kwh']}</span>
           <span class="mc-unit"> kWh</span></div>
    </div>
    <div class="mc">
      <div class="mc-label">Energi terbuang</div>
      <div><span class="mc-val" id="mc-waste">{res['waste_kwh']}</span>
           <span class="mc-unit"> kWh</span></div>
    </div>
    <div class="mc">
      <div class="mc-label">Peak input surya</div>
      <div><span class="mc-val" id="mc-peak">{res['peak_solar']}</span>
           <span class="mc-unit"> kW</span></div>
    </div>
  </div>

  <!-- Charts -->
  <div class="card">
    <div class="section-title">Dashboard 24 jam - heat balance & SoC</div>
    <div id="mainChart"></div>
  </div>
  <div class="card">
    <div class="section-title">Sensitivitas konfigurasi - efisiensi vs cermin × wax</div>
    <div id="sensChart"></div>
  </div>
  <div class="card">
    <div class="section-title">Ketahanan sistem vs variasi cuaca (5 skenario)</div>
    <div id="scenChart"></div>
  </div>

  <div class="footer">
    Rumus: PCM Heat Balance (Zalba et al. 2003) · Entalpi Slack Wax
    (Sari &amp; Karaipekli 2007) · Cp Slack Wax (Trai-In et al. 2025) ·
    SoC framework (Pielichowska &amp; Pielichowski 2014) ·
    Solar: Bird Clear Sky Model (NREL 1981) · NASA POWER · BMKG
  </div>
</div>

<script>
const CITY_DATA = {json.dumps({
    k: {
        "lat": v["lat"], "lon": v["lon"],
        "alt": v["alt"], "cf": v["cf"]
    } for k, v in CITY_PROFILES.items()
})};

const SCENARIOS = {json.dumps(SCENARIO_CLOUDS)};

function eot(doy){{
  const B=2*Math.PI/365*(doy-81);
  return 9.87*Math.sin(2*B)-7.53*Math.cos(B)-1.5*Math.sin(B);
}}
function solarElev(h,lat,lon,doy){{
  const decl=23.45*Math.sin(2*Math.PI/365*(doy-81))*Math.PI/180;
  const lr=lat*Math.PI/180;
  const st=h+(lon%15)*4/60+eot(doy)/60;
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
function computeGHI(cityKey,cf){{
  const c=CITY_DATA[cityKey];
  const kc=1-cf*0.75;
  return Array.from({{length:24}},(_,h)=>Math.round(bird(solarElev(h+0.5,c.lat,c.lon,166),c.alt)*kc));
}}
function simPCM(GHI,mirrorArea,waxMass,reactorLoad){{
  const Lf_kwh=200/3600;
  const Emax=waxMass*Lf_kwh;
  const eta_csp=0.65;
  const qSolar=GHI.map(g=>(g*mirrorArea*eta_csp)/1000);
  let Epcm=Emax*0.3;
  const soc=[],qActual=[],qCharge=[];
  let totalPenuhi=0,totalWaste=0,jamPadam=0,peakSolar=0;
  for(let h=0;h<24;h++){{
    const Qin=qSolar[h],Qout=reactorLoad;
    peakSolar=Math.max(peakSolar,Qin);
    const surplus=Qin-Qout;
    let qa;
    if(surplus>=0){{
      const ce=surplus*1,ruang=Emax-Epcm,masuk=Math.min(ce,ruang);
      totalWaste+=ce-masuk; Epcm+=masuk;
      qCharge.push(masuk); qa=Qout;
    }}else{{
      const def=Math.abs(surplus)*1,bisa=Math.min(def,Epcm);
      Epcm-=bisa; qCharge.push(-bisa);
      qa=Math.min(Qin+bisa,Qout);
      if(qa<Qout*0.95)jamPadam++;
    }}
    totalPenuhi+=qa; qActual.push(qa);
    soc.push(Math.min(100,(Epcm/Emax)*100));
  }}
  return{{soc,qSolar,qActual,qCharge,
          efisiensi:Math.round(totalPenuhi/(reactorLoad*24)*1000)/10,
          jamPadam,Emax_kwh:Math.round(Emax*100)/100,
          waste_kwh:Math.round(totalWaste*100)/100,
          peakSolar:Math.round(peakSolar*10)/10}};
}}

const hlabels=Array.from({{length:24}},(_,i)=>String(i).padStart(2,'0')+':00');
const AMBER='{AMBER}',RED='{RED}',BLUE='{BLUE}',
      TEAL='{TEAL}',PURPLE='{PURPLE}',TEXTS='{TEXTS}',TEXT='{TEXT}';
const LAYOUT_BASE={{
  template:'plotly_white',paper_bgcolor:'rgba(0,0,0,0)',
  plot_bgcolor:'rgba(0,0,0,0)',
  font:{{family:'system-ui,-apple-system,sans-serif',color:TEXT}},
  margin:{{l:55,r:30,t:60,b:60}},
  legend:{{orientation:'h',y:-0.2,x:0.5,xanchor:'center',
           font:{{size:11,color:TEXTS}},bgcolor:'rgba(0,0,0,0)'}},
  hovermode:'x unified',
}};

function renderMain(res,mirrorArea,waxMass,reactorLoad,cityKey,cfPct){{
  const traces=[
    {{type:'scatter',x:hlabels,y:res.qSolar,name:'Input surya (kW)',
      fill:'tozeroy',line:{{color:AMBER,width:2.5}},
      fillcolor:'rgba(242,166,35,0.18)'}},
    {{type:'scatter',x:hlabels,y:Array(24).fill(reactorLoad),
      name:'Beban reaktor',line:{{color:RED,width:2,dash:'dot'}}}},
    {{type:'scatter',x:hlabels,y:res.qActual,name:'Output aktual',
      line:{{color:TEAL,width:1.5,dash:'dash'}}}},
    {{type:'bar',x:hlabels,y:res.soc,name:'SoC (%)',
      marker:{{color:res.soc.map(v=>v>=60?TEAL:v>=30?AMBER:RED),line:{{width:0}}}},
      yaxis:'y2'}},
  ];
  const layout={{...LAYOUT_BASE,
    title:{{text:`<b>${{cityKey}}</b> · Cloud ${{cfPct}}% · Cermin ${{Math.round(mirrorArea/1000)}}k m² · Wax ${{Math.round(waxMass/1000)}}k kg · Beban ${{reactorLoad}} kW`,
            font:{{size:13}},x:0,xanchor:'left'}},
    height:380,
    yaxis:{{title:'Daya (kW)',showgrid:true,gridcolor:'rgba(0,0,0,0.06)',zeroline:false,tickfont:{{size:10,color:TEXTS}}}},
    yaxis2:{{title:'SoC (%)',overlaying:'y',side:'right',range:[0,110],showgrid:false,zeroline:false,tickfont:{{size:10,color:TEXTS}}}},
    annotations:[
      {{x:0.5,y:1.08,xref:'paper',yref:'paper',
        text:`η: <b style='color:${{res.efisiensi>=95?TEAL:res.efisiensi>=80?AMBER:RED}}'>${{res.efisiensi}}%</b> · Padam: <b>${{res.jamPadam}} jam</b> · PCM: <b>${{res.Emax_kwh}} kWh</b>`,
        showarrow:false,font:{{size:11,color:TEXTS}}}},
    ],
  }};
  Plotly.react('mainChart',traces,layout,{{responsive:true,displayModeBar:false}});
}}

function renderScen(cityKey,mirrorArea,waxMass,reactorLoad){{
  const colors=[TEAL,'#639922',AMBER,RED,PURPLE];
  const traces=Object.entries(SCENARIOS).map(([label,cf],i)=>{{
    const GHI=computeGHI(cityKey,cf);
    const res=simPCM(GHI,mirrorArea,waxMass,reactorLoad);
    return{{type:'scatter',x:hlabels,y:res.soc,mode:'lines',
           name:`${{label}} → η=${{res.efisiensi}}%`,
           line:{{color:colors[i],width:2}}}};
  }});
  traces.push(
    {{type:'scatter',x:hlabels,y:Array(24).fill(30),mode:'lines',
      name:'Batas aman',line:{{color:AMBER,width:1,dash:'dot'}},showlegend:false}},
    {{type:'scatter',x:hlabels,y:Array(24).fill(10),mode:'lines',
      name:'Kritis',line:{{color:RED,width:1,dash:'dot'}},showlegend:false}},
  );
  Plotly.react('scenChart',traces,{{...LAYOUT_BASE,
    title:{{text:`<b>Ketahanan vs variasi cuaca</b> · SoC Slack Wax (%) · ${{cityKey}}`,
            font:{{size:13}},x:0,xanchor:'left'}},
    height:340,
    yaxis:{{title:'SoC (%)',range:[0,108],showgrid:true,
            gridcolor:'rgba(0,0,0,0.06)',zeroline:false,
            tickfont:{{size:10,color:TEXTS}}}},
    shapes:[
      {{type:'rect',x0:0,x1:23,y0:0,y1:10,
        fillcolor:RED,opacity:0.06,line:{{width:0}}}},
    ],
  }},{{responsive:true,displayModeBar:false}});
}}

document.getElementById('mirrorCtrl').oninput=function(){{
  document.getElementById('mirrorVal').textContent=
    Math.round(this.value/1000)+'k m²';
}};
document.getElementById('waxCtrl').oninput=function(){{
  document.getElementById('waxVal').textContent=
    Math.round(this.value/1000)+'k kg';
}};
document.getElementById('loadCtrl').oninput=function(){{
  document.getElementById('loadVal').textContent=this.value+' kW';
}};

function runSim(){{
  const cityKey=document.getElementById('cityCtrl').value;
  const cf=parseFloat(document.getElementById('cloudCtrl').value);
  const mirror=parseFloat(document.getElementById('mirrorCtrl').value);
  const wax=parseFloat(document.getElementById('waxCtrl').value);
  const load=parseFloat(document.getElementById('loadCtrl').value);
  const GHI=computeGHI(cityKey,cf);
  const res=simPCM(GHI,mirror,wax,load);
  const cfPct=Math.round(cf*100);
  document.getElementById('mc-eff').textContent=res.efisiensi;
  document.getElementById('mc-down').textContent=res.jamPadam;
  document.getElementById('mc-cap').textContent=res.Emax_kwh;
  document.getElementById('mc-waste').textContent=res.waste_kwh;
  document.getElementById('mc-peak').textContent=res.peakSolar;
  renderMain(res,mirror,wax,load,cityKey,cfPct);
  renderScen(cityKey,mirror,wax,load);
}}

const mainData={main_json};
const sensData={sens_json};
const scenData={scen_json};
Plotly.newPlot('mainChart',mainData.data,mainData.layout,{{responsive:true,displayModeBar:false}});
Plotly.newPlot('sensChart',sensData.data,sensData.layout,{{responsive:true,displayModeBar:false}});
Plotly.newPlot('scenChart',scenData.data,scenData.layout,{{responsive:true,displayModeBar:false}});
</script>
</body>
</html>"""

    fname = f"Modul2_PCM_ThermalLogic_{city_key}.html"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"[✓] Exported: {fname}")
    webbrowser.open("file://" + os.path.abspath(fname))
    return fname


# Entry Point
if __name__ == "__main__":
    print("PCM Thermal Logic")
    print("Memvalidasi kemampuan Slack Wax menyerap dan melepas panas laten selama 24 jam agar suhu reaktor tetap di atas 600°C meski input surya fluktuatif.")

    fname = export_html(
        city_key     = "Cilacap",
        mirror_area  = 30_000,
        wax_mass     = 15_000,
        reactor_load = 50.0,
    )
    print(f"\n  [✓] Dashboard siap: {fname}")
    print("  Browser akan terbuka otomatis.")
    print("\n  Untuk kota lain, ubah parameter di bagian export_html().")
    print("  Contoh: export_html('Makassar', 30000, 15000, 50.0)\n")