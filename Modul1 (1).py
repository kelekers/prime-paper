# MODUL 1 (SOLAR PROFILER)
# Memetakan fluktuasi radiasi matahari (GHI/DNI/DHI) per jam di 39 kota Indonesia untuk menentukan potensi input energi surya ke sistem CSP berdasarkan posisi geografis dan kondisi awan

import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback_context

CITIES = [
    {"name": "Banda Aceh",    "prov": "Aceh",              "lat":  5.55, "lon":  95.32, "alt":  15, "cf": [.68,.65,.62,.65,.68,.66,.62,.60,.62,.64,.66,.68]},
    {"name": "Medan",         "prov": "Sumatera Utara",    "lat":  3.59, "lon":  98.67, "alt":  25, "cf": [.65,.63,.68,.72,.74,.70,.65,.62,.64,.68,.70,.67]},
    {"name": "Padang",        "prov": "Sumatera Barat",    "lat": -0.95, "lon": 100.35, "alt":  10, "cf": [.78,.76,.73,.70,.66,.62,.60,.58,.60,.64,.70,.76]},
    {"name": "Pekanbaru",     "prov": "Riau",              "lat":  0.53, "lon": 101.45, "alt":  30, "cf": [.70,.68,.65,.68,.72,.68,.62,.60,.62,.66,.70,.70]},
    {"name": "Tanjung Pinang","prov": "Kepulauan Riau",    "lat":  0.92, "lon": 104.45, "alt":  10, "cf": [.68,.65,.62,.65,.68,.64,.60,.58,.60,.64,.66,.68]},
    {"name": "Jambi",         "prov": "Jambi",             "lat": -1.61, "lon": 103.62, "alt":  35, "cf": [.73,.71,.67,.63,.57,.51,.47,.45,.49,.57,.65,.71]},
    {"name": "Bengkulu",      "prov": "Bengkulu",          "lat": -3.80, "lon": 102.27, "alt":  15, "cf": [.76,.74,.70,.65,.60,.55,.52,.50,.52,.58,.66,.72]},
    {"name": "Palembang",     "prov": "Sumatera Selatan",  "lat": -2.99, "lon": 104.75, "alt":   8, "cf": [.75,.72,.68,.64,.58,.52,.48,.46,.50,.58,.66,.72]},
    {"name": "Pangkalpinang", "prov": "Bangka Belitung",   "lat": -2.13, "lon": 106.12, "alt":  25, "cf": [.72,.70,.66,.62,.56,.50,.46,.44,.48,.56,.64,.70]},
    {"name": "Serang",        "prov": "Banten",            "lat": -6.12, "lon": 106.15, "alt":  40, "cf": [.78,.76,.72,.68,.60,.52,.48,.46,.50,.60,.70,.76]},
    {"name": "Jakarta",       "prov": "DKI Jakarta",       "lat": -6.21, "lon": 106.85, "alt":   8, "cf": [.80,.78,.74,.70,.62,.54,.50,.48,.52,.62,.72,.78]},
    {"name": "Bandung",       "prov": "Jawa Barat",        "lat": -6.92, "lon": 107.61, "alt": 768, "cf": [.82,.80,.76,.72,.64,.56,.52,.50,.54,.64,.74,.80]},
    {"name": "Cilacap",       "prov": "Jawa Tengah",       "lat": -7.70, "lon": 109.01, "alt":   6, "cf": [.75,.73,.70,.66,.58,.50,.46,.44,.48,.58,.68,.74]},
    {"name": "Semarang",      "prov": "Jawa Tengah",       "lat": -6.97, "lon": 110.42, "alt":  10, "cf": [.78,.76,.72,.68,.60,.52,.48,.46,.50,.60,.70,.76]},
    {"name": "Yogyakarta",    "prov": "D.I. Yogyakarta",   "lat": -7.80, "lon": 110.37, "alt": 113, "cf": [.76,.74,.70,.64,.56,.48,.44,.42,.46,.56,.66,.74]},
    {"name": "Surabaya",      "prov": "Jawa Timur",        "lat": -7.25, "lon": 112.75, "alt":   5, "cf": [.72,.70,.66,.60,.52,.44,.40,.38,.42,.54,.64,.70]},
    {"name": "Denpasar",      "prov": "Bali",              "lat": -8.65, "lon": 115.22, "alt":  10, "cf": [.70,.68,.62,.54,.44,.36,.32,.30,.34,.46,.60,.68]},
    {"name": "Mataram",       "prov": "NTB",               "lat": -8.65, "lon": 116.12, "alt":  15, "cf": [.68,.66,.60,.52,.42,.34,.28,.26,.30,.40,.56,.65]},
    {"name": "Kupang",        "prov": "NTT",               "lat":-10.17, "lon": 123.61, "alt":  90, "cf": [.68,.65,.58,.46,.35,.28,.24,.22,.26,.36,.52,.65]},
    {"name": "Pontianak",     "prov": "Kalimantan Barat",  "lat": -0.02, "lon": 109.34, "alt":   5, "cf": [.74,.72,.70,.68,.65,.62,.60,.58,.60,.64,.68,.72]},
    {"name": "Banjarmasin",   "prov": "Kalimantan Selatan","lat": -3.32, "lon": 114.59, "alt":   5, "cf": [.72,.70,.66,.62,.56,.50,.46,.44,.48,.56,.64,.70]},
    {"name": "Samarinda",     "prov": "Kalimantan Timur",  "lat": -0.50, "lon": 117.15, "alt":  20, "cf": [.70,.68,.64,.60,.54,.48,.44,.42,.46,.54,.62,.68]},
    {"name": "Balikpapan",    "prov": "Kalimantan Timur",  "lat": -1.27, "lon": 116.83, "alt":  15, "cf": [.68,.66,.62,.58,.52,.46,.42,.40,.44,.52,.62,.68]},
    {"name": "Mamuju",        "prov": "Sulawesi Barat",    "lat": -2.68, "lon": 118.89, "alt":  30, "cf": [.70,.68,.62,.54,.46,.38,.32,.30,.34,.44,.58,.68]},
    {"name": "Palu",          "prov": "Sulawesi Tengah",   "lat": -0.90, "lon": 119.88, "alt":  25, "cf": [.62,.60,.56,.50,.44,.38,.32,.30,.34,.42,.54,.60]},
    {"name": "Makassar",      "prov": "Sulawesi Selatan",  "lat": -5.14, "lon": 119.43, "alt":  10, "cf": [.70,.68,.62,.54,.44,.36,.30,.28,.32,.42,.58,.68]},
    {"name": "Kendari",       "prov": "Sulawesi Tenggara", "lat": -3.94, "lon": 122.51, "alt":  20, "cf": [.72,.70,.65,.58,.50,.42,.36,.34,.38,.48,.60,.70]},
    {"name": "Gorontalo",     "prov": "Gorontalo",         "lat":  0.54, "lon": 123.06, "alt":  20, "cf": [.68,.66,.62,.58,.54,.50,.46,.44,.46,.52,.60,.66]},
    {"name": "Manado",        "prov": "Sulawesi Utara",    "lat":  1.49, "lon": 124.84, "alt":  30, "cf": [.72,.70,.65,.60,.58,.55,.52,.50,.52,.58,.64,.70]},
    {"name": "Sofifi",        "prov": "Maluku Utara",      "lat":  0.74, "lon": 127.56, "alt":  10, "cf": [.70,.68,.64,.60,.56,.52,.48,.46,.48,.54,.62,.68]},
    {"name": "Ternate",       "prov": "Maluku Utara",      "lat":  0.79, "lon": 127.37, "alt":  10, "cf": [.70,.68,.64,.60,.56,.52,.48,.46,.48,.54,.62,.68]},
    {"name": "Ambon",         "prov": "Maluku",            "lat": -3.70, "lon": 128.17, "alt":  15, "cf": [.76,.74,.70,.65,.58,.52,.48,.46,.50,.56,.64,.72]},
    {"name": "Sorong",        "prov": "Papua Barat",       "lat": -0.87, "lon": 131.25, "alt":  10, "cf": [.74,.72,.68,.64,.60,.56,.52,.50,.52,.58,.64,.72]},
    {"name": "Manokwari",     "prov": "Papua Barat",       "lat": -0.87, "lon": 134.08, "alt":  10, "cf": [.73,.71,.67,.63,.60,.57,.54,.52,.54,.58,.64,.70]},
    {"name": "Jayapura",      "prov": "Papua",             "lat": -2.53, "lon": 140.72, "alt":  40, "cf": [.72,.70,.68,.65,.62,.60,.58,.56,.58,.62,.66,.70]},
    {"name": "Merauke",       "prov": "Papua Selatan",     "lat": -8.49, "lon": 140.40, "alt":   5, "cf": [.70,.68,.65,.60,.55,.50,.46,.44,.48,.55,.63,.68]},
    {"name": "Nabire",        "prov": "Papua Tengah",      "lat": -3.37, "lon": 135.50, "alt":   8, "cf": [.74,.72,.70,.67,.64,.61,.58,.56,.58,.63,.68,.72]},
    {"name": "Wamena",        "prov": "Papua Pegunungan",  "lat": -4.08, "lon": 138.95, "alt":1672, "cf": [.76,.74,.71,.68,.65,.62,.59,.57,.60,.65,.70,.74]},
    {"name": "Sorong Selatan","prov": "Papua Barat Daya",  "lat": -1.21, "lon": 131.85, "alt":  10, "cf": [.74,.72,.68,.64,.60,.56,.52,.50,.52,.58,.64,.72]},
]

BULAN  = ["Januari","Februari","Maret","April","Mei","Juni",
          "Juli","Agustus","September","Oktober","November","Desember"]
DOY    = [15,46,74,105,135,166,196,227,258,288,319,349]

AMBER  = "#F2A623"
RED    = "#E24B4A"
BLUE   = "#378ADD"
TEAL   = "#1D9E75"
GRAY   = "#888780"
BG     = "#FFFFFF"
BG2    = "#F5F5F0"
BORDER = "rgba(0,0,0,0.10)"
TEXT   = "#1E293B"
TEXTS  = "#64748B"


def equation_of_time(doy: int) -> float:
    """Spencer (1971) — koreksi waktu matahari (menit)"""
    B = np.radians(360 / 365 * (doy - 81))
    return 9.87 * np.sin(2*B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)


def solar_elevation(hour: float, lat: float, lon: float, doy: int) -> float:
    """
    Sudut elevasi matahari (derajat).
    sin(α) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(ω)
    Ref: Duffie & Beckman (2013) Pers. 1.6.5
    """
    decl      = np.radians(23.45 * np.sin(np.radians(360 / 365 * (doy - 81))))
    lat_r     = np.radians(lat)
    eot       = equation_of_time(doy)
    solar_time = hour + (lon % 15) * 4 / 60 + eot / 60
    omega     = np.radians((solar_time - 12) * 15)
    sin_e     = (np.sin(lat_r) * np.sin(decl) +
                 np.cos(lat_r) * np.cos(decl) * np.cos(omega))
    return float(np.degrees(np.arcsin(np.clip(sin_e, -1, 1))))


def bird_clear_sky(elev_deg: float, altitude_m: float) -> tuple:
    """
    Bird Clear Sky Model — GHI, DNI, DHI (W/m²) pada langit cerah.
    Ref: Bird & Hulstrom (1981) SERI/TR-642-761
    Air mass: Kasten & Young (1989) Applied Optics 28(22)
    """
    if elev_deg <= 0:
        return 0.0, 0.0, 0.0
    elev_r = np.radians(elev_deg)
    cos_z  = np.sin(elev_r)
    AM     = min(1 / (cos_z + 0.50572 * (96.07995 - elev_deg) ** (-1.6364)), 38.0)
    af     = np.exp(-altitude_m / 8500)
    Tr     = np.clip(np.exp(-0.0903 * (AM * af) ** 0.84 * (1 + AM * af - (AM * af) ** 1.01)), 0.0, 1.0)
    Ta     = np.clip(np.exp(-0.0688 * (AM * af) ** 0.90 * 0.66), 0.0, 1.0)
    Tw     = np.clip(np.exp(-0.2700 * (0.04 * AM) ** 0.45), 0.0, 1.0)
    DNI    = max(0.0, 1353 * 0.9662 * Tr * Ta * Tw)
    DHI    = max(0.0, 1353 * cos_z * 0.95 * Tr ** 1.01 * 0.93 ** 0.69 * 0.12)
    GHI    = max(0.0, DNI * cos_z + DHI)
    return GHI, DNI, DHI


def compute_profile(city: dict, month_idx: int) -> dict:
    """
    Profil radiasi 24 jam dengan koreksi awan.
    Cloud correction: kc = 1 - cf * 0.75
    Ref: Liu & Jordan (1960) Solar Energy 4(3)
    """
    doy = DOY[month_idx]
    cf  = city["cf"][month_idx]
    kc  = 1 - cf * 0.75

    hours = np.arange(24)
    GHI, DNI, DHI = np.zeros(24), np.zeros(24), np.zeros(24)

    for i, h in enumerate(hours):
        el = solar_elevation(h + 0.5, city["lat"], city["lon"], doy)
        if el > 0:
            g, d, dh = bird_clear_sky(el, city["alt"])
            GHI[i] = round(g * kc)
            DNI[i] = round(d * kc * 0.90)
            DHI[i] = round(dh * kc + g * cf * 0.12)

    return {
        "GHI":       GHI,
        "DNI":       DNI,
        "DHI":       DHI,
        "peak_ghi":  int(GHI.max()),
        "peak_dni":  int(DNI.max()),
        "daily_kwh": round(float(GHI.sum()) / 1000, 3),
        "sun_hours": int((GHI > 50).sum()),
        "csp_hours": int((DNI > 500).sum()),
    }


def make_map_fig(city_idx: int, month_idx: int) -> go.Figure:
    """Scatter map Indonesia dengan warna GHI harian per kota."""
    lats, lons, names, colors, sizes, texts = [], [], [], [], [], []

    for i, c in enumerate(CITIES):
        p = compute_profile(c, month_idx)
        lats.append(c["lat"])
        lons.append(c["lon"])
        names.append(c["name"])
        colors.append(p["daily_kwh"])
        sizes.append(14 if i == city_idx else 9)
        texts.append(
            f"<b>{c['name']}</b><br>{c['prov']}<br>"
            f"GHI: {p['daily_kwh']} kWh/m²/hari<br>"
            f"Peak: {p['peak_ghi']} W/m²<br>"
            f"Jam surya: {p['sun_hours']} jam"
        )

    fig = go.Figure(go.Scattergeo(
        lat=lats, lon=lons,
        text=texts,
        hoverinfo="text",
        mode="markers",
        marker=dict(
            size=sizes,
            color=colors,
            colorscale=[
                [0.0,  "#185FA5"],
                [0.25, "#378ADD"],
                [0.5,  "#F2A623"],
                [0.75, "#D85A30"],
                [1.0,  "#A32D2D"],
            ],
            cmin=2.0, cmax=7.0,
            showscale=True,
            colorbar=dict(
                title=dict(text="kWh/m²/hari", font=dict(size=11, color=TEXTS)),
                thickness=12, len=0.7,
                tickfont=dict(size=10, color=TEXTS),
                outlinewidth=0,
            ),
            line=dict(width=1.5, color="white"),
            opacity=0.92,
        ),
    ))

    sel = CITIES[city_idx]
    fig.add_trace(go.Scattergeo(
        lat=[sel["lat"]], lon=[sel["lon"]],
        mode="markers",
        marker=dict(size=18, color=AMBER,
                    line=dict(width=2, color="white"), opacity=1),
        hoverinfo="skip",
        showlegend=False,
    ))

    fig.update_geos(
        visible=False,
        resolution=50,
        projection_type="mercator",
        lonaxis_range=[94, 142],
        lataxis_range=[-12, 7],
        showland=True,    landcolor="#E0DDD4",
        showocean=True,   oceancolor="#D6E8F5",
        showlakes=True,   lakecolor="#D6E8F5",
        showcoastlines=True, coastlinecolor="#AAAAAA",
        coastlinewidth=0.5,
        showframe=False,
        showcountries=True, countrycolor="#CCCCCC",
        countrywidth=0.4,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        height=320,
        showlegend=False,
    )
    return fig


def make_chart_fig(city_idx: int, month_idx: int) -> go.Figure:
    """Grafik GHI/DNI/DHI 24 jam dengan dual y-axis."""
    c = CITIES[city_idx]
    p = compute_profile(c, month_idx)
    hours = list(range(24))
    labels = [f"{h:02d}:00" for h in hours]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(
        x=labels, y=p["GHI"].tolist(),
        name="GHI (W/m²)", fill="tozeroy",
        line=dict(color=AMBER, width=2.5),
        fillcolor="rgba(242,166,35,0.18)",
        hovertemplate="%{x}<br>GHI: <b>%{y:.0f} W/m²</b><extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=labels, y=p["DNI"].tolist(),
        name="DNI (W/m²)",
        line=dict(color=RED, width=1.8, dash="dash"),
        hovertemplate="%{x}<br>DNI: <b>%{y:.0f} W/m²</b><extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=labels, y=p["DHI"].tolist(),
        name="DHI (W/m²)", fill="tozeroy",
        line=dict(color=BLUE, width=1.5),
        fillcolor="rgba(55,138,221,0.12)",
        hovertemplate="%{x}<br>DHI: <b>%{y:.0f} W/m²</b><extra></extra>",
    ), secondary_y=False)

    fig.add_hline(
        y=500, line_dash="dot", line_color=TEAL, line_width=1,
        annotation_text="CSP threshold (500 W/m²)",
        annotation_font_size=10, annotation_font_color=TEAL,
        annotation_position="bottom right",
    )

    lat_str = f"{abs(c['lat']):.2f}°{'S' if c['lat'] < 0 else 'N'}"
    fig.update_layout(
        title=dict(
            text=(f"<b>{c['name']}</b> · Estimasi Operasional 2026<br>"
                  f"<sup style='color:{TEXTS}'>"
                  f"{lat_str} · {c['lon']:.2f}°E · Alt {c['alt']} m · "
                  f"{BULAN[month_idx]} · Cloud cover {c['cf'][month_idx]*100:.0f}%"
                  f"</sup>"),
            x=0, xanchor="left",
            font=dict(size=15, color=TEXT),
        ),
        hovermode="x unified",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=20, t=80, b=110),
        height=360,
        legend=dict(
            orientation="h",
            y=-0.65,
            x=0.5, xanchor="center",
            font=dict(size=11, color=TEXTS),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            title=dict(text="Jam operasional", standoff=40),
            tickangle=-45, tickfont=dict(size=10, color=TEXTS),
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Irradiance (W/m²)",
            tickfont=dict(size=10, color=TEXTS),
            showgrid=True, gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
        ),
    )
    return fig


def make_monthly_fig(city_idx: int) -> go.Figure:
    """Bar chart GHI bulanan (perbandingan 12 bulan)."""
    c = CITIES[city_idx]
    daily_vals = []
    sun_hours  = []
    for mi in range(12):
        p = compute_profile(c, mi)
        daily_vals.append(p["daily_kwh"])
        sun_hours.append(p["sun_hours"])

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=BULAN, y=daily_vals,
        name="GHI harian (kWh/m²)",
        marker_color=[
            "#E24B4A" if v == max(daily_vals)
            else "#378ADD" if v == min(daily_vals)
            else "#F2A623"
            for v in daily_vals
        ],
        marker_line_width=0,
        hovertemplate="%{x}<br>GHI: <b>%{y:.3f} kWh/m²/hari</b><extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=BULAN, y=sun_hours,
        name="Jam surya efektif",
        mode="lines+markers",
        line=dict(color=TEAL, width=2),
        marker=dict(size=6, color=TEAL),
        hovertemplate="%{x}<br>Jam surya: <b>%{y} jam</b><extra></extra>",
    ), secondary_y=True)

    fig.update_layout(
        title=dict(
            text=f"<b>Proyeksi Variasi Bulanan 2026</b> — {c['name']}",
            x=0, xanchor="left",
            font=dict(size=13, color=TEXT),
        ),
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=50, r=50, t=55, b=60),
        height=250,
        hovermode="x unified",
        legend=dict(
            orientation="h", y=-0.25, x=0.5, xanchor="center",
            font=dict(size=10, color=TEXTS),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(tickfont=dict(size=10, color=TEXTS),
                   showgrid=False, zeroline=False),
        yaxis=dict(title="kWh/m²/hari",
                   tickfont=dict(size=10, color=TEXTS),
                   showgrid=True, gridcolor="rgba(0,0,0,0.06)",
                   zeroline=False),
        yaxis2=dict(title="Jam",
                    tickfont=dict(size=10, color=TEXTS),
                    showgrid=False, zeroline=False),
        bargap=0.25,
    )
    return fig


# HTML Views
def metric_cards(city_idx: int, month_idx: int):
    c = CITIES[city_idx]
    p = compute_profile(c, month_idx)
    card_style = {
        "background": BG2,
        "borderRadius": "8px",
        "padding": "14px 16px",
        "flex": "1",
        "minWidth": "140px",
        "overflow": "hidden",
        "boxSizing": "border-box",
    }
    hi_style = {**card_style,
                "borderLeft": f"3px solid {AMBER}",
                "borderRadius": "0 8px 8px 0"}
    label_s = {"fontSize": "11px", "color": TEXTS, "marginBottom": "6px",
               "whiteSpace": "nowrap"}
    val_s   = {"fontSize": "22px", "fontWeight": "500",
               "color": TEXT, "lineHeight": "1",
               "whiteSpace": "nowrap", "display": "block"}
    unit_s  = {"fontSize": "11px", "color": TEXTS,
               "marginTop": "4px", "display": "block"}

    def card(label, val, unit, style=card_style):
        return html.Div([
            html.Div(label, style=label_s),
            html.Span(str(val), style=val_s),
            html.Span(unit, style=unit_s),
        ], style=style)

    return html.Div([
        card("Peak GHI",          int(p["peak_ghi"]),              "W/m²",        hi_style),
        card("Peak DNI",          int(p["peak_dni"]),              "W/m²"),
        card("Daily total",       f"{float(p['daily_kwh']):.3f}",  "kWh/m²"),
        card("Jam surya efektif", int(p["sun_hours"]),             "jam"),
        card("Potensi CSP",       int(p["csp_hours"]),             "jam (DNI>500)"),
        card("Cloud cover",       f"{c['cf'][month_idx]*100:.0f}", "%"),
    ], style={
        "display": "flex", "gap": "8px",
        "flexWrap": "wrap", "marginBottom": "16px",
    })


# Dashboard
CITY_NAMES  = [c["name"] for c in CITIES]
INIT_CITY   = 12   # Cilacap (index 12)
INIT_MONTH  = 5    # Juni

app = dash.Dash(
    __name__,
    title="Solar Profiler Indonesia",
    meta_tags=[{"name": "viewport",
                "content": "width=device-width, initial-scale=1"}],
)

HEADER_STYLE = {
    "fontFamily": "system-ui, -apple-system, sans-serif",
    "borderBottom": f"1px solid {BORDER}",
    "paddingBottom": "14px",
    "marginBottom": "18px",
}
LABEL_STYLE  = {"fontSize": "12px", "color": TEXTS, "marginBottom": "4px"}
SELECT_STYLE = {
    "fontSize": "13px",
    "padding": "6px 10px",
    "border": f"0.5px solid {BORDER}",
    "borderRadius": "8px",
    "background": BG,
    "color": TEXT,
    "cursor": "pointer",
    "minWidth": "200px",
}

app.layout = html.Div([

    # Header
    html.Div([
        html.H1("Solar Profiler", style={
            "fontSize": "18px", "fontWeight": "500",
            "color": TEXT, "margin": "0 0 2px",
        }),
        html.P(
            "Memetakan fluktuasi radiasi matahari (GHI/DNI/DHI) per jam di "
            "39 kota Indonesia untuk menentukan potensi input energi surya ke "
            "sistem CSP berdasarkan posisi geografis dan kondisi awan",
            style={"fontSize": "12px", "color": TEXTS, "margin": "0"},
        ),
    ], style=HEADER_STYLE),

    # Controls
    html.Div([
        html.Div([
            html.Div("Kota", style=LABEL_STYLE),
            dcc.Dropdown(
                id="city-dd",
                options=[{"label": f"{c['name']} ({c['prov']})",
                          "value": i} for i, c in enumerate(CITIES)],
                value=INIT_CITY,
                clearable=False,
                style={**SELECT_STYLE, "minWidth": "260px"},
            ),
        ]),
        html.Div([
            html.Div("Bulan", style=LABEL_STYLE),
            dcc.Dropdown(
                id="month-dd",
                options=[{"label": b, "value": i}
                         for i, b in enumerate(BULAN)],
                value=INIT_MONTH,
                clearable=False,
                style=SELECT_STYLE,
            ),
        ]),
        html.Div(id="info-badge", style={
            "alignSelf": "flex-end",
            "fontSize": "11px",
            "padding": "6px 12px",
            "borderRadius": "99px",
            "background": "#FAEEDA",
            "color": "#633806",
            "fontWeight": "500",
        }),
    ], style={
        "display": "flex", "gap": "14px",
        "alignItems": "flex-start", "flexWrap": "wrap",
        "marginBottom": "18px",
    }),

    # Metric Cards
    html.Div(id="metrics"),

    # Peta
    html.Div([
        html.P("Klik titik pada peta untuk memuat detail kota",
               style={"fontSize": "11px", "color": TEXTS, "margin": "0 0 6px"}),
        dcc.Graph(id="map-fig",
                  config={"displayModeBar": False},
                  style={"borderRadius": "12px",
                         "border": f"0.5px solid {BORDER}",
                         "overflow": "hidden"}),
    ], style={"marginBottom": "16px"}),

    # Chart 24 jam
    html.Div([
        dcc.Graph(id="chart-fig",
                  config={"displayModeBar": False, "displaylogo": False}),
    ], style={
        "background": BG,
        "border": f"0.5px solid {BORDER}",
        "borderRadius": "12px",
        "padding": "16px",
        "marginBottom": "16px",
    }),

    # Chart bulanan
    html.Div([
        dcc.Graph(id="monthly-fig",
                  config={"displayModeBar": False}),
    ], style={
        "background": BG,
        "border": f"0.5px solid {BORDER}",
        "borderRadius": "12px",
        "padding": "16px",
        "marginBottom": "16px",
    }),

    # Footer
    html.Div([
        html.P([
            "Rumus: Bird Clear Sky Model (NREL, 1981) · "
            "Solar Position Algorithm (Reda & Andreas, 2004) · "
            "Cloud correction: Liu & Jordan (1960) · "
            "Data: NASA POWER · Global Solar Atlas 2.0 (World Bank) · BMKG",
        ], style={"fontSize": "10px", "color": TEXTS, "margin": "0"}),
    ], style={
        "borderTop": f"0.5px solid {BORDER}",
        "paddingTop": "10px",
    }),

], style={
    "fontFamily": "system-ui, -apple-system, sans-serif",
    "maxWidth": "1100px",
    "margin": "0 auto",
    "padding": "24px 20px",
    "background": BG,
    "color": TEXT,
})


# Callbacks

@app.callback(
    Output("map-fig",     "figure"),
    Output("chart-fig",   "figure"),
    Output("monthly-fig", "figure"),
    Output("metrics",     "children"),
    Output("info-badge",  "children"),
    Output("city-dd",     "value"),
    Input("city-dd",      "value"),
    Input("month-dd",     "value"),
    Input("map-fig",      "clickData"),
)
def update(city_idx, month_idx, click_data):
    ctx = callback_context

    # gunakan 'is None' agar index 0 (Banda Aceh) tetap valid
    if city_idx is None:
        city_idx = INIT_CITY
    if month_idx is None:
        month_idx = INIT_MONTH

    # Override dari klik peta
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "map-fig.clickData":
        if click_data and click_data.get("points"):
            pt = click_data["points"][0]
            pt_lat = pt.get("lat")
            pt_lon = pt.get("lon")
            if pt_lat is not None and pt_lon is not None:
                for i, c in enumerate(CITIES):
                    if (abs(c["lat"] - pt_lat) < 0.05 and
                            abs(c["lon"] - pt_lon) < 0.05):
                        city_idx = i
                        break

    c = CITIES[city_idx]
    lat_str = f"{abs(c['lat']):.2f}°{'S' if c['lat'] < 0 else 'N'}"
    badge   = f"{c['name']} · {lat_str} · Alt {c['alt']} m"

    return (
        make_map_fig(city_idx, month_idx),
        make_chart_fig(city_idx, month_idx),
        make_monthly_fig(city_idx),
        metric_cards(city_idx, month_idx),
        badge,
        city_idx,
    )


# Export HTML

def export_standalone(city_idx=INIT_CITY, month_idx=INIT_MONTH):
    import plotly.io as pio
    c  = CITIES[city_idx]
    p  = compute_profile(c, month_idx)

    fig_map     = make_map_fig(city_idx, month_idx)
    fig_chart   = make_chart_fig(city_idx, month_idx)
    fig_monthly = make_monthly_fig(city_idx)

    map_html     = pio.to_html(fig_map,     full_html=False, include_plotlyjs=False)
    chart_html   = pio.to_html(fig_chart,   full_html=False, include_plotlyjs=False)
    monthly_html = pio.to_html(fig_monthly, full_html=False, include_plotlyjs=False)

    lat_str = f"{abs(c['lat']):.2f}°{'S' if c['lat'] < 0 else 'N'}"

    html_out = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Solar Profiler · {c['name']}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:system-ui,-apple-system,sans-serif;
        background:#f8f8f5;color:#1E293B;padding:24px 20px}}
  .wrap{{max-width:1100px;margin:0 auto}}
  h1{{font-size:18px;font-weight:500;margin-bottom:2px}}
  .sub{{font-size:12px;color:#64748B;margin-bottom:20px}}
  .metrics{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}}
  .mc{{background:#F5F5F0;border-radius:8px;padding:12px 16px;flex:1;min-width:130px}}
  .mc.hi{{border-left:3px solid {AMBER};border-radius:0 8px 8px 0}}
  .mc-label{{font-size:11px;color:#64748B;margin-bottom:3px}}
  .mc-val{{font-size:24px;font-weight:500;line-height:1}}
  .mc-unit{{font-size:11px;color:#64748B}}
  .card{{background:#fff;border:0.5px solid rgba(0,0,0,.1);
         border-radius:12px;padding:16px;margin-bottom:14px}}
  .hint{{font-size:11px;color:#64748B;margin-bottom:6px}}
  .footer{{border-top:0.5px solid rgba(0,0,0,.1);padding-top:10px;
           font-size:10px;color:#64748B;margin-top:4px}}
</style>
</head>
<body>
<div class="wrap">
  <h1>Modul 1 — Solar Profiler · {c['name']}</h1>
  <p class="sub">
    {c['prov']} · {lat_str} · {c['lon']:.2f}°E · Alt {c['alt']} m ·
    {BULAN[month_idx]} · Cloud cover {c['cf'][month_idx]*100:.0f}%
  </p>

  <div class="metrics">
    <div class="mc hi">
      <div class="mc-label">Peak GHI</div>
      <div><span class="mc-val">{p['peak_ghi']}</span>
           <span class="mc-unit"> W/m²</span></div>
    </div>
    <div class="mc">
      <div class="mc-label">Peak DNI</div>
      <div><span class="mc-val">{p['peak_dni']}</span>
           <span class="mc-unit"> W/m²</span></div>
    </div>
    <div class="mc">
      <div class="mc-label">Daily total</div>
      <div><span class="mc-val">{p['daily_kwh']}</span>
           <span class="mc-unit"> kWh/m²</span></div>
    </div>
    <div class="mc">
      <div class="mc-label">Jam surya efektif</div>
      <div><span class="mc-val">{p['sun_hours']}</span>
           <span class="mc-unit"> jam</span></div>
    </div>
    <div class="mc">
      <div class="mc-label">Potensi CSP</div>
      <div><span class="mc-val">{p['csp_hours']}</span>
           <span class="mc-unit"> jam DNI&gt;500</span></div>
    </div>
  </div>

  <div class="card">
    <p class="hint">Peta distribusi GHI nasional — {BULAN[month_idx]}</p>
    {map_html}
  </div>
  <div class="card">{chart_html}</div>
  <div class="card">{monthly_html}</div>

  <div class="footer">
    Rumus: Bird Clear Sky Model (NREL, 1981) ·
    Solar Position Algorithm (Reda &amp; Andreas, 2004) ·
    Cloud correction: Liu &amp; Jordan (1960) ·
    Data: NASA POWER · Global Solar Atlas 2.0 (World Bank) · BMKG
  </div>
</div>
</body>
</html>"""

    fname = f"Solar_Profiler_{c['name'].replace(' ','_')}_{BULAN[month_idx]}.html"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(html_out)
    print(f"[✓] Exported: {fname}")
    return fname


# Entry Point

if __name__ == "__main__":
    if "--export" in sys.argv:
        # Export 5 kota: Cilacap, Makassar, Yogyakarta, Surabaya, Mataram
        top_cities = [12, 25, 14, 15, 17]
        for ci in top_cities:
            export_standalone(city_idx=ci, month_idx=5)
        print("\n[✓] Semua file HTML siap dibawa presentasi.")
    else:
        print("\n  Solar Profiler — Modul 1")
        print("  Dashboard: http://localhost:5000")
        print("  Export HTML: python modul1_solar_profiler.py --export\n")
        app.run(debug=False, host="localhost", port=5000)