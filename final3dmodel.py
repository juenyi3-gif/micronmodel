# integrated_3d_fvm_reliability_app.py

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import factorized


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Micron HBM Packaging + Thermal Reliability Model",
    layout="wide",
    page_icon="🔥"
)


# ============================================================
# PACKAGING CONSTANTS
# ============================================================

MSL_DB = {
    "1":  {"limit": 999999, "label": "Unlimited"},
    "2":  {"limit": 8760,   "label": "1 Year"},
    "2a": {"limit": 672,    "label": "4 Weeks"},
    "3":  {"limit": 168,    "label": "7 Days"},
    "4":  {"limit": 72,     "label": "72 Hours"},
    "5":  {"limit": 48,     "label": "48 Hours"},
    "5a": {"limit": 24,     "label": "24 Hours"},
    "6":  {"limit": 0,      "label": "Mandatory Bake"}
}

EA = 0.35
K_BOLTZMANN = 8.617e-5


# ============================================================
# THERMAL MATERIAL CONSTANTS
# Approximate values based on package thermal literature.
# ============================================================

K_SI = 148.0
K_EPOXY = 0.77
K_SUBSTRATE = 0.30
K_SOLDER = 54.0
K_CU = 401.0

RHO_SI = 2330.0
RHO_EPOXY = 1200.0
RHO_SUBSTRATE = 1850.0
RHO_SOLDER = 7300.0
RHO_CU = 8960.0

CP_SI = 705.0
CP_EPOXY = 800.0
CP_SUBSTRATE = 440.0
CP_SOLDER = 230.0
CP_CU = 385.0

T_GLASS = 175.0
WARNING_T = 110.0


# ============================================================
# PACKAGING MODEL
# ============================================================

def run_diffusion_model(temp_c, rh_pct, time_hr, thickness_mm, d0_cm2s):
    """
    Simple Fickian moisture uptake screening model.
    """
    temp_k = temp_c + 273.15
    d0_m2s = d0_cm2s * 1e-4

    d_m2s = d0_m2s * np.exp(-EA / (K_BOLTZMANN * temp_k))

    half_thickness_m = (thickness_mm / 2.0) / 1000.0
    time_s = time_hr * 3600.0

    tau = d_m2s * time_s / (half_thickness_m ** 2)

    c_surface = (rh_pct / 85.0) * 0.4
    c_internal = c_surface * (1.0 - np.exp(-tau))

    d_cm2s = d_m2s * 1e4

    return d_cm2s, c_surface, c_internal


def evaluate_packaging(temp_c, rh_pct, exposure_hr, thickness_mm, d0_cm2s,
                       msl_rating, glass_stoppage, reflow_count):
    d_coeff, c_surface, c_internal = run_diffusion_model(
        temp_c, rh_pct, exposure_hr, thickness_mm, d0_cm2s
    )

    moisture_limit = 0.07 if glass_stoppage else 0.15

    floor_life_limit = MSL_DB[msl_rating]["limit"]
    usage_ratio = exposure_hr / floor_life_limit if floor_life_limit > 0 else 1.0

    reflow_fail = reflow_count > 3
    msl_fail = msl_rating == "6"
    floor_life_fail = usage_ratio >= 1.0
    moisture_fail = c_internal >= moisture_limit

    if reflow_fail or msl_fail or floor_life_fail or moisture_fail:
        status = "FAILURE"
    elif usage_ratio > 0.9 or c_internal > 0.8 * moisture_limit:
        status = "WARNING"
    else:
        status = "SAFE"

    reasons = []

    if reflow_fail:
        reasons.append("Reflow count exceeds 3.")
    if msl_fail:
        reasons.append("MSL 6 requires mandatory bake.")
    if floor_life_fail:
        reasons.append("Floor life has expired.")
    if moisture_fail:
        reasons.append("Internal moisture exceeds threshold.")
    if not reasons:
        reasons.append("Packaging condition is within screening limits.")

    return {
        "status": status,
        "reason": " ".join(reasons),
        "d_coeff": d_coeff,
        "c_surface": c_surface,
        "c_internal": c_internal,
        "moisture_limit": moisture_limit,
        "usage_ratio": usage_ratio,
        "floor_life_limit": floor_life_limit
    }


# ============================================================
# 3D FVM THERMAL MODEL
# ============================================================

def harmonic_mean(a, b):
    if a + b <= 0:
        return 0.0
    return 2.0 * a * b / (a + b)


def build_layered_material_grid(nx, ny, nz, rh_pct, c_internal):
    """
    Creates 3D material-property fields.

    z direction:
      bottom -> top

    This is a simplified stacked package:
      substrate
      solder / microbump
      die 1
      underfill
      die 2
      underfill
      die 3
      underfill
      die 4
    """

    k = np.zeros((nx, ny, nz))
    rho = np.zeros((nx, ny, nz))
    cp = np.zeros((nx, ny, nz))
    mat = np.empty((nx, ny, nz), dtype=object)

    # Moisture degradation of epoxy conductivity.
    # This is a screening assumption, not a calibrated material law.
    moisture_ratio = np.clip(c_internal / 0.15, 0.0, 1.5)
    humidity_ratio = np.clip(rh_pct / 85.0, 0.0, 1.2)

    epoxy_degradation = 0.35 * humidity_ratio + 0.15 * moisture_ratio
    epoxy_degradation = np.clip(epoxy_degradation, 0.0, 0.50)

    k_epoxy_eff = max(K_EPOXY * (1.0 - epoxy_degradation), 0.35)

    # Assign layer boundaries by z index.
    for z in range(nz):
        frac = z / max(nz - 1, 1)

        if frac < 0.20:
            material = "Substrate"
            kk, rr, cc = K_SUBSTRATE, RHO_SUBSTRATE, CP_SUBSTRATE

        elif frac < 0.27:
            material = "Solder / Microbump"
            kk, rr, cc = K_SOLDER, RHO_SOLDER, CP_SOLDER

        elif frac < 0.43:
            material = "Silicon Die 1"
            kk, rr, cc = K_SI, RHO_SI, CP_SI

        elif frac < 0.50:
            material = "Underfill 1"
            kk, rr, cc = k_epoxy_eff, RHO_EPOXY, CP_EPOXY

        elif frac < 0.63:
            material = "Silicon Die 2"
            kk, rr, cc = K_SI, RHO_SI, CP_SI

        elif frac < 0.70:
            material = "Underfill 2"
            kk, rr, cc = k_epoxy_eff, RHO_EPOXY, CP_EPOXY

        elif frac < 0.83:
            material = "Silicon Die 3"
            kk, rr, cc = K_SI, RHO_SI, CP_SI

        elif frac < 0.90:
            material = "Underfill 3"
            kk, rr, cc = k_epoxy_eff, RHO_EPOXY, CP_EPOXY

        else:
            material = "Silicon Die 4"
            kk, rr, cc = K_SI, RHO_SI, CP_SI

        k[:, :, z] = kk
        rho[:, :, z] = rr
        cp[:, :, z] = cc
        mat[:, :, z] = material

    return k, rho, cp, mat, k_epoxy_eff


def reflow_air_temperature(t, ambient_c, peak_c, heating_time_s):
    """
    Simplified reflow profile:
      0%   ambient
      35%  ramp to 150 C
      65%  soak to 180 C
      85%  ramp to peak
      100% cool toward 100 C

    This is not a JEDEC-certified profile. It is a practical simulation input.
    """
    total = heating_time_s
    x = t / total

    if x <= 0.35:
        return ambient_c + (150.0 - ambient_c) * (x / 0.35)

    elif x <= 0.65:
        return 150.0 + (170.0 - 150.0) * ((x - 0.35) / 0.30)

    elif x <= 0.85:
        return 170.0 + (peak_c - 170.0) * ((x - 0.65) / 0.20)

    else:
        return peak_c + (50.0 - peak_c) * ((x - 0.85) / 0.15)


def idx(i, j, k, nx, ny, nz):
    return (k * nx * ny) + (j * nx) + i


def build_implicit_fvm_matrix(nx, ny, nz, dx, dy, dz,
                              k_field, rho_field, cp_field,
                              dt, h_conv):
    """
    Builds the backward-Euler 3D FVM matrix.

    Equation:
      rho Cp V (T_new - T_old) / dt
      =
      sum_faces G_face (T_neighbor_new - T_cell_new)
      + Q V
      + boundary convection

    Rearranged:
      A T_new = b
    """

    n_total = nx * ny * nz
    A = lil_matrix((n_total, n_total))

    volume = dx * dy * dz

    ax = dy * dz
    ay = dx * dz
    az = dx * dy

    for kk in range(nz):
        for jj in range(ny):
            for ii in range(nx):
                p = idx(ii, jj, kk, nx, ny, nz)

                rho_cp_v = rho_field[ii, jj, kk] * cp_field[ii, jj, kk] * volume
                diag = rho_cp_v / dt

                k_cell = k_field[ii, jj, kk]

                # X minus
                if ii > 0:
                    k_face = harmonic_mean(k_cell, k_field[ii - 1, jj, kk])
                    g = k_face * ax / dx
                    diag += g
                    A[p, idx(ii - 1, jj, kk, nx, ny, nz)] = -g
                else:
                    hA = h_conv * ax
                    diag += hA

                # X plus
                if ii < nx - 1:
                    k_face = harmonic_mean(k_cell, k_field[ii + 1, jj, kk])
                    g = k_face * ax / dx
                    diag += g
                    A[p, idx(ii + 1, jj, kk, nx, ny, nz)] = -g
                else:
                    hA = h_conv * ax
                    diag += hA

                # Y minus
                if jj > 0:
                    k_face = harmonic_mean(k_cell, k_field[ii, jj - 1, kk])
                    g = k_face * ay / dy
                    diag += g
                    A[p, idx(ii, jj - 1, kk, nx, ny, nz)] = -g
                else:
                    hA = h_conv * ay
                    diag += hA

                # Y plus
                if jj < ny - 1:
                    k_face = harmonic_mean(k_cell, k_field[ii, jj + 1, kk])
                    g = k_face * ay / dy
                    diag += g
                    A[p, idx(ii, jj + 1, kk, nx, ny, nz)] = -g
                else:
                    hA = h_conv * ay
                    diag += hA

                # Z minus
                if kk > 0:
                    k_face = harmonic_mean(k_cell, k_field[ii, jj, kk - 1])
                    g = k_face * az / dz
                    diag += g
                    A[p, idx(ii, jj, kk - 1, nx, ny, nz)] = -g
                else:
                    hA = h_conv * az
                    diag += hA

                # Z plus
                if kk < nz - 1:
                    k_face = harmonic_mean(k_cell, k_field[ii, jj, kk + 1])
                    g = k_face * az / dz
                    diag += g
                    A[p, idx(ii, jj, kk + 1, nx, ny, nz)] = -g
                else:
                    hA = h_conv * az
                    diag += hA

                A[p, p] = diag

    return csr_matrix(A)


def build_heat_source(nx, ny, nz, dx, dy, dz, mat_field, power_w):
    """
    Localized 3D hotspot in the silicon die region.
    Power is distributed into a small NPU-like block.
    """
    qv = np.zeros((nx, ny, nz))

    # Hotspot dimensions in grid cells
    cx0 = int(nx * 0.45)
    cx1 = int(nx * 0.65)

    cy0 = int(ny * 0.40)
    cy1 = int(ny * 0.60)

    target_cells = []

    for kk in range(nz):
        for jj in range(cy0, cy1):
            for ii in range(cx0, cx1):
                if "Silicon" in mat_field[ii, jj, kk]:
                    # Place heat mainly in upper/middle silicon layers
                    if kk > int(nz * 0.35):
                        target_cells.append((ii, jj, kk))

    if len(target_cells) == 0:
        return qv

    volume = dx * dy * dz
    q_per_volume = power_w / (len(target_cells) * volume)

    for ii, jj, kk in target_cells:
        qv[ii, jj, kk] = q_per_volume

    return qv


def run_3d_transient_fvm(
    ambient_c,
    rh_pct,
    c_internal,
    thickness_mm,
    power_w,
    reflow_peak_c,
    heating_time_s,
    h_conv,
    nx=22,
    ny=22,
    nz=14,
    n_steps=80
):
    """
    Full 3D transient structured-grid FVM thermal solver.

    Outputs:
      T_final
      T_max_history
      T_underfill_history
      time_history
      air_temp_history
      material field
      effective epoxy k
    """

    # Package footprint. Keep moderate for Streamlit speed.
    length_m = 12.7e-3
    width_m = 12.7e-3
    height_m = max(thickness_mm * 1e-3, 0.5e-3)

    dx = length_m / nx
    dy = width_m / ny
    dz = height_m / nz

    dt = heating_time_s / n_steps

    k_field, rho_field, cp_field, mat_field, k_epoxy_eff = build_layered_material_grid(
        nx, ny, nz, rh_pct, c_internal
    )

    qv = build_heat_source(nx, ny, nz, dx, dy, dz, mat_field, power_w)

    A = build_implicit_fvm_matrix(
        nx, ny, nz, dx, dy, dz,
        k_field, rho_field, cp_field,
        dt, h_conv
    )

    solve_A = factorized(A)

    volume = dx * dy * dz

    T = np.ones((nx, ny, nz)) * ambient_c

    T_max_history = []
    T_underfill_history = []
    time_history = []
    air_temp_history = []

    # Boundary face areas
    ax = dy * dz
    ay = dx * dz
    az = dx * dy

    for step in range(n_steps):
        t = (step + 1) * dt
        t_air = reflow_air_temperature(t, ambient_c, reflow_peak_c, heating_time_s)

        b = np.zeros(nx * ny * nz)

        for kk in range(nz):
            for jj in range(ny):
                for ii in range(nx):
                    p = idx(ii, jj, kk, nx, ny, nz)

                    rho_cp_v = rho_field[ii, jj, kk] * cp_field[ii, jj, kk] * volume

                    rhs = rho_cp_v / dt * T[ii, jj, kk]
                    rhs += qv[ii, jj, kk] * volume

                    # Boundary convection RHS terms: h A T_air
                    if ii == 0:
                        rhs += h_conv * ax * t_air
                    if ii == nx - 1:
                        rhs += h_conv * ax * t_air
                    if jj == 0:
                        rhs += h_conv * ay * t_air
                    if jj == ny - 1:
                        rhs += h_conv * ay * t_air
                    if kk == 0:
                        rhs += h_conv * az * t_air
                    if kk == nz - 1:
                        rhs += h_conv * az * t_air

                    b[p] = rhs

        T_vec = solve_A(b)
        T = T_vec.reshape((nz, ny, nx)).transpose(2, 1, 0)

        underfill_mask = np.char.find(mat_field.astype(str), "Underfill") >= 0

        T_max_history.append(float(np.max(T)))
        T_underfill_history.append(float(np.max(T[underfill_mask])))
        time_history.append(t)
        air_temp_history.append(t_air)

    result = {
        "T_final": T,
        "T_max_history": np.array(T_max_history),
        "T_underfill_history": np.array(T_underfill_history),
        "time_history": np.array(time_history),
        "air_temp_history": np.array(air_temp_history),
        "mat_field": mat_field,
        "k_epoxy_eff": k_epoxy_eff,
        "grid": {
            "nx": nx,
            "ny": ny,
            "nz": nz,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "length_m": length_m,
            "width_m": width_m,
            "height_m": height_m
        }
    }

    return result


def evaluate_thermal_fvm(fvm_result):
    T_underfill = fvm_result["T_underfill_history"]
    T_max = fvm_result["T_max_history"]
    time = fvm_result["time_history"]

    dt = time[1] - time[0] if len(time) > 1 else 0.0

    max_stack = float(np.max(T_max))
    max_underfill = float(np.max(T_underfill))

    time_above_tg = float(np.sum(T_underfill >= T_GLASS) * dt)
    time_above_warning = float(np.sum(T_underfill >= WARNING_T) * dt)

    if max_stack > 260.0:
        status = "CRITICAL"
        reason = f"Peak package temperature ({max_stack:.1f}°C) exceeds absolute JEDEC reflow limit (260°C)."
    elif time_above_tg > 60.0:
        status = "CRITICAL"
        reason = f"Time above Tg ({time_above_tg:.1f}s) exceeds absolute JEDEC limit (60s)."
    elif time_above_tg >= 30.0:
        status = "WARNING"
        reason = f"Time above Tg ({time_above_tg:.1f}s) is in the warning zone (30-60s)."
    else:
        status = "STABLE"
        reason = "Thermal reflow profile remains within safe processing limits."

    return {
        "status": status,
        "reason": reason,
        "max_stack": max_stack,
        "max_underfill": max_underfill,
        "time_above_tg": time_above_tg,
        "time_above_warning": time_above_warning
    }


# ============================================================
# PLOTS
# ============================================================

def plot_3d_temperature_volume(fvm_result):
    T = fvm_result["T_final"]
    grid = fvm_result["grid"]

    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    dx, dy, dz = grid["dx"], grid["dy"], grid["dz"]

    X, Y, Z = np.meshgrid(
        np.arange(nx) * dx * 1e3,
        np.arange(ny) * dy * 1e3,
        np.arange(nz) * dz * 1e6,
        indexing="ij"
    )

    fig = go.Figure(data=go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=T.flatten(),
        isomin=float(np.min(T)), isomax=float(np.max(T)),
        opacity=0.25, surface_count=21, colorscale="Turbo",
        colorbar=dict(
            title=dict(text="Temp (°C)", font=dict(color="white")), 
            tickfont=dict(color="white")
        ),
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.5, roughness=0.1)
    ))
    
    fig.update_layout(
        title=dict(text="3D Thermal Distribution — FVM Final Temperature", font=dict(color="white", size=18, family="Inter, sans-serif")),
        scene=dict(
            xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Height (µm)",
            bgcolor="#0f172a",
            xaxis=dict(color="white", gridcolor="#334155", zerolinecolor="#334155", backgroundcolor="#0f172a", showbackground=True),
            yaxis=dict(color="white", gridcolor="#334155", zerolinecolor="#334155", backgroundcolor="#0f172a", showbackground=True),
            zaxis=dict(color="white", gridcolor="#334155", zerolinecolor="#334155", backgroundcolor="#0f172a", showbackground=True),
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.8))
        ),
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font=dict(color="white", family="Inter, sans-serif"),
        height=550, margin=dict(l=0, r=0, t=60, b=0),
    )

    return fig


def plot_vertical_temperature_profile(fvm_result):
    T = fvm_result["T_final"]
    grid = fvm_result["grid"]
    dz = grid["dz"]
    nz = grid["nz"]

    z_um = np.arange(nz) * dz * 1e6

    avg_profile = np.mean(T, axis=(0, 1))
    max_profile = np.max(T, axis=(0, 1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=avg_profile,
        y=z_um,
        mode="lines+markers",
        name="Average Temp",
        line=dict(color="#0ea5e9", width=3),
        marker=dict(size=6, color="#0ea5e9", line=dict(color="white", width=1))
    ))

    fig.add_trace(go.Scatter(
        x=max_profile,
        y=z_um,
        mode="lines+markers",
        name="Max Temp",
        line=dict(color="#f43f5e", width=3),
        marker=dict(size=6, color="#f43f5e", line=dict(color="white", width=1))
    ))

    fig.add_vline(
        x=T_GLASS,
        line_dash="dash",
        line_color="#fbbf24",
        annotation_text=f"Tg {T_GLASS:.0f}°C",
        annotation_font_color="#fbbf24"
    )

    fig.update_layout(
        title=dict(text="Temperature Profile Through Package Thickness", font=dict(color="white", size=16, family="Inter, sans-serif")),
        xaxis_title="Temperature (°C)",
        yaxis_title="Height (µm)",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="white", family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
        height=480,
        margin=dict(l=20, r=20, t=50, b=40)
    )

    return fig


def plot_reflow_history(fvm_result):
    time_min = fvm_result["time_history"] / 60.0

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=time_min,
        y=fvm_result["air_temp_history"],
        mode="lines",
        name="Oven Air Temp",
        line=dict(color="#94a3b8", width=2, dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=time_min,
        y=fvm_result["T_max_history"],
        mode="lines",
        name="Max Package Temp",
        line=dict(color="#f43f5e", width=3),
        fill='tozeroy', fillcolor='rgba(244, 63, 94, 0.1)'
    ))

    fig.add_trace(go.Scatter(
        x=time_min,
        y=fvm_result["T_underfill_history"],
        mode="lines",
        name="Max Underfill Temp",
        line=dict(color="#10b981", width=3)
    ))

    fig.add_hline(
        y=T_GLASS,
        line_dash="dash",
        line_color="#fbbf24",
        annotation_text=f"Tg {T_GLASS:.0f}°C",
        annotation_font_color="#fbbf24"
    )
    
    fig.add_hline(
        y=260.0,
        line_dash="dot",
        line_color="#ef4444",
        annotation_text=f"Reflow Limit 260°C",
        annotation_font_color="#ef4444"
    )

    fig.update_layout(
        title=dict(text="Transient Thermal History", font=dict(color="white", size=16, family="Inter, sans-serif")),
        xaxis_title="Time (min)",
        yaxis_title="Temperature (°C)",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="white", family="Inter, sans-serif"),
        xaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        yaxis=dict(gridcolor="#334155", zerolinecolor="#334155"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="white")),
        height=480,
        margin=dict(l=20, r=20, t=50, b=40)
    )

    return fig


# ============================================================
# FINAL DECISION
# ============================================================

def combine_decision(packaging_result, thermal_result):
    if packaging_result["status"] == "FAILURE":
        return "REJECT / MANDATORY BAKE", "Packaging moisture, MSL, or reflow-count rule failed."

    if thermal_result["status"] == "CRITICAL":
        return "HOLD BATCH — THERMAL LIMIT EXCEEDED", thermal_result["reason"]

    if packaging_result["status"] == "WARNING" or thermal_result["status"] == "WARNING":
        return "ENGINEERING REVIEW", "Moisture or thermal profile is close to reliability limit."

    return "APPROVED", "Packaging and thermal screening results are acceptable."


# ============================================================
# STREAMLIT UI
# ============================================================

st.title("Micron HBM Packaging + Thermal Reliability Model")

st.markdown(
    """
This version uses packaging inputs to calculate moisture risk, then feeds the result into a
**3D transient finite-volume thermal model** with reflow heating, convection, localized heat generation,
and time-above-Tg reliability logic.
"""
)

with st.sidebar:
    st.header("Packaging Inputs")

    ambient_temp = st.slider("Ambient Temperature (°C)", 10.0, 50.0, 25.0, 0.5)
    humidity = st.slider("Humidity (%RH)", 10.0, 95.0, 60.0, 1.0)
    exposure_time = st.slider("Exposure Time (hours)", 0, 300, 24, 1)
    package_thickness = st.slider("Package Thickness (mm)", 0.5, 5.0, 1.27, 0.01)
    d0 = st.slider("D0 Diffusion Constant (cm²/s)", 0.0001, 0.0500, 0.0050, 0.0001)
    msl_rating = st.selectbox("MSL Rating", list(MSL_DB.keys()), index=3)
    glass_stoppage = st.toggle("Glass Stoppage Detected", value=False)
    reflow_count = st.number_input("Reflow Count", min_value=1, max_value=10, value=1)

    st.divider()
    st.header("Thermal Inputs")

    power_w = 0.0  # Chip is unpowered during reflow
    reflow_peak = st.slider("Reflow Peak Temperature (°C)", 180.0, 280.0, 245.0, 1.0)
    heating_time = st.slider("Reflow Cycle Time (s)", 60, 600, 240, 10)
    h_conv = st.slider("Convection Coefficient h (W/m²·K)", 5.0, 80.0, 20.0, 1.0)

    st.divider()
    st.header("3D Solver Resolution")

    nx = st.slider("Grid NX", 12, 30, 22, 2)
    ny = st.slider("Grid NY", 12, 30, 22, 2)
    nz = st.slider("Grid NZ", 8, 20, 14, 1)
    n_steps = st.slider("Time Steps", 30, 150, 80, 10)

run_button = st.button("Run 3D FVM Reliability Decision", type="primary", use_container_width=True)

if run_button:
    with st.spinner("Running packaging model and 3D transient FVM solver..."):
        packaging = evaluate_packaging(
            temp_c=ambient_temp,
            rh_pct=humidity,
            exposure_hr=exposure_time,
            thickness_mm=package_thickness,
            d0_cm2s=d0,
            msl_rating=msl_rating,
            glass_stoppage=glass_stoppage,
            reflow_count=reflow_count
        )

        fvm = run_3d_transient_fvm(
            ambient_c=ambient_temp,
            rh_pct=humidity,
            c_internal=packaging["c_internal"],
            thickness_mm=package_thickness,
            power_w=power_w,
            reflow_peak_c=reflow_peak,
            heating_time_s=heating_time,
            h_conv=h_conv,
            nx=nx,
            ny=ny,
            nz=nz,
            n_steps=n_steps
        )

        thermal = evaluate_thermal_fvm(fvm)
        final_status, final_reason = combine_decision(packaging, thermal)

        st.session_state["results"] = {
            "packaging": packaging,
            "fvm": fvm,
            "thermal": thermal,
            "final_status": final_status,
            "final_reason": final_reason
        }

if "results" in st.session_state:
    results = st.session_state["results"]
    packaging = results["packaging"]
    fvm = results["fvm"]
    thermal = results["thermal"]
    final_status = results["final_status"]
    final_reason = results["final_reason"]

    st.divider()
    st.subheader("Reliability Decision")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Packaging Status", packaging["status"])
    c2.metric("Internal Moisture", f"{packaging['c_internal']:.4f}%")
    c3.metric("Max Package Temp", f"{thermal['max_stack']:.1f} °C")
    c4.metric("Max Underfill Temp", f"{thermal['max_underfill']:.1f} °C")
    c5.metric("Time Above Tg", f"{thermal['time_above_tg']:.1f} s")

    if "APPROVED" in final_status:
        st.success(f"✅ {final_status} — {final_reason}")
    elif "REVIEW" in final_status:
        st.warning(f"⚠️ {final_status} — {final_reason}")
    else:
        st.error(f"❌ {final_status} — {final_reason}")

    left, right = st.columns(2)

    with left:
        st.markdown("### Packaging Assessment")
        
        if packaging["status"] == "SAFE":
            st.success(f"✅ **SAFE** — {packaging['reason']}")
        elif packaging["status"] == "WARNING":
            st.warning(f"⚠️ **WARNING** — {packaging['reason']}")
        else:
            st.error(f"❌ **FAILURE** — {packaging['reason']}")
            
        st.write(f"**Moisture threshold:** {packaging['moisture_limit']:.2f}%")
        st.write(f"**Floor life usage:** {packaging['usage_ratio']:.1%}")
        st.write(f"**Diffusion coefficient:** {packaging['d_coeff']:.4e} cm²/s")

    with right:
        st.markdown("### 3D Thermal Assessment")
        
        if thermal["status"] == "STABLE":
            st.success(f"✅ **STABLE** — {thermal['reason']}")
        elif thermal["status"] == "WARNING":
            st.warning(f"⚠️ **WARNING** — {thermal['reason']}")
        else:
            st.error(f"❌ **CRITICAL** — {thermal['reason']}")
            
        st.write(f"**Effective epoxy k:** {fvm['k_epoxy_eff']:.3f} W/m·K")
        st.write(f"**Time above warning temperature:** {thermal['time_above_warning']:.1f} s")
        st.write(f"**Time above Tg:** {thermal['time_above_tg']:.1f} s")

    with st.expander("Optional Expandable Section: 3D Thermal CAD Render + Temperature Profile", expanded=True):
        fig_3d = plot_3d_temperature_volume(fvm)
        fig_profile = plot_vertical_temperature_profile(fvm)
        fig_history = plot_reflow_history(fvm)

        st.plotly_chart(fig_3d, width="stretch", theme=None)

        col_a, col_b = st.columns(2)

        with col_a:
            st.plotly_chart(fig_profile, width="stretch", theme=None)

        with col_b:
            st.plotly_chart(fig_history, width="stretch", theme=None)

else:
    st.info("Set the inputs in the sidebar, then click **Run 3D FVM Reliability Decision**.")
