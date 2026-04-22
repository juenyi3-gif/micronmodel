import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── PHYSICAL CONSTANTS (ScienceDirect 2025 HBM study) ─────────────────────
K_SI        = 148.0   # Silicon thermal conductivity  [W/m·K]
K_EPOXY     = 0.77    # Epoxy underfill conductivity  [W/m·K]
K_SUBSTRATE = 0.30    # Package substrate (FR4)       [W/m·K]
T_GLASS     = 125.0   # Epoxy glass-transition temp   [°C]  ← critical alarm
NPU_SLOPE   = 8.584   # Empirical: T_max = slope·P + intercept (study Eq.)
NPU_INTERC  = 29.524

# ── STACK GEOMETRY ─────────────────────────────────────────────────────────
N  = 12          # number of 1-D nodes (vertical slice through HBM stack)
DX = 60e-6       # node spacing: 60 µm  →  720 µm total height
CROSS_SEC = 1e-4 # die cross-section area [m²]  (10 mm × 10 mm)

LAYER_NAMES = [
    "Substrate (BGA)",       "Microbumps/Solder",
    "HBM Die-1 (Si)",        "HBM Die-1 (Si)",
    "Underfill — IF1 ⚠",    "HBM Die-2 (Si)",
    "HBM Die-2 (Si)",        "Underfill — IF2 ⚠",
    "HBM Die-3 (Si)",        "HBM Die-3 (Si)",
    "Underfill — IF3 ⚠",    "HBM Die-4 (Si) Top",
]
DIE_NODES      = [2, 3, 5, 6, 8, 9, 11]   # silicon nodes (heat sources)
UNDERFILL_NODES = [1, 4, 7, 10]            # epoxy nodes (failure risk)

# ── PHYSICS: build conductivity array ──────────────────────────────────────
def get_k_array(humidity_pct):
    """Moisture degrades epoxy k via Fick absorption (up to 35% at 85 %RH)."""
    deg    = max(1.0 - 0.35 * (humidity_pct / 85.0), 0.55)
    k_ep   = K_EPOXY * deg
    return np.array([
        K_SUBSTRATE,          # 0  substrate
        k_ep,                 # 1  microbumps
        K_SI, K_SI,           # 2-3 Die 1
        k_ep,                 # 4  underfill IF1  ← delamination start
        K_SI, K_SI,           # 5-6 Die 2
        k_ep,                 # 7  underfill IF2
        K_SI, K_SI,           # 8-9 Die 3
        k_ep,                 # 10 underfill IF3
        K_SI,                 # 11 Die 4 (top)
    ])

# ── PHYSICS: steady-state finite volume solver ─────────────────────────────
def solve_steady_state(npu_power_W, humidity_pct, T_ambient=25.0):
    """
    Solves  d/dx[k(x) · dT/dx] + Q(x) = 0  in steady state.

    Uses Finite Volume Method with harmonic-mean interface conductivity
    — no time-stepping, no CFL stability constraint.

    BCs:
      Node 0  : Dirichlet  — package substrate (heated via board-level Rth)
      Node N-1: Robin      — natural convection to ambient (h=8 W/m²·K)
    """
    k = get_k_array(humidity_pct)

    # Heat source density [W/m³] distributed across silicon die nodes
    Q = np.zeros(N)
    q_den = npu_power_W / (len(DIE_NODES) * DX * CROSS_SEC)
    for n in DIE_NODES:
        Q[n] = q_den

    A = np.zeros((N, N))
    b = np.zeros(N)

    # Node 0: substrate temperature (package Rth ≈ 1.5 °C/W)
    T_sub = T_ambient + npu_power_W * 1.5
    A[0, 0] = 1.0
    b[0]    = T_sub

    # Interior nodes: harmonic-mean conductivity at cell faces
    for i in range(1, N - 1):
        k_e = 2 * k[i] * k[i+1] / (k[i] + k[i+1])
        k_w = 2 * k[i-1] * k[i] / (k[i-1] + k[i])
        A[i, i-1] =  k_w / DX**2
        A[i, i]   = -(k_e + k_w) / DX**2
        A[i, i+1] =  k_e / DX**2
        b[i]      = -Q[i]

    # Node N-1: convection BC  →  k·dT/dx = h·(T - T_amb)
    h = 8.0
    A[-1, -2] = -k[-1] / DX
    A[-1, -1] =  k[-1] / DX + h
    b[-1]     =  h * T_ambient

    return np.linalg.solve(A, b)

# ── STREAMLIT UI ───────────────────────────────────────────────────────────
st.set_page_config(page_title="Micron Sentinel-AI", layout="wide", page_icon="🛡️")
st.title("🛡️ Sentinel-AI: HBM Reliability Digital Twin")
st.markdown("### Physics-Informed Manufacturing Risk Detection — Batu Kawan Fab")

with st.sidebar:
    st.header("⚙️ Factory Inputs")
    npu_power    = st.slider("NPU Power (W)", 1.0, 15.0, 5.0, step=0.5,
                              help="Power dissipated inside HBM stack")
    humidity     = st.slider("Cleanroom Humidity (%RH)", 10, 90, 45)
    material_age = st.number_input("Mold Compound Age (days)", 1, 30, 5)
    T_amb        = st.slider("Ambient Temperature (°C)", 20, 40, 25)
    st.divider()
    st.markdown("**Material Properties**")
    st.code(f"k_Si    = {K_SI} W/m·K\nk_Epoxy = {K_EPOXY} W/m·K\nTg      = {T_GLASS} °C")
    st.info("Solver: Steady-state FVM  |  No CFL constraint  |  Harmonic-mean interfaces")

# ── CALCULATIONS ───────────────────────────────────────────────────────────
# Older mold compound absorbs more moisture
eff_humidity = min(humidity * (1 + material_age * 0.008), 90.0)

T_profile  = solve_steady_state(npu_power, eff_humidity, T_amb)
T_max_sim  = float(np.max(T_profile))
T_max_emp  = NPU_SLOPE * npu_power + NPU_INTERC   # study empirical formula
T_underfill_max = max(T_profile[n] for n in UNDERFILL_NODES)

# Risk score (normalised to JEDEC thresholds)
risk_score = (T_max_sim / 260.0) * (eff_humidity / 50.0)

# ── FIGURES ────────────────────────────────────────────────────────────────
z_um = np.arange(N) * DX * 1e6   # convert to µm for axis

# Figure 1: Temperature profile through the stack
fig_profile = go.Figure()
colors = ["#ef4444" if T_profile[i] > T_GLASS
          else "#f97316" if T_profile[i] > 100
          else "#22c55e" for i in range(N)]

fig_profile.add_trace(go.Bar(
    x=T_profile, y=LAYER_NAMES, orientation='h',
    marker_color=colors, name="Simulated T",
    text=[f"{t:.1f} °C" for t in T_profile], textposition="outside",
))
fig_profile.add_vline(x=T_GLASS, line_dash="dash", line_color="#ef4444",
                      annotation_text=f"Tg = {T_GLASS}°C (epoxy glass transition)")
fig_profile.update_layout(
    title="Steady-State Temperature Profile — HBM Stack",
    xaxis_title="Temperature (°C)",
    height=420, margin=dict(l=10, r=80, t=50, b=40),
    plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
    font=dict(color="white"), xaxis=dict(gridcolor="#1e293b"),
)

# Figure 2: 3D volumetric heat map
nx, ny = 5, 5
X, Y, Z_idx = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(N), indexing='ij')
T_vol = T_profile[Z_idx]

fig_3d = go.Figure(data=go.Volume(
    x=X.flatten(), y=Y.flatten(), z=(Z_idx * DX * 1e6).flatten(),
    value=T_vol.flatten(),
    isomin=float(T_profile.min()), isomax=float(T_profile.max()),
    opacity=0.20, surface_count=18, colorscale="Hot",
    colorbar=dict(title="°C", tickfont=dict(color="white")),
))
fig_3d.update_layout(
    title="3D Thermal Distribution — HBM Stack",
    scene=dict(
        xaxis_title="X", yaxis_title="Y", zaxis_title="Height (µm)",
        bgcolor="#0f172a",
        xaxis=dict(color="white"), yaxis=dict(color="white"), zaxis=dict(color="white"),
    ),
    paper_bgcolor="#0f172a", font=dict(color="white"),
    height=480, margin=dict(l=0, r=0, t=50, b=0),
)

# ── DASHBOARD LAYOUT ───────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.plotly_chart(fig_3d,      width="stretch", key="vol3d")
    st.plotly_chart(fig_profile, width="stretch", key="profile")

with col2:
    st.subheader("📊 Reliability Analysis")

    # ── Risk alert ──
    if T_underfill_max >= T_GLASS:
        st.error(f"🔴 CRITICAL — Underfill exceeds Tg ({T_underfill_max:.1f} °C ≥ {T_GLASS} °C)")
        st.write("**Failure Mode:** Epoxy softening → Interfacial delamination")
        st.write("**Action:** Intercept batch. Schedule CSAM acoustic microscopy.")
    elif T_underfill_max >= 100:
        st.warning(f"🟡 WARNING — Underfill at {T_underfill_max:.1f} °C (approaching Tg)")
        st.write("**Risk:** CTE mismatch stress, early fatigue cracking.")
    else:
        st.success(f"✅ STABLE — Underfill at {T_underfill_max:.1f} °C")
        st.write("Operating within JEDEC safe-zone.")

    st.divider()

    # ── Key metrics ──
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Max Die Temp (Simulated)", f"{T_max_sim:.1f} °C",
                  delta=f"{T_max_sim - T_GLASS:.1f} °C vs Tg")
    with col_b:
        st.metric("Max Temp (Empirical)", f"{T_max_emp:.1f} °C",
                  help="Study formula: T = 8.584·P + 29.524")

    st.metric("Worst Underfill Temp", f"{T_underfill_max:.1f} °C")
    st.metric("Effective Humidity (age-adjusted)", f"{eff_humidity:.1f} %RH")
    st.metric("Risk Score", f"{risk_score:.3f}",
              delta="CRITICAL" if risk_score > 1.2 else "WARNING" if risk_score > 0.8 else "SAFE")

    st.divider()
    st.markdown("**Empirical vs Simulated T_max**")
    fig_comp = go.Figure()
    P_range = np.linspace(1, 15, 50)
    fig_comp.add_trace(go.Scatter(
        x=P_range,
        y=NPU_SLOPE * P_range + NPU_INTERC,
        mode='lines', name="Empirical (study)", line=dict(color="#60a5fa"),
    ))
    fig_comp.add_trace(go.Scatter(
        x=[npu_power], y=[T_max_sim],
        mode='markers', name="Current sim",
        marker=dict(color="#f97316", size=12, symbol="diamond"),
    ))
    fig_comp.add_hline(y=T_GLASS, line_dash="dash", line_color="#ef4444",
                       annotation_text="Tg = 125°C")
    fig_comp.update_layout(
        xaxis_title="NPU Power (W)", yaxis_title="T_max (°C)",
        height=250, margin=dict(l=10, r=10, t=20, b=40),
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font=dict(color="white", size=11),
        legend=dict(orientation="h", y=1.15),
        xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
    )
    st.plotly_chart(fig_comp, width="stretch", key="empirical")