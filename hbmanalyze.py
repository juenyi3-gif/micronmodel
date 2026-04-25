import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 0. GLOBAL CONFIGURATION ---
st.set_page_config(page_title="Micron Chip Analysis System", layout="wide", page_icon="🛡️")

# --- 1. SHARED CONSTANTS & INITIALIZATION ---

# --- Packaging Variables ---
initial_states = {
    'temp': 25.0, 'temp_slider': 25.0, 'temp_input': 25.0,
    'rh': 60.0, 'rh_slider': 60.0, 'rh_input': 60.0,
    'thickness': 1.27, 'thickness_slider': 1.27, 'thickness_input': 1.27,
    'time': 24, 'time_slider': 24, 'time_input': 24,
    'd0': 0.005, 'd0_slider': 0.005, 'd0_input': 0.005,
    'reflow_count': 1
}

for key, value in initial_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

def update_sync(param):
    st.session_state[param] = st.session_state[f"{param}_slider"]
    st.session_state[f"{param}_input"] = st.session_state[param]

def update_input_sync(param):
    st.session_state[param] = st.session_state[f"{param}_input"]
    st.session_state[f"{param}_slider"] = st.session_state[param]

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

Ea = 0.35 
k_boltzmann = 8.617e-5 

def run_diffusion_model(temp, rh, time_hr, thickness_mm, d0_cm2s):
    temp_k = temp + 273.15
    d0_m2s = d0_cm2s * 1e-4
    D_m2s = d0_m2s * np.exp(-Ea / (k_boltzmann * temp_k))
    L_m = (thickness_mm / 2) / 1000 
    time_s = time_hr * 3600
    tau = (D_m2s * time_s) / (L_m**2)
    c_surface = (rh / 85) * 0.4
    c_internal = c_surface * (1 - np.exp(-tau))
    D_display = D_m2s * 1e4
    return D_display, c_surface, c_internal

# --- Thermal PDE Variables ---
K_SI        = 148.0   # Silicon thermal conductivity  [W/m·K]
K_EPOXY     = 0.77    # Epoxy underfill conductivity  [W/m·K]
K_SUBSTRATE = 0.30    # Package substrate (FR4)       [W/m·K]
T_GLASS     = 125.0   # Epoxy glass-transition temp   [°C]
NPU_SLOPE   = 8.584   
NPU_INTERC  = 29.524

N  = 12          
DX = 60e-6       
CROSS_SEC = 1e-4 

LAYER_NAMES = [
    "Substrate (BGA)",       "Microbumps/Solder",
    "HBM Die-1 (Si)",        "HBM Die-1 (Si)",
    "Underfill — IF1 ⚠",    "HBM Die-2 (Si)",
    "HBM Die-2 (Si)",        "Underfill — IF2 ⚠",
    "HBM Die-3 (Si)",        "HBM Die-3 (Si)",
    "Underfill — IF3 ⚠",    "HBM Die-4 (Si) Top",
]
DIE_NODES      = [2, 3, 5, 6, 8, 9, 11]   
UNDERFILL_NODES = [1, 4, 7, 10]            

def get_k_array(humidity_pct):
    deg    = max(1.0 - 0.35 * (humidity_pct / 85.0), 0.55)
    k_ep   = K_EPOXY * deg
    return np.array([
        K_SUBSTRATE, k_ep, K_SI, K_SI, k_ep, K_SI, K_SI, k_ep, K_SI, K_SI, k_ep, K_SI,
    ])

def solve_steady_state(npu_power_W, humidity_pct, T_ambient=25.0):
    k = get_k_array(humidity_pct)
    Q = np.zeros(N)
    q_den = npu_power_W / (len(DIE_NODES) * DX * CROSS_SEC)
    for n in DIE_NODES:
        Q[n] = q_den
    A = np.zeros((N, N))
    b = np.zeros(N)

    T_sub = T_ambient + npu_power_W * 1.5
    A[0, 0] = 1.0
    b[0]    = T_sub

    for i in range(1, N - 1):
        k_e = 2 * k[i] * k[i+1] / (k[i] + k[i+1])
        k_w = 2 * k[i-1] * k[i] / (k[i-1] + k[i])
        A[i, i-1] =  k_w / DX**2
        A[i, i]   = -(k_e + k_w) / DX**2
        A[i, i+1] =  k_e / DX**2
        b[i]      = -Q[i]

    h = 8.0
    A[-1, -2] = -k[-1] / DX
    A[-1, -1] =  k[-1] / DX + h
    b[-1]     =  h * T_ambient

    return np.linalg.solve(A, b)


# --- 2. MAIN APPLICATION UI ---
st.title("Micron Integrated Chip Analysis")

tab1, tab2 = st.tabs(["Predictive Moisture Analysis (Packaging)", "Sentinel-AI Thermal Reliability (PDE)"])

# ----------------- TAB 1: PACKAGING -----------------
with tab1:
    tab1_ctrl, tab1_main = st.columns([1, 3])
    
    with tab1_ctrl:
        st.header("🕹️ Control Panel")
        
        st.subheader("Ambient Environment")
        st.slider("Temperature (°C)", 10.0, 50.0, key="temp_slider", on_change=update_sync, args=('temp',))
        st.number_input("Type Temperature", 10.0, 50.0, key="temp_input", on_change=update_input_sync, args=('temp',))
        
        st.slider("Humidity (% RH)", 10.0, 95.0, key="rh_slider", on_change=update_sync, args=('rh',))
        st.number_input("Type Humidity", 10.0, 95.0, key="rh_input", on_change=update_input_sync, args=('rh',))
        
        st.subheader("Component Details")
        st.slider("Thickness (mm)", 0.5, 5.0, key="thickness_slider", on_change=update_sync, args=('thickness',))
        st.number_input("Type Thickness", 0.5, 5.0, key="thickness_input", on_change=update_input_sync, args=('thickness',))
        
        st.subheader("Reflow Process History")
        reflow_count = st.number_input("Current Reflow Count", 1, 10, key="reflow_count_input")
        
        st.subheader("Exposure Timer")
        st.slider("Time (Hours)", 0, 300, key="time_slider", on_change=update_sync, args=('time',))
        st.number_input("Type Time", 0, 300, key="time_input", on_change=update_input_sync, args=('time',))
        
        st.subheader("Physics Parameters")
        st.slider("D0 (Ref. Diffusion) [cm²/s]", 0.0001, 0.0500, key="d0_slider", format="%.4f", on_change=update_sync, args=('d0',))
        st.number_input("Type D0", 0.0001, 0.0500, key="d0_input", format="%.4f", on_change=update_input_sync, args=('d0',))
        
        msl_val = st.selectbox("MSL Rating", list(MSL_DB.keys()), index=3)
        glass_stoppage = st.toggle("Glass Stoppage Defect Detected")

    with tab1_main:
        st.title("Predictive Moisture Analysis System")
        st.markdown("### *Real-Time Moisture Diffusion & Reflow Risk Assessment*")
        
        D_coeff, sat_level, c_result = run_diffusion_model(
            st.session_state.temp, 
            st.session_state.rh, 
            st.session_state.time, 
            st.session_state.thickness, 
            st.session_state.d0
        )
        
        threshold = 0.07 if glass_stoppage else 0.15 
        floor_life_limit = MSL_DB[msl_val]["limit"]
        usage_ratio = st.session_state.time / floor_life_limit if floor_life_limit > 0 else 1.0
        max_reflow_limit = 3
        
        reflow_fail = reflow_count > max_reflow_limit
        moisture_fail = c_result >= threshold
        floor_life_fail = usage_ratio >= 1.0
        
        status = "SAFE"
        if reflow_fail or moisture_fail or floor_life_fail or msl_val == "6":
            status = "FAILURE"
        elif usage_ratio > 0.9 or c_result > (threshold * 0.8):
            status = "WARNING"
            
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Internal Moisture (C)", f"{c_result:.4f}%", 
                      delta=f"{c_result - threshold:.4f}%", delta_color="inverse")
            st.caption(f"Safety Limit: {threshold}%")
        with m2:
            st.metric("Floor Life Usage", f"{usage_ratio:.1%}")
            st.caption(f"MSL {msl_val} Limit: {MSL_DB[msl_val]['label']}")
        with m3:
            st.metric("Reflow Cycles", f"{reflow_count} / {max_reflow_limit}")
            st.caption("Structural limit is 3")
        with m4:
            st.metric("System Health", status)
            st.caption("Overall Assessment")
            
        st.divider()
        if status == "FAILURE":
            st.error(f"❌ **CRITICAL: REFLOW REJECTED**")
            if reflow_fail:
                st.write(f"**Reason:** Reflow count ({reflow_count}) exceeds the industry limit ({max_reflow_limit}). Material fatigue risk.")
            elif msl_val == "6":
                st.write("**Reason:** MSL 6 components require a mandatory bake before any reflow.")
            elif moisture_fail:
                st.write(f"**Reason:** Moisture level ({c_result:.4f}%) is above the safety threshold.")
            else:
                st.write("**Reason:** Component Floor Life has expired.")
            st.button("📋 Recommended Action: Mandatory Bake 24h @ 125°C", use_container_width=True)
            
        elif status == "WARNING":
            st.warning(f"⚠️ **PROACTIVE ALERT: NEARING LIMITS**")
            st.write("The component is nearing its moisture or life limit. Move to reflow immediately.")
        else:
            st.success(f"✅ **BATCH APPROVED**")
            st.write("Parameters are within safe chemical and physical limits.")
            
        with st.expander("View Diffusion Math & Detail Parameters"):
            col_calc1, col_calc2 = st.columns(2)
            with col_calc1:
                st.metric("Diffusion Coefficient (D)", f"{D_coeff:.4e} cm²/s")
                st.caption("Calculated via Arrhenius equation")
            with col_calc2:
                st.metric("Saturation Level (C_surface)", f"{sat_level:.4f}%")
                st.caption("Max moisture capacity at current RH")


# ----------------- TAB 2: THERMAL PDE -----------------
with tab2:
    tab2_ctrl, tab2_main = st.columns([1, 4])
    
    with tab2_ctrl:
        st.header("⚙️ Factory Inputs")
        npu_power    = st.slider("NPU Power (W)", 1.0, 15.0, 5.0, step=0.5,
                                  help="Power dissipated inside HBM stack")
        humidity_pde = st.slider("Cleanroom Humidity (%RH)", 10, 90, 45, key="tab2_hum")
        material_age = st.number_input("Mold Compound Age (days)", 1, 30, 5)
        T_amb        = st.slider("Ambient Temperature (°C)", 20, 40, 25, key="tab2_tamb")
        st.divider()
        st.markdown("**Material Properties**")
        st.code(f"k_Si    = {K_SI} W/m·K\nk_Epoxy = {K_EPOXY} W/m·K\nTg      = {T_GLASS} °C")
        st.info("Solver: Steady-state FVM  |  No CFL constraint  |  Harmonic-mean interfaces")
        
    with tab2_main:
        st.title("🛡️ Sentinel-AI: HBM Reliability Digital Twin")
        st.markdown("### Physics-Informed Manufacturing Risk Detection — Batu Kawan Fab")
        
        eff_humidity = min(humidity_pde * (1 + material_age * 0.008), 90.0)
        T_profile  = solve_steady_state(npu_power, eff_humidity, T_amb)
        T_max_sim  = float(np.max(T_profile))
        T_max_emp  = NPU_SLOPE * npu_power + NPU_INTERC
        T_underfill_max = max(T_profile[n] for n in UNDERFILL_NODES)
        
        risk_score = (T_max_sim / 260.0) * (eff_humidity / 50.0)
        
        z_um = np.arange(N) * DX * 1e6
        
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
        
        col1, col2 = st.columns([3, 2])
        with col1:
            st.plotly_chart(fig_3d,      width="stretch", key="vol3d")
            st.plotly_chart(fig_profile, width="stretch", key="profile")
            
        with col2:
            st.subheader("📊 Reliability Analysis")
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
