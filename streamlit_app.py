"""
BOMOCO - Business-Outcome-Aware Multi-Objective Cloud Optimizer
Streamlit Cloud Optimized Version

A self-driving cloud platform that optimizes cost, carbon, water, 
performance, and business KPIs simultaneously.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    AWS_REGIONS, AZURE_REGIONS, GCP_REGIONS,
    GRID_CARBON_BASELINE, DEFAULT_WEIGHTS,
    PUE_VALUES, INSTANCE_PRICING
)
from data.sample_data import (
    generate_workload_data, generate_carbon_intensity_forecast,
    generate_cost_forecast, generate_business_metrics,
    generate_optimization_history, calculate_sustainability_metrics
)
from utils.multi_objective import (
    MultiObjectiveOptimizer, BusinessKPICorrelator,
    SustainabilityScorer, OptimizationAction
)

# =============================================================================
# STREAMLIT CLOUD CONFIGURATION
# =============================================================================

def get_secret(section: str, key: str, default: str = "") -> str:
    """Safely get secret from Streamlit secrets."""
    try:
        return st.secrets.get(section, {}).get(key, default)
    except Exception:
        return default

def is_demo_mode() -> bool:
    """Check if running in demo mode."""
    try:
        return st.secrets.get("app", {}).get("demo_mode", True)
    except Exception:
        return True

def get_aws_credentials():
    """Get AWS credentials from secrets."""
    try:
        return {
            "aws_access_key_id": st.secrets.get("aws", {}).get("access_key_id", ""),
            "aws_secret_access_key": st.secrets.get("aws", {}).get("secret_access_key", ""),
            "region_name": st.secrets.get("aws", {}).get("region", "us-east-1"),
        }
    except Exception:
        return {"aws_access_key_id": "", "aws_secret_access_key": "", "region_name": "us-east-1"}


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="BOMOCO | Self-Driving Cloud",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/bomoco',
        'Report a bug': 'https://github.com/your-repo/bomoco/issues',
        'About': '''
        ## BOMOCO
        Business-Outcome-Aware Multi-Objective Cloud Optimizer
        
        **Patent Pending** - A revolutionary cloud optimization platform.
        '''
    }
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    .main { background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%); }
    .stApp { background: linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 100%); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle { font-size: 1.2rem; color: #94a3b8; margin-bottom: 2rem; }
    
    .demo-badge {
        background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
        color: white; padding: 4px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    }
    
    .live-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white; padding: 4px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; text-transform: uppercase;
    }
    
    div[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; }
    .stSlider > div > div { background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444); }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

if 'workloads' not in st.session_state:
    st.session_state.workloads = generate_workload_data(50)
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = MultiObjectiveOptimizer()
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []


# =============================================================================
# CACHED DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=300)
def cached_business_metrics(days: int = 30):
    """Cache business metrics generation."""
    return generate_business_metrics(days)

@st.cache_data(ttl=300)
def cached_carbon_forecast(hours: int, region: str):
    """Cache carbon intensity forecast."""
    return generate_carbon_intensity_forecast(hours, region)

@st.cache_data(ttl=300)
def cached_cost_forecast(hours: int, region: str):
    """Cache cost forecast."""
    return generate_cost_forecast(hours, region)

@st.cache_data(ttl=60)
def cached_sustainability_metrics(workloads_hash: str, workloads):
    """Cache sustainability metrics calculation."""
    return calculate_sustainability_metrics(workloads)


# =============================================================================
# COMPONENTS
# =============================================================================

def render_header():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="hero-title">BOMOCO</h1>', unsafe_allow_html=True)
        st.markdown('<p class="hero-subtitle">Business-Outcome-Aware Multi-Objective Cloud Optimizer</p>', unsafe_allow_html=True)
    with col2:
        demo_mode = is_demo_mode()
        badge = "demo-badge" if demo_mode else "live-badge"
        text = "DEMO MODE" if demo_mode else "LIVE DATA"
        st.markdown(f'''
        <div style="text-align: right; padding-top: 20px;">
            <span class="{badge}">{text}</span><br>
            <span style="color: #10b981; font-size: 0.9rem; margin-top: 8px; display: block;">● System Active</span>
            <span style="color: #64748b; font-size: 0.8rem;">{datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
        </div>
        ''', unsafe_allow_html=True)


def render_kpi_metrics(workloads, business_metrics):
    sustainability = calculate_sustainability_metrics(workloads)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Monthly Cloud Spend", f"${sustainability['total_monthly_cost']:,.0f}",
                  f"-{sustainability['rightsizing_opportunity_percent']:.0f}% opportunity", delta_color="inverse")
    with col2:
        st.metric("Carbon Footprint", f"{sustainability['total_monthly_carbon_kg']:,.0f} kg",
                  f"-{sustainability['carbon_shift_opportunity_percent']:.0f}% potential", delta_color="inverse")
    with col3:
        st.metric("Water Usage", f"{sustainability['total_monthly_water_liters']/1000:,.0f} kL",
                  "-12% vs benchmark", delta_color="inverse")
    with col4:
        if not business_metrics.empty:
            rev = business_metrics.iloc[-1]['daily_revenue']
            prev = business_metrics.iloc[-2]['daily_revenue'] if len(business_metrics) > 1 else rev
            st.metric("Daily Revenue", f"${rev:,.0f}", f"{((rev-prev)/prev)*100:+.1f}%")
    with col5:
        if not business_metrics.empty:
            st.metric("Conversion Rate", f"{business_metrics.iloc[-1]['conversion_rate']:.2f}%", "+0.3%")


def render_sidebar():
    st.sidebar.markdown("### ⚙️ Optimization Weights")
    
    cost_w = st.sidebar.slider("💰 Cost", 0.0, 1.0, 0.35, 0.05)
    carbon_w = st.sidebar.slider("🌱 Carbon", 0.0, 1.0, 0.25, 0.05)
    water_w = st.sidebar.slider("💧 Water", 0.0, 1.0, 0.10, 0.05)
    perf_w = st.sidebar.slider("⚡ Performance", 0.0, 1.0, 0.20, 0.05)
    biz_w = st.sidebar.slider("📈 Business KPI", 0.0, 1.0, 0.10, 0.05)
    
    total = cost_w + carbon_w + water_w + perf_w + biz_w
    if total > 0:
        st.session_state.optimizer.set_weights({
            "cost": cost_w/total, "carbon": carbon_w/total, "water": water_w/total,
            "performance": perf_w/total, "business_kpi": biz_w/total,
        })
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Quick Actions")
    
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.session_state.workloads = generate_workload_data(50)
        st.rerun()
    
    if st.sidebar.button("⚡ Run Optimization", use_container_width=True, type="primary"):
        with st.spinner("Analyzing..."):
            carbon_f = generate_carbon_intensity_forecast(48, "us-east-1")
            cost_f = generate_cost_forecast(24, "us-east-1")
            recs = st.session_state.optimizer.generate_recommendations(
                st.session_state.workloads, carbon_f, cost_f, 20)
            st.session_state.recommendations = recs
        st.success(f"Generated {len(recs)} recommendations!")
        st.rerun()
    
    st.sidebar.markdown("---")
    if is_demo_mode():
        st.sidebar.info("📊 Running with simulated data.\n\nConfigure secrets for live AWS/carbon data.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align:center;color:#64748b;font-size:0.8rem;">
        BOMOCO v0.1.0 | Patent Pending
    </div>
    """, unsafe_allow_html=True)


def render_carbon_map(workloads):
    data = []
    for region, info in AWS_REGIONS.items():
        grid = info["grid"]
        carbon = GRID_CARBON_BASELINE.get(grid, 400)
        wl_count = len(workloads[workloads["region"] == region])
        cost = workloads[workloads["region"] == region]["monthly_cost"].sum()
        data.append({"region": region, "name": info["name"], "lat": info["lat"], "lon": info["lon"],
                    "carbon_intensity": carbon, "workload_count": wl_count, "monthly_cost": cost, "grid": grid})
    
    df = pd.DataFrame(data)
    fig = px.scatter_geo(df, lat="lat", lon="lon", size="workload_count", color="carbon_intensity",
                        hover_name="name", color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
                        size_max=40, title="Global Carbon Intensity & Workload Distribution")
    fig.update_layout(
        geo=dict(showland=True, landcolor="#1a1a2e", showocean=True, oceancolor="#0a0a0f",
                showcountries=True, countrycolor="#2d3748", bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"), height=400, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)


def render_carbon_forecast(region):
    forecast = generate_carbon_intensity_forecast(48, region)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast["timestamp"], y=forecast["carbon_intensity_gco2_kwh"],
                            mode="lines", line=dict(color="#10b981", width=2),
                            fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.1)"))
    fig.update_layout(title=f"48-Hour Carbon Forecast - {region}", xaxis_title="Time", yaxis_title="gCO₂/kWh",
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     font=dict(color="#e2e8f0"), height=300)
    st.plotly_chart(fig, use_container_width=True)


def render_pareto(recs):
    if not recs:
        st.info("🎯 Click 'Run Optimization' to see trade-off analysis")
        return
    
    cost_imp = [-r.cost_impact for r in recs]
    carbon_imp = [-r.carbon_impact for r in recs]
    colors = {"rightsize_down": "#10b981", "spot_conversion": "#0ea5e9", "carbon_shift": "#22c55e",
             "region_migration": "#8b5cf6", "rightsize_up": "#f59e0b"}
    c = [colors.get(r.action_type, "#64748b") for r in recs]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cost_imp, y=carbon_imp, mode="markers",
                            marker=dict(size=12, color=c, line=dict(width=2, color="white")),
                            text=[r.description[:40] for r in recs],
                            hovertemplate="<b>%{text}</b><br>Cost: %{x:.1f}%<br>Carbon: %{y:.1f}%<extra></extra>"))
    fig.update_layout(title="Cost vs Carbon Trade-offs", xaxis_title="Cost Savings (%)", yaxis_title="Carbon Reduction (%)",
                     paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"), height=400)
    st.plotly_chart(fig, use_container_width=True)


def render_recommendations(recs):
    if not recs:
        st.info("🎯 Click 'Run Optimization' to generate recommendations")
        return
    
    impact = st.session_state.optimizer.estimate_total_impact(recs)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Savings", f"${impact['total_monthly_savings']:,.0f}/mo")
    col2.metric("Cost Reduction", f"{impact['total_cost_reduction_percent']:.1f}%")
    col3.metric("Carbon Reduction", f"{impact['total_carbon_reduction_percent']:.1f}%")
    col4.metric("Avg Confidence", f"{impact['avg_confidence']*100:.0f}%")
    
    st.markdown("---")
    for i, r in enumerate(recs[:10]):
        with st.expander(f"**{i+1}. {r.action_type.replace('_',' ').title()}** | Score: {r.composite_score:.3f} | ${r.estimated_savings_monthly:,.0f}/mo", expanded=(i<3)):
            c1, c2 = st.columns([2,1])
            with c1:
                st.markdown(f"**{r.description}**")
                st.markdown(f"Workload: `{r.workload_id}`")
                for lbl, val in [("💰 Cost", r.cost_impact), ("🌱 Carbon", r.carbon_impact), ("⚡ Perf", r.performance_impact)]:
                    color = "#10b981" if val < 0 else "#ef4444"
                    st.markdown(f'<span style="color:{color}">{lbl}: {val:+.1f}%</span>', unsafe_allow_html=True)
            with c2:
                st.markdown(f"**Confidence:** {r.confidence:.0%}")
                st.markdown(f"**Risk:** {r.risk_level}")
                st.markdown(f"**Effort:** {r.implementation_effort}")


def render_business_correlation(biz_metrics, opt_history):
    if biz_metrics.empty:
        return
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=biz_metrics["date"], y=biz_metrics["daily_revenue"],
                            name="Revenue", line=dict(color="#8b5cf6", width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=biz_metrics["date"], y=biz_metrics["infrastructure_health_score"],
                            name="Infra Health", line=dict(color="#10b981", width=2, dash="dot")), secondary_y=True)
    fig.update_layout(title="Business KPI Correlation", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     font=dict(color="#e2e8f0"), height=350)
    st.plotly_chart(fig, use_container_width=True)
    
    correlator = BusinessKPICorrelator()
    corr = correlator.analyze_correlations(opt_history, biz_metrics)
    
    st.markdown("**Learned Correlations (Patent Innovation):**")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div style="text-align:center;padding:12px;background:rgba(139,92,246,0.1);border-radius:8px;"><div style="font-size:1.5rem;color:#8b5cf6;">{corr["performance_to_revenue"]:.0%}</div><div style="color:#94a3b8;">Performance → Revenue</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div style="text-align:center;padding:12px;background:rgba(239,68,68,0.1);border-radius:8px;"><div style="font-size:1.5rem;color:#ef4444;">{corr["latency_to_conversions"]:.0%}</div><div style="color:#94a3b8;">Latency → Conversions</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div style="text-align:center;padding:12px;background:rgba(34,197,94,0.1);border-radius:8px;"><div style="font-size:1.5rem;color:#22c55e;">{corr["carbon_to_brand"]:.0%}</div><div style="color:#94a3b8;">Sustainability → Brand</div></div>', unsafe_allow_html=True)


# =============================================================================
# MAIN
# =============================================================================

def main():
    render_header()
    render_sidebar()
    
    biz_metrics = generate_business_metrics(30)
    opt_history = generate_optimization_history(20)
    
    render_kpi_metrics(st.session_state.workloads, biz_metrics)
    st.markdown("---")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌍 Sustainability", "📊 Recommendations", "📈 Business Intel", "🔮 Forecasts", "📋 Inventory"])
    
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            render_carbon_map(st.session_state.workloads)
        with c2:
            region = st.selectbox("Region", list(AWS_REGIONS.keys()), format_func=lambda x: f"{AWS_REGIONS[x]['name']} ({x})")
            wl = st.session_state.workloads[st.session_state.workloads["region"] == region]
            grid = AWS_REGIONS[region]["grid"]
            st.markdown(f'''
            <div style="background:rgba(255,255,255,0.05);padding:16px;border-radius:8px;">
                <h4>{AWS_REGIONS[region]['name']}</h4>
                <p>Grid: {grid} | Carbon: {GRID_CARBON_BASELINE.get(grid, 400)} gCO₂/kWh</p>
                <p>Workloads: {len(wl)} | Cost: ${wl["monthly_cost"].sum():,.0f}/mo</p>
            </div>
            ''', unsafe_allow_html=True)
        render_carbon_forecast(region)
    
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🎯 Recommendations")
            render_recommendations(st.session_state.recommendations)
        with c2:
            st.markdown("### 📊 Trade-offs")
            render_pareto(st.session_state.recommendations)
    
    with tab3:
        st.markdown("### 📈 Business KPI Correlation")
        st.markdown("*Patent-pending innovation: business-aware cloud optimization*")
        render_business_correlation(biz_metrics, opt_history)
    
    with tab4:
        st.markdown("### 🔮 Forecasts")
        c1, c2 = st.columns(2)
        with c1:
            cf = generate_cost_forecast(24, "us-east-1")
            fig = px.line(cf, x="timestamp", y="spot_discount_percentage", title="Spot Discount Forecast")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"), height=300)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            forecasts = pd.concat([generate_carbon_intensity_forecast(24, r).assign(region_name=AWS_REGIONS[r]["name"]) for r in ["us-east-1", "us-west-2", "eu-west-1"]])
            fig = px.line(forecasts, x="timestamp", y="carbon_intensity_gco2_kwh", color="region_name", title="Carbon by Region")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"), height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.markdown("### 📋 Workload Inventory")
        c1, c2, c3 = st.columns(3)
        tf = c1.multiselect("Type", st.session_state.workloads["workload_type"].unique())
        rf = c2.multiselect("Region", st.session_state.workloads["region"].unique())
        bf = c3.multiselect("Business Unit", st.session_state.workloads["business_unit"].unique())
        
        filtered = st.session_state.workloads.copy()
        if tf: filtered = filtered[filtered["workload_type"].isin(tf)]
        if rf: filtered = filtered[filtered["region"].isin(rf)]
        if bf: filtered = filtered[filtered["business_unit"].isin(bf)]
        
        display = filtered[["workload_id", "workload_name", "workload_type", "region", "instance_type", "cpu_utilization", "monthly_cost"]].copy()
        display["cpu_utilization"] = display["cpu_utilization"].apply(lambda x: f"{x*100:.0f}%")
        display["monthly_cost"] = display["monthly_cost"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(display, use_container_width=True, height=400)


if __name__ == "__main__":
    main()
