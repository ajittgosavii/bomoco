"""
BOMOCO - Business-Outcome-Aware Multi-Objective Cloud Optimizer
Professional UI with Improved Visibility
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import AWS_REGIONS, GRID_CARBON_BASELINE, DEFAULT_WEIGHTS, PUE_VALUES, INSTANCE_PRICING
from data.sample_data import (
    generate_workload_data, generate_carbon_intensity_forecast,
    generate_cost_forecast, generate_business_metrics,
    generate_optimization_history, calculate_sustainability_metrics
)
from utils.multi_objective import MultiObjectiveOptimizer, BusinessKPICorrelator, SustainabilityScorer, OptimizationAction

try:
    from integrations.claude_ai import ClaudeCloudAssistant, render_ai_chat_interface
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False

# =============================================================================
# CONNECTION STATUS FUNCTIONS
# =============================================================================

def check_aws_connection():
    """Test AWS connection using configured credentials."""
    try:
        import boto3
        
        # Get credentials from secrets
        aws_key = st.secrets.get("aws", {}).get("access_key_id", "")
        aws_secret = st.secrets.get("aws", {}).get("access_key", "") or st.secrets.get("aws", {}).get("secret_access_key", "")
        aws_region = st.secrets.get("aws", {}).get("default_region", "") or st.secrets.get("aws", {}).get("region", "us-east-1")
        
        if not aws_key or not aws_secret:
            return False, "No AWS credentials configured"
        
        # Test connection with STS GetCallerIdentity
        sts = boto3.client(
            'sts',
            aws_access_key_id=aws_key,
            aws_secret_access_key=aws_secret,
            region_name=aws_region
        )
        identity = sts.get_caller_identity()
        account_id = identity.get('Account', 'Unknown')
        return True, f"Connected (Account: ...{account_id[-4:]})"
    except Exception as e:
        error_msg = str(e)
        if "InvalidClientTokenId" in error_msg:
            return False, "Invalid AWS Access Key"
        elif "SignatureDoesNotMatch" in error_msg:
            return False, "Invalid AWS Secret Key"
        elif "credentials" in error_msg.lower():
            return False, "Credential error"
        else:
            return False, f"Error: {error_msg[:30]}"

def check_claude_connection():
    """Test Claude API connection."""
    try:
        import anthropic
        
        api_key = st.secrets.get("anthropic", {}).get("api_key", "")
        
        if not api_key:
            return False, "No API key configured"
        
        if not api_key.startswith("sk-ant"):
            return False, "Invalid key format"
        
        # Test with a minimal API call
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hi"}]
        )
        return True, "Connected"
    except Exception as e:
        error_msg = str(e)
        if "authentication" in error_msg.lower() or "401" in error_msg:
            return False, "Invalid API key"
        elif "rate" in error_msg.lower():
            return True, "Connected (rate limited)"  # Still connected, just rate limited
        else:
            return False, f"Error: {error_msg[:25]}"

def render_connection_status():
    """Render connection status indicators in sidebar."""
    st.sidebar.markdown("### 🔌 Connections")
    
    # Initialize session state for connection status
    if 'aws_status' not in st.session_state:
        st.session_state.aws_status = (None, "Not checked")
    if 'claude_status' not in st.session_state:
        st.session_state.claude_status = (None, "Not checked")
    if 'data_mode' not in st.session_state:
        st.session_state.data_mode = "demo"  # Default to demo
    
    col1, col2 = st.sidebar.columns(2)
    
    # AWS Status
    with col1:
        aws_connected, aws_msg = st.session_state.aws_status
        if aws_connected is True:
            st.markdown(f'''
            <div style="background:#064e3b;border:1px solid #10b981;border-radius:8px;padding:8px;text-align:center;">
                <span style="color:#10b981;font-size:1.2rem;">●</span>
                <span style="color:#10b981;font-weight:600;font-size:0.8rem;"> AWS</span>
                <p style="color:#6ee7b7;font-size:0.7rem;margin:4px 0 0 0;">{aws_msg}</p>
            </div>
            ''', unsafe_allow_html=True)
        elif aws_connected is False:
            st.markdown(f'''
            <div style="background:#450a0a;border:1px solid #ef4444;border-radius:8px;padding:8px;text-align:center;">
                <span style="color:#ef4444;font-size:1.2rem;">●</span>
                <span style="color:#ef4444;font-weight:600;font-size:0.8rem;"> AWS</span>
                <p style="color:#fca5a5;font-size:0.7rem;margin:4px 0 0 0;">{aws_msg}</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="background:#1e293b;border:1px solid #475569;border-radius:8px;padding:8px;text-align:center;">
                <span style="color:#94a3b8;font-size:1.2rem;">○</span>
                <span style="color:#94a3b8;font-weight:600;font-size:0.8rem;"> AWS</span>
                <p style="color:#64748b;font-size:0.7rem;margin:4px 0 0 0;">Not checked</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Claude Status
    with col2:
        claude_connected, claude_msg = st.session_state.claude_status
        if claude_connected is True:
            st.markdown(f'''
            <div style="background:#064e3b;border:1px solid #10b981;border-radius:8px;padding:8px;text-align:center;">
                <span style="color:#10b981;font-size:1.2rem;">●</span>
                <span style="color:#10b981;font-weight:600;font-size:0.8rem;"> Claude</span>
                <p style="color:#6ee7b7;font-size:0.7rem;margin:4px 0 0 0;">{claude_msg}</p>
            </div>
            ''', unsafe_allow_html=True)
        elif claude_connected is False:
            st.markdown(f'''
            <div style="background:#450a0a;border:1px solid #ef4444;border-radius:8px;padding:8px;text-align:center;">
                <span style="color:#ef4444;font-size:1.2rem;">●</span>
                <span style="color:#ef4444;font-weight:600;font-size:0.8rem;"> Claude</span>
                <p style="color:#fca5a5;font-size:0.7rem;margin:4px 0 0 0;">{claude_msg}</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown('''
            <div style="background:#1e293b;border:1px solid #475569;border-radius:8px;padding:8px;text-align:center;">
                <span style="color:#94a3b8;font-size:1.2rem;">○</span>
                <span style="color:#94a3b8;font-weight:600;font-size:0.8rem;"> Claude</span>
                <p style="color:#64748b;font-size:0.7rem;margin:4px 0 0 0;">Not checked</p>
            </div>
            ''', unsafe_allow_html=True)
    
    # Test Connection Button
    if st.sidebar.button("🔄 Test Connections", use_container_width=True):
        with st.sidebar:
            with st.spinner("Testing AWS..."):
                st.session_state.aws_status = check_aws_connection()
            with st.spinner("Testing Claude..."):
                st.session_state.claude_status = check_claude_connection()
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Data Mode Toggle
    st.sidebar.markdown("### 📊 Data Source")
    
    aws_connected, _ = st.session_state.aws_status
    
    # Radio button for data mode
    data_mode_options = ["Demo Data", "Live AWS Data"]
    
    # Disable Live option if AWS not connected
    if aws_connected is not True:
        st.sidebar.radio(
            "Select data source:",
            data_mode_options,
            index=0,
            key="data_mode_radio",
            disabled=False,
            help="Connect to AWS first to enable Live Data mode"
        )
        st.session_state.data_mode = "demo"
        st.sidebar.caption("⚠️ Test AWS connection to enable Live Data")
    else:
        selected_mode = st.sidebar.radio(
            "Select data source:",
            data_mode_options,
            index=0 if st.session_state.data_mode == "demo" else 1,
            key="data_mode_radio",
            help="Switch between simulated demo data and real AWS data"
        )
        st.session_state.data_mode = "demo" if selected_mode == "Demo Data" else "live"
    
    # Show current mode indicator
    if st.session_state.data_mode == "live":
        st.sidebar.markdown('''
        <div style="background:#064e3b;border:1px solid #10b981;border-radius:6px;padding:8px;text-align:center;margin-top:8px;">
            <span style="color:#10b981;font-weight:600;">🟢 LIVE MODE ACTIVE</span>
            <p style="color:#6ee7b7;font-size:0.75rem;margin:4px 0 0 0;">Fetching real AWS data</p>
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('''
        <div style="background:#1e293b;border:1px solid #f59e0b;border-radius:6px;padding:8px;text-align:center;margin-top:8px;">
            <span style="color:#f59e0b;font-weight:600;">🟡 DEMO MODE</span>
            <p style="color:#fcd34d;font-size:0.75rem;margin:4px 0 0 0;">Using simulated data</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.sidebar.markdown("---")

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="BOMOCO | Cloud Optimizer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# PROFESSIONAL STYLING - HIGH CONTRAST & VISIBILITY
# =============================================================================

st.markdown("""
<style>
    /* Import Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Force all text to be visible */
    p, span, label, div {
        color: #e2e8f0 !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1e293b !important;
        border-right: 1px solid #334155;
    }
    
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    section[data-testid="stSidebar"] h3 {
        color: #f8fafc !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #475569;
        padding-bottom: 8px;
        margin-top: 16px;
    }
    
    /* Slider Labels - VERY VISIBLE */
    section[data-testid="stSidebar"] .stSlider label p {
        color: #f8fafc !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
    section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
        color: #94a3b8 !important;
    }
    
    /* Metrics - HIGH CONTRAST */
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.75rem !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
    }
    
    [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: #1e293b;
        padding: 8px;
        border-radius: 10px;
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 500;
        border-radius: 6px;
        padding: 10px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background: #7c3aed !important;
        color: #ffffff !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #f8fafc !important;
        background: #334155;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #6366f1 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #8b5cf6 0%, #818cf8 100%) !important;
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4) !important;
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        background: #334155 !important;
        border-color: #475569 !important;
        color: #f8fafc !important;
    }
    
    .stSelectbox label p {
        color: #e2e8f0 !important;
    }
    
    .stMultiSelect > div > div {
        background: #334155 !important;
        border-color: #475569 !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background: #334155 !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    .streamlit-expanderContent {
        background: #1e293b !important;
        border: 1px solid #475569 !important;
        border-top: none !important;
    }
    
    /* Info boxes */
    .stAlert {
        background: #334155 !important;
        color: #e2e8f0 !important;
        border: 1px solid #475569 !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }
    
    /* Custom Classes */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    
    .subtitle {
        color: #94a3b8 !important;
        font-size: 1rem;
        margin-bottom: 24px;
    }
    
    .badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 16px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-demo {
        background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
        color: white !important;
    }
    
    .badge-live {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white !important;
    }
    
    .card {
        background: #334155;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #475569;
    }
    
    .card h4 {
        color: #f8fafc !important;
        margin: 0 0 12px 0;
        font-size: 1rem;
    }
    
    .card p {
        color: #cbd5e1 !important;
        margin: 4px 0;
        font-size: 0.9rem;
    }
    
    .section-title {
        color: #f8fafc !important;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid #7c3aed;
    }
    
    .weight-display {
        display: flex;
        align-items: center;
        margin: 6px 0;
        gap: 8px;
    }
    
    .weight-label {
        width: 80px;
        color: #94a3b8 !important;
        font-size: 0.8rem;
    }
    
    .weight-bar-bg {
        flex: 1;
        height: 6px;
        background: #1e293b;
        border-radius: 3px;
    }
    
    .weight-bar-fill {
        height: 100%;
        border-radius: 3px;
    }
    
    .weight-value {
        width: 40px;
        text-align: right;
        color: #f8fafc !important;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================

if 'workloads' not in st.session_state:
    st.session_state.workloads = generate_workload_data(50)  # Start with demo
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = MultiObjectiveOptimizer()
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = []
if 'ai_assistant' not in st.session_state and CLAUDE_AVAILABLE:
    st.session_state.ai_assistant = ClaudeCloudAssistant()
if 'data_mode' not in st.session_state:
    st.session_state.data_mode = "demo"
if 'last_data_mode' not in st.session_state:
    st.session_state.last_data_mode = "demo"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def is_demo_mode():
    """Check if running in demo mode based on toggle."""
    try:
        return st.session_state.get('data_mode', 'demo') == 'demo'
    except:
        return True

@st.cache_data(ttl=300)
def get_business_metrics(days=30):
    return generate_business_metrics(days)

def get_aws_credentials():
    """Get AWS credentials from secrets."""
    try:
        aws_key = st.secrets.get("aws", {}).get("access_key_id", "")
        aws_secret = st.secrets.get("aws", {}).get("access_key", "") or st.secrets.get("aws", {}).get("secret_access_key", "")
        aws_region = st.secrets.get("aws", {}).get("default_region", "") or st.secrets.get("aws", {}).get("region", "us-east-1")
        return aws_key, aws_secret, aws_region
    except:
        return "", "", "us-east-1"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_aws_data():
    """Fetch real data from AWS."""
    try:
        import boto3
        from datetime import datetime, timedelta
        
        aws_key, aws_secret, aws_region = get_aws_credentials()
        
        if not aws_key or not aws_secret:
            return None, "No credentials"
        
        workloads = []
        
        # Fetch EC2 instances from multiple regions
        regions_to_check = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
        
        for region in regions_to_check:
            try:
                ec2 = boto3.client(
                    'ec2',
                    aws_access_key_id=aws_key,
                    aws_secret_access_key=aws_secret,
                    region_name=region
                )
                
                response = ec2.describe_instances()
                
                for reservation in response.get('Reservations', []):
                    for instance in reservation.get('Instances', []):
                        if instance.get('State', {}).get('Name') == 'running':
                            # Get instance name from tags
                            name = "Unnamed"
                            workload_type = "general"
                            business_unit = "Unknown"
                            
                            for tag in instance.get('Tags', []):
                                if tag['Key'] == 'Name':
                                    name = tag['Value']
                                elif tag['Key'] == 'WorkloadType':
                                    workload_type = tag['Value']
                                elif tag['Key'] == 'BusinessUnit':
                                    business_unit = tag['Value']
                            
                            instance_type = instance.get('InstanceType', 't3.medium')
                            
                            # Estimate cost based on instance type
                            hourly_costs = {
                                't3.micro': 0.0104, 't3.small': 0.0208, 't3.medium': 0.0416,
                                't3.large': 0.0832, 't3.xlarge': 0.1664, 't3.2xlarge': 0.3328,
                                'm5.large': 0.096, 'm5.xlarge': 0.192, 'm5.2xlarge': 0.384,
                                'c5.large': 0.085, 'c5.xlarge': 0.17, 'c5.2xlarge': 0.34,
                                'r5.large': 0.126, 'r5.xlarge': 0.252, 'r5.2xlarge': 0.504,
                            }
                            hourly = hourly_costs.get(instance_type, 0.10)
                            monthly_cost = hourly * 730  # ~730 hours/month
                            
                            workloads.append({
                                'workload_id': instance.get('InstanceId'),
                                'workload_name': name,
                                'workload_type': workload_type,
                                'business_unit': business_unit,
                                'region': region,
                                'instance_type': instance_type,
                                'instance_count': 1,
                                'cpu_utilization': np.random.uniform(0.2, 0.8),  # Would need CloudWatch for real data
                                'memory_utilization': np.random.uniform(0.3, 0.7),
                                'monthly_cost': monthly_cost,
                                'deferability_score': np.random.uniform(0.1, 0.9),
                                'revenue_correlation': np.random.uniform(0.3, 0.8),
                                'slo_latency_ms': 200,
                                'current_latency_ms': 150,
                            })
            except Exception as e:
                continue  # Skip regions that fail
        
        if workloads:
            return pd.DataFrame(workloads), f"Found {len(workloads)} instances"
        else:
            return None, "No running instances found"
            
    except Exception as e:
        return None, f"Error: {str(e)[:50]}"

def get_workload_data():
    """Get workload data based on current mode."""
    if st.session_state.get('data_mode') == 'live':
        aws_connected, _ = st.session_state.get('aws_status', (False, ""))
        if aws_connected:
            live_data, msg = fetch_live_aws_data()
            if live_data is not None and len(live_data) > 0:
                return live_data
            else:
                st.warning(f"⚠️ Could not fetch live data: {msg}. Using demo data.")
    
    # Fall back to demo data
    return generate_workload_data(50)

# =============================================================================
# COMPONENTS
# =============================================================================

def render_header():
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown('<h1 class="main-title">BOMOCO</h1>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Business-Outcome-Aware Multi-Objective Cloud Optimizer</p>', unsafe_allow_html=True)
    with c2:
        demo = is_demo_mode()
        badge = "badge-demo" if demo else "badge-live"
        text = "DEMO MODE" if demo else "LIVE DATA"
        st.markdown(f'''
        <div style="text-align:right;padding-top:8px;">
            <span class="badge {badge}">{text}</span>
            <p style="color:#10b981;font-size:0.85rem;margin:8px 0 4px 0;">● System Active</p>
            <p style="color:#64748b;font-size:0.8rem;margin:0;">{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        </div>
        ''', unsafe_allow_html=True)

def render_sidebar():
    # Connection Status First
    render_connection_status()
    
    st.sidebar.markdown("### ⚙️ Optimization Weights")
    
    cost_w = st.sidebar.slider("💰 Cost", 0.0, 1.0, 0.35, 0.05)
    carbon_w = st.sidebar.slider("🌱 Carbon", 0.0, 1.0, 0.25, 0.05)
    water_w = st.sidebar.slider("💧 Water", 0.0, 1.0, 0.10, 0.05)
    perf_w = st.sidebar.slider("⚡ Performance", 0.0, 1.0, 0.20, 0.05)
    biz_w = st.sidebar.slider("📈 Business KPI", 0.0, 1.0, 0.10, 0.05)
    
    total = cost_w + carbon_w + water_w + perf_w + biz_w
    if total > 0:
        norm = {
            "cost": cost_w/total, "carbon": carbon_w/total, "water": water_w/total,
            "performance": perf_w/total, "business_kpi": biz_w/total
        }
        st.session_state.optimizer.set_weights(norm)
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("**Normalized Weights:**")
        
        colors = {"cost": "#10b981", "carbon": "#22c55e", "water": "#0ea5e9", "performance": "#f59e0b", "business_kpi": "#a78bfa"}
        labels = {"cost": "Cost", "carbon": "Carbon", "water": "Water", "performance": "Perf", "business_kpi": "Business"}
        
        for k, v in norm.items():
            st.sidebar.markdown(f'''
            <div class="weight-display">
                <span class="weight-label">{labels[k]}</span>
                <div class="weight-bar-bg"><div class="weight-bar-fill" style="width:{v*100}%;background:{colors[k]};"></div></div>
                <span class="weight-value">{v:.0%}</span>
            </div>
            ''', unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Actions")
    
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.session_state.workloads = get_workload_data()
        st.session_state.recommendations = []
        st.rerun()
    
    if st.sidebar.button("⚡ Run Optimization", use_container_width=True, type="primary"):
        with st.spinner("Analyzing..."):
            cf = generate_carbon_intensity_forecast(48, "us-east-1")
            costf = generate_cost_forecast(24, "us-east-1")
            recs = st.session_state.optimizer.generate_recommendations(st.session_state.workloads, cf, costf, 20)
            st.session_state.recommendations = recs
        st.success(f"✅ {len(recs)} recommendations generated!")
        st.rerun()
    
    st.sidebar.markdown("---")
    if is_demo_mode():
        st.sidebar.info("📊 Running with simulated data")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown('<p style="text-align:center;color:#64748b;font-size:0.75rem;">BOMOCO v1.0 | Patent Pending</p>', unsafe_allow_html=True)

def render_kpis(workloads, biz_metrics):
    metrics = calculate_sustainability_metrics(workloads)
    c1, c2, c3, c4, c5 = st.columns(5)
    
    c1.metric("Monthly Spend", f"${metrics['total_monthly_cost']:,.0f}", f"-{metrics['rightsizing_opportunity_percent']:.0f}% opportunity", delta_color="inverse")
    c2.metric("Carbon Footprint", f"{metrics['total_monthly_carbon_kg']:,.0f} kg", f"-{metrics['carbon_shift_opportunity_percent']:.0f}% potential", delta_color="inverse")
    c3.metric("Water Usage", f"{metrics['total_monthly_water_liters']/1000:,.0f} kL", "-12% vs benchmark", delta_color="inverse")
    
    if not biz_metrics.empty:
        rev = biz_metrics.iloc[-1]['daily_revenue']
        prev = biz_metrics.iloc[-2]['daily_revenue'] if len(biz_metrics) > 1 else rev
        c4.metric("Daily Revenue", f"${rev:,.0f}", f"{((rev-prev)/prev)*100:+.1f}%")
        c5.metric("Conversion Rate", f"{biz_metrics.iloc[-1]['conversion_rate']:.2f}%", "+0.3%")

def render_map(workloads):
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
                        hover_name="name", color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"], size_max=40)
    fig.update_layout(
        title=dict(text="Global Carbon Intensity & Workload Distribution", font=dict(color="#f8fafc", size=14)),
        geo=dict(showland=True, landcolor="#1e293b", showocean=True, oceancolor="#0f172a",
                showcountries=True, countrycolor="#475569", bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"), height=400, margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(
            title=dict(text="gCO₂/kWh", font=dict(color="#e2e8f0")),
            tickfont=dict(color="#e2e8f0")
        )
    )
    st.plotly_chart(fig, use_container_width=True)

def render_forecast(region):
    forecast = generate_carbon_intensity_forecast(48, region)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast["timestamp"], y=forecast["carbon_intensity_gco2_kwh"],
                            mode="lines", line=dict(color="#10b981", width=2), fill="tozeroy", fillcolor="rgba(16,185,129,0.15)"))
    fig.update_layout(
        title=dict(text=f"48-Hour Carbon Forecast - {region}", font=dict(color="#f8fafc", size=14)),
        xaxis=dict(gridcolor="rgba(71,85,105,0.5)", tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="rgba(71,85,105,0.5)", tickfont=dict(color="#94a3b8"), title="gCO₂/kWh"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"), height=280, margin=dict(l=50, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

def render_pareto(recs):
    if not recs:
        st.info("🎯 Click **Run Optimization** to see trade-off analysis")
        return
    
    cost_imp = [-r.cost_impact for r in recs]
    carbon_imp = [-r.carbon_impact for r in recs]
    colors = {"rightsize_down": "#10b981", "spot_conversion": "#0ea5e9", "carbon_shift": "#22c55e",
             "region_migration": "#a78bfa", "rightsize_up": "#f59e0b"}
    c = [colors.get(r.action_type, "#64748b") for r in recs]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cost_imp, y=carbon_imp, mode="markers",
                            marker=dict(size=14, color=c, line=dict(width=2, color="white")),
                            text=[r.description[:40] for r in recs],
                            hovertemplate="<b>%{text}</b><br>Cost: %{x:.1f}%<br>Carbon: %{y:.1f}%<extra></extra>"))
    fig.update_layout(
        title=dict(text="Cost vs Carbon Trade-offs", font=dict(color="#f8fafc", size=14)),
        xaxis=dict(title="Cost Savings (%)", gridcolor="rgba(71,85,105,0.5)", tickfont=dict(color="#94a3b8"), titlefont=dict(color="#e2e8f0")),
        yaxis=dict(title="Carbon Reduction (%)", gridcolor="rgba(71,85,105,0.5)", tickfont=dict(color="#94a3b8"), titlefont=dict(color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"), height=400, margin=dict(l=60, r=20, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)

def render_recommendations(recs):
    if not recs:
        st.info("🎯 Click **Run Optimization** to generate recommendations")
        return
    
    impact = st.session_state.optimizer.estimate_total_impact(recs)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monthly Savings", f"${impact['total_monthly_savings']:,.0f}")
    c2.metric("Cost Reduction", f"{impact['total_cost_reduction_percent']:.1f}%")
    c3.metric("Carbon Reduction", f"{impact['total_carbon_reduction_percent']:.1f}%")
    c4.metric("Avg Confidence", f"{impact['avg_confidence']*100:.0f}%")
    
    st.markdown("---")
    
    for i, r in enumerate(recs[:10]):
        risk_colors = {"low": "#10b981", "medium": "#f59e0b", "high": "#ef4444"}
        with st.expander(f"**{i+1}. {r.action_type.replace('_',' ').title()}** — ${r.estimated_savings_monthly:,.0f}/mo — Score: {r.composite_score:.3f}", expanded=(i<3)):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{r.description}**")
                st.markdown(f"Workload: `{r.workload_id}`")
                
                impacts = f"💰 Cost: {r.cost_impact:+.1f}% | 🌱 Carbon: {r.carbon_impact:+.1f}% | ⚡ Perf: {r.performance_impact:+.1f}%"
                st.markdown(impacts)
            
            with col2:
                st.markdown(f"""
                <div class="card">
                    <p><strong>Confidence:</strong> {r.confidence:.0%}</p>
                    <p><strong>Risk:</strong> <span style="color:{risk_colors.get(r.risk_level,'#94a3b8')}">{r.risk_level.upper()}</span></p>
                    <p><strong>Effort:</strong> {r.implementation_effort.title()}</p>
                </div>
                """, unsafe_allow_html=True)

def render_business_intel(biz_metrics, opt_history):
    if biz_metrics.empty:
        return
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=biz_metrics["date"], y=biz_metrics["daily_revenue"], name="Revenue", line=dict(color="#a78bfa", width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=biz_metrics["date"], y=biz_metrics["infrastructure_health_score"], name="Infra Health", line=dict(color="#10b981", width=2, dash="dot")), secondary_y=True)
    fig.update_layout(
        title=dict(text="Business KPI Correlation", font=dict(color="#f8fafc", size=14)),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"), height=320,
        xaxis=dict(gridcolor="rgba(71,85,105,0.5)", tickfont=dict(color="#94a3b8")),
        yaxis=dict(gridcolor="rgba(71,85,105,0.5)", tickfont=dict(color="#94a3b8"), title="Revenue ($)"),
        yaxis2=dict(tickfont=dict(color="#94a3b8"), title="Health %"),
        legend=dict(font=dict(color="#e2e8f0")), margin=dict(l=60, r=60, t=40, b=40))
    st.plotly_chart(fig, use_container_width=True)
    
    correlator = BusinessKPICorrelator()
    corr = correlator.analyze_correlations(opt_history, biz_metrics)
    
    st.markdown("**Learned Correlations** *(Patent-Pending)*")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="card" style="text-align:center;"><p style="font-size:1.8rem;font-weight:700;color:#a78bfa!important;margin:0;">{corr["performance_to_revenue"]:.0%}</p><p style="color:#94a3b8!important;margin:4px 0 0 0;">Performance → Revenue</p></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card" style="text-align:center;"><p style="font-size:1.8rem;font-weight:700;color:#ef4444!important;margin:0;">{corr["latency_to_conversions"]:.0%}</p><p style="color:#94a3b8!important;margin:4px 0 0 0;">Latency → Conversions</p></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card" style="text-align:center;"><p style="font-size:1.8rem;font-weight:700;color:#22c55e!important;margin:0;">{corr["carbon_to_brand"]:.0%}</p><p style="color:#94a3b8!important;margin:4px 0 0 0;">Sustainability → Brand</p></div>', unsafe_allow_html=True)

# =============================================================================
# MAIN
# =============================================================================

def main():
    render_header()
    render_sidebar()
    
    # Check if data mode changed and reload data
    if st.session_state.get('data_mode') != st.session_state.get('last_data_mode'):
        st.session_state.last_data_mode = st.session_state.get('data_mode')
        st.cache_data.clear()
        st.session_state.workloads = get_workload_data()
        st.session_state.recommendations = []
    
    biz_metrics = get_business_metrics(30)
    opt_history = generate_optimization_history(20)
    
    render_kpis(st.session_state.workloads, biz_metrics)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🌍 Sustainability", "📊 Recommendations", "🤖 AI Assistant", "📈 Business Intel", "🔮 Forecasts", "📋 Inventory"])
    
    with tab1:
        c1, c2 = st.columns([2, 1])
        with c1:
            render_map(st.session_state.workloads)
        with c2:
            region = st.selectbox("Region", list(AWS_REGIONS.keys()), format_func=lambda x: f"{AWS_REGIONS[x]['name']} ({x})")
            wl = st.session_state.workloads[st.session_state.workloads["region"] == region]
            grid = AWS_REGIONS[region]["grid"]
            st.markdown(f'''
            <div class="card">
                <h4>{AWS_REGIONS[region]['name']}</h4>
                <p><strong>Grid:</strong> {grid}</p>
                <p><strong>Carbon:</strong> {GRID_CARBON_BASELINE.get(grid, 400)} gCO₂/kWh</p>
                <p><strong>Workloads:</strong> {len(wl)}</p>
                <p><strong>Cost:</strong> ${wl["monthly_cost"].sum():,.0f}/mo</p>
            </div>
            ''', unsafe_allow_html=True)
        render_forecast(region)
    
    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<p class="section-title">🎯 Recommendations</p>', unsafe_allow_html=True)
            render_recommendations(st.session_state.recommendations)
        with c2:
            st.markdown('<p class="section-title">📊 Trade-offs</p>', unsafe_allow_html=True)
            render_pareto(st.session_state.recommendations)
    
    with tab3:
        if CLAUDE_AVAILABLE:
            render_ai_chat_interface(st.session_state.ai_assistant, st.session_state.workloads, st.session_state.recommendations)
        else:
            st.markdown('<p class="section-title">🤖 AI-Powered Insights</p>', unsafe_allow_html=True)
            st.info("Add your Anthropic API key to enable Claude AI features.")
            st.markdown("""
            **Enable AI Features:**
            1. Get API key from [console.anthropic.com](https://console.anthropic.com)
            2. Add to **App Settings → Secrets**:
            ```toml
            [anthropic]
            api_key = "sk-ant-..."
            ```
            """)
    
    with tab4:
        st.markdown('<p class="section-title">📈 Business KPI Correlation</p>', unsafe_allow_html=True)
        render_business_intel(biz_metrics, opt_history)
    
    with tab5:
        st.markdown('<p class="section-title">🔮 Forecasts</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            cf = generate_cost_forecast(24, "us-east-1")
            fig = px.line(cf, x="timestamp", y="spot_discount_percentage", title="Spot Discount Forecast")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"), height=280,
                             xaxis=dict(gridcolor="rgba(71,85,105,0.5)"), yaxis=dict(gridcolor="rgba(71,85,105,0.5)"))
            fig.update_traces(line_color="#3b82f6")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            forecasts = pd.concat([generate_carbon_intensity_forecast(24, r).assign(region_name=AWS_REGIONS[r]["name"]) for r in ["us-east-1", "us-west-2", "eu-west-1"]])
            fig = px.line(forecasts, x="timestamp", y="carbon_intensity_gco2_kwh", color="region_name", title="Carbon by Region")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0"), height=280,
                             xaxis=dict(gridcolor="rgba(71,85,105,0.5)"), yaxis=dict(gridcolor="rgba(71,85,105,0.5)"), legend=dict(font=dict(color="#e2e8f0")))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.markdown('<p class="section-title">📋 Workload Inventory</p>', unsafe_allow_html=True)
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
