import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="A/B Test Simulation Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv('simulation_results.csv')
    return df

df = load_data()
control_rate = df['control_reward'].mean()
bandit_rate = df['bandit_reward'].mean()
lift = (bandit_rate - control_rate) / control_rate * 100

# --- Dashboard UI ---
st.title("📈 Dynamic Promotion A/B Test Results")
st.markdown("""
This dashboard shows the results of a simulated A/B test comparing a static promotion strategy (Control Group) 
with a dynamic, personalized strategy powered by a Contextual Multi-Armed Bandit (Treatment Group).
""")

# --- Key Metrics ---
st.header("Overall Performance")
col1, col2, col3 = st.columns(3)
col1.metric("Control Conversion Rate", f"{control_rate:.4f}")
col2.metric("Bandit Conversion Rate", f"{bandit_rate:.4f}")
col3.metric("Performance Lift", f"{lift:.2f}%")

st.markdown("---")

# --- Visualizations ---
st.header("Performance Over Time")

# 1. Cumulative Rewards Plot
fig_cumulative = px.line(df, x=df.index, y=['cumulative_control_reward', 'cumulative_bandit_reward'],
                         title="Cumulative Conversions (Bandit vs. Control)",
                         labels={'value': 'Total Conversions', 'index': 'Users Processed'})
fig_cumulative.update_layout(legend_title_text='Group')
st.plotly_chart(fig_cumulative, use_container_width=True)


# 2. Arm Distribution Plot
st.header("Bandit's Promotion Strategy")
arm_counts = df['bandit_arm'].value_counts().reset_index()
arm_counts.columns = ['Promotion Type', 'Count']
fig_arms = px.bar(arm_counts, x='Promotion Type', y='Count',
                  title="Promotions Chosen by the Bandit",
                  color='Promotion Type')
st.plotly_chart(fig_arms, use_container_width=True)

st.info("Notice how the bandit learned to use different promotions. This is personalization in action!")