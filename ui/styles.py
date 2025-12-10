import streamlit as st

def load_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
        html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
        
        /* Hiding Streamlit Branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .block-container {padding-top: 2rem; padding-bottom: 5rem;}
        
        /* Glassmorphism Containers */
        div[data-testid="stExpander"], div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            backdrop-filter: blur(10px);
            color: white;
            transition: transform 0.2s;
        }
        
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: rgba(255, 255, 255, 0.3);
        }
        
        /* Input Fields */
        .stTextInput input, .stSelectbox, .stNumberInput input {
            background-color: #0E1117 !important;
            color: white !important;
            border-radius: 8px !important;
        }
        
        /* Primary Buttons */
        button[kind="primary"] {
            background: linear-gradient(90deg, #2b5876 0%, #4e4376 100%);
            border: none;
            color: white;
            font-weight: 600;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            transition: all 0.3s ease;
        }
        button[kind="primary"]:hover {box-shadow: 0 0 15px rgba(78, 67, 118, 0.6);}
        
        /* Progress Bar */
        .stProgress > div > div > div > div {background-image: linear-gradient(to right, #00c6ff, #0072ff);}
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {gap: 10px;}
        .stTabs [data-baseweb="tab"] {
            height: 50px; white-space: pre-wrap; background-color: rgba(255,255,255,0.02);
            border-radius: 8px; padding: 0 20px; color: #ccc;
        }
        .stTabs [aria-selected="true"] {background-color: rgba(255,255,255,0.1); color: white; font-weight: bold;}
        </style>
    """, unsafe_allow_html=True)