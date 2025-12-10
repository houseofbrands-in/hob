import streamlit as st

def load_custom_css():
    st.markdown("""
        <style>
        /* --- FONTS & BASICS --- */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            background-color: #050505; 
            color: #e0e0e0;
        }

        /* --- HIDE STREAMLIT BRANDING --- */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* --- LAYOUT ADJUSTMENTS --- */
        .block-container {
            padding-top: 1.5rem; 
            padding-bottom: 5rem;
            max-width: 95% !important;
        }

        /* --- SIDEBAR --- */
        section[data-testid="stSidebar"] {
            background-color: #0a0a0a;
            border-right: 1px solid #1f1f1f;
        }
        
        /* --- INPUT FIELDS (Text, Select, Number) --- */
        .stTextInput input, .stSelectbox div[data-baseweb="select"], .stNumberInput input {
            background-color: #111 !important;
            color: white !important;
            border: 1px solid #333 !important;
            border-radius: 6px !important;
        }
        .stTextInput input:focus, .stSelectbox div[data-baseweb="select"]:focus {
            border-color: #4f46e5 !important; /* Indigo accent */
            box-shadow: 0 0 0 1px #4f46e5 !important;
        }

        /* --- BUTTONS --- */
        button[kind="primary"] {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            border: 1px solid rgba(255,255,255,0.1);
            color: white;
            font-weight: 600;
            padding: 0.5rem 1rem;
            border-radius: 6px;
            transition: all 0.2s ease;
            box-shadow: 0 4px 14px 0 rgba(124, 58, 237, 0.3);
        }
        button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px 0 rgba(124, 58, 237, 0.5);
            border-color: white;
        }
        button[kind="secondary"] {
            background: #1a1a1a;
            border: 1px solid #333;
            color: #ccc;
        }
        button[kind="secondary"]:hover {
            border-color: #666;
            color: white;
        }

        /* --- TABS --- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            white-space: pre-wrap;
            background-color: #111;
            border: 1px solid #222;
            border-radius: 6px;
            padding: 0 20px;
            color: #888;
            font-size: 14px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f1f1f;
            border-color: #4f46e5;
            color: #fff;
        }

        /* --- EXPANDERS & CONTAINERS --- */
        div[data-testid="stExpander"] {
            background: #0e0e0e;
            border: 1px solid #222;
            border-radius: 8px;
        }
        
        /* --- PROGRESS BAR --- */
        .stProgress > div > div > div > div {
            background-image: linear-gradient(90deg, #4f46e5, #ec4899);
        }
        
        /* --- DATAFRAME --- */
        div[data-testid="stDataFrame"] {
            border: 1px solid #333;
            border-radius: 8px;
        }
        
        </style>
    """, unsafe_allow_html=True)
