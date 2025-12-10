import streamlit as st

def kpi_card(title, value, icon="⚡", color="blue"):
    """
    Renders a glassmorphism KPI card.
    color options: blue, purple, green, red
    """
    colors = {
        "blue": "linear-gradient(135deg, rgba(79, 70, 229, 0.1) 0%, rgba(79, 70, 229, 0.05) 100%)",
        "purple": "linear-gradient(135deg, rgba(124, 58, 237, 0.1) 0%, rgba(124, 58, 237, 0.05) 100%)",
        "green": "linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%)",
        "red": "linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%)"
    }
    
    border_colors = {
        "blue": "rgba(79, 70, 229, 0.3)",
        "purple": "rgba(124, 58, 237, 0.3)",
        "green": "rgba(16, 185, 129, 0.3)",
        "red": "rgba(239, 68, 68, 0.3)"
    }
    
    bg = colors.get(color, colors["blue"])
    border = border_colors.get(color, border_colors["blue"])
    
    html = f"""
    <div style="
        background: {bg};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 15px;
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 10px;
        backdrop-filter: blur(10px);
    ">
        <div style="
            font-size: 24px;
            background: rgba(255,255,255,0.05);
            width: 45px;
            height: 45px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
        ">
            {icon}
        </div>
        <div>
            <div style="font-size: 12px; text-transform: uppercase; letter-spacing: 1px; color: #888; margin-bottom: 4px;">{title}</div>
            <div style="font-size: 20px; font-weight: 700; color: #fff;">{value}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def status_badge(status_text, type="success"):
    """
    Renders a small pill badge.
    type: success, error, warning, neutral
    """
    c_map = {
        "success": {"bg": "rgba(16, 185, 129, 0.2)", "c": "#34d399", "b": "rgba(16, 185, 129, 0.3)"},
        "error":   {"bg": "rgba(239, 68, 68, 0.2)", "c": "#f87171", "b": "rgba(239, 68, 68, 0.3)"},
        "warning": {"bg": "rgba(245, 158, 11, 0.2)", "c": "#fbbf24", "b": "rgba(245, 158, 11, 0.3)"},
        "neutral": {"bg": "rgba(255, 255, 255, 0.1)", "c": "#e5e7eb", "b": "rgba(255, 255, 255, 0.2)"},
    }
    style = c_map.get(type, c_map["neutral"])
    
    html = f"""
    <span style="
        background-color: {style['bg']};
        color: {style['c']};
        border: 1px solid {style['b']};
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
    ">{status_text}</span>
    """
    st.markdown(html, unsafe_allow_html=True)
