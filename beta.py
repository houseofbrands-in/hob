import streamlit as st
import pandas as pd
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- USE BETA LOGIC ---
from ui.styles import load_custom_css
import core.database as db
import core.logic_beta as logic_beta # <--- Using Beta Logic
import ui.components as ui

st.set_page_config(page_title="HOB OS - BETA LABS", layout="wide", page_icon="🧪")
load_custom_css()

# ... (Authentication code same as main.py - Copy lines 23-50 from main.py if needed) ...
# For brevity, I assume you are logged in or can copy the login block.
if "logged_in" not in st.session_state: st.session_state.logged_in = True; st.session_state.username = "Beta Tester"; st.session_state.user_role = "admin"

clients = logic_beta.init_clients() # Returns (gpt, gemini, openrouter)

with st.sidebar:
    st.markdown("### 🧪 HOB BETA LABS")
    st.warning("EXPERIMENTAL BUILD")
    selected_mp = st.selectbox("Marketplace", ["Myntra", "Flipkart", "Ajio", "Amazon", "Nykaa"])
    mp_cats = db.get_categories_for_marketplace(selected_mp)
    concurrency_limit = st.slider("Threads", 1, 5, 3)

tab_run, tab_setup = st.tabs(["🧪 Experiment", "⚙️ Config"])

with tab_run:
    with st.expander("📂 Input", expanded=True):
        if not mp_cats: st.stop()
        run_cat = st.selectbox("Category", mp_cats)
        config = db.load_config(selected_mp, run_cat)
        active_kws = db.get_seo(selected_mp, run_cat)
        input_file = st.file_uploader("Upload Excel", type=["xlsx"])

    if input_file and config:
        df_input = pd.read_excel(input_file)
        
        st.markdown("#### 🧪 Engine Selection")
        c1, c2, c3 = st.columns(3)
        with c1:
            # THE NEW DROPDOWN
            arch_mode = st.selectbox("Select Model", [
                "🚀 Economy (DeepSeek)", 
                "💎 Precision (Claude 3.5)", 
                "⚖️ Standard (Gemini)"
            ])
        
        with c2:
            all_cols = df_input.columns.tolist()
            img_col = st.selectbox("Image Col", all_cols)
        with c3:
            sku_col = st.selectbox("SKU Col", all_cols)

        # ... (Rest of the Execution Logic is identical to main.py, just calling logic_beta) ...
        # Copy the START ENGINE block from main.py but change:
        # logic.process_row_workflow -> logic_beta.process_row_workflow
        # Pass 'clients' tuple instead of just 'client'
        
        if st.button("▶️ RUN BETA TEST", type="primary"):
            st.session_state.gen_results = []
            results_container = st.container()
            
            # Deduplication Logic
            valid_rows = df_input[df_input[img_col].notna()]
            grouped = valid_rows.groupby(img_col)
            tasks = [group.iloc[0] for _, group in grouped]
            
            ai_cache = {}
            
            with ThreadPoolExecutor(max_workers=concurrency_limit) as executor:
                future_to_url = {
                    executor.submit(
                        logic_beta.process_row_workflow, # <--- CALLING BETA
                        row, img_col, sku_col, config, clients, arch_mode, active_kws, selected_mp
                    ): row[img_col] 
                    for row in tasks
                }
                
                for future in as_completed(future_to_url):
                    res = future.result()
                    if res['success']:
                        ai_cache[future_to_url[future]] = res['ai_data']
                        with results_container:
                            st.success(f"Generated: {res['sku']}")
                            st.json(res['ai_data']) # Debug View
                    else:
                        st.error(f"Failed: {res['error']}")

            # Reconstruct and Download (Same as main)
            final_rows = []
            for idx, row in valid_rows.iterrows():
                u = str(row[img_col]).strip()
                if u in ai_cache:
                    final_rows.append(logic_beta.merge_ai_data_to_row(row, ai_cache[u], config))
                else:
                    final_rows.append(logic_beta.merge_ai_data_to_row(row, {}, config))
            
            df_final = pd.DataFrame(final_rows)
            st.dataframe(df_final)
