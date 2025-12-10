import streamlit as st
import pandas as pd
import time
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- IMPORT MODULES ---
from ui.styles import load_custom_css
import core.database as db
import core.logic_beta as logic_beta 
import ui.components as ui 

# --- PAGE CONFIG ---
st.set_page_config(page_title="HOB OS - BETA LABS", layout="wide", page_icon="🧪")
load_custom_css()

# --- AUTH (Simplified for Beta) ---
if "logged_in" not in st.session_state: 
    # Auto-login for beta testing ease
    st.session_state.logged_in = True
    st.session_state.username = "Beta Tester"
    st.session_state.user_role = "admin"

clients = logic_beta.init_clients() # Returns (gpt, gemini, openrouter)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🧪 HOB BETA LABS")
    st.warning("EXPERIMENTAL BUILD")
    selected_mp = st.selectbox("Marketplace", ["Myntra", "Flipkart", "Ajio", "Amazon", "Nykaa"])
    mp_cats = db.get_categories_for_marketplace(selected_mp)
    concurrency_limit = st.slider("Threads", 1, 5, 3)

tab_run, tab_setup = st.tabs(["🧪 Experiment", "⚙️ Config"])

# --- TAB 1: RUN ---
with tab_run:
    with st.expander("📂 Input & Setup", expanded=True):
        if not mp_cats: st.stop()
        c_setup1, c_setup2 = st.columns(2)
        
        with c_setup1:
            run_cat = st.selectbox("Category", mp_cats)
            config = db.load_config(selected_mp, run_cat)
            active_kws = db.get_seo(selected_mp, run_cat)
        
        with c_setup2:
            input_file = st.file_uploader("Upload Excel", type=["xlsx"])

    if input_file and config:
        df_input = pd.read_excel(input_file)
        
        st.divider()
        st.markdown("#### 🧪 Engine Configuration")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            arch_mode = st.selectbox("Select Model", [
                "🚀 Economy (DeepSeek)", 
                "💎 Precision (Claude 3.5)", 
                "⚖️ Standard (Gemini)"
            ])
        
        with c2:
            all_cols = df_input.columns.tolist()
            # Auto-detect columns
            img_guess = next((c for c in all_cols if "url" in c.lower() or "image" in c.lower()), all_cols[0])
            img_col = st.selectbox("Image Col", all_cols, index=all_cols.index(img_guess))
        with c3:
            sku_guess = next((c for c in all_cols if "sku" in c.lower()), all_cols[0])
            sku_col = st.selectbox("SKU Col", all_cols, index=all_cols.index(sku_guess))

        # Filter valid rows
        valid_rows = df_input[df_input[img_col].notna()]
        
        # --- UI: PRE-FLIGHT STATUS ---
        st.markdown("### 📊 Test Parameters")
        c_kpi1, c_kpi2, c_kpi3 = st.columns(3)
        with c_kpi1: ui.kpi_card("Target Rows", str(len(valid_rows)), "📦", "blue")
        with c_kpi2: ui.kpi_card("Engine", arch_mode.split()[1], "⚙️", "purple")
        with c_kpi3: ui.kpi_card("Est. Cost", "Negligible" if "DeepSeek" in arch_mode else "High", "💰", "green")

        # --- EXECUTION ---
        if st.button("▶️ RUN BETA TEST", type="primary", use_container_width=True):
            
            # 1. IMMEDIATE FEEDBACK
            with st.spinner(f"Initializing {arch_mode}... Connecting to OpenRouter..."):
                st.session_state.beta_results = []
                results_container = st.container()
                
                # Deduplication Logic
                grouped = valid_rows.groupby(img_col)
                tasks = [group.iloc[0] for _, group in grouped]
                total_tasks = len(tasks)
                
                ai_cache = {}
                prog_bar = st.progress(0)
                completed = 0
                
                with ThreadPoolExecutor(max_workers=concurrency_limit) as executor:
                    future_to_url = {
                        executor.submit(
                            logic_beta.process_row_workflow, 
                            row, img_col, sku_col, config, clients, arch_mode, active_kws, selected_mp
                        ): row[img_col] 
                        for row in tasks
                    }
                    
                    for future in as_completed(future_to_url):
                        completed += 1
                        prog_bar.progress(completed / total_tasks)
                        
                        try:
                            res = future.result()
                            if res['success']:
                                ai_cache[future_to_url[future]] = res['ai_data']
                                
                                # UI Feedback (Visual Card)
                                with results_container:
                                    with st.container():
                                        c_img, c_txt = st.columns([1, 5])
                                        with c_img:
                                            if res['img_display']: st.image(res['img_display'], width=60)
                                        with c_txt:
                                            ui.status_badge(f"Generated: {res['sku']}", "success")
                                            with st.expander("View JSON Data"):
                                                st.json(res['ai_data'])
                                        st.divider()
                            else:
                                with results_container:
                                    ui.status_badge(f"Failed: {res['error']}", "error")
                                    
                        except Exception as e:
                            st.error(f"Critical Error: {str(e)}")

                # Reconstruct Rows (Sync Logic)
                final_rows = []
                for idx, row in valid_rows.iterrows():
                    u = str(row[img_col]).strip()
                    if u in ai_cache:
                        final_rows.append(logic_beta.merge_ai_data_to_row(row, ai_cache[u], config))
                    else:
                        final_rows.append(logic_beta.merge_ai_data_to_row(row, {}, config))
                
                # SAVE TO STATE (So download button stays)
                st.session_state.beta_results = final_rows
                st.success("✅ Test Complete! Scroll down to download.")

        # --- DOWNLOAD BLOCK (Outside the button, persistent) ---
        if "beta_results" in st.session_state and len(st.session_state.beta_results) > 0:
            st.divider()
            st.markdown("### 📥 Test Results")
            
            df_final = pd.DataFrame(st.session_state.beta_results)
            st.dataframe(df_final, use_container_width=True)
            
            output_gen = BytesIO()
            with pd.ExcelWriter(output_gen, engine='xlsxwriter') as writer: 
                df_final.to_excel(writer, index=False)
            
            st.download_button(
                label="⬇️ Download Beta Result (.xlsx)", 
                data=output_gen.getvalue(), 
                file_name=f"BETA_RESULT_{arch_mode.split()[1]}.xlsx", 
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                type="primary",
                use_container_width=True
            )
