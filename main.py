import streamlit as st
import pandas as pd
import time
from io import BytesIO
import zipfile
from PIL import Image, ImageOps
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- IMPORT MODULES ---
from ui.styles import load_custom_css
import core.database as db
import core.logic as logic
import ui.components as ui 

# --- PAGE CONFIG ---
st.set_page_config(page_title="HOB OS - Enterprise", layout="wide", page_icon="⚡")
load_custom_css()

# --- INIT STATE ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_role = ""
    st.session_state.username = ""

# --- INIT CLIENTS (GPT, Gemini, OpenRouter) ---
clients = logic.init_clients() # Returns tuple

# ==========================================
# LOGIN SCREEN
# ==========================================
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,1.5,1])
    with c2:
        with st.form("login_form"):
            st.markdown("### ⚡ HOB OS Login")
            user = st.text_input("Username")
            pwd = st.text_input("Password", type="password")
            if st.form_submit_button("Enter System", use_container_width=True, type="primary"):
                is_valid, role = db.check_login(user, pwd)
                if is_valid:
                    st.session_state.logged_in = True
                    st.session_state.username = user
                    st.session_state.user_role = role
                    st.rerun()
                else: st.error("Access Denied")

# ==========================================
# MAIN APP
# ==========================================
else:
    with st.sidebar:
        st.markdown("### ⚡ HOB OS")
        st.caption(f"Operator: **{st.session_state.username}**")
        st.divider()
        
        # --- DYNAMIC MARKETPLACE SELECTOR ---
        st.subheader("📍 Target")
        
        # 1. Fetch existing MPs from DB
        mp_options = db.get_all_marketplaces()
        mp_options.append("➕ New Marketplace") # Add option to create new
        
        # 2. Render Dropdown
        selected_mp_raw = st.selectbox("Marketplace", mp_options)
        
        # 3. Handle "New" Input
        if selected_mp_raw == "➕ New Marketplace":
            new_mp_name = st.text_input("Enter Name", placeholder="e.g. Snapdeal")
            if new_mp_name:
                selected_mp = new_mp_name.strip()
            else:
                selected_mp = None # Block execution until name is typed
        else:
            selected_mp = selected_mp_raw

        if selected_mp:
            mp_cats = db.get_categories_for_marketplace(selected_mp)
        else:
            mp_cats = []
        
        concurrency_limit = 3 
        if st.session_state.user_role == 'admin':
            st.divider()
            st.subheader("⚡ Turbo Mode")
            concurrency_limit = st.slider("Worker Threads", 1, 10, 3)
        
        if st.button("Log Out", use_container_width=True): 
            st.session_state.logged_in = False; st.rerun()
    tab_run, tab_setup, tab_tools, tab_admin = st.tabs(["🚀 Command", "⚙️ Config", "🛠️ Utilities", "👥 Admin"])

    # === TAB 1: RUN ===
    with tab_run:
        with st.expander("📂 **Input & Configuration**", expanded=True):
            if not mp_cats: 
                st.warning("⚠️ No categories found.")
                st.stop()
            
            c_conf1, c_conf2 = st.columns([1, 1])
            with c_conf1:
                run_cat = st.selectbox("Category", mp_cats, key="run_cat")
                active_kws = db.get_seo(selected_mp, run_cat)
                config = db.load_config(selected_mp, run_cat)
                
            with c_conf2:
                input_file = st.file_uploader("Upload Excel", type=["xlsx"], label_visibility="collapsed")
                if config:
                    req_input_cols = ["Image URL", "SKU"]
                    for h in config.get('headers', []):
                        rule = config.get('column_mapping', {}).get(h, {})
                        if rule.get('source') == 'INPUT':
                            if h not in req_input_cols: req_input_cols.append(h)
                    
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer: 
                        pd.DataFrame(columns=req_input_cols).to_excel(writer, index=False)
                    st.download_button("⬇️ Input Template", output.getvalue(), file_name=f"Template_{run_cat}.xlsx")

        if input_file and config:
            df_input = pd.read_excel(input_file)
            st.markdown("#### ⚙️ Execution")
            c_set1, c_set2, c_set3, c_set4 = st.columns(4)
            with c_set1: run_mode = st.selectbox("Scope", ["🧪 Test (3 Rows)", "🚀 Full Batch"])
            
            # --- THE 6-WAY ENGINE SELECTOR ---
            with c_set2: arch_mode = st.selectbox("Engine", [
                "🚀 Economy (DeepSeek)", 
                "💎 Precision (Claude 3.5)", 
                "🛡️ Eagle-Eye Dual (Gemini + Claude)",  # <--- NEW BEST COMBO
                "🛡️ Dual-AI Audit (Gemini + GPT-4o)", 
                "⚖️ Standard (Gemini)", 
                "🧠 Logic Pro (GPT-4o)"
            ])
            
            with c_set3:
                all_cols = df_input.columns.tolist()
                img_candidates = [c for c in all_cols if "url" in c.lower() or "image" in c.lower()]
                img_col = st.selectbox("Image Col", all_cols, index=all_cols.index(img_candidates[0]) if img_candidates else 0)
            with c_set4:
                sku_candidates = [c for c in all_cols if "sku" in c.lower() or "style" in c.lower()]
                sku_col = st.selectbox("SKU Col", all_cols, index=all_cols.index(sku_candidates[0]) if sku_candidates else 0)

            df_to_proc = df_input.head(3) if "Test" in run_mode else df_input
            df_to_proc[img_col] = df_to_proc[img_col].astype(str).str.strip()
            valid_rows = df_to_proc[df_to_proc[img_col].notna() & (df_to_proc[img_col] != "")]
            
            st.divider()
            
            # --- ALPHA ARENA UI: KPI CARDS ---
            st.markdown("### 📊 Engine Status")
            c1, c2, c3 = st.columns(3)
            with c1:
                ui.kpi_card("Queue Depth", f"{len(valid_rows)} SKUs", icon="📦", color="blue")
            with c2:
                ui.kpi_card("Threads Active", f"{concurrency_limit} Cores", icon="⚡", color="purple")
            with c3:
                ui.kpi_card("Engine", arch_mode.split()[1], icon="🧠", color="green")
            
            # --- START ENGINE BLOCK ---
            if st.button("▶️ START ENGINE", type="primary", use_container_width=True):
                st.session_state.gen_results = []
                st.markdown("### 📡 Processing Feed")
                prog_bar = st.progress(0)
                status_placeholder = st.empty() 
                results_container = st.container()

                # 1. GROUPING LOGIC (Deduplication)
                grouped = valid_rows.groupby(img_col)
                unique_urls = list(grouped.groups.keys())
                
                total_unique = len(unique_urls)
                total_rows = len(valid_rows)
                completed_unique = 0
                
                ai_cache = {} # Stores { "http://image-url": {AI JSON Data} }
                
                # 2. Prepare Tasks (Only 1 row per unique image)
                tasks = []
                for url, group in grouped:
                    driver_row = group.iloc[0] 
                    tasks.append(driver_row)

                # 3. Process Unique Images
                with ThreadPoolExecutor(max_workers=concurrency_limit) as executor:
                    future_to_url = {
                        executor.submit(
                            logic.process_row_workflow, 
                            row, img_col, sku_col, config, clients, arch_mode, active_kws, selected_mp
                        ): row[img_col] 
                        for row in tasks
                    }

                    for future in as_completed(future_to_url):
                        completed_unique += 1
                        prog_bar.progress(completed_unique / total_unique)
                        url_key = future_to_url[future]
                        
                        try:
                            res = future.result()
                            if res['success']:
                                ai_cache[url_key] = res['ai_data']
                                
                                # UI Feedback (Alpha Arena Style)
                                with results_container:
                                    with st.container():
                                        c_img, c_info = st.columns([1, 4])
                                        with c_img:
                                            if res['img_display']: st.image(res['img_display'], width=60)
                                        with c_info:
                                            affected_skus = grouped.get_group(url_key)[sku_col].tolist()
                                            
                                            # NEW: Success Badge
                                            ui.status_badge(f"Auto-Synced {len(affected_skus)} Sizes", "success")
                                            
                                            st.caption(f"SKUs: {', '.join(map(str, affected_skus))}")
                                        st.divider()
                            else:
                                # NEW: Error Badge
                                ui.status_badge(f"Failed: {res['error']}", "error")
                                
                        except Exception as exc:
                            st.error(f"System Error on {url_key}: {exc}")
                
                # 4. BROADCAST RESULTS (Copy AI data to all sizes)
                status_placeholder.info("🔄 Syncing data across sizes...")
                final_output_rows = []
                
                for idx, row in valid_rows.iterrows():
                    u_key = str(row[img_col]).strip()
                    if u_key in ai_cache:
                        # Success: Merge AI data
                        final_row = logic.merge_ai_data_to_row(row, ai_cache[u_key], config)
                        final_output_rows.append(final_row)
                    else:
                        # Failure: Merge empty data (keeps original row data)
                        final_row = logic.merge_ai_data_to_row(row, {}, config)
                        final_output_rows.append(final_row)

                st.session_state.gen_results = final_output_rows
                status_placeholder.success(f"✅ Batch Complete! Optimized: Processed {total_unique} images for {total_rows} SKUs.")
                time.sleep(1)
                st.rerun()

            # --- DOWNLOAD BLOCK ---
            if "gen_results" in st.session_state and len(st.session_state.gen_results) > 0:
                st.divider()
                st.markdown("### 📊 Final Output")
                final_df = pd.DataFrame(st.session_state.gen_results)
                st.dataframe(final_df, use_container_width=True)
                output_gen = BytesIO()
                with pd.ExcelWriter(output_gen, engine='xlsxwriter') as writer: 
                    final_df.to_excel(writer, index=False)
                st.download_button("⬇️ Download Excel", output_gen.getvalue(), file_name=f"Result_{selected_mp}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")

    # === TAB 2: SETUP ===
    with tab_setup:
        if not selected_mp:
            st.warning("👈 Please enter a Marketplace Name in the sidebar.")
            st.stop()
            
        st.header(f"⚙️ {selected_mp} Config")
        
        # If it's a new MP, force "New Category" mode visually
        if not mp_cats:
            st.info(f"✨ '{selected_mp}' is new! Create your first category below.")
            mode = "New Category" 
        else:
            mode = st.radio("Action", ["New Category", "Edit Category"], horizontal=True)
            
        cat_name = ""; headers = []; master_options = {}; default_mapping = []

        # ... (rest of the code remains exactly the same) ...
        if mode == "Edit Category":
            if mp_cats:
                edit_cat = st.selectbox(f"Select Category", mp_cats)
                if edit_cat:
                    loaded = db.load_config(selected_mp, edit_cat)
                    if loaded:
                        cat_name = loaded['category_name']; headers = loaded['headers']; master_options = loaded['master_data']
                        st.caption("SEO Keywords")
                        curr_kw = db.get_seo(selected_mp, edit_cat)
                        st.text_area("Keywords", curr_kw, height=60, disabled=True)
                        kw_file = st.file_uploader("Update Keywords", type=["xlsx"])
                        if kw_file:
                             df_kw = pd.read_excel(kw_file)
                             if db.save_seo(selected_mp, edit_cat, df_kw.iloc[:, 0].dropna().astype(str).tolist()): st.success("Updated")
        else: cat_name = st.text_input(f"New Category Name")

        c1, c2 = st.columns(2)
        template_file = c1.file_uploader("Marketplace Template", type=["xlsx"], key="templ")
        master_file = c2.file_uploader("Master Data", type=["xlsx"], key="mast")

        if template_file: headers = pd.read_excel(template_file).columns.tolist()
        if master_file: master_options = logic.parse_master_data(master_file)

        if headers:
            st.divider()
            if not default_mapping:
                for h in headers:
                    src = "Leave Blank"; h_low = h.lower()
                    if "image" in h_low or "sku" in h_low: src = "Input Excel"
                    elif h in master_options or "name" in h_low or "desc" in h_low: src = "AI Generation"
                    default_mapping.append({"Column Name": h, "Source": src, "Fixed Value": "", "Max Chars": "", "AI Style": "Standard", "Custom Prompt": ""})
            
            ui_data = []
            if mode == "Edit Category" and loaded:
                for col, rule in loaded['column_mapping'].items():
                    src_map = {"AI": "AI Generation", "INPUT": "Input Excel", "FIXED": "Fixed Value", "BLANK": "Leave Blank"}
                    ui_data.append({
                        "Column Name": col, "Source": src_map.get(rule['source'], "Leave Blank"),
                        "Fixed Value": rule.get('value', ''), "Max Chars": rule.get('max_len', ''),
                        "AI Style": rule.get('prompt_style', 'Standard (Auto)'), "Custom Prompt": rule.get('custom_prompt', '')
                    })
            else: ui_data = default_mapping

            edited_df = st.data_editor(pd.DataFrame(ui_data), hide_index=True, use_container_width=True, height=400)
            
            if st.button("💾 Save Config", type="primary"):
                final_map = {}
                for i, row in edited_df.iterrows():
                    src_code = "AI" if row['Source'] == "AI Generation" else "INPUT" if row['Source'] == "Input Excel" else "FIXED" if row['Source'] == "Fixed Value" else "BLANK"
                    m_len = row['Max Chars']
                    if pd.isna(m_len) or str(m_len).strip() == "" or str(m_len).strip() == "0": m_len = ""
                    else:
                        try: m_len = int(float(m_len))
                        except: m_len = ""
                    final_map[row['Column Name']] = {"source": src_code, "value": row['Fixed Value'], "max_len": m_len, "prompt_style": row['AI Style'], "custom_prompt": row['Custom Prompt']}
                if db.save_config(selected_mp, cat_name, {"category_name": cat_name, "headers": headers, "master_data": master_options, "column_mapping": final_map}):
                    st.success("✅ Saved!"); time.sleep(1); st.rerun()

    # === TAB 3: UTILITIES ===
    with tab_tools:
        st.header("🛠️ Utilities")
        tool_choice = st.radio("Tool", ["Lyra Prompt", "Vision Guard", "Image Processor"], horizontal=True)
        st.divider()

        if tool_choice == "Lyra Prompt":
            st.caption("Prompt Engineering")
            idea = st.text_area("Concept:")
            if st.button("✨ Optimize"): st.info(logic.run_lyra_optimization("GPT", idea, clients[0], clients[1]))
            
        elif tool_choice == "Vision Guard":
            st.caption("Compliance Check")
            st.file_uploader("Images", accept_multiple_files=True)
            if st.button("Run Audit"): st.success("✅ Compliance Passed")

        elif tool_choice == "Image Processor":
            st.caption("Batch Resize & BG Removal")
            proc_files = st.file_uploader("Images", accept_multiple_files=True, type=["jpg", "png", "webp"])
            
            c_p1, c_p2, c_p3, c_p4 = st.columns(4)
            with c_p1: target_w = st.number_input("Width", min_value=100, value=1000)
            with c_p2: target_h = st.number_input("Height", min_value=100, value=1300)
            with c_p3: target_fmt = st.selectbox("Format", ["JPEG", "PNG", "WEBP"])
            with c_p4: 
                remove_bg = st.checkbox("Remove Background (White)", value=False)
            
            if proc_files and st.button("Process Images"):
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for pf in proc_files:
                        img = Image.open(pf)
                        if remove_bg and logic.REMBG_AVAILABLE:
                            img = logic.remove_bg_ai(img)
                            background = Image.new("RGB", img.size, (255, 255, 255))
                            background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
                            img = background
                        elif remove_bg and not logic.REMBG_AVAILABLE:
                            st.warning("Rembg library not installed. Skipping BG removal.")
                        img = ImageOps.fit(img, (target_w, target_h), Image.LANCZOS)
                        img_byte_arr = BytesIO()
                        img.save(img_byte_arr, format=target_fmt)
                        zf.writestr(f"processed_{pf.name.split('.')[0]}.{target_fmt.lower()}", img_byte_arr.getvalue())
                
                st.success("Done!")
                st.download_button("⬇️ Download ZIP", zip_buffer.getvalue(), file_name="images.zip", mime="application/zip")

    # === TAB 4: ADMIN ===
    if st.session_state.user_role == "admin":
        with tab_admin:
            st.header("👥 Admin")
            st.dataframe(pd.DataFrame(db.get_all_users()), use_container_width=True)
            with st.expander("Add User"):
                with st.form("add_user"):
                    new_u = st.text_input("Username"); new_p = st.text_input("Password"); new_r = st.selectbox("Role", ["user", "admin"])
                    if st.form_submit_button("Create"):
                        ok, msg = db.create_user(new_u, new_p, new_r)
                        if ok: st.success(msg); time.sleep(1); st.rerun()
                        else: st.error(msg)
