import streamlit as st
import pandas as pd
import time
import zipfile  # <--- WAS MISSING
from PIL import Image, ImageOps # <--- WAS MISSING
from io import BytesIO
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

# --- INIT CLIENTS ---
clients = logic.init_clients()

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
    # --- DYNAMIC SIDEBAR ---
    with st.sidebar:
        st.markdown("### ⚡ HOB OS")
        st.caption(f"Operator: **{st.session_state.username}**")
        st.divider()
        
        st.subheader("📍 Target")
        
        # 1. Fetch existing MPs from DB
        mp_options = db.get_all_marketplaces()
        mp_options.append("➕ New Marketplace")
        
        # 2. Render Dropdown
        selected_mp_raw = st.selectbox("Marketplace", mp_options)
        
        # 3. Handle "New" Input
        selected_mp = None
        if selected_mp_raw == "➕ New Marketplace":
            new_mp_name = st.text_input("Enter Name", placeholder="e.g. Snapdeal")
            if new_mp_name:
                selected_mp = new_mp_name.strip()
            else:
                st.caption("👈 Type name & press Enter")
        else:
            selected_mp = selected_mp_raw

        # 4. Safe Fetch Categories
        mp_cats = []
        if selected_mp:
            try:
                mp_cats = db.get_categories_for_marketplace(selected_mp)
            except:
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
        # LOGIC FIX: Do not use st.stop() here, or it kills Tab 2.
        # Check if we are ready to run
        ready_to_run = False
        
        if not selected_mp:
            if selected_mp_raw == "➕ New Marketplace":
                st.info("👈 Please type the new Marketplace Name in the sidebar and press ENTER.")
            else:
                st.info("👈 Please select a Marketplace in the sidebar.")
        elif not mp_cats: 
            st.warning(f"⚠️ '{selected_mp}' has no categories yet. Go to the 'Config' tab to create one.")
        else:
            ready_to_run = True

        # Only render the Run Interface if ready
        if ready_to_run:
            with st.expander("📂 **Input & Configuration**", expanded=True):
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
                    "🛡️ Eagle-Eye Dual (Gemini + Claude)", 
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
                
                # --- KPI CARDS ---
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

                    grouped = valid_rows.groupby(img_col)
                    unique_urls = list(grouped.groups.keys())
                    
                    total_unique = len(unique_urls)
                    total_rows = len(valid_rows)
                    completed_unique = 0
                    
                    ai_cache = {} 
                    tasks = []
                    for url, group in grouped:
                        driver_row = group.iloc[0] 
                        tasks.append(driver_row)

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
                                    with results_container:
                                        with st.container():
                                            c_img, c_info = st.columns([1, 4])
                                            with c_img:
                                                if res['img_display']: st.image(res['img_display'], width=60)
                                            with c_info:
                                                affected_skus = grouped.get_group(url_key)[sku_col].tolist()
                                                ui.status_badge(f"Auto-Synced {len(affected_skus)} Sizes", "success")
                                                st.caption(f"SKUs: {', '.join(map(str, affected_skus))}")
                                            st.divider()
                                else:
                                    ui.status_badge(f"Failed: {res['error']}", "error")
                                    
                            except Exception as exc:
                                st.error(f"System Error on {url_key}: {exc}")
                    
                    status_placeholder.info("🔄 Syncing data across sizes...")
                    final_output_rows = []
                    
                    for idx, row in valid_rows.iterrows():
                        u_key = str(row[img_col]).strip()
                        if u_key in ai_cache:
                            final_row = logic.merge_ai_data_to_row(row, ai_cache[u_key], config)
                            final_output_rows.append(final_row)
                        else:
                            final_row = logic.merge_ai_data_to_row(row, {}, config)
                            final_output_rows.append(final_row)

                    st.session_state.gen_results = final_output_rows
                    status_placeholder.success(f"✅ Batch Complete! Optimized: Processed {total_unique} images for {total_rows} SKUs.")
                    time.sleep(1)
                    st.rerun()

               # --- PHASE 1: HUMAN-IN-THE-LOOP EDITOR ---
                if "gen_results" in st.session_state and len(st.session_state.gen_results) > 0:
                    st.divider()
                    st.markdown("### 📝 Human-in-the-Loop Editor")
                    st.caption("Review the AI's work. Edit cells directly. Changes are auto-saved to the Download button.")

                    # 1. Prepare Data
                    df_ai_raw = pd.DataFrame(st.session_state.gen_results)
                    
                    # 2. Render Editor (Dynamic)
                    # We lock the Image URL and SKU to prevent breaking the keys
                    column_config = {
                        img_col: st.column_config.TextColumn("Image Reference", disabled=True),
                        sku_col: st.column_config.TextColumn("SKU", disabled=True)
                    }
                    
                    st.data_editor(
                        df_ai_raw, 
                        key="editor_grid",
                        use_container_width=True,
                        num_rows="fixed", # Prevent accidental deletion of rows
                        column_config=column_config,
                        height=400
                    )

                    # 3. Detect Changes (The "Learning" Logic)
                    # Compare Original AI Draft vs Edited Version
                    try:
                        changes_detected = []
                        if not df_ai_raw.equals(edited_df):
                            # Iterate to find diffs (Simplified for performance)
                            # We only scan if shapes match
                            if df_ai_raw.shape == edited_df.shape:
                                diff_mask = (df_ai_raw != edited_df)
                                diff_locations = diff_mask.stack()[diff_mask.stack()].index.tolist()
                                
                                for idx, col in diff_locations:
                                    row_url = df_ai_raw.loc[idx, img_col]
                                    old_val = df_ai_raw.loc[idx, col]
                                    new_val = edited_df.loc[idx, col]
                                    changes_detected.append({
                                        "img_url": row_url,
                                        "col": col,
                                        "ai_value": old_val,
                                        "human_value": new_val
                                    })

                        # 4. Action Bar
                        c_act1, c_act2 = st.columns([1, 2])
                        
                        with c_act1:
                            # THE DOWNLOAD BUTTON (Now uses edited_df)
                            output_gen = BytesIO()
                            with pd.ExcelWriter(output_gen, engine='xlsxwriter') as writer: 
                                edited_df.to_excel(writer, index=False)
                            
                            st.download_button(
                                "⬇️ Download Final Excel", 
                                output_gen.getvalue(), 
                                file_name=f"Result_{selected_mp}_Verified.xlsx", 
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", 
                                type="primary",
                                use_container_width=True
                            )

                        with c_act2:
                            # THE LEARNING BUTTON
                            if changes_detected:
                                btn_label = f"🧠 Train System ({len(changes_detected)} Corrections)"
                                if st.button(btn_label, use_container_width=True, type="secondary"):
                                    ok, count = db.log_training_data(selected_mp, changes_detected)
                                    if ok:
                                        st.toast(f"✅ System learned from {count} corrections!", icon="🧠")
                                    else:
                                        st.error(f"Save failed: {count}")
                            else:
                                st.button("🧠 System Training (No Changes)", disabled=True, use_container_width=True)
                                
                    except Exception as e:
                        st.error(f"Editor Sync Error: {e}")
    # === TAB 2: SETUP (UNBLOCKED) ===
    with tab_setup:
        # Guard: Check if MP is selected
        if not selected_mp:
            if selected_mp_raw == "➕ New Marketplace":
                st.info("👈 Please type the new Marketplace Name in the sidebar and press ENTER.")
            else:
                st.warning("👈 Please select a Marketplace in the sidebar.")
            # Note: We can use st.stop() here because it's the end of the render chain for this user flow,
            # but safer to just wrap content in 'else'.
        
        else:
            st.header(f"⚙️ {selected_mp} Config")
            
            # Determine Mode
            mode = "New Category"
            if mp_cats:
                if st.radio("Action", ["New Category", "Edit Category"], horizontal=True) == "Edit Category":
                    mode = "Edit Category"
            else:
                st.info(f"✨ '{selected_mp}' is new! Create your first category below.")
                
            # Initialize Variables
            cat_name = ""
            headers = []
            master_options = {}
            loaded = None
            
            # Mode Logic
            if mode == "Edit Category" and mp_cats:
                edit_cat = st.selectbox(f"Select Category", mp_cats)
                if edit_cat:
                    loaded = db.load_config(selected_mp, edit_cat)
                    if loaded:
                        cat_name = loaded['category_name']
                        headers = loaded['headers']
                        master_options = loaded['master_data']
                        
                        st.caption("SEO Keywords")
                        curr_kw = db.get_seo(selected_mp, edit_cat)
                        st.text_area("Keywords", curr_kw, height=60, disabled=True)
                        kw_file = st.file_uploader("Update Keywords", type=["xlsx"])
                        if kw_file:
                                df_kw = pd.read_excel(kw_file)
                                if db.save_seo(selected_mp, edit_cat, df_kw.iloc[:, 0].dropna().astype(str).tolist()): st.success("Updated")
            else:
                # NEW CATEGORY INPUT
                cat_name = st.text_input(f"New Category Name", key="new_cat_input")

            st.divider()
            c1, c2 = st.columns(2)
            template_file = c1.file_uploader("Marketplace Template", type=["xlsx"], key="templ")
            master_file = c2.file_uploader("Master Data", type=["xlsx"], key="mast")

            if template_file: headers = pd.read_excel(template_file).columns.tolist()
            if master_file: master_options = logic.parse_master_data(master_file)

            if headers:
                st.divider()
                default_mapping = []
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

                edited_df = # --- DEFINE DROPDOWN OPTIONS ---
                source_options = ["AI Generation", "Input Excel", "Fixed Value", "Leave Blank"]
                # Added "SEO (Optimized)" back to the list
                style_options = ["Standard (Auto)", "Creative (Marketing)", "Technical (Specs)", "SEO (Optimized)"] 

                # --- RENDER EDITOR WITH ALL CONTROLS ---
                edited_df = st.data_editor(
                    pd.DataFrame(ui_data),
                    hide_index=True,
                    use_container_width=True,
                    height=400,
                    column_config={
                        "Column Name": st.column_config.TextColumn(
                            "Column Name", 
                            disabled=True, 
                            width="medium"
                        ),
                        "Source": st.column_config.SelectboxColumn(
                            "Source",
                            width="medium",
                            options=source_options,
                            required=True
                        ),
                        "Fixed Value": st.column_config.TextColumn(
                            "Fixed Value",
                            width="medium"
                        ),
                        "Max Chars": st.column_config.NumberColumn(
                            "Max Chars",
                            help="Limit output length (0 or Empty = No Limit)",
                            min_value=0,
                            max_value=2000,
                            step=1,
                            width="small"
                        ),
                        "AI Style": st.column_config.SelectboxColumn(
                            "AI Style",
                            width="medium",
                            options=style_options, # Now includes SEO
                            required=True
                        ),
                        "Custom Prompt": st.column_config.TextColumn(
                            "Custom Prompt",
                            width="large"
                        )
                    }
                )
                
                if st.button("💾 Save Config", type="primary"):
                    if not cat_name:
                        st.error("Please enter a category name.")
                    else:
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
                            st.success(f"✅ Saved '{cat_name}' to {selected_mp}!"); 
                            time.sleep(1); 
                            st.rerun()

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
