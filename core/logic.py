import streamlit as st
import pandas as pd
import json
import base64
import requests
import time
import difflib
from openai import OpenAI
import google.generativeai as genai
from io import BytesIO
from PIL import Image

# --- DEPENDENCY CHECK ---
try:
    from rembg import remove as remove_bg_ai
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# --- INIT CLIENTS ---
def init_clients():
    # 1. Standard OpenAI (Direct)
    gpt_client = None
    try:
        if "OPENAI_API_KEY" in st.secrets:
            gpt_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except: pass

    # 2. Google Gemini (Direct)
    gemini_avail = False
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            gemini_avail = True
    except: pass
    
    # 3. OpenRouter (Multi-Model Adapter)
    or_client = None
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            or_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=st.secrets["OPENROUTER_API_KEY"],
            )
    except: pass
    
    return gpt_client, gemini_avail, or_client

# --- UTILS ---
def parse_master_data(file):
    df = pd.read_excel(file)
    valid_options = {}
    for col in df.columns:
        options = df[col].dropna().astype(str).unique().tolist()
        if len(options) > 0: valid_options[col] = options
    return valid_options

def smart_truncate(text, max_length):
    if not text: return ""
    text = str(text).strip()
    if len(text) <= max_length: return text
    truncated = text[:max_length]
    if len(text) > max_length and text[max_length] != " ":
        if " " in truncated: truncated = truncated.rsplit(" ", 1)[0]
    return truncated.strip()

def enforce_master_data_fallback(value, options):
    if not value: return ""
    ai_text = str(value).strip().lower()
    for opt in options:
        if str(opt).strip().lower() == ai_text: return opt
    sorted_options = sorted(options, key=lambda x: len(str(x)), reverse=True)
    for opt in sorted_options:
        opt_val = str(opt).strip().lower()
        if not opt_val: continue
        if opt_val in ai_text: return opt
    matches = difflib.get_close_matches(ai_text, [str(o).lower() for o in options], n=1, cutoff=0.7)
    if matches:
        match_lower = matches[0]
        for opt in options:
            if str(opt).lower() == match_lower: return opt
    return value

def run_lyra_optimization(model_choice, raw_instruction, client, gemini_avail):
    lyra_system_prompt = "You are Lyra, a master-level AI prompt optimization specialist..."
    user_msg = f"Optimize: '{raw_instruction}'"
    try:
        if "GPT" in model_choice and client:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": lyra_system_prompt},{"role": "user", "content": user_msg}]
            )
            return response.choices[0].message.content
        elif "Gemini" in model_choice:
            if not gemini_avail: return "Gemini API Key missing."
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(f"{lyra_system_prompt}\n\nUSER REQUEST: {user_msg}")
            return response.text
    except Exception as e: return f"Error: {str(e)}"

# --- THE UNIVERSAL ANALYSIS ENGINE ---
def analyze_image_multimodal(base64_image, user_hints, keywords, config, marketplace, engine_mode, clients):
    gpt_c, gemini_avail, or_c = clients
    
    # Prepare Constraints
    target_columns = []
    strict_constraints = {} 
    creative_columns = []   
    
    for col, settings in config['column_mapping'].items():
        if settings['source'] == 'AI':
            target_columns.append(col)
            best_match_key = None
            best_match_len = -1
            for master_col in config['master_data'].keys():
                c_clean = col.lower().strip()
                m_clean = master_col.lower().strip()
                if c_clean == m_clean:
                    best_match_key = master_col; break
                elif m_clean in c_clean:
                    if len(m_clean) > best_match_len:
                        best_match_len = len(m_clean); best_match_key = master_col
            if best_match_key: strict_constraints[col] = config['master_data'][best_match_key]
            else: creative_columns.append(col)

    # Base System Prompt
    system_prompt = f"""
    Role: E-commerce Expert for {marketplace}.
    Task: Analyze image and generate JSON.
    SECTION A: ALLOWED OPTIONS: {json.dumps(strict_constraints)}
    SECTION B: CREATIVE: {creative_columns} - Keywords: {keywords}
    Hints: {user_hints}
    Output: JSON Only.
    """

    # --- ROUTING LOGIC ---

    # 1. CLAUDE 3.5 SONNET (Vision King)
    if "Precision" in engine_mode:
        if not or_c: return None, "OpenRouter Key Missing"
        try:
            response = or_c.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": system_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                temperature=0.1
            )
            txt = response.choices[0].message.content
            if "```json" in txt: txt = txt.split("```json")[1].split("```")[0]
            return json.loads(txt), None
        except Exception as e: return None, f"Claude Error: {str(e)}"

    # 2. DEEPSEEK V3 (Economy)
    elif "Economy" in engine_mode:
        if not gemini_avail: return None, "Gemini Key Missing (Required for Vision)"
        if not or_c: return None, "OpenRouter Key Missing"
        try:
            # Step A: Vision (Gemini)
            model = genai.GenerativeModel('gemini-2.5-flash')
            img_data = base64.b64decode(base64_image)
            image_part = {"mime_type": "image/jpeg", "data": img_data}
            vis_prompt = "Describe this product in extreme detail: Fabric, Color, Pattern, Neckline, Sleeves, Fit, Branding."
            vis_res = model.generate_content([vis_prompt, image_part])
            visual_desc = vis_res.text
            
            # Step B: Logic (DeepSeek)
            ds_prompt = f"""
            Context: {visual_desc}
            Task: Map this description to the following JSON schema.
            {system_prompt}
            """
            response = or_c.chat.completions.create(
                model="deepseek/deepseek-chat",
                messages=[{"role": "user", "content": ds_prompt}],
                temperature=0.0
            )
            txt = response.choices[0].message.content
            if "```json" in txt: txt = txt.split("```json")[1].split("```")[0]
            return json.loads(txt), None
        except Exception as e: return None, f"DeepSeek Error: {str(e)}"

    # 3. EAGLE-EYE DUAL (Gemini + Claude Audit)
    elif "Eagle-Eye" in engine_mode:
        if not gemini_avail or not or_c: return None, "Need Gemini and OpenRouter Keys"
        try:
            # Maker (Gemini)
            model = genai.GenerativeModel('gemini-2.5-flash')
            img_data = base64.b64decode(base64_image)
            image_part = {"mime_type": "image/jpeg", "data": img_data}
            res = model.generate_content([system_prompt, image_part])
            maker_txt = res.text
            if "```json" in maker_txt: maker_txt = maker_txt.split("```json")[1].split("```")[0]
            elif "```" in maker_txt: maker_txt = maker_txt.split("```")[1].split("```")[0]
            maker_draft = json.loads(maker_txt)
            
            # Checker (Claude)
            audit_prompt = f"""
            You are the QUALITY CONTROL CHIEF.
            1. Look at the image.
            2. Review this Draft JSON: {json.dumps(maker_draft)}
            3. CRITICAL: If the fabric, color, or neck/sleeve type in the Draft does not match the Image perfectly, CORRECT IT.
            4. Ensure these allowed options are respected: {json.dumps(strict_constraints)}
            5. Output the cleaned, final JSON.
            """
            
            response = or_c.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": audit_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                temperature=0.1
            )
            txt = response.choices[0].message.content
            if "```json" in txt: txt = txt.split("```json")[1].split("```")[0]
            return json.loads(txt), None
            
        except Exception as e: return None, f"Eagle-Eye Error: {str(e)}"

    # 4. STANDARD DUAL (Gemini + GPT-4o Audit)
    elif "Dual-AI" in engine_mode:
        if not gemini_avail or not gpt_c: return None, "Need both Gemini and OpenAI Keys"
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            img_data = base64.b64decode(base64_image)
            image_part = {"mime_type": "image/jpeg", "data": img_data}
            res = model.generate_content([system_prompt, image_part])
            maker_txt = res.text
            if "```json" in maker_txt: maker_txt = maker_txt.split("```json")[1].split("```")[0]
            elif "```" in maker_txt: maker_txt = maker_txt.split("```")[1].split("```")[0]
            maker_draft = json.loads(maker_txt)
            
            audit_prompt = f"""
            You are the AUDITOR.
            Visual Inputs provided.
            Draft JSON: {json.dumps(maker_draft)}
            Constraints: {json.dumps(strict_constraints)}
            Mission: Verify Draft against Image. Fix errors.
            Output: Final JSON.
            """
            response = gpt_c.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Auditor Mode. Temp=0."},
                    {"role": "user", "content": [{"type": "text", "text": audit_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
                ],
                response_format={"type": "json_object"}, temperature=0.0
            )
            return json.loads(response.choices[0].message.content), None
        except Exception as e: return None, f"Dual-AI Error: {str(e)}"

    # 5. GPT-4o ONLY (Logic Pro) -- [RESTORED]
    elif "GPT" in engine_mode:
        if not gpt_c: return None, "OpenAI Key Missing"
        try:
            response = gpt_c.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "E-commerce Expert."},
                    {"role": "user", "content": [{"type": "text", "text": system_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
                ],
                response_format={"type": "json_object"}, temperature=0.0
            )
            return json.loads(response.choices[0].message.content), None
        except Exception as e: return None, f"GPT Error: {str(e)}"

    # 6. GEMINI ONLY
    else:
        if not gemini_avail: return None, "Gemini Key Missing"
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            img_data = base64.b64decode(base64_image)
            image_part = {"mime_type": "image/jpeg", "data": img_data}
            res = model.generate_content([system_prompt, image_part])
            txt = res.text
            if "```json" in txt: txt = txt.split("```json")[1].split("```")[0]
            elif "```" in txt: txt = txt.split("```")[1].split("```")[0]
            return json.loads(txt), None
        except Exception as e: return None, f"Gemini Error: {str(e)}"

# --- HELPER: MERGE AI DATA ---
def merge_ai_data_to_row(row_data, ai_data, config):
    new_row = {}
    mapping = config['column_mapping']
    for col in config['headers']:
        rule = mapping.get(col, {'source': 'BLANK'})
        val = ""
        if rule['source'] == 'INPUT': val = row_data.get(col, "")
        elif rule['source'] == 'FIXED': val = rule['value']
        elif rule['source'] == 'AI' and ai_data:
            if col in ai_data: val = ai_data[col]
            else: 
                clean_col = col.lower().replace(" ", "").replace("_", "")
                for k,v in ai_data.items():
                    if k.lower().replace(" ", "") in clean_col: val = v; break
            m_list = []
            for mc, opts in config['master_data'].items():
                if mc.lower() in col.lower(): m_list = opts; break
            if m_list and val: val = enforce_master_data_fallback(val, m_list)
        if isinstance(val, (list, tuple)): val = ", ".join(map(str, val))
        elif isinstance(val, dict): val = json.dumps(val)
        val = str(val).strip()
        if rule.get('max_len'): val = smart_truncate(val, int(float(rule['max_len'])))
        new_row[col] = val
    return new_row

# --- WORKER ---
def process_row_workflow(row_data, img_col, sku_col, config, clients, arch_mode, active_kws, selected_mp):
    u_key = str(row_data.get(img_col, "")).strip()
    sku_label = str(row_data.get(sku_col, "Unknown SKU"))
    
    result_package = {
        "success": False,
        "sku": sku_label,
        "u_key": u_key,
        "img_display": None,
        "ai_data": {},
        "final_row": {},
        "error": None
    }
    
    # Download
    download_url = u_key 
    if "dropbox.com" in download_url: download_url = download_url.replace("?dl=0", "") + "&dl=1"
    
    base64_img = None
    try:
        response = requests.get(download_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if response.status_code == 200:
            result_package["img_display"] = response.content
            base64_img = base64.b64encode(response.content).decode('utf-8')
        else:
            result_package["error"] = f"Img Error: {response.status_code}"
            return result_package
    except Exception as e:
        result_package["error"] = f"Net Error: {str(e)}"
        return result_package

    hints = ", ".join([f"{k}: {v}" for k,v in row_data.items() if k != img_col and str(v).lower() != "nan"])
    hints = smart_truncate(hints, 300)

    # Call Universal Engine
    ai_data = {}; err = None
    for attempt in range(2):
        try:
            ai_data, err = analyze_image_multimodal(base64_img, hints, active_kws, config, selected_mp, arch_mode, clients)
            if err: 
                time.sleep(2); continue
            break
        except Exception as e: err = str(e)

    if err: result_package["error"] = err; return result_package
    if not ai_data: result_package["error"] = "No Data Gen"; return result_package

    result_package["ai_data"] = ai_data
    result_package["success"] = True
    result_package["final_row"] = merge_ai_data_to_row(row_data, ai_data, config)
    
    return result_package
