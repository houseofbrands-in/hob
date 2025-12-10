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

# --- DEPENDENCY CHECK ---
try:
    from rembg import remove as remove_bg_ai
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# --- INIT CLIENTS ---
def init_clients():
    # 1. Standard OpenAI (Legacy)
    gpt_client = None
    try:
        if "OPENAI_API_KEY" in st.secrets:
            gpt_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except: pass

    # 2. Google Gemini (Legacy)
    gemini_avail = False
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            gemini_avail = True
    except: pass
    
    # 3. OpenRouter (The New Multi-Model Engine)
    or_client = None
    try:
        if "OPENROUTER_API_KEY" in st.secrets:
            or_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=st.secrets["OPENROUTER_API_KEY"],
            )
    except: pass
    
    return gpt_client, gemini_avail, or_client

# --- UTILS (Shared) ---
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

# --- THE NEW UNIVERSAL ENGINE ---
def analyze_image_universal(base64_image, user_hints, keywords, config, marketplace, engine_mode, clients):
    gpt_c, gemini_avail, or_c = clients
    
    # Define Model IDs
    MODEL_MAP = {
        "DeepSeek V3": "deepseek/deepseek-chat",  # Text powerhouse (requires vision extraction first)
        "Claude 3.5 Sonnet": "anthropic/claude-3.5-sonnet", # Vision King
        "GPT-4o": "openai/gpt-4o",
        "Gemini 2.5": "google/gemini-flash-1.5"
    }

    target_columns = []
    strict_constraints = {} 
    creative_columns = []   
    
    for col, settings in config['column_mapping'].items():
        if settings['source'] == 'AI':
            target_columns.append(col)
            # ... (Constraint extraction logic same as before) ...
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

    # --- PROMPT CONSTRUCTION ---
    system_prompt = f"""
    You are an expert E-commerce Cataloger for {marketplace}.
    Analyze the product image strictly.
    OUTPUT FORMAT: JSON only. No markdown.
    
    CONSTRAINTS (You MUST pick from these lists):
    {json.dumps(strict_constraints)}
    
    FREE FIELDS (Be creative & detailed, use keywords: {keywords}):
    {creative_columns}
    
    USER HINTS: {user_hints}
    """

    # --- ROUTING LOGIC ---
    
    # ROUTE A: CLAUDE 3.5 SONNET (High Precision Vision)
    if engine_mode == "💎 Precision (Claude 3.5)" and or_c:
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

    # ROUTE B: ECONOMY (Gemini Vision -> DeepSeek Refiner)
    # DeepSeek V3 is text-only (mostly), so we use Gemini to "See" and DeepSeek to "Think/Format"
    elif engine_mode == "🚀 Economy (DeepSeek)" and gemini_avail:
        try:
            # Step 1: Gemini sees the image
            model = genai.GenerativeModel('gemini-2.5-flash')
            img_data = base64.b64decode(base64_image)
            image_part = {"mime_type": "image/jpeg", "data": img_data}
            
            vision_prompt = "Describe this product image in extreme detail. Fabric, Pattern, Neckline, Sleeves, Color, Fit, and any Visible Logos."
            vision_res = model.generate_content([vision_prompt, image_part])
            visual_description = vision_res.text
            
            # Step 2: DeepSeek formats it (Cheap & Smart)
            if or_c:
                ds_prompt = f"""
                Taking this visual description: "{visual_description}"
                
                Map it to this JSON Schema for {marketplace}:
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
        except Exception as e: return None, f"DeepSeek/Gemini Error: {str(e)}"

    # ROUTE C: STANDARD (Original Logic - Gemini Direct)
    else:
        # Fallback to original Gemini Logic
        if not gemini_avail: return None, "Gemini Key Missing"
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            img_data = base64.b64decode(base64_image)
            image_part = {"mime_type": "image/jpeg", "data": img_data}
            response = model.generate_content([system_prompt, image_part])
            txt = response.text
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
            ai_data, err = analyze_image_universal(base64_img, hints, active_kws, config, selected_mp, arch_mode, clients)
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
