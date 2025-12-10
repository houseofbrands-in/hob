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
import zipfile
from PIL import Image, ImageOps

# --- DEPENDENCY CHECK ---
try:
    from rembg import remove as remove_bg_ai
    REMBG_AVAILABLE = True
except ImportError:
    REMBG_AVAILABLE = False

# --- INIT CLIENTS ---
def init_clients():
    gpt_client = None
    gemini_avail = False
    
    try:
        if "OPENAI_API_KEY" in st.secrets:
            gpt_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except: pass

    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            gemini_avail = True
    except: pass
    
    return gpt_client, gemini_avail

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

# --- UNIFIED AI ANALYSIS ---
def analyze_image_unified(client, base64_image, user_hints, keywords, config, marketplace, mode="Dual-AI"):
    _, gemini_avail = init_clients() 
    
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
    
    maker_draft = {}
    
    # 1. GEMINI PATH
    if "Gemini" in mode or "Dual" in mode:
        try:
            if not gemini_avail: return None, "Gemini Missing"
            model = genai.GenerativeModel('gemini-2.5-flash')
            img_data = base64.b64decode(base64_image)
            image_part = {"mime_type": "image/jpeg", "data": img_data}
            
            maker_prompt = f"""
            Role: E-commerce Expert for {marketplace}.
            Task: Analyze image and generate JSON.
            SECTION A: ALLOWED OPTIONS: {json.dumps(strict_constraints)}
            SECTION B: CREATIVE: {creative_columns} - Keywords: {keywords}
            Hints: {user_hints}
            Output: JSON Only.
            """
            response = model.generate_content([maker_prompt, image_part], generation_config=genai.types.GenerationConfig(temperature=0.4))
            text_out = response.text
            if "```json" in text_out: text_out = text_out.split("```json")[1].split("```")[0]
            elif "```" in text_out: text_out = text_out.split("```")[1].split("```")[0]
            maker_draft = json.loads(text_out)
            if "Gemini" in mode: return maker_draft, None 

        except Exception as e:
            if "Gemini" in mode: return None, f"Gemini Failed: {str(e)}"
            return None, f"Maker (Gemini) Failed: {str(e)}"

    # 2. GPT PATH
    if "GPT" in mode or "Dual" in mode:
        try:
            if "GPT" in mode:
                gpt_prompt = f"""
                Role: E-commerce Expert.
                Task: Analyze image and generate JSON.
                ALLOWED OPTIONS: {json.dumps(strict_constraints)}
                CREATIVE COLS: {creative_columns}
                Hints: {user_hints}
                Output: JSON Only.
                """
            else: # Dual
                gpt_prompt = f"""
                You are the LEAD DATA AUDITOR.
                INPUTS: 1. Visual 2. Draft: {json.dumps(maker_draft)} 3. Options: {json.dumps(strict_constraints)}
                MISSION: Enforce consistency. If Draft conflicts with Image or Options, OVERWRITE it.
                OUTPUT: Final JSON for columns: {", ".join(target_columns)}
                """

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a Data Engine. Temperature=0.0."},
                    {"role": "user", "content": [{"type": "text", "text": gpt_prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}
                ],
                response_format={"type": "json_object"}, temperature=0.0
            )
            return json.loads(response.choices[0].message.content), None
        except Exception as e: return None, f"GPT Failed: {str(e)}"
    
    return None, "Invalid Mode"

# --- HELPER: MERGE AI DATA INTO ROW ---
# This allows us to reuse the AI result for multiple rows (S, M, L)
def merge_ai_data_to_row(row_data, ai_data, config):
    new_row = {}
    mapping = config['column_mapping']
    
    for col in config['headers']:
        rule = mapping.get(col, {'source': 'BLANK'})
        val = ""
        
        # SOURCE: INPUT (Grab from Excel, e.g. SKU or Size)
        if rule['source'] == 'INPUT': 
            val = row_data.get(col, "")
            
        # SOURCE: FIXED
        elif rule['source'] == 'FIXED': 
            val = rule['value']
            
        # SOURCE: AI (Use the cached AI data)
        elif rule['source'] == 'AI' and ai_data:
            if col in ai_data: val = ai_data[col]
            else: 
                clean_col = col.lower().replace(" ", "").replace("_", "")
                for k,v in ai_data.items():
                    if k.lower().replace(" ", "") in clean_col: val = v; break
            
            # Apply Master Data Constraints
            m_list = []
            for mc, opts in config['master_data'].items():
                if mc.lower() in col.lower(): m_list = opts; break
            if m_list and val: val = enforce_master_data_fallback(val, m_list)
        
        # Formatting
        if isinstance(val, (list, tuple)): val = ", ".join(map(str, val))
        elif isinstance(val, dict): val = json.dumps(val)
        val = str(val).strip()
        if rule.get('max_len'): val = smart_truncate(val, int(float(rule['max_len'])))
        new_row[col] = val
        
    return new_row

# --- WORKER ---
def process_row_workflow(row_data, img_col, sku_col, config, client, arch_mode, active_kws, selected_mp):
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
    
    # 1. Download Image
    download_url = u_key 
    if "dropbox.com" in download_url: 
        download_url = download_url.replace("?dl=0", "").replace("&dl=0", "") + "&dl=1"
    
    base64_img = None
    try:
        response = requests.get(download_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if response.status_code == 200:
            result_package["img_display"] = response.content
            base64_img = base64.b64encode(response.content).decode('utf-8')
        else:
            result_package["error"] = f"Download Failed: {response.status_code}"
            return result_package
    except Exception as e:
        result_package["error"] = f"Network Error: {str(e)}"
        return result_package

    # 2. Hints
    hints = "Product analysis."
    try:
        hints = ", ".join([f"{k}: {v}" for k,v in row_data.items() if k != img_col and str(v).lower() != "nan"])
        hints = smart_truncate(hints, 300)
    except: pass

    # 3. AI Execution
    ai_data = {}
    err = None
    mode_arg = "Dual-AI"
    if "Gemini" in arch_mode: mode_arg = "Gemini"
    elif "GPT" in arch_mode: mode_arg = "GPT"

    for attempt in range(3):
        try:
            ai_data, err = analyze_image_unified(client, base64_img, hints, active_kws, config, selected_mp, mode=mode_arg)
            if err: 
                if "429" in str(err): 
                    time.sleep(60) 
                    continue
                else: raise Exception(err)
            break
        except Exception as e:
            err = str(e)
            time.sleep(2)

    if err:
        result_package["error"] = err
        return result_package

    result_package["ai_data"] = ai_data
    result_package["success"] = True

    # 4. Construct Row (Using the new helper)
    result_package["final_row"] = merge_ai_data_to_row(row_data, ai_data, config)
    
    return result_package
