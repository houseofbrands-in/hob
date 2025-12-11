import streamlit as st
import pandas as pd
import json
import base64
import requests
import time
import difflib
import urllib.parse
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

print("DEBUG: HOB OS Logic Module Loaded (v12.0.4 - Robust Image Handling)")

# --- INIT CLIENTS ---
def init_clients():
    gpt_client = None
    try:
        if "OPENAI_API_KEY" in st.secrets:
            gpt_client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except: pass

    gemini_avail = False
    try:
        if "GEMINI_API_KEY" in st.secrets:
            genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            gemini_avail = True
    except: pass
    
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

# --- CORE ANALYSIS ENGINE ---
def analyze_image_multimodal(base64_image, user_hints, keywords, config, marketplace, engine_mode, clients):
    gpt_c, gemini_avail, or_c = clients
    strict_constraints = {} 
    creative_columns = []   
    
    for col, settings in config['column_mapping'].items():
        if settings['source'] == 'AI':
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

    system_prompt = f"""
    Role: E-commerce Expert for {marketplace}.
    Task: Analyze image and generate JSON.
    SECTION A: ALLOWED OPTIONS: {json.dumps(strict_constraints)}
    SECTION B: CREATIVE: {creative_columns} - Keywords: {keywords}
    Hints: {user_hints}
    Output: JSON Only.
    """

    # --- SIMPLIFIED ROUTING ---
    if "Precision" in engine_mode and or_c:
        try:
            response = or_c.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[{"role": "user", "content": [{"type": "text", "text": system_prompt},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
                temperature=0.1
            )
            txt = response.choices[0].message.content
            if "```json" in txt: txt = txt.split("```json")[1].split("```")[0]
            return json.loads(txt), None
        except Exception as e: return None, str(e)
    
    elif "Economy" in engine_mode and gemini_avail and or_c:
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            img_data = base64.b64decode(base64_image)
            image_part = {"mime_type": "image/jpeg", "data": img_data}
            vis_res = model.generate_content(["Describe this product in extreme detail.", image_part])
            ds_prompt = f"Context: {vis_res.text}\nTask: Map to JSON.\n{system_prompt}"
            response = or_c.chat.completions.create(
                model="deepseek/deepseek-chat", messages=[{"role": "user", "content": ds_prompt}], temperature=0.0
            )
            txt = response.choices[0].message.content
            if "```json" in txt: txt = txt.split("```json")[1].split("```")[0]
            return json.loads(txt), None
        except Exception as e: return None, str(e)
        
    elif "Dual-AI" in engine_mode and gemini_avail and gpt_c:
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            img_data = base64.b64decode(base64_image)
            image_part = {"mime_type": "image/jpeg", "data": img_data}
            res = model.generate_content([system_prompt, image_part])
            maker_draft = json.loads(res.text.replace("```json", "").replace("```", ""))
            audit_prompt = f"Review Draft: {json.dumps(maker_draft)}. Constraints: {json.dumps(strict_constraints)}. Output JSON."
            response = gpt_c.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [{"type": "text", "text": audit_prompt},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
                response_format={"type": "json_object"}, temperature=0.0
            )
            return json.loads(response.choices[0].message.content), None
        except Exception as e: return None, str(e)

    elif "GPT" in engine_mode and gpt_c:
        try:
            response = gpt_c.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [{"type": "text", "text": system_prompt},{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}]}],
                response_format={"type": "json_object"}, temperature=0.0
            )
            return json.loads(response.choices[0].message.content), None
        except Exception as e: return None, str(e)

    else: # Fallback Gemini
        if not gemini_avail: return None, "Gemini Key Missing"
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            img_data = base64.b64decode(base64_image)
            image_part = {"mime_type": "image/jpeg", "data": img_data}
            res = model.generate_content([system_prompt, image_part])
            txt = res.text.replace("```json", "").replace("```", "")
            return json.loads(txt), None
        except Exception as e: return None, str(e)

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

def process_row_workflow(row_data, img_col, sku_col, config, clients, arch_mode, active_kws, selected_mp):
    u_key = str(row_data.get(img_col, "")).strip()
    sku_label = str(row_data.get(sku_col, "Unknown SKU"))
    result_package = {"success": False, "sku": sku_label, "u_key": u_key, "img_display": None, "ai_data": {}, "final_row": {}, "error": None}
    
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

    ai_data = {}; err = None
    for attempt in range(2):
        try:
            ai_data, err = analyze_image_multimodal(base64_img, hints, active_kws, config, selected_mp, arch_mode, clients)
            if err: time.sleep(2); continue
            break
        except Exception as e: err = str(e)

    if err: result_package["error"] = err; return result_package
    if not ai_data: result_package["error"] = "No Data Gen"; return result_package

    result_package["ai_data"] = ai_data
    result_package["success"] = True
    result_package["final_row"] = merge_ai_data_to_row(row_data, ai_data, config)
    return result_package

def estimate_cost(engine_mode, num_skus):
    rates = {"GPT": 0.02, "Claude": 0.01, "DeepSeek": 0.002, "Gemini": 0.0005, "Eagle-Eye": 0.0105, "Dual-AI": 0.0205}
    active_rate = 0.02
    if "DeepSeek" in engine_mode: active_rate = rates["DeepSeek"]
    elif "Precision" in engine_mode: active_rate = rates["Claude"]
    elif "Eagle-Eye" in engine_mode: active_rate = rates["Eagle-Eye"]
    elif "Dual-AI" in engine_mode: active_rate = rates["Dual-AI"]
    elif "Standard" in engine_mode: active_rate = rates["Gemini"]
    
    total = active_rate * num_skus
    bench = rates["GPT"] * num_skus
    return total, bench, bench - total

# --- PHASE 3: AI STUDIO (ROBUST VERSION) ---
def generate_ai_background(prompt, _unused_client=None):
    """
    Tries OpenRouter (Flux) -> Fallback to Pollinations.ai
    """
    # 1. STRATEGY: OPENROUTER FLUX
    if "OPENROUTER_API_KEY" in st.secrets:
        try:
            api_key = st.secrets["OPENROUTER_API_KEY"]
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "HTTP-Referer": "https://hob-os.com", "X-Title": "HOB"}
            
            # Trying the free endpoint explicitly
            model_id = "black-forest-labs/flux-1-schnell:free" 
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": f"professional product photography background, {prompt}, soft lighting, 8k resolution, photorealistic"}]
            }
            res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=25)
            
            if res.status_code == 200:
                content = res.json()['choices'][0]['message']['content']
                if "(" in content and ")" in content:
                    return content[content.find("(")+1:content.find(")")], None
                if content.strip().startswith("http"):
                    return content.strip(), None
            else:
                print(f"DEBUG: OpenRouter Flux Failed ({res.status_code}). Switching to Fallback.")
        except: pass

    # 2. STRATEGY: POLLINATIONS (Guaranteed)
    try:
        # Use simple prompt encoding
        safe_prompt = urllib.parse.quote(f"product photography background {prompt}")
        poll_url = f"https://image.pollinations.ai/prompt/{safe_prompt}?width=1024&height=1024&nologo=true&seed={int(time.time())}"
        
        # Verify it's reachable
        return poll_url, None
            
    except Exception as e:
        return None, f"All strategies failed. {e}"

def composite_product(product_img, bg_url):
    """
    Robust Composition with File Identification
    """
    try:
        # A. Remove Background
        if REMBG_AVAILABLE:
            try:
                product_img = remove_bg_ai(product_img)
            except: pass 

        # B. Download Background (Stealth Mode)
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # We need a session to handle redirects correctly for Pollinations
        session = requests.Session()
        bg_response = session.get(bg_url, headers=headers, timeout=25, stream=True)
        
        # --- CRITICAL FIX: Check if we actually got an image ---
        content_type = bg_response.headers.get('Content-Type', '').lower()
        
        # If not an image, create a fallback color background
        if 'image' not in content_type:
            print(f"DEBUG: Failed to download valid image. Got {content_type}. Using fallback color.")
            bg_img = Image.new('RGB', (1024, 1024), (240, 240, 240)) # Light Grey Fallback
        else:
            try:
                # Try to open the image
                bg_img = Image.open(BytesIO(bg_response.content)).convert("RGBA")
            except Exception as e:
                print(f"DEBUG: Image Open Failed: {e}. Using fallback.")
                bg_img = Image.new('RGB', (1024, 1024), (200, 200, 200)) # Grey Fallback

        # C. Resize Background
        bg_img = bg_img.resize((1024, 1024))
        
        # D. Resize Product
        p_w, p_h = product_img.size
        target_h = int(1024 * 0.7)
        ratio = target_h / p_h
        target_w = int(p_w * ratio)
        product_resized = product_img.resize((target_w, target_h), Image.LANCZOS)
        
        # E. Center
        bg_w, bg_h = bg_img.size
        offset_x = (bg_w - target_w) // 2
        offset_y = (bg_h - target_h) // 2 + 50
        
        # F. Paste
        final_comp = bg_img.copy()
        final_comp.paste(product_resized, (offset_x, offset_y), product_resized)
        
        return final_comp.convert("RGB"), None
        
    except Exception as e:
        return None, f"Composite Error: {str(e)}"
