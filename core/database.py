import streamlit as st
import json
import time  # <--- ADD THIS LINE
import gspread
import bcrypt  # <--- NEW IMPORT
from oauth2client.service_account import ServiceAccountCredentials

SHEET_NAME = "Testing_Agency_OS_Database"
# ... rest of the file

@st.cache_resource
def init_connection():
    try:
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        # Assumes st.secrets is available globally in Streamlit context
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        return client
    except Exception:
        return None

@st.cache_data
def get_worksheet_data(sheet_name, worksheet_name):
    client = init_connection()
    if not client: return []
    try:
        sh = client.open(sheet_name)
        ws = sh.worksheet(worksheet_name)
        return ws.get_all_values()
    except: return []

def get_worksheet_object(ws_name):
    gc = init_connection()
    return gc.open(SHEET_NAME).worksheet(ws_name)

# --- SECURE AUTH ---
def check_login(username, password):
    """
    Verifies username and password against BCrypt hash.
    """
    rows = get_worksheet_data(SHEET_NAME, "Users")
    if not rows: return False, None
    
    # Skip header
    for row in rows[1:]: 
        if len(row) >= 3:
            db_user = str(row[0]).strip()
            db_hash = str(row[1]).strip()
            db_role = row[2]
            
            if db_user == username:
                try:
                    # Verify password against the stored hash
                    # encode() converts string to bytes for bcrypt
                    if bcrypt.checkpw(password.encode('utf-8'), db_hash.encode('utf-8')):
                        return True, db_role
                except ValueError:
                    # Happens if the DB contains legacy plain text
                    return False, "Security Update: Old password format detected. Ask Admin to reset."
                    
    return False, None

def create_user(username, password, role):
    try:
        ws = get_worksheet_object("Users")
        existing = [r[0] for r in get_worksheet_data(SHEET_NAME, "Users")]
        
        if username in existing: 
            return False, "User exists"
        
        # --- HASHING LOGIC ---
        # Generate a salt and hash the password
        hashed_pwd = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        ws.append_row([username, hashed_pwd, role])
        st.cache_data.clear()
        return True, "User Created Successfully"
    except Exception as e: 
        return False, str(e)

# --- CONFIGS ---
def get_categories_for_marketplace(marketplace):
    rows = get_worksheet_data(SHEET_NAME, "Configs")
    cats = [row[1] for row in rows if len(row) > 1 and row[0] == marketplace]
    return list(set([c for c in cats if c and c != "Category"]))

def save_config(marketplace, category, data):
    try:
        ws = get_worksheet_object("Configs")
        json_str = json.dumps(data)
        all_vals = ws.get_all_values()
        for i, row in enumerate(all_vals):
            if len(row) > 1 and row[0] == marketplace and row[1] == category:
                ws.update_cell(i + 1, 3, json_str)
                st.cache_data.clear()
                return True
        ws.append_row([marketplace, category, json_str])
        st.cache_data.clear()
        return True
    except: return False

def load_config(marketplace, category):
    rows = get_worksheet_data(SHEET_NAME, "Configs")
    for row in rows:
        if len(row) > 2 and row[0] == marketplace and row[1] == category:
            return json.loads(row[2])
    return None

def save_seo(marketplace, category, keywords_list):
    try:
        ws = get_worksheet_object("SEO_Data")
        kw_string = ", ".join([str(k).strip() for k in keywords_list if str(k).strip()])
        all_vals = ws.get_all_values()
        for i, row in enumerate(all_vals):
            if len(row) > 1 and row[0] == marketplace and row[1] == category:
                ws.update_cell(i + 1, 3, kw_string)
                st.cache_data.clear()
                return True
        ws.append_row([marketplace, category, kw_string])
        st.cache_data.clear()
        return True
    except: return False

def get_seo(marketplace, category):
    rows = get_worksheet_data(SHEET_NAME, "SEO_Data")
    for row in rows:
        if len(row) > 2 and row[0] == marketplace and row[1] == category:
            return row[2]
    return ""

# --- NEW: DYNAMIC MARKETPLACES ---
def get_all_marketplaces():
    """
    Fetches unique Marketplaces from the Configs sheet.
    Merges them with a default list to ensure the app doesn't look empty on fresh install.
    """
    defaults = ["Myntra", "Flipkart", "Ajio", "Amazon", "Nykaa"]
    try:
        rows = get_worksheet_data(SHEET_NAME, "Configs")
        if not rows: return defaults
        
        # Column 0 is Marketplace. Skip header (row 0).
        db_mps = []
        for r in rows[1:]:
            if r and len(r) > 0 and str(r[0]).strip() != "":
                db_mps.append(str(r[0]).strip())
        
        # Combine defaults with DB values, remove duplicates, and sort
        combined = sorted(list(set(defaults + db_mps)))
        return combined
    except:
        return defaults
# --- PHASE 1: HUMAN-IN-THE-LOOP LOGGING ---
def log_training_data(marketplace, corrections_list):
    """
    Saves user corrections to a 'Training_Data' sheet.
    corrections_list = [{'img_url': '...', 'col': 'Fabric', 'ai_value': 'Cotton', 'human_value': 'Polyester'}]
    """
    try:
        # Check if sheet exists, if not, creates a header (soft check)
        ws = get_worksheet_object("Training_Data")
        
        # Prepare rows
        rows_to_append = []
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        for item in corrections_list:
            rows_to_append.append([
                timestamp, 
                marketplace, 
                item['img_url'], 
                item['col'], 
                str(item['ai_value']), 
                str(item['human_value'])
            ])
            
        if rows_to_append:
            ws.append_rows(rows_to_append)
            return True, len(rows_to_append)
        return False, 0
    except Exception as e:
        return False, str(e)
# --- PHASE 2: FINANCIAL LOGGING ---
def log_financials(marketplace, engine, skus, cost, savings):
    try:
        ws = get_worksheet_object("Usage_Ledger")
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # Format: Time, MP, Engine, SKU_Count, Real_Cost, Savings
        ws.append_row([timestamp, marketplace, engine, skus, round(cost, 4), round(savings, 4)])
        return True
    except Exception as e:
        return False

# --- APPEND TO core/database.py ---

def delete_user(username):
    """
    Removes a user row from the 'Users' worksheet.
    """
    # 1. Safety Checks
    if not username: return False, "No username selected."
    if username.lower() == "admin": return False, "Cannot delete the root Admin."
    
    try:
        ws = get_worksheet_object("Users")
        
        # 2. Find the row index
        # We fetch only the first column to minimize API load and search safely
        usernames_col = ws.col_values(1) 
        
        try:
            # list.index raises ValueError if not found
            # +1 because gspread rows are 1-indexed
            row_index = usernames_col.index(username) + 1 
        except ValueError:
            return False, "User not found in database."

        # 3. Protect Header
        if row_index == 1:
            return False, "Cannot delete the Header row."

        # 4. Execute Deletion
        ws.delete_rows(row_index)
        
        # 5. Clear Cache so the UI updates immediately
        st.cache_data.clear()
        
        return True, f"User '{username}' permanently deleted."
        
    except Exception as e:
        return False, f"GSheet Error: {str(e)}"
