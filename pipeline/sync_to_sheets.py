
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd

# Sheet config
SHEET_ID = "1tt8_mmJb2wixl-gTcXLE_oqIVgh2vs_2J8ZM9lhfL2Q"
SHEET_NAME = "trade_log"
CREDENTIALS_PATH = "resources/joystrategy-btc-11536b5d09b1.json"

# 將單筆交易記錄寫入 Google Sheet
def sync_trade_to_sheet(trade_dict):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_PATH, scope)
    client = gspread.authorize(creds)

    sheet = client.open_by_key(SHEET_ID)
    worksheet = sheet.worksheet(SHEET_NAME)

    row = [
        trade_dict.get("entry_time"),
        trade_dict.get("exit_time"),
        trade_dict.get("entry_price"),
        trade_dict.get("exit_price"),
        trade_dict.get("return"),
        trade_dict.get("holding_minutes"),
        trade_dict.get("horizon"),
        trade_dict.get("side", "LONG"),
    ]
    worksheet.append_row(row, value_input_option="USER_ENTERED")
