import os
import re
import urllib.parse
import asyncio
import aiohttp
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Item → ASIN Exporter (1 row per Item)", page_icon="📦", layout="wide")

# =========================
# Secrets
# =========================
CRAWLBASE_TOKEN = st.secrets.get("CRAWLBASE_TOKEN", "")
APP_PASSWORD    = st.secrets.get("APP_PASSWORD", "")

if not CRAWLBASE_TOKEN:
    st.error("⚠️ Missing CRAWLBASE_TOKEN in Streamlit secrets.")
    st.stop()
if not APP_PASSWORD:
    st.error("⚠️ Missing APP_PASSWORD in Streamlit secrets.")
    st.stop()

# =========================
# Auth
# =========================
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

def do_login(pw: str):
    st.session_state.auth_ok = (pw == APP_PASSWORD)
    if not st.session_state.auth_ok:
        st.error("Incorrect password.")

def do_logout():
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("🔒 Item → ASIN Exporter")
    st.caption("Enter password to continue.")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        do_login(pw)
    st.stop()

# =========================
# UI
# =========================
st.title("Item → ASIN Exporter (1 row per Item number)")
st.caption(
    "• Deduplicates on **Item number** (each listing = 1 row)\n"
    "• ASINs derived from **Custom label (SKU)** only if it starts with J17 (case-insensitive)\n"
    "• Queries **valid** 10-alphanumeric ASINs only\n"
    "• Computes fee-adjusted eBay price and compares to Amazon price\n"
    "• Orders key columns first"
)

st.button("Logout", on_click=do_logout)

left, right = st.columns([3, 2])
with left:
    c1, c2 = st.columns(2)
    with c1:
        domain = st.text_input("Amazon domain", value="www.amazon.com.au")
    with c2:
        max_concurrency = st.slider("Max concurrent requests", 1, 20, 20)

    upl = st.file_uploader(
        "Upload file (.csv or .xlsx) with **Item number**, **Title**, **Custom label (SKU)**, **Available quantity**, **Current price**",
        type=["csv", "xlsx"]
    )

with right:
    st.subheader("Live log")
    log_box = st.container()
    progress_bar = st.progress(0, text="Idle")
    counter_txt = st.empty()

# =========================
# Helpers
# =========================
ITEM_COL = "Item number"
SKU_COL  = "Custom label (SKU)"
TITLE_COL = "Title"
QTY_COL   = "Available quantity"
CURR_PRICE_COL = "Current price"

ASIN_RE  = re.compile(r"^[A-Za-z0-9]{10}$")
AMZ_COLS = ["TITLE","body_html","price","highResolutionImages","Brand","isPrime","inStock","stockDetail"]

def sanitize_text(s: str) -> str:
    # Replace the standalone word 'amazon' with 'ams'
    if not isinstance(s, str):
        return s
    return re.sub(r'\bamazon\b', 'ams', s, flags=re.IGNORECASE)

def extract_asin_from_j17(sku: str) -> str | None:
    """
    From a SKU that starts with j17/J17, remove first 3 chars,
    then extract the FIRST 10-char alphanumeric token (ASIN-like).
    Return it only if it matches the ASIN pattern.
    """
    if not isinstance(sku, str):
        return None
    s = sku.strip()
    if not s.lower().startswith("j17"):
        return None
    remainder = s[3:].strip().upper()
    m = re.search(r"[A-Z0-9]{10}", remainder)
    if not m:
        return None
    cand = m.group(0)
    return cand if ASIN_RE.fullmatch(cand) else None

def unique_preserve_order(values):
    seen = set()
    out = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def add_log(line: str, holder, buf: list[str]):
    buf.append(line)
    holder.write("\n".join(buf[-200:]))

def update_progress(done: int, total: int):
    pct = int((done / total) * 100) if total else 0
    progress_bar.progress(pct, text=f"Processing {done}/{total}…")
    counter_txt.write(f"Processed {done} of {total}")

def parse_money(val):
    """
    Robust price parser: strips symbols, handles commas and decimals.
    Returns float or None.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s:
        return None
    # Keep digits, dot, comma, minus; drop everything else
    s = re.sub(r"[^0-9\.,-]", "", s)
    if s == "":
        return None
    # If both comma and dot, treat comma as thousands sep
    if "," in s and "." in s:
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        # Only comma -> decimal comma
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None

# =========================
# Crawlbase fetch (valid ASINs ONLY)
# =========================
async def fetch_asin(session: aiohtt_
