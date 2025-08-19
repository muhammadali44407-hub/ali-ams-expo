import os
import io
import re
import urllib.parse
import asyncio
import aiohttp
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="ASIN → Product Exporter", page_icon="🛒", layout="centered")

# =========================
# Secrets
# =========================
CRAWLBASE_TOKEN = st.secrets.get("CRAWLBASE_TOKEN", "")
APP_PASSWORD = st.secrets.get("APP_PASSWORD", "")

if not CRAWLBASE_TOKEN:
    st.error("⚠️ Missing CRAWLBASE_TOKEN in Streamlit secrets.")
    st.stop()
if not APP_PASSWORD:
    st.error("⚠️ Missing APP_PASSWORD in Streamlit secrets.")
    st.stop()

# =========================
# Auth (simple password gate)
# =========================
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

def do_login(pw: str):
    if pw and pw == APP_PASSWORD:
        st.session_state.auth_ok = True
        st.session_state["_pw"] = None
    else:
        st.error("Incorrect password.")

def do_logout():
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("🔒 ASIN → Product Exporter")
    st.caption("Enter password to continue.")
    pw = st.text_input("Password", type="password")
    col_l, col_r = st.columns([1,1])
    with col_l:
        if st.button("Login"):
            do_login(pw)
    st.stop()

# =========================
# App UI (behind password)
# =========================
st.title("ASIN → Product Exporter")
st.caption("Upload an Excel with an 'ASIN' column. The app fetches product details and returns an Excel file.")
st.button("Logout", on_click=do_logout)

# Controls
col1, col2 = st.columns(2)
with col1:
    domain = st.text_input("Domain", value="www.amazon.com.au")
with col2:
    max_concurrency = st.slider("Max concurrent requests", 1, 20, 20)

uploaded = st.file_uploader("Upload Excel (.xlsx) with an 'ASIN' column", type=["xlsx"])

# =========================
# Helpers
# =========================
def sanitize_text(s: str) -> str:
    """Replace the standalone word 'amazon' (any case) with 'ams'."""
    if not isinstance(s, str):
        return s
    return re.sub(r'\bamazon\b', 'ams', s, flags=re.IGNORECASE)

async def fetch_asin(session: aiohttp.ClientSession, token: str, domain: str, asin: str, sem: asyncio.Semaphore):
    amazon_url = f"https://{domain}/dp/{asin}"
    encoded_url = urllib.parse.quote(amazon_url, safe="")
    api_url = f"https://api.crawlbase.com/?token={token}&url={encoded_url}&scraper=amazon-product-details"
    async with sem:
        try:
            async with session.get(api_url, timeout=60) as resp:
                try:
                    data = await resp.json(content_type=None)
                except Exception as e:
                    text = await resp.text()
                    return {
                        'ASIN': asin, 'TITLE': None,
                        'body_html': f"Error parsing response: {e}\n{text[:500]}",
                        'price': None, 'highResolutionImages': None, 'Brand': None,
                        'isPrime': None, 'inStock': None, 'stockDetail': None
                    }
                if isinstance(data, dict) and 'error' in data:
                    return {
                        'ASIN': asin, 'TITLE': None,
                        'body_html': f"Error: {data.get('error')}",
                        'price': None, 'highResolutionImages': None, 'Brand': None,
                        'isPrime': None, 'inStock': None, 'stockDetail': None
                    }
                body = (data or {}).get('body', {}) if isinstance(data, dict) else {}
                features = body.get('features', []) or []
                description = body.get('description', '') or ''
                images = body.get('highResolutionImages', []) or []

                html_features = '<br>'.join(features) if features else ''
                html_combined = f"{html_features}<br><br>{description}" if html_features else description
                images_cleaned = ', '.join(images) if isinstance(images, list) else ''

                return {
                    'ASIN': asin,
                    'TITLE': sanitize_text(body.get('name')),
                    'body_html': sanitize_text(html_combined),
                    'price': body.get('rawPrice'),
                    'highResolutionImages': images_cleaned,
                    'Brand': sanitize_tex_
