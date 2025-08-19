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
st.title("ASIN → Product Exporter")
st.caption("Upload an Excel with an 'ASIN' column. Fetches product details and returns an Excel.")

CRAWLBASE_TOKEN = st.secrets.get("CRAWLBASE_TOKEN", "")
if not CRAWLBASE_TOKEN:
    st.error("⚠️ Missing CRAWLBASE_TOKEN in Streamlit secrets.")
    st.stop()

def sanitize_text(s: str) -> str:
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
                    'Brand': sanitize_text(body.get('brand')),
                    'isPrime': body.get('isPrime'),
                    'inStock': body.get('inStock'),
                    'stockDetail': body.get('stockDetail')
                }
        except Exception as e:
            return {
                'ASIN': asin, 'TITLE': None,
                'body_html': f"Exception: {e}",
                'price': None, 'highResolutionImages': None, 'Brand': None,
                'isPrime': None, 'inStock': None, 'stockDetail': None
            }

async def process_asins(token: str, domain: str, asins: list, max_concurrency: int, progress_cb=None):
    sem = asyncio.Semaphore(max_concurrency)
    results, total, done = [], len(asins), 0
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_asin(session, token, domain, a, sem) for a in asins]
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            done += 1
            if progress_cb: progress_cb(done, total)
    return results

def df_to_excel_bytes(df: pd.DataFrame, base_name: str) -> (bytes, str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"{base_name}_results_{ts}.xlsx"
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False)
    buf.seek(0)
    return buf.read(), out_name

col1, col2 = st.columns(2)
with col1:
    domain = st.text_input("Domain", value="www.amazon.com.au")
with col2:
    max_concurrency = st.slider("Max concurrent requests", 1, 20, 20)

uploaded = st.file_uploader("Upload Excel (.xlsx) with an 'ASIN' column", type=["xlsx"])

if uploaded is not None:
    try:
        df_in = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"Could not read Excel: {e}")
        st.stop()
    if 'ASIN' not in df_in.columns:
        st.error("Input file must contain 'ASIN' column.")
        st.stop()
    asins = df_in['ASIN'].dropna().astype(str).tolist()
    st.success(f"Loaded {len(asins)} ASINs.")
    if st.button("Run"):
        progress = st.progress(0); status = st.empty()
        def cb(done, total):
            progress.progress(int((done/total)*100))
            status.write(f"Processing {done}/{total}…")
        with st.spinner("Fetching product details…"):
            results = asyncio.run(process_asins(CRAWLBASE_TOKEN, domain, asins, max_concurrency, cb))
        desired = ['ASIN','TITLE','body_html','price','highResolutionImages','Brand','isPrime','inStock','stockDetail']
        df_out = pd.DataFrame(results)
        for c in desired:
            if c not in df_out.columns: df_out[c] = None
        df_out = df_out[desired]
        st.success("Done! Preview:")
        st.dataframe(df_out.head(20), use_container_width=True)
        base = os.path.splitext(uploaded.name)[0]
        xbytes, out_name = df_to_excel_bytes(df_out, base)
        st.download_button("⬇️ Download results (.xlsx)", data=xbytes, file_name=out_name,
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
