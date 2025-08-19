import os
import io
import re
import json
import urllib.parse
import asyncio
import aiohttp
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import List, Dict, Callable

st.set_page_config(page_title="ASIN → Product Exporter", page_icon="🛒", layout="wide")

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
    else:
        st.error("Incorrect password.")

def do_logout():
    st.session_state.auth_ok = False

if not st.session_state.auth_ok:
    st.title("🔒 ASIN → Product Exporter")
    st.caption("Enter password to continue.")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        do_login(pw)
    st.stop()

# =========================
# History storage (ephemeral)
# =========================
HISTORY_DIR = "history"
INDEX_PATH = os.path.join(HISTORY_DIR, "index.json")

os.makedirs(HISTORY_DIR, exist_ok=True)
if not os.path.exists(INDEX_PATH):
    with open(INDEX_PATH, "w") as f:
        json.dump([], f)

def load_history() -> List[Dict]:
    try:
        with open(INDEX_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_history_entry(entry: Dict):
    entries = load_history()
    entries.insert(0, entry)  # newest first
    with open(INDEX_PATH, "w") as f:
        json.dump(entries, f, indent=2)

def ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_uploaded_copy(uploaded_file) -> str:
    stamp = ts()
    base = os.path.splitext(uploaded_file.name)[0]
    dest = os.path.join(HISTORY_DIR, f"{base}_INPUT_{stamp}.xlsx")
    with open(dest, "wb") as out:
        out.write(uploaded_file.getvalue())
    return dest

def save_output_excel(df: pd.DataFrame, base_name: str) -> str:
    out_name = os.path.join(HISTORY_DIR, f"{base_name}_RESULTS_{ts()}.xlsx")
    df.to_excel(out_name, index=False)
    return out_name

# =========================
# Layout
# =========================
st.title("ASIN → Product Exporter")
st.caption("Upload an Excel with an **ASIN** column. Get back an Excel with product details.")
top_bar = st.container()
col_left, col_right = st.columns([3, 2])

with top_bar:
    st.button("Logout", on_click=do_logout)

# Sidebar: History viewer
with st.sidebar:
    st.header("🗂️ History")
    hist = load_history()
    if not hist:
        st.caption("No runs yet.")
    else:
        for i, item in enumerate(hist[:20]):  # show last 20
            with st.expander(f"Run {item.get('timestamp')} • {item.get('count',0)} ASINs"):
                st.caption(f"Domain: {item.get('domain')}")
                in_path = item.get("input_path")
                out_path = item.get("output_path")
                if in_path and os.path.exists(in_path):
                    with open(in_path, "rb") as f:
                        st.download_button("⬇️ Download input", f.read(),
                                           file_name=os.path.basename(in_path),
                                           key=f"in_{i}")
                if out_path and os.path.exists(out_path):
                    with open(out_path, "rb") as f:
                        st.download_button("⬇️ Download output", f.read(),
                                           file_name=os.path.basename(out_path),
                                           key=f"out_{i}")

# =========================
# Controls + File upload
# =========================
with col_left:
    c1, c2 = st.columns(2)
    with c1:
        domain = st.text_input("Domain", value="www.amazon.com.au")
    with c2:
        max_concurrency = st.slider("Max concurrent requests", 1, 20, 20)

    uploaded = st.file_uploader("Upload Excel (.xlsx) with an 'ASIN' column", type=["xlsx"])

# Right column: live log & progress
with col_right:
    st.subheader("Live log")
    log_box = st.container()
    progress_bar = st.progress(0, text="Idle")
    counter_txt = st.empty()

# =========================
# Helpers / Scraper
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
                        'body_html': f"Error parsing response: {e}\n{text[:300]}",
                        'price': None, 'highResolutionImages': None, 'Brand': None,
                        'isPrime': None, 'inStock': None, 'stockDetail': None,
                        '_ok': False, '_msg': f"Parse error: {e}"
                    }
                if isinstance(data, dict) and 'error' in data:
                    return {
                        'ASIN': asin, 'TITLE': None,
                        'body_html': f"Error: {data.get('error')}",
                        'price': None, 'highResolutionImages': None, 'Brand': None,
                        'isPrime': None, 'inStock': None, 'stockDetail': None,
                        '_ok': False, '_msg': f"API error: {data.get('error')}"
                    }
                body = (data or {}).get('body', {}) if isinstance(data, dict) else {}
                features = body.get('features', []) or []
                description = body.get('description', '') or ''
                images = body.get('highResolutionImages', []) or []
                html_features = '<br>'.join(features) if features else ''
                html_combined = f"{html_features}<br><br>{description}" if html_features else description
                images_cleaned = ', '.join(images) if isinstance(images, list) else ''
                result = {
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
                result['_ok'] = True
                result['_msg'] = "OK"
                return result
        except Exception as e:
            return {
                'ASIN': asin, 'TITLE': None,
                'body_html': f"Exception: {e}",
                'price': None, 'highResolutionImages': None, 'Brand': None,
                'isPrime': None, 'inStock': None, 'stockDetail': None,
                '_ok': False, '_msg': f"Exception: {e}"
            }

async def process_asins(
    token: str,
    domain: str,
    asins: List[str],
    max_concurrency: int,
    progress_cb: Callable[[int, int], None],
    log_cb: Callable[[Dict], None]
):
    sem = asyncio.Semaphore(max_concurrency)
    results, total, done = [], len(asins), 0
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_asin(session, token, domain, a, sem) for a in asins]
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            done += 1
            if log_cb:
                log_cb(res)
            if progress_cb:
                progress_cb(done, total)
    return results

def render_log_line(entry: Dict):
    ok = entry.get("_ok", False)
    asin = entry.get("ASIN", "")
    title = entry.get("TITLE") or ""
    msg = entry.get("_msg", "")
    stock = entry.get("inStock")
    prime = entry.get("isPrime")
    emoji = "✅" if ok else "❌"
    title_snip = title[:60] + ("…" if title and len(title) > 60 else "")
    stock_txt = "inStock" if stock else "outOfStock" if stock is not None else "stock?"
    prime_txt = "Prime" if prime else "NoPrime" if prime is not None else "prime?"
    return f"{emoji} {asin} • {title_snip} • {stock_txt} • {prime_txt} • {msg}"

def update_progress(done: int, total: int):
    pct = int((done / total) * 100) if total else 0
    progress_bar.progress(pct, text=f"Processing {done}/{total}…")
    counter_txt.write(f"Processed {done} of {total}")

# =========================
# Main action
# =========================
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
        # Save a copy of the uploaded file to history
        input_copy_path = save_uploaded_copy(uploaded)

        # Prepare live log UI
        log_holder = log_box.empty()
        log_lines: List[str] = []
        def log_cb(res: Dict):
            line = render_log_line(res)
            log_lines.append(line)
            # show last 200 lines
            log_holder.write("\n".join(log_lines[-200:]))

        with st.spinner("Fetching product details…"):
            results = asyncio.run(
                process_asins(
                    CRAWLBASE_TOKEN,
                    domain,
                    asins,
                    max_concurrency,
                    progress_cb=update_progress,
                    log_cb=log_cb
                )
            )

        desired = ['ASIN','TITLE','body_html','price','highResolutionImages','Brand','isPrime','inStock','stockDetail']
        df_out = pd.DataFrame(results)
        for c in desired:
            if c not in df_out.columns:
                df_out[c] = None
        df_out = df_out[desired]

        st.success("Done! Preview:")
        st.dataframe(df_out.head(20), use_container_width=True)

        # Save output + add to history
        base = os.path.splitext(uploaded.name)[0]
        output_path = save_output_excel(df_out, base)
        save_history_entry({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "domain": domain,
            "count": len(asins),
            "input_path": input_copy_path,
            "output_path": output_path
        })

        # Download button for this run
        with open(output_path, "rb") as f:
            st.download_button(
                "⬇️ Download results (.xlsx)",
                data=f.read(),
                file_name=os.path.basename(output_path),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
