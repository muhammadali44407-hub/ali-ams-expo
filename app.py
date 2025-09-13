import os
import re
import urllib.parse
import asyncio
import aiohttp
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Item → ASIN Exporter (1 row per Item number)", page_icon="📦", layout="wide")

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
    "• Deduplicates on **Item number** (so each listing = 1 row)\n"
    "• ASINs are derived from **Custom label (SKU)** if it starts with J17\n"
    "• Only valid 10-character ASINs are queried\n"
    "• Others remain in output with Amazon fields = **not found**"
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
        "Upload file (.csv or .xlsx) with **Item number** and **Custom label (SKU)**",
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
SKU_COL = "Custom label (SKU)"
ITEM_COL = "Item number"
ASIN_RE = re.compile(r"^[A-Za-z0-9]{10}$")
AMZ_COLS = ["TITLE","body_html","price","highResolutionImages","Brand","isPrime","inStock","stockDetail"]

def sanitize_text(s: str) -> str:
    if not isinstance(s, str):
        return s
    return re.sub(r'\bamazon\b', 'ams', s, flags=re.IGNORECASE)

def extract_asin_from_j17(sku: str) -> str | None:
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

def update_progress(done: int, total: int, bar, txt):
    pct = int((done / total) * 100) if total else 0
    bar.progress(pct, text=f"Processing {done}/{total}…")
    txt.write(f"Processed {done} of {total}")

# =========================
# Crawlbase fetch
# =========================
async def fetch_asin(session: aiohttp.ClientSession, token: str, domain: str, asin: str, sem: asyncio.Semaphore):
    amazon_url = f"https://{domain}/dp/{asin}"
    encoded_url = urllib.parse.quote(amazon_url, safe="")
    api_url = f"https://api.crawlbase.com/?token={token}&url={encoded_url}&scraper=amazon-product-details"
    async with sem:
        try:
            async with session.get(api_url, timeout=60) as resp:
                data = await resp.json(content_type=None)
                if isinstance(data, dict) and data.get("error"):
                    return {"ASIN": asin, "_ok": False, "_msg": f"API error: {data.get('error')}"}
                body = (data or {}).get("body", {}) if isinstance(data, dict) else {}
                features = body.get("features", []) or []
                description = body.get("description", "") or ""
                images = body.get("highResolutionImages", []) or []
                html_features = "<br>".join(features) if features else ""
                html_combined = f"{html_features}<br><br>{description}" if html_features else description
                images_cleaned = ", ".join(images) if isinstance(images, list) else ""
                return {
                    "ASIN": asin,
                    "TITLE": sanitize_text(body.get("name")),
                    "body_html": sanitize_text(html_combined),
                    "price": body.get("rawPrice"),
                    "highResolutionImages": images_cleaned,
                    "Brand": sanitize_text(body.get("brand")),
                    "isPrime": body.get("isPrime"),
                    "inStock": body.get("inStock"),
                    "stockDetail": body.get("stockDetail"),
                    "_ok": True, "_msg": "OK"
                }
        except Exception as e:
            return {"ASIN": asin, "_ok": False, "_msg": f"Exception: {e}"}

async def fetch_all(asins, token, domain, max_conc, log_holder, log_buf, bar, txt):
    sem = asyncio.Semaphore(max_conc)
    out, total, done = [], len(asins), 0
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_asin(session, token, domain, a, sem) for a in asins]
        for coro in asyncio.as_completed(tasks):
            res = await coro
            out.append(res)
            done += 1
            status = res.get("_msg", "")
            title  = res.get("TITLE") or ""
            add_log(f"{'✅' if res.get('_ok') else '❌'} {res.get('ASIN')} • {title[:60]} • {status}",
                    log_holder, log_buf)
            update_progress(done, total, bar, txt)
    return out

# =========================
# Main
# =========================
if upl is not None:
    # Read CSV/XLSX
    try:
        if upl.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(upl)
        else:
            df_raw = pd.read_excel(upl)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    if ITEM_COL not in df_raw.columns or SKU_COL not in df_raw.columns:
        st.error("Input must contain both 'Item number' and 'Custom label (SKU)' columns.")
        st.stop()

    # Deduplicate by Item number
    df = df_raw.drop_duplicates(subset=[ITEM_COL], keep="first").reset_index(drop=True)

    # Extract ASINs from SKU
    df["ASIN"] = df[SKU_COL].apply(extract_asin_from_j17)

    valid_asins = [a for a in df["ASIN"].tolist() if isinstance(a, str) and ASIN_RE.fullmatch(a)]
    valid_asins = unique_preserve_order(valid_asins)

    skipped_count = int(df["ASIN"].isna().sum())
    st.info(
        f"Input rows: {len(df_raw)} | Unique Item numbers: {df[ITEM_COL].nunique()} | "
        f"Valid ASINs to fetch: {len(valid_asins)} | Skipped (invalid/non-J17): {skipped_count}"
    )

    if st.button("Run"):
        log_holder = log_box.empty()
        log_buf = []

        if skipped_count > 0:
            add_log(f"Skipping {skipped_count} rows (invalid/non-J17 ASIN)", log_holder, log_buf)

        if len(valid_asins) == 0:
            df_out = df.copy()
            for c in AMZ_COLS:
                df_out[c] = "not found"
        else:
            with st.spinner("Fetching Amazon details…"):
                results = asyncio.run(
                    fetch_all(valid_asins, CRAWLBASE_TOKEN, domain, max_concurrency, log_holder, log_buf, progress_bar, counter_txt)
                )

            df_amz = pd.DataFrame(results) if results else pd.DataFrame(columns=["ASIN"] + AMZ_COLS + ["_ok","_msg"])
            if not df_amz.empty:
                df_amz = df_amz.sort_values(by=["_ok","price","TITLE"], ascending=[False, False, False])
                df_amz = df_amz.drop_duplicates(subset="ASIN", keep="first")

            df_out = df.merge(
                df_amz[["ASIN"] + AMZ_COLS] if not df_amz.empty else pd.DataFrame(columns=["ASIN"]+AMZ_COLS),
                how="left", on="ASIN"
            )
            for c in AMZ_COLS:
                if c not in df_out.columns:
                    df_out[c] = "not found"
                else:
                    df_out[c] = df_out[c].where(df_out[c].notna(), "not found")

        st.success(f"Done! Output rows: {len(df_out)} (1 row per Item number)")
        preview_cols = [ITEM_COL, SKU_COL, "ASIN"] + AMZ_COLS
        have = [c for c in preview_cols if c in df_out.columns]
        st.dataframe(df_out[have].head(20), use_container_width=True)

        ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{os.path.splitext(upl.name)[0]}_ENRICHED_{ts_now}.csv"
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download enriched CSV", data=csv_bytes, file_name=out_name, mime="text/csv", use_container_width=True)
