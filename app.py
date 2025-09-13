import os
import re
import urllib.parse
import asyncio
import aiohttp
import pandas as pd
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="SKU → ASIN Exporter (J17 only)", page_icon="📦", layout="wide")

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
    st.title("🔒 SKU → ASIN Exporter (J17 only)")
    st.caption("Enter password to continue.")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        do_login(pw)
    st.stop()

# =========================
# UI
# =========================
st.title("SKU → ASIN Exporter (J17 only)")
st.caption(
    "Processes only rows where **Custom label (SKU)** starts with `J17`. "
    "For those, removes the first 3 characters to form an ASIN and fetches data only if it matches the Amazon ASIN pattern (10 alphanumeric). "
    "All other rows are kept with Amazon fields marked as **not found**."
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
        "Upload file (.csv or .xlsx) with a **Custom label (SKU)** column",
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
ASIN_RE = re.compile(r"^[A-Za-z0-9]{10}$")
AMZ_COLS = ["TITLE","body_html","price","highResolutionImages","Brand","isPrime","inStock","stockDetail"]
SKU_COL = "Custom label (SKU)"   # back to original

def sanitize_text(s: str) -> str:
    """Replace the standalone word 'amazon' (any case) with 'ams'."""
    if not isinstance(s, str):
        return s
    return re.sub(r'\bamazon\b', 'ams', s, flags=re.IGNORECASE)

def asin_from_sku_j17(sku: str) -> str | None:
    if not isinstance(sku, str):
        return None
    sku = sku.strip()
    if not sku.startswith("J17"):
        return None
    candidate = sku[3:].strip().upper()
    if ASIN_RE.fullmatch(candidate):
        return candidate
    return None

def unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x is None:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def render_log_line(asin: str | None, status: str, title: str | None = None):
    if asin is None:
        return f"❌ (skipped) • not J17 or invalid"
    title_snip = (title or "")[:60]
    emoji = "✅" if status == "OK" else "❌"
    return f"{emoji} {asin} • {title_snip} • {status}"

def update_progress(done: int, total: int):
    pct = int((done / total) * 100) if total else 0
    progress_bar.progress(pct, text=f"Processing {done}/{total}…")
    counter_txt.write(f"Processed {done} of {total}")

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

async def fetch_all(asins, token, domain, max_conc, log_cb=None, prog_cb=None):
    sem = asyncio.Semaphore(max_conc)
    out, total, done = [], len(asins), 0
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_asin(session, token, domain, a, sem) for a in asins]
        for coro in asyncio.as_completed(tasks):
            res = await coro
            out.append(res)
            done += 1
            if log_cb:
                log_cb(render_log_line(res.get("ASIN"), res.get("_msg", ""), res.get("TITLE")))
            if prog_cb:
                prog_cb(done, total)
    return out

# =========================
# Main
# =========================
if upl is not None:
    try:
        if upl.name.lower().endswith(".csv"):
            df_in = pd.read_csv(upl)
        else:
            df_in = pd.read_excel(upl)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        st.stop()

    if SKU_COL not in df_in.columns:
        st.error(f"Input must contain a column named '{SKU_COL}'.")
        st.stop()

    df_in = df_in.copy()
    df_in["ASIN"] = df_in[SKU_COL].apply(asin_from_sku_j17)

    valid_asins = unique_preserve_order([a for a in df_in["ASIN"].tolist() if a is not None])

    st.success(
        f"Rows: {len(df_in)} | J17 ASIN candidates: {df_in['ASIN'].notna().sum()} | "
        f"Unique valid ASINs to fetch: {len(valid_asins)}"
    )

    if st.button("Run"):
        log_holder = log_box.empty()
        log_lines = []

        def add_log(line: str):
            log_lines.append(line)
            log_holder.write("\n".join(log_lines[-200:]))

        skipped = df_in[df_in["ASIN"].isna()]
        for _ in range(min(len(skipped), 200)):
            add_log(render_log_line(None, "skipped (not J17 or invalid)"))

        with st.spinner("Fetching Amazon details…"):
            results = asyncio.run(
                fetch_all(valid_asins, CRAWLBASE_TOKEN, domain, max_concurrency, log_cb=add_log, prog_cb=update_progress)
            )

        df_amz = pd.DataFrame(results) if results else pd.DataFrame(columns=["ASIN"]+AMZ_COLS+["_ok","_msg"])
        # deduplicate: keep best per ASIN
        df_amz = df_amz.sort_values(by=["_ok","price","TITLE"], ascending=[False, False, False])
        df_amz = df_amz.drop_duplicates(subset="ASIN", keep="first")

        df_merge = df_in.merge(df_amz[["ASIN"]+AMZ_COLS], how="left", on="ASIN")

        for c in AMZ_COLS:
            df_merge[c] = df_merge[c].where(df_merge[c].notna(), "not found")

        st.success("Done! Preview:")
        preview_cols = [SKU_COL, "ASIN"] + AMZ_COLS
        st.dataframe(df_merge[preview_cols].head(20), use_container_width=True)

        ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_name = f"{os.path.splitext(upl.name)[0]}_ENRICHED_{ts_now}.csv"
        csv_bytes = df_merge.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download enriched CSV", data=csv_bytes, file_name=out_name, mime="text/csv")
