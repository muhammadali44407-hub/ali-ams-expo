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
    "• 1 row per **Item number**\n"
    "• ASINs derived from **Custom label (SKU)** only if it starts with J17 (case-insensitive)\n"
    "• Queries valid 10-alphanumeric ASINs only (no API calls for invalid)\n"
    "• Computes eBay fee-adjusted price and compares to Amazon price\n"
    "• Includes `soldBy` (seller name) and `maximumQuantity` from Amazon"
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
# Added soldBy and maximumQuantity here:
AMZ_COLS = [
    "TITLE","body_html","price","highResolutionImages","Brand",
    "isPrime","inStock","stockDetail","soldBy","maximumQuantity"
]

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

                # NEW: soldBy and maximumQuantity (sanitized)
                sold_by = sanitize_text(body.get("soldBy"))
                maximum_quantity = body.get("maximumQuantity")

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
                    "soldBy": sold_by,
                    "maximumQuantity": maximum_quantity,
                    "_ok": True, "_msg": "OK"
                }
        except Exception as e:
            return {"ASIN": asin, "_ok": False, "_msg": f"Exception: {e}"}

async def fetch_all(asins, token, domain, max_conc, log_holder, log_buf):
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
            add_log(f"{'✅' if res.get('_ok') else '❌'} {res.get('ASIN')} • {title[:60]} • {status}", log_holder, log_buf)
            update_progress(done, total)
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

    # Validate required columns exist
    required = [ITEM_COL, TITLE_COL, SKU_COL, QTY_COL, CURR_PRICE_COL]
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.stop()

    # 1) One row per Item number
    df = df_raw.drop_duplicates(subset=[ITEM_COL], keep="first").reset_index(drop=True)

    # 2) Extract ASINs from SKU (J17 rule, case-insensitive)
    df["ASIN"] = df[SKU_COL].apply(extract_asin_from_j17)

    # 3) Build list of VALID asins only (strict 10-alnum), unique
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
            add_log(f"Skipping {skipped_count} rows (invalid/non-J17 ASIN) — no API calls made for these.", log_holder, log_buf)

        # 4) Fetch Amazon only for valid ASINs
        if len(valid_asins) == 0:
            df_amz = pd.DataFrame(columns=["ASIN"] + AMZ_COLS)
        else:
            with st.spinner("Fetching Amazon details…"):
                results = asyncio.run(
                    fetch_all(valid_asins, CRAWLBASE_TOKEN, domain, max_concurrency, log_holder, log_buf)
                )
            df_amz = pd.DataFrame(results) if results else pd.DataFrame(columns=["ASIN"] + AMZ_COLS + ["_ok","_msg"])
            if not df_amz.empty:
                # Prefer successful rows with data; simple sort heuristic
                df_amz = df_amz.sort_values(by=["_ok","price","TITLE"], ascending=[False, False, False])
                df_amz = df_amz.drop_duplicates(subset="ASIN", keep="first")

        # 5) Merge once per ASIN into the one-row-per-item table
        df_out = df.merge(
            df_amz[["ASIN"] + AMZ_COLS] if not df_amz.empty else pd.DataFrame(columns=["ASIN"] + AMZ_COLS),
            how="left",
            on="ASIN"
        )

        # 6) Fill missing Amazon fields with "not found"
        for c in AMZ_COLS:
            if c not in df_out.columns:
                df_out[c] = "not found"
            else:
                df_out[c] = df_out[c].where(df_out[c].notna(), "not found")

        # 7) Compute fee-adjusted eBay price and comparison
        df_out["Current price_num"] = df_out[CURR_PRICE_COL].apply(parse_money)
        df_out["Amazon price_num"]  = df_out["price"].apply(parse_money)

        df_out["Ebay price after fee"] = df_out["Current price_num"].apply(
            lambda x: round(x * 0.89 - 0.30, 2) if x is not None else None
        )

        def cmp(amz, ebay):
            if amz is None or ebay is None:
                return "not found"
            return bool(amz > ebay)

        df_out["Amazon price higher then ebay"] = df_out.apply(
            lambda r: cmp(r["Amazon price_num"], r["Ebay price after fee"]),
            axis=1
        )

        # 8) Reorder columns: your required set first, then the rest
        preferred = [
            ITEM_COL,
            TITLE_COL,
            SKU_COL,
            QTY_COL,
            CURR_PRICE_COL,
            "ASIN",
            "price",
            "isPrime",
            "inStock",
            "stockDetail",
            "soldBy",            # NEW
            "maximumQuantity",   # NEW
            "Ebay price after fee",
            "Amazon price higher then ebay",
        ]
        have_pref = [c for c in preferred if c in df_out.columns]
        other_cols = [c for c in df_out.columns if c not in have_pref]
        df_out = df_out[have_pref + other_cols]

        st.success(f"Done! Output rows: {len(df_out)} (1 per Item number). Preview:")
        st.dataframe(df_out.head(20), use_container_width=True)

        # 9) Download CSV
        ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{os.path.splitext(upl.name)[0]}_ENRICHED_{ts_now}"
        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv_bytes,
            file_name=f"{base_name}.csv",
            mime="text/csv",
            use_container_width=True
        )
