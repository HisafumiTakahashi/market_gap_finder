"""Land price collection utilities."""

from __future__ import annotations

import io
import logging
import re
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd
import requests

from config import settings

logger = logging.getLogger(__name__)

_KOKUDO_API_BASE = "https://nlftp.mlit.go.jp/ksj/api/1.0b/index.php/app"
_KOKUDO_APP_ID = "ksjapibeta1"
_DIRECT_URL_TEMPLATE = (
    "https://nlftp.mlit.go.jp/ksj/gml/data/L01/L01-{year_short}/"
    "L01-{year_short}_{pref_code}_GML.zip"
)

PREFECTURE_CODES = {
    "東京都": "13",
    "大阪府": "27",
    "愛知県": "23",
}


def _get_download_url(pref_code: str, fiscal_year: int = 2024) -> str | None:
    """Fetch the official download URL via the KSJ API."""
    url = f"{_KOKUDO_API_BASE}/getGMLStList.json"
    params = {
        "appId": _KOKUDO_APP_ID,
        "dataformat": 1,
        "identifier": "L01",
        "prefCode": pref_code,
        "fiscalyear": str(fiscal_year),
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        items = data.get("GML_LIST", {}).get("GML_INF", [])
        if isinstance(items, dict):
            items = [items]
        for item in items:
            dl_url = item.get("zipFileUrl")
            if dl_url:
                logger.info("Resolved land price URL: %s", dl_url)
                return dl_url
    except Exception:
        logger.warning("Failed to resolve API download URL (pref=%s, year=%d)", pref_code, fiscal_year)
    return None


def _build_direct_url(pref_code: str, fiscal_year: int = 2024) -> str:
    """Build a direct download URL using the fiscal year suffix."""
    year_short = str(fiscal_year)[-2:]
    return _DIRECT_URL_TEMPLATE.format(year_short=year_short, pref_code=pref_code)


def _parse_land_price_gml(gml_content: bytes) -> pd.DataFrame:
    """Parse GML/XML land price data."""
    root = ET.fromstring(gml_content)
    records = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag in ("PublicLandPrice", "LandPrice", "土地価格"):
            record = _extract_price_record(elem)
            if record:
                records.append(record)

    if not records:
        records = _fallback_parse(root)

    logger.info("Parsed %d land price records from GML", len(records))
    return pd.DataFrame(records)


def _extract_price_record(elem: ET.Element) -> dict | None:
    """Extract one land price record from an XML element."""
    lat, lng, price = None, None, None

    for child in elem.iter():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        text = (child.text or "").strip()

        if tag == "pos" and text:
            parts = text.split()
            if len(parts) >= 2:
                try:
                    lat, lng = float(parts[0]), float(parts[1])
                except ValueError:
                    pass
        elif tag in ("publicLandPrice", "standardPrice", "price", "L01_006") and text:
            price_str = re.sub(r"[^\d.]", "", text)
            if price_str:
                try:
                    price = float(price_str)
                except ValueError:
                    pass

    if lat is not None and lng is not None and price is not None:
        return {"lat": lat, "lng": lng, "price_per_sqm": price}
    return None


def _fallback_parse(root: ET.Element) -> list[dict]:
    """Fallback parser for unsupported or unusual GML structures."""
    records: list[dict] = []
    for elem in root.iter():
        record = _extract_price_record(elem)
        if record:
            records.append(record)
    return records


def _find_parent_id(root: ET.Element, target: ET.Element) -> str:
    """Compatibility helper retained for legacy callers."""
    del root
    return str(id(target))


def download_land_price(pref_code: str, fiscal_year: int = 2024) -> pd.DataFrame:
    """Download and parse land price data, falling back to direct URLs."""
    candidate_urls: list[str] = []

    api_url = _get_download_url(pref_code, fiscal_year)
    if api_url:
        candidate_urls.append(api_url)

    for year in range(fiscal_year, fiscal_year - 3, -1):
        fallback_url = _build_direct_url(pref_code, year)
        if fallback_url not in candidate_urls:
            candidate_urls.append(fallback_url)

    for download_url in candidate_urls:
        try:
            logger.info("Downloading land price ZIP: %s", download_url)
            resp = requests.get(download_url, timeout=120)
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                gml_files = [name for name in zf.namelist() if name.endswith((".xml", ".gml"))]
                if not gml_files:
                    logger.warning("No GML/XML files found in ZIP: %s", download_url)
                    continue

                all_records = []
                for gml_file in gml_files:
                    content = zf.read(gml_file)
                    parsed = _parse_land_price_gml(content)
                    if not parsed.empty:
                        all_records.append(parsed)

                if all_records:
                    result = pd.concat(all_records, ignore_index=True)
                    logger.info("Downloaded %d land price records", len(result))
                    return result
        except Exception:
            logger.warning("Failed to download land price data: %s", download_url, exc_info=True)

    return pd.DataFrame(columns=["lat", "lng", "price_per_sqm"])


def save_land_price_cache(df: pd.DataFrame, tag: str) -> None:
    """Save land price cache."""
    settings.EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = settings.EXTERNAL_DATA_DIR / f"{tag}_land_price.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logger.info("Saved land price cache: %s (%d rows)", path, len(df))


def load_land_price_cache(tag: str) -> pd.DataFrame | None:
    """Load land price cache if present."""
    path = settings.EXTERNAL_DATA_DIR / f"{tag}_land_price.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    logger.info("Loaded land price cache: %s (%d rows)", path, len(df))
    return df
