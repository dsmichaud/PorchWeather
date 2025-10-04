import os
import re
import time
import json
import math
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from dateutil import tz

import streamlit as st

# ----------------------------- Config ---------------------------------------
APP_TITLE = "Porchability — Live Gust Wind-Chill & Work Windows"
LOCAL_TZ = tz.gettz("America/New_York")
WORK_HOURS = range(9, 16)   # 9,10,11,12,13,14,15
DEFAULT_THRESHOLDS = [20, 32, 40]  # °F

WL_API = "https://api.weatherlink.com/v2"
NWS_POINTS = "https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
NWS_HOURLY = "https://api.weather.gov/gridpoints/{office}/{gridX},{gridY}/forecast/hourly"

# ----------------------------- Helpers --------------------------------------
def parse_station_uuid(s: str) -> str | None:
    """
    Accepts raw UUID or a WeatherLink URL like:
      https://www.weatherlink.com/browse/9722cfc3-a4ef-47b9-befb-72f52592d6ed
    Returns the UUID string if found.
    """
    pat = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
    m = pat.search(s.strip())
    return m.group(0) if m else None

def nws_wind_chill_f(temp_f: float, wind_mph: float) -> float:
    """NWS wind-chill using gust; valid for T<=50°F and V>=3 mph; else returns T."""
    if pd.isna(temp_f) or pd.isna(wind_mph):
        return np.nan
    if temp_f <= 50 and wind_mph >= 3:
        return 35.74 + 0.6215*temp_f - 35.75*(wind_mph**0.16) + 0.4275*temp_f*(wind_mph**0.16)
    return temp_f

def twoconsecutive_mask(series_bool: pd.Series, min_len=2) -> bool:
    """True if series has at least 'min_len' consecutive True values."""
    run = 0
    for v in series_bool.astype(bool).tolist():
        run = run + 1 if v else 0
        if run >= min_len:
            return True
    return False

def wl_get(session: requests.Session, path: str, api_key: str, api_secret: str, params=None, demo=False):
    """WeatherLink v2 API GET. Adds api-key and X-Api-Secret as required."""
    params = params or {}
    params["api-key"] = api_key
    if demo:
        params["demo"] = "true"  # include demo station in your access
    headers = {"X-Api-Secret": api_secret}
    r = session.get(f"{WL_API}{path}", params=params, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()

def wl_list_stations(session, api_key, api_secret, demo=False):
    # /stations — returns stations accessible to your key
    # https://weatherlink.github.io/v2-api/api-reference  (Authentication & Demo mode docs)
    return wl_get(session, "/stations", api_key, api_secret, params=None, demo=demo)

def wl_current(session, station_id, api_key, api_secret, demo=False):
    return wl_get(session, f"/current/{station_id}", api_key, api_secret, params=None, demo=demo)

def wl_historic(session, station_id, api_key, api_secret, start_ts, end_ts, demo=False):
    return wl_get(session, f"/historic/{station_id}", api_key, api_secret,
                  params={"start-timestamp": int(start_ts), "end-timestamp": int(end_ts)}, demo=demo)

def get_station_coords(meta_json, target_uuid: str | None):
    # WeatherLink returns a list of stations with metadata including lat/lon
    stations = meta_json.get("stations", [])
    for s in stations:
        # station_id may be an int; 'station_id_uuid' is present in v2 metadata
        if str(s.get("station_id_uuid")) == str(target_uuid) or target_uuid is None:
            lat = s.get("latitude")
            lon = s.get("longitude")
            name = s.get("station_name", "Station")
            return name, lat, lon, str(s.get("station_id_uuid"))
    return None, None, None, None

def fetch_nws_hourly(lat, lon):
    # Step 1: get office/grid point for this lat/lon
    r = requests.get(NWS_POINTS.format(lat=lat, lon=lon),
                     headers={"User-Agent": "Porchability/1.0 (personal use)"},
                     timeout=20)
    r.raise_for_status()
    p = r.json()["properties"]
    office = p["gridId"]; x = p["gridX"]; y = p["gridY"]
    # Step 2: hourly forecast
    r2 = requests.get(NWS_HOURLY.format(office=office, gridX=x, gridY=y),
                      headers={"User-Agent": "Porchability/1.0 (personal use)"},
                      timeout=20)
    r2.raise_for_status()
    return r2.json()["properties"]["periods"]

def mph_from_nws_field(field):
    # NWS returns "13 mph" or None
    if not field:
        return np.nan
    m = re.search(r"([0-9.]+)", str(field))
    return float(m.group(1)) if m else np.nan

def dataframe_from_nws(periods, local_tz=LOCAL_TZ):
    rows = []
    for p in periods:
        ts = datetime.fromisoformat(p["startTime"].replace("Z","+00:00")).astimezone(local_tz)
        tF = float(p.get("temperature")) if p.get("temperature") is not None else np.nan
        wind = mph_from_nws_field(p.get("windSpeed"))
        gust = mph_from_nws_field(p.get("windGust"))
        rows.append({"ts": ts, "temp_f": tF, "wind_mph": wind, "wind_gust_mph": gust})
    df = pd.DataFrame(rows).set_index("ts").sort_index()
    v = df["wind_gust_mph"].fillna(df["wind_mph"])
    df["gust_wc_f"] = [nws_wind_chill_f(t, w) for t, w in zip(df["temp_f"], v)]
    return df

def porchability_badge(wc_f: float, thresholds=DEFAULT_THRESHOLDS):
    if pd.isna(wc_f):
        return "—", "gray"
    if wc_f >= thresholds[2]:  # 40
        return "✅ Great", "green"
    if wc_f >= thresholds[1]:  # 32
        return "✅ Good", "green"
    if wc_f >= thresholds[0]:  # 20
        return "⚠ Borderline", "orange"
    return "❄ Too cold", "red"

# ----------------------------- UI -------------------------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🪵")
st.title(APP_TITLE)
st.caption("Gust-based wind chill + your work window (9–3) = realistic outdoor laptop comfort.")

st.sidebar.header("WeatherLink v2 Access")
api_key = st.sidebar.text_input("API Key", type="password")
api_secret = st.sidebar.text_input("API Secret (header X-Api-Secret)", type="password")
station_input = st.sidebar.text_input("Station UUID or WeatherLink URL (/browse/{uuid})", value="")
use_demo = st.sidebar.checkbox("Use Demo Station (for testing)", value=False)
thresholds = st.sidebar.multiselect("Comfort thresholds (°F)", DEFAULT_THRESHOLDS, default=DEFAULT_THRESHOLDS)
st.sidebar.caption("Docs: Authentication & Demo Mode on the WeatherLink v2 developer portal.")

uuid = parse_station_uuid(station_input) if station_input else None

# Live card rows
col1, col2 = st.columns([1,1], gap="large")

with col1:
    st.subheader("Live 'Porchability' (now)")
    if use_demo and not api_key:
        st.info("Demo mode still requires a WeatherLink API key/secret (you won't need access to your own station).")
    if (api_key and api_secret) and (uuid or use_demo):
        sess = requests.Session()
        try:
            # Get stations (to find lat/lon + verify UUID)
            stations_json = wl_list_stations(sess, api_key, api_secret, demo=use_demo)
            name, lat, lon, resolved_uuid = get_station_coords(stations_json, uuid)
            if not resolved_uuid:
                st.error("Could not find station metadata. Check your UUID or your account access.")
            else:
                sid = resolved_uuid
                cur = wl_current(sess, sid, api_key, api_secret, demo=use_demo)

                # Parse current obs (WeatherLink JSON nests by sensors; we take outside temp and wind/gust)
                # Fallbacks if sensor types vary:
                temp_f = np.nan; wind = np.nan; gust = np.nan
                sensors = cur.get("sensors", [])
                for s in sensors:
                    obs = s.get("data", [{}])[-1] if s.get("data") else {}
                    # Common fields on VP2: temp_out, wind_speed_last, wind_speed_hi_last_10_min
                    for k,v in obs.items():
                        k_low = str(k).lower()
                        if "temp" in k_low and ("out" in k_low or "outside" in k_low):
                            temp_f = float(v)
                        if k_low in ("wind_speed_last", "wind_speed", "wind_avg"):
                            wind = float(v)
                        if "gust" in k_low or "hi_last_10_min" in k_low or "wind_speed_hi" in k_low:
                            try:
                                gust = float(v)
                            except Exception:
                                pass
                v = gust if (gust == gust) else wind
                wc_now = nws_wind_chill_f(temp_f, v)
                badge, color = porchability_badge(wc_now, thresholds=sorted(thresholds))

                st.metric("Station", name or "Station")
                st.metric("Now — Gust WC (°F)", None if pd.isna(wc_now) else f"{wc_now:.1f}")
                st.write(f"**Status:** :{color}[{badge}]  •  Temp {temp_f if temp_f==temp_f else '—'}°F · Gust {v if v==v else '—'} mph")

                if lat and lon:
                    st.caption(f"Lat/Lon from WeatherLink: {lat:.4f}, {lon:.4f}")
                else:
                    st.caption("Lat/Lon unavailable from station metadata.")

        except requests.HTTPError as e:
            st.error(f"WeatherLink error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")
    else:
        st.info("Enter your API key/secret and station UUID (or enable demo) to see live status.")

with col2:
    st.subheader("Next work window today (9–3)")
    if (api_key and api_secret) and (uuid or use_demo):
        try:
            sess = requests.Session()
            stations_json = wl_list_stations(sess, api_key, api_secret, demo=use_demo)
            name, lat, lon, resolved_uuid = get_station_coords(stations_json, uuid)
            if lat and lon:
                periods = fetch_nws_hourly(lat, lon)
                fc = dataframe_from_nws(periods, local_tz=LOCAL_TZ)
                # Keep today 9–3 only
                now_local = datetime.now(LOCAL_TZ)
                today = now_local.date()
                end = datetime.combine(today, datetime.min.time()).replace(tzinfo=LOCAL_TZ) + timedelta(hours=15, minutes=59)
                start = datetime.combine(today, datetime.min.time()).replace(tzinfo=LOCAL_TZ) + timedelta(hours=9)
                mask = (fc.index >= start) & (fc.index <= end)
                fc_today = fc.loc[mask]

                # Evaluate thresholds and two-hour runs
                for th in sorted(thresholds):
                    ok = fc_today["gust_wc_f"] >= th
                    has_run = twoconsecutive_mask(ok, min_len=2)
                    if has_run:
                        # find first 2-hour run block start
                        idx = ok.index.tolist()
                        vals = ok.astype(bool).tolist()
                        first = None
                        for i in range(len(vals)-1):
                            if vals[i] and vals[i+1]:
                                first = idx[i]
                                break
                        if first:
                            st.success(f"≥ {th}°F: ✅ two-hour window starts ~ {first.strftime('%-I %p')}")
                        else:
                            st.success(f"≥ {th}°F: ✅ somewhere between 9–3")
                    else:
                        st.warning(f"≥ {th}°F: no 2-hour run between 9–3 today")
            else:
                st.info("Need lat/lon from station metadata to use NWS hourly. Check your station access/UUID.")
        except Exception as e:
            st.error(f"NWS/Forecast error: {e}")
    else:
        st.info("Enter your API key/secret and station UUID (or enable demo) to check forecast windows.")

st.divider()
st.subheader("Climatology (optional: upload your hourly CSV)")
st.caption("Upload a CSV with at least a 'timestamp' column (local time) and 'gust_wc_f'. I’ll build diurnal×monthly heatmaps and a monthly two-hour-run chart (9–3 only).")

up = st.file_uploader("Upload hourly gust wind-chill CSV", type=["csv"])
if up:
    try:
        df = pd.read_csv(up, parse_dates=[0])
        df = df.set_index(df.columns[0]).sort_index()
        # Normalize column names
        df.columns = [c.strip().lower() for c in df.columns]
        if "gust_wc_f" not in df.columns:
            st.error("CSV must include 'gust_wc_f'.")
        else:
            # Diurnal × monthly frequencies
            wc = df["gust_wc_f"].dropna()
            tmp = pd.DataFrame({"wc": wc})
            tmp["month"] = tmp.index.month
            tmp["hour"] = tmp.index.hour

            def freq_table(th):
                hit = (tmp["wc"] >= th).astype(int)
                table = pd.pivot_table(pd.DataFrame({"hit": hit, "month": tmp["month"], "hour": tmp["hour"]}),
                                       index="month", columns="hour", values="hit", aggfunc="mean")
                table = table.reindex(index=range(1,13), columns=range(24))
                return table

            for th in DEFAULT_THRESHOLDS:
                st.write(f"**Diurnal–Monthly fraction of hours with gust WC ≥ {th}°F**")
                st.dataframe(freq_table(th).style.format("{:.2f}"))

            # Monthly % of days with ≥2 consecutive hours (9–3 only)
            wc_day = wc[(wc.index.hour>=9) & (wc.index.hour<=15)]
            def monthly_two_hour(series, th):
                g = (series >= th).astype(int)
                by_day = g.groupby(g.index.date).apply(lambda s: (s.rolling(2).sum() >= 2).any())
                mo = pd.Series(by_day.values, index=pd.to_datetime(by_day.index)).to_frame("ok")
                mo["month"] = mo.index.to_period("M")
                return (mo.groupby("month")["ok"].mean()*100).rename(f"≥{th}°F")
            out = pd.concat([monthly_two_hour(wc_day, th) for th in DEFAULT_THRESHOLDS], axis=1)
            st.write("**Monthly % of days with ≥2 consecutive hours meeting threshold (9–3 only)**")
            st.dataframe(out.fillna(0).round(1))
    except Exception as e:
        st.error(f"Could not parse CSV: {e}")

st.divider()
st.caption