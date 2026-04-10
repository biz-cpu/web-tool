"""
ローカライゼーション用座標変換
平面直角座標（1〜19系）↔ 緯度経度
Kawase (2011) 高精度ガウス・クリューゲル投影
ジオイドモデル: JPGEO2024（国土地理院API）
"""

import math
import io
import csv
import json
import re
import time
import requests
import streamlit as st
import pandas as pd
import pydeck as pdk

# ═══════════════════════════════════════════════════════
# 1. 定数・定義
# ═══════════════════════════════════════════════════════

DEG = math.pi / 180
RAD = 180 / math.pi

DATUMS = {
    "JGD2024": {"label": "JGD2024（測地成果2024）",   "a": 6378137.0,   "f": 1/298.257222101,
                "toWGS84": {"dx":0,"dy":0,"dz":0,"rx":0,"ry":0,"rz":0,"ds":0}},
    "JGD2011": {"label": "JGD2011（測地成果2011）",   "a": 6378137.0,   "f": 1/298.257222101,
                "toWGS84": {"dx":0,"dy":0,"dz":0,"rx":0,"ry":0,"rz":0,"ds":0}},
    "JGD2000": {"label": "JGD2000（測地成果2000）",   "a": 6378137.0,   "f": 1/298.257222101,
                "toWGS84": {"dx":0,"dy":0,"dz":0,"rx":0,"ry":0,"rz":0,"ds":0}},
    "WGS84":   {"label": "WGS84",                     "a": 6378137.0,   "f": 1/298.257223563,
                "toWGS84": None},
    "TOKYO":   {"label": "旧日本測地系（Tokyo97）",   "a": 6377397.155, "f": 1/299.1528128,
                "toWGS84": {"dx":-148,"dy":507,"dz":685,"rx":0,"ry":0,"rz":0,"ds":0}},
}
# JGD2024 補足: 測地成果2024（令和6年告示）
# JGD2011との差は数cm以内。準拠楕円体はGRS80（JGD2011と同一）。
# 実用上はJGD2011と同パラメータで問題なし（プレート運動補正は別途必要な場合のみ）。

JPC_ORIGINS = {
     1:(33.0,129.5),        2:(33.0,131.0),        3:(36.0,132.166666667),
     4:(33.0,133.5),        5:(36.0,134.333333333), 6:(36.0,136.0),
     7:(36.0,137.166666667),8:(36.0,138.5),         9:(36.0,139.833333333),
    10:(40.0,140.833333333),11:(44.0,140.25),       12:(44.0,142.25),
    13:(44.0,144.25),       14:(26.0,142.0),        15:(26.0,127.5),
    16:(26.0,124.0),        17:(26.0,131.0),        18:(20.0,136.0),
    19:(26.0,154.0),
}

# 各系のカバー範囲（緯度経度の近似BBox）- 系候補提案に使用
JPC_ZONE_BBOX = {
     1: (30.5, 34.5, 127.5, 131.5),   # 長崎・鹿児島
     2: (30.5, 35.0, 129.5, 133.0),   # 福岡・佐賀・熊本 等
     3: (33.5, 36.5, 130.5, 134.0),   # 山口・島根・広島
     4: (32.5, 35.0, 132.0, 135.5),   # 香川・愛媛・徳島・高知
     5: (34.0, 37.0, 133.0, 136.0),   # 兵庫・鳥取・岡山
     6: (33.5, 37.5, 134.5, 137.5),   # 京都・大阪・福井 等
     7: (35.0, 38.5, 135.5, 138.5),   # 石川・富山・岐阜・愛知
     8: (35.0, 38.5, 137.0, 140.5),   # 新潟・長野・山梨・静岡
     9: (34.5, 40.5, 138.5, 141.5),   # 東京・神奈川・千葉 等
    10: (37.5, 42.0, 139.5, 142.5),   # 青森・秋田・山形・岩手・宮城
    11: (41.0, 45.5, 138.5, 142.0),   # 小樽・函館 等
    12: (42.0, 46.0, 140.5, 143.5),   # 札幌・旭川 等
    13: (42.5, 46.0, 142.5, 146.5),   # 網走・北見・釧路・帯広
    14: (23.0, 28.0, 140.0, 144.5),   # 諸島
    15: (24.0, 27.5, 126.0, 130.0),   # 沖縄本島
    16: (24.0, 26.5, 122.5, 126.0),   # 石垣・宮古
    17: (25.5, 27.5, 129.0, 133.0),   # 大東諸島
    18: (19.0, 21.5, 134.0, 138.0),   # 沖ノ鳥島
    19: (23.5, 25.5, 152.5, 155.5),   # 南鳥島
}

def suggest_zone_from_latlon(lat: float, lon: float) -> list[int]:
    """緯度経度から候補系番号リストを返す（BBox内に入る系）"""
    candidates = []
    for zone, (lat_s, lat_n, lon_w, lon_e) in JPC_ZONE_BBOX.items():
        if lat_s <= lat <= lat_n and lon_w <= lon <= lon_e:
            candidates.append(zone)
    if not candidates:
        # BBox外の場合は最近傍の系を返す
        def dist(zone):
            la0, lo0 = JPC_ORIGINS[zone]
            return (lat - la0)**2 + (lon - lo0)**2
        candidates = [min(JPC_ORIGINS.keys(), key=dist)]
    return sorted(candidates)

def suggest_zone_from_jpc(x: float, y: float) -> list[int]:
    """平面直角座標値から系候補を返す（原点からの距離が近い系）"""
    # 平面直角座標の典型範囲: X が -700km〜+700km、Y が -400km〜+400km
    # 距離が近い上位3系を候補として返す
    def score(zone):
        # X方向の重みを少し大きく（X範囲の方が大きいため）
        return abs(x) / 800000 + abs(y) / 400000
    # X,Yが極端に大きければ範囲外の可能性
    if abs(x) > 800000 or abs(y) > 400000:
        return []
    # 全系とも同じ変換式なので原点からの数値的な近さで候補を絞れない
    # → 入力中の系番号をそのまま使うのが正解、警告のみ返す
    return []

JPC_ZONE_LABELS = {
     1:"1系 — 長崎・鹿児島（一部）",
     2:"2系 — 福岡・佐賀・熊本・大分・宮崎・鹿児島（一部）",
     3:"3系 — 山口・島根・広島",
     4:"4系 — 香川・愛媛・徳島・高知",
     5:"5系 — 兵庫・鳥取・岡山",
     6:"6系 — 京都・大阪・福井・滋賀・三重・奈良・和歌山",
     7:"7系 — 石川・富山・岐阜・愛知",
     8:"8系 — 新潟・長野・山梨・静岡",
     9:"9系 — 東京・神奈川・千葉・埼玉・茨城・栃木・群馬・福島",
    10:"10系 — 青森・秋田・山形・岩手・宮城",
    11:"11系 — 小樽・函館・旭川（一部）",
    12:"12系 — 札幌・旭川（一部）・室蘭（一部）",
    13:"13系 — 網走・北見・釧路・帯広",
    14:"14系 — 諸島",
    15:"15系 — 沖縄本島",
    16:"16系 — 石垣・宮古",
    17:"17系 — 大東諸島",
    18:"18系 — 沖ノ鳥島",
    19:"19系 — 南鳥島",
}

GEOID_MODELS = {
    "JPGEO2024": "ジオイドモデル2024（国土地理院・推奨）",
    "JPGEO2011": "ジオイドモデル2011（国土地理院）",
    "NONE":      "ジオイド補正なし（標高≈楕円体高）",
}

# 出力フォーマット（入力にも共用）
OUTPUT_FORMATS = {
    "DD.DDDDDDDD°（十進角度）":  "decimal",
    "DD°MM′SS.SSS″（度分秒）":  "dms",
    "NDD°MM′SS.SSS″（方位角）": "bearing",
    "DD.MMSSSSSS（度分秒圧縮）": "ddmmssss",
    "Gons（グラード）":           "gons",
}

# 地図タイル
MAP_STYLES = {
    "標準地図（OpenStreetMap）": "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
    "航空写真（Esri World Imagery）": "mapbox://styles/mapbox/satellite-streets-v11",
}
# Esri Tileの代替（mapbox不要）
ESRI_TILE  = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
OSM_TILE   = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

# ═══════════════════════════════════════════════════════
# 2. Kawase (2011) 高精度変換
# ═══════════════════════════════════════════════════════

_a=6378137.0; _f=1/298.257222101; _m0=0.9999
_n=_f/(2-_f); _n2=_n**2; _n3=_n**3; _n4=_n**4
_A=_a/(1+_n)*(1+_n2/4+_n4/64)
_e=math.sqrt(2*_f-_f*_f)
_alpha=[0,_n/2-2*_n2/3+5*_n3/16+41*_n4/180,
          13*_n2/48-3*_n3/5+557*_n4/1440,
          61*_n3/240-103*_n4/140,49561*_n4/161280]
_beta=[0,_n/2-2*_n2/3+37*_n3/96-_n4/360,
         _n2/48+_n3/15-437*_n4/1440,
         17*_n3/480-37*_n4/840,4397*_n4/161280]
_delta=[0,2*_n-2*_n2/3-2*_n3+116*_n4/45,
          7*_n2/3-8*_n3/5-227*_n4/45,
          56*_n3/15-136*_n4/35,4279*_n4/630]
_c2=3*_n/2-9*_n3/16; _c4=15*_n2/16-15*_n4/32
_c6=35*_n3/48; _c8=315*_n4/512

def _S(phi):
    return _A*_m0*(phi-_c2*math.sin(2*phi)+_c4*math.sin(4*phi)
                      -_c6*math.sin(6*phi)+_c8*math.sin(8*phi))

@st.cache_data(show_spinner=False)
def latlon_to_jpc(lat_deg, lon_deg, zone):
    if zone not in JPC_ORIGINS: return None
    la0,lo0=JPC_ORIGINS[zone]
    phi=lat_deg*DEG; lam=lon_deg*DEG; phi0=la0*DEG; lam0=lo0*DEG
    sinP=math.sin(phi); psi=math.atanh(sinP)-_e*math.atanh(_e*sinP); dl=lam-lam0
    xi_=math.atan2(math.sinh(psi),math.cos(dl)); eta_=math.atanh(math.sin(dl)/math.cosh(psi))
    xi=xi_+sum(_alpha[j]*math.sin(2*j*xi_)*math.cosh(2*j*eta_) for j in range(1,5))
    eta=eta_+sum(_alpha[j]*math.cos(2*j*xi_)*math.sinh(2*j*eta_) for j in range(1,5))
    return _m0*_A*xi-_S(phi0), _m0*_A*eta

@st.cache_data(show_spinner=False)
def jpc_to_latlon(X, Y, zone):
    if zone not in JPC_ORIGINS: return None
    la0,lo0=JPC_ORIGINS[zone]; phi0=la0*DEG; lam0=lo0*DEG
    xi=(X+_S(phi0))/(_m0*_A); eta=Y/(_m0*_A)
    xi_=xi-sum(_beta[j]*math.sin(2*j*xi)*math.cosh(2*j*eta) for j in range(1,5))
    eta_=eta-sum(_beta[j]*math.cos(2*j*xi)*math.sinh(2*j*eta) for j in range(1,5))
    chi=math.asin(min(1.0,max(-1.0,math.sin(xi_)/math.cosh(eta_))))
    phi=chi+sum(_delta[j]*math.sin(2*j*chi) for j in range(1,5))
    lam=lam0+math.atan2(math.sinh(eta_),math.cos(xi_))
    return phi*RAD, lam*RAD

# ═══════════════════════════════════════════════════════
# 3. ジオイド高 API
# ═══════════════════════════════════════════════════════

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_geoid(lat: float, lon: float, model: str = "JPGEO2024"):
    """国土地理院ジオイド高API（24hキャッシュ）。失敗時は最大2回リトライ。"""
    if model == "NONE": return 0.0
    select = "0" if model == "JPGEO2024" else "1"
    url = (
        "https://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh/cgi/geoidcalc.pl"
        f"?select={select}&tanni=1&outputType=json&latitude={lat:.8f}&longitude={lon:.8f}"
    )
    for attempt in range(3):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return float(r.json()["OutputData"]["geoidHeight"])
        except Exception:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))  # 1.5s, 3.0s
    return None

# ═══════════════════════════════════════════════════════
# 4. 角度フォーマット（入力パース + 出力フォーマット）
# ═══════════════════════════════════════════════════════

def fmt_decimal(dd):
    return f"{dd:.8f}"

def fmt_dms(dd):
    sg="-" if dd<0 else ""; a=abs(dd)
    d=int(a); m=int((a-d)*60); s=(a-d-m/60)*3600
    return f"{sg}{d}°{m:02d}′{s:08.5f}″"

def fmt_bearing(dd):
    a=abs(dd); d=int(a); m=int((a-d)*60); s=(a-d-m/60)*3600
    return f"{'N' if dd>=0 else 'S'}{d}°{m:02d}′{s:08.5f}″"

def fmt_ddmmssss(dd):
    sg="-" if dd<0 else ""; a=abs(dd)
    d=int(a); m=int((a-d)*60); s=(a-d-m/60)*3600
    ss=f"{s:09.6f}".replace(".","")
    return f"{sg}{d}.{m:02d}{ss}"

def fmt_gons(dd):
    return f"{dd*10/9:.8f}"

def format_angle(dd, fk):
    return {"decimal":fmt_decimal,"dms":fmt_dms,"bearing":fmt_bearing,
            "ddmmssss":fmt_ddmmssss,"gons":fmt_gons}.get(fk, fmt_decimal)(dd)

# ── 入力パーサー（各フォーマット → 十進度）──
def parse_angle(val: str, fk: str) -> float:
    """フォーマットキーに応じた文字列を十進角度に変換。失敗時 ValueError。"""
    s = val.strip()
    if not s:
        raise ValueError("空欄です")

    if fk == "decimal":
        return float(s.replace("°",""))

    if fk == "dms":
        # DD°MM′SS.SSS″ or DD d MM m SS.SSS s
        m = re.match(r'(-?\d+)[°d°]\s*(\d+)[′\'m′]\s*([\d.]+)', s)
        if not m:
            raise ValueError(f"度分秒形式が不正: {s}")
        sign = -1 if float(m.group(1)) < 0 else 1
        return sign*(abs(float(m.group(1)))+float(m.group(2))/60+float(m.group(3))/3600)

    if fk == "bearing":
        # NDD°MM′SS.SSS″ or SDD°...
        m = re.match(r'([NS])\s*(\d+)[°d°]\s*(\d+)[′\'m′]\s*([\d.]+)', s, re.IGNORECASE)
        if not m:
            raise ValueError(f"方位角形式が不正: {s}")
        dd = float(m.group(2))+float(m.group(3))/60+float(m.group(4))/3600
        return dd if m.group(1).upper()=="N" else -dd

    if fk == "ddmmssss":
        # DD.MMSSSSSS: 整数部=度, 小数先2桁=分, 残り=秒(整数2桁+小数部)
        sign = -1 if s.startswith("-") else 1
        s2 = s.lstrip("-")
        if "." not in s2:
            return sign * float(s2)
        int_part, dec_part = s2.split(".")
        d = int(int_part)
        dec_part = dec_part.ljust(10, "0")
        mm = int(dec_part[:2])
        ss_raw = dec_part[2:]
        ss = float(ss_raw[:2] + "." + ss_raw[2:]) if len(ss_raw) >= 2 else float(ss_raw or "0")
        return sign * (d + mm/60 + ss/3600)

    if fk == "gons":
        return float(s.replace("gon","").strip())*9/10

    return float(s)


def csv_row(*fields) -> str:
    """CSV行を生成。点名にカンマ等が含まれても正しくクォートする。"""
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="")
    writer.writerow([str(f) for f in fields])
    return buf.getvalue()


def _csv_filename_ui(key: str, placeholder: str) -> str | None:
    """
    ファイル名入力欄を描画し、確定したファイル名（.csv付き）を返す。
    空欄の場合は None を返しダウンロードボタンを無効化する。
    使用可能文字: 大小英数字・漢字・ひらがな・カタカナ・ハイフン・アンダースコア・スペース
    NG文字: バックスラッシュ / コロン / アスタリスク / 疑問符 / 引用符 / 不等号 / パイプ
    """
    import re as _re
    raw = st.text_input(
        "📄 ファイル名（.csv）",
        value=st.session_state.get(key, ""),
        key=key + "_widget",
        placeholder=placeholder,
        help="拡張子 .csv は自動で付きます。空欄のままだとダウンロードできません。",
    )
    # NG文字を除去（OSで使えない文字）
    cleaned = _re.sub(r'[/\\:*?"<>|]', "", raw).strip()
    st.session_state[key] = cleaned
    if not cleaned:
        st.warning("⚠️ ファイル名を入力してください", icon="📄")
        return None
    return cleaned + ".csv"


def auto_parse_angle(val: str) -> tuple[float, str]:
    """
    5フォーマットを自動判別して (十進角度, フォーマットキー) を返す。
    失敗時は ValueError。

    判別優先順位:
      1. bearing  : 先頭が N / S（大小文字不問）
      2. dms      : °′″ の Unicode 記号を含む
      3. gons     : 末尾に gon / gons / g / gr キーワード（大小無視）
      4. ddmmssss : DD.MMSSSSSS 構造（小数6桁以上 かつ 分0-59・秒整数0-59）
      5. decimal  : それ以外の純数値
    """
    s = val.strip()
    if not s:
        raise ValueError("空欄")

    # 1. bearing: 先頭 N or S
    if re.match(r"^[NSns]", s):
        return parse_angle(s, "bearing"), "bearing"

    # 2. dms: Unicode度分秒記号（° ′ ″）を含む ← 半角英字 d/m/s は除外して誤判定防止
    if any(c in s for c in ("°", "′", "″", "°", "′", "″")):
        return parse_angle(s, "dms"), "dms"

    # 3. gons: 末尾に gon/gons/g/gr（純数値以外）
    if re.search(r"(?i)(gons?|gr?)\s*$", s):
        return parse_angle(s, "gons"), "gons"

    # 4. ddmmssss: DD.MMSSSSSS（小数12桁以上 かつ 分0-59・秒0-59）
    # 十進角度との誤判定を防ぐため、小数部12桁以上を必須条件とする
    # 例: 35.404052440000 (14桁) → ddmmssss確定
    #     140.55591438   (8桁)  → decimal（十進角度として扱う）
    m = re.match(r"^(-?)(\d{1,3})\.(\d{12,})$", s)
    if m:
        deg_int = int(m.group(2))
        dec_str = m.group(3).ljust(14, "0")
        mm_val  = int(dec_str[:2])
        ss_int  = int(dec_str[2:4])
        if deg_int <= 180 and 0 <= mm_val <= 59 and ss_int <= 59:
            return parse_angle(s, "ddmmssss"), "ddmmssss"

    # 5. decimal: 純数値（° 記号付きも可）
    try:
        return float(s.replace("°", "")), "decimal"
    except ValueError:
        pass

    raise ValueError(f"角度フォーマットを判別できません: {s!r}")

# ── 各フォーマットのプレースホルダー ──
FORMAT_PLACEHOLDER = {
    "decimal":  "35.68123456",
    "dms":      "35°40′52.44″",
    "bearing":  "N35°40′52.44″",
    "ddmmssss": "35.404052440000",
    "gons":     "39.64590684",
}

# ═══════════════════════════════════════════════════════
# 5. pydeck 地図描画ユーティリティ
# ═══════════════════════════════════════════════════════

# カラーパレット（点ごとに色を変える）
PIN_COLORS = [
    [239, 68,  68 ],  # red
    [59,  130, 246],  # blue
    [16,  185, 129],  # green
    [245, 158, 11 ],  # amber
    [139, 92,  246],  # purple
    [236, 72,  153],  # pink
    [20,  184, 166],  # teal
    [249, 115, 22 ],  # orange
    [99,  102, 241],  # indigo
    [34,  197, 94 ],  # emerald
]

def render_map(points: list[dict], map_style_key: str, zoom: int = 13):
    """Leaflet.js ベースの地図を st.components.v1.html で描画。
    APIキー不要。標準地図=OSM, 航空写真=Esri World Imagery。"""
    if not points:
        return
    import streamlit.components.v1 as components

    is_aerial = "航空" in map_style_key
    tile_url  = ESRI_TILE if is_aerial else OSM_TILE
    attr      = ("&copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, "
                 "Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
                 if is_aerial
                 else '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors')

    clat = sum(p["lat"] for p in points) / len(points)
    clon = sum(p["lon"] for p in points) / len(points)
    z    = zoom if len(points) == 1 else max(8, zoom - 3)

    # ピンデータをJSONに
    pins_js = json.dumps([
        {
            "name": p["name"],
            "lat":  p["lat"],
            "lon":  p["lon"],
            "tip":  p.get("tooltip", p["name"]),
            "color": "#{:02x}{:02x}{:02x}".format(*PIN_COLORS[i % len(PIN_COLORS)]),
        }
        for i, p in enumerate(points)
    ])

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body{{margin:0;padding:0;height:100%;}}#map{{height:420px;width:100%;border-radius:12px;}}</style>
</head><body>
<div id="map"></div>
<script>
var map = L.map('map').setView([{clat},{clon}],{z});
L.tileLayer('{tile_url}',{{
  attribution:'{attr}',
  maxZoom:19,
  tileSize:256
}}).addTo(map);

var pins = {pins_js};
pins.forEach(function(p){{
  var svgIcon = L.divIcon({{
    className:'',
    html:'<svg width="32" height="44" viewBox="0 0 32 44" xmlns="http://www.w3.org/2000/svg">'
      +'<path d="M16 0C7.163 0 0 7.163 0 16c0 10.667 16 28 16 28s16-17.333 16-28C32 7.163 24.837 0 16 0z" fill="'+p.color+'"/>'
      +'<circle cx="16" cy="16" r="7" fill="white" fill-opacity="0.9"/>'
      +'</svg>',
    iconSize:[32,44],iconAnchor:[16,44],popupAnchor:[0,-44]
  }});
  L.marker([p.lat,p.lon],{{icon:svgIcon}})
   .bindPopup('<b>'+p.name+'</b><br/>'+p.tip+'<br/>lat:'+p.lat.toFixed(6)+'<br/>lon:'+p.lon.toFixed(6))
   .addTo(map);
  L.marker([p.lat,p.lon],{{
    icon:L.divIcon({{
      className:'',
      html:'<div style="font:bold 12px/1 Noto Sans JP,sans-serif;color:#1e293b;white-space:nowrap;'
          +'text-shadow:0 1px 3px #fff,0 -1px 3px #fff,1px 0 3px #fff,-1px 0 3px #fff;'
          +'margin-top:-48px;margin-left:4px;">'+p.name+'</div>',
      iconAnchor:[0,0]
    }})
  }}).addTo(map);
}});
if(pins.length>1){{
  var bounds=L.latLngBounds(pins.map(function(p){{return[p.lat,p.lon];}}));
  map.fitBounds(bounds,{{padding:[40,40]}});
}}
</script></body></html>"""

    components.html(html, height=430, scrolling=False)

# ═══════════════════════════════════════════════════════
# 6. ページ設定・CSS
# ═══════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────
# ミス防止チェック関数
# ─────────────────────────────────────────────────────────

def check_datum_zone_mismatch(datum_key: str, zone: int) -> str | None:
    """
    測地系と座標系の組み合わせ不整合を検出してメッセージを返す。
    問題なければ None。
    """
    # 旧日本測地系（Tokyo）は旧座標系用であり、JGD系との混在は誤差が大きい
    if datum_key == "TOKYO":
        return (
            "⚠️ **測地系の確認** — 旧日本測地系（Tokyo97）が選択されています。"
            "現在の公共測量では **JGD2011 または JGD2024** を使用します。"
            "旧測地系のデータを処理する場合のみ選択してください。"
        )
    # 沖縄・離島系（15〜19系）に本土測地系が指定された場合の注意
    if zone in (15, 16, 17, 18, 19) and datum_key in ("JGD2024","JGD2011","JGD2000"):
        return None  # 問題なし
    return None

def check_geoid_warning(geoid_key: str, has_elevation_input: bool) -> str | None:
    """ジオイド補正なしでZ標高が入力されている場合に警告を返す。"""
    if geoid_key == "NONE" and has_elevation_input:
        return (
            "⚠️ **ジオイド補正なし** — 「ジオイド補正なし」が選択されています。"
            "Z標高（正標高）から楕円体高を計算するには"
            "**JPGEO2024 または JPGEO2011** を選択してください。"
            "現在は標高＝楕円体高として処理されます。"
        )
    return None

def render_zone_suggestion_jpc(x_str: str, y_str: str, current_zone: int):
    """JPC入力時の系候補を表示（入力値がある場合のみ）"""
    if not (x_str.strip() and y_str.strip()):
        return
    try:
        Xv, Yv = float(x_str), float(y_str)
        # 極端に大きな値は系の選択ミスの可能性
        if abs(Xv) > 600000 or abs(Yv) > 350000:
            st.warning(
                f"⚠️ **座標値の確認** — X={Xv:,.0f} m / Y={Yv:,.0f} m は"
                "平面直角座標として非常に大きい値です。"
                "**系番号の選択** または **X/Y の入力順序** を確認してください。"
            )
    except ValueError:
        pass

def render_zone_suggestion_ll(lat_str: str, lon_str: str, fmt_key: str, current_zone: int):
    """緯度経度入力時の系自動提案を表示"""
    if not (lat_str.strip() and lon_str.strip()):
        return
    try:
        lat_dd = parse_angle(lat_str, fmt_key)
        lon_dd = parse_angle(lon_str, fmt_key)
        candidates = suggest_zone_from_latlon(lat_dd, lon_dd)
        if candidates and current_zone not in candidates:
            zone_names = ", ".join(
                f"**{z}系**（{JPC_ZONE_LABELS[z].split(' — ')[1]}）"
                for z in candidates
            )
            st.info(
                f"💡 **系番号の提案** — 入力座標（緯度 {lat_dd:.4f}°, 経度 {lon_dd:.4f}°）には"
                f" {zone_names} が適合します。"
                f"現在選択中の **第 {current_zone} 系** と異なります。"
                "サイドバーで系番号を変更してください。"
            )
        elif candidates and current_zone in candidates:
            # 一致の場合は軽くOK表示
            pass
    except (ValueError, Exception):
        pass


st.set_page_config(
    page_title="GNSS SmartShift ICT",
    page_icon="📐", layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* フォント: ローカル優先でシステムフォントにフォールバック（外部読み込み不要） */
html, body, [class*="css"] {
  font-family: 'Noto Sans JP', 'Hiragino Sans', 'Yu Gothic UI', 'Meiryo', sans-serif;
}

/* ── グローバル余白の圧縮 ── */
/* Streamlit のブロック間デフォルトギャップを削減 */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
  gap: 0 !important;
}
/* 各ウィジェットの上下マージンを締める */
div[data-testid="stVerticalBlock"] > * {
  margin-top: 0 !important;
  margin-bottom: 0 !important;
}
/* ボタン・セレクトボックス周りの余白 */
div[data-testid="stButton"] { margin-top: 2px !important; margin-bottom: 2px !important; }
div[data-testid="stSelectbox"] { margin-top: 0 !important; margin-bottom: 0 !important; }
/* radio の上下 */
div[data-testid="stRadio"] { margin-top: 4px !important; margin-bottom: 0 !important; }
/* タブコンテンツ内の先頭余白 */
div[data-testid="stTabsContent"] > div { padding-top: 8px !important; }
/* hr / divider を締める */
hr { margin: 8px 0 !important; }
/* markdown ブロックの余白 */
div[data-testid="stMarkdownContainer"] p { margin-bottom: 4px !important; }

/* ── サイドバー ── */
section[data-testid="stSidebar"] {
  background:
    url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMjAwIiBoZWlnaHQ9IjQwMCIgdmlld0JveD0iMCAwIDEyMDAgNDAwIj4KICA8ZGVmcz4KICAgIDxyYWRpYWxHcmFkaWVudCBpZD0ic3BhY2UiIGN4PSI1MCUiIGN5PSI1MCUiIHI9IjcwJSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiMwYTE2MjgiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSI2MCUiIHN0b3AtY29sb3I9IiMwNTBkMWEiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjMDIwODEwIi8+CiAgICA8L3JhZGlhbEdyYWRpZW50PgogICAgPHJhZGlhbEdyYWRpZW50IGlkPSJlYXJ0aCIgY3g9IjQwJSIgY3k9IjM1JSIgcj0iNjAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzFhNmI5ZSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjQwJSIgc3RvcC1jb2xvcj0iIzBmNGY3YSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjcwJSIgc3RvcC1jb2xvcj0iIzBhM2E1ZSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiMwNjFkMzAiLz4KICAgIDwvcmFkaWFsR3JhZGllbnQ+CiAgICA8cmFkaWFsR3JhZGllbnQgaWQ9ImVhcnRoR2xvdyIgY3g9IjQwJSIgY3k9IjM1JSIgcj0iNjAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSI2MCUiIHN0b3AtY29sb3I9InRyYW5zcGFyZW50Ii8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iIzFlODhlNTgwIi8+CiAgICA8L3JhZGlhbEdyYWRpZW50PgogICAgPHJhZGlhbEdyYWRpZW50IGlkPSJzYXRHbG93IiBjeD0iNTAlIiBjeT0iNTAlIiByPSI1MCUiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjNjBhNWZhIiBzdG9wLW9wYWNpdHk9IjAuOSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiM2MGE1ZmEiIHN0b3Atb3BhY2l0eT0iMCIvPgogICAgPC9yYWRpYWxHcmFkaWVudD4KICAgIDxmaWx0ZXIgaWQ9Imdsb3ciPgogICAgICA8ZmVHYXVzc2lhbkJsdXIgc3RkRGV2aWF0aW9uPSIyIiByZXN1bHQ9ImJsdXIiLz4KICAgICAgPGZlTWVyZ2U+PGZlTWVyZ2VOb2RlIGluPSJibHVyIi8+PGZlTWVyZ2VOb2RlIGluPSJTb3VyY2VHcmFwaGljIi8+PC9mZU1lcmdlPgogICAgPC9maWx0ZXI+CiAgICA8ZmlsdGVyIGlkPSJzb2Z0R2xvdyI+CiAgICAgIDxmZUdhdXNzaWFuQmx1ciBzdGREZXZpYXRpb249IjQiIHJlc3VsdD0iYmx1ciIvPgogICAgICA8ZmVNZXJnZT48ZmVNZXJnZU5vZGUgaW49ImJsdXIiLz48ZmVNZXJnZU5vZGUgaW49IlNvdXJjZUdyYXBoaWMiLz48L2ZlTWVyZ2U+CiAgICA8L2ZpbHRlcj4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0ib3JiaXQxIiB4MT0iMCUiIHkxPSIwJSIgeDI9IjEwMCUiIHkyPSIxMDAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzYwYTVmYSIgc3RvcC1vcGFjaXR5PSIwIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iNTAlIiBzdG9wLWNvbG9yPSIjNjBhNWZhIiBzdG9wLW9wYWNpdHk9IjAuNSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiM2MGE1ZmEiIHN0b3Atb3BhY2l0eT0iMCIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0ib3JiaXQyIiB4MT0iMTAwJSIgeTE9IjAlIiB4Mj0iMCUiIHkyPSIxMDAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzM0ZDM5OSIgc3RvcC1vcGFjaXR5PSIwIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iNTAlIiBzdG9wLWNvbG9yPSIjMzRkMzk5IiBzdG9wLW9wYWNpdHk9IjAuMzUiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjMzRkMzk5IiBzdG9wLW9wYWNpdHk9IjAiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8Y2xpcFBhdGggaWQ9ImNsaXAiPjxyZWN0IHdpZHRoPSIxMjAwIiBoZWlnaHQ9IjQwMCIvPjwvY2xpcFBhdGg+CiAgPC9kZWZzPgoKICA8IS0tIOWuh+WumeiDjOaZryAtLT4KICA8cmVjdCB3aWR0aD0iMTIwMCIgaGVpZ2h0PSI0MDAiIGZpbGw9InVybCgjc3BhY2UpIi8+CgogIDwhLS0g5pif77yI5bCP44GV44GE44KC44Gu77yJIC0tPgogIDxnIGNsaXAtcGF0aD0idXJsKCNjbGlwKSIgb3BhY2l0eT0iMC45Ij4KICAgIDwhLS0g5piO44KL44GE5pifIC0tPgogICAgPGNpcmNsZSBjeD0iNDUiIGN5PSIyMiIgcj0iMS4yIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjk1Ii8+CiAgICA8Y2lyY2xlIGN4PSIxMjAiIGN5PSI2NyIgcj0iMC44IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjciLz4KICAgIDxjaXJjbGUgY3g9IjE5OCIgY3k9IjE1IiByPSIxLjAiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuODUiLz4KICAgIDxjaXJjbGUgY3g9IjI2NyIgY3k9Ijg4IiByPSIwLjciIGZpbGw9IiNlOGY0ZmQiIG9wYWNpdHk9IjAuNiIvPgogICAgPGNpcmNsZSBjeD0iMzQ1IiBjeT0iMzQiIHI9IjEuMSIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC45Ii8+CiAgICA8Y2lyY2xlIGN4PSI0MjMiIGN5PSIxOCIgcj0iMC45IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjc1Ii8+CiAgICA8Y2lyY2xlIGN4PSI1MTIiIGN5PSI1NSIgcj0iMS4zIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjk1Ii8+CiAgICA8Y2lyY2xlIGN4PSI1NzgiIGN5PSIyOCIgcj0iMC44IiBmaWxsPSIjY2ZlOGZmIiBvcGFjaXR5PSIwLjciLz4KICAgIDxjaXJjbGUgY3g9IjY0NSIgY3k9IjcyIiByPSIxLjAiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuOCIvPgogICAgPGNpcmNsZSBjeD0iNzIzIiBjeT0iMjAiIHI9IjAuOSIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC44NSIvPgogICAgPGNpcmNsZSBjeD0iODEyIiBjeT0iNDUiIHI9IjEuMiIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC45Ii8+CiAgICA8Y2lyY2xlIGN4PSI4NzYiIGN5PSIxNSIgcj0iMC43IiBmaWxsPSIjZThmNGZkIiBvcGFjaXR5PSIwLjY1Ii8+CiAgICA8Y2lyY2xlIGN4PSI5MzQiIGN5PSI2MCIgcj0iMS4xIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjg4Ii8+CiAgICA8Y2lyY2xlIGN4PSIxMDIzIiBjeT0iMzAiIHI9IjAuOCIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC43MiIvPgogICAgPGNpcmNsZSBjeD0iMTA5OCIgY3k9IjUwIiByPSIxLjAiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuOCIvPgogICAgPGNpcmNsZSBjeD0iMTE1NiIgY3k9IjI1IiByPSIwLjkiIGZpbGw9IiNjZmU4ZmYiIG9wYWNpdHk9IjAuNzUiLz4KICAgIDwhLS0g5Lit5q61IC0tPgogICAgPGNpcmNsZSBjeD0iNzgiIGN5PSIxNDUiIHI9IjAuOSIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC43Ii8+CiAgICA8Y2lyY2xlIGN4PSIxNTYiIGN5PSIxNzgiIHI9IjEuMSIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC44NSIvPgogICAgPGNpcmNsZSBjeD0iMjM0IiBjeT0iMTMwIiByPSIwLjciIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNiIvPgogICAgPGNpcmNsZSBjeD0iMzEyIiBjeT0iMTY1IiByPSIwLjgiIGZpbGw9IiNlOGY0ZmQiIG9wYWNpdHk9IjAuNzIiLz4KICAgIDxjaXJjbGUgY3g9IjM4OSIgY3k9IjE0OCIgcj0iMS4wIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjgiLz4KICAgIDxjaXJjbGUgY3g9IjQ2NyIgY3k9IjE5MiIgcj0iMC45IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjY4Ii8+CiAgICA8Y2lyY2xlIGN4PSI1NTYiIGN5PSIxNDAiIHI9IjEuMiIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC45Ii8+CiAgICA8Y2lyY2xlIGN4PSI2MzQiIGN5PSIxNzUiIHI9IjAuOCIgZmlsbD0iI2NmZThmZiIgb3BhY2l0eT0iMC42MiIvPgogICAgPGNpcmNsZSBjeD0iNzAwIiBjeT0iMTU1IiByPSIxLjAiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNzgiLz4KICAgIDxjaXJjbGUgY3g9Ijc5MCIgY3k9IjE4NSIgcj0iMC43IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjY1Ii8+CiAgICA8Y2lyY2xlIGN4PSI4NTYiIGN5PSIxNDUiIHI9IjEuMSIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC44NyIvPgogICAgPGNpcmNsZSBjeD0iOTQ1IiBjeT0iMTcwIiByPSIwLjkiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNzMiLz4KICAgIDxjaXJjbGUgY3g9IjEwMzQiIGN5PSIxNTIiIHI9IjAuOCIgZmlsbD0iI2U4ZjRmZCIgb3BhY2l0eT0iMC42NyIvPgogICAgPGNpcmNsZSBjeD0iMTExMiIgY3k9IjEzNSIgcj0iMS4wIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjgyIi8+CiAgICA8Y2lyY2xlIGN4PSIxMTc4IiBjeT0iMTYyIiByPSIwLjkiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNyIvPgogICAgPCEtLSDkuIvmrrUgLS0+CiAgICA8Y2lyY2xlIGN4PSI1NiIgY3k9IjI4NSIgcj0iMC44IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjYiLz4KICAgIDxjaXJjbGUgY3g9IjE2NyIgY3k9IjMxMiIgcj0iMS4wIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjcyIi8+CiAgICA8Y2lyY2xlIGN4PSIyNzgiIGN5PSIyNzAiIHI9IjAuOSIgZmlsbD0iI2NmZThmZiIgb3BhY2l0eT0iMC42NSIvPgogICAgPGNpcmNsZSBjeD0iMzg5IiBjeT0iMzQwIiByPSIwLjciIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNTUiLz4KICAgIDxjaXJjbGUgY3g9IjUwMCIgY3k9IjI5OCIgcj0iMS4xIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjgiLz4KICAgIDxjaXJjbGUgY3g9IjYxMSIgY3k9IjMyNSIgcj0iMC44IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjY4Ii8+CiAgICA8Y2lyY2xlIGN4PSI3MjIiIGN5PSIyNzgiIHI9IjEuMCIgZmlsbD0iI2U4ZjRmZCIgb3BhY2l0eT0iMC43NSIvPgogICAgPGNpcmNsZSBjeD0iODMzIiBjeT0iMzYwIiByPSIwLjkiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNjIiLz4KICAgIDxjaXJjbGUgY3g9Ijk0NCIgY3k9IjI5MCIgcj0iMC43IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjU4Ii8+CiAgICA8Y2lyY2xlIGN4PSIxMDU1IiBjeT0iMzMyIiByPSIxLjEiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNzgiLz4KICAgIDxjaXJjbGUgY3g9IjExNDQiIGN5PSIzMDUiIHI9IjAuOCIgZmlsbD0iI2NmZThmZiIgb3BhY2l0eT0iMC43Ii8+CiAgPC9nPgoKICA8IS0tIOWcsOeQg++8iOWPs+S4i+OBruabsumdou+8iSAtLT4KICA8Y2lyY2xlIGN4PSIxMDUwIiBjeT0iNTIwIiByPSIzMjAiIGZpbGw9InVybCgjZWFydGgpIiBvcGFjaXR5PSIwLjg1Ii8+CiAgPGNpcmNsZSBjeD0iMTA1MCIgY3k9IjUyMCIgcj0iMzIwIiBmaWxsPSJ1cmwoI2VhcnRoR2xvdykiIG9wYWNpdHk9IjAuNSIvPgogIDwhLS0g5aSn5rCX5YWJIC0tPgogIDxjaXJjbGUgY3g9IjEwNTAiIGN5PSI1MjAiIHI9IjMzMiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMWU4OGU1IiBzdHJva2Utd2lkdGg9IjUiIG9wYWNpdHk9IjAuMTUiLz4KICA8Y2lyY2xlIGN4PSIxMDUwIiBjeT0iNTIwIiByPSIzNDAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzQyYTVmNSIgc3Ryb2tlLXdpZHRoPSIzIiBvcGFjaXR5PSIwLjA4Ii8+CiAgPCEtLSDpmbjlnLDjgrfjg6vjgqjjg4Pjg4jvvIjnsKHnlaXvvIkgLS0+CiAgPHBhdGggZD0iTTc2MCAzMDAgUTc4MCAyODAgODEwIDI5NSBRODMwIDI3NSA4NTAgMjg1IFE4NzAgMjY1IDg5MCAyODAgUTkwMCAyOTUgODg1IDMxMCBRODcwIDMyNSA4NDUgMzE4IFE4MjAgMzMwIDc5NSAzMjAgUTc3MCAzMTUgNzYwIDMwMFoiIGZpbGw9IiMyZDhhNGUiIG9wYWNpdHk9IjAuNTUiLz4KICA8cGF0aCBkPSJNOTAwIDM0MCBROTIwIDMyNSA5NDAgMzM1IFE5NjAgMzE4IDk3NSAzMzAgUTk4NSAzNDUgOTcwIDM1OCBROTUwIDM2NSA5MzAgMzU1IFE5MTAgMzYwIDkwMCAzNDBaIiBmaWxsPSIjMmQ4YTRlIiBvcGFjaXR5PSIwLjQ1Ii8+CiAgPCEtLSDlpKfmtIvjga7lj43lsIQgLS0+CiAgPGVsbGlwc2UgY3g9IjgyMCIgY3k9IjM4MCIgcng9IjYwIiByeT0iMjAiIGZpbGw9IiM0ZmMzZjciIG9wYWNpdHk9IjAuMTIiIHRyYW5zZm9ybT0icm90YXRlKC0xNSw4MjAsMzgwKSIvPgoKICA8IS0tIOihm+aYn+i7jOmBk+ODqeOCpOODszEgLS0+CiAgPGVsbGlwc2UgY3g9IjQwMCIgY3k9IjIwMCIgcng9IjYyMCIgcnk9IjEyMCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ1cmwoI29yYml0MSkiIHN0cm9rZS13aWR0aD0iMSIgb3BhY2l0eT0iMC42IiB0cmFuc2Zvcm09InJvdGF0ZSgtMTIsNDAwLDIwMCkiLz4KCiAgPCEtLSDooZvmmJ/ou4zpgZPjg6njgqTjg7MyIC0tPgogIDxlbGxpcHNlIGN4PSIzMDAiIGN5PSIxODAiIHJ4PSI1ODAiIHJ5PSI5MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ1cmwoI29yYml0MikiIHN0cm9rZS13aWR0aD0iMC44IiBvcGFjaXR5PSIwLjUiIHRyYW5zZm9ybT0icm90YXRlKDgsMzAwLDE4MCkiLz4KCiAgPCEtLSBHTlNT6KGb5pifMSAtLT4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxODUsNzUpIiBmaWx0ZXI9InVybCgjZ2xvdykiPgogICAgPHJlY3QgeD0iLTE4IiB5PSItMyIgd2lkdGg9IjM2IiBoZWlnaHQ9IjYiIGZpbGw9IiM2MGE1ZmEiIG9wYWNpdHk9IjAuOSIgcng9IjEiLz4KICAgIDxyZWN0IHg9Ii0zIiB5PSItMTIiIHdpZHRoPSI2IiBoZWlnaHQ9IjI0IiBmaWxsPSIjNjBhNWZhIiBvcGFjaXR5PSIwLjkiIHJ4PSIxIi8+CiAgICA8Y2lyY2xlIGN4PSIwIiBjeT0iMCIgcj0iNSIgZmlsbD0iIzkzYzVmZCIgb3BhY2l0eT0iMC45NSIvPgogICAgPGNpcmNsZSBjeD0iMCIgY3k9IjAiIHI9IjgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzYwYTVmYSIgc3Ryb2tlLXdpZHRoPSIwLjgiIG9wYWNpdHk9IjAuNCIvPgogICAgPGNpcmNsZSBjeD0iMCIgY3k9IjAiIHI9IjE0IiBmaWxsPSJ1cmwoI3NhdEdsb3cpIiBvcGFjaXR5PSIwLjMiLz4KICA8L2c+CgogIDwhLS0gR05TU+ihm+aYnzLvvIjlsI/jgZXjgoHvvIkgLS0+CiAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODgwLDYyKSIgZmlsdGVyPSJ1cmwoI2dsb3cpIj4KICAgIDxyZWN0IHg9Ii0xNCIgeT0iLTIiIHdpZHRoPSIyOCIgaGVpZ2h0PSI0IiBmaWxsPSIjMzRkMzk5IiBvcGFjaXR5PSIwLjg1IiByeD0iMSIvPgogICAgPHJlY3QgeD0iLTIiIHk9Ii05IiB3aWR0aD0iNCIgaGVpZ2h0PSIxOCIgZmlsbD0iIzM0ZDM5OSIgb3BhY2l0eT0iMC44NSIgcng9IjEiLz4KICAgIDxjaXJjbGUgY3g9IjAiIGN5PSIwIiByPSI0IiBmaWxsPSIjNmVlN2I3IiBvcGFjaXR5PSIwLjkiLz4KICAgIDxjaXJjbGUgY3g9IjAiIGN5PSIwIiByPSIxMCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMzRkMzk5IiBzdHJva2Utd2lkdGg9IjAuNyIgb3BhY2l0eT0iMC4zNSIvPgogIDwvZz4KCiAgPCEtLSDkv6Hlj7fms6LntIvvvIjooZvmmJ8x77yJIC0tPgogIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4NSw3NSkiIG9wYWNpdHk9IjAuMjUiPgogICAgPGNpcmNsZSBjeD0iMCIgY3k9IjAiIHI9IjI1IiBmaWxsPSJub25lIiBzdHJva2U9IiM2MGE1ZmEiIHN0cm9rZS13aWR0aD0iMC44Ii8+CiAgICA8Y2lyY2xlIGN4PSIwIiBjeT0iMCIgcj0iNDAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzYwYTVmYSIgc3Ryb2tlLXdpZHRoPSIwLjYiLz4KICAgIDxjaXJjbGUgY3g9IjAiIGN5PSIwIiByPSI1OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNjBhNWZhIiBzdHJva2Utd2lkdGg9IjAuNCIvPgogIDwvZz4KCiAgPCEtLSDkv6Hlj7fnt5rvvIjooZvmmJ/ihpLlnLDnkIPvvIkgLS0+CiAgPGxpbmUgeDE9IjE4NSIgeTE9Ijg1IiB4Mj0iODUwIiB5Mj0iMzEwIiBzdHJva2U9IiM2MGE1ZmEiIHN0cm9rZS13aWR0aD0iMC42IiBzdHJva2UtZGFzaGFycmF5PSI0LDgiIG9wYWNpdHk9IjAuMiIvPgogIDxsaW5lIHgxPSI4ODAiIHkxPSI3MiIgeDI9IjgyMCIgeTI9IjI5NSIgc3Ryb2tlPSIjMzRkMzk5IiBzdHJva2Utd2lkdGg9IjAuNiIgc3Ryb2tlLWRhc2hhcnJheT0iNCw4IiBvcGFjaXR5PSIwLjE4Ii8+CgogIDwhLS0g5Y+z56uv44Kw44Op44OH44O844K344On44Oz77yI44OV44Kn44O844OJ44Ki44Km44OI77yJIC0tPgogIDxyZWN0IHdpZHRoPSIxMjAwIiBoZWlnaHQ9IjQwMCIgZmlsbD0idXJsKCNzcGFjZSkiIG9wYWNpdHk9IjAiIGNsaXAtcGF0aD0idXJsKCNjbGlwKSIvPgo8L3N2Zz4=") center/cover no-repeat fixed !important;
  position: relative;
}
/* 半透明ダークオーバーレイ */
section[data-testid="stSidebar"]::before {
  content: "";
  position: absolute; inset: 0;
  background: rgba(10,18,36,0.78);
  z-index: 0;
  pointer-events: none;
}
section[data-testid="stSidebar"] > div { position: relative; z-index: 1; }
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background: #1e293b !important; color: #f1f5f9 !important; border-color: #334155 !important; }
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span { color: #f1f5f9 !important; }
[data-baseweb="popover"] [role="option"] { background: #1e293b !important; color: #f1f5f9 !important; }
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] { background: #334155 !important; color: #fbbf24 !important; }
[data-baseweb="popover"] { background: #1e293b !important; }

/* ── 結果カード ── */
.rc { background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:14px 18px 12px; margin-bottom:8px; }
.rc-lbl { font-size:9px; font-weight:700; letter-spacing:.18em; text-transform:uppercase; color:#94a3b8; margin-bottom:5px; }
.rc-val { font-family:'DM Mono',monospace; font-size:17px; font-weight:500; color:#0f172a; }
.rc-sub { font-family:'DM Mono',monospace; font-size:10px; color:#94a3b8; margin-top:3px; }

/* ── 点カード ── */
.pt-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px;
           padding:18px 20px 14px; margin-bottom:14px; }
.pt-title { font-size:13px; font-weight:700; color:#475569; margin-bottom:12px; }

/* ── ヘッダー ── */
.app-hdr {
  background: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNDAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDE0MDAgMjAwIj4KICA8ZGVmcz4KICAgIDwhLS0gQmFubmVyIGNsaXAgLS0+CiAgICA8Y2xpcFBhdGggaWQ9ImJhbm5lciI+PHJlY3Qgd2lkdGg9IjE0MDAiIGhlaWdodD0iMjAwIiByeD0iMTQiLz48L2NsaXBQYXRoPgoKICAgIDwhLS0gRWFydGggY2xpcDogY2lyY2xlIGV4YWN0bHkgZml0dGluZyByaWdodCBwb3J0aW9uIC0tPgogICAgPGNsaXBQYXRoIGlkPSJlYXJ0aENsaXAiPgogICAgICA8Y2lyY2xlIGN4PSIxMTgwIiBjeT0iMTAwIiByPSIxNzAiLz4KICAgIDwvY2xpcFBhdGg+CgogICAgPCEtLSBEZWVwIHNwYWNlIGJnIC0tPgogICAgPGxpbmVhckdyYWRpZW50IGlkPSJza3kiIHgxPSIwJSIgeTE9IjAlIiB4Mj0iMTAwJSIgeTI9IjEwMCUiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiAgIHN0b3AtY29sb3I9IiMwMDA2MTIiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSI1MCUiICBzdG9wLWNvbG9yPSIjMDEwYjIwIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iIzAyMGQxNiIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KCiAgICA8IS0tIE9jZWFuIGdyYWRpZW50IC0tPgogICAgPHJhZGlhbEdyYWRpZW50IGlkPSJvY2VhbiIgY3g9IjM4JSIgY3k9IjQyJSIgcj0iNjAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgICBzdG9wLWNvbG9yPSIjMWU3ZGI4Ii8+CiAgICAgIDxzdG9wIG9mZnNldD0iMzAlIiAgc3RvcC1jb2xvcj0iIzBlNTU4NSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjY1JSIgIHN0b3AtY29sb3I9IiMwODNkNjIiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjMDMyMjQwIi8+CiAgICA8L3JhZGlhbEdyYWRpZW50PgoKICAgIDwhLS0gQXRtb3NwaGVyZSBoYWxvIC0tPgogICAgPHJhZGlhbEdyYWRpZW50IGlkPSJhdG1vIiBjeD0iMzglIiBjeT0iNDIlIiByPSI1MiUiPgogICAgICA8c3RvcCBvZmZzZXQ9IjcyJSIgIHN0b3AtY29sb3I9InRyYW5zcGFyZW50Ii8+CiAgICAgIDxzdG9wIG9mZnNldD0iODglIiAgc3RvcC1jb2xvcj0iIzFlNzBiMDM4Ii8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iIzQwYThlODcwIi8+CiAgICA8L3JhZGlhbEdyYWRpZW50PgoKICAgIDwhLS0gTmlnaHQtc2lkZSBzaGFkb3cgKHRlcm1pbmF0b3IpIC0tPgogICAgPHJhZGlhbEdyYWRpZW50IGlkPSJuaWdodCIgY3g9IjcyJSIgY3k9IjU4JSIgcj0iNTUlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgICBzdG9wLWNvbG9yPSIjMDAwMDAwYTAiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSI1NSUiICBzdG9wLWNvbG9yPSIjMDAwMDAwNTUiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSJ0cmFuc3BhcmVudCIvPgogICAgPC9yYWRpYWxHcmFkaWVudD4KCiAgICA8IS0tIENpdHkgZ2xvdyBmaWx0ZXIgLS0+CiAgICA8ZmlsdGVyIGlkPSJjaXR5R2xvdyIgeD0iLTYwJSIgeT0iLTYwJSIgd2lkdGg9IjIyMCUiIGhlaWdodD0iMjIwJSI+CiAgICAgIDxmZUdhdXNzaWFuQmx1ciBzdGREZXZpYXRpb249IjEuOCIgcmVzdWx0PSJiIi8+CiAgICAgIDxmZU1lcmdlPjxmZU1lcmdlTm9kZSBpbj0iYiIvPjxmZU1lcmdlTm9kZSBpbj0iU291cmNlR3JhcGhpYyIvPjwvZmVNZXJnZT4KICAgIDwvZmlsdGVyPgoKICAgIDwhLS0gU2F0ZWxsaXRlIGdsb3cgLS0+CiAgICA8ZmlsdGVyIGlkPSJzYXRHbG93IiB4PSItMTIwJSIgeT0iLTEyMCUiIHdpZHRoPSIzNDAlIiBoZWlnaHQ9IjM0MCUiPgogICAgICA8ZmVHYXVzc2lhbkJsdXIgc3RkRGV2aWF0aW9uPSIzLjUiIHJlc3VsdD0iYiIvPgogICAgICA8ZmVNZXJnZT48ZmVNZXJnZU5vZGUgaW49ImIiLz48ZmVNZXJnZU5vZGUgaW49IlNvdXJjZUdyYXBoaWMiLz48L2ZlTWVyZ2U+CiAgICA8L2ZpbHRlcj4KCiAgICA8IS0tIFNvZnQgcmluZyBnbG93IC0tPgogICAgPGZpbHRlciBpZD0icmluZ0dsb3ciIHg9Ii04MCUiIHk9Ii04MCUiIHdpZHRoPSIyNjAlIiBoZWlnaHQ9IjI2MCUiPgogICAgICA8ZmVHYXVzc2lhbkJsdXIgc3RkRGV2aWF0aW9uPSI0IiByZXN1bHQ9ImIiLz4KICAgICAgPGZlTWVyZ2U+PGZlTWVyZ2VOb2RlIGluPSJiIi8+PGZlTWVyZ2VOb2RlIGluPSJTb3VyY2VHcmFwaGljIi8+PC9mZU1lcmdlPgogICAgPC9maWx0ZXI+CgogICAgPCEtLSBUZXh0IG92ZXJsYXk6IGRhcmstbGVmdCBmYWRlIGZvciByZWFkYWJpbGl0eSAtLT4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0idGV4dEZhZGUiIHgxPSIwJSIgeTE9IjAlIiB4Mj0iMTAwJSIgeTI9IjAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgICBzdG9wLWNvbG9yPSIjMDEwYTFlIiBzdG9wLW9wYWNpdHk9IjAuNjgiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSI0MiUiICBzdG9wLWNvbG9yPSIjMDEwYTFlIiBzdG9wLW9wYWNpdHk9IjAuMjIiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSI2NSUiICBzdG9wLWNvbG9yPSIjMDEwYTFlIiBzdG9wLW9wYWNpdHk9IjAuMDQiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjMDEwYTFlIiBzdG9wLW9wYWNpdHk9IjAuMDAiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CgogICAgPCEtLSBTaGltbWVyIGxpbmUgLS0+CiAgICA8bGluZWFyR3JhZGllbnQgaWQ9InNoaW1tZXIiIHgxPSIwJSIgeTE9IjAlIiB4Mj0iMTAwJSIgeTI9IjAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgICBzdG9wLWNvbG9yPSJ0cmFuc3BhcmVudCIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjMwJSIgIHN0b3AtY29sb3I9IiM2MGQwZmYiIHN0b3Atb3BhY2l0eT0iMC41NSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjU1JSIgIHN0b3AtY29sb3I9IiNmZmZmZmYiICBzdG9wLW9wYWNpdHk9IjAuMzUiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSI4MCUiICBzdG9wLWNvbG9yPSIjNjBkMGZmIiBzdG9wLW9wYWNpdHk9IjAuNDAiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSJ0cmFuc3BhcmVudCIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICA8L2RlZnM+CgogIDwhLS0g4pWQ4pWQ4pWQIEJBQ0tHUk9VTkQg4pWQ4pWQ4pWQIC0tPgogIDxyZWN0IHdpZHRoPSIxNDAwIiBoZWlnaHQ9IjIwMCIgZmlsbD0idXJsKCNza3kpIiBjbGlwLXBhdGg9InVybCgjYmFubmVyKSIvPgoKICA8IS0tIOKVkOKVkOKVkCBTVEFSUyDilZDilZDilZAgLS0+CiAgPGcgY2xpcC1wYXRoPSJ1cmwoI2Jhbm5lcikiPgogICAgPCEtLSBicmlnaHQgLS0+CiAgICA8Y2lyY2xlIGN4PSIyMiIgIGN5PSIxNCIgIHI9IjEuNCIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC45NSIvPgogICAgPGNpcmNsZSBjeD0iNTgiICBjeT0iNDIiICByPSIwLjkiIGZpbGw9IiNlOGY0ZmQiIG9wYWNpdHk9IjAuODAiLz4KICAgIDxjaXJjbGUgY3g9IjkyIiAgY3k9IjkiICAgcj0iMS4yIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjkyIi8+CiAgICA8Y2lyY2xlIGN4PSIxMzgiIGN5PSIzMSIgIHI9IjAuNyIgZmlsbD0iI2NjZThmZiIgb3BhY2l0eT0iMC43MCIvPgogICAgPGNpcmNsZSBjeD0iMTc4IiBjeT0iNTYiICByPSIxLjMiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuOTUiLz4KICAgIDxjaXJjbGUgY3g9IjIxNSIgY3k9IjE1IiAgcj0iMC45IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjc4Ii8+CiAgICA8Y2lyY2xlIGN4PSIyNTIiIGN5PSI3NCIgIHI9IjEuMSIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC44OCIvPgogICAgPGNpcmNsZSBjeD0iMjkwIiBjeT0iMjIiICByPSIwLjciIGZpbGw9IiNlMGYwZmYiIG9wYWNpdHk9IjAuNjUiLz4KICAgIDxjaXJjbGUgY3g9IjMyOCIgY3k9IjQ5IiAgcj0iMS4yIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjkwIi8+CiAgICA8Y2lyY2xlIGN4PSIzNjgiIGN5PSIxMSIgIHI9IjAuOSIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC44MiIvPgogICAgPGNpcmNsZSBjeD0iNDA1IiBjeT0iNjYiICByPSIxLjAiIGZpbGw9IiNjY2U4ZmYiIG9wYWNpdHk9IjAuODUiLz4KICAgIDxjaXJjbGUgY3g9IjQ0NSIgY3k9IjI4IiAgcj0iMC44IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjc0Ii8+CiAgICA8Y2lyY2xlIGN4PSI0ODUiIGN5PSI4MiIgIHI9IjEuMSIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC44NCIvPgogICAgPGNpcmNsZSBjeD0iNTIyIiBjeT0iMTgiICByPSIwLjkiIGZpbGw9IiNlOGY0ZmQiIG9wYWNpdHk9IjAuNzkiLz4KICAgIDxjaXJjbGUgY3g9IjU2MCIgY3k9IjUzIiAgcj0iMS4zIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjkzIi8+CiAgICA8Y2lyY2xlIGN4PSI1OTgiIGN5PSI5IiAgIHI9IjAuOCIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC44MiIvPgogICAgPGNpcmNsZSBjeD0iNjM1IiBjeT0iNzAiICByPSIwLjciIGZpbGw9IiNjY2U4ZmYiIG9wYWNpdHk9IjAuNjQiLz4KICAgIDxjaXJjbGUgY3g9IjY2OCIgY3k9IjMzIiAgcj0iMS4wIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjg2Ii8+CiAgICA8Y2lyY2xlIGN4PSI3MDUiIGN5PSIxNSIgIHI9IjAuOSIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC43OCIvPgogICAgPGNpcmNsZSBjeD0iNzQyIiBjeT0iNjAiICByPSIxLjEiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuODgiLz4KICAgIDxjaXJjbGUgY3g9Ijc3OCIgY3k9IjIyIiAgcj0iMC43IiBmaWxsPSIjZThmNGZkIiBvcGFjaXR5PSIwLjcwIi8+CiAgICA8Y2lyY2xlIGN4PSI4MTIiIGN5PSI0OCIgIHI9IjEuMiIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC45MCIvPgogICAgPGNpcmNsZSBjeD0iODQ4IiBjeT0iMTIiICByPSIwLjgiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNzYiLz4KICAgIDwhLS0gbWlkIGJyaWdodG5lc3MgLS0+CiAgICA8Y2lyY2xlIGN4PSI0MCIgIGN5PSI4MiIgIHI9IjAuNyIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC41OCIvPgogICAgPGNpcmNsZSBjeD0iNzgiICBjeT0iOTgiICByPSIwLjYiIGZpbGw9IiNkZGYiIG9wYWNpdHk9IjAuNTIiLz4KICAgIDxjaXJjbGUgY3g9IjExOCIgY3k9Ijg4IiAgcj0iMC44IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjYyIi8+CiAgICA8Y2lyY2xlIGN4PSIxNjAiIGN5PSIxMDIiIHI9IjAuNiIgZmlsbD0iI2U4ZjRmZCIgb3BhY2l0eT0iMC41NiIvPgogICAgPGNpcmNsZSBjeD0iMjAwIiBjeT0iOTAiICByPSIwLjkiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNjgiLz4KICAgIDxjaXJjbGUgY3g9IjI0MiIgY3k9IjExMCIgcj0iMC41IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjUwIi8+CiAgICA8Y2lyY2xlIGN4PSIyNzgiIGN5PSI5NiIgIHI9IjAuNyIgZmlsbD0iI2NjZThmZiIgb3BhY2l0eT0iMC42MCIvPgogICAgPGNpcmNsZSBjeD0iMzE1IiBjeT0iMTE0IiByPSIwLjYiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNTQiLz4KICAgIDxjaXJjbGUgY3g9IjM1MiIgY3k9IjkyIiAgcj0iMC44IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjY2Ii8+CiAgICA8Y2lyY2xlIGN4PSIzOTIiIGN5PSIxMDgiIHI9IjAuNiIgZmlsbD0iI2RkZiIgb3BhY2l0eT0iMC41MiIvPgogICAgPGNpcmNsZSBjeD0iNDI4IiBjeT0iOTQiICByPSIwLjkiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNzAiLz4KICAgIDxjaXJjbGUgY3g9IjQ2NSIgY3k9IjExOCIgcj0iMC41IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjU0Ii8+CiAgICA8Y2lyY2xlIGN4PSI1MDUiIGN5PSIxMDAiIHI9IjAuNyIgZmlsbD0iI2U4ZjRmZCIgb3BhY2l0eT0iMC42MiIvPgogICAgPGNpcmNsZSBjeD0iNTQyIiBjeT0iMTEwIiByPSIwLjYiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNTYiLz4KICAgIDxjaXJjbGUgY3g9IjU3OCIgY3k9IjkwIiAgcj0iMC44IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjY4Ii8+CiAgICA8Y2lyY2xlIGN4PSI2MTUiIGN5PSIxMDQiIHI9IjAuNiIgZmlsbD0iI2NjZThmZiIgb3BhY2l0eT0iMC41OCIvPgogICAgPGNpcmNsZSBjeD0iNjUyIiBjeT0iODQiICByPSIwLjkiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNzIiLz4KICAgIDxjaXJjbGUgY3g9IjY4OCIgY3k9IjEwOCIgcj0iMC41IiBmaWxsPSIjZGRmIiBvcGFjaXR5PSIwLjUwIi8+CiAgICA8Y2lyY2xlIGN4PSI3MjUiIGN5PSI5MiIgIHI9IjAuNyIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC42NCIvPgogICAgPGNpcmNsZSBjeD0iNzYwIiBjeT0iMTE1IiByPSIwLjYiIGZpbGw9IiNlOGY0ZmQiIG9wYWNpdHk9IjAuNTYiLz4KICAgIDxjaXJjbGUgY3g9Ijc5NSIgY3k9Ijk2IiAgcj0iMC44IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjcwIi8+CiAgICA8Y2lyY2xlIGN4PSI4MzAiIGN5PSIxMTAiIHI9IjAuNSIgZmlsbD0iI2NjZThmZiIgb3BhY2l0eT0iMC41MiIvPgogICAgPCEtLSBkaW0gLS0+CiAgICA8Y2lyY2xlIGN4PSIzMCIgIGN5PSIxNDgiIHI9IjAuNSIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC40MCIvPgogICAgPGNpcmNsZSBjeD0iNzIiICBjeT0iMTYyIiByPSIwLjQiIGZpbGw9IiNkZGYiIG9wYWNpdHk9IjAuMzYiLz4KICAgIDxjaXJjbGUgY3g9IjExMiIgY3k9IjE0NSIgcj0iMC42IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjQ2Ii8+CiAgICA8Y2lyY2xlIGN4PSIxNTIiIGN5PSIxNzAiIHI9IjAuNCIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC4zNCIvPgogICAgPGNpcmNsZSBjeD0iMTkyIiBjeT0iMTQ4IiByPSIwLjUiIGZpbGw9IiNlOGY0ZmQiIG9wYWNpdHk9IjAuNDQiLz4KICAgIDxjaXJjbGUgY3g9IjIzMiIgY3k9IjE2NSIgcj0iMC40IiBmaWxsPSIjY2NlOGZmIiBvcGFjaXR5PSIwLjM4Ii8+CiAgICA8Y2lyY2xlIGN4PSIyNzIiIGN5PSIxNTIiIHI9IjAuNiIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC40OCIvPgogICAgPGNpcmNsZSBjeD0iMzEyIiBjeT0iMTY4IiByPSIwLjMiIGZpbGw9IiNkZGYiIG9wYWNpdHk9IjAuMzIiLz4KICAgIDxjaXJjbGUgY3g9IjM1MiIgY3k9IjE1NSIgcj0iMC41IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjQ0Ii8+CiAgICA8Y2lyY2xlIGN4PSIzOTIiIGN5PSIxNzUiIHI9IjAuNCIgZmlsbD0iI2U4ZjRmZCIgb3BhY2l0eT0iMC4zOCIvPgogICAgPGNpcmNsZSBjeD0iNDMyIiBjeT0iMTU4IiByPSIwLjYiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNTAiLz4KICAgIDxjaXJjbGUgY3g9IjQ3MiIgY3k9IjE3MiIgcj0iMC4zIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjMwIi8+CiAgICA8Y2lyY2xlIGN4PSI1MTIiIGN5PSIxNjAiIHI9IjAuNSIgZmlsbD0iI2NjZThmZiIgb3BhY2l0eT0iMC40MiIvPgogICAgPGNpcmNsZSBjeD0iNTUyIiBjeT0iMTQ4IiByPSIwLjQiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuMzgiLz4KICAgIDxjaXJjbGUgY3g9IjU5MiIgY3k9IjE2OCIgcj0iMC42IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjQ4Ii8+CiAgICA8Y2lyY2xlIGN4PSI2MzIiIGN5PSIxNTUiIHI9IjAuMyIgZmlsbD0iI2RkZiIgb3BhY2l0eT0iMC4zNCIvPgogICAgPGNpcmNsZSBjeD0iNjcyIiBjeT0iMTcyIiByPSIwLjUiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNDIiLz4KICAgIDxjaXJjbGUgY3g9IjcxMiIgY3k9IjE1OCIgcj0iMC40IiBmaWxsPSIjZThmNGZkIiBvcGFjaXR5PSIwLjM4Ii8+CiAgICA8Y2lyY2xlIGN4PSI3NTIiIGN5PSIxNDgiIHI9IjAuNiIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC40NiIvPgogICAgPGNpcmNsZSBjeD0iNzkyIiBjeT0iMTY1IiByPSIwLjMiIGZpbGw9IiNjY2U4ZmYiIG9wYWNpdHk9IjAuMzIiLz4KICAgIDwhLS0gTWlsa3kgd2F5IGhhemUgLS0+CiAgICA8ZWxsaXBzZSBjeD0iMzgwIiBjeT0iMTA1IiByeD0iMzQwIiByeT0iNDgiIGZpbGw9IiNhMGM4ZmYiIG9wYWNpdHk9IjAuMDE2IiB0cmFuc2Zvcm09InJvdGF0ZSgtOCwzODAsMTA1KSIvPgogICAgPGVsbGlwc2UgY3g9IjM5MCIgY3k9IjEwOCIgcng9IjI5MCIgcnk9IjMyIiBmaWxsPSIjZmZmZmZmIiAgb3BhY2l0eT0iMC4wMTQiIHRyYW5zZm9ybT0icm90YXRlKC04LDM5MCwxMDgpIi8+CiAgPC9nPgoKICA8IS0tIOKVkOKVkOKVkCBFQVJUSCAoY2xpcHBlZCB0byBjaXJjbGUsIHJpZ2h0IHNpZGUpIOKVkOKVkOKVkCAtLT4KICA8ZyBjbGlwLXBhdGg9InVybCgjZWFydGhDbGlwKSI+CiAgICA8IS0tIE9jZWFuIGJhc2UgLS0+CiAgICA8Y2lyY2xlIGN4PSIxMTgwIiBjeT0iMTAwIiByPSIxNzAiIGZpbGw9InVybCgjb2NlYW4pIi8+CgogICAgPCEtLSDilIDilIAgTGFuZCBtYXNzZXMgKGFsbCB3aXRoaW4gcj0xNzAgZnJvbSAxMTgwLDEwMCkg4pSA4pSACiAgICAgICAgIFVzaW5nIGFjY3VyYXRlIGJ1dCBzaW1wbGlmaWVkIG91dGxpbmVzIGZvciBBc2lhL1BhY2lmaWMgcmVnaW9uLgogICAgICAgICBLZXk6IGtlZXAgYWxsIHBhdGggY29vcmRpbmF0ZXMgd2l0aGluIGNpcmNsZSBjeD0xMTgwIGN5PTEwMCByPTE3MAogICAgICAgICBzbyBub3RoaW5nIGJsZWVkcyBvdXRzaWRlLiAtLT4KCiAgICA8IS0tIEFzaWFuIGNvbnRpbmVudCBib2R5IChTaWJlcmlhL0NoaW5hL1NFIEFzaWEpIC0tPgogICAgPHBhdGggZD0iCiAgICAgIE0xMDU1LDggIFExMDcyLDQgIDEwOTAsNiAgUTExMDgsMiAgMTEyNSw4ICBRMTE0MCw0ICAxMTU1LDEyCiAgICAgIFExMTY1LDggIDExNzgsMTUgUTExODgsMTAgMTE5OCwxOCBRMTIwOCwxNSAxMjE1LDI1CiAgICAgIFExMjI1LDIyIDEyMzIsMzUgUTEyMzgsMzAgMTI0Miw0NCBRMTI0OCw0MCAxMjUwLDU1CiAgICAgIFExMjUyLDUwIDEyNTIsNjUgUTEyNTAsNjIgMTI0OCw3NSBRMTI1MCw3MiAxMjQ4LDg4CiAgICAgIFExMjQ2LDg1IDEyNDIsOTUgUTEyNDQsOTIgMTI0MCwxMDUgUTEyMzgsMTAyIDEyMzIsMTEyCiAgICAgIFExMjM0LDExMCAxMjI4LDEyMCBRMTIyNSwxMTggMTIxOCwxMjggUTEyMTUsMTI2IDEyMDgsMTM1CiAgICAgIFExMjA1LDEzMyAxMTk4LDE0MiBRMTE5NSwxNDAgMTE4OCwxNDggUTExODUsMTQ2IDExNzgsMTU0CiAgICAgIFExMTc1LDE1MiAxMTY4LDE1OCBRMTE2MiwxNTYgMTE1NSwxNjIgUTExNDgsMTYwIDExNDAsMTY0CiAgICAgIFExMTMyLDE2MiAxMTIyLDE2NSBRMTExMiwxNjMgMTEwMiwxNjAKICAgICAgUTEwOTAsMTU4IDEwNzgsMTUyIFExMDY4LDE0OCAxMDU4LDE0MAogICAgICBRMTA0OCwxMzUgMTA0MCwxMjUgUTEwMzIsMTE4IDEwMjUsMTA4CiAgICAgIFExMDIwLDEwMCAxMDE4LDkwICBRMTAxNSw4MCAgMTAxNiw3MAogICAgICBRMTAxNSw2MCAgMTAxOCw1MCAgUTEwMjIsNDIgIDEwMjgsMzQKICAgICAgUTEwMzUsMjYgIDEwNDQsMTggIFExMDUwLDEyICAxMDU1LDhaCiAgICAiIGZpbGw9IiMyZTZlNDAiIG9wYWNpdHk9IjAuNzgiLz4KCiAgICA8IS0tIEphcGFuIG1haW4gaXNsYW5kIChIb25zaHUpIC0tPgogICAgPHBhdGggZD0iCiAgICAgIE0xMTQ4LDU1IFExMTUyLDUwIDExNTgsNTIgUTExNjMsNDggMTE2OCw1MwogICAgICBRMTE3Miw1MCAxMTc1LDU2IFExMTc4LDU0IDExODAsNjAKICAgICAgUTExODIsNTggMTE4Myw2NSBRMTE4MSw2MyAxMTgwLDcwCiAgICAgIFExMTc4LDY4IDExNzUsNzQgUTExNzIsNzIgMTE2OCw3OAogICAgICBRMTE2NCw3NiAxMTYwLDgyIFExMTU2LDgwIDExNTIsODUKICAgICAgUTExNDgsODMgMTE0NSw3OCBRMTE0Myw3NCAxMTQ0LDY4CiAgICAgIFExMTQyLDY0IDExNDQsNTggUTExNDYsNTQgMTE0OCw1NVoKICAgICIgZmlsbD0iIzM1NjY0NCIgb3BhY2l0eT0iMC44NCIvPgoKICAgIDwhLS0gSmFwYW4gS3l1c2h1IC0tPgogICAgPHBhdGggZD0iCiAgICAgIE0xMTM4LDgwIFExMTQyLDc2IDExNDUsODAgUTExNDcsNzggMTE0OCw4NAogICAgICBRMTE0Niw4MiAxMTQ0LDg4IFExMTQxLDg2IDExMzgsOTAKICAgICAgUTExMzUsODggMTEzNCw4NCBRMTEzNCw4MCAxMTM4LDgwWgogICAgIiBmaWxsPSIjMzU2NjQ0IiBvcGFjaXR5PSIwLjgwIi8+CgogICAgPCEtLSBKYXBhbiBTaGlrb2t1IC0tPgogICAgPHBhdGggZD0iCiAgICAgIE0xMTQ4LDg0IFExMTUyLDgwIDExNTYsODQgUTExNTgsODIgMTE1OCw4OAogICAgICBRMTE1NSw4NiAxMTUyLDkwIFExMTQ4LDg4IDExNDcsODQgUTExNDcsODMgMTE0OCw4NFoKICAgICIgZmlsbD0iIzM1NjY0NCIgb3BhY2l0eT0iMC43OCIvPgoKICAgIDwhLS0gS29yZWFuIHBlbmluc3VsYSAtLT4KICAgIDxwYXRoIGQ9IgogICAgICBNMTEyOCw2OCBRMTEzMyw2NCAxMTM2LDY4IFExMTM4LDY2IDExMzgsNzQKICAgICAgUTExMzYsNzIgMTEzNCw3OCBRMTEzMCw3NiAxMTI4LDgwCiAgICAgIFExMTI1LDc4IDExMjUsNzIgUTExMjUsNjggMTEyOCw2OFoKICAgICIgZmlsbD0iIzJlNmU0MCIgb3BhY2l0eT0iMC44MCIvPgoKICAgIDwhLS0gVGFpd2FuIC0tPgogICAgPHBhdGggZD0iCiAgICAgIE0xMTQ4LDEwMCBRMTE1MSw5NyAxMTUzLDEwMCBRMTE1NCw5OCAxMTUzLDEwNAogICAgICBRMTE1MSwxMDMgMTE0OCwxMDUgUTExNDYsMTAzIDExNDcsMTAwWgogICAgIiBmaWxsPSIjMmU2ZTQwIiBvcGFjaXR5PSIwLjc1Ii8+CgogICAgPCEtLSBTYWtoYWxpbiAtLT4KICAgIDxwYXRoIGQ9IgogICAgICBNMTE1NSwzMCBRMTE1OCwyNSAxMTYxLDI4IFExMTYyLDI2IDExNjIsMzQKICAgICAgUTExNjAsMzIgMTE1OCwzOCBRMTE1NSwzNiAxMTU0LDMyIFExMTU0LDI5IDExNTUsMzBaCiAgICAiIGZpbGw9IiMyZTZlNDAiIG9wYWNpdHk9IjAuNzAiLz4KCiAgICA8IS0tIEF0bW9zcGhlcmUgKyB0ZXJtaW5hdG9yIChhcHBsaWVkIG92ZXIgdGhlIGNsaXBwZWQgZWFydGgpIC0tPgogICAgPGNpcmNsZSBjeD0iMTE4MCIgY3k9IjEwMCIgcj0iMTcwIiBmaWxsPSJ1cmwoI2F0bW8pIi8+CiAgICA8Y2lyY2xlIGN4PSIxMTgwIiBjeT0iMTAwIiByPSIxNzAiIGZpbGw9InVybCgjbmlnaHQpIiBvcGFjaXR5PSIwLjU1Ii8+CgogICAgPCEtLSBDbG91ZCB3aXNwcyAod2l0aGluIGVhcnRoQ2xpcCkgLS0+CiAgICA8ZyBvcGFjaXR5PSIwLjI4Ij4KICAgICAgPGVsbGlwc2UgY3g9IjEwNzIiIGN5PSI0MiIgIHJ4PSIzMCIgcnk9IjYiICBmaWxsPSIjZWVmNmZmIiBvcGFjaXR5PSIwLjY1IiB0cmFuc2Zvcm09InJvdGF0ZSgtMTAsMTA3Miw0MikiLz4KICAgICAgPGVsbGlwc2UgY3g9IjExMTUiIGN5PSIzMiIgIHJ4PSIyMiIgcnk9IjUiICBmaWxsPSIjZWVmNmZmIiBvcGFjaXR5PSIwLjU1IiB0cmFuc2Zvcm09InJvdGF0ZSgtNiwxMTE1LDMyKSIvPgogICAgICA8ZWxsaXBzZSBjeD0iMTE2MCIgY3k9IjM4IiAgcng9IjI4IiByeT0iNSIgIGZpbGw9IiNlZWY2ZmYiIG9wYWNpdHk9IjAuNTAiIHRyYW5zZm9ybT0icm90YXRlKC00LDExNjAsMzgpIi8+CiAgICAgIDxlbGxpcHNlIGN4PSIxMjAwIiBjeT0iNTUiICByeD0iMjAiIHJ5PSI0IiAgZmlsbD0iI2VlZjZmZiIgb3BhY2l0eT0iMC40NSIgdHJhbnNmb3JtPSJyb3RhdGUoMywxMjAwLDU1KSIvPgogICAgICA8ZWxsaXBzZSBjeD0iMTEwMCIgY3k9IjE0MiIgcng9IjMyIiByeT0iNiIgIGZpbGw9IiNlZWY2ZmYiIG9wYWNpdHk9IjAuNDIiIHRyYW5zZm9ybT0icm90YXRlKDUsMTEwMCwxNDIpIi8+CiAgICAgIDxlbGxpcHNlIGN4PSIxMTU1IiBjeT0iMTUyIiByeD0iMjUiIHJ5PSI1IiAgZmlsbD0iI2VlZjZmZiIgb3BhY2l0eT0iMC4zOCIgdHJhbnNmb3JtPSJyb3RhdGUoOCwxMTU1LDE1MikiLz4KICAgICAgPGVsbGlwc2UgY3g9IjEwNDUiIGN5PSI4OCIgIHJ4PSIxOCIgcnk9IjQiICBmaWxsPSIjZWVmNmZmIiBvcGFjaXR5PSIwLjM1IiB0cmFuc2Zvcm09InJvdGF0ZSgtOCwxMDQ1LDg4KSIvPgogICAgPC9nPgoKICAgIDwhLS0gQ2l0eSBsaWdodHMgKG5pZ2h0IHNpZGUpIC0tPgogICAgPGcgZmlsdGVyPSJ1cmwoI2NpdHlHbG93KSI+CiAgICAgIDwhLS0gVG9reW8gLS0+CiAgICAgIDxjaXJjbGUgY3g9IjExNjUiIGN5PSI3MiIgIHI9IjIuMiIgZmlsbD0iI2ZmZDg4MCIgb3BhY2l0eT0iMC45MiIvPgogICAgICA8Y2lyY2xlIGN4PSIxMTY4IiBjeT0iNzciICByPSIxLjQiIGZpbGw9IiNmZmNjNjAiIG9wYWNpdHk9IjAuODUiLz4KICAgICAgPGNpcmNsZSBjeD0iMTE2MSIgY3k9Ijc4IiAgcj0iMS4yIiBmaWxsPSIjZmZkODgwIiBvcGFjaXR5PSIwLjgwIi8+CiAgICAgIDwhLS0gT3Nha2EgLS0+CiAgICAgIDxjaXJjbGUgY3g9IjExNTYiIGN5PSI4MCIgIHI9IjEuNSIgZmlsbD0iI2ZmZDA3MCIgb3BhY2l0eT0iMC44MiIvPgogICAgICA8IS0tIE5hZ295YSAtLT4KICAgICAgPGNpcmNsZSBjeD0iMTE2MCIgY3k9Ijc0IiAgcj0iMS4wIiBmaWxsPSIjZmZlMDkwIiBvcGFjaXR5PSIwLjc2Ii8+CiAgICAgIDwhLS0gU2VvdWwgLS0+CiAgICAgIDxjaXJjbGUgY3g9IjExMzMiIGN5PSI3MyIgIHI9IjEuOCIgZmlsbD0iI2ZmYzg2MCIgb3BhY2l0eT0iMC44NCIvPgogICAgICA8Y2lyY2xlIGN4PSIxMTM2IiBjeT0iNzciICByPSIxLjEiIGZpbGw9IiNmZmQ4ODAiIG9wYWNpdHk9IjAuNzYiLz4KICAgICAgPCEtLSBTaGFuZ2hhaSAtLT4KICAgICAgPGNpcmNsZSBjeD0iMTE0MCIgY3k9IjkwIiAgcj0iMS45IiBmaWxsPSIjZmZkMDcwIiBvcGFjaXR5PSIwLjg2Ii8+CiAgICAgIDxjaXJjbGUgY3g9IjExNDUiIGN5PSI5NCIgIHI9IjEuMiIgZmlsbD0iI2ZmY2M2MCIgb3BhY2l0eT0iMC43OCIvPgogICAgICA8IS0tIEJlaWppbmcgLS0+CiAgICAgIDxjaXJjbGUgY3g9IjExMjAiIGN5PSI2NSIgIHI9IjEuNiIgZmlsbD0iI2ZmYzg1MCIgb3BhY2l0eT0iMC44MiIvPgogICAgICA8IS0tIFNoZW55YW5nIC0tPgogICAgICA8Y2lyY2xlIGN4PSIxMTI4IiBjeT0iNTgiICByPSIxLjIiIGZpbGw9IiNmZmQwNjAiIG9wYWNpdHk9IjAuNzQiLz4KICAgICAgPCEtLSBGdWt1b2thIC0tPgogICAgICA8Y2lyY2xlIGN4PSIxMTQzIiBjeT0iODIiICByPSIxLjAiIGZpbGw9IiNmZmQwODAiIG9wYWNpdHk9IjAuNzIiLz4KICAgICAgPCEtLSBTY2F0dGVyZWQgY29hc3RhbCAtLT4KICAgICAgPGNpcmNsZSBjeD0iMTEwOCIgY3k9Ijg4IiAgcj0iMC45IiBmaWxsPSIjZmZiYjQwIiBvcGFjaXR5PSIwLjYyIi8+CiAgICAgIDxjaXJjbGUgY3g9IjEwOTUiIGN5PSI5NSIgIHI9IjAuOCIgZmlsbD0iI2ZmZDA2MCIgb3BhY2l0eT0iMC41OCIvPgogICAgICA8Y2lyY2xlIGN4PSIxMDgyIiBjeT0iMTAyIiByPSIxLjAiIGZpbGw9IiNmZmNjNTAiIG9wYWNpdHk9IjAuNjIiLz4KICAgICAgPGNpcmNsZSBjeD0iMTA2OCIgY3k9IjExMCIgcj0iMC44IiBmaWxsPSIjZmZkMDcwIiBvcGFjaXR5PSIwLjU2Ii8+CiAgICAgIDxjaXJjbGUgY3g9IjExNDgiIGN5PSIxMDAiIHI9IjAuOCIgZmlsbD0iI2ZmY2M2MCIgb3BhY2l0eT0iMC42NSIvPgogICAgPC9nPgogIDwvZz4KCiAgPCEtLSBFYXJ0aCByaW0gKGF0bW9zcGhlcmUgZ2xvdyByaW5nLCBvdXRzaWRlIGVhcnRoQ2xpcCBzbyBpdCByZW5kZXJzIG9uIHRvcCkgLS0+CiAgPGcgY2xpcC1wYXRoPSJ1cmwoI2Jhbm5lcikiPgogICAgPGNpcmNsZSBjeD0iMTE4MCIgY3k9IjEwMCIgcj0iMTcyIiBmaWxsPSJub25lIiBzdHJva2U9IiMzYWIwZTgiIHN0cm9rZS13aWR0aD0iMi41IiBvcGFjaXR5PSIwLjM4Ii8+CiAgICA8Y2lyY2xlIGN4PSIxMTgwIiBjeT0iMTAwIiByPSIxNzYiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzFhODBjMCIgc3Ryb2tlLXdpZHRoPSIxLjIiIG9wYWNpdHk9IjAuMTgiLz4KICA8L2c+CgogIDwhLS0g4pWQ4pWQ4pWQIEdOU1MgU0FURUxMSVRFUyDilZDilZDilZAgLS0+CiAgPGcgY2xpcC1wYXRoPSJ1cmwoI2Jhbm5lcikiPgoKICAgIDwhLS0gU0FUIDE6IEdQUyAoYmx1ZSBwYW5lbHMpIC0tPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTk1LDM2KSIgZmlsdGVyPSJ1cmwoI3NhdEdsb3cpIj4KICAgICAgPHJlY3QgeD0iLTMwIiB5PSItNC41IiB3aWR0aD0iMjQiIGhlaWdodD0iOSIgZmlsbD0iIzFlNGQ5OSIgb3BhY2l0eT0iMC45MyIgcng9IjEuNSIvPgogICAgICA8bGluZSB4MT0iLTIwIiB5MT0iLTQuNSIgeDI9Ii0yMCIgeTI9IjQuNSIgc3Ryb2tlPSIjNTU4OGRkIiBzdHJva2Utd2lkdGg9IjAuNiIgb3BhY2l0eT0iMC43Ii8+CiAgICAgIDxsaW5lIHgxPSItMTIiIHkxPSItNC41IiB4Mj0iLTEyIiB5Mj0iNC41IiBzdHJva2U9IiM1NTg4ZGQiIHN0cm9rZS13aWR0aD0iMC42IiBvcGFjaXR5PSIwLjciLz4KICAgICAgPHJlY3QgeD0iNiIgICB5PSItNC41IiB3aWR0aD0iMjQiIGhlaWdodD0iOSIgZmlsbD0iIzFlNGQ5OSIgb3BhY2l0eT0iMC45MyIgcng9IjEuNSIvPgogICAgICA8bGluZSB4MT0iMTMiIHkxPSItNC41IiB4Mj0iMTMiIHkyPSI0LjUiIHN0cm9rZT0iIzU1ODhkZCIgc3Ryb2tlLXdpZHRoPSIwLjYiIG9wYWNpdHk9IjAuNyIvPgogICAgICA8bGluZSB4MT0iMjAiIHkxPSItNC41IiB4Mj0iMjAiIHkyPSI0LjUiIHN0cm9rZT0iIzU1ODhkZCIgc3Ryb2tlLXdpZHRoPSIwLjYiIG9wYWNpdHk9IjAuNyIvPgogICAgICA8cmVjdCB4PSItNiIgeT0iLTciIHdpZHRoPSIxMiIgaGVpZ2h0PSIxNCIgZmlsbD0iI2MwZDBlOCIgb3BhY2l0eT0iMC45NSIgcng9IjIiLz4KICAgICAgPHJlY3QgeD0iLTQiIHk9Ii01IiB3aWR0aD0iOCIgaGVpZ2h0PSIxMCIgZmlsbD0iIzg4YThjYyIgb3BhY2l0eT0iMC44MCIgcng9IjEiLz4KICAgICAgPGVsbGlwc2UgY3g9IjAiIGN5PSItMTAiIHJ4PSI0LjUiIHJ5PSIyIiBmaWxsPSJub25lIiBzdHJva2U9IiNkOGVhZjgiIHN0cm9rZS13aWR0aD0iMC45IiBvcGFjaXR5PSIwLjg1Ii8+CiAgICAgIDxsaW5lIHgxPSIwIiB5MT0iLTgiIHgyPSIwIiB5Mj0iLTciIHN0cm9rZT0iI2Q4ZWFmOCIgc3Ryb2tlLXdpZHRoPSIwLjkiIG9wYWNpdHk9IjAuODUiLz4KICAgICAgPGNpcmNsZSBjeD0iMCIgY3k9IjAiIHI9IjMuNSIgZmlsbD0iIzYwYTVmYSIgb3BhY2l0eT0iMC41NSIvPgogICAgPC9nPgoKICAgIDwhLS0gU0FUIDI6IEdMT05BU1MgKGdyZWVuIHBhbmVscykgLS0+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg1NDAsMjApIiBmaWx0ZXI9InVybCgjc2F0R2xvdykiPgogICAgICA8cmVjdCB4PSItMjYiIHk9Ii0zLjUiIHdpZHRoPSIyMCIgaGVpZ2h0PSI3IiBmaWxsPSIjMTY1ZTMwIiBvcGFjaXR5PSIwLjkyIiByeD0iMS41Ii8+CiAgICAgIDxsaW5lIHgxPSItMTgiIHkxPSItMy41IiB4Mj0iLTE4IiB5Mj0iMy41IiBzdHJva2U9IiMzNGQzOTkiIHN0cm9rZS13aWR0aD0iMC42IiBvcGFjaXR5PSIwLjY4Ii8+CiAgICAgIDxsaW5lIHgxPSItMTAiIHkxPSItMy41IiB4Mj0iLTEwIiB5Mj0iMy41IiBzdHJva2U9IiMzNGQzOTkiIHN0cm9rZS13aWR0aD0iMC42IiBvcGFjaXR5PSIwLjY4Ii8+CiAgICAgIDxyZWN0IHg9IjYiICB5PSItMy41IiB3aWR0aD0iMjAiIGhlaWdodD0iNyIgZmlsbD0iIzE2NWUzMCIgb3BhY2l0eT0iMC45MiIgcng9IjEuNSIvPgogICAgICA8bGluZSB4MT0iMTIiIHkxPSItMy41IiB4Mj0iMTIiIHkyPSIzLjUiIHN0cm9rZT0iIzM0ZDM5OSIgc3Ryb2tlLXdpZHRoPSIwLjYiIG9wYWNpdHk9IjAuNjgiLz4KICAgICAgPGxpbmUgeDE9IjE4IiB5MT0iLTMuNSIgeDI9IjE4IiB5Mj0iMy41IiBzdHJva2U9IiMzNGQzOTkiIHN0cm9rZS13aWR0aD0iMC42IiBvcGFjaXR5PSIwLjY4Ii8+CiAgICAgIDxyZWN0IHg9Ii01IiB5PSItNiIgd2lkdGg9IjEwIiBoZWlnaHQ9IjEyIiBmaWxsPSIjYThjOGIwIiBvcGFjaXR5PSIwLjkzIiByeD0iMiIvPgogICAgICA8cmVjdCB4PSItMyIgeT0iLTQiIHdpZHRoPSI2IiBoZWlnaHQ9IjgiIGZpbGw9IiM4MGE4ODgiIG9wYWNpdHk9IjAuNzgiIHJ4PSIxIi8+CiAgICAgIDxlbGxpcHNlIGN4PSIwIiBjeT0iLTkiIHJ4PSI0IiByeT0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjYzBkY2MwIiBzdHJva2Utd2lkdGg9IjAuOSIgb3BhY2l0eT0iMC44MiIvPgogICAgICA8bGluZSB4MT0iMCIgeTE9Ii03IiB4Mj0iMCIgeTI9Ii02IiBzdHJva2U9IiNjMGRjYzAiIHN0cm9rZS13aWR0aD0iMC45Ii8+CiAgICAgIDxjaXJjbGUgY3g9IjAiIGN5PSIwIiByPSIzIiBmaWxsPSIjMzRkMzk5IiBvcGFjaXR5PSIwLjUwIi8+CiAgICA8L2c+CgogICAgPCEtLSBTQVQgMzogR2FsaWxlbyAoZ29sZCBwYW5lbHMpIC0tPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzY4LDU4KSIgZmlsdGVyPSJ1cmwoI3NhdEdsb3cpIj4KICAgICAgPHJlY3QgeD0iLTM0IiB5PSItNCIgd2lkdGg9IjI4IiBoZWlnaHQ9IjgiIGZpbGw9IiM2YjQ4MDAiIG9wYWNpdHk9IjAuOTAiIHJ4PSIxLjUiLz4KICAgICAgPGxpbmUgeDE9Ii0yNCIgeTE9Ii00IiB4Mj0iLTI0IiB5Mj0iNCIgc3Ryb2tlPSIjZjU5ZTBiIiBzdHJva2Utd2lkdGg9IjAuNiIgb3BhY2l0eT0iMC43MiIvPgogICAgICA8bGluZSB4MT0iLTE1IiB5MT0iLTQiIHgyPSItMTUiIHkyPSI0IiBzdHJva2U9IiNmNTllMGIiIHN0cm9rZS13aWR0aD0iMC42IiBvcGFjaXR5PSIwLjcyIi8+CiAgICAgIDxyZWN0IHg9IjYiICB5PSItNCIgd2lkdGg9IjI4IiBoZWlnaHQ9IjgiIGZpbGw9IiM2YjQ4MDAiIG9wYWNpdHk9IjAuOTAiIHJ4PSIxLjUiLz4KICAgICAgPGxpbmUgeDE9IjE1IiB5MT0iLTQiIHgyPSIxNSIgeTI9IjQiIHN0cm9rZT0iI2Y1OWUwYiIgc3Ryb2tlLXdpZHRoPSIwLjYiIG9wYWNpdHk9IjAuNzIiLz4KICAgICAgPGxpbmUgeDE9IjI0IiB5MT0iLTQiIHgyPSIyNCIgeTI9IjQiIHN0cm9rZT0iI2Y1OWUwYiIgc3Ryb2tlLXdpZHRoPSIwLjYiIG9wYWNpdHk9IjAuNzIiLz4KICAgICAgPHJlY3QgeD0iLTYiIHk9Ii04IiB3aWR0aD0iMTIiIGhlaWdodD0iMTYiIGZpbGw9IiNjY2MwODAiIG9wYWNpdHk9IjAuOTIiIHJ4PSIyIi8+CiAgICAgIDxyZWN0IHg9Ii00IiB5PSItNiIgd2lkdGg9IjgiIGhlaWdodD0iMTIiIGZpbGw9IiNhYWEwNjAiIG9wYWNpdHk9IjAuNzYiIHJ4PSIxIi8+CiAgICAgIDxlbGxpcHNlIGN4PSIwIiBjeT0iLTExIiByeD0iNC41IiByeT0iMiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjZWNlMGEwIiBzdHJva2Utd2lkdGg9IjAuOSIgb3BhY2l0eT0iMC44MiIvPgogICAgICA8bGluZSB4MT0iMCIgeTE9Ii05IiB4Mj0iMCIgeTI9Ii04IiBzdHJva2U9IiNlY2UwYTAiIHN0cm9rZS13aWR0aD0iMC45Ii8+CiAgICAgIDxjaXJjbGUgY3g9IjAiIGN5PSIwIiByPSIzLjUiIGZpbGw9IiNmYmJmMjQiIG9wYWNpdHk9IjAuNTUiLz4KICAgIDwvZz4KCiAgICA8IS0tIFNBVCA0OiBRWlNTIChwdXJwbGUsIHNtYWxsKSAtLT4KICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDY1LDUwKSIgZmlsdGVyPSJ1cmwoI3NhdEdsb3cpIiBvcGFjaXR5PSIwLjgyIj4KICAgICAgPHJlY3QgeD0iLTIwIiB5PSItMyIgd2lkdGg9IjE1IiBoZWlnaHQ9IjYiIGZpbGw9IiM0YTFhOTkiIG9wYWNpdHk9IjAuOTAiIHJ4PSIxLjUiLz4KICAgICAgPGxpbmUgeDE9Ii0xMyIgeTE9Ii0zIiB4Mj0iLTEzIiB5Mj0iMyIgc3Ryb2tlPSIjYTc4YmZhIiBzdHJva2Utd2lkdGg9IjAuNSIgb3BhY2l0eT0iMC42OCIvPgogICAgICA8cmVjdCB4PSI1IiAgeT0iLTMiIHdpZHRoPSIxNSIgaGVpZ2h0PSI2IiBmaWxsPSIjNGExYTk5IiBvcGFjaXR5PSIwLjkwIiByeD0iMS41Ii8+CiAgICAgIDxsaW5lIHgxPSIxMCIgeTE9Ii0zIiB4Mj0iMTAiIHkyPSIzIiBzdHJva2U9IiNhNzhiZmEiIHN0cm9rZS13aWR0aD0iMC41IiBvcGFjaXR5PSIwLjY4Ii8+CiAgICAgIDxyZWN0IHg9Ii00IiB5PSItNSIgd2lkdGg9IjgiIGhlaWdodD0iMTAiIGZpbGw9IiNiOGE4ZDgiIG9wYWNpdHk9IjAuOTIiIHJ4PSIyIi8+CiAgICAgIDxjaXJjbGUgY3g9IjAiIGN5PSIwIiByPSIyLjUiIGZpbGw9IiNhNzhiZmEiIG9wYWNpdHk9IjAuNTAiLz4KICAgIDwvZz4KCiAgICA8IS0tIFNpZ25hbCBiZWFtcyBzYXTihpJlYXJ0aCAoZGFzaGVkIGxpbmVzKSAtLT4KICAgIDxsaW5lIHgxPSIxOTUiIHkxPSI0MCIgIHgyPSIxMTYzIiB5Mj0iNzQiICBzdHJva2U9IiM2MGQwZmYiIHN0cm9rZS13aWR0aD0iMC44IiBzdHJva2UtZGFzaGFycmF5PSI3LDExIiBvcGFjaXR5PSIwLjQyIi8+CiAgICA8bGluZSB4MT0iNTQwIiB5MT0iMjQiICB4Mj0iMTEzMyIgeTI9Ijc2IiAgc3Ryb2tlPSIjYTBmZmIwIiBzdHJva2Utd2lkdGg9IjAuNyIgc3Ryb2tlLWRhc2hhcnJheT0iNiwxMCIgb3BhY2l0eT0iMC4zOCIvPgogICAgPGxpbmUgeDE9IjM2OCIgeTE9IjYyIiAgeDI9IjExNDAiIHkyPSI5MiIgIHN0cm9rZT0iI2ZmZGQ2MCIgc3Ryb2tlLXdpZHRoPSIwLjciIHN0cm9rZS1kYXNoYXJyYXk9IjYsMTEiIG9wYWNpdHk9IjAuMzYiLz4KICAgIDxsaW5lIHgxPSI2NSIgIHkxPSI1NCIgIHgyPSIxMTIwIiB5Mj0iNjgiICBzdHJva2U9IiNjMDg0ZmMiIHN0cm9rZS13aWR0aD0iMC42IiBzdHJva2UtZGFzaGFycmF5PSI1LDExIiBvcGFjaXR5PSIwLjI4Ii8+CgogICAgPCEtLSBPcmJpdCBhcmNzIC0tPgogICAgPGVsbGlwc2UgY3g9IjcwMCIgY3k9IjM1MCIgcng9Ijg2MCIgcnk9IjE1NSIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNjBhNWZhIiBzdHJva2Utd2lkdGg9IjAuOCIgb3BhY2l0eT0iMC4xNiIgdHJhbnNmb3JtPSJyb3RhdGUoLTUsNzAwLDM1MCkiIGNsaXAtcGF0aD0idXJsKCNiYW5uZXIpIi8+CiAgICA8ZWxsaXBzZSBjeD0iNjIwIiBjeT0iMzEwIiByeD0iNzQwIiByeT0iMTE4IiBmaWxsPSJub25lIiBzdHJva2U9IiMzNGQzOTkiIHN0cm9rZS13aWR0aD0iMC42IiBvcGFjaXR5PSIwLjE0IiB0cmFuc2Zvcm09InJvdGF0ZSg0LDYyMCwzMTApIiAgY2xpcC1wYXRoPSJ1cmwoI2Jhbm5lcikiLz4KICAgIDxlbGxpcHNlIGN4PSI2NjAiIGN5PSIzMzAiIHJ4PSI4MDAiIHJ5PSIxMzYiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI2ZiYmYyNCIgc3Ryb2tlLXdpZHRoPSIwLjUiIG9wYWNpdHk9IjAuMTMiIHRyYW5zZm9ybT0icm90YXRlKC0yLDY2MCwzMzApIiBjbGlwLXBhdGg9InVybCgjYmFubmVyKSIvPgoKICAgIDwhLS0gUHVsc2UgcmluZ3Mgb24gY2l0eSBsaWdodHMgLS0+CiAgICA8ZyBmaWx0ZXI9InVybCgjcmluZ0dsb3cpIiBjbGlwLXBhdGg9InVybCgjZWFydGhDbGlwKSI+CiAgICAgIDxjaXJjbGUgY3g9IjExNjUiIGN5PSI3MiIgIHI9IjEwIiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmQ4ODAiIHN0cm9rZS13aWR0aD0iMC44IiBvcGFjaXR5PSIwLjQ4Ii8+CiAgICAgIDxjaXJjbGUgY3g9IjExNjUiIGN5PSI3MiIgIHI9IjE4IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmQ4ODAiIHN0cm9rZS13aWR0aD0iMC41IiBvcGFjaXR5PSIwLjI2Ii8+CiAgICAgIDxjaXJjbGUgY3g9IjExMzMiIGN5PSI3MyIgIHI9IjgiICBmaWxsPSJub25lIiBzdHJva2U9IiNhMGZmYjAiIHN0cm9rZS13aWR0aD0iMC44IiBvcGFjaXR5PSIwLjQwIi8+CiAgICAgIDxjaXJjbGUgY3g9IjExNDAiIGN5PSI5MCIgIHI9IjciICBmaWxsPSJub25lIiBzdHJva2U9IiNmZmRkNjAiIHN0cm9rZS13aWR0aD0iMC44IiBvcGFjaXR5PSIwLjM2Ii8+CiAgICA8L2c+CgogICAgPCEtLSBUZWNoIGdyaWQgKGZhaW50KSAtLT4KICAgIDxnIG9wYWNpdHk9IjAuMDU1IiBjbGlwLXBhdGg9InVybCgjYmFubmVyKSI+CiAgICAgIDxsaW5lIHgxPSIwIiB5MT0iNjciICB4Mj0iOTUwIiB5Mj0iNjciICBzdHJva2U9IiM2MGE1ZmEiIHN0cm9rZS13aWR0aD0iMC41Ii8+CiAgICAgIDxsaW5lIHgxPSIwIiB5MT0iMTMzIiB4Mj0iOTUwIiB5Mj0iMTMzIiBzdHJva2U9IiM2MGE1ZmEiIHN0cm9rZS13aWR0aD0iMC41Ii8+CiAgICAgIDxsaW5lIHgxPSIyMjAiIHkxPSIwIiAgeDI9IjIyMCIgeTI9IjIwMCIgc3Ryb2tlPSIjNjBhNWZhIiBzdHJva2Utd2lkdGg9IjAuNSIvPgogICAgICA8bGluZSB4MT0iNDQwIiB5MT0iMCIgIHgyPSI0NDAiIHkyPSIyMDAiIHN0cm9rZT0iIzYwYTVmYSIgc3Ryb2tlLXdpZHRoPSIwLjUiLz4KICAgICAgPGxpbmUgeDE9IjY2MCIgeTE9IjAiICB4Mj0iNjYwIiB5Mj0iMjAwIiBzdHJva2U9IiM2MGE1ZmEiIHN0cm9rZS13aWR0aD0iMC41Ii8+CiAgICAgIDxsaW5lIHgxPSI4ODAiIHkxPSIwIiAgeDI9Ijg4MCIgeTI9IjIwMCIgc3Ryb2tlPSIjNjBhNWZhIiBzdHJva2Utd2lkdGg9IjAuNSIvPgogICAgPC9nPgogIDwvZz4KCiAgPCEtLSBUZXh0IHJlYWRhYmlsaXR5IG92ZXJsYXkgLS0+CiAgPHJlY3Qgd2lkdGg9IjE0MDAiIGhlaWdodD0iMjAwIiBmaWxsPSJ1cmwoI3RleHRGYWRlKSIgY2xpcC1wYXRoPSJ1cmwoI2Jhbm5lcikiLz4KCiAgPCEtLSBUb3Agc2hpbW1lciBsaW5lIC0tPgogIDxyZWN0IHg9IjAiIHk9IjAiIHdpZHRoPSIxNDAwIiBoZWlnaHQ9IjEuNSIgZmlsbD0idXJsKCNzaGltbWVyKSIgY2xpcC1wYXRoPSJ1cmwoI2Jhbm5lcikiIG9wYWNpdHk9IjAuOSIvPgo8L3N2Zz4=") center/cover no-repeat;
  border-radius: 16px;
  padding: 28px 36px 26px;
  position: relative;
  overflow: hidden;
  margin-bottom: 0;
  box-shadow: 0 4px 32px rgba(0,10,40,0.55), 0 1px 0 rgba(255,255,255,0.06) inset;
}
/* top shimmer line */
.app-hdr::after {
  content: "";
  position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, rgba(96,200,255,0.6), rgba(255,255,255,0.4), rgba(96,200,255,0.6), transparent);
  z-index: 2;
}
.app-hdr h1 {
  margin: 0;
  font-size: 28px;
  font-weight: 800;
  color: #f0f8ff;
  letter-spacing: -0.02em;
  text-shadow: 0 0 24px rgba(96,180,255,0.7), 0 2px 6px rgba(0,0,0,0.8);
  position: relative; z-index: 1;
}
.app-hdr p  {
  margin: 7px 0 0;
  color: #7ec8f0;
  font-size: 13px;
  font-weight: 500;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  text-shadow: 0 1px 8px rgba(0,20,60,0.9);
  position: relative; z-index: 1;
}
.app-hdr-btn a:hover { background: rgba(255,255,255,0.28); }

/* ── バッジ類 ── */
.zbadge { display:inline-block; background:#f59e0b; color:#fff; font-weight:700;
          font-size:11px; padding:2px 10px; border-radius:6px; margin:4px 0; }
.zinfo  { font-size:11px; color:#94a3b8; margin-top:2px; }
.acc    { display:inline-block; background:#dcfce7; color:#15803d; font-size:10px;
          font-weight:700; padding:2px 8px; border-radius:4px; }
.geoid-ok { color:#16a34a; font-size:11px; font-weight:600; }
.geoid-ng { color:#dc2626; font-size:11px; font-weight:600; }
.err  { color:#ef4444; font-size:12px; }
.sec-label { font-size:10px; font-weight:700; color:#94a3b8;
             letter-spacing:.18em; text-transform:uppercase; margin:16px 0 8px; }

/* 入力フォーマット選択チップ */
.fmt-pill { display:inline-block; background:#f1f5f9; color:#475569; border:1px solid #e2e8f0;
            border-radius:20px; font-size:11px; padding:3px 12px; margin:2px; cursor:pointer;
            font-family:'DM Mono',monospace; }
.fmt-pill.active { background:#0f172a; color:#f1f5f9; border-color:#0f172a; }

/* pydeck地図の高さ統一 */
.element-container iframe { border-radius:12px; }

/* ── サイドバーを完全非表示 ── */
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

/* ── タブ：見やすく・レスポンシブ対応 ── */
div[data-testid="stTabs"] > div:first-child {
  gap: 8px;
  border-bottom: 2px solid #e2e8f0;
  margin-bottom: 18px;
  flex-wrap: nowrap;
}
div[data-testid="stTabs"] button[role="tab"] {
  font-size: 15px !important;
  font-weight: 700 !important;
  padding: 10px 24px !important;
  border-radius: 10px 10px 0 0 !important;
  border: 1.5px solid #e2e8f0 !important;
  border-bottom: none !important;
  background: #f8fafc !important;
  color: #64748b !important;
  white-space: nowrap;
  flex: 1 1 0;
  min-width: 0;
  transition: background .15s, color .15s;
}
div[data-testid="stTabs"] button[role="tab"]:hover {
  background: #e0f2fe !important;
  color: #0369a1 !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  background: linear-gradient(135deg, #1d4ed8, #0ea5e9) !important;
  color: #fff !important;
  border-color: #1d4ed8 !important;
  box-shadow: 0 2px 10px rgba(29,78,216,0.25);
}
div[data-testid="stTabs"] button[role="tab"] p {
  font-size: 15px !important;
  font-weight: 700 !important;
  margin: 0 !important;
  overflow: hidden;
  text-overflow: ellipsis;
}

/* モバイル */
@media (max-width: 640px) {
  div[data-testid="stTabs"] button[role="tab"] {
    font-size: 13px !important;
    padding: 8px 10px !important;
  }
  div[data-testid="stTabs"] button[role="tab"] p {
    font-size: 13px !important;
  }
}

</style>
""", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════
# 7. サイドバー共通設定
# ═══════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════
# 7. 共通設定（メインエリア上部）
# ═══════════════════════════════════════════════════════

@st.dialog("❓ 使い方ガイド", width="large")
def _dlg_help():
    with st.expander("📌 1. 基本的な使い方", expanded=True):
        st.markdown("""
**GNSS SmartShift ICT でできること**

| 機能 | 内容 |
|---|---|
| **平面直角 ↔ 緯度経度 変換** | X・Y・Z標高 と 緯度・経度・楕円体高 の相互変換 |
| **緯度経度 形式変換** | 十進角度・度分秒・度分秒圧縮など5形式を相互変換 |
| **ジオイド高計算** | 国土地理院API（JPGEO2024/2011）で Z標高 ↔ 楕円体高 を自動補正 |
| **CSV一括変換** | 複数点を一括処理・結果をCSV出力（UTF-8 BOM付き） |
| **他社機ローカライゼーション対応** | 各社 GNSS データを建設 ICT ソフトウェア向け形式に変換 |

**基本の流れ**

1. **座標系（系番号）・測地系・ジオイドモデル** を設定
2. **単点変換** タブで座標を入力 → 変換結果・CSV出力を確認
3. 複数点は **CSV 一括変換** タブでまとめて処理
""")
    with st.expander("🗾 2. 座標系（系番号）の選び方 ＆ 一覧"):
        st.markdown("現場の都道府県で系番号が決まります。下表で確認してください。")
        st.caption("国土交通省告示（昭和48年建設省告示第143号）")
        _rows_z = []
        for _z in range(1, 20):
            _l0, _o0 = JPC_ORIGINS[_z]
            _rows_z.append({"系番号": _z, "適用地域": JPC_ZONE_LABELS[_z].split(" — ")[1],
                            "原点緯度 φ₀": f"{_l0}°", "原点経度 λ₀": f"{_o0}°",
                            "縮尺係数": "0.9999"})
        _dfz = pd.DataFrame(_rows_z)
        st.dataframe(_dfz, use_container_width=True, hide_index=True, height=660)
    with st.expander("🔗 3. 他社機ローカライゼーション対応 — 建設 ICT ソフトへの入力手順"):
        st.markdown("""
GNSS ローバー（Leica・Trimble・Topcon・Sokkia 等）で取得した **緯度・経度・楕円体高** を、建設 ICT ソフトウェア（3DOffice・SiTECH 3D・TREND-POINT 等）のローカライゼーション設定に必要な形式に変換します。

---

**STEP 1｜共通設定を現場に合わせる**

| 設定項目 | 内容 |
|---|---|
| **座標系（系番号）** | 現場の都道府県に対応する系番号（「使い方」→ 2 で確認） |
| **測地系** | JGD2024 または JGD2011（推奨） |
| **ジオイドモデル** | JPGEO2024（推奨）— Z標高を自動計算 |

---

**STEP 2｜単点変換タブ →「緯度経度 形式変換」を選択**

ローバーで取得した基準点（既知点）データを入力します。

---

**STEP 3｜入力フォーマットをローバーに合わせる**

| ローバー表示形式 | 選択フォーマット |
|---|---|
| 39.61921024 | DD.DDDDDDDD°（十進角度） |
| 39°37′09.157″ | DD°MM′SS.SSS″（度分秒） |
| N39°37′09.157″ | NDD°MM′SS.SSS″（方位角） |
| 39.370915700000 | DD.MMSSSSSS（度分秒圧縮） |

---

**STEP 4｜点名・緯度・経度・楕円体高（h）を入力**

- **「＋ 点を追加」** で複数の基準点を追加できます
- 楕円体高（h）は GNSS ローバーが出力する **楕円体高（Ellipsoidal Height）** を入力してください（標高ではありません）

---

**STEP 5｜出力フォーマットを建設 ICT ソフトに合わせる**

| ソフトウェア | 推奨出力フォーマット |
|---|---|
| 3DOffice / SiTECH 3D | DD.MMSSSSSS（度分秒圧縮） |
| TREND-POINT / FIELD 等 | DD.DDDDDDDD°（十進角度）または DD°MM′SS.SSS″ |
| CSV 取込（汎用） | DD.DDDDDDDD°（十進角度）推奨 |

---

**STEP 6｜変換結果を確認 → CSV ダウンロード**

変換結果テーブルの各列を建設 ICT ソフトのローカライゼーション設定に入力します。

| 出力列 | 建設 ICT ソフトでの使用箇所 |
|---|---|
| **X（m）** | 北方向座標（平面直角 X） |
| **Y（m）** | 東方向座標（平面直角 Y） |
| **Z標高（m）** | 標高（ジオイド補正済み） |
| **緯度** | 基準点緯度 |
| **経度** | 基準点経度 |
| **楕円体高（m）** | 楕円体高（h） |

「📥 全点 CSV ダウンロード」からファイルをエクスポートできます。
""")

    with st.expander("📐 4. 3DOffice 等 建設 ICT ソフトへの入力手順（詳細）"):
        st.markdown("""
**STEP 1｜座標系・測地系を設定する**

共通設定で **座標系（系番号）** と **測地系** を現場に合わせて設定してください。

---

**STEP 2｜「緯度経度 形式変換」タブを選択する**

**単点変換** タブ上部の **「緯度経度 形式変換」** を選択します。

---

**STEP 3｜入力フォーマットを合わせる**

| ローバーの表示形式 | 選択するフォーマット |
|---|---|
| 39.61921024 | DD.DDDDDDDD°（十進角度） |
| 39°37′09.157″ | DD°MM′SS.SSS″（度分秒） |
| N39°37′09.157″ | NDD°MM′SS.SSS″（方位角） |
| 39.370915700000 | DD.MMSSSSSS（度分秒圧縮） |

---

**STEP 4〜6｜入力・出力・CSV 出力**

- 点名・緯度・経度・楕円体高（h）を入力し **「＋ 点を追加」** で複数点追加
- 出力フォーマットを **DD.MMSSSSSS（度分秒圧縮）** に設定
- 変換結果の X・Y・Z標高・緯度・経度・楕円体高を建設 ICT ソフトに入力
""")
    with st.expander("📋 5. CSV一括変換の使い方"):
        st.markdown("""
**入力CSVのフォーマット（ヘッダーなし）**

| 変換方向 | A列 | B列 | C列 | D列 |
|---|---|---|---|---|
| **平面直角 → 緯度経度** | 点名 | X(m) | Y(m) | Z標高(m)・省略可 |
| **緯度経度 → 平面直角** | 点名 | 緯度 | 経度 | 楕円体高(m)・省略可 |

- 1行目からデータ（ヘッダー行不要）
- 5列目以降は無視
""")
    with st.expander("💬 6. よくある質問（FAQ）"):
        st.markdown("""
**Q. 系番号はどれを選べばいいですか？**

A. 上の「2. 座標系（系番号）の選び方 ＆ 一覧」で都道府県を確認してください。

---

**Q. ジオイド高の取得に時間がかかります / 失敗します**

A. 同じ座標は **24時間キャッシュ** されるため2回目以降は即座に表示されます。補正不要な場合は「ジオイドモデル」を **「補正なし」** に設定してください。

---

**Q. 往復変換するとXY座標が数mm変わります**

A. 正常な動作です。往復変換時は出力フォーマットを **十進角度（DD.DDDDDDDD°）** に設定してください。

---

**Q. 度分秒圧縮（DD.MMSSSSSS）が十進角度として判別されます**

A. **小数部12桁以上** にしてください。例：`140.555914380000`（14桁）→ 正しく処理されます。
""")

    st.markdown("---")
    st.markdown("#### 🎯 変換精度・仕様")
    st.markdown("""
| 項目 | 仕様 |
|---|---|
| **変換式** | Kawase (2011) 高次ガウス・クリューゲル展開式 |
| **準拠楕円体** | GRS80（JGD2024・JGD2011・JGD2000）／WGS84楕円体（WGS84） |
| **縮尺係数** | m₀ = 0.9999（全系共通） |
| **対応座標系** | 公共測量 平面直角座標系 1〜19系 |
| **往復変換誤差** | < 0.01 mm |
| **ジオイド高** | 国土地理院 API（JPGEO2024 / JPGEO2011）・24hキャッシュ |
| **旧日本測地系変換** | Helmert 3パラメータ（Δx=−148, Δy=+507, Δz=+685 m）|
| **JGD2024** | GRS80楕円体・令和6年告示。JGD2011と同一楕円体パラメータ |
""")


zone_inv = {v:k for k,v in JPC_ZONE_LABELS.items()}
datum_inv = {v["label"]:k for k,v in DATUMS.items()}

# ── タイトル ──
st.markdown("""
<div class="app-hdr">
  <h1>🛰️ GNSS SmartShift ICT</h1>
  <p>マルチメーカー対応 ローカライゼーション統合システム</p>
</div>""", unsafe_allow_html=True)

# ── 使い方ボタン（タイトルと共通設定の間） ──
_bBtn, _bSpc = st.columns([2, 8])
with _bBtn:
    if st.button("❓ 使い方", use_container_width=True, key="btn_help"):
        _dlg_help()

# ── 共通設定 ──
st.markdown("<div style='background:#f1f5f9;border:1.5px solid #cbd5e1;border-radius:12px;padding:8px 16px 6px;margin-bottom:6px;margin-top:4px'><span style='font-size:14px;font-weight:700;color:#1e3a5f;letter-spacing:.02em'>⚙️ 共通設定</span></div>", unsafe_allow_html=True)
_col1, _col2, _col3, _col4 = st.columns(4)
with _col1:
    st.markdown("<div style='font-size:11px;font-weight:700;color:#374151;margin-bottom:4px'>📌 座標系（系番号）</div>", unsafe_allow_html=True)
    zone_lbl = st.selectbox("座標系", list(JPC_ZONE_LABELS.values()),
                             index=8, label_visibility="collapsed", key="sel_zone")
    Z = zone_inv[zone_lbl]
    la0, lo0 = JPC_ORIGINS[Z]
with _col2:
    st.markdown("<div style='font-size:11px;font-weight:700;color:#374151;margin-bottom:4px'>🌐 測地系</div>", unsafe_allow_html=True)
    datum_lbl = st.selectbox("測地系", list(datum_inv.keys()),
                              index=0, label_visibility="collapsed", key="sel_datum")
    DATUM = datum_inv[datum_lbl]
with _col3:
    st.markdown("<div style='font-size:11px;font-weight:700;color:#374151;margin-bottom:4px'>📡 ジオイドモデル</div>", unsafe_allow_html=True)
    geoid_lbl = st.selectbox("ジオイドモデル", list(GEOID_MODELS.values()),
                              index=0, label_visibility="collapsed", key="sel_geoid")
    GEOID_KEY = [k for k,v in GEOID_MODELS.items() if v==geoid_lbl][0]
with _col4:
    st.markdown("<div style='font-size:11px;font-weight:700;color:#374151;margin-bottom:4px'>🗺️ 地図スタイル</div>", unsafe_allow_html=True)
    map_style_lbl = st.selectbox("地図スタイル", list(MAP_STYLES.keys()),
                                  index=1, label_visibility="collapsed", key="sel_map")

# ═══════════════════════════════════════════════════════
# 8. メインヘッダー
# ═══════════════════════════════════════════════════════

# FMT/fmt_lbl はTAB内の各モードで個別に定義するため、ここではデフォルトのみ設定
_FMT_DEFAULT = "decimal"
_fmt_lbl_default = list(OUTPUT_FORMATS.keys())[0]

# ── グローバルチェック（サイドバー設定の不整合を常時表示）──────
_datum_warn = check_datum_zone_mismatch(DATUM, Z)
if _datum_warn:
    st.warning(_datum_warn)

tab1, tab2 = st.tabs(["📍 単点変換", "📋 CSV 一括変換"])

# ═══════════════════════════════════════════════════════
# 9. TAB 1: 単点変換（複数点 / 地図ピン / 入力形式選択）
# ═══════════════════════════════════════════════════════

# ── session_state 直接バインド方式のヘルパー ─────────────
# text_input は key= のみ使い value= を渡さない。
# これにより サイドバー変更・地図スタイル変更時も入力値が保持される。
# スワップ時は session_state のキーを直接書き換えるため即時反映される。

def _init_jpc():
    if "pts_jpc" not in st.session_state:
        st.session_state["pts_jpc"] = [{"name":"","x":"","y":"","z":""}]
    for i, pt in enumerate(st.session_state["pts_jpc"]):
        for f,v in [("name",pt["name"]),("x",pt["x"]),("y",pt["y"]),("z",pt["z"])]:
            k = f"jpc_{f}_{i}"
            if k not in st.session_state:
                st.session_state[k] = v

def _read_jpc():
    """ウィジェットキーの現在値を pts_jpc に反映して返す"""
    pts = st.session_state["pts_jpc"]
    for i, pt in enumerate(pts):
        for f in ("name","x","y","z"):
            k = f"jpc_{f}_{i}"
            if k in st.session_state:
                pt[f] = st.session_state[k]
    return pts

def _init_ll():
    if "pts_ll" not in st.session_state:
        st.session_state["pts_ll"] = [{"name":"","lat":"","lon":"","h":""}]
    for i, pt in enumerate(st.session_state["pts_ll"]):
        for f,v in [("name",pt["name"]),("lat",pt["lat"]),("lon",pt["lon"]),("h",pt["h"])]:
            k = f"ll_{f}_{i}"
            if k not in st.session_state:
                st.session_state[k] = v

def _read_ll():
    pts = st.session_state["pts_ll"]
    for i, pt in enumerate(pts):
        for f in ("name","lat","lon","h"):
            k = f"ll_{f}_{i}"
            if k in st.session_state:
                pt[f] = st.session_state[k]
    return pts

def _clear_keys(prefix):
    for k in [k for k in st.session_state if k.startswith(prefix)]:
        del st.session_state[k]

def _swap_jpc():
    """X と Y を全点入替（session_state キーを直接書き換え）"""
    pts = _read_jpc()
    for i, pt in enumerate(pts):
        old_x, old_y = pt["x"], pt["y"]
        st.session_state[f"jpc_x_{i}"] = old_y
        st.session_state[f"jpc_y_{i}"] = old_x
        pt["x"], pt["y"] = old_y, old_x

def _swap_ll():
    """緯度と経度を全点入替"""
    pts = _read_ll()
    for i, pt in enumerate(pts):
        old_lat, old_lon = pt["lat"], pt["lon"]
        st.session_state[f"ll_lat_{i}"] = old_lon
        st.session_state[f"ll_lon_{i}"] = old_lat
        pt["lat"], pt["lon"] = old_lon, old_lat

# ─────────────────────────────────────────────────────────
with tab1:
    dir1 = st.radio("変換方向",
                    ["緯度経度 形式変換", "平面直角 → 緯度経度", "緯度経度 → 平面直角", "📐 ローカライゼーション計算"],
                    horizontal=True, key="d1")
    st.markdown("---")

    # ══════════════════════════════
    # 平面直角 → 緯度経度
    # ══════════════════════════════
    if dir1 == "平面直角 → 緯度経度":
        _init_jpc()

        col_add, col_clr, col_swap, _ = st.columns([1,1,1.3,4])
        with col_add:
            if st.button("＋ 点を追加", key="add_jpc"):
                _read_jpc()
                n = len(st.session_state["pts_jpc"]) + 1
                st.session_state["pts_jpc"].append({"name":"","x":"","y":"","z":""})
                st.rerun()
        with col_clr:
            if st.button("🗑 全クリア", key="clr_jpc"):
                st.session_state["pts_jpc"] = [{"name":"","x":"","y":"","z":""}]
                _clear_keys("jpc_")
                st.rerun()
        with col_swap:
            if st.button("⇄ X↔Y 入替", key="swap_jpc", help="全点のXとYを入れ替えます"):
                _swap_jpc()
                st.rerun()

        st.markdown("<div class='sec-label'>入力（平面直角座標）</div>", unsafe_allow_html=True)

        pts = st.session_state["pts_jpc"]
        del_idx = None
        for i, pt in enumerate(pts):
            c0, c1, c2, c3, c4, c5 = st.columns([0.6,1.4,2,2,2,0.45])
            with c0:
                toppad = "32" if i==0 else "8"
                st.markdown(
                    f"<div style='padding-top:{toppad}px;font-size:12px;font-weight:700;color:#64748b'>#{i+1}</div>",
                    unsafe_allow_html=True)
            with c1:
                st.text_input("点名", placeholder=f"pt{i+1}", key=f"jpc_name_{i}",
                              label_visibility="visible" if i==0 else "collapsed")
            with c2:
                st.text_input("X 北が正 (m)" if i==0 else "X (m)",
                              placeholder="-42090.367", key=f"jpc_x_{i}",
                              label_visibility="visible" if i==0 else "collapsed")
            with c3:
                st.text_input("Y 東が正 (m)" if i==0 else "Y (m)",
                              placeholder="-23809.574", key=f"jpc_y_{i}",
                              label_visibility="visible" if i==0 else "collapsed")
            with c4:
                st.text_input("Z 標高 (m)" if i==0 else "Z (m)",
                              placeholder="52.340", key=f"jpc_z_{i}",
                              label_visibility="visible" if i==0 else "collapsed")
            with c5:
                if i == 0:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("✕", key=f"del_jpc_{i}", disabled=len(pts)==1):
                    del_idx = i
        if del_idx is not None:
            _read_jpc()
            new_pts = [p for j,p in enumerate(st.session_state["pts_jpc"]) if j != del_idx]
            _clear_keys("jpc_")
            st.session_state["pts_jpc"] = new_pts
            st.rerun()

        # 最新値を読み取り
        pts = _read_jpc()

        # 出力フォーマット選択（入力欄のすぐ下）
        st.markdown("<div class='sec-label'>出力フォーマット（緯度・経度）</div>", unsafe_allow_html=True)
        out_fmt_jpc_lbl = st.selectbox(
            "出力フォーマットJPC",
            list(OUTPUT_FORMATS.keys()),
            index=3, label_visibility="collapsed", key="out_fmt_jpc"
        )
        FMT_JPC = OUTPUT_FORMATS[out_fmt_jpc_lbl]

        # ── 入力中のリアルタイムチェック ──
        for _pt in pts:
            if _pt["x"].strip() and _pt["y"].strip():
                render_zone_suggestion_jpc(_pt["x"], _pt["y"], Z)
                break  # 最初の有効点のみ表示

        st.markdown("---")
        has_input = any(pt["x"].strip() and pt["y"].strip() for pt in pts)

        # ジオイド未設定チェック
        _has_z = any(pt["z"].strip() for pt in pts)
        _geoid_warn = check_geoid_warning(GEOID_KEY, _has_z)
        if _geoid_warn and has_input and _has_z:
            st.warning(_geoid_warn)

        if has_input:
            st.markdown("<div class='sec-label'>変換結果</div>", unsafe_allow_html=True)
            map_rows, csv_rows = [], []
            # CSV統一フォーマット（ヘッダーなし）

            for i, pt in enumerate(pts):
                if not (pt["x"].strip() and pt["y"].strip()):
                    continue
                try:
                    Xv = float(pt["x"]); Yv = float(pt["y"])
                    Zv = float(pt["z"]) if pt["z"].strip() else None
                    res = jpc_to_latlon(Xv, Yv, Z)
                    if res is None:
                        st.error(f"[{pt['name']}] 系番号 {Z} が無効"); continue
                    lat_dd, lon_dd = res

                    N = None; ellH = None; geoid_status = ""
                    if GEOID_KEY != "NONE":
                        N = fetch_geoid(lat_dd, lon_dd, GEOID_KEY)
                        if N is not None:
                            geoid_status = f"<span class='geoid-ok'>N={N:.4f} m ({GEOID_KEY})</span>"
                            if Zv is not None: ellH = Zv + N
                        else:
                            geoid_status = "<span class='geoid-ng'>⚠ ジオイド高取得失敗</span>"
                    else:
                        geoid_status = "<span style='color:#94a3b8;font-size:11px'>補正なし</span>"
                        if Zv is not None: ellH = Zv

                    pin_color = PIN_COLORS[i % len(PIN_COLORS)]
                    color_css = f"rgb({pin_color[0]},{pin_color[1]},{pin_color[2]})"
                    st.markdown(
                        f"<div class='pt-card'><div class='pt-title'>"
                        f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
                        f"background:{color_css};margin-right:6px'></span>"
                        f"{pt['name']} &nbsp; {geoid_status}</div>",
                        unsafe_allow_html=True)

                    rc1,rc2,rc3 = st.columns(3)
                    with rc1:
                        st.markdown(
                            f"<div class='rc'><div class='rc-lbl' style='color:#3b82f6'>緯度 LAT</div>"
                            f"<div class='rc-val'>{format_angle(lat_dd,FMT_JPC)}</div>"
                            f"<div class='rc-sub'>{fmt_decimal(lat_dd)} deg</div></div>",
                            unsafe_allow_html=True)
                    with rc2:
                        st.markdown(
                            f"<div class='rc'><div class='rc-lbl' style='color:#10b981'>経度 LON</div>"
                            f"<div class='rc-val'>{format_angle(lon_dd,FMT_JPC)}</div>"
                            f"<div class='rc-sub'>{fmt_decimal(lon_dd)} deg</div></div>",
                            unsafe_allow_html=True)
                    with rc3:
                        hs  = f"{ellH:.4f} m" if ellH is not None else "---"
                        sub = f"Z={Zv:.4f}+N={N:.4f}" if (ellH is not None and N is not None and Zv is not None) else ""
                        st.markdown(
                            f"<div class='rc'><div class='rc-lbl' style='color:#8b5cf6'>楕円体高 h (m)</div>"
                            f"<div class='rc-val'>{hs}</div>"
                            f"<div class='rc-sub'>{sub}</div></div>",
                            unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                    tip = f"Z={Zv:.4f}m / N={N:.4f}m / h={ellH:.4f}m" if ellH is not None else fmt_decimal(lat_dd)
                    map_rows.append({"name":pt["name"],"lat":lat_dd,"lon":lon_dd,"tooltip":tip})
                    # A=点名, B=緯度, C=経度, D=楕円体高
                    csv_rows.append(csv_row(
                        pt["name"],
                        format_angle(lat_dd, FMT_JPC),
                        format_angle(lon_dd, FMT_JPC),
                        f"{ellH:.4f}" if ellH is not None else "",
                    ))
                except (ValueError, Exception) as ex:
                    st.error(f"[{pt['name']}] エラー: {ex}")

            if map_rows:
                st.markdown("#### 📍 地図")
                render_map(map_rows, map_style_lbl, zoom=13)
                csv_out = "\ufeff" + "\n".join(csv_rows)
                _fn_jpc_out = _csv_filename_ui("csv_fn_jpc", "例: 現場名_JPC変換")
                st.download_button("📥 全点 CSV ダウンロード", csv_out, _fn_jpc_out or "dummy.csv",
                    "text/csv; charset=utf-8-sig", disabled=(_fn_jpc_out is None))
        else:
            st.info("X・Y 座標を入力してください。")

    # ══════════════════════════════
    # 緯度経度 → 平面直角
    # ══════════════════════════════
    elif dir1 == "緯度経度 → 平面直角":
        _init_ll()

        st.markdown("<div class='sec-label'>入力フォーマット（緯度・経度）</div>", unsafe_allow_html=True)
        in_fmt_lbl = st.selectbox(
            "入力フォーマット",
            list(OUTPUT_FORMATS.keys()),
            index=0,
            label_visibility="collapsed",
            key="in_fmt_ll",
        )
        IN_FMT = OUTPUT_FORMATS[in_fmt_lbl]
        ph_lat = FORMAT_PLACEHOLDER[IN_FMT]
        ph_lon = FORMAT_PLACEHOLDER[IN_FMT].replace("35","139").replace("40","47")

        col_add2, col_clr2, col_swap2, _ = st.columns([1,1,1.4,4])
        with col_add2:
            if st.button("＋ 点を追加", key="add_ll"):
                _read_ll()
                n = len(st.session_state["pts_ll"]) + 1
                st.session_state["pts_ll"].append({"name":"","lat":"","lon":"","h":""})
                st.rerun()
        with col_clr2:
            if st.button("🗑 全クリア", key="clr_ll"):
                st.session_state["pts_ll"] = [{"name":"","lat":"","lon":"","h":""}]
                _clear_keys("ll_")
                st.rerun()
        with col_swap2:
            if st.button("⇄ 緯↔経 入替", key="swap_ll", help="全点の緯度と経度を入れ替えます"):
                _swap_ll()
                st.rerun()

        st.markdown(f"<div class='sec-label'>入力（{in_fmt_lbl}）</div>", unsafe_allow_html=True)

        pts2 = st.session_state["pts_ll"]
        del_idx2 = None
        for i, pt in enumerate(pts2):
            c0, c1, c2, c3, c4, c5 = st.columns([0.6,1.4,2.3,2.3,1.8,0.45])
            with c0:
                toppad = "32" if i==0 else "8"
                st.markdown(
                    f"<div style='padding-top:{toppad}px;font-size:12px;font-weight:700;color:#64748b'>#{i+1}</div>",
                    unsafe_allow_html=True)
            with c1:
                st.text_input("点名", placeholder=f"pt{i+1}", key=f"ll_name_{i}",
                              label_visibility="visible" if i==0 else "collapsed")
            with c2:
                st.text_input(
                    f"緯度（{in_fmt_lbl.split('（')[0]}）" if i==0 else "緯度",
                    placeholder=ph_lat, key=f"ll_lat_{i}",
                    label_visibility="visible" if i==0 else "collapsed")
            with c3:
                st.text_input(
                    f"経度（{in_fmt_lbl.split('（')[0]}）" if i==0 else "経度",
                    placeholder=ph_lon, key=f"ll_lon_{i}",
                    label_visibility="visible" if i==0 else "collapsed")
            with c4:
                st.text_input("楕円体高 h (m)" if i==0 else "h(m)",
                              placeholder="89.555", key=f"ll_h_{i}",
                              label_visibility="visible" if i==0 else "collapsed")
            with c5:
                if i == 0:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("✕", key=f"del_ll_{i}", disabled=len(pts2)==1):
                    del_idx2 = i
        if del_idx2 is not None:
            _read_ll()
            new_pts2 = [p for j,p in enumerate(st.session_state["pts_ll"]) if j != del_idx2]
            _clear_keys("ll_name_"); _clear_keys("ll_lat_"); _clear_keys("ll_lon_"); _clear_keys("ll_h_")
            st.session_state["pts_ll"] = new_pts2
            st.rerun()

        pts2 = _read_ll()

        # ── 入力中のリアルタイムチェック ──
        for _pt in pts2:
            if _pt["lat"].strip() and _pt["lon"].strip():
                render_zone_suggestion_ll(_pt["lat"], _pt["lon"], IN_FMT, Z)
                break  # 最初の有効点のみ表示

        st.markdown("---")
        has_input2 = any(pt["lat"].strip() and pt["lon"].strip() for pt in pts2)

        # ジオイド未設定チェック
        _has_h2 = any(pt["h"].strip() for pt in pts2)
        _geoid_warn2 = check_geoid_warning(GEOID_KEY, _has_h2)
        if _geoid_warn2 and has_input2 and _has_h2:
            st.warning(_geoid_warn2)

        if has_input2:
            st.markdown("<div class='sec-label'>変換結果</div>", unsafe_allow_html=True)
            map_rows2, csv_rows2 = [], []
            # CSV統一フォーマット（ヘッダーなし）

            for i, pt in enumerate(pts2):
                if not (pt["lat"].strip() and pt["lon"].strip()):
                    continue
                try:
                    lv  = parse_angle(pt["lat"], IN_FMT)
                    lov = parse_angle(pt["lon"], IN_FMT)
                    hv  = float(pt["h"]) if pt["h"].strip() else 0.0
                    res = latlon_to_jpc(lv, lov, Z)
                    if res is None:
                        st.error(f"[{pt['name']}] 系番号 {Z} が無効"); continue
                    Xr, Yr = res

                    # 楕円体高 → 標高: ジオイド高を差し引く
                    N_ll = None; elev_ll = None; geoid_status_ll = ""
                    if pt["h"].strip():
                        if GEOID_KEY != "NONE":
                            N_ll = fetch_geoid(lv, lov, GEOID_KEY)
                            if N_ll is not None:
                                elev_ll = hv - N_ll
                                geoid_status_ll = f"<span class='geoid-ok'>N={N_ll:.4f} m</span>"
                            else:
                                geoid_status_ll = "<span class='geoid-ng'>⚠ ジオイド高取得失敗</span>"
                        else:
                            geoid_status_ll = "<span style='color:#94a3b8;font-size:11px'>補正なし</span>"
                            elev_ll = hv

                    pin_color = PIN_COLORS[i % len(PIN_COLORS)]
                    color_css = f"rgb({pin_color[0]},{pin_color[1]},{pin_color[2]})"
                    st.markdown(
                        f"<div class='pt-card'><div class='pt-title'>"
                        f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
                        f"background:{color_css};margin-right:6px'></span>"
                        f"{pt['name']} &nbsp; {geoid_status_ll}</div>",
                        unsafe_allow_html=True)

                    rc1,rc2,rc3,rc4 = st.columns(4)
                    with rc1:
                        st.markdown(
                            f"<div class='rc'><div class='rc-lbl' style='color:#3b82f6'>X 北が正 (m)</div>"
                            f"<div class='rc-val'>{Xr:,.4f}</div></div>",
                            unsafe_allow_html=True)
                    with rc2:
                        st.markdown(
                            f"<div class='rc'><div class='rc-lbl' style='color:#10b981'>Y 東が正 (m)</div>"
                            f"<div class='rc-val'>{Yr:,.4f}</div></div>",
                            unsafe_allow_html=True)
                    with rc3:
                        if elev_ll is not None:
                            z_str = f"{elev_ll:.4f} m"
                            z_sub = f"h={hv:.4f} - N={N_ll:.4f}" if N_ll is not None else "補正なし"
                        elif pt["h"].strip():
                            z_str = "---"; z_sub = "ジオイド高取得失敗"
                        else:
                            z_str = "---"; z_sub = "楕円体高未入力"
                        st.markdown(
                            f"<div class='rc'><div class='rc-lbl' style='color:#f59e0b'>Z 標高 (m)</div>"
                            f"<div class='rc-val'>{z_str}</div>"
                            f"<div class='rc-sub'>{z_sub}</div></div>",
                            unsafe_allow_html=True)
                    with rc4:
                        st.markdown(
                            f"<div class='rc'><div class='rc-lbl' style='color:#8b5cf6'>座標系</div>"
                            f"<div class='rc-val'>第 {Z} 系</div>"
                            f"<div class='rc-sub'>GRS80 / m0=0.9999</div></div>",
                            unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    map_rows2.append({"name":pt["name"],"lat":lv,"lon":lov,
                                      "tooltip":f"X={Xr:.4f} / Y={Yr:.4f}" + (f" / Z={elev_ll:.4f}m" if elev_ll else "")})
                    # A=点名, B=X, C=Y, D=Z標高
                    csv_rows2.append(csv_row(
                        pt["name"],
                        f"{Xr:.4f}",
                        f"{Yr:.4f}",
                        f"{elev_ll:.4f}" if elev_ll is not None else "",
                    ))

                except (ValueError, Exception) as ex:
                    st.error(f"[{pt['name']}] エラー: {ex}")

            if map_rows2:
                st.markdown("#### 📍 地図")
                render_map(map_rows2, map_style_lbl, zoom=13)
                csv_out2 = "\ufeff" + "\n".join(csv_rows2)
                _fn_ll_out = _csv_filename_ui("csv_fn_ll", "例: 現場名_LL変換")
                st.download_button("📥 全点 CSV ダウンロード", csv_out2, _fn_ll_out or "dummy.csv",
                    "text/csv; charset=utf-8-sig", disabled=(_fn_ll_out is None))
        else:
            pass

    # ══════════════════════════════
    # 緯度経度 形式変換
    # ══════════════════════════════
    elif dir1 == "緯度経度 形式変換":
        # session_state 初期化
        if "pts_cvt" not in st.session_state:
            st.session_state["pts_cvt"] = [{"name":"","lat":"","lon":"","h":""}]
        for i, pt in enumerate(st.session_state["pts_cvt"]):
            for f,v in [("name",pt.get("name","")),("lat",pt.get("lat","")),
                        ("lon",pt.get("lon","")),("h",pt.get("h",""))]:
                k = f"cvt_{f}_{i}"
                if k not in st.session_state:
                    st.session_state[k] = v

        # 入力フォーマット
        st.markdown("<div class='sec-label'>入力フォーマット</div>", unsafe_allow_html=True)
        in_fmt_cvt_lbl = st.selectbox(
            "入力フォーマット（形式変換）",
            list(OUTPUT_FORMATS.keys()),
            index=0, label_visibility="collapsed", key="in_fmt_cvt"
        )
        IN_FMT_CVT = OUTPUT_FORMATS[in_fmt_cvt_lbl]

        # OUT_FMT_CVT は入力欄より後で定義するため、ここでは key だけ事前参照
        # （session_state に既存値があれば使い、なければデフォルト index=3）
        _cvt_fmt_keys = list(OUTPUT_FORMATS.keys())
        _cvt_fmt_default = _cvt_fmt_keys[3]
        out_fmt_cvt_lbl = st.session_state.get("out_fmt_cvt", _cvt_fmt_default)
        if out_fmt_cvt_lbl not in OUTPUT_FORMATS:
            out_fmt_cvt_lbl = _cvt_fmt_default
        OUT_FMT_CVT = OUTPUT_FORMATS[out_fmt_cvt_lbl]

        ph_cvt_lat = FORMAT_PLACEHOLDER[IN_FMT_CVT]
        ph_cvt_lon = FORMAT_PLACEHOLDER[IN_FMT_CVT].replace("35","139").replace("40","47")

        col_add_c, col_clr_c, col_swap_c, _ = st.columns([1,1,1.4,4])
        with col_add_c:
            if st.button("＋ 点を追加", key="add_cvt"):
                for i, pt in enumerate(st.session_state["pts_cvt"]):
                    for f in ("name","lat","lon","h"):
                        k = f"cvt_{f}_{i}"
                        if k in st.session_state: pt[f] = st.session_state[k]
                n = len(st.session_state["pts_cvt"]) + 1
                st.session_state["pts_cvt"].append({"name":"","lat":"","lon":"","h":""})
                st.rerun()
        with col_clr_c:
            if st.button("🗑 全クリア", key="clr_cvt"):
                st.session_state["pts_cvt"] = [{"name":"","lat":"","lon":""}]
                for k in [k for k in st.session_state if k.startswith("cvt_")]:
                    del st.session_state[k]
                st.rerun()
        with col_swap_c:
            if st.button("⇄ 緯↔経 入替", key="swap_cvt"):
                for i, pt in enumerate(st.session_state["pts_cvt"]):
                    for f in ("name","lat","lon","h"):
                        k = f"cvt_{f}_{i}"
                        if k in st.session_state: pt[f] = st.session_state[k]
                for i, pt in enumerate(st.session_state["pts_cvt"]):
                    old_lat, old_lon = pt["lat"], pt["lon"]
                    st.session_state[f"cvt_lat_{i}"] = old_lon
                    st.session_state[f"cvt_lon_{i}"] = old_lat
                    pt["lat"], pt["lon"] = old_lon, old_lat
                st.rerun()

        st.markdown(f"<div class='sec-label'>入力（{in_fmt_cvt_lbl}）</div>", unsafe_allow_html=True)

        pts_cvt = st.session_state["pts_cvt"]
        del_idx_c = None
        for i, pt in enumerate(pts_cvt):
            c0, c1, c2, c3, c4, c5 = st.columns([0.6,1.4,2,2,1.6,0.45])
            with c0:
                toppad = "32" if i==0 else "8"
                st.markdown(
                    f"<div style='padding-top:{toppad}px;font-size:12px;font-weight:700;color:#64748b'>#{i+1}</div>",
                    unsafe_allow_html=True)
            with c1:
                st.text_input("点名", placeholder=f"pt{i+1}", key=f"cvt_name_{i}",
                              label_visibility="visible" if i==0 else "collapsed")
            with c2:
                st.text_input(
                    f"緯度（{in_fmt_cvt_lbl.split('（')[0]}）" if i==0 else "緯度",
                    placeholder=ph_cvt_lat, key=f"cvt_lat_{i}",
                    label_visibility="visible" if i==0 else "collapsed")
            with c3:
                st.text_input(
                    f"経度（{in_fmt_cvt_lbl.split('（')[0]}）" if i==0 else "経度",
                    placeholder=ph_cvt_lon, key=f"cvt_lon_{i}",
                    label_visibility="visible" if i==0 else "collapsed")
            with c4:
                st.text_input("楕円体高 h (m)" if i==0 else "h(m)",
                              placeholder="89.555", key=f"cvt_h_{i}",
                              label_visibility="visible" if i==0 else "collapsed")
            with c5:
                if i == 0:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("✕", key=f"del_cvt_{i}", disabled=len(pts_cvt)==1):
                    del_idx_c = i
        if del_idx_c is not None:
            for i, pt in enumerate(st.session_state["pts_cvt"]):
                for f in ("name","lat","lon","h"):
                    k = f"cvt_{f}_{i}"
                    if k in st.session_state: pt[f] = st.session_state[k]
            new_pts_c = [p for j,p in enumerate(st.session_state["pts_cvt"]) if j != del_idx_c]
            for k in [k for k in st.session_state if k.startswith("cvt_name_") or k.startswith("cvt_lat_") or k.startswith("cvt_lon_") or k.startswith("cvt_h_")]:
                del st.session_state[k]
            st.session_state["pts_cvt"] = new_pts_c
            st.rerun()

        # 現在値を同期
        for i, pt in enumerate(pts_cvt):
            for f in ("name","lat","lon","h"):
                k = f"cvt_{f}_{i}"
                if k in st.session_state: pt[f] = st.session_state[k]

        # 出力フォーマット選択（入力欄のすぐ下）
        st.markdown("<div class='sec-label'>出力フォーマット</div>", unsafe_allow_html=True)
        out_fmt_cvt_lbl = st.selectbox(
            "出力フォーマット（形式変換）",
            list(OUTPUT_FORMATS.keys()),
            index=_cvt_fmt_keys.index(out_fmt_cvt_lbl),
            label_visibility="collapsed", key="out_fmt_cvt"
        )
        OUT_FMT_CVT = OUTPUT_FORMATS[out_fmt_cvt_lbl]

        st.markdown("---")
        has_cvt = any(pt["lat"].strip() and pt["lon"].strip() for pt in pts_cvt)

        if has_cvt:
            st.markdown(
                f"<div class='sec-label'>変換結果 &nbsp;"
                f"<span style='font-size:10px;color:#64748b;font-weight:400'>"
                f"{in_fmt_cvt_lbl} &rarr; {out_fmt_cvt_lbl}</span></div>",
                unsafe_allow_html=True)

            map_rowsc, csv_rowsc = [], []
            # CSV統一フォーマット（ヘッダーなし）

            for i, pt in enumerate(pts_cvt):
                if not (pt["lat"].strip() and pt["lon"].strip()):
                    continue
                try:
                    lat_dd = parse_angle(pt["lat"], IN_FMT_CVT)
                    lon_dd = parse_angle(pt["lon"], IN_FMT_CVT)
                    lat_out = format_angle(lat_dd, OUT_FMT_CVT)
                    lon_out = format_angle(lon_dd, OUT_FMT_CVT)

                    pin_color = PIN_COLORS[i % len(PIN_COLORS)]
                    color_css = f"rgb({pin_color[0]},{pin_color[1]},{pin_color[2]})"
                    st.markdown(
                        f"<div class='pt-card'><div class='pt-title'>"
                        f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
                        f"background:{color_css};margin-right:6px'></span>"
                        f"{pt['name']}</div>",
                        unsafe_allow_html=True)

                    # pt["h"] は現在値同期済み → session_state再取得不要
                    h_cvt_val = float(pt["h"]) if pt["h"].strip() else None

                    rc1,rc2,rc3 = st.columns(3)
                    with rc1:
                        st.markdown(
                            f"<div class='rc'><div class='rc-lbl' style='color:#3b82f6'>緯度 LAT</div>"
                            f"<div class='rc-val'>{lat_out}</div>"
                            f"<div class='rc-sub'>{fmt_decimal(lat_dd)} deg</div></div>",
                            unsafe_allow_html=True)
                    with rc2:
                        st.markdown(
                            f"<div class='rc'><div class='rc-lbl' style='color:#10b981'>経度 LON</div>"
                            f"<div class='rc-val'>{lon_out}</div>"
                            f"<div class='rc-sub'>{fmt_decimal(lon_dd)} deg</div></div>",
                            unsafe_allow_html=True)
                    with rc3:
                        hc_str = f"{h_cvt_val:.4f} m" if h_cvt_val is not None else "---"
                        hc_sub = "楕円体高（そのまま）" if h_cvt_val is not None else "未入力"
                        st.markdown(
                            f"<div class='rc'><div class='rc-lbl' style='color:#8b5cf6'>楕円体高 h (m)</div>"
                            f"<div class='rc-val'>{hc_str}</div>"
                            f"<div class='rc-sub'>{hc_sub}</div></div>",
                            unsafe_allow_html=True)

                    st.markdown("</div>", unsafe_allow_html=True)
                    tip_h = f" / h={h_cvt_val:.4f}m" if h_cvt_val is not None else ""
                    map_rowsc.append({"name":pt["name"],"lat":lat_dd,"lon":lon_dd,
                                      "tooltip":f"{lat_out} / {lon_out}{tip_h}"})
                    # A=点名, B=緯度, C=経度, D=楕円体高
                    csv_rowsc.append(csv_row(
                        pt["name"],
                        lat_out,
                        lon_out,
                        f"{h_cvt_val:.4f}" if h_cvt_val is not None else "",
                    ))
                except (ValueError, Exception) as ex:
                    st.error(f"[{pt['name']}] エラー: {ex}")

            if map_rowsc:
                st.markdown("#### 📍 地図")
                render_map(map_rowsc, map_style_lbl, zoom=13)
                csv_outc = "\ufeff" + "\n".join(csv_rowsc)
                _fn_cvt_out = _csv_filename_ui("csv_fn_cvt", "例: 現場名_形式変換")
                st.download_button("📥 全点 CSV ダウンロード", csv_outc, _fn_cvt_out or "dummy.csv",
                    "text/csv; charset=utf-8-sig", disabled=(_fn_cvt_out is None))
        else:
            pass

    # ══════════════════════════════
    # ローカライゼーション計算
    # ══════════════════════════════
    elif dir1 == "📐 ローカライゼーション計算":
        import math as _math
        import numpy as _np

        st.markdown("""
<div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:10px 16px;margin-bottom:10px;font-size:13px;color:#1e40af'>
📌 <b>入力フォーマット（CSVヘッダーなし）：</b>　点名, 測量X(m), 測量Y(m), 測量Z(m), GNSS緯度, GNSS経度, 楕円体高(m)<br>
2点以上でScale・Rotation・Tx・Ty・ΔZを算出。3点以上で残差も表示します。
</div>""", unsafe_allow_html=True)

        # ── セッション初期化 ──
        _lc_empty = {"name":"","sx":"","sy":"","sz":"","lat":"","lon":"","h":""}
        if "pts_local" not in st.session_state:
            st.session_state["pts_local"] = [dict(_lc_empty), dict(_lc_empty)]
        if "lc_csv_ver" not in st.session_state:
            st.session_state["lc_csv_ver"] = 0
        if "lc_csv_msg" not in st.session_state:
            st.session_state["lc_csv_msg"] = ""

        # ── CSV一括インポート ──
        _lc_up = st.file_uploader(
            "📂 CSVインポート（点名, 測量X, 測量Y, 測量Z, GNSS緯度, GNSS経度, 楕円体高）",
            type=["csv","txt"],
            key=f"lc_csv_up_{st.session_state['lc_csv_ver']}"
        )
        if _lc_up is not None:
            try:
                _lc_text = _lc_up.read().decode("utf-8-sig")
                _lc_rows = []
                for _lc_line in _lc_text.splitlines():
                    _lc_line = _lc_line.strip()
                    if not _lc_line:
                        continue
                    _lc_cols = [c.strip() for c in _lc_line.split(",")]
                    if len(_lc_cols) >= 7:
                        _lc_rows.append({
                            "name": _lc_cols[0],
                            "sx":   _lc_cols[1],
                            "sy":   _lc_cols[2],
                            "sz":   _lc_cols[3],
                            "lat":  _lc_cols[4],
                            "lon":  _lc_cols[5],
                            "h":    _lc_cols[6],
                        })
                if _lc_rows:
                    st.session_state["pts_local"] = _lc_rows
                    st.session_state["lc_csv_ver"] += 1
                    st.session_state["lc_csv_msg"] = f"✅ {len(_lc_rows)} 点を読み込みました"
                    st.rerun()
                else:
                    st.error("❌ 読み込める行がありません（7列以上必要）")
            except Exception as _lc_ex:
                st.error(f"❌ CSV読み込みエラー: {_lc_ex}")

        # 読み込み完了メッセージ（rerun後に表示）
        if st.session_state.get("lc_csv_msg"):
            st.success(st.session_state["lc_csv_msg"])
            st.session_state["lc_csv_msg"] = ""

        pts_local = st.session_state["pts_local"]
        _lc_v = st.session_state["lc_csv_ver"]

        col_add_l, col_clr_l, _ = st.columns([1, 1, 6])
        with col_add_l:
            if st.button("＋ 点を追加", key="add_local"):
                pts_local.append(dict(_lc_empty))
                st.rerun()
        with col_clr_l:
            if st.button("🗑 全クリア", key="clr_local"):
                st.session_state["pts_local"] = [dict(_lc_empty), dict(_lc_empty)]
                st.session_state["lc_csv_ver"] += 1
                st.rerun()

        # ── ヘッダー行 ──
        _lc_h1, _lc_h2, _lc_h3, _lc_h4, _lc_h5, _lc_h6, _lc_h7, _lc_hd = st.columns([1.5, 2, 2, 1.5, 2, 2, 1.5, 0.4])
        for _col, _lbl in zip(
            [_lc_h1,_lc_h2,_lc_h3,_lc_h4,_lc_h5,_lc_h6,_lc_h7],
            ["点名","測量X(m)","測量Y(m)","測量Z(m)","GNSS緯度","GNSS経度","楕円体高(m)"]
        ):
            _col.markdown(f"<div style='font-size:11px;font-weight:700;color:#374151;padding-bottom:2px'>{_lbl}</div>", unsafe_allow_html=True)

        for i, pt in enumerate(pts_local):
            ca, cb, cc, cd, ce, cf, cg, cdel = st.columns([1.5, 2, 2, 1.5, 2, 2, 1.5, 0.4])
            with ca: pt["name"] = st.text_input("点名",      value=pt.get("name",""), key=f"lc_name_{i}_{_lc_v}", placeholder="BM-1",          label_visibility="collapsed")
            with cb: pt["sx"]   = st.text_input("測量X",     value=pt.get("sx",""),   key=f"lc_sx_{i}_{_lc_v}",   placeholder="151940.92",      label_visibility="collapsed")
            with cc: pt["sy"]   = st.text_input("測量Y",     value=pt.get("sy",""),   key=f"lc_sy_{i}_{_lc_v}",   placeholder="44023.112",      label_visibility="collapsed")
            with cd: pt["sz"]   = st.text_input("測量Z",     value=pt.get("sz",""),   key=f"lc_sz_{i}_{_lc_v}",   placeholder="249.837",        label_visibility="collapsed")
            with ce: pt["lat"]  = st.text_input("GNSS緯度",  value=pt.get("lat",""),  key=f"lc_lat_{i}_{_lc_v}",  placeholder="37.36827786",    label_visibility="collapsed")
            with cf: pt["lon"]  = st.text_input("GNSS経度",  value=pt.get("lon",""),  key=f"lc_lon_{i}_{_lc_v}",  placeholder="140.33036534",   label_visibility="collapsed")
            with cg: pt["h"]    = st.text_input("楕円体高",  value=pt.get("h",""),    key=f"lc_h_{i}_{_lc_v}",    placeholder="292.8286",       label_visibility="collapsed")
            with cdel:
                if len(pts_local) > 2 and st.button("✕", key=f"lc_del_{i}_{_lc_v}"):
                    pts_local.pop(i); st.rerun()
            st.session_state["pts_local"][i] = pt

        st.markdown("---")

        if st.button("🔢 ローカライゼーション計算", key="calc_local", type="primary", use_container_width=True):
            valid_pts = []
            errors = []

            # キャッシュ付き関数を避けて純粋計算関数を直接定義
            def _ll2jpc(lat_deg, lon_deg, zone):
                if zone not in JPC_ORIGINS: return None
                la0, lo0 = JPC_ORIGINS[zone]
                phi = lat_deg * DEG; lam = lon_deg * DEG
                phi0 = la0 * DEG;    lam0 = lo0 * DEG
                sinP = _math.sin(phi)
                psi  = _math.atanh(sinP) - _e * _math.atanh(_e * sinP)
                dl   = lam - lam0
                xi_  = _math.atan2(_math.sinh(psi), _math.cos(dl))
                eta_ = _math.atanh(_math.sin(dl) / _math.cosh(psi))
                xi   = xi_  + sum(_alpha[j]*_math.sin(2*j*xi_) *_math.cosh(2*j*eta_) for j in range(1,5))
                eta  = eta_ + sum(_alpha[j]*_math.cos(2*j*xi_) *_math.sinh(2*j*eta_) for j in range(1,5))
                return _m0*_A*xi - _S(phi0), _m0*_A*eta

            def _parse_ll(s):
                """文字列→十進角度 (float)"""
                s = s.strip()
                # 度分秒記号
                if any(c in s for c in ("°","′","″")):
                    import re as _re
                    m = _re.match(r'(-?\d+)[°]\s*(\d+)[′]\s*([\d.]+)', s)
                    if m:
                        sg = -1 if float(m.group(1)) < 0 else 1
                        return sg*(abs(float(m.group(1)))+float(m.group(2))/60+float(m.group(3))/3600)
                # N/S prefix
                if s[0] in 'NnSs':
                    import re as _re
                    m = _re.match(r'([NSns])\s*(\d+)[°]?\s*(\d+)[′]?\s*([\d.]+)', s)
                    if m:
                        v = float(m.group(2))+float(m.group(3))/60+float(m.group(4))/3600
                        return v if s[0] in 'Nn' else -v
                # 純数値
                return float(s.replace("°",""))

            for i, pt in enumerate(pts_local):
                name = pt.get("name","").strip() or f"#{i+1}"
                try:
                    lat_s = pt.get("lat","").strip()
                    lon_s = pt.get("lon","").strip()
                    sx_s  = pt.get("sx","").strip()
                    sy_s  = pt.get("sy","").strip()
                    if not lat_s or not lon_s or not sx_s or not sy_s:
                        errors.append(f"{name}: 必須項目（緯度・経度・測量X・Y）が未入力")
                        continue

                    lat_dd = _parse_ll(lat_s)
                    lon_dd = _parse_ll(lon_s)
                    h_val  = float(pt.get("h","0").strip() or "0")
                    sx_val = float(sx_s)
                    sy_val = float(sy_s)
                    sz_s   = pt.get("sz","").strip()
                    sz_val = float(sz_s) if sz_s else None

                    res = _ll2jpc(lat_dd, lon_dd, Z)
                    if res is None:
                        errors.append(f"{name}: 平面直角変換失敗（系番号を確認）")
                        continue
                    gx, gy = float(res[0]), float(res[1])

                    # ジオイド高（APIキャッシュ版を使用・結果をfloatに強制）
                    try:
                        _N = fetch_geoid(lat_dd, lon_dd, GEOID_KEY)
                        N_val = float(_N) if _N is not None else 0.0
                    except Exception:
                        N_val = 0.0
                    z_ortho = h_val - N_val

                    valid_pts.append({
                        "name": name,
                        "gx": gx, "gy": gy,
                        "sx": sx_val, "sy": sy_val,
                        "gz": z_ortho, "sz": sz_val,
                    })
                except Exception as ex:
                    errors.append(f"{name}: {ex}")

            for e in errors:
                st.error(f"❌ {e}")

            if len(valid_pts) < 2:
                st.warning("⚠️ 有効な基準点が2点以上必要です。")
            else:
                n = len(valid_pts)
                A = _np.zeros((2*n, 4))
                L = _np.zeros(2*n)
                for idx, p in enumerate(valid_pts):
                    A[2*idx,   :] = [ p["gx"], -p["gy"], 1, 0]
                    A[2*idx+1, :] = [ p["gy"],  p["gx"], 0, 1]
                    L[2*idx]      = p["sx"]
                    L[2*idx+1]    = p["sy"]
                params, _, _, _ = _np.linalg.lstsq(A, L, rcond=None)
                a_p, b_p, Tx, Ty = float(params[0]), float(params[1]), float(params[2]), float(params[3])
                scale    = _math.sqrt(a_p**2 + b_p**2)
                rotation = _math.degrees(_math.atan2(b_p, a_p))
                z_pairs  = [p for p in valid_pts if p["sz"] is not None]
                dz_mean  = sum(p["sz"] - p["gz"] for p in z_pairs) / max(len(z_pairs), 1) if z_pairs else 0.0

                residuals = []
                for p in valid_pts:
                    cx = a_p*p["gx"] - b_p*p["gy"] + Tx
                    cy = b_p*p["gx"] + a_p*p["gy"] + Ty
                    dx = (p["sx"] - cx) * 1000
                    dy = (p["sy"] - cy) * 1000
                    dr = _math.sqrt(dx**2 + dy**2)
                    residuals.append({"name": p["name"], "dx_mm": dx, "dy_mm": dy, "dr_mm": dr})
                rmse = _math.sqrt(sum(r["dr_mm"]**2 for r in residuals) / n)

                st.success("✅ 計算完了")
                rc1, rc2, rc3, rc4 = st.columns(4)
                with rc1:
                    st.markdown(f"""<div class='rc'><div class='rc-lbl'>SCALE（スケール）</div>
                        <div class='rc-val'>{scale:.8f}</div>
                        <div class='rc-sub'>ppm: {(scale-1)*1e6:+.3f}</div></div>""", unsafe_allow_html=True)
                with rc2:
                    st.markdown(f"""<div class='rc'><div class='rc-lbl'>ROTATION（回転角）</div>
                        <div class='rc-val'>{rotation:.6f}°</div>
                        <div class='rc-sub'>{rotation*3600:.3f}″</div></div>""", unsafe_allow_html=True)
                with rc3:
                    st.markdown(f"""<div class='rc'><div class='rc-lbl'>Tx（X平行移動）</div>
                        <div class='rc-val'>{Tx:.4f} m</div>
                        <div class='rc-sub'>Ty: {Ty:.4f} m</div></div>""", unsafe_allow_html=True)
                with rc4:
                    st.markdown(f"""<div class='rc'><div class='rc-lbl'>ΔZ（標高オフセット）</div>
                        <div class='rc-val'>{dz_mean:.4f} m</div>
                        <div class='rc-sub'>RMSE: {rmse:.1f} mm</div></div>""", unsafe_allow_html=True)

                st.markdown("#### 📊 各点の残差")
                _df_res = pd.DataFrame([{
                    "点名": r["name"],
                    "ΔX (mm)": f"{r['dx_mm']:+.1f}",
                    "ΔY (mm)": f"{r['dy_mm']:+.1f}",
                    "水平残差 (mm)": f"{r['dr_mm']:.1f}",
                    "判定": "✅ 良好" if r["dr_mm"] < 20 else ("⚠️ 注意" if r["dr_mm"] < 50 else "❌ 要確認"),
                } for r in residuals])
                st.dataframe(_df_res, use_container_width=True, hide_index=True)

                csv_lines_l = [csv_row("パラメータ","値","備考")]
                csv_lines_l += [
                    csv_row("Scale",         f"{scale:.8f}",    f"ppm: {(scale-1)*1e6:+.3f}"),
                    csv_row("Rotation(deg)", f"{rotation:.6f}", f"秒: {rotation*3600:.3f}"),
                    csv_row("Tx(m)",         f"{Tx:.4f}",       "X平行移動"),
                    csv_row("Ty(m)",         f"{Ty:.4f}",       "Y平行移動"),
                    csv_row("dZ(m)",         f"{dz_mean:.4f}",  "標高オフセット"),
                    csv_row("RMSE(mm)",      f"{rmse:.1f}",     f"使用点数: {n}"),
                    csv_row("","",""),
                    csv_row("点名","ΔX(mm)","ΔY(mm)","水平残差(mm)"),
                ] + [csv_row(r["name"], f"{r['dx_mm']:+.1f}", f"{r['dy_mm']:+.1f}", f"{r['dr_mm']:.1f}") for r in residuals]
                _fn_local = _csv_filename_ui("csv_fn_local", "例: 現場名_ローカライゼーション")
                st.download_button("📥 パラメータ CSV ダウンロード",
                    "\ufeff" + "\n".join(csv_lines_l),
                    _fn_local or "dummy.csv",
                    "text/csv; charset=utf-8-sig",
                    disabled=(_fn_local is None))


# ═══════════════════════════════════════════════════════
# 10. TAB 2: CSV 一括変換
# ═══════════════════════════════════════════════════════

with tab2:
    dir2 = st.radio("変換方向",
                    ["平面直角 → 緯度経度", "緯度経度 → 平面直角"],
                    horizontal=True, key="d2")
    st.markdown("---")

    # ── 出力フォーマット選択（アップロード前に必ず設定） ──────────
    c_fmt_lbl, c_fmt_sel = st.columns([2, 4])
    with c_fmt_lbl:
        st.markdown(
            "<div style='padding-top:8px;font-size:12px;font-weight:700;color:#374151'>"
            "📐 出力フォーマット（緯度・経度）</div>",
            unsafe_allow_html=True)
    with c_fmt_sel:
        out_fmt_b2_lbl = st.selectbox(
            "CSV出力フォーマット",
            list(OUTPUT_FORMATS.keys()),
            index=3,
            label_visibility="collapsed",
            key="out_fmt_batch",
        )
    FMT_B2 = OUTPUT_FORMATS[out_fmt_b2_lbl]

    st.markdown("---")

    # ── 変換方向ごとの入力・出力仕様カード ──────────────────────
    _auto_note = (
        "・緯度・経度は <b>5フォーマット自動判別</b>"
        "（十進度 / 度分秒 / 度分秒圧縮 DD.MMSSSSSS / 方位角 N/S / Gons）<br>"
        "・混在入力（行ごとに異なるフォーマット）も対応"
    )
    _col_in_jpc  = "A列=点名　B列=X(m)　C列=Y(m)　D列=Z標高(m)"
    _col_in_ll   = "A列=点名　B列=緯度　C列=経度　D列=楕円体高(m)"
    _col_out     = "A列=点名　B列=X(m)　C列=Y(m)　D列=Z標高(m)　E列=緯度　F列=経度　G列=楕円体高(m)"
    _in_note_jpc = "・D列（Z標高）は省略可&emsp;・5列目以降は無視"
    _in_note_ll  = f"{_auto_note}<br>・D列（楕円体高）は省略可&emsp;・5列目以降は無視"

    col_in  = _col_in_jpc  if dir2 == "平面直角 → 緯度経度" else _col_in_ll
    note_in = _in_note_jpc if dir2 == "平面直角 → 緯度経度" else _in_note_ll

    # 往復変換する場合の注意説明
    _roundtrip_note = ""
    if dir2 == "緯度経度 → 平面直角":
        _roundtrip_note = """
<div style='background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;padding:10px 14px;margin-bottom:10px'>
  <div style='font-size:11px;font-weight:700;color:#92400e;margin-bottom:4px'>⚠️ 往復変換（JPC→LL→JPC）をする場合の注意</div>
  <div style='font-size:10.5px;color:#78350f;line-height:1.8'>
    <b>平面直角 → 緯度経度</b> で出力された緯度経度をそのまま <b>緯度経度 → 平面直角</b> に入力した場合、<br>
    元のXY座標と数mm程度の差が生じることがあります。<br>
    <b>原因：</b>緯度経度の小数桁数が不足すると丸め誤差が積み重なるためです。<br>
    例）緯度経度の小数桁数が不足していると逆変換時に数mm〜十数mmの誤差が生じます<br>
    <b>往復精度を上げるには</b>出力フォーマットを <b>DD.DDDDDDDD°（十進角度）</b> にして<br>
    小数8桁以上の緯度経度をそのまま入力してください。
  </div>
</div>"""

    st.markdown(_roundtrip_note, unsafe_allow_html=True)

    st.markdown(f"""
<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:14px'>
  <div style='background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;padding:12px 14px'>
    <div style='font-size:11px;font-weight:700;color:#0369a1;margin-bottom:5px'>📥 入力CSV（ヘッダーなし・4列固定）</div>
    <code style='font-size:11px;display:block;margin-bottom:5px'>{col_in}</code>
    <div style='font-size:10.5px;color:#64748b;line-height:1.6'>{note_in}</div>
  </div>
  <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:12px 14px'>
    <div style='font-size:11px;font-weight:700;color:#15803d;margin-bottom:5px'>📤 出力CSV（ヘッダーなし・7列）　緯度経度 → <span style='color:#0369a1;font-weight:700'>{out_fmt_b2_lbl}</span></div>
    <code style='font-size:11px;display:block'>{_col_out}</code>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("")
    # 変換方向ごとにキーを変えることで、方向切り替え時にファイルを自動リセット
    _upload_key = "u1_jpc" if dir2 == "平面直角 → 緯度経度" else "u1_ll"
    up1 = st.file_uploader("CSVファイルをアップロード", ["csv","txt"], key=_upload_key)
    src = up1.read().decode("utf-8-sig") if up1 else ""

    if src.strip():
        try:
            # ヘッダーなし・先頭4列のみ読み込み（5列目以降は無視）
            df_in = pd.read_csv(io.StringIO(src), header=None, dtype=str)
            # 4列未満は空列補完
            while len(df_in.columns) < 4:
                df_in[len(df_in.columns)] = ""
            # 5列目以降を切り捨て
            df_in = df_in.iloc[:, :4]

            def _v(s):
                s = str(s).strip()
                return "" if s.lower() in ("nan","none","") else s

            rows_out = []
            pb = st.progress(0, "変換中...")
            total = len(df_in)

            for idx, (_, row) in enumerate(df_in.iterrows()):
                pb.progress((idx+1)/total, f"{idx+1}/{total} 点処理中")
                try:
                    name  = _v(row.iloc[0])
                    col_b = _v(row.iloc[1])
                    col_c = _v(row.iloc[2])
                    col_d = _v(row.iloc[3]) if len(row) > 3 else ""

                    out_x = out_y = out_z = out_lat = out_lon = out_h = ""
                    lat_dd = lon_dd = None

                    if dir2 == "平面直角 → 緯度経度":
                        # B=X(数値) C=Y(数値) D=Z標高(数値・省略可)
                        if not (col_b and col_c):
                            raise ValueError("B列(X)・C列(Y)が必要です")
                        Xv = float(col_b); Yv = float(col_c)
                        Zv = float(col_d) if col_d else None
                        res = jpc_to_latlon(Xv, Yv, Z)
                        if res is None: raise ValueError(f"系番号 {Z} が無効")
                        lat_dd, lon_dd = res
                        N = None; ellH = None
                        if GEOID_KEY != "NONE" and Zv is not None:
                            if idx > 0:
                                time.sleep(0.3)  # API連続呼び出しのレート制限回避
                            N = fetch_geoid(lat_dd, lon_dd, GEOID_KEY)
                            if N is not None:
                                ellH = Zv + N
                            else:
                                ellH = Zv  # N取得失敗時はZをそのまま楕円体高として使用
                        elif Zv is not None:
                            ellH = Zv

                        out_x   = f"{Xv:.4f}"
                        out_y   = f"{Yv:.4f}"
                        out_z   = f"{Zv:.4f}" if Zv is not None else ""
                        out_lat = format_angle(lat_dd, FMT_B2)
                        out_lon = format_angle(lon_dd, FMT_B2)
                        out_h   = f"{ellH:.4f}" if ellH is not None else ""
                        out_n   = f"{N:.4f}" if N is not None else ""
                        _detected_fmt = ""

                    else:
                        # B=緯度（5フォーマット自動判別）C=経度 D=楕円体高（省略可）
                        if not (col_b and col_c):
                            raise ValueError("B列(緯度)・C列(経度)が必要です")
                        lv,  _fmt_lat = auto_parse_angle(col_b)
                        lov, _fmt_lon = auto_parse_angle(col_c)
                        hv  = float(col_d) if col_d else None
                        res = latlon_to_jpc(lv, lov, Z)
                        if res is None: raise ValueError(f"系番号 {Z} が無効")
                        Xr, Yr = res
                        lat_dd, lon_dd = lv, lov
                        N_b = None; elev_b = None
                        if hv is not None and GEOID_KEY != "NONE":
                            if idx > 0:
                                time.sleep(0.3)  # API連続呼び出しのレート制限回避
                            N_b = fetch_geoid(lv, lov, GEOID_KEY)
                            if N_b is not None:
                                elev_b = hv - N_b
                            else:
                                elev_b = hv  # N取得失敗時はhをそのまま使用
                        elif hv is not None:
                            elev_b = hv
                        out_x   = f"{Xr:.4f}"
                        out_y   = f"{Yr:.4f}"
                        out_z   = f"{elev_b:.4f}" if elev_b is not None else ""
                        out_lat = format_angle(lv, FMT_B2)
                        out_lon = format_angle(lov, FMT_B2)
                        out_h   = f"{hv:.4f}" if hv is not None else ""
                        out_n   = f"{N_b:.4f}" if N_b is not None else ""
                        _detected_fmt = _fmt_lat

                    _fmt_labels = {
                        "decimal":"十進角度","dms":"度分秒","bearing":"方位角",
                        "ddmmssss":"度分秒圧縮","gons":"Gons","":"—",
                    }
                    rows_out.append({
                        "点名":          name,
                        "X(m)":          out_x,
                        "Y(m)":          out_y,
                        "Z標高(m)":      out_z,
                        "緯度":          out_lat,
                        "経度":          out_lon,
                        "楕円体高(m)":   out_h,
                        "ジオイド高N(m)": out_n,
                        "判別FMT":       _fmt_labels.get(_detected_fmt, _detected_fmt),
                        "_lat": lat_dd, "_lon": lon_dd, "_err": None,
                    })
                except Exception as ex:
                    rows_out.append({
                        "点名": _v(row.iloc[0]) if len(row) > 0 else "?",
                        "_err": str(ex), "_lat": None, "_lon": None,
                    })

            pb.empty()
            dfr = pd.DataFrame(rows_out)
            ok = dfr[dfr["_err"].isna()]
            ng = dfr[dfr["_err"].notna()]

            st.success(f"✅ {len(ok)} 点完了" + (f"　⚠️ {len(ng)} 件エラー" if len(ng) else ""))
            if len(ng):
                with st.expander("⚠️ エラー詳細"):
                    for _, r in ng.iterrows():
                        st.markdown(f"<span class='err'>❌ {r['点名']} — {r['_err']}</span>",
                                    unsafe_allow_html=True)

            if dir2 == "緯度経度 → 平面直角":
                show = ["点名","判別FMT","X(m)","Y(m)","Z標高(m)","緯度","経度","楕円体高(m)","ジオイド高N(m)"]
                st.caption("💡 判別FMT = 入力の緯度・経度に自動判別されたフォーマット")
            else:
                show = ["点名","X(m)","Y(m)","Z標高(m)","緯度","経度","楕円体高(m)","ジオイド高N(m)"]
            show = [c for c in show if c in ok.columns]
            st.dataframe(ok[show], use_container_width=True, hide_index=True)

            if ok["_lat"].notna().any():
                st.markdown("#### 📍 地図")
                map_pts = [{"name": r["点名"], "lat": r["_lat"], "lon": r["_lon"]}
                           for _, r in ok[ok["_lat"].notna()].iterrows()]
                render_map(map_pts, map_style_lbl, zoom=9)

            # 出力CSV（ヘッダーなし: A=点名,B=X,C=Y,D=Z標高,E=緯度,F=経度,G=楕円体高）
            csv_lines = [
                ",".join([
                    r["点名"], r["X(m)"], r["Y(m)"], r["Z標高(m)"],
                    r["緯度"], r["経度"], r["楕円体高(m)"],
                ])
                for _, r in ok.iterrows()
            ]
            _fn_batch_out = _csv_filename_ui("csv_fn_batch", "例: 現場名_一括変換")
            st.download_button(
                f"📥 結果 CSV ダウンロード（{out_fmt_b2_lbl}）",
                "\ufeff" + "\n".join(csv_lines),
                _fn_batch_out or "dummy.csv",
                "text/csv; charset=utf-8-sig",
                disabled=(_fn_batch_out is None),
            )

        except Exception as ex:
            st.error(f"処理エラー: {ex}")
    else:
        st.info("CSVファイルをアップロードしてください。")

# ── 免責事項・プライバシーポリシー ──
with st.expander("📋 免責事項・プライバシーポリシー"):
    st.markdown("""
本ツール（GNSS SmartShift ICT）による座標変換・ジオイド高計算・各種出力値はすべて参考値です。
変換結果を実際の測量・設計・施工・出来形管理等に使用する場合は、必ず有資格者（測量士・測量士補等）による検証・確認を行ってください。
計算結果の利用によって生じた損害・損失・不利益について、開発者は一切の責任を負いません。

国土地理院ジオイドAPI（JPGEO2024/JPGEO2011）の取得失敗時は N=0 として処理するため Z標高 = 楕円体高 として出力されます。

**プライバシー：** 入力データはサーバーに保存されません。外部通信は国土地理院ジオイドAPI（vldb.gsi.go.jp）のみです。

© 2026 biz-cpu　｜　本ソフトウェアの無断複製・改変・再配布・商用転用を禁じます。
""")
st.caption("© 2026 biz-cpu　｜　GNSS SmartShift ICT　｜　Kawase (2011) 高次ガウス・クリューゲル展開式")

# ── ページトップへ戻るボタン（parent.document に直接注入）──
import streamlit.components.v1 as _comp_top
_comp_top.html("""
<script>
(function(){
  // すでに注入済みなら何もしない
  if (window.parent.document.getElementById('gss-back-to-top')) return;

  var p = window.parent.document;

  // スタイル注入
  var style = p.createElement('style');
  style.textContent = [
    '#gss-back-to-top{',
    '  position:fixed;',
    '  bottom:32px;',
    '  left:50%;',
    '  transform:translateX(-50%) translateY(16px);',
    '  z-index:99999;',
    '  width:52px;height:52px;',
    '  border-radius:50%;',
    '  background:linear-gradient(135deg,#3b82f6,#1d4ed8);',
    '  color:#fff;border:none;',
    '  font-size:24px;line-height:52px;',
    '  text-align:center;cursor:pointer;',
    '  box-shadow:0 4px 16px rgba(59,130,246,0.55);',
    '  opacity:0;',
    '  transition:opacity .3s,transform .3s;',
    '  display:flex;align-items:center;justify-content:center;',
    '}',
    '#gss-back-to-top.show{opacity:1;transform:translateX(-50%) translateY(0);}',
    '#gss-back-to-top:hover{background:linear-gradient(135deg,#60a5fa,#2563eb);}'
  ].join('');
  p.head.appendChild(style);

  // ボタン注入
  var btn = p.createElement('button');
  btn.id = 'gss-back-to-top';
  btn.title = 'ページトップへ戻る';
  btn.innerHTML = '&#8679;';
  p.body.appendChild(btn);

  // スクロール監視
  function getMain(){
    return p.querySelector('section[data-testid="stMain"]') ||
           p.querySelector('.main')  ||
           p.querySelector('section.main') ||
           null;
  }

  function onScroll(){
    var main = getMain();
    var top = main ? main.scrollTop : p.documentElement.scrollTop;
    if(top > 300){ btn.classList.add('show'); }
    else          { btn.classList.remove('show'); }
  }

  btn.addEventListener('click', function(){
    var main = getMain();
    if(main){ main.scrollTo({top:0, behavior:'smooth'}); }
    else    { p.documentElement.scrollTo({top:0, behavior:'smooth'}); }
  });

  function attach(n){
    var main = getMain();
    if(main){
      main.addEventListener('scroll', onScroll);
    } else {
      p.addEventListener('scroll', onScroll);
    }
    if(n > 0 && !getMain()) setTimeout(function(){ attach(n-1); }, 500);
  }

  if(p.readyState === 'loading'){
    p.addEventListener('DOMContentLoaded', function(){ attach(10); });
  } else {
    attach(10);
  }
})();
</script>
""", height=0)
