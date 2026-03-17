"""
ローカライゼーション用座標変換
平面直角座標（1〜19系）↔ 緯度経度
Kawase (2011) 高精度ガウス・クリューゲル投影
ジオイドモデル: JPGEO2024（国土地理院API）
"""

import math
import io
import json
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
    "JGD2011": {"label": "JGD2011（測地成果2011）",   "a": 6378137.0,   "f": 1/298.257222101,
                "toWGS84": {"dx":0,"dy":0,"dz":0,"rx":0,"ry":0,"rz":0,"ds":0}},
    "JGD2000": {"label": "JGD2000（測地成果2000）",   "a": 6378137.0,   "f": 1/298.257222101,
                "toWGS84": {"dx":0,"dy":0,"dz":0,"rx":0,"ry":0,"rz":0,"ds":0}},
    "WGS84":   {"label": "WGS84",                     "a": 6378137.0,   "f": 1/298.257223563,
                "toWGS84": None},
    "TOKYO":   {"label": "旧日本測地系（Tokyo97）",   "a": 6377397.155, "f": 1/299.1528128,
                "toWGS84": {"dx":-148,"dy":507,"dz":685,"rx":0,"ry":0,"rz":0,"ds":0}},
}

JPC_ORIGINS = {
     1:(33.0,129.5),        2:(33.0,131.0),        3:(36.0,132.166666667),
     4:(33.0,133.5),        5:(36.0,134.333333333), 6:(36.0,136.0),
     7:(36.0,137.166666667),8:(36.0,138.5),         9:(36.0,139.833333333),
    10:(40.0,140.833333333),11:(44.0,140.25),       12:(44.0,142.25),
    13:(44.0,144.25),       14:(26.0,142.0),        15:(26.0,127.5),
    16:(26.0,124.0),        17:(26.0,131.0),        18:(20.0,136.0),
    19:(26.0,154.0),
}

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

def latlon_to_jpc(lat_deg, lon_deg, zone):
    if zone not in JPC_ORIGINS: return None
    la0,lo0=JPC_ORIGINS[zone]
    phi=lat_deg*DEG; lam=lon_deg*DEG; phi0=la0*DEG; lam0=lo0*DEG
    sinP=math.sin(phi); psi=math.atanh(sinP)-_e*math.atanh(_e*sinP); dl=lam-lam0
    xi_=math.atan2(math.sinh(psi),math.cos(dl)); eta_=math.atanh(math.sin(dl)/math.cosh(psi))
    xi=xi_+sum(_alpha[j]*math.sin(2*j*xi_)*math.cosh(2*j*eta_) for j in range(1,5))
    eta=eta_+sum(_alpha[j]*math.cos(2*j*xi_)*math.sinh(2*j*eta_) for j in range(1,5))
    return _m0*_A*xi-_S(phi0), _m0*_A*eta

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

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_geoid(lat: float, lon: float, model: str = "JPGEO2024"):
    if model == "NONE": return 0.0
    select = "0" if model == "JPGEO2024" else "1"
    url = (
        "https://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh/cgi/geoidcalc.pl"
        f"?select={select}&tanni=1&outputType=json&latitude={lat:.8f}&longitude={lon:.8f}"
    )
    try:
        r = requests.get(url, timeout=8)
        return float(r.json()["OutputData"]["geoidHeight"])
    except Exception:
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
    import re
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
    import json as _json

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
    pins_js = _json.dumps([
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

st.set_page_config(
    page_title="ローカライゼーション用座標変換",
    page_icon="📐", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Noto Sans JP', sans-serif; }

/* ── サイドバー ── */
section[data-testid="stSidebar"] { background: #0f172a !important; }
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
.app-hdr { background:linear-gradient(135deg,#0f172a 0%,#1e3a5f 100%);
           border-radius:14px; padding:22px 28px; margin-bottom:20px; }
.app-hdr h1 { margin:0; font-size:22px; font-weight:700; color:#f1f5f9; }
.app-hdr p  { margin:4px 0 0; color:#64748b; font-size:11px; letter-spacing:.2em; text-transform:uppercase; }

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
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# 7. サイドバー共通設定
# ═══════════════════════════════════════════════════════

with st.sidebar:
    # サイドバーのマージンを CSS でさらに詰める
    st.markdown("""<style>
section[data-testid="stSidebar"] .block-container{padding-top:0.8rem!important;padding-bottom:0.5rem!important;}
section[data-testid="stSidebar"] .stMarkdown p{margin:0!important;padding:0!important;line-height:1.3!important;}
section[data-testid="stSidebar"] .element-container{margin-bottom:0!important;}
section[data-testid="stSidebar"] hr{margin:6px 0!important;}
section[data-testid="stSidebar"] .stSelectbox{margin-bottom:0!important;}
section[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"]{gap:0!important;}
</style>""", unsafe_allow_html=True)

    st.markdown("<div style='font-size:15px;font-weight:700;color:#f1f5f9;padding:4px 0 6px'>⚙️ 共通設定</div>", unsafe_allow_html=True)
    st.divider()

    # 座標系
    st.markdown("<div style='font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:.12em;text-transform:uppercase;margin-bottom:3px'>📌 座標系（系番号）</div>", unsafe_allow_html=True)
    zone_inv = {v:k for k,v in JPC_ZONE_LABELS.items()}
    zone_lbl = st.selectbox("座標系", list(JPC_ZONE_LABELS.values()),
                             index=8, label_visibility="collapsed")
    Z = zone_inv[zone_lbl]
    la0, lo0 = JPC_ORIGINS[Z]
    st.markdown(f"<div style='display:flex;align-items:center;gap:6px;margin:3px 0 2px'>"
                f"<span class='zbadge' style='font-size:10px;padding:1px 8px'>第 {Z} 系</span>"
                f"<span style='font-size:10px;color:#64748b'>φ₀={la0}° λ₀={lo0}°</span></div>", unsafe_allow_html=True)

    st.divider()

    # 測地系
    st.markdown("<div style='font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:.12em;text-transform:uppercase;margin-bottom:3px'>🌐 測地系</div>", unsafe_allow_html=True)
    datum_inv = {v["label"]:k for k,v in DATUMS.items()}
    datum_lbl = st.selectbox("測地系", list(datum_inv.keys()),
                              index=0, label_visibility="collapsed")
    DATUM = datum_inv[datum_lbl]

    st.divider()

    # ジオイドモデル
    st.markdown("<div style='font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:.12em;text-transform:uppercase;margin-bottom:3px'>📡 ジオイドモデル</div>", unsafe_allow_html=True)
    geoid_lbl = st.selectbox("ジオイドモデル", list(GEOID_MODELS.values()),
                              index=0, label_visibility="collapsed")
    GEOID_KEY = [k for k,v in GEOID_MODELS.items() if v==geoid_lbl][0]

    st.divider()

    # 地図スタイル
    st.markdown("<div style='font-size:10px;font-weight:700;color:#94a3b8;letter-spacing:.12em;text-transform:uppercase;margin-bottom:3px'>🗺️ 地図スタイル</div>", unsafe_allow_html=True)
    map_style_lbl = st.selectbox("地図スタイル", list(MAP_STYLES.keys()),
                                  index=0, label_visibility="collapsed")

    st.divider()

    st.markdown(f"""<div class='acc' style='margin-bottom:4px'>往復誤差 &lt; 0.01mm</div>
<div style='font-size:10px;color:#475569;line-height:1.6'>
GRS80 / m₀=0.9999 / Kawase2011
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# 8. メインヘッダー
# ═══════════════════════════════════════════════════════

# FMT/fmt_lbl はTAB内の各モードで個別に定義するため、ここではデフォルトのみ設定
_FMT_DEFAULT = "decimal"
_fmt_lbl_default = list(OUTPUT_FORMATS.keys())[0]

st.markdown(f"""
<div class="app-hdr">
  <h1>📐 ローカライゼーション用座標変換</h1>
  <p>第 {Z} 系 &nbsp;·&nbsp; {datum_lbl} &nbsp;·&nbsp; {map_style_lbl}</p>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📍 単点変換", "📋 CSV 一括変換", "ℹ️ 系番号一覧"])

# ═══════════════════════════════════════════════════════
# 9. TAB 1: 単点変換（複数点 / 地図ピン / 入力形式選択）
# ═══════════════════════════════════════════════════════

# ── session_state 直接バインド方式のヘルパー ─────────────
# text_input は key= のみ使い value= を渡さない。
# これにより サイドバー変更・地図スタイル変更時も入力値が保持される。
# スワップ時は session_state のキーを直接書き換えるため即時反映される。

def _init_jpc():
    if "pts_jpc" not in st.session_state:
        st.session_state["pts_jpc"] = [{"name":"pt1","x":"","y":"","z":""}]
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
        st.session_state["pts_ll"] = [{"name":"pt1","lat":"","lon":"","h":""}]
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
                    ["平面直角 → 緯度経度", "緯度経度 → 平面直角", "緯度経度 形式変換"],
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
                st.session_state["pts_jpc"].append({"name":f"pt{n}","x":"","y":"","z":""})
                st.rerun()
        with col_clr:
            if st.button("🗑 全クリア", key="clr_jpc"):
                st.session_state["pts_jpc"] = [{"name":"pt1","x":"","y":"","z":""}]
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
            c0,c1,c2,c3,c4,c5 = st.columns([0.6,1.4,2,2,2,0.45])
            with c0:
                toppad = "32" if i==0 else "8"
                st.markdown(
                    f"<div style='padding-top:{toppad}px;font-size:12px;font-weight:700;color:#64748b'>#{i+1}</div>",
                    unsafe_allow_html=True)
            with c1:
                st.text_input("点名", key=f"jpc_name_{i}",
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
            index=0, label_visibility="collapsed", key="out_fmt_jpc"
        )
        FMT_JPC = OUTPUT_FORMATS[out_fmt_jpc_lbl]

        st.markdown("---")
        has_input = any(pt["x"].strip() and pt["y"].strip() for pt in pts)

        if has_input:
            st.markdown("<div class='sec-label'>変換結果</div>", unsafe_allow_html=True)
            map_rows, csv_rows = [], []
            csv_hdr = f"点名,X(m),Y(m),Z標高(m),ジオイド高N(m),楕円体高h(m),緯度({out_fmt_jpc_lbl}),経度({out_fmt_jpc_lbl}),緯度_DD,経度_DD"

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
                        with st.spinner(f"[{pt['name']}] ジオイド高 取得中..."):
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
                        hs  = f"{ellH:.3f} m" if ellH is not None else "---"
                        sub = f"Z={Zv:.3f}+N={N:.4f}" if (ellH is not None and N is not None and Zv is not None) else ""
                        st.markdown(
                            f"<div class='rc'><div class='rc-lbl' style='color:#8b5cf6'>楕円体高 h (m)</div>"
                            f"<div class='rc-val'>{hs}</div>"
                            f"<div class='rc-sub'>{sub}</div></div>",
                            unsafe_allow_html=True)

                    with st.expander(f"🔢 {pt['name']} 全フォーマット"):
                        st.dataframe(pd.DataFrame([
                            {"フォーマット":fl,"緯度":format_angle(lat_dd,fk),"経度":format_angle(lon_dd,fk)}
                            for fl,fk in OUTPUT_FORMATS.items()
                        ]), use_container_width=True, hide_index=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                    tip = f"Z={Zv:.3f}m / N={N:.4f}m / h={ellH:.3f}m" if ellH is not None else fmt_decimal(lat_dd)
                    map_rows.append({"name":pt["name"],"lat":lat_dd,"lon":lon_dd,"tooltip":tip})
                    csv_rows.append(
                        f"{pt['name']},{Xv},{Yv},"
                        + (f"{Zv:.3f}" if Zv is not None else "") + ","
                        + (f"{N:.4f}"  if N  is not None else "") + ","
                        + (f"{ellH:.3f}" if ellH is not None else "") + ","
                        + f"{format_angle(lat_dd,FMT_JPC)},{format_angle(lon_dd,FMT_JPC)},"
                        + f"{fmt_decimal(lat_dd)},{fmt_decimal(lon_dd)}"
                    )
                except (ValueError, Exception) as ex:
                    st.error(f"[{pt['name']}] エラー: {ex}")

            if map_rows:
                st.markdown("#### 📍 地図")
                render_map(map_rows, map_style_lbl, zoom=13)
                csv_out = "\ufeff" + csv_hdr + "\n" + "\n".join(csv_rows)
                st.download_button("📥 全点 CSV ダウンロード", csv_out, "converted.csv", "text/csv")
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

        # 出力フォーマット選択（入力フォーマットのすぐ下）
        st.markdown("<div class='sec-label'>出力フォーマット（緯度経度・参照用）</div>", unsafe_allow_html=True)
        out_fmt_ll_lbl = st.selectbox(
            "出力フォーマットLL",
            list(OUTPUT_FORMATS.keys()),
            index=0, label_visibility="collapsed", key="out_fmt_ll"
        )
        FMT_LL = OUTPUT_FORMATS[out_fmt_ll_lbl]

        col_add2, col_clr2, col_swap2, _ = st.columns([1,1,1.4,4])
        with col_add2:
            if st.button("＋ 点を追加", key="add_ll"):
                _read_ll()
                n = len(st.session_state["pts_ll"]) + 1
                st.session_state["pts_ll"].append({"name":f"pt{n}","lat":"","lon":"","h":""})
                st.rerun()
        with col_clr2:
            if st.button("🗑 全クリア", key="clr_ll"):
                st.session_state["pts_ll"] = [{"name":"pt1","lat":"","lon":"","h":""}]
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
            c0,c1,c2,c3,c4,c5 = st.columns([0.6,1.4,2.3,2.3,1.8,0.45])
            with c0:
                toppad = "32" if i==0 else "8"
                st.markdown(
                    f"<div style='padding-top:{toppad}px;font-size:12px;font-weight:700;color:#64748b'>#{i+1}</div>",
                    unsafe_allow_html=True)
            with c1:
                st.text_input("点名", key=f"ll_name_{i}",
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

        st.markdown("---")
        has_input2 = any(pt["lat"].strip() and pt["lon"].strip() for pt in pts2)

        if has_input2:
            st.markdown("<div class='sec-label'>変換結果</div>", unsafe_allow_html=True)
            map_rows2, csv_rows2 = [], []
            csv_hdr2 = "点名,緯度_入力,経度_入力,楕円体高h(m),X(m),Y(m),Z標高(m),系番号"

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
                            with st.spinner(f"[{pt['name']}] ジオイド高 取得中..."):
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
                            z_str = f"{elev_ll:.3f} m"
                            z_sub = f"h={hv:.3f} - N={N_ll:.4f}" if N_ll is not None else "補正なし"
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
                                      "tooltip":f"X={Xr:.3f} / Y={Yr:.3f}" + (f" / Z={elev_ll:.3f}m" if elev_ll else "")})
                    csv_rows2.append(
                        f"{pt['name']},{fmt_decimal(lv)},{fmt_decimal(lov)},{hv:.3f},"
                        + f"{Xr:.4f},{Yr:.4f},"
                        + (f"{elev_ll:.3f}" if elev_ll is not None else "") + ","
                        + f"{Z}"
                    )

                except (ValueError, Exception) as ex:
                    st.error(f"[{pt['name']}] エラー: {ex}")

            if map_rows2:
                st.markdown("#### 📍 地図")
                render_map(map_rows2, map_style_lbl, zoom=13)
                csv_out2 = "\ufeff" + csv_hdr2 + "\n" + "\n".join(csv_rows2)
                st.download_button("📥 全点 CSV ダウンロード", csv_out2, "converted.csv", "text/csv")
        else:
            st.info(f"緯度・経度を {in_fmt_lbl} 形式で入力してください。")

    # ══════════════════════════════
    # 緯度経度 形式変換
    # ══════════════════════════════
    else:  # dir1 == "緯度経度 形式変換"
        # session_state 初期化
        if "pts_cvt" not in st.session_state:
            st.session_state["pts_cvt"] = [{"name":"pt1","lat":"","lon":""}]
        for i, pt in enumerate(st.session_state["pts_cvt"]):
            for f,v in [("name",pt["name"]),("lat",pt["lat"]),("lon",pt["lon"])]:
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

        # 出力フォーマット
        st.markdown("<div class='sec-label'>出力フォーマット</div>", unsafe_allow_html=True)
        out_fmt_cvt_lbl = st.selectbox(
            "出力フォーマット（形式変換）",
            list(OUTPUT_FORMATS.keys()),
            index=1, label_visibility="collapsed", key="out_fmt_cvt"
        )
        OUT_FMT_CVT = OUTPUT_FORMATS[out_fmt_cvt_lbl]

        ph_cvt_lat = FORMAT_PLACEHOLDER[IN_FMT_CVT]
        ph_cvt_lon = FORMAT_PLACEHOLDER[IN_FMT_CVT].replace("35","139").replace("40","47")

        col_add_c, col_clr_c, col_swap_c, _ = st.columns([1,1,1.4,4])
        with col_add_c:
            if st.button("＋ 点を追加", key="add_cvt"):
                for i, pt in enumerate(st.session_state["pts_cvt"]):
                    for f in ("name","lat","lon"):
                        k = f"cvt_{f}_{i}"
                        if k in st.session_state: pt[f] = st.session_state[k]
                n = len(st.session_state["pts_cvt"]) + 1
                st.session_state["pts_cvt"].append({"name":f"pt{n}","lat":"","lon":""})
                st.rerun()
        with col_clr_c:
            if st.button("🗑 全クリア", key="clr_cvt"):
                st.session_state["pts_cvt"] = [{"name":"pt1","lat":"","lon":""}]
                for k in [k for k in st.session_state if k.startswith("cvt_")]:
                    del st.session_state[k]
                st.rerun()
        with col_swap_c:
            if st.button("⇄ 緯↔経 入替", key="swap_cvt"):
                for i, pt in enumerate(st.session_state["pts_cvt"]):
                    for f in ("name","lat","lon"):
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
            c0,c1,c2,c3,c4 = st.columns([0.6,1.4,2.5,2.5,0.45])
            with c0:
                toppad = "32" if i==0 else "8"
                st.markdown(
                    f"<div style='padding-top:{toppad}px;font-size:12px;font-weight:700;color:#64748b'>#{i+1}</div>",
                    unsafe_allow_html=True)
            with c1:
                st.text_input("点名", key=f"cvt_name_{i}",
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
                if i == 0:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                if st.button("✕", key=f"del_cvt_{i}", disabled=len(pts_cvt)==1):
                    del_idx_c = i

        if del_idx_c is not None:
            for i, pt in enumerate(st.session_state["pts_cvt"]):
                for f in ("name","lat","lon"):
                    k = f"cvt_{f}_{i}"
                    if k in st.session_state: pt[f] = st.session_state[k]
            new_pts_c = [p for j,p in enumerate(st.session_state["pts_cvt"]) if j != del_idx_c]
            for k in [k for k in st.session_state if k.startswith("cvt_name_") or k.startswith("cvt_lat_") or k.startswith("cvt_lon_")]:
                del st.session_state[k]
            st.session_state["pts_cvt"] = new_pts_c
            st.rerun()

        # 現在値を同期
        for i, pt in enumerate(pts_cvt):
            for f in ("name","lat","lon"):
                k = f"cvt_{f}_{i}"
                if k in st.session_state: pt[f] = st.session_state[k]

        st.markdown("---")
        has_cvt = any(pt["lat"].strip() and pt["lon"].strip() for pt in pts_cvt)

        if has_cvt:
            st.markdown(
                f"<div class='sec-label'>変換結果 &nbsp;"
                f"<span style='font-size:10px;color:#64748b;font-weight:400'>"
                f"{in_fmt_cvt_lbl} &rarr; {out_fmt_cvt_lbl}</span></div>",
                unsafe_allow_html=True)

            map_rowsc, csv_rowsc = [], []
            csv_hdrc = f"点名,緯度_入力({in_fmt_cvt_lbl}),経度_入力({in_fmt_cvt_lbl}),緯度_出力({out_fmt_cvt_lbl}),経度_出力({out_fmt_cvt_lbl}),緯度_DD,経度_DD"

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

                    rc1,rc2 = st.columns(2)
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

                    with st.expander(f"🔢 {pt['name']} 全フォーマット"):
                        st.dataframe(pd.DataFrame([
                            {"フォーマット":fl,"緯度":format_angle(lat_dd,fk),"経度":format_angle(lon_dd,fk)}
                            for fl,fk in OUTPUT_FORMATS.items()
                        ]), use_container_width=True, hide_index=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                    map_rowsc.append({"name":pt["name"],"lat":lat_dd,"lon":lon_dd,
                                      "tooltip":f"{lat_out} / {lon_out}"})
                    csv_rowsc.append(
                        f"{pt['name']},{pt['lat']},{pt['lon']},{lat_out},{lon_out},"
                        + f"{fmt_decimal(lat_dd)},{fmt_decimal(lon_dd)}"
                    )
                except (ValueError, Exception) as ex:
                    st.error(f"[{pt['name']}] エラー: {ex}")

            if map_rowsc:
                st.markdown("#### 📍 地図")
                render_map(map_rowsc, map_style_lbl, zoom=13)
                csv_outc = "\ufeff" + csv_hdrc + "\n" + "\n".join(csv_rowsc)
                st.download_button("📥 全点 CSV ダウンロード", csv_outc, "converted_fmt.csv", "text/csv")
        else:
            st.info(f"緯度・経度を {in_fmt_cvt_lbl} 形式で入力してください。")


# ═══════════════════════════════════════════════════════
# 10. TAB 2: CSV 一括変換
# ═══════════════════════════════════════════════════════

with tab2:
    dir2 = st.radio("変換方向",
                    ["平面直角 → 緯度経度（ヘッダーなし）","緯度経度 → 平面直角（ヘッダーあり）"],
                    horizontal=True, key="d2")
    st.markdown("---")
    S1 = "t1,-42090.367,-23809.574,67.222\nt2,-42089.211,-23951.174,67.659\nt3,-42238.931,-23876.726,66.813"
    S2 = "name,lat,lon,h,datum\npt1,35.68123,139.76712,10.5,JGD2011\npt2,34.69374,135.50218,5.2,JGD2011\npt3,38.26822,140.86940,52.3,JGD2011"

    if "平面直角" in dir2:
        st.markdown("""
**CSV フォーマット（ヘッダー行なし）**
```
点名,X(m),Y(m),標高Z(m)
t1,-42090.367,-23809.574,67.222
```
標高列を入力するとジオイド高APIで楕円体高を自動計算します。
        """)
        up1 = st.file_uploader("CSVファイル",["csv","txt"],key="u1")
        ca,cb = st.columns([3,1])
        with ca: tx1 = st.text_area("または貼り付け",height=130,placeholder=S1,key="t1")
        with cb:
            if st.button("サンプル",key="s1"): st.session_state["t1"]=S1; st.rerun()
        src = (up1.read().decode("utf-8-sig") if up1 else "") or (tx1 if tx1 else "")

        if src.strip():
            try:
                df_in = pd.read_csv(io.StringIO(src), header=None, dtype=str)
                rows = []
                pb = st.progress(0, "変換中...")
                total = len(df_in)
                for idx,(_, row) in enumerate(df_in.iterrows()):
                    pb.progress((idx+1)/total, f"{idx+1}/{total} 点処理中")
                    try:
                        name = str(row.iloc[0])
                        X = float(row.iloc[1]); Y = float(row.iloc[2])
                        elev = float(row.iloc[3]) if len(row)>3 and pd.notna(row.iloc[3]) else None
                        res = jpc_to_latlon(X, Y, Z)
                        if res is None: raise ValueError(f"系番号 {Z} が無効")
                        lat_dd, lon_dd = res
                        N=None; ellH=None
                        if GEOID_KEY != "NONE" and elev is not None:
                            N = fetch_geoid(lat_dd, lon_dd, GEOID_KEY)
                            if N is not None: ellH = elev + N
                        rows.append({
                            "点名":name,"X(m)":X,"Y(m)":Y,
                            "標高Z(m)": f"{elev:.3f}" if elev is not None else "",
                            "ジオイド高N(m)": f"{N:.4f}" if N is not None else "",
                            "楕円体高h(m)": f"{ellH:.3f}" if ellH is not None else "",
                            "緯度": format_angle(lat_dd, FMT),
                            "経度": format_angle(lon_dd, FMT),
                            "緯度_DD": fmt_decimal(lat_dd),
                            "経度_DD": fmt_decimal(lon_dd),
                            "_lat":lat_dd,"_lon":lon_dd,"_err":None,
                        })
                    except Exception as ex:
                        rows.append({"点名":str(row.iloc[0]) if len(row)>0 else "?",
                                     "_err":str(ex),"_lat":None,"_lon":None})
                pb.empty()
                dfr = pd.DataFrame(rows)
                ok = dfr[dfr["_err"].isna()]; ng = dfr[dfr["_err"].notna()]
                st.success(f"✅ {len(ok)} 点完了" + (f"　⚠️ {len(ng)} 件エラー" if len(ng) else ""))
                if len(ng):
                    with st.expander("⚠️ エラー"):
                        for _, r in ng.iterrows():
                            st.markdown(f"<span class='err'>❌ {r['点名']} — {r['_err']}</span>", unsafe_allow_html=True)
                show = [c for c in ["点名","X(m)","Y(m)","標高Z(m)","ジオイド高N(m)","楕円体高h(m)","緯度","経度"] if c in ok.columns]
                st.dataframe(ok[show], use_container_width=True, hide_index=True)
                if ok["_lat"].notna().any():
                    st.markdown("#### 📍 地図")
                    map_pts = [{"name":r["点名"],"lat":r["_lat"],"lon":r["_lon"]}
                               for _,r in ok[ok["_lat"].notna()].iterrows()]
                    render_map(map_pts, map_style_lbl, zoom=9)
                out = [c for c in ["点名","X(m)","Y(m)","標高Z(m)","ジオイド高N(m)","楕円体高h(m)","緯度","経度","緯度_DD","経度_DD"] if c in ok.columns]
                st.download_button("📥 結果 CSV", "\ufeff"+ok[out].to_csv(index=False), "batch.csv","text/csv")
            except Exception as ex:
                st.error(f"処理エラー: {ex}")
        else:
            st.info("CSV をアップロードまたは貼り付けてください。")

    else:
        st.markdown("""
**CSV フォーマット（1行目ヘッダー必須）**
```
name,lat,lon,h,datum
pt1,35.68123,139.76712,10.5,JGD2011
```
name / h / datum 列は省略可。lat・lon は十進角度（DD.DDDDDDDD）で入力してください。
        """)
        up2 = st.file_uploader("CSVファイル",["csv","txt"],key="u2")
        cc,cd = st.columns([3,1])
        with cc: tx2 = st.text_area("または貼り付け",height=130,placeholder=S2,key="t2")
        with cd:
            if st.button("サンプル",key="s2"): st.session_state["t2"]=S2; st.rerun()
        src2 = (up2.read().decode("utf-8-sig") if up2 else "") or (tx2 if tx2 else "")

        if src2.strip():
            try:
                df_in2 = pd.read_csv(io.StringIO(src2), dtype=str)
                cols = [c.strip().lower() for c in df_in2.columns]
                def ci(ns):
                    for n in ns:
                        if n in cols: return cols.index(n)
                    return None
                il=ci(["lat","緯度","latitude"]); iol=ci(["lon","lng","経度","longitude"])
                ih=ci(["h","z","ellh","height","楕円体高"]); inm=ci(["name","点名","id","no"])
                idt=ci(["datum","測地系"])
                if il is None or iol is None:
                    st.error("ヘッダーに 'lat' と 'lon' 列が必要です"); st.stop()
                rows2 = []
                for i, row in df_in2.iterrows():
                    try:
                        name = str(row.iloc[inm]) if inm is not None else str(i+1)
                        lv   = float(row.iloc[il]); lov = float(row.iloc[iol])
                        hv   = float(row.iloc[ih]) if ih is not None and pd.notna(row.iloc[ih]) else 0.0
                        res  = latlon_to_jpc(lv, lov, Z)
                        if res is None: raise ValueError(f"系番号 {Z} が無効")
                        px, py = res
                        rows2.append({"点名":name,"緯度_入力":fmt_decimal(lv),"経度_入力":fmt_decimal(lov),
                                      "楕円体高(m)":f"{hv:.3f}","X(m)":f"{px:.4f}","Y(m)":f"{py:.4f}",
                                      "_lat":lv,"_lon":lov,"_err":None})
                    except Exception as ex:
                        rows2.append({"点名":str(row.iloc[inm]) if inm is not None else "?",
                                      "_err":str(ex),"_lat":None,"_lon":None})
                dfr2 = pd.DataFrame(rows2)
                ok2 = dfr2[dfr2["_err"].isna()]; ng2 = dfr2[dfr2["_err"].notna()]
                st.success(f"✅ {len(ok2)} 点完了" + (f"　⚠️ {len(ng2)} 件エラー" if len(ng2) else ""))
                show2 = [c for c in ["点名","緯度_入力","経度_入力","楕円体高(m)","X(m)","Y(m)"] if c in ok2.columns]
                st.dataframe(ok2[show2], use_container_width=True, hide_index=True)
                if ok2["_lat"].notna().any():
                    st.markdown("#### 📍 地図")
                    map_pts2 = [{"name":r["点名"],"lat":r["_lat"],"lon":r["_lon"]}
                                for _,r in ok2[ok2["_lat"].notna()].iterrows()]
                    render_map(map_pts2, map_style_lbl, zoom=9)
                st.download_button("📥 結果 CSV", "\ufeff"+ok2[show2].to_csv(index=False), "batch.csv","text/csv")
            except Exception as ex:
                st.error(f"処理エラー: {ex}")
        else:
            st.info("CSV をアップロードまたは貼り付けてください。")

# ═══════════════════════════════════════════════════════
# 11. TAB 3: 系番号一覧
# ═══════════════════════════════════════════════════════

with tab3:
    st.markdown("### 公共測量 平面直角座標系 系番号一覧")
    st.caption("国土交通省告示（昭和48年建設省告示第143号）")
    rows_z = []
    for z in range(1,20):
        l0,o0 = JPC_ORIGINS[z]
        rows_z.append({"系番号":z,"適用地域":JPC_ZONE_LABELS[z].split(" — ")[1],
                       "原点緯度 φ₀":f"{l0}°","原点経度 λ₀":f"{o0}°",
                       "縮尺係数":"0.9999","選択中":"✅" if z==Z else ""})
    dfz = pd.DataFrame(rows_z)
    def hl(r): return ["background:#fef3c7"]*len(r) if r["選択中"]=="✅" else [""]*len(r)
    st.dataframe(dfz.style.apply(hl,axis=1), use_container_width=True, hide_index=True, height=680)
    st.markdown("""
---
**仕様**
- 準拠楕円体: **GRS80**（JGD2011 / JGD2000 共通）
- 投影法: **ガウス・クリューゲル正角投影**（Kawase 2011 高次展開式）
- 縮尺係数: **m₀ = 0.9999**（全系共通）
- 変換精度: **往復誤差 < 0.01 mm**
- ジオイド高: **国土地理院API**（JPGEO2024 / JPGEO2011）
- 旧日本測地系: Helmert 3パラメータ（Δx=−148, Δy=+507, Δz=+685 m）
    """)
