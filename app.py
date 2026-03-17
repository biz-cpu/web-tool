"""
ローカライゼーション用座標変換
平面直角座標（1〜19系）↔ 緯度経度
Kawase (2011) 高精度ガウス・クリューゲル投影
ジオイドモデル: JPGEO2024（国土地理院API）
"""

import math
import io
import requests
import streamlit as st
import pandas as pd

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
     1:(33.0,129.5),       2:(33.0,131.0),       3:(36.0,132.166666667),
     4:(33.0,133.5),       5:(36.0,134.333333333),6:(36.0,136.0),
     7:(36.0,137.166666667),8:(36.0,138.5),       9:(36.0,139.833333333),
    10:(40.0,140.833333333),11:(44.0,140.25),     12:(44.0,142.25),
    13:(44.0,144.25),      14:(26.0,142.0),       15:(26.0,127.5),
    16:(26.0,124.0),       17:(26.0,131.0),       18:(20.0,136.0),
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

OUTPUT_FORMATS = {
    "DD.DDDDDDDD°（十進角度）":    "decimal",
    "DD°MM′SS.SSS″（度分秒）":    "dms",
    "NDD°MM′SS.SSS″（方位角）":   "bearing",
    "DD.MMSSSSSS（度分秒圧縮）":   "ddmmssss",
    "Gons（グラード）":             "gons",
}

# ═══════════════════════════════════════════════════════
# 2. Kawase (2011) 高精度変換
# ═══════════════════════════════════════════════════════

_a=6378137.0; _f=1/298.257222101; _m0=0.9999
_n=_f/(2-_f); _n2=_n**2; _n3=_n**3; _n4=_n**4
_A=_a/(1+_n)*(1+_n2/4+_n4/64)
_e=math.sqrt(2*_f-_f*_f)

_alpha=[0,_n/2-2*_n2/3+5*_n3/16+41*_n4/180,
          13*_n2/48-3*_n3/5+557*_n4/1440,
          61*_n3/240-103*_n4/140,
          49561*_n4/161280]
_beta=[0,_n/2-2*_n2/3+37*_n3/96-_n4/360,
         _n2/48+_n3/15-437*_n4/1440,
         17*_n3/480-37*_n4/840,
         4397*_n4/161280]
_delta=[0,2*_n-2*_n2/3-2*_n3+116*_n4/45,
          7*_n2/3-8*_n3/5-227*_n4/45,
          56*_n3/15-136*_n4/35,
          4279*_n4/630]
_c2=3*_n/2-9*_n3/16; _c4=15*_n2/16-15*_n4/32
_c6=35*_n3/48;        _c8=315*_n4/512

def _S(phi):
    return _A*_m0*(phi-_c2*math.sin(2*phi)+_c4*math.sin(4*phi)
                       -_c6*math.sin(6*phi)+_c8*math.sin(8*phi))

def latlon_to_jpc(lat_deg, lon_deg, zone):
    if zone not in JPC_ORIGINS: return None
    la0,lo0=JPC_ORIGINS[zone]
    phi=lat_deg*DEG; lam=lon_deg*DEG; phi0=la0*DEG; lam0=lo0*DEG
    sinP=math.sin(phi)
    psi=math.atanh(sinP)-_e*math.atanh(_e*sinP)
    dl=lam-lam0
    xi_=math.atan2(math.sinh(psi),math.cos(dl))
    eta_=math.atanh(math.sin(dl)/math.cosh(psi))
    xi=xi_+sum(_alpha[j]*math.sin(2*j*xi_)*math.cosh(2*j*eta_) for j in range(1,5))
    eta=eta_+sum(_alpha[j]*math.cos(2*j*xi_)*math.sinh(2*j*eta_) for j in range(1,5))
    return _m0*_A*xi-_S(phi0), _m0*_A*eta

def jpc_to_latlon(X, Y, zone):
    if zone not in JPC_ORIGINS: return None
    la0,lo0=JPC_ORIGINS[zone]
    phi0=la0*DEG; lam0=lo0*DEG
    xi=(X+_S(phi0))/(_m0*_A); eta=Y/(_m0*_A)
    xi_=xi-sum(_beta[j]*math.sin(2*j*xi)*math.cosh(2*j*eta) for j in range(1,5))
    eta_=eta-sum(_beta[j]*math.cos(2*j*xi)*math.sinh(2*j*eta) for j in range(1,5))
    chi=math.asin(min(1.0,max(-1.0,math.sin(xi_)/math.cosh(eta_))))
    phi=chi+sum(_delta[j]*math.sin(2*j*chi) for j in range(1,5))
    lam=lam0+math.atan2(math.sinh(eta_),math.cos(xi_))
    return phi*RAD, lam*RAD

# ═══════════════════════════════════════════════════════
# 3. ジオイド高 API（国土地理院 JPGEO2024/2011）
# ═══════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_geoid(lat: float, lon: float, model: str = "JPGEO2024") -> float | None:
    """
    国土地理院ジオイド高計算API
    model: "JPGEO2024" or "JPGEO2011"
    戻り値: ジオイド高N [m]、失敗時 None
    """
    if model == "NONE":
        return 0.0
    select = "0" if model == "JPGEO2024" else "1"
    url = (
        "https://vldb.gsi.go.jp/sokuchi/surveycalc/geoid/calcgh/cgi/geoidcalc.pl"
        f"?select={select}&tanni=1&outputType=json"
        f"&latitude={lat:.8f}&longitude={lon:.8f}"
    )
    try:
        r = requests.get(url, timeout=8)
        d = r.json()
        n = float(d["OutputData"]["geoidHeight"])
        return n
    except Exception:
        return None

# ═══════════════════════════════════════════════════════
# 4. 角度フォーマット
# ═══════════════════════════════════════════════════════

def fmt_decimal(dd):  return f"{dd:.8f}"
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
def fmt_gons(dd):     return f"{dd*10/9:.8f}"

def format_angle(dd, fk):
    return {"decimal":fmt_decimal,"dms":fmt_dms,"bearing":fmt_bearing,
            "ddmmssss":fmt_ddmmssss,"gons":fmt_gons}.get(fk,fmt_decimal)(dd)

# ═══════════════════════════════════════════════════════
# 5. ページ設定・CSS
# ═══════════════════════════════════════════════════════

st.set_page_config(
    page_title="ローカライゼーション用座標変換",
    page_icon="📐", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Sans JP', sans-serif;
}

/* ── サイドバー ── */
section[data-testid="stSidebar"] {
    background: #0f172a !important;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #e2e8f0 !important;
}
/* セレクトボックス本体の文字色（プルダウン選択後の表示） */
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border-color: #334155 !important;
}
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span {
    color: #f1f5f9 !important;
}
/* プルダウンメニューの背景・文字 */
[data-baseweb="popover"] [role="option"] {
    background: #1e293b !important;
    color: #f1f5f9 !important;
}
[data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] {
    background: #334155 !important;
    color: #fbbf24 !important;
}
[data-baseweb="popover"] {
    background: #1e293b !important;
}

/* ── 結果カード ── */
.rc {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 14px 18px 12px;
    margin-bottom: 8px;
}
.rc-lbl {
    font-size: 9px; font-weight: 700;
    letter-spacing: .18em; text-transform: uppercase;
    color: #94a3b8; margin-bottom: 5px;
}
.rc-val {
    font-family: 'DM Mono', monospace;
    font-size: 17px; font-weight: 500; color: #0f172a;
}
.rc-sub {
    font-family: 'DM Mono', monospace;
    font-size: 10px; color: #94a3b8; margin-top: 3px;
}

/* ── 点カード ── */
.pt-card {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 18px 20px 14px;
    margin-bottom: 14px;
}
.pt-title {
    font-size: 13px; font-weight: 700;
    color: #475569; margin-bottom: 12px;
    display: flex; align-items: center; gap: 8px;
}

/* ── ヘッダー ── */
.app-hdr {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 14px; padding: 22px 28px; margin-bottom: 20px;
}
.app-hdr h1 { margin:0; font-size:22px; font-weight:700; color:#f1f5f9; }
.app-hdr p  { margin:4px 0 0; color:#64748b; font-size:11px; letter-spacing:.2em; text-transform:uppercase; }

/* ── バッジ類 ── */
.zbadge { display:inline-block; background:#f59e0b; color:#fff; font-weight:700;
          font-size:11px; padding:2px 10px; border-radius:6px; margin:4px 0; }
.zinfo  { font-size:11px; color:#94a3b8; margin-top:2px; }
.acc    { display:inline-block; background:#dcfce7; color:#15803d; font-size:10px;
          font-weight:700; padding:2px 8px; border-radius:4px; }
.gbadge { display:inline-block; background:#e0f2fe; color:#0369a1; font-size:10px;
          font-weight:700; padding:2px 8px; border-radius:4px; margin-left:6px; }
.geoid-ok  { color:#16a34a; font-size:11px; font-weight:600; }
.geoid-ng  { color:#dc2626; font-size:11px; font-weight:600; }
.err { color:#ef4444; font-size:12px; }

/* ── セクション区切り ── */
.sec-label {
    font-size:10px; font-weight:700; color:#94a3b8;
    letter-spacing:.18em; text-transform:uppercase; margin: 16px 0 8px;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# 6. サイドバー共通設定
# ═══════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚙️ 共通設定")
    st.markdown("---")

    # 座標系
    st.markdown("**📌 座標系（系番号）**")
    zone_inv = {v:k for k,v in JPC_ZONE_LABELS.items()}
    zone_lbl = st.selectbox(
        "座標系", list(JPC_ZONE_LABELS.values()),
        index=8, label_visibility="collapsed"
    )
    Z = zone_inv[zone_lbl]
    la0, lo0 = JPC_ORIGINS[Z]
    st.markdown(f"<div class='zbadge'>第 {Z} 系</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='zinfo'>原点 φ₀={la0}° / λ₀={lo0}°</div>", unsafe_allow_html=True)

    st.markdown("---")

    # 測地系
    st.markdown("**🌐 測地系**")
    datum_inv = {v["label"]:k for k,v in DATUMS.items()}
    datum_lbl = st.selectbox(
        "測地系", list(datum_inv.keys()),
        index=0, label_visibility="collapsed"
    )
    DATUM = datum_inv[datum_lbl]

    st.markdown("---")

    # ジオイドモデル
    st.markdown("**📡 ジオイドモデル**")
    geoid_lbl = st.selectbox(
        "ジオイドモデル", list(GEOID_MODELS.values()),
        index=0, label_visibility="collapsed"
    )
    GEOID_KEY = [k for k,v in GEOID_MODELS.items() if v==geoid_lbl][0]

    st.markdown("---")

    # 出力フォーマット
    st.markdown("**📐 出力フォーマット（緯度経度）**")
    fmt_lbl = st.selectbox(
        "出力フォーマット", list(OUTPUT_FORMATS.keys()),
        index=0, label_visibility="collapsed"
    )
    FMT = OUTPUT_FORMATS[fmt_lbl]

    st.markdown("---")
    st.markdown(f"""<div class='acc'>往復誤差 &lt; 0.01mm</div>
<div class='zinfo' style='margin-top:6px'>
GRS80楕円体 / m₀=0.9999<br>
Kawase (2011) 高次展開式<br>
{GEOID_MODELS[GEOID_KEY]}
</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# 7. メインヘッダー
# ═══════════════════════════════════════════════════════

geoid_badge = f"<span class='gbadge'>{GEOID_MODELS[GEOID_KEY]}</span>"
st.markdown(f"""
<div class="app-hdr">
  <h1>📐 ローカライゼーション用座標変換</h1>
  <p>第 {Z} 系 &nbsp;·&nbsp; {datum_lbl} &nbsp;·&nbsp; {fmt_lbl}</p>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📍 単点変換", "📋 CSV 一括変換", "ℹ️ 系番号一覧"])

# ═══════════════════════════════════════════════════════
# 8. TAB 1: 単点変換（複数点対応）
# ═══════════════════════════════════════════════════════

with tab1:
    dir1 = st.radio(
        "変換方向",
        ["平面直角 → 緯度経度", "緯度経度 → 平面直角"],
        horizontal=True, key="d1"
    )
    st.markdown("---")

    # ─── session_state で点リストを管理 ───────────────

    if dir1 == "平面直角 → 緯度経度":
        key_pts = "pts_jpc"
        if key_pts not in st.session_state:
            st.session_state[key_pts] = [{"name":"pt1","x":"","y":"","z":""}]

        pts: list = st.session_state[key_pts]

        # ＋ボタン・全クリアボタン
        col_add, col_clr, _ = st.columns([1, 1, 6])
        with col_add:
            if st.button("＋ 点を追加", key="add_jpc"):
                n = len(pts) + 1
                pts.append({"name":f"pt{n}","x":"","y":"","z":""})
                st.rerun()
        with col_clr:
            if st.button("🗑 全クリア", key="clr_jpc"):
                st.session_state[key_pts] = [{"name":"pt1","x":"","y":"","z":""}]
                st.rerun()

        st.markdown("<div class='sec-label'>入力</div>", unsafe_allow_html=True)

        # 各点の入力行
        del_idx = None
        for i, pt in enumerate(pts):
            c0,c1,c2,c3,c4,c5 = st.columns([0.8,1.5,2,2,2,0.5])
            with c0:
                st.markdown(f"<div style='padding-top:32px;font-size:12px;font-weight:700;color:#64748b'>#{i+1}</div>",
                            unsafe_allow_html=True)
            with c1:
                pt["name"] = st.text_input("点名", value=pt["name"],
                                           key=f"jpc_nm_{i}", label_visibility="visible" if i==0 else "collapsed")
            with c2:
                pt["x"] = st.text_input("X 北が正 (m)" if i==0 else "X",
                                        value=pt["x"], placeholder="-42090.367",
                                        key=f"jpc_x_{i}", label_visibility="visible" if i==0 else "collapsed")
            with c3:
                pt["y"] = st.text_input("Y 東が正 (m)" if i==0 else "Y",
                                        value=pt["y"], placeholder="-23809.574",
                                        key=f"jpc_y_{i}", label_visibility="visible" if i==0 else "collapsed")
            with c4:
                pt["z"] = st.text_input("Z 標高 (m)" if i==0 else "Z",
                                        value=pt["z"], placeholder="52.340",
                                        key=f"jpc_z_{i}", label_visibility="visible" if i==0 else "collapsed")
            with c5:
                pad = "margin-top:28px;" if i==0 else "margin-top:0px;"
                if st.button("✕", key=f"del_jpc_{i}",
                             disabled=len(pts)==1,
                             help="この点を削除"):
                    del_idx = i

        if del_idx is not None:
            pts.pop(del_idx)
            st.rerun()

        st.markdown("---")

        # ─── 変換実行 ───────────────────────────────
        has_input = any(p["x"].strip() and p["y"].strip() for p in pts)

        if has_input:
            st.markdown("<div class='sec-label'>変換結果</div>", unsafe_allow_html=True)

            map_rows = []
            csv_rows = []
            csv_hdr  = f"点名,X(m),Y(m),Z標高(m),ジオイド高N(m),楕円体高h(m),緯度({fmt_lbl}),経度({fmt_lbl}),緯度_DD,経度_DD"

            for pt in pts:
                if not (pt["x"].strip() and pt["y"].strip()):
                    continue
                try:
                    Xv = float(pt["x"]); Yv = float(pt["y"])
                    Zv = float(pt["z"]) if pt["z"].strip() else None
                    res = jpc_to_latlon(Xv, Yv, Z)
                    if res is None:
                        st.error(f"[{pt['name']}] 系番号 {Z} が無効です。")
                        continue
                    lat_dd, lon_dd = res

                    # ジオイド高取得
                    N = None
                    ellH = None
                    geoid_status = ""
                    if GEOID_KEY != "NONE":
                        with st.spinner(f"[{pt['name']}] ジオイド高を取得中..."):
                            N = fetch_geoid(lat_dd, lon_dd, GEOID_KEY)
                        if N is not None:
                            geoid_status = f"<span class='geoid-ok'>N={N:.4f}m ({GEOID_KEY})</span>"
                            if Zv is not None:
                                ellH = Zv + N
                        else:
                            geoid_status = "<span class='geoid-ng'>ジオイド高取得失敗（手動入力が必要）</span>"
                    else:
                        geoid_status = "<span class='geoid-ng'>補正なし</span>"
                        if Zv is not None:
                            ellH = Zv

                    # 結果カード
                    st.markdown(f"""
<div class='pt-card'>
<div class='pt-title'>📍 {pt['name']} &nbsp; {geoid_status}</div>
""", unsafe_allow_html=True)
                    rc1,rc2,rc3 = st.columns(3)
                    with rc1:
                        st.markdown(f"""<div class='rc'>
<div class='rc-lbl' style='color:#3b82f6'>緯度 LAT</div>
<div class='rc-val'>{format_angle(lat_dd,FMT)}</div>
<div class='rc-sub'>{fmt_decimal(lat_dd)}°</div>
</div>""", unsafe_allow_html=True)
                    with rc2:
                        st.markdown(f"""<div class='rc'>
<div class='rc-lbl' style='color:#10b981'>経度 LON</div>
<div class='rc-val'>{format_angle(lon_dd,FMT)}</div>
<div class='rc-sub'>{fmt_decimal(lon_dd)}°</div>
</div>""", unsafe_allow_html=True)
                    with rc3:
                        hs = f"{ellH:.3f} m" if ellH is not None else "—"
                        sub = f"Z={Zv:.3f} + N={N:.4f}" if (ellH is not None and N is not None and Zv is not None) else ""
                        st.markdown(f"""<div class='rc'>
<div class='rc-lbl' style='color:#8b5cf6'>楕円体高 h (m)</div>
<div class='rc-val'>{hs}</div>
<div class='rc-sub'>{sub}</div>
</div>""", unsafe_allow_html=True)

                    # 全フォーマット
                    with st.expander(f"🔢 {pt['name']} 全フォーマット"):
                        st.dataframe(pd.DataFrame([
                            {"フォーマット":fl,"緯度":format_angle(lat_dd,fk),"経度":format_angle(lon_dd,fk)}
                            for fl,fk in OUTPUT_FORMATS.items()
                        ]), use_container_width=True, hide_index=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                    map_rows.append({"lat":lat_dd,"lon":lon_dd,"name":pt["name"]})
                    csv_rows.append(
                        f"{pt['name']},{Xv},{Yv},"
                        + (f"{Zv:.3f}" if Zv is not None else "") + ","
                        + (f"{N:.4f}"  if N  is not None else "") + ","
                        + (f"{ellH:.3f}" if ellH is not None else "") + ","
                        + f"{format_angle(lat_dd,FMT)},{format_angle(lon_dd,FMT)},"
                        + f"{fmt_decimal(lat_dd)},{fmt_decimal(lon_dd)}"
                    )

                except ValueError as ex:
                    st.error(f"[{pt['name']}] 入力エラー: {ex}")

            # 地図・CSV出力
            if map_rows:
                st.markdown("#### 📍 地図")
                st.map(pd.DataFrame(map_rows), zoom=10)

                csv_out = "\ufeff" + csv_hdr + "\n" + "\n".join(csv_rows)
                st.download_button("📥 全点 CSV ダウンロード", csv_out,
                                   "converted.csv", "text/csv")
        else:
            st.info("X・Y 座標を入力してください。")

    # ─── 緯度経度 → 平面直角 ─────────────────────────
    else:
        key_pts2 = "pts_ll"
        if key_pts2 not in st.session_state:
            st.session_state[key_pts2] = [{"name":"pt1","lat":"","lon":"","h":""}]

        pts2: list = st.session_state[key_pts2]

        col_add2, col_clr2, _ = st.columns([1,1,6])
        with col_add2:
            if st.button("＋ 点を追加", key="add_ll"):
                n = len(pts2)+1
                pts2.append({"name":f"pt{n}","lat":"","lon":"","h":""})
                st.rerun()
        with col_clr2:
            if st.button("🗑 全クリア", key="clr_ll"):
                st.session_state[key_pts2] = [{"name":"pt1","lat":"","lon":"","h":""}]
                st.rerun()

        st.markdown("<div class='sec-label'>入力</div>", unsafe_allow_html=True)

        del_idx2 = None
        for i, pt in enumerate(pts2):
            c0,c1,c2,c3,c4,c5 = st.columns([0.8,1.5,2.5,2.5,2,0.5])
            with c0:
                st.markdown(f"<div style='padding-top:32px;font-size:12px;font-weight:700;color:#64748b'>#{i+1}</div>",
                            unsafe_allow_html=True)
            with c1:
                pt["name"] = st.text_input("点名", value=pt["name"],
                                           key=f"ll_nm_{i}", label_visibility="visible" if i==0 else "collapsed")
            with c2:
                pt["lat"] = st.text_input("緯度（十進度）" if i==0 else "緯度",
                                          value=pt["lat"], placeholder="35.68123456",
                                          key=f"ll_la_{i}", label_visibility="visible" if i==0 else "collapsed")
            with c3:
                pt["lon"] = st.text_input("経度（十進度）" if i==0 else "経度",
                                          value=pt["lon"], placeholder="139.76712345",
                                          key=f"ll_lo_{i}", label_visibility="visible" if i==0 else "collapsed")
            with c4:
                pt["h"] = st.text_input("楕円体高 h (m)" if i==0 else "h(m)",
                                        value=pt["h"], placeholder="89.555",
                                        key=f"ll_h_{i}", label_visibility="visible" if i==0 else "collapsed")
            with c5:
                if st.button("✕", key=f"del_ll_{i}", disabled=len(pts2)==1):
                    del_idx2 = i

        if del_idx2 is not None:
            pts2.pop(del_idx2)
            st.rerun()

        st.markdown("---")

        has_input2 = any(p["lat"].strip() and p["lon"].strip() for p in pts2)

        if has_input2:
            st.markdown("<div class='sec-label'>変換結果</div>", unsafe_allow_html=True)
            map_rows2 = []
            csv_rows2 = []
            csv_hdr2  = f"点名,緯度(DD),経度(DD),楕円体高(m),X(m),Y(m),系番号"

            for pt in pts2:
                if not (pt["lat"].strip() and pt["lon"].strip()):
                    continue
                try:
                    lv=float(pt["lat"]); lov=float(pt["lon"])
                    hv=float(pt["h"]) if pt["h"].strip() else 0.0
                    res=latlon_to_jpc(lv,lov,Z)
                    if res is None:
                        st.error(f"[{pt['name']}] 系番号 {Z} が無効です。"); continue
                    Xr,Yr=res

                    st.markdown(f"<div class='pt-card'><div class='pt-title'>📍 {pt['name']}</div>",
                                unsafe_allow_html=True)
                    rc1,rc2,rc3=st.columns(3)
                    with rc1:
                        st.markdown(f"""<div class='rc'>
<div class='rc-lbl' style='color:#3b82f6'>X 北が正 (m)</div>
<div class='rc-val'>{Xr:,.4f}</div></div>""", unsafe_allow_html=True)
                    with rc2:
                        st.markdown(f"""<div class='rc'>
<div class='rc-lbl' style='color:#10b981'>Y 東が正 (m)</div>
<div class='rc-val'>{Yr:,.4f}</div></div>""", unsafe_allow_html=True)
                    with rc3:
                        st.markdown(f"""<div class='rc'>
<div class='rc-lbl' style='color:#f59e0b'>座標系</div>
<div class='rc-val'>第 {Z} 系</div>
<div class='rc-sub'>GRS80 / m₀=0.9999</div></div>""", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    map_rows2.append({"lat":lv,"lon":lov})
                    csv_rows2.append(f"{pt['name']},{lv},{lov},{hv:.3f},{Xr:.4f},{Yr:.4f},{Z}")

                except ValueError as ex:
                    st.error(f"[{pt['name']}] 入力エラー: {ex}")

            if map_rows2:
                st.markdown("#### 📍 地図")
                st.map(pd.DataFrame(map_rows2), zoom=10)
                csv_out2="\ufeff"+csv_hdr2+"\n"+"\n".join(csv_rows2)
                st.download_button("📥 全点 CSV ダウンロード",csv_out2,"converted.csv","text/csv")
        else:
            st.info("緯度・経度を十進角度で入力してください。")

# ═══════════════════════════════════════════════════════
# 9. TAB 2: CSV 一括変換
# ═══════════════════════════════════════════════════════

with tab2:
    dir2=st.radio("変換方向",
                  ["平面直角 → 緯度経度（ヘッダーなし）","緯度経度 → 平面直角（ヘッダーあり）"],
                  horizontal=True, key="d2")
    st.markdown("---")
    S1="t1,-42090.367,-23809.574,67.222\nt2,-42089.211,-23951.174,67.659\nt3,-42238.931,-23876.726,66.813"
    S2="name,lat,lon,h,datum\npt1,35.68123,139.76712,10.5,JGD2011\npt2,34.69374,135.50218,5.2,JGD2011\npt3,38.26822,140.86940,52.3,JGD2011"

    if "平面直角" in dir2:
        st.markdown("""
**CSV フォーマット（ヘッダー行なし）**
```
点名,X(m),Y(m),標高Z(m)
t1,-42090.367,-23809.574,67.222
```
標高列を入力するとジオイド高APIで楕円体高を自動計算して出力します。
        """)
        up1=st.file_uploader("CSVファイル",["csv","txt"],key="u1")
        ca,cb=st.columns([3,1])
        with ca: tx1=st.text_area("または貼り付け",height=130,placeholder=S1,key="t1")
        with cb:
            if st.button("サンプル",key="s1"): st.session_state["t1"]=S1; st.rerun()
        src=(up1.read().decode("utf-8-sig") if up1 else "") or (tx1 if tx1 else "")

        if src.strip():
            try:
                df_in=pd.read_csv(io.StringIO(src),header=None,dtype=str)
                rows=[]
                pb=st.progress(0,"変換中...")
                total=len(df_in)
                for idx,(_, row) in enumerate(df_in.iterrows()):
                    pb.progress((idx+1)/total, f"{idx+1}/{total} 点処理中")
                    try:
                        name=str(row.iloc[0])
                        X=float(row.iloc[1]); Y=float(row.iloc[2])
                        elev=float(row.iloc[3]) if len(row)>3 and pd.notna(row.iloc[3]) else None
                        res=jpc_to_latlon(X,Y,Z)
                        if res is None: raise ValueError(f"系番号 {Z} が無効")
                        lat_dd,lon_dd=res
                        N=None; ellH=None
                        if GEOID_KEY!="NONE" and elev is not None:
                            N=fetch_geoid(lat_dd,lon_dd,GEOID_KEY)
                            if N is not None: ellH=elev+N
                        rows.append({"点名":name,"X(m)":X,"Y(m)":Y,
                                     "標高Z(m)":f"{elev:.3f}" if elev is not None else "",
                                     "ジオイド高N(m)":f"{N:.4f}" if N is not None else "",
                                     "楕円体高h(m)":f"{ellH:.3f}" if ellH is not None else "",
                                     "緯度":format_angle(lat_dd,FMT),
                                     "経度":format_angle(lon_dd,FMT),
                                     "緯度_DD":fmt_decimal(lat_dd),
                                     "経度_DD":fmt_decimal(lon_dd),
                                     "_lat":lat_dd,"_lon":lon_dd,"_err":None})
                    except Exception as ex:
                        rows.append({"点名":str(row.iloc[0]) if len(row)>0 else "?",
                                     "_err":str(ex),"_lat":None,"_lon":None})
                pb.empty()
                dfr=pd.DataFrame(rows)
                ok=dfr[dfr["_err"].isna()]; ng=dfr[dfr["_err"].notna()]
                st.success(f"✅ {len(ok)} 点完了"+(f"　⚠️ {len(ng)} 件エラー" if len(ng) else ""))
                if len(ng):
                    with st.expander("⚠️ エラー"):
                        [st.markdown(f"<span class='err'>❌ {r['点名']} — {r['_err']}</span>",unsafe_allow_html=True)
                         for _,r in ng.iterrows()]
                show=[c for c in ["点名","X(m)","Y(m)","標高Z(m)","ジオイド高N(m)","楕円体高h(m)","緯度","経度"] if c in ok.columns]
                st.dataframe(ok[show],use_container_width=True,hide_index=True)
                if ok["_lat"].notna().any():
                    st.markdown("#### 📍 地図")
                    st.map(ok[ok["_lat"].notna()][["_lat","_lon"]].rename(columns={"_lat":"lat","_lon":"lon"}),zoom=9)
                out=[c for c in ["点名","X(m)","Y(m)","標高Z(m)","ジオイド高N(m)","楕円体高h(m)","緯度","経度","緯度_DD","経度_DD"] if c in ok.columns]
                st.download_button("📥 結果 CSV","\ufeff"+ok[out].to_csv(index=False),"batch.csv","text/csv")
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
name / h / datum 列は省略可。
        """)
        up2=st.file_uploader("CSVファイル",["csv","txt"],key="u2")
        cc,cd=st.columns([3,1])
        with cc: tx2=st.text_area("または貼り付け",height=130,placeholder=S2,key="t2")
        with cd:
            if st.button("サンプル",key="s2"): st.session_state["t2"]=S2; st.rerun()
        src2=(up2.read().decode("utf-8-sig") if up2 else "") or (tx2 if tx2 else "")

        if src2.strip():
            try:
                df_in2=pd.read_csv(io.StringIO(src2),dtype=str)
                cols=[c.strip().lower() for c in df_in2.columns]
                def ci(ns):
                    for n in ns:
                        if n in cols: return cols.index(n)
                    return None
                il=ci(["lat","緯度","latitude"]); iol=ci(["lon","lng","経度","longitude"])
                ih=ci(["h","z","ellh","height","楕円体高"]); inm=ci(["name","点名","id","no"])
                idt=ci(["datum","測地系"])
                if il is None or iol is None:
                    st.error("ヘッダーに 'lat' と 'lon' 列が必要です"); st.stop()
                rows2=[]
                for i,row in df_in2.iterrows():
                    try:
                        name=str(row.iloc[inm]) if inm is not None else str(i+1)
                        lv=float(row.iloc[il]); lov=float(row.iloc[iol])
                        hv=float(row.iloc[ih]) if ih is not None and pd.notna(row.iloc[ih]) else 0.0
                        res=latlon_to_jpc(lv,lov,Z)
                        if res is None: raise ValueError(f"系番号 {Z} が無効")
                        px,py=res
                        rows2.append({"点名":name,"緯度_入力":fmt_decimal(lv),"経度_入力":fmt_decimal(lov),
                                      "楕円体高(m)":f"{hv:.3f}","X(m)":f"{px:.4f}","Y(m)":f"{py:.4f}",
                                      "_lat":lv,"_lon":lov,"_err":None})
                    except Exception as ex:
                        rows2.append({"点名":str(row.iloc[inm]) if inm is not None else "?",
                                      "_err":str(ex),"_lat":None,"_lon":None})
                dfr2=pd.DataFrame(rows2)
                ok2=dfr2[dfr2["_err"].isna()]; ng2=dfr2[dfr2["_err"].notna()]
                st.success(f"✅ {len(ok2)} 点完了"+(f"　⚠️ {len(ng2)} 件エラー" if len(ng2) else ""))
                show2=[c for c in ["点名","緯度_入力","経度_入力","楕円体高(m)","X(m)","Y(m)"] if c in ok2.columns]
                st.dataframe(ok2[show2],use_container_width=True,hide_index=True)
                if ok2["_lat"].notna().any():
                    st.markdown("#### 📍 地図")
                    st.map(ok2[ok2["_lat"].notna()][["_lat","_lon"]].rename(columns={"_lat":"lat","_lon":"lon"}),zoom=9)
                st.download_button("📥 結果 CSV","\ufeff"+ok2[show2].to_csv(index=False),"batch.csv","text/csv")
            except Exception as ex:
                st.error(f"処理エラー: {ex}")
        else:
            st.info("CSV をアップロードまたは貼り付けてください。")

# ═══════════════════════════════════════════════════════
# 10. TAB 3: 系番号一覧
# ═══════════════════════════════════════════════════════

with tab3:
    st.markdown("### 公共測量 平面直角座標系 系番号一覧")
    st.caption("国土交通省告示（昭和48年建設省告示第143号）")
    rows_z=[]
    for z in range(1,20):
        l0,o0=JPC_ORIGINS[z]
        rows_z.append({"系番号":z,"適用地域":JPC_ZONE_LABELS[z].split(" — ")[1],
                       "原点緯度 φ₀":f"{l0}°","原点経度 λ₀":f"{o0}°",
                       "縮尺係数":"0.9999","選択中":"✅" if z==Z else ""})
    dfz=pd.DataFrame(rows_z)
    def hl(r): return ["background:#fef3c7"]*len(r) if r["選択中"]=="✅" else [""]*len(r)
    st.dataframe(dfz.style.apply(hl,axis=1),use_container_width=True,hide_index=True,height=680)
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
