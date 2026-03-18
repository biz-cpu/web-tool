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

# ─────────────────────────────────────────────────────────
# 写真OCR：Claude Vision API で座標値を読み取る
# ─────────────────────────────────────────────────────────

def ocr_image_to_points(image_bytes: bytes, mime_type: str, mode: str) -> list[dict]:
    """
    写真から座標データを読み取る（Anthropic SDK 使用）。
    APIキーは st.secrets["ANTHROPIC_API_KEY"] または環境変数 ANTHROPIC_API_KEY から取得。
    """
    import base64, json, re as _re

    b64 = base64.standard_b64encode(image_bytes).decode()

    if mode == "jpc":
        field_desc = "点名(name), X座標(x), Y座標(y), Z標高(z)"
        json_schema = '{"points":[{"name":"","x":"","y":"","z":""}]}'
        hint = "平面直角座標（X北が正・Y東が正・Z標高、単位m）"
    else:
        field_desc = "点名(name), 緯度(lat), 経度(lon), 楕円体高(h)"
        json_schema = '{"points":[{"name":"","lat":"","lon":"","h":""}]}'
        hint = "緯度・経度・楕円体高（h）"

    prompt = f"""この画像から測量・GNSSデータの数値を読み取ってください。
読み取り対象: {hint}
各点の {field_desc} を抽出してください。

- 数値は画像に表示されている通りに読み取る（単位は不要、数値のみ）
- 点名が不明な場合は "pt1", "pt2"... と連番
- 空欄・不明な値は "" (空文字)
- 複数点ある場合はすべて抽出

以下のJSONフォーマットのみで回答してください（前後の説明不要）:
{json_schema}
"""

    # APIキー取得: secrets → 環境変数の順に試みる
    import os
    api_key = ""
    try:
        # st.secrets は dict-like だが .get() が使えない場合があるため [] でアクセス
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "APIキーが未設定です。\n"
            "Streamlit Cloud の Settings → Secrets に\n"
            "ANTHROPIC_API_KEY = \"sk-ant-...\"\n"
            "を追加してください。"
        )

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        raw = msg.content[0].text.strip()
        # JSON 部分を抽出（コードブロック等を除去）
        m = _re.search(r'\{[^{}]*"points"[^{}]*\[.*?\][^{}]*\}', raw, _re.DOTALL)
        if not m:
            m = _re.search(r'\{.*\}', raw, _re.DOTALL)
        if m:
            data = json.loads(m.group())
            return data.get("points", [])
        raise ValueError(f"JSONが返されませんでした: {raw[:200]}")
    except ImportError:
        raise ValueError(
            "anthropic ライブラリが見つかりません。\n"
            "requirements.txt に anthropic を追加してください。"
        )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"API呼び出しエラー: {type(e).__name__}: {e}")
    return []


def camera_ocr_button(mode: str, row_idx: int):
    """列内に 📷 ボタンだけ描画。タップで open フラグを反転。"""
    open_key = f"cam_open_{mode}_{row_idx}"
    if open_key not in st.session_state:
        st.session_state[open_key] = False
    if st.button("📷", key=f"cam_btn_{mode}_{row_idx}",
                 help="カメラで撮影して読み取る"):
        st.session_state[open_key] = not st.session_state[open_key]
        st.rerun()


def camera_ocr_panel(mode: str, row_idx: int):
    """列の外（フル幅）にカメラ撮影 + OCR パネルを展開する。"""
    open_key = f"cam_open_{mode}_{row_idx}"
    cam_key  = f"cam_{mode}_{row_idx}"
    if not st.session_state.get(open_key, False):
        return

    fields  = ("name","x","y","z") if mode == "jpc" else ("name","lat","lon","h")
    pts_key = {"jpc":"pts_jpc","ll":"pts_ll","cvt":"pts_cvt"}[mode]
    lbl_fields = "点名・X・Y・Z" if mode == "jpc" else "点名・緯度・経度・楕円体高"

    st.markdown(
        f"<div style='background:#f0f9ff;border:1px solid #bae6fd;"
        f"border-radius:10px;padding:10px 14px;margin:4px 0 8px'>"
        f"<b>📷 #{row_idx+1} をカメラで読み取る</b>"
        f"<span style='font-size:11px;color:#64748b;margin-left:8px'>{lbl_fields}</span>"
        f"</div>",
        unsafe_allow_html=True
    )
    img = st.camera_input(
        f"#{row_idx+1} 撮影",
        key=cam_key,
        label_visibility="collapsed",
    )
    if img is not None:
        col_img, col_act = st.columns([3, 1])
        with col_img:
            st.image(img, use_container_width=True)
        with col_act:
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            if st.button("🔍 読取", key=f"ocr_run_{mode}_{row_idx}",
                         use_container_width=True):
                with st.spinner("AI解析中..."):
                    try:
                        pts_data = ocr_image_to_points(img.getvalue(), "image/jpeg", mode)
                        if not pts_data:
                            st.error("読み取れませんでした。数値が写った部分を正面から撮り直してください。")
                        else:
                            p = pts_data[0]
                            while len(st.session_state[pts_key]) <= row_idx:
                                st.session_state[pts_key].append(
                                    {"name":"","x":"","y":"","z":""} if mode=="jpc"
                                    else {"name":"","lat":"","lon":"","h":""}
                                )
                            for f in fields:
                                val = str(p.get(f, "")).strip()
                                st.session_state[pts_key][row_idx][f] = val
                                st.session_state[f"{mode}_{f}_{row_idx}"] = val
                            st.session_state[open_key] = False
                            st.success(f"✅ #{row_idx+1} 読み取り完了")
                            st.rerun()
                    except ValueError as e:
                        st.error(str(e))
            if st.button("✕ 閉じる", key=f"cam_close_{mode}_{row_idx}",
                         use_container_width=True):
                st.session_state[open_key] = False
                st.rerun()


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
    import re as _re
    s = val.strip()
    if not s:
        raise ValueError("空欄")

    # 1. bearing: 先頭 N or S
    if _re.match(r"^[NSns]", s):
        return parse_angle(s, "bearing"), "bearing"

    # 2. dms: Unicode度分秒記号（° ′ ″）を含む ← 半角英字 d/m/s は除外して誤判定防止
    if any(c in s for c in ("°", "′", "″", "°", "′", "″")):
        return parse_angle(s, "dms"), "dms"

    # 3. gons: 末尾に gon/gons/g/gr（純数値以外）
    if _re.search(r"(?i)(gons?|gr?)\s*$", s):
        return parse_angle(s, "gons"), "gons"

    # 4. ddmmssss: DD.MMSSSSSS（小数6桁以上で分・秒整数が有効範囲）
    m = _re.match(r"^(-?)(\d{1,3})\.(\d{6,})$", s)
    if m:
        deg_int = int(m.group(2))
        dec_str = m.group(3).ljust(10, "0")
        mm_val  = int(dec_str[:2])
        ss_int  = int(dec_str[2:4])
        if deg_int <= 180 and mm_val <= 59 and ss_int <= 59:
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
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;500;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Noto Sans JP', sans-serif; }

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
  background: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMjAwIiBoZWlnaHQ9IjQwMCIgdmlld0JveD0iMCAwIDEyMDAgNDAwIj4KICA8ZGVmcz4KICAgIDxyYWRpYWxHcmFkaWVudCBpZD0ic3BhY2UiIGN4PSI1MCUiIGN5PSI1MCUiIHI9IjcwJSI+CiAgICAgIDxzdG9wIG9mZnNldD0iMCUiIHN0b3AtY29sb3I9IiMwYTE2MjgiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSI2MCUiIHN0b3AtY29sb3I9IiMwNTBkMWEiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjMDIwODEwIi8+CiAgICA8L3JhZGlhbEdyYWRpZW50PgogICAgPHJhZGlhbEdyYWRpZW50IGlkPSJlYXJ0aCIgY3g9IjQwJSIgY3k9IjM1JSIgcj0iNjAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzFhNmI5ZSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjQwJSIgc3RvcC1jb2xvcj0iIzBmNGY3YSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjcwJSIgc3RvcC1jb2xvcj0iIzBhM2E1ZSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiMwNjFkMzAiLz4KICAgIDwvcmFkaWFsR3JhZGllbnQ+CiAgICA8cmFkaWFsR3JhZGllbnQgaWQ9ImVhcnRoR2xvdyIgY3g9IjQwJSIgY3k9IjM1JSIgcj0iNjAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSI2MCUiIHN0b3AtY29sb3I9InRyYW5zcGFyZW50Ii8+CiAgICAgIDxzdG9wIG9mZnNldD0iMTAwJSIgc3RvcC1jb2xvcj0iIzFlODhlNTgwIi8+CiAgICA8L3JhZGlhbEdyYWRpZW50PgogICAgPHJhZGlhbEdyYWRpZW50IGlkPSJzYXRHbG93IiBjeD0iNTAlIiBjeT0iNTAlIiByPSI1MCUiPgogICAgICA8c3RvcCBvZmZzZXQ9IjAlIiBzdG9wLWNvbG9yPSIjNjBhNWZhIiBzdG9wLW9wYWNpdHk9IjAuOSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiM2MGE1ZmEiIHN0b3Atb3BhY2l0eT0iMCIvPgogICAgPC9yYWRpYWxHcmFkaWVudD4KICAgIDxmaWx0ZXIgaWQ9Imdsb3ciPgogICAgICA8ZmVHYXVzc2lhbkJsdXIgc3RkRGV2aWF0aW9uPSIyIiByZXN1bHQ9ImJsdXIiLz4KICAgICAgPGZlTWVyZ2U+PGZlTWVyZ2VOb2RlIGluPSJibHVyIi8+PGZlTWVyZ2VOb2RlIGluPSJTb3VyY2VHcmFwaGljIi8+PC9mZU1lcmdlPgogICAgPC9maWx0ZXI+CiAgICA8ZmlsdGVyIGlkPSJzb2Z0R2xvdyI+CiAgICAgIDxmZUdhdXNzaWFuQmx1ciBzdGREZXZpYXRpb249IjQiIHJlc3VsdD0iYmx1ciIvPgogICAgICA8ZmVNZXJnZT48ZmVNZXJnZU5vZGUgaW49ImJsdXIiLz48ZmVNZXJnZU5vZGUgaW49IlNvdXJjZUdyYXBoaWMiLz48L2ZlTWVyZ2U+CiAgICA8L2ZpbHRlcj4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0ib3JiaXQxIiB4MT0iMCUiIHkxPSIwJSIgeDI9IjEwMCUiIHkyPSIxMDAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzYwYTVmYSIgc3RvcC1vcGFjaXR5PSIwIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iNTAlIiBzdG9wLWNvbG9yPSIjNjBhNWZhIiBzdG9wLW9wYWNpdHk9IjAuNSIvPgogICAgICA8c3RvcCBvZmZzZXQ9IjEwMCUiIHN0b3AtY29sb3I9IiM2MGE1ZmEiIHN0b3Atb3BhY2l0eT0iMCIvPgogICAgPC9saW5lYXJHcmFkaWVudD4KICAgIDxsaW5lYXJHcmFkaWVudCBpZD0ib3JiaXQyIiB4MT0iMTAwJSIgeTE9IjAlIiB4Mj0iMCUiIHkyPSIxMDAlIj4KICAgICAgPHN0b3Agb2Zmc2V0PSIwJSIgc3RvcC1jb2xvcj0iIzM0ZDM5OSIgc3RvcC1vcGFjaXR5PSIwIi8+CiAgICAgIDxzdG9wIG9mZnNldD0iNTAlIiBzdG9wLWNvbG9yPSIjMzRkMzk5IiBzdG9wLW9wYWNpdHk9IjAuMzUiLz4KICAgICAgPHN0b3Agb2Zmc2V0PSIxMDAlIiBzdG9wLWNvbG9yPSIjMzRkMzk5IiBzdG9wLW9wYWNpdHk9IjAiLz4KICAgIDwvbGluZWFyR3JhZGllbnQ+CiAgICA8Y2xpcFBhdGggaWQ9ImNsaXAiPjxyZWN0IHdpZHRoPSIxMjAwIiBoZWlnaHQ9IjQwMCIvPjwvY2xpcFBhdGg+CiAgPC9kZWZzPgoKICA8IS0tIOWuh+WumeiDjOaZryAtLT4KICA8cmVjdCB3aWR0aD0iMTIwMCIgaGVpZ2h0PSI0MDAiIGZpbGw9InVybCgjc3BhY2UpIi8+CgogIDwhLS0g5pif77yI5bCP44GV44GE44KC44Gu77yJIC0tPgogIDxnIGNsaXAtcGF0aD0idXJsKCNjbGlwKSIgb3BhY2l0eT0iMC45Ij4KICAgIDwhLS0g5piO44KL44GE5pifIC0tPgogICAgPGNpcmNsZSBjeD0iNDUiIGN5PSIyMiIgcj0iMS4yIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjk1Ii8+CiAgICA8Y2lyY2xlIGN4PSIxMjAiIGN5PSI2NyIgcj0iMC44IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjciLz4KICAgIDxjaXJjbGUgY3g9IjE5OCIgY3k9IjE1IiByPSIxLjAiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuODUiLz4KICAgIDxjaXJjbGUgY3g9IjI2NyIgY3k9Ijg4IiByPSIwLjciIGZpbGw9IiNlOGY0ZmQiIG9wYWNpdHk9IjAuNiIvPgogICAgPGNpcmNsZSBjeD0iMzQ1IiBjeT0iMzQiIHI9IjEuMSIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC45Ii8+CiAgICA8Y2lyY2xlIGN4PSI0MjMiIGN5PSIxOCIgcj0iMC45IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjc1Ii8+CiAgICA8Y2lyY2xlIGN4PSI1MTIiIGN5PSI1NSIgcj0iMS4zIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjk1Ii8+CiAgICA8Y2lyY2xlIGN4PSI1NzgiIGN5PSIyOCIgcj0iMC44IiBmaWxsPSIjY2ZlOGZmIiBvcGFjaXR5PSIwLjciLz4KICAgIDxjaXJjbGUgY3g9IjY0NSIgY3k9IjcyIiByPSIxLjAiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuOCIvPgogICAgPGNpcmNsZSBjeD0iNzIzIiBjeT0iMjAiIHI9IjAuOSIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC44NSIvPgogICAgPGNpcmNsZSBjeD0iODEyIiBjeT0iNDUiIHI9IjEuMiIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC45Ii8+CiAgICA8Y2lyY2xlIGN4PSI4NzYiIGN5PSIxNSIgcj0iMC43IiBmaWxsPSIjZThmNGZkIiBvcGFjaXR5PSIwLjY1Ii8+CiAgICA8Y2lyY2xlIGN4PSI5MzQiIGN5PSI2MCIgcj0iMS4xIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjg4Ii8+CiAgICA8Y2lyY2xlIGN4PSIxMDIzIiBjeT0iMzAiIHI9IjAuOCIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC43MiIvPgogICAgPGNpcmNsZSBjeD0iMTA5OCIgY3k9IjUwIiByPSIxLjAiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuOCIvPgogICAgPGNpcmNsZSBjeD0iMTE1NiIgY3k9IjI1IiByPSIwLjkiIGZpbGw9IiNjZmU4ZmYiIG9wYWNpdHk9IjAuNzUiLz4KICAgIDwhLS0g5Lit5q61IC0tPgogICAgPGNpcmNsZSBjeD0iNzgiIGN5PSIxNDUiIHI9IjAuOSIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC43Ii8+CiAgICA8Y2lyY2xlIGN4PSIxNTYiIGN5PSIxNzgiIHI9IjEuMSIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC44NSIvPgogICAgPGNpcmNsZSBjeD0iMjM0IiBjeT0iMTMwIiByPSIwLjciIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNiIvPgogICAgPGNpcmNsZSBjeD0iMzEyIiBjeT0iMTY1IiByPSIwLjgiIGZpbGw9IiNlOGY0ZmQiIG9wYWNpdHk9IjAuNzIiLz4KICAgIDxjaXJjbGUgY3g9IjM4OSIgY3k9IjE0OCIgcj0iMS4wIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjgiLz4KICAgIDxjaXJjbGUgY3g9IjQ2NyIgY3k9IjE5MiIgcj0iMC45IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjY4Ii8+CiAgICA8Y2lyY2xlIGN4PSI1NTYiIGN5PSIxNDAiIHI9IjEuMiIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC45Ii8+CiAgICA8Y2lyY2xlIGN4PSI2MzQiIGN5PSIxNzUiIHI9IjAuOCIgZmlsbD0iI2NmZThmZiIgb3BhY2l0eT0iMC42MiIvPgogICAgPGNpcmNsZSBjeD0iNzAwIiBjeT0iMTU1IiByPSIxLjAiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNzgiLz4KICAgIDxjaXJjbGUgY3g9Ijc5MCIgY3k9IjE4NSIgcj0iMC43IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjY1Ii8+CiAgICA8Y2lyY2xlIGN4PSI4NTYiIGN5PSIxNDUiIHI9IjEuMSIgZmlsbD0iI2ZmZiIgb3BhY2l0eT0iMC44NyIvPgogICAgPGNpcmNsZSBjeD0iOTQ1IiBjeT0iMTcwIiByPSIwLjkiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNzMiLz4KICAgIDxjaXJjbGUgY3g9IjEwMzQiIGN5PSIxNTIiIHI9IjAuOCIgZmlsbD0iI2U4ZjRmZCIgb3BhY2l0eT0iMC42NyIvPgogICAgPGNpcmNsZSBjeD0iMTExMiIgY3k9IjEzNSIgcj0iMS4wIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjgyIi8+CiAgICA8Y2lyY2xlIGN4PSIxMTc4IiBjeT0iMTYyIiByPSIwLjkiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNyIvPgogICAgPCEtLSDkuIvmrrUgLS0+CiAgICA8Y2lyY2xlIGN4PSI1NiIgY3k9IjI4NSIgcj0iMC44IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjYiLz4KICAgIDxjaXJjbGUgY3g9IjE2NyIgY3k9IjMxMiIgcj0iMS4wIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjcyIi8+CiAgICA8Y2lyY2xlIGN4PSIyNzgiIGN5PSIyNzAiIHI9IjAuOSIgZmlsbD0iI2NmZThmZiIgb3BhY2l0eT0iMC42NSIvPgogICAgPGNpcmNsZSBjeD0iMzg5IiBjeT0iMzQwIiByPSIwLjciIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNTUiLz4KICAgIDxjaXJjbGUgY3g9IjUwMCIgY3k9IjI5OCIgcj0iMS4xIiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjgiLz4KICAgIDxjaXJjbGUgY3g9IjYxMSIgY3k9IjMyNSIgcj0iMC44IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjY4Ii8+CiAgICA8Y2lyY2xlIGN4PSI3MjIiIGN5PSIyNzgiIHI9IjEuMCIgZmlsbD0iI2U4ZjRmZCIgb3BhY2l0eT0iMC43NSIvPgogICAgPGNpcmNsZSBjeD0iODMzIiBjeT0iMzYwIiByPSIwLjkiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNjIiLz4KICAgIDxjaXJjbGUgY3g9Ijk0NCIgY3k9IjI5MCIgcj0iMC43IiBmaWxsPSIjZmZmIiBvcGFjaXR5PSIwLjU4Ii8+CiAgICA8Y2lyY2xlIGN4PSIxMDU1IiBjeT0iMzMyIiByPSIxLjEiIGZpbGw9IiNmZmYiIG9wYWNpdHk9IjAuNzgiLz4KICAgIDxjaXJjbGUgY3g9IjExNDQiIGN5PSIzMDUiIHI9IjAuOCIgZmlsbD0iI2NmZThmZiIgb3BhY2l0eT0iMC43Ii8+CiAgPC9nPgoKICA8IS0tIOWcsOeQg++8iOWPs+S4i+OBruabsumdou+8iSAtLT4KICA8Y2lyY2xlIGN4PSIxMDUwIiBjeT0iNTIwIiByPSIzMjAiIGZpbGw9InVybCgjZWFydGgpIiBvcGFjaXR5PSIwLjg1Ii8+CiAgPGNpcmNsZSBjeD0iMTA1MCIgY3k9IjUyMCIgcj0iMzIwIiBmaWxsPSJ1cmwoI2VhcnRoR2xvdykiIG9wYWNpdHk9IjAuNSIvPgogIDwhLS0g5aSn5rCX5YWJIC0tPgogIDxjaXJjbGUgY3g9IjEwNTAiIGN5PSI1MjAiIHI9IjMzMiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMWU4OGU1IiBzdHJva2Utd2lkdGg9IjUiIG9wYWNpdHk9IjAuMTUiLz4KICA8Y2lyY2xlIGN4PSIxMDUwIiBjeT0iNTIwIiByPSIzNDAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzQyYTVmNSIgc3Ryb2tlLXdpZHRoPSIzIiBvcGFjaXR5PSIwLjA4Ii8+CiAgPCEtLSDpmbjlnLDjgrfjg6vjgqjjg4Pjg4jvvIjnsKHnlaXvvIkgLS0+CiAgPHBhdGggZD0iTTc2MCAzMDAgUTc4MCAyODAgODEwIDI5NSBRODMwIDI3NSA4NTAgMjg1IFE4NzAgMjY1IDg5MCAyODAgUTkwMCAyOTUgODg1IDMxMCBRODcwIDMyNSA4NDUgMzE4IFE4MjAgMzMwIDc5NSAzMjAgUTc3MCAzMTUgNzYwIDMwMFoiIGZpbGw9IiMyZDhhNGUiIG9wYWNpdHk9IjAuNTUiLz4KICA8cGF0aCBkPSJNOTAwIDM0MCBROTIwIDMyNSA5NDAgMzM1IFE5NjAgMzE4IDk3NSAzMzAgUTk4NSAzNDUgOTcwIDM1OCBROTUwIDM2NSA5MzAgMzU1IFE5MTAgMzYwIDkwMCAzNDBaIiBmaWxsPSIjMmQ4YTRlIiBvcGFjaXR5PSIwLjQ1Ii8+CiAgPCEtLSDlpKfmtIvjga7lj43lsIQgLS0+CiAgPGVsbGlwc2UgY3g9IjgyMCIgY3k9IjM4MCIgcng9IjYwIiByeT0iMjAiIGZpbGw9IiM0ZmMzZjciIG9wYWNpdHk9IjAuMTIiIHRyYW5zZm9ybT0icm90YXRlKC0xNSw4MjAsMzgwKSIvPgoKICA8IS0tIOihm+aYn+i7jOmBk+ODqeOCpOODszEgLS0+CiAgPGVsbGlwc2UgY3g9IjQwMCIgY3k9IjIwMCIgcng9IjYyMCIgcnk9IjEyMCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ1cmwoI29yYml0MSkiIHN0cm9rZS13aWR0aD0iMSIgb3BhY2l0eT0iMC42IiB0cmFuc2Zvcm09InJvdGF0ZSgtMTIsNDAwLDIwMCkiLz4KCiAgPCEtLSDooZvmmJ/ou4zpgZPjg6njgqTjg7MyIC0tPgogIDxlbGxpcHNlIGN4PSIzMDAiIGN5PSIxODAiIHJ4PSI1ODAiIHJ5PSI5MCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSJ1cmwoI29yYml0MikiIHN0cm9rZS13aWR0aD0iMC44IiBvcGFjaXR5PSIwLjUiIHRyYW5zZm9ybT0icm90YXRlKDgsMzAwLDE4MCkiLz4KCiAgPCEtLSBHTlNT6KGb5pifMSAtLT4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxODUsNzUpIiBmaWx0ZXI9InVybCgjZ2xvdykiPgogICAgPHJlY3QgeD0iLTE4IiB5PSItMyIgd2lkdGg9IjM2IiBoZWlnaHQ9IjYiIGZpbGw9IiM2MGE1ZmEiIG9wYWNpdHk9IjAuOSIgcng9IjEiLz4KICAgIDxyZWN0IHg9Ii0zIiB5PSItMTIiIHdpZHRoPSI2IiBoZWlnaHQ9IjI0IiBmaWxsPSIjNjBhNWZhIiBvcGFjaXR5PSIwLjkiIHJ4PSIxIi8+CiAgICA8Y2lyY2xlIGN4PSIwIiBjeT0iMCIgcj0iNSIgZmlsbD0iIzkzYzVmZCIgb3BhY2l0eT0iMC45NSIvPgogICAgPGNpcmNsZSBjeD0iMCIgY3k9IjAiIHI9IjgiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzYwYTVmYSIgc3Ryb2tlLXdpZHRoPSIwLjgiIG9wYWNpdHk9IjAuNCIvPgogICAgPGNpcmNsZSBjeD0iMCIgY3k9IjAiIHI9IjE0IiBmaWxsPSJ1cmwoI3NhdEdsb3cpIiBvcGFjaXR5PSIwLjMiLz4KICA8L2c+CgogIDwhLS0gR05TU+ihm+aYnzLvvIjlsI/jgZXjgoHvvIkgLS0+CiAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODgwLDYyKSIgZmlsdGVyPSJ1cmwoI2dsb3cpIj4KICAgIDxyZWN0IHg9Ii0xNCIgeT0iLTIiIHdpZHRoPSIyOCIgaGVpZ2h0PSI0IiBmaWxsPSIjMzRkMzk5IiBvcGFjaXR5PSIwLjg1IiByeD0iMSIvPgogICAgPHJlY3QgeD0iLTIiIHk9Ii05IiB3aWR0aD0iNCIgaGVpZ2h0PSIxOCIgZmlsbD0iIzM0ZDM5OSIgb3BhY2l0eT0iMC44NSIgcng9IjEiLz4KICAgIDxjaXJjbGUgY3g9IjAiIGN5PSIwIiByPSI0IiBmaWxsPSIjNmVlN2I3IiBvcGFjaXR5PSIwLjkiLz4KICAgIDxjaXJjbGUgY3g9IjAiIGN5PSIwIiByPSIxMCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMzRkMzk5IiBzdHJva2Utd2lkdGg9IjAuNyIgb3BhY2l0eT0iMC4zNSIvPgogIDwvZz4KCiAgPCEtLSDkv6Hlj7fms6LntIvvvIjooZvmmJ8x77yJIC0tPgogIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4NSw3NSkiIG9wYWNpdHk9IjAuMjUiPgogICAgPGNpcmNsZSBjeD0iMCIgY3k9IjAiIHI9IjI1IiBmaWxsPSJub25lIiBzdHJva2U9IiM2MGE1ZmEiIHN0cm9rZS13aWR0aD0iMC44Ii8+CiAgICA8Y2lyY2xlIGN4PSIwIiBjeT0iMCIgcj0iNDAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzYwYTVmYSIgc3Ryb2tlLXdpZHRoPSIwLjYiLz4KICAgIDxjaXJjbGUgY3g9IjAiIGN5PSIwIiByPSI1OCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNjBhNWZhIiBzdHJva2Utd2lkdGg9IjAuNCIvPgogIDwvZz4KCiAgPCEtLSDkv6Hlj7fnt5rvvIjooZvmmJ/ihpLlnLDnkIPvvIkgLS0+CiAgPGxpbmUgeDE9IjE4NSIgeTE9Ijg1IiB4Mj0iODUwIiB5Mj0iMzEwIiBzdHJva2U9IiM2MGE1ZmEiIHN0cm9rZS13aWR0aD0iMC42IiBzdHJva2UtZGFzaGFycmF5PSI0LDgiIG9wYWNpdHk9IjAuMiIvPgogIDxsaW5lIHgxPSI4ODAiIHkxPSI3MiIgeDI9IjgyMCIgeTI9IjI5NSIgc3Ryb2tlPSIjMzRkMzk5IiBzdHJva2Utd2lkdGg9IjAuNiIgc3Ryb2tlLWRhc2hhcnJheT0iNCw4IiBvcGFjaXR5PSIwLjE4Ii8+CgogIDwhLS0g5Y+z56uv44Kw44Op44OH44O844K344On44Oz77yI44OV44Kn44O844OJ44Ki44Km44OI77yJIC0tPgogIDxyZWN0IHdpZHRoPSIxMjAwIiBoZWlnaHQ9IjQwMCIgZmlsbD0idXJsKCNzcGFjZSkiIG9wYWNpdHk9IjAiIGNsaXAtcGF0aD0idXJsKCNjbGlwKSIvPgo8L3N2Zz4=") center/cover no-repeat;
  border-radius:14px; padding:22px 28px; margin-bottom:20px;
  position: relative; overflow: hidden;
}
/* 半透明オーバーレイで文字を読みやすく */
.app-hdr::before {
  content: "";
  position: absolute; inset: 0;
  background: linear-gradient(135deg,rgba(15,23,42,0.62) 0%,rgba(30,58,95,0.55) 100%);
  border-radius: 14px;
  z-index: 0;
}
.app-hdr > * { position: relative; z-index: 1; }
.app-hdr h1 { margin:0; font-size:22px; font-weight:700; color:#f1f5f9;
               text-shadow: 0 1px 8px rgba(0,0,0,0.6); }
.app-hdr p  { margin:4px 0 0; color:#cbd5e1; font-size:11px; letter-spacing:.2em; text-transform:uppercase; }

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

/* ══════════════════════════════════════════════════════
   スマホ対応 CSS（ボタン行は縦並びのまま・サイドバーボタンのみ対応）
   ══════════════════════════════════════════════════════ */
@media (max-width: 768px) {

  /* ── Streamlit ネイティブのサイドバー折りたたみボタン ──
     スマホ時に常時表示・左上固定・デザイン統一             */
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"] {
    position: fixed !important;
    top: 8px !important;
    left: 8px !important;
    z-index: 999999 !important;
    background: #0f172a !important;
    border-radius: 10px !important;
    border: 1.5px solid #475569 !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.55) !important;
    width: 44px !important;
    height: 44px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    /* iOS: GPU レイヤーに昇格して fixed をキープ */
    -webkit-transform: translateZ(0) !important;
    transform: translateZ(0) !important;
    will-change: transform !important;
  }
  [data-testid="collapsedControl"] button,
  [data-testid="stSidebarCollapsedControl"] button {
    color: #f1f5f9 !important;
    font-size: 20px !important;
    width: 44px !important;
    height: 44px !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
  }
  [data-testid="collapsedControl"] svg,
  [data-testid="stSidebarCollapsedControl"] svg {
    fill: #f1f5f9 !important;
    width: 22px !important;
    height: 22px !important;
  }
}
</style>
""", unsafe_allow_html=True)

# ── ② iOS キーボード表示時の fixed 位置ずれ補正 ──────────────────
# st.markdown の HTML は Streamlit メインページ DOM に直接注入される。
# visualViewport API でキーボード表示を検知し、
# collapsedControl ボタンの top を動的に補正する。
st.markdown("""
<script>
(function(){
  if (typeof window === 'undefined') return;

  var SELECTORS = [
    '[data-testid="collapsedControl"]',
    '[data-testid="stSidebarCollapsedControl"]'
  ];

  function getBtn() {
    for (var i = 0; i < SELECTORS.length; i++) {
      var el = document.querySelector(SELECTORS[i]);
      if (el) return el;
    }
    return null;
  }

  function adjust() {
    var btn = getBtn();
    if (!btn) return;
    if (window.innerWidth > 768) { btn.style.top = ''; return; }

    if (window.visualViewport) {
      // visualViewport.offsetTop = キーボードなどでずれた上端の量
      var offset = Math.round(window.visualViewport.offsetTop);
      btn.style.top = (offset + 8) + 'px';
    } else {
      btn.style.top = '8px';
    }
  }

  // visualViewport イベント（iOS15+ / Chrome Mobile）
  if (window.visualViewport) {
    window.visualViewport.addEventListener('resize', adjust);
    window.visualViewport.addEventListener('scroll', adjust);
  }

  // フォールバック
  window.addEventListener('resize', adjust);

  // DOM 構築完了後に初回実行（Streamlit は動的レンダリングのため少し待つ）
  function tryInit(n) {
    var btn = getBtn();
    if (btn) { adjust(); return; }
    if (n > 0) setTimeout(function(){ tryInit(n-1); }, 400);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function(){ tryInit(10); });
  } else {
    tryInit(10);
  }


})();
</script>
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
                                  index=1, label_visibility="collapsed")

    st.divider()

    st.markdown("""
<div style='background:#1e293b;border:1px solid #334155;border-radius:8px;padding:8px 10px;margin:2px 0'>
  <div style='display:flex;align-items:center;gap:6px;margin-bottom:4px'>
    <span style='background:#16a34a;color:#fff;font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px'>
      ✓ 往復誤差 &lt; 0.01mm
    </span>
  </div>
  <div style='font-size:10px;color:#94a3b8;line-height:1.7'>
    GRS80楕円体 / m₀ = 0.9999<br>
    Kawase (2011) 高次展開式
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── 免責事項・プライバシーポリシー ──
    with st.expander("📋 免責事項・プライバシーポリシー"):
        st.markdown("""
<div style='font-size:11px;color:#94a3b8;line-height:1.8'>

<div style='font-size:12px;font-weight:700;color:#e2e8f0;margin-bottom:6px'>⚠️ 免責事項</div>

本ツールによる座標変換・計算結果はあくまで参考値です。<br>
計算結果の利用によって生じた損害・損失について、<br>
開発者は一切の責任を負いません。<br>
実際の測量・施工においては、有資格者による確認を行ってください。

<div style='font-size:12px;font-weight:700;color:#e2e8f0;margin:12px 0 6px'>🔒 プライバシーポリシー</div>

本ツールに入力されたデータ（座標値・高さ等）は<br>
サーバーに一切保存・記録されません。<br>
すべての処理はセッション内のみで完結し、<br>
ブラウザを閉じると同時にデータは消去されます。<br>
外部への送信はジオイド高取得API（国土地理院）のみです。

<div style='font-size:12px;font-weight:700;color:#e2e8f0;margin:12px 0 6px'>©️ 著作権</div>

© 2026 biz-cpu<br>
本ソフトウェアの無断複製・転用を禁じます。

</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════
# 8. メインヘッダー
# ═══════════════════════════════════════════════════════

# FMT/fmt_lbl はTAB内の各モードで個別に定義するため、ここではデフォルトのみ設定
_FMT_DEFAULT = "decimal"
_fmt_lbl_default = list(OUTPUT_FORMATS.keys())[0]

st.markdown(f"""
<div class="app-hdr">
  <h1>🛰️ GNSS SmartShift ICT</h1>
  <p style='font-size:13px;color:#cbd5e1;margin:2px 0 0;font-weight:500;letter-spacing:.05em'>マルチメーカー対応 ローカライゼーション統合システム</p>
  <p style='margin-top:6px'>第 {Z} 系 &nbsp;·&nbsp; {datum_lbl} &nbsp;·&nbsp; {geoid_lbl} &nbsp;·&nbsp; {map_style_lbl}</p>
</div>""", unsafe_allow_html=True)

# ── グローバルチェック（サイドバー設定の不整合を常時表示）──────
_datum_warn = check_datum_zone_mismatch(DATUM, Z)
if _datum_warn:
    st.warning(_datum_warn)

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
            cam_col, c0, c1, c2, c3, c4, c5 = st.columns([0.45,0.5,1.4,2,2,2,0.45])
            with cam_col:
                if i == 0:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                camera_ocr_button("jpc", i)
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
            # カメラパネルを各行のすぐ下に展開（全幅）
            camera_ocr_panel("jpc", i)

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
                        + f"{format_angle(lat_dd,FMT_JPC)},{format_angle(lon_dd,FMT_JPC)}"
                    )
                except (ValueError, Exception) as ex:
                    st.error(f"[{pt['name']}] エラー: {ex}")

            if map_rows:
                st.markdown("#### 📍 地図")
                render_map(map_rows, map_style_lbl, zoom=13)
                csv_out = "\n".join(csv_rows)
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
            cam_col, c0, c1, c2, c3, c4, c5 = st.columns([0.45,0.5,1.4,2.3,2.3,1.8,0.45])
            with cam_col:
                if i == 0:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                camera_ocr_button("ll", i)
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
            # カメラパネルを各行のすぐ下に展開（全幅）
            camera_ocr_panel("ll", i)

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
                csv_out2 = "\n".join(csv_rows2)
                st.download_button("📥 全点 CSV ダウンロード", csv_out2, "converted.csv", "text/csv")
        else:
            st.info(f"緯度・経度を {in_fmt_lbl} 形式で入力してください。")

    # ══════════════════════════════
    # 緯度経度 形式変換
    # ══════════════════════════════
    else:  # dir1 == "緯度経度 形式変換"
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
            cam_col, c0, c1, c2, c3, c4, c5 = st.columns([0.45,0.5,1.4,2,2,1.6,0.45])
            with cam_col:
                if i == 0:
                    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
                camera_ocr_button("cvt", i)
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
            # カメラパネルを各行のすぐ下に展開（全幅）
            camera_ocr_panel("cvt", i)
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

                    h_cvt_raw = st.session_state.get(f"cvt_h_{i}", "")
                    h_cvt_v = float(h_cvt_raw) if h_cvt_raw.strip() else None

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
                        hc_str = f"{h_cvt_v:.3f} m" if h_cvt_v is not None else "---"
                        hc_sub = "楕円体高（そのまま）" if h_cvt_v is not None else "未入力"
                        st.markdown(
                            f"<div class='rc'><div class='rc-lbl' style='color:#8b5cf6'>楕円体高 h (m)</div>"
                            f"<div class='rc-val'>{hc_str}</div>"
                            f"<div class='rc-sub'>{hc_sub}</div></div>",
                            unsafe_allow_html=True)

                    with st.expander(f"🔢 {pt['name']} 全フォーマット"):
                        st.dataframe(pd.DataFrame([
                            {"フォーマット":fl,"緯度":format_angle(lat_dd,fk),"経度":format_angle(lon_dd,fk)}
                            for fl,fk in OUTPUT_FORMATS.items()
                        ]), use_container_width=True, hide_index=True)

                    st.markdown("</div>", unsafe_allow_html=True)

                    h_cvt_raw2 = st.session_state.get(f"cvt_h_{i}", "")
                    h_cvt_v2 = float(h_cvt_raw2) if h_cvt_raw2.strip() else None
                    tip_h = f" / h={h_cvt_v2:.3f}m" if h_cvt_v2 is not None else ""
                    map_rowsc.append({"name":pt["name"],"lat":lat_dd,"lon":lon_dd,
                                      "tooltip":f"{lat_out} / {lon_out}{tip_h}"})
                    h_cvt = st.session_state.get(f"cvt_h_{i}", "")
                    h_cvt_val = float(h_cvt) if h_cvt.strip() else ""
                    csv_rowsc.append(
                        f"{pt['name']},{pt['lat']},{pt['lon']},"
                        + (f"{h_cvt_val:.3f}" if isinstance(h_cvt_val, float) else "") + ","
                        + f"{lat_out},{lon_out}"
                    )
                except (ValueError, Exception) as ex:
                    st.error(f"[{pt['name']}] エラー: {ex}")

            if map_rowsc:
                st.markdown("#### 📍 地図")
                render_map(map_rowsc, map_style_lbl, zoom=13)
                csv_outc = "\n".join(csv_rowsc)
                st.download_button("📥 全点 CSV ダウンロード", csv_outc, "converted_fmt.csv", "text/csv")
        else:
            st.info(f"緯度・経度を {in_fmt_cvt_lbl} 形式で入力してください。")


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
    up1 = st.file_uploader("CSVファイルをアップロード", ["csv","txt"], key="u1")
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

            rows_out = []
            pb = st.progress(0, "変換中...")
            total = len(df_in)

            for idx, (_, row) in enumerate(df_in.iterrows()):
                pb.progress((idx+1)/total, f"{idx+1}/{total} 点処理中")
                try:
                    def _v(s):
                        s = str(s).strip()
                        return "" if s.lower() in ("nan","none","") else s

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
                            N = fetch_geoid(lat_dd, lon_dd, GEOID_KEY)
                            if N is not None: ellH = Zv + N
                        elif Zv is not None:
                            ellH = Zv
                        out_x   = f"{Xv:.4f}"
                        out_y   = f"{Yv:.4f}"
                        out_z   = f"{Zv:.3f}" if Zv is not None else ""
                        out_lat = format_angle(lat_dd, FMT_B2)
                        out_lon = format_angle(lon_dd, FMT_B2)
                        out_h   = f"{ellH:.3f}" if ellH is not None else ""
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
                        elev_b = None
                        if hv is not None and GEOID_KEY != "NONE":
                            N_b = fetch_geoid(lv, lov, GEOID_KEY)
                            if N_b is not None: elev_b = hv - N_b
                        elif hv is not None:
                            elev_b = hv
                        out_x   = f"{Xr:.4f}"
                        out_y   = f"{Yr:.4f}"
                        out_z   = f"{elev_b:.3f}" if elev_b is not None else ""
                        out_lat = format_angle(lv, FMT_B2)
                        out_lon = format_angle(lov, FMT_B2)
                        out_h   = f"{hv:.3f}" if hv is not None else ""
                        _detected_fmt = _fmt_lat

                    _fmt_labels = {
                        "decimal":"十進角度","dms":"度分秒","bearing":"方位角",
                        "ddmmssss":"度分秒圧縮","gons":"Gons","":"—",
                    }
                    rows_out.append({
                        "点名":        name,
                        "X(m)":        out_x,
                        "Y(m)":        out_y,
                        "Z標高(m)":    out_z,
                        "緯度":        out_lat,
                        "経度":        out_lon,
                        "楕円体高(m)": out_h,
                        "判別FMT":     _fmt_labels.get(_detected_fmt, _detected_fmt),
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
                show = ["点名","判別FMT","X(m)","Y(m)","Z標高(m)","緯度","経度","楕円体高(m)"]
                st.caption("💡 判別FMT = 入力の緯度・経度に自動判別されたフォーマット")
            else:
                show = ["点名","X(m)","Y(m)","Z標高(m)","緯度","経度","楕円体高(m)"]
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
            st.download_button(
                f"📥 結果 CSV ダウンロード（{out_fmt_b2_lbl}）",
                "\n".join(csv_lines),
                "batch_result.csv", "text/csv"
            )

        except Exception as ex:
            st.error(f"処理エラー: {ex}")
    else:
        st.info("CSVファイルをアップロードしてください。")

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
- **JGD2024**（測地成果2024）: GRS80楕円体・JGD2011と同準拠楕円体。令和6年告示。実用的にはJGD2011と同等の精度で利用可能。
    """)
