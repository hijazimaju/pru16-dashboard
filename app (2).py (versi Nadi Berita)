"""Dashboard Ramalan Politik Malaysia PRU16 — Streamlit.

Semua teks UI dalam Bahasa Melayu. Semua nombor kerusi datang dari
data/pru15_results.csv + forecast_model.py; berita dari Google News RSS.
Tiada angka direka. Model adalah ILUSTRATIF, bukan ramalan muktamad.

Jalankan: streamlit run app.py
"""

from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from forecast_model import (
    COALITIONS,
    COALITION_COLORS,
    COALITION_NAMES_BM,
    load_data,
    win_probabilities,
    monte_carlo,
    marginal_seats,
    state_projection,
    nadi_berita,
)

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "pru15_results.csv"
REFRESH_MS = 10 * 60 * 1000  # 10 minit (permintaan jazi: refresh setiap 10 minit)

st.set_page_config(
    page_title="Dashboard Ramalan Politik Malaysia PRU16",
    page_icon="🇲🇾",
    layout="wide",
)

# CSS mesra telefon: padding lebih ketat & tajuk lebih kecil pada skrin sempit.
st.markdown(
    """<style>
@media (max-width: 768px) {
  .block-container { padding: 1rem 0.75rem !important; }
  h1 { font-size: 1.5rem !important; line-height: 1.25 !important; }
  h2 { font-size: 1.2rem !important; }
  h3 { font-size: 1.05rem !important; }
  .stMetric { padding: 0.25rem 0 !important; }
}
.kpi-card {
  border-top: 4px solid #888;
  background: #fafafa;
  border-radius: 8px;
  padding: 0.6rem 0.8rem;
  margin-bottom: 0.5rem;
}
.kpi-num { font-size: 1.7rem; font-weight: 700; line-height: 1.1; }
.kpi-range { font-size: 0.8rem; color: #555; }
</style>""",
    unsafe_allow_html=True,
)

# Auto-refresh seluruh app setiap 10 minit; Nadi Berita + berita di-cache 10 minit.
st_autorefresh(interval=REFRESH_MS, key="pru16_refresh")


def _now_my():
    return datetime.now(timezone.utc).astimezone()


# ---------------------------------------------------------------------------
# Data: normalisasi KDM/PBM -> BEBAS sebelum load_data (lihat Metodologi)
# ---------------------------------------------------------------------------

@st.cache_data
def load_baseline() -> pd.DataFrame:
    """Muat CSV PRU15, normalisasi gabungan tidak dikenali ke BEBAS.

    'gabungan_asal' menyimpan label asal (cth. KDM, PBM) untuk dipaparkan
    dalam jadual; model hanya menerima 8 gabungan dalam COALITIONS.
    """
    df = pd.read_csv(DATA_PATH)
    df["gabungan_asal"] = df["gabungan"]
    df["gabungan"] = df["gabungan"].where(
        df["gabungan"].isin(COALITIONS), "BEBAS"
    )
    # load_data membaca dari path; tulis salinan ternormal ke fail sementara.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as tmp:
        df.to_csv(tmp.name, index=False)
        return load_data(tmp.name)


df_base = load_baseline()

# ---------------------------------------------------------------------------
# Sidebar: andaian swing
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Andaian Swing")
st.sidebar.caption(
    "Laraskan swing nasional dan lihat unjuran dikira semula secara langsung."
)
swing_ph = st.sidebar.slider("Swing nasional — PH (mata peratus)", -10.0, 10.0, 0.0, 0.5)
swing_pn = st.sidebar.slider("Swing nasional — PN (mata peratus)", -10.0, 10.0, 0.0, 0.5)
swing_bn = st.sidebar.slider("Swing nasional — BN (mata peratus)", -10.0, 10.0, 0.0, 0.5)
youth_swing = st.sidebar.slider("Swing undi muda Undi18 (mata peratus)", 0.0, 10.0, 0.0, 0.5)
turnout_assumption = st.sidebar.slider("Peratus keluar mengundi andaian", 60, 85, 74, 1)
st.sidebar.caption(
    "ℹ️ Peratus keluar mengundi di atas adalah konteks paparan sahaja — "
    "ia tidak mengubah model kerusi secara langsung."
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "📡 Pelarasan Nadi Berita (berita terkini, terhad ±2pp) ditambah "
    "automatik ATAS nilai slider di atas. Lihat panel Nadi Berita dalam "
    "tab 🏠 Ramalan Kerusi."
)
st.sidebar.caption(
    "Model ilustratif berasaskan penyandang PRU15. Bukan ramalan muktamad. "
    "Lihat tab 📐 Metodologi."
)


# ---------------------------------------------------------------------------
# Nadi Berita: pelarasan swing automatik dari tajuk berita terkini (10 minit)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600)
def get_nadi() -> dict:
    """Tarik Nadi Berita; di-cache 10 minit.

    Gagal-fetch dikendalikan gracefully oleh nadi_berita() (pelarasan = 0);
    try/except luar ini pelindung tambahan supaya app tidak crash.
    """
    try:
        return nadi_berita()
    except Exception:  # noqa: BLE001
        return {
            "pelarasan_pp": {c: 0.0 for c in COALITIONS},
            "sebutan": {c: 0 for c in COALITIONS},
            "bilangan_tajuk": 0,
            "timestamp": None,
            "gagal": True,
            "nota": "Ralat tidak dijangka semasa mengira Nadi Berita.",
        }


nadi = get_nadi()
adj_pp = nadi["pelarasan_pp"]

# Swing efektif = slider pengguna + pelarasan Nadi Berita (ON TOP, telus).
eff_swing_ph = swing_ph + adj_pp.get("PH", 0.0)
eff_swing_pn = swing_pn + adj_pp.get("PN", 0.0)
eff_swing_bn = swing_bn + adj_pp.get("BN", 0.0)
# Tuple untuk cache key (nilai nadi dibundarkan supaya stabil).
news_tuple = tuple(round(float(adj_pp.get(c, 0.0)), 3) for c in COALITIONS)


@st.cache_data
def compute_forecast(swing_ph, swing_pn, swing_bn, youth_swing, news_tuple):
    """Kira unjuran + Monte Carlo; di-cache mengikut slider + Nadi Berita."""
    news_adj_pp = dict(zip(COALITIONS, news_tuple))
    df_proj = win_probabilities(
        df_base,
        swing_ph=swing_ph / 100.0,
        swing_pn=swing_pn / 100.0,
        swing_bn=swing_bn / 100.0,
        youth_swing=youth_swing / 100.0,
        news_adj_pp=news_adj_pp,
    )
    mc = monte_carlo(
        df_base,
        swing_ph=swing_ph / 100.0,
        swing_pn=swing_pn / 100.0,
        swing_bn=swing_bn / 100.0,
        youth_swing=youth_swing / 100.0,
        news_adj_pp=news_adj_pp,
        n=2000,
        seed=42,
    )
    return df_proj, mc


with st.spinner("Mengira unjuran kerusi..."):
    df_proj, mc = compute_forecast(
        eff_swing_ph, eff_swing_pn, eff_swing_bn, youth_swing, news_tuple
    )

proj_counts = df_proj["proj_gabungan"].value_counts()
marginal = marginal_seats(df_proj, threshold=0.05)
super_marginal = marginal_seats(df_proj, threshold=0.02)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.title("🇲🇾 Dashboard Ramalan Politik Malaysia PRU16")
st.caption(
    "Unjuran kerusi Parlimen berasaskan data PRU15 (2022) + model swing ilustratif. "
    "Parlimen bubar automatik: 19 Dis 2027 — PRU16 selewat-lewatnya 17 Feb 2028. "
    f"Andaian keluar mengundi: {turnout_assumption}% (konteks sahaja). "
    "🔄 Auto-refresh setiap 10 minit (termasuk Nadi Berita)."
)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Ramalan Kerusi",
    "📊 Pecahan Negeri",
    "🧭 Halatuju Politik",
    "💬 Sentimen Awam",
    "📰 Berita Terkini",
    "📐 Metodologi",
])

# ================= TAB 1: RAMALAN KERUSI =================
with tab1:
    st.header("🏠 Ramalan Kerusi")
    st.info("ℹ️ **112 kerusi = majoriti mudah** untuk membentuk kerajaan.")

    # ---- Panel Nadi Berita: pelarasan automatik dari berita terkini ----
    st.subheader("📡 Nadi Berita (auto-refresh setiap 10 minit)")
    if nadi.get("gagal"):
        st.warning(
            "⚠️ Nadi Berita tidak tersedia buat masa ini — "
            f"{nadi.get('nota', '')} Unjuran di bawah hanya menggunakan "
            "andaian slider anda."
        )
    else:
        ts = nadi.get("timestamp")
        if ts is not None:
            ts_my = ts.astimezone(ZoneInfo("Asia/Kuala_Lumpur"))
            ts_str = ts_my.strftime("%d %b %Y, %H:%M")
        else:
            ts_str = "-"
        st.caption(
            f"Kemas kini terakhir: **{ts_str}** (waktu Malaysia) · "
            f"**{nadi.get('bilangan_tajuk', 0)}** tajuk berita diimbas "
            "(6 jam terakhir)."
        )
    # Jadual ketelusan: slider pengguna vs Nadi Berita vs jumlah efektif.
    slider_map = {"PH": swing_ph, "PN": swing_pn, "BN": swing_bn}
    nadi_rows = []
    for c in COALITIONS:
        a = float(adj_pp.get(c, 0.0))
        s = slider_map.get(c)
        jumlah = (s if s is not None else 0.0) + a
        nadi_rows.append({
            "Gabungan": f"{COALITION_NAMES_BM[c]} ({c})",
            "Nadi Berita (pp)": f"{a:+.2f}",
            "Slider anda (pp)": f"{s:+.1f}" if s is not None else "—",
            "Jumlah efektif (pp)": f"{jumlah:+.2f}",
            "Sebutan": int(nadi.get("sebutan", {}).get(c, 0)),
        })
    st.dataframe(pd.DataFrame(nadi_rows), use_container_width=True,
                 hide_index=True)
    st.caption(
        "Pelarasan Nadi Berita terhad **±2.00 mata peratus** setiap gabungan "
        "dan ditambah atas andaian slider anda (lajur 'Jumlah efektif'). "
        "Berita ≠ pengundi sebenar — lihat batasan dalam tab 📐 Metodologi."
    )

    # Kad KPI setiap gabungan
    for chunk_start in range(0, len(COALITIONS), 4):
        cols = st.columns(4)
        for i, coal in enumerate(COALITIONS[chunk_start:chunk_start + 4]):
            proj = int(proj_counts.get(coal, 0))
            stats = mc.get(coal, {"median": 0, "p10": 0, "p90": 0})
            color = COALITION_COLORS[coal]
            with cols[i]:
                st.markdown(
                    f"""<div class="kpi-card" style="border-top-color:{color}">
<div style="font-size:0.85rem;font-weight:600">{COALITION_NAMES_BM[coal]} ({coal})</div>
<div class="kpi-num">{proj}</div>
<div class="kpi-range">Monte Carlo: median {stats['median']} """
                    f"""(P10–P90: {stats['p10']}–{stats['p90']})</div></div>""",
                    unsafe_allow_html=True,
                )

    # Carta parlimen hemicycle
    st.subheader("Carta Parlimen (hemicycle)")
    st.caption("Setiap titik = satu kerusi, diwarnakan mengikut unjuran gabungan.")

    def hemicycle_fig(df):
        order = {c: i for i, c in enumerate(COALITIONS)}
        seats = df.sort_values(
            by="proj_gabungan", key=lambda s: s.map(order)
        ).reset_index(drop=True)
        n = len(seats)
        n_rows = 8
        base, rem = divmod(n, n_rows)
        counts = [base + (1 if i < rem else 0) for i in range(n_rows)]
        xs, ys, colors, texts = [], [], [], []
        R = 10.0
        idx = 0
        for r in range(n_rows):
            radius = R - r * 1.15
            cnt = counts[r]
            for k in range(cnt):
                frac = (k + 0.5) / cnt
                ang = np.pi * (1 - frac)
                xs.append(radius * np.cos(ang))
                ys.append(radius * np.sin(ang))
                row = seats.iloc[idx]
                idx += 1
                colors.append(COALITION_COLORS[row["proj_gabungan"]])
                texts.append(
                    f"{row['kod']} {row['nama_kerusi']}<br>"
                    f"Unjuran: {COALITION_NAMES_BM[row['proj_gabungan']]} "
                    f"({row['proj_gabungan']})<br>"
                    f"Majoriti PRU15: {int(row['majoriti']):,}"
                )
        fig = go.Figure(
            go.Scatter(
                x=xs, y=ys, mode="markers",
                marker=dict(size=11, color=colors,
                            line=dict(width=0.5, color="white")),
                text=texts, hoverinfo="text",
            )
        )
        fig.update_layout(
            height=430, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(visible=False, range=[-R - 1, R + 1]),
            yaxis=dict(visible=False, range=[-1.5, R + 1]),
            showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        return fig

    st.plotly_chart(hemicycle_fig(df_proj), use_container_width=True)

    # Jadual 222 kerusi: carian + tapis negeri
    st.subheader("Jadual 222 Kerusi")
    col_search, col_state = st.columns([2, 1])
    with col_search:
        q = st.text_input("🔍 Cari kerusi / parti", "")
    with col_state:
        negeri_opts = ["Semua"] + sorted(df_proj["negeri"].unique().tolist())
        negeri_sel = st.selectbox("Tapis negeri", negeri_opts)

    tbl = df_proj.copy()
    if negeri_sel != "Semua":
        tbl = tbl[tbl["negeri"] == negeri_sel]
    if q.strip():
        ql = q.strip().lower()
        mask = (
            tbl["kod"].str.lower().str.contains(ql, na=False)
            | tbl["nama_kerusi"].str.lower().str.contains(ql, na=False)
            | tbl["parti_pemenang"].str.lower().str.contains(ql, na=False)
        )
        tbl = tbl[mask]

    prob_cols = [f"prob_{c}" for c in COALITIONS]
    tbl_disp = tbl[["kod", "nama_kerusi", "negeri", "parti_pemenang",
                    "gabungan_asal", "proj_gabungan", "majoriti",
                    "peratus_keluar_mengundi"]].copy()
    tbl_disp["prob_tertinggi"] = tbl[prob_cols].max(axis=1).round(3)
    tbl_disp = tbl_disp.rename(columns={
        "kod": "Kod", "nama_kerusi": "Kerusi", "negeri": "Negeri",
        "parti_pemenang": "Parti (PRU15)", "gabungan_asal": "Gabungan (PRU15)",
        "proj_gabungan": "Unjuran Gabungan", "majoriti": "Majoriti (PRU15)",
        "peratus_keluar_mengundi": "Keluar Mengundi % (PRU15)",
        "prob_tertinggi": "Kebarangkalian Tertinggi",
    })
    st.dataframe(tbl_disp, use_container_width=True, hide_index=True,
                 height=450)
    st.caption(
        "Lajur 'Gabungan (PRU15)' mengekalkan label asal dari data, termasuk "
        "KDM (P168 Kota Marudu) dan PBM (P209 Julau)."
    )

    # Kerusi marginal
    with st.expander(f"⚔️ Kerusi marginal (<5% margin PRU15): {len(marginal)} kerusi"):
        m_disp = marginal.copy()
        m_disp["margin_share"] = (m_disp["margin_share"] * 100).round(2)
        m_disp = m_disp.rename(columns={
            "kod": "Kod", "nama_kerusi": "Kerusi", "negeri": "Negeri",
            "parti_pemenang": "Parti (PRU15)", "gabungan": "Gabungan (PRU15)",
            "majoriti": "Majoriti", "margin_share": "Margin %",
            "peratus_keluar_mengundi": "Keluar Mengundi %",
        })
        st.dataframe(m_disp, use_container_width=True, hide_index=True)
    with st.expander(
        f"🔥 Kerusi 'super marginal' (<2% margin PRU15): {len(super_marginal)} kerusi"
    ):
        sm_disp = super_marginal.copy()
        sm_disp["margin_share"] = (sm_disp["margin_share"] * 100).round(2)
        sm_disp = sm_disp.rename(columns={
            "kod": "Kod", "nama_kerusi": "Kerusi", "negeri": "Negeri",
            "parti_pemenang": "Parti (PRU15)", "gabungan": "Gabungan (PRU15)",
            "majoriti": "Majoriti", "margin_share": "Margin %",
            "peratus_keluar_mengundi": "Keluar Mengundi %",
        })
        st.dataframe(sm_disp, use_container_width=True, hide_index=True)
        st.caption("Kerusi paling rapuh — sedikit swing sahaja boleh mengubah tangan.")

# ================= TAB 2: PECAHAN NEGERI =================
with tab2:
    st.header("📊 Pecahan Negeri")
    st.caption("Unjuran kerusi mengikut negeri (proj_gabungan).")
    pivot = state_projection(df_proj)
    # Susun negeri mengikut jumlah kerusi, menurun
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    fig = go.Figure()
    for coal in [c for c in COALITIONS if c in pivot.columns]:
        fig.add_trace(go.Bar(
            y=pivot.index, x=pivot[coal], name=coal,
            orientation="h", marker_color=COALITION_COLORS[coal],
            hovertemplate=f"%{{y}} — {coal}: %{{x}} kerusi<extra></extra>",
        ))
    fig.update_layout(
        barmode="stack", height=max(420, 34 * len(pivot)),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Bilangan kerusi (unjuran)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Lihat jadual pecahan negeri"):
        st.dataframe(pivot, use_container_width=True)

# ================= TAB 3: HALATUJU POLITIK =================
with tab3:
    st.header("🧭 Halatuju Politik")
    st.caption("Gambaran neutral landskap politik Malaysia menuju PRU16.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🏛️ Kerajaan Perpaduan")
        st.write(
            "Gabungan kerajaan persekutuan: **PH + BN + GPS + GRS** serta "
            "parti-parti lain. Perdana Menteri: **Anwar Ibrahim** "
            "(sejak November 2022, selepas PRU15)."
        )
    with c2:
        st.markdown("#### ⚔️ Perikatan Nasional")
        st.write(
            "Blok pembangkang utama: **PAS, BERSATU, GERAKAN**. "
            "Kuat di Pantai Timur dan utara Semenanjung berdasarkan "
            "keputusan PRU15."
        )
    with c3:
        st.markdown("#### 🤝 Blok Borneo")
        st.write(
            "**GPS (Sarawak)** dan **GRS (Sabah)** — penentu kuasa "
            "(\"kingmaker\"): sokongan mereka boleh menentukan siapa "
            "membentuk kerajaan persekutuan."
        )

    st.subheader("Garis Masa ke PRU16")
    st.markdown(
        """
- **19 Nov 2022** — PRU15: parlimen tergantung, Kerajaan Perpaduan dibentuk
- **Ogos 2023** — PRN 6 negeri (Selangor, Negeri Sembilan, Pulau Pinang, Kedah, Kelantan, Terengganu)
- **9 Okt 2026** — Belanjawan 2027 dibentangkan
- **14 Nov 2026** — PRN Melaka
- **19 Dis 2027** — Bubar automatik Parlimen
- **17 Feb 2028** — PRU16 selewat-lewatnya (60 hari selepas bubar)
"""
    )

    # Countdown ke bubar automatik
    today_my = datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).date()
    target = date(2027, 12, 19)
    days_left = (target - today_my).days
    if days_left >= 0:
        st.metric("⏳ Hari ke bubar automatik Parlimen (19 Dis 2027)",
                  f"{days_left:,} hari")
    else:
        st.metric("⏳ Bubar automatik Parlimen (19 Dis 2027)",
                  "Telah berlalu")

    st.subheader("Janji vs Pelaksanaan")
    st.warning(
        "⚠️ **Rangkuman editorial berdasarkan liputan berita — bukan data keras.** "
        "Status di bawah adalah tafsiran umum, bukan ukuran rasmi."
    )
    st.table(pd.DataFrame([
        {"Bidang": "Reformasi institusi",
         "Contoh": "Had tempoh PM, RUU perkhidmatan parlimen, pemisahan Peguam Negara",
         "Status": "Sedang berjalan"},
        {"Bidang": "Kos sara hidup & subsidi",
         "Contoh": "Penyasaran subsidi, bantuan tunai, kawalan harga barang",
         "Status": "Sedang berjalan"},
        {"Bidang": "Tadbir urus & antirasuah",
         "Contoh": "Pendakwaan kes berprofil tinggi, reformasi perolehan",
         "Status": "Sedang berjalan"},
        {"Bidang": "Ekonomi & pelaburan",
         "Contoh": "NIMP 2030, NETR, sasaran defisit fiskal",
         "Status": "Dilaksana (berperingkat)"},
        {"Bidang": "Pendidikan",
         "Contoh": "Kurikulum baharu, isu sekolah daif Sabah/Sarawak",
         "Status": "Sedang berjalan"},
        {"Bidang": "Perjanjian Malaysia 1963 (MA63)",
         "Contoh": "Pemerkasaan autonomi Sabah & Sarawak",
         "Status": "Sedang berjalan"},
    ]))

# ================= TAB 4: SENTIMEN AWAM =================
with tab4:
    st.header("💬 Sentimen Awam")
    st.caption("Peranan setiap platform dalam politik Malaysia — gambaran umum.")

    platforms = [
        ("🎵 TikTok", "Video pendek; dominan dalam kalangan pengundi muda Undi18. "
         "Naratif politik tersebar pantas melalui trend dan 'duet'."),
        ("🐦 X (Twitter)", "Elit politik, wartawan dan aktivis; tempat kenyataan "
         "rasmi, perdebatan dasar dan 'ratio' tular."),
        ("📘 Facebook", "Penetrasi tertinggi merentas umur; kumpulan komuniti dan "
         "halaman parti menjadi medan kempen setempat."),
        ("🧵 Threads", "Perbualan teks santai; berkembang dalam kalangan pengguna "
         "bandar dan profesional muda."),
        ("▶️ YouTube", "Podcast politik, ceramah penuh dan analisis panjang; "
         "rujukan untuk hujah mendalam."),
        ("📸 Instagram", "Visual kempen: infografik dasar, reels calon dan "
         "'behind-the-scenes' jentera parti."),
    ]
    for name, desc in platforms:
        st.markdown(f"**{name}** — {desc}")

    st.warning(
        "⚠️ Media sosial **bukan** cerminan pengundi sebenar. Setiap platform "
        "adalah *echo chamber*: algoritma menayangkan kandungan yang menguatkan "
        "pandangan sedia ada, dan pengguna media sosial tidak mewakili taburan "
        "umur, kaum dan kawasan pengundi berdaftar."
    )

    st.subheader("Trend Kata Kunci (proksi dari tajuk berita)")
    st.caption("Bukan data media sosial — hanya kekerapan perkataan dalam tajuk berita semasa.")

# Berita di-fetch sekali di sini (dikongsi tab 4 & 5) — di-cache 10 minit.
NEWS_QUERIES = ["politik Malaysia", "PRU16", "pilihan raya Malaysia", "PRN Melaka"]

STOPWORDS = set("""
yang dan di ke dari untuk dengan pada ini itu adalah tidak dalam sebagai atau
lebih telah juga akan ada kerana oleh terhadap antara semua mereka kita kamu
dia ia nya lah kah pun tapi namun serta iaitu agar supaya jika kalau apabila
sementara manakala setiap beberapa para tersebut ini itu
the and to of in for on with is are was were be by as at from new said after
over will would has has have had not but about into out up down per via vs
amid among during says say year years day days news report reports malaysia
malaysian kuala lumpur bernama astro awani sinar harian metro
""".split())


@st.cache_data(ttl=600)
def fetch_news() -> pd.DataFrame:
    """Berita politik terkini dari Google News RSS (pelbagai outlet berlesen).

    Di-cache 10 minit — seiring auto-refresh app.
    """
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
    rows = []
    for q in NEWS_QUERIES:
        url = (f"https://news.google.com/rss/search?q={q}%20when%3A30d"
               f"&hl=ms&gl=MY&ceid=MY%3Ams")
        try:
            r = requests.get(url, headers=headers, timeout=20)
            r.raise_for_status()
            items = ET.fromstring(r.content).find("channel").findall("item")
        except Exception:  # noqa: BLE001 — langkau query yang gagal
            continue
        for it in items:
            title = (it.findtext("title") or "").rsplit(" - ", 1)[0].strip()
            src_el = it.find("source")
            link = (it.findtext("link") or "").strip()
            try:
                pub = parsedate_to_datetime(it.findtext("pubDate"))
            except Exception:  # noqa: BLE001
                pub = None
            if title and link:
                rows.append({
                    "tajuk": title,
                    "sumber": src_el.text if src_el is not None else "-",
                    "masa": pub, "pautan": link,
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        # hanya outlet berita — buang sumber media sosial
        sosmed = ("facebook", "youtube", "tiktok", "instagram", "twitter", "x ")
        df = df[~df["sumber"].str.lower().str.startswith(sosmed)]
        df["kunci"] = df["tajuk"].str.lower().str[:60]
        df = (df.sort_values("masa", ascending=False)
                .drop_duplicates("kunci").head(20).reset_index(drop=True))
    df.attrs["fetched_at"] = _now_my()
    return df


def masa_relatif(dt) -> str:
    """Format masa relatif dalam BM ('3 jam lalu')."""
    if dt is None or (isinstance(dt, float) and pd.isna(dt)):
        return "-"
    now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    secs = (now - dt).total_seconds()
    if secs < 0:
        return "baru sahaja"
    if secs < 3600:
        return f"{max(int(secs // 60), 1)} minit lalu"
    if secs < 86400:
        return f"{int(secs // 3600)} jam lalu"
    return f"{int(secs // 86400)} hari lalu"


try:
    news_df = fetch_news()
    news_ok = not news_df.empty
except Exception as e:  # noqa: BLE001 — papar amaran, jangan crash
    news_df, news_ok = pd.DataFrame(), False
    st.warning(f"Gagal tarik berita buat masa ini ({e}). Cuba lagi sebentar.")

with tab4:
    if news_ok:
        words = re.findall(r"[a-zA-Z]{4,}", " ".join(
            news_df["tajuk"].str.lower()))
        freq = (pd.Series([w for w in words if w not in STOPWORDS])
                .value_counts().head(15))
        if not freq.empty:
            fig_kw = go.Figure(go.Bar(
                x=freq.values, y=freq.index, orientation="h",
                marker_color="#3949AB",
                hovertemplate="‘%{y}’: %{x} kali<extra></extra>",
            ))
            fig_kw.update_layout(
                height=420, margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="Kekerapan dalam tajuk berita",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_kw, use_container_width=True)
        else:
            st.info("Tiada kata kunci dapat diekstrak buat masa ini.")
    else:
        st.info("Berita tidak tersedia — trend kata kunci tidak dapat dikira.")

# ================= TAB 5: BERITA TERKINI =================
with tab5:
    st.header("📰 Berita Terkini")
    if news_ok:
        st.caption(
            f"Dikemas kini: {news_df.attrs.get('fetched_at', '-'):%d %b %Y, %H:%M} "
            "(waktu Malaysia). Sumber: Google News RSS — politik Malaysia, PRU16, "
            "pilihan raya Malaysia, PRN Melaka (30 hari)."
        )
        for _, row in news_df.iterrows():
            st.markdown(
                f"- [{row['tajuk']}]({row['pautan']})  \n"
                f"  <small>{row['sumber']} · {masa_relatif(row['masa'])}</small>",
                unsafe_allow_html=True,
            )
    else:
        st.warning("Tiada berita dapat ditarik buat masa ini. Cuba lagi sebentar.")

# ================= TAB 6: METODOLOGI =================
with tab6:
    st.header("📐 Metodologi")
    st.markdown(
        """
### Cara model dikira (ringkas, dalam Bahasa Melayu)

**1. Baseline = keputusan PRU15 sebenar.** Setiap daripada 222 kerusi bermula
dengan penyandang sebenar dari data Wikipedia (keputusan PRU15 mengikut
kawasan parlimen). Tiada andaian lompat parti — baseline adalah keputusan
pilihan raya, bukan komposisi Dewan semasa.

**2. Kekuatan setiap gabungan.** Untuk setiap kerusi, setiap gabungan diberi
skor "kekuatan":

- **Bonus penyandang:** `30 × (majoriti ÷ jumlah pengundi) + 0.5` — kerusi yang
  dimenangi besar lebih "selamat"; semua penyandang dapat bonus asas 0.5.
- **Prior serantau:** GPS kuat di Sarawak, GRS/WARISAN kuat di Sabah/Labuan
  (dan sangat lemah di luar); MUDA dan Bebas diberi skor negatif di mana-mana.
- **Swing nasional (dari slider anda):** setiap 1 mata peratus swing menambah
  kekuatan PH/PN/BN mengikut arah yang anda tetapkan.
- **Swing undi muda (andaian ilustratif):** memberi manfaat kepada PH dan MUDA,
  merugikan PN.

**3. Kebarangkalian menang.** Skor ditukar kepada kebarangkalian melalui
*softmax* — jumlah kebarangkalian kelapan-lapan gabungan bagi setiap kerusi
sentiasa = 1.0. Unjuran kerusi (`proj_gabungan`) ialah gabungan dengan
kebarangkalian tertinggi di kerusi itu.

**4. Monte Carlo.** Model diulang 2,000 kali; setiap ulangan menambah hingar
rawak (taburan normal, sisihan 1.1) pada skor untuk mewakili ketidaktentuan
tempatan (calon, isu setempat, cuaca hari mengundi). Julat P10–P90 ialah
selang 80% jumlah kerusi merentas 2,000 ulangan. Seed ditetapkan (42) supaya
keputusan boleh diulang.

### Batasan model
1. Model ini **ilustratif, bukan anggaran statistik** — nilai swing, kekuatan
   penyandang dan hingar adalah andaian pembina model.
2. Model **tidak tahu** tentang persempadanan semula kerusi, pertandingan
   berbilang penjuru sebenar, atau calon spesifik.
3. Kekuatan penyandang dianggar **linear dari margin PRU15** — tidak mengambil
   kira perubahan demografi, perpindahan pengundi atau sentimen tempatan.
4. Output adalah **distribusi kebarangkalian, bukan ramalan pasti** — julat
   Monte Carlo menunjukkan ketidaktentuan, bukan keyakinan.

### Nadi Berita — pelarasan automatik setiap 10 minit

**Apa itu?** Selain andaian slider anda, dashboard ini membaca "nadi"
berita politik semasa dan menyesuaikan unjuran secara automatik:

1. **Tarik tajuk berita** — setiap 10 minit, app mengambil tajuk berita
   politik Malaysia (Bahasa Melayu) yang diterbitkan dalam **6 jam terakhir**
   dari Google News RSS.
2. **Kira sebutan** — setiap tajuk disemak: gabungan mana yang disebut?
   (cth. "Anwar"/"PKR"/"DAP" = PH; "PAS"/"BERSATU"/"Muhyiddin" = PN;
   "UMNO"/"Zahid" = BN; dan seterusnya untuk GPS, GRS, WARISAN, MUDA).
3. **Skor sentimen** — setiap tajuk diberi skor ringkas: +1 untuk setiap
   kata positif ("menang", "sokong", "berjaya"...) dan -1 untuk setiap kata
   negatif ("kalah", "skandal", "gagal", "bantah"...).
4. **Pelarasan kecil** — purata sentimen setiap gabungan ditukar kepada
   pelarasan swing, **terhad kepada ±2 mata peratus** setiap gabungan.
   Gabungan yang disebut kurang dari 2 kali tidak diusik langsung
   (anti-hingar: satu tajuk sensasi tidak boleh menggerakkan model).
5. **Telus** — pelarasan ini **ditambah atas** nilai slider anda. Panel
   "📡 Nadi Berita" dalam tab Ramalan Kerusi menunjukkan kedua-duanya
   secara berasingan: lajur "Slider anda", "Nadi Berita", dan
   "Jumlah efektif".

**Jika berita gagal ditarik** (masalah rangkaian dsb.), pelarasan menjadi
**0 untuk semua gabungan** dan app memaparkan amaran — unjuran kembali
kepada andaian slider anda sahaja. Modul ini tidak pernah mereka data.

### Batasan Nadi Berita (jujur)
1. **Berita ≠ pengundi sebenar.** Tajuk berita berat kepada peristiwa
   sensasi dan kontroversi; liputan tidak seimbang antara gabungan.
2. Skor sentimen adalah **kiraan kata kunci yang kasar** — ia tidak faham
   konteks, sindiran, atau tajuk yang berlapis maksud.
3. Pelarasan **sengaja kecil (±2pp)** — ia "nadi", bukan ramalan. Kesan
   sebenar berita kepada undi tidak dapat diukur dengan cara ini.
4. Sumber hanya **tajuk RSS**, bukan analisis kandungan penuh artikel.

### Nota data
- **P168 Kota Marudu (KDM)** dan **P209 Julau (PBM)** — parti serantau satu
  kerusi — dimodelkan sebagai **Bebas**; label asal dikekalkan dalam jadual.
- **P044 Permatang Pauh** mengikut data CSV: PAS/PN (bukan BERSATU).
- Sumber data: Wikipedia — *Results of the 2022 Malaysian general election
  by parliamentary constituency*
  (https://en.wikipedia.org/wiki/Results_of_the_2022_Malaysian_general_election_by_parliamentary_constituency).
- Berita: Google News RSS (dikemas kini setiap 10 minit).
"""
    )
    st.caption(
        f"Tarikh paparan: {_now_my():%d %B %Y}. Model: forecast_model.py "
        "(fungsi pure, tiada data direka)."
    )

st.markdown("---")
st.caption(
    "🇲🇾 Dashboard Ramalan Politik Malaysia PRU16 — model ilustratif untuk "
    "tujuan pendidikan. Bukan ramalan muktamad atau nasihat pengundian."
)
