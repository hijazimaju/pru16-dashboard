"""Model ramalan PRU16 — telus, ilustratif, BUKAN ramalan muktamad.

Modul ini menyediakan fungsi pure untuk aplikasi Streamlit dashboard ramalan
PRU16. Semua andaian didokumenkan dalam docstring setiap fungsi dalam
Bahasa Melayu. Baseline 100% datang dari data CSV (keputusan PRU15);
modul ini TIDAK mengandungi sebarang data kerusi yang direka.

BATASAN UTAMA (mesti dipaparkan dalam UI):
  1. Model berasaskan penyandang PRU15 + swing nasional yang ditetapkan
     pengguna — ia tidak tahu tentang persempadanan semula kerusi,
     pertandingan berbilang penjuru sebenar, atau calon spesifik.
  2. Nilai swing, kekuatan penyandang, dan hingar simulasi adalah
     ILUSTRATIF, bukan anggaran statistik dari data sebenar.
  3. Kekuatan penyandang dianggar linear dari margin kemenangan PRU15 —
     tidak mengambil kira perubahan demografi, perpindahan pengundi,
     atau sentimen tempatan.
  4. Keputusan adalah distribusi kebarangkalian, bukan ramalan pasti.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Tetapan gabungan
# ---------------------------------------------------------------------------

COALITIONS = ["PH", "PN", "BN", "GPS", "GRS", "WARISAN", "MUDA", "BEBAS"]

COALITION_COLORS = {
    "PH": "#E53935",
    "PN": "#3949AB",
    "BN": "#1E88E5",
    "GPS": "#FF9800",
    "GRS": "#009688",
    "WARISAN": "#8E24AA",
    "MUDA": "#FFEB3B",
    "BEBAS": "#9E9E9E",
}

COALITION_NAMES_BM = {
    "PH": "Pakatan Harapan",
    "PN": "Perikatan Nasional",
    "BN": "Barisan Nasional",
    "GPS": "Gabungan Parti Sarawak",
    "GRS": "Gabungan Rakyat Sabah",
    "WARISAN": "WARISAN",
    "MUDA": "MUDA",
    "BEBAS": "Bebas",
}

_REQUIRED_COLUMNS = [
    "kod",
    "nama_kerusi",
    "negeri",
    "parti_pemenang",
    "gabungan",
    "majoriti",
    "peratus_keluar_mengundi",
    "jumlah_pengundi",
]

_NOISE_SD = 1.1  # sisihan piawai hingar simulasi Monte Carlo

# ---------------------------------------------------------------------------
# Utiliti dalaman (pure, tanpa I/O)
# ---------------------------------------------------------------------------


def _strength_matrix(df: pd.DataFrame, swing_ph: float, swing_pn: float,
                     swing_bn: float, youth_swing: float,
                     news_adj_pp: dict | None = None) -> np.ndarray:
    """Kira matriks kekuatan (n_kerusi × 8) mengikut formula spesifikasi.

    - Penyandang (gabungan = baseline PRU15): s = 30 * margin_share + 0.5.
    - Prior serantau untuk GPS/GRS/WARISAN; MUDA dan BEBAS negatif di
      mana-mana; PH/PN/BN sifar.
    - Swing nasional diskalakan ×40; swing undi muda mengikut andaian
      yang didokumenkan.
    - ``news_adj_pp`` (pilihan): pelarasan Nadi Berita dalam mata peratus
      (sudah terhad ±2pp di peringkat pengiraan) — ditambah atas swing
      manual pengguna, diskala ×40 seperti swing nasional.
    """
    n = len(df)
    m_share = margin_share(df).to_numpy(dtype=float)
    incumbent = df["gabungan"].to_numpy()
    negeri = df["negeri"].to_numpy()

    strengths = np.zeros((n, len(COALITIONS)))

    # Bonus penyandang: kekuatan linear terhadap margin kemenangan PRU15
    # campur bonus asas 0.5 untuk kelebihan penyandang am.
    for j, coal in enumerate(COALITIONS):
        strengths[:, j] = np.where(incumbent == coal, 30.0 * m_share + 0.5, 0.0)

    # Prior serantau: gabungan borneo dominan di negeri masing-masing,
    # sebaliknya sangat lemah di luar (nilai -8 ≈ tiada peluang).
    sarawak = negeri == "Sarawak"
    sabah_labuan = np.isin(negeri, ["Sabah", "Labuan"])

    strengths[:, COALITIONS.index("GPS")] += np.where(sarawak, 2.0, -8.0)
    strengths[:, COALITIONS.index("GRS")] += np.where(sabah_labuan, 2.0, -8.0)
    strengths[:, COALITIONS.index("WARISAN")] += np.where(sabah_labuan, 1.5, -8.0)

    # MUDA dan BEBAS: pesaing kecil tanpa jentera nasional — negatif di mana-mana.
    strengths[:, COALITIONS.index("MUDA")] += -1.0
    strengths[:, COALITIONS.index("BEBAS")] += -1.5

    # Swing nasional: perubahan sentimen negara dikongsi semua kerusi.
    strengths[:, COALITIONS.index("PH")] += swing_ph * 40.0
    strengths[:, COALITIONS.index("PN")] += swing_pn * 40.0
    strengths[:, COALITIONS.index("BN")] += swing_bn * 40.0

    # Undi muda (ANDAIAN ilustratif): peningkatan sokongan pengundi muda
    # memberi manfaat kepada PH dan MUDA, merugikan PN.
    strengths[:, COALITIONS.index("PH")] += youth_swing * 25.0
    strengths[:, COALITIONS.index("MUDA")] += youth_swing * 25.0
    strengths[:, COALITIONS.index("PN")] -= youth_swing * 15.0

    # Nadi Berita: pelarasan automatik dari sentimen tajuk berita terkini.
    # Nilai dalam mata peratus (sudah terhad ±2pp oleh nadi_berita);
    # ditambah atas swing manual pengguna, diskala ×40 seperti swing nasional.
    if news_adj_pp:
        for j, coal in enumerate(COALITIONS):
            adj = news_adj_pp.get(coal, 0.0) or 0.0
            if adj:
                strengths[:, j] += (adj / 100.0) * 40.0

    return strengths


def _softmax(strengths: np.ndarray) -> np.ndarray:
    """Softmax stabil numerikal merentas lajur (setiap baris jumlah = 1.0)."""
    shifted = strengths - strengths.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Fungsi awam (kontrak API)
# ---------------------------------------------------------------------------


def load_data(path: str = "data/pru15_results.csv") -> pd.DataFrame:
    """Muat data baseline keputusan PRU15 dari fail CSV.

    Andaian: fail mesti mengandungi lajur
    ``kod, nama_kerusi, negeri, parti_pemenang, gabungan, majoriti,
    peratus_keluar_mengundi, jumlah_pengundi``. Setiap baris = satu kerusi
    parlimen PRU15 (dijangka 222 baris untuk Parlimen Malaysia).

    Kegagalan membaca fail atau lajur yang tiada akan menimbulkan
    pengecualian — modul tidak pernah menggantikan dengan data rekaan.
    """
    df = pd.read_csv(path)
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Lajur tiada dalam CSV: {missing}")
    unknown = sorted(set(df["gabungan"].dropna().unique()) - set(COALITIONS))
    if unknown:
        raise ValueError(f"Gabungan tidak dikenali dalam CSV: {unknown}")
    return df.reset_index(drop=True)


def margin_share(df: pd.DataFrame) -> pd.Series:
    """Nisbah margin kemenangan: majoriti / jumlah_pengundi (float).

    Andaian: ukuran betapa 'selamat' sesebuah kerusi — semakin besar nisbah,
    semakin sukar kerusi itu dirampas. Nilai antara 0 dan 1.
    """
    return df["majoriti"].astype(float) / df["jumlah_pengundi"].astype(float)


def win_probabilities(df: pd.DataFrame, swing_ph: float = 0.0,
                      swing_pn: float = 0.0, swing_bn: float = 0.0,
                      youth_swing: float = 0.0,
                      news_adj_pp: dict | None = None) -> pd.DataFrame:
    """Kebarangkalian menang setiap gabungan bagi setiap kerusi.

    Andaian:
    - Model kekuatan mengikut ``_strength_matrix``: bonus penyandang dari
      margin PRU15, prior serantau, swing nasional, dan andaian undi muda.
    - ``news_adj_pp`` (pilihan): pelarasan Nadi Berita (mata peratus,
      terhad ±2pp) ditambah atas swing manual.
    - Kekuatan ditukar kepada kebarangkalian melalui softmax; jumlah setiap
      baris = 1.0 (disemak dengan assert).
    - ``proj_gabungan`` = gabungan dengan kebarangkalian tertinggi
      (mod projeksi, bukan jangkaan).

    Pulangan: salinan ``df`` + lajur ``prob_PH ... prob_BEBAS`` dan
    ``proj_gabungan``.
    """
    strengths = _strength_matrix(df, swing_ph, swing_pn, swing_bn,
                                 youth_swing, news_adj_pp)
    probs = _softmax(strengths)
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-9), \
        f"Jumlah kebarangkalian tidak = 1.0: {row_sums.min()}..{row_sums.max()}"

    out = df.copy()
    for j, coal in enumerate(COALITIONS):
        out[f"prob_{coal}"] = probs[:, j]
    out["proj_gabungan"] = [COALITIONS[i] for i in probs.argmax(axis=1)]
    return out


def monte_carlo(df: pd.DataFrame, swing_ph: float = 0.0,
                swing_pn: float = 0.0, swing_bn: float = 0.0,
                youth_swing: float = 0.0, n: int = 2000,
                seed: int = 42,
                news_adj_pp: dict | None = None) -> dict:
    """Simulasi Monte Carlo jumlah kerusi setiap gabungan.

    Andaian:
    - Matriks kekuatan (n_kerusi × 8) dikira sekali secara vektor (numpy).
    - ``news_adj_pp`` (pilihan): pelarasan Nadi Berita (mata peratus,
      terhad ±2pp) ditambah atas swing manual sebelum simulasi.
    - Setiap simulasi menambah hingar Normal(0, 1.1) yang bebas kepada
      setiap kekuatan — hingar mewakili ketidaktentuan tempatan yang tidak
      dapat dimodelkan (calon, isu setempat, cuaca hari mengundi, dll.).
    - Pemenang kerusi = argmax kekuatan berhingar; jumlah kerusi setiap
      gabungan dikira merentas ``n`` simulasi.
    - ``seed`` memastikan keputusan boleh diulang (reproducible).
    - Pengiraan sepenuhnya bervektor: mesti siap < 10 saat untuk n=2000.

    Pulangan: {gabungan: {"median": int, "p10": int, "p90": int}} — median
    dan selang persentil 10/90 jumlah kerusi merentas simulasi.
    """
    rng = np.random.default_rng(seed)
    strengths = _strength_matrix(df, swing_ph, swing_pn, swing_bn,
                                 youth_swing, news_adj_pp)
    n_seats = strengths.shape[0]

    # Vektor merentas simulasi: (n, n_kerusi, 8) — ~28 MB untuk 2000×222×8 float64.
    noise = rng.normal(0.0, _NOISE_SD, size=(n, n_seats, len(COALITIONS)))
    winners = np.argmax(strengths[None, :, :] + noise, axis=2)  # (n, n_kerusi)

    result: dict = {}
    for j, coal in enumerate(COALITIONS):
        counts = (winners == j).sum(axis=1).astype(float)
        result[coal] = {
            "median": int(np.median(counts)),
            "p10": int(np.percentile(counts, 10)),
            "p90": int(np.percentile(counts, 90)),
        }
    return result


def marginal_seats(df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Kerusi marginal: margin_share < threshold, disusun menaik.

    Andaian: kerusi dengan margin kemenangan kurang dari 5% undi dianggap
    'marginal' — paling mudah berubah tangan jika berlaku swing. Ambang
    boleh dilaras; susunan menaik bermakna kerusi paling rapuh di atas.
    """
    out = df.copy()
    out["margin_share"] = margin_share(df)
    out = out[out["margin_share"] < threshold].sort_values("margin_share")
    return out[["kod", "nama_kerusi", "negeri", "parti_pemenang", "gabungan",
                "majoriti", "margin_share",
                "peratus_keluar_mengundi"]].reset_index(drop=True)


def state_projection(df_proj: pd.DataFrame) -> pd.DataFrame:
    """Pivot negeri × proj_gabungan: bilangan kerusi projeksi setiap negeri.

    Andaian: input ialah output ``win_probabilities`` (mesti ada lajur
    ``negeri`` dan ``proj_gabungan``). Sel kosong diisi 0. Lajur disusun
    mengikut tertib COALITIONS (hanya yang hadir dikekalkan).
    """
    pivot = pd.crosstab(df_proj["negeri"], df_proj["proj_gabungan"])
    ordered = [c for c in COALITIONS if c in pivot.columns]
    return pivot[ordered]


# ---------------------------------------------------------------------------
# Nadi Berita — pelarasan swing automatik dari tajuk berita terkini
# ---------------------------------------------------------------------------
# Mekanisme (telus, ilustratif):
#   1. Tarik Google News RSS (politik Malaysia, BM) — tajuk dalam N jam terakhir.
#   2. Kira sebutan setiap gabungan melalui senarai kata kunci (padanan
#      sempadan perkataan, tidak sensitif huruf besar/kecil).
#   3. Skor sentimen setiap tajuk: (+1 setiap kata positif, -1 setiap kata
#      negatif) dalam BM + EN.
#   4. Pelarasan = 2.0 × (sentimen bersih / bilangan sebutan), TERHAD ±2pp,
#      dan 0 jika sebutan < 2 (anti-hingar: satu tajuk tidak boleh
#      menggerakkan model).
#   5. Jika RSS gagal: pelarasan = 0 untuk semua gabungan (graceful) —
#      modul TIDAK PERNAH mereka data.
#
# BATASAN (mesti dipaparkan dalam UI): tajuk berita berat kepada peristiwa
# sensasi; liputan tidak seimbang antara gabungan; berita ≠ pengundi sebenar.

NADI_QUERIES = [
    "politik Malaysia",
    "PRU16",
    "pilihan raya Malaysia",
    "Anwar Ibrahim",
    "Perikatan Nasional",
]

NADI_KEYWORDS = {
    "PH": ["anwar", "pkr", "dap", "amanah", "pakatan harapan", "rafizi",
           "anthony loke"],
    "PN": ["muhyiddin", "pas", "bersatu", "perikatan nasional", "sanusi",
           "hamzah zainudin"],
    "BN": ["umno", "zahid", "barisan nasional", "tok mat"],
    "GPS": ["gps", "sarawak", "abang jo"],
    "GRS": ["grs", "hajiiji", "sabah"],
    "WARISAN": ["warisan", "shafie"],
    "MUDA": ["muda", "syed saddiq"],
    "BEBAS": [],
}

NADI_POSITIF = [
    "menang", "sokong", "sokongan", "baik", "puas hati", "berjaya",
    "kejayaan", "kukuh", "popular", "puji", "lulus", "untung", "cemerlang",
    "hebat", "rekod", "terima baik",
    "win", "support", "success", "victory", "praise",
]

NADI_NEGATIF = [
    "kalah", "bantah", "bantahan", "skandal", "gagal", "kegagalan",
    "kecewa", "tolak", "kritik", "kritikan", "rasuah", "jatuh", "merosot",
    "protes", "saman", "tuduh", "hilang", "tewas", "desak",
    "lose", "lost", "scandal", "corrupt", "reject", "critic",
]

NADI_MAX_PP = 2.0      # had pelarasan setiap gabungan (mata peratus)
NADI_MIN_SEBUTAN = 2   # sebutan minimum sebelum pelarasan dikenakan


def _nadi_skor(tajuk_list: list[str]) -> tuple[dict, dict]:
    """Pure: kira pelarasan (pp) + bilangan sebutan dari senarai tajuk.

    ``tajuk_list``: tajuk dalam huruf kecil. Pulangan:
    ``(pelarasan_pp, sebutan)`` — dict mengikut COALITIONS. Pelarasan
    sentiasa dalam [-2.0, 2.0]; 0.0 jika sebutan < NADI_MIN_SEBUTAN.
    """
    rx = {
        c: [re.compile(r"\b" + re.escape(k) + r"\b") for k in kws]
        for c, kws in NADI_KEYWORDS.items()
    }
    bersih = {c: 0 for c in COALITIONS}
    sebutan = {c: 0 for c in COALITIONS}
    for t in tajuk_list:
        skor = (sum(1 for w in NADI_POSITIF if w in t)
                - sum(1 for w in NADI_NEGATIF if w in t))
        for c in COALITIONS:
            if rx[c] and any(p.search(t) for p in rx[c]):
                sebutan[c] += 1
                bersih[c] += skor
    pelarasan = {}
    for c in COALITIONS:
        if sebutan[c] >= NADI_MIN_SEBUTAN:
            mentah = 2.0 * bersih[c] / sebutan[c]
            pelarasan[c] = round(max(-NADI_MAX_PP,
                                     min(NADI_MAX_PP, mentah)), 2)
        else:
            pelarasan[c] = 0.0
    return pelarasan, sebutan


def nadi_berita(hours: int = 6, timeout: int = 15) -> dict:
    """Pelarasan swing automatik dari sentimen tajuk berita terkini.

    Menarik Google News RSS (hl=ms, gl=MY) untuk query politik Malaysia,
    menapis tajuk dalam ``hours`` jam terakhir, mengira sebutan + sentimen
    setiap gabungan, dan menghasilkan pelarasan swing terhad ±2 mata
    peratus setiap gabungan.

    Pulangan dict:
      - "pelarasan_pp": {gabungan: float} dalam [-2.0, 2.0]
      - "sebutan": {gabungan: int} bilangan tajuk menyebut gabungan
      - "bilangan_tajuk": int — jumlah tajuk unik diimbas
      - "timestamp": datetime (UTC, tz-aware) masa pengiraan, atau None
      - "gagal": bool — True jika fetch gagal
      - "nota": str — penjelasan ringkas / mesej kegagalan

    Jaminan: sebarang kegagalan rangkaian atau parsing menghasilkan
    pelarasan 0.0 untuk semua gabungan + flag "gagal" — fungsi tidak
    pernah mereka data dan tidak pernah menimbulkan pengecualian.
    """
    kosong = {
        "pelarasan_pp": {c: 0.0 for c in COALITIONS},
        "sebutan": {c: 0 for c in COALITIONS},
        "bilangan_tajuk": 0,
        "timestamp": None,
        "gagal": True,
        "nota": "",
    }

    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime

    import requests

    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                             "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
    now = datetime.now(timezone.utc)
    had = now - timedelta(hours=hours)

    tajuk = []
    dilihat = set()
    dapat_respons = False
    try:
        for q in NADI_QUERIES:
            url = (f"https://news.google.com/rss/search?q={quote(q)}"
                   f"%20when%3A1d&hl=ms&gl=MY&ceid=MY%3Ams")
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            dapat_respons = True
            channel = ET.fromstring(r.content).find("channel")
            if channel is None:
                continue
            for it in channel.findall("item"):
                nama = (it.findtext("title") or "").rsplit(" - ", 1)[0].strip()
                if not nama:
                    continue
                kunci = nama.lower()[:60]
                if kunci in dilihat:
                    continue
                try:
                    terbit = parsedate_to_datetime(it.findtext("pubDate"))
                except Exception:  # noqa: BLE001
                    continue
                if terbit is None:
                    continue
                if terbit.tzinfo is None:
                    terbit = terbit.replace(tzinfo=timezone.utc)
                if terbit < had:
                    continue
                dilihat.add(kunci)
                tajuk.append(nama.lower())
    except Exception as exc:  # noqa: BLE001 — graceful: pelarasan = 0
        kosong["nota"] = (f"Gagal tarik berita ({type(exc).__name__}); "
                          "pelarasan = 0.")
        return kosong

    if not dapat_respons:
        kosong["nota"] = "Tiada respons dari Google News RSS; pelarasan = 0."
        return kosong

    pelarasan, sebutan = _nadi_skor(tajuk)
    return {
        "pelarasan_pp": pelarasan,
        "sebutan": sebutan,
        "bilangan_tajuk": len(tajuk),
        "timestamp": now,
        "gagal": False,
        "nota": (f"{len(tajuk)} tajuk dalam {hours} jam terakhir."
                 if tajuk else
                 f"Tiada tajuk berita dalam {hours} jam terakhir; "
                 "pelarasan = 0."),
    }


# ---------------------------------------------------------------------------
# Ujian kendiri
# ---------------------------------------------------------------------------


def _synthetic_data(n_seats: int = 222, seed: int = 7) -> pd.DataFrame:
    """Jana data sintetik 222 kerusi ikut skema CSV — untuk ujian sahaja.

    Taburan negeri dan gabungan penyandang dianggar kasar dari PRU15 sebenar
    (bukan angka tepat), mencukupi untuk menguji bentuk dan logik model.
    """
    rng = np.random.default_rng(seed)
    state_seats = {
        "Johor": 26, "Kedah": 15, "Kelantan": 14, "Melaka": 6,
        "Negeri Sembilan": 8, "Pahang": 14, "Perak": 24, "Perlis": 3,
        "Pulau Pinang": 13, "Sabah": 25, "Sarawak": 31, "Selangor": 22,
        "Terengganu": 8, "W.P. Kuala Lumpur": 11, "W.P. Labuan": 1,
        "W.P. Putrajaya": 1,
    }
    assert sum(state_seats.values()) == n_seats

    # Gabungan penyandang mengikut negeri (anggaran kasar corak PRU15).
    incumbent_map = {
        "Sarawak": (["GPS"] * 23 + ["PH"] * 5 + ["PN"] * 2 + ["BEBAS"] * 1),
        "Sabah": (["GRS"] * 6 + ["PH"] * 7 + ["PN"] * 4 + ["BN"] * 3
                  + ["WARISAN"] * 3 + ["BEBAS"] * 2),
        "W.P. Labuan": ["BN"],
    }
    peninsular = {
        "PH": 0.40, "PN": 0.34, "BN": 0.24, "MUDA": 0.015, "BEBAS": 0.005,
    }

    rows = []
    i = 0
    for state, count in state_seats.items():
        if state in incumbent_map:
            incumbents = incumbent_map[state]
        else:
            coals = list(peninsular.keys())
            probs = np.array([peninsular[c] for c in coals])
            probs /= probs.sum()
            incumbents = rng.choice(coals, size=count, p=probs).tolist()
        for k, coal in enumerate(incumbents):
            i += 1
            voters = int(rng.integers(30000, 150000))
            margin = int(rng.integers(200, int(voters * 0.35)))
            turnout = round(float(rng.uniform(0.60, 0.80)), 4)
            rows.append({
                "kod": f"P{i:03d}",
                "nama_kerusi": f"Kerusi Ujian {i}",
                "negeri": state,
                "parti_pemenang": f"Parti-{coal}-{k + 1}",
                "gabungan": coal,
                "majoriti": margin,
                "peratus_keluar_mengundi": turnout,
                "jumlah_pengundi": voters,
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("== Ujian kendiri forecast_model.py ==")
    df = _synthetic_data()
    assert len(df) == 222, f"jangka 222 baris, dapat {len(df)}"
    assert list(df.columns) == _REQUIRED_COLUMNS
    print(f"[OK] data sintetik: {len(df)} kerusi, lajur betul")

    ms = margin_share(df)
    assert isinstance(ms, pd.Series) and len(ms) == 222
    assert ((ms >= 0) & (ms < 1)).all()
    print(f"[OK] margin_share: min={ms.min():.4f} max={ms.max():.4f}")

    dfp = win_probabilities(df, swing_ph=0.03, swing_pn=-0.02,
                            swing_bn=0.0, youth_swing=0.05)
    prob_cols = [f"prob_{c}" for c in COALITIONS]
    assert all(c in dfp.columns for c in prob_cols + ["proj_gabungan"])
    assert np.allclose(dfp[prob_cols].sum(axis=1), 1.0, atol=1e-9)
    assert set(dfp["proj_gabungan"].unique()) <= set(COALITIONS)
    print(f"[OK] win_probabilities: jumlah kebarangkalian = 1.0 setiap baris")
    print(f"     taburan proj_gabungan: {dfp['proj_gabungan'].value_counts().to_dict()}")

    # Monte Carlo: pastikan deterministik dengan seed sama.
    t0 = time.perf_counter()
    mc1 = monte_carlo(df, swing_ph=0.03, n=2000, seed=42)
    t1 = time.perf_counter()
    mc2 = monte_carlo(df, swing_ph=0.03, n=2000, seed=42)
    elapsed = t1 - t0
    assert mc1 == mc2, "Monte Carlo mesti deterministik dengan seed sama"
    assert set(mc1.keys()) == set(COALITIONS)
    for coal, stats in mc1.items():
        assert set(stats.keys()) == {"median", "p10", "p90"}
        assert all(isinstance(v, int) for v in stats.values())
        assert stats["p10"] <= stats["median"] <= stats["p90"]
    total_median = sum(v["median"] for v in mc1.values())
    assert total_median == 222, f"jumlah median kerusi = {total_median}, jangka 222"
    assert elapsed < 10.0, f"Monte Carlo terlalu perlahan: {elapsed:.2f}s"
    print(f"[OK] monte_carlo n=2000: {elapsed:.2f}s (< 10s), jumlah median = 222")
    print("     contoh median:")
    for coal in COALITIONS:
        s = mc1[coal]
        print(f"       {coal:8s} median={s['median']:3d}  p10={s['p10']:3d}  p90={s['p90']:3d}")

    marg = marginal_seats(df, threshold=0.05)
    assert list(marg.columns) == ["kod", "nama_kerusi", "negeri",
                                  "parti_pemenang", "gabungan", "majoriti",
                                  "margin_share", "peratus_keluar_mengundi"]
    assert (marg["margin_share"] < 0.05).all()
    assert marg["margin_share"].is_monotonic_increasing
    print(f"[OK] marginal_seats: {len(marg)} kerusi marginal (<5%)")

    st = state_projection(dfp)
    assert st.values.sum() == 222
    print(f"[OK] state_projection: {st.shape[0]} negeri, jumlah = {st.values.sum()}")

    # Semak fungsi pure: input tidak diubah.
    df_before = df.copy()
    _ = win_probabilities(df)
    _ = marginal_seats(df)
    pd.testing.assert_frame_equal(df, df_before)
    print("[OK] fungsi tidak mengubah input (pure)")

    # --- Nadi Berita: ujian logik skor (pure, tanpa rangkaian) ---
    adj, sebutan = _nadi_skor([
        "anwar menang besar, sokongan kukuh untuk ph",
        "anwar hebat, pkr berjaya tawan hati pengundi",
        "pas kalah teruk, skandal gagalkan bersatu",
        "sanusi kecewa, pn ditolak pengundi",
    ])
    assert set(adj.keys()) == set(COALITIONS)
    assert all(-2.0 <= v <= 2.0 for v in adj.values()), \
        f"pelarasan mesti dalam ±2pp: {adj}"
    assert adj["PH"] == 2.0, f"PH patut +2.0 (positif penuh), dapat {adj['PH']}"
    assert adj["PN"] == -2.0, f"PN patut -2.0 (negatif penuh), dapat {adj['PN']}"
    assert sebutan["PH"] == 2 and sebutan["PN"] == 2
    assert adj["BEBAS"] == 0.0 and sebutan["BEBAS"] == 0
    print(f"[OK] _nadi_skor: PH={adj['PH']}, PN={adj['PN']}, bound ±2pp dipatuhi")

    # Anti-hingar: satu sebutan sahaja -> pelarasan 0.
    adj1, _ = _nadi_skor(["gps menang besar di sarawak"])
    assert adj1["GPS"] == 0.0, "satu sebutan tidak boleh gerakkan model"
    # Senarai kosong -> semua sifar.
    adj0, seb0 = _nadi_skor([])
    assert all(v == 0.0 for v in adj0.values())
    assert all(v == 0 for v in seb0.values())
    print("[OK] _nadi_skor: anti-hingar (min sebutan) + senarai kosong -> 0")

    # news_adj_pp disalurkan ke model dan mengubah unjuran secara konsisten.
    dfp_n = win_probabilities(df, news_adj_pp={c: 0.0 for c in COALITIONS})
    pd.testing.assert_frame_equal(
        dfp_n[[f"prob_{c}" for c in COALITIONS]],
        win_probabilities(df)[[f"prob_{c}" for c in COALITIONS]],
    )
    dfp_pos = win_probabilities(df, news_adj_pp={"PH": 2.0, "PN": -2.0})
    dfp_base = win_probabilities(df)
    assert (dfp_pos["prob_PH"] >= dfp_base["prob_PH"]).all(), \
        "pelarasan +2pp PH mesti tidak mengurangkan prob PH"
    assert (dfp_pos["prob_PN"] <= dfp_base["prob_PN"]).all(), \
        "pelarasan -2pp PN mesti tidak menambah prob PN"
    mc_n = monte_carlo(df, n=200, seed=1,
                       news_adj_pp={"PH": 2.0, "PN": -2.0})
    assert set(mc_n.keys()) == set(COALITIONS)
    for coal, s in mc_n.items():
        assert 0 <= s["median"] <= 222
        assert s["p10"] <= s["median"] <= s["p90"]
    print("[OK] news_adj_pp: disalur ke win_probabilities + monte_carlo")

    print("== SEMUA UJIAN LULUS ==")
