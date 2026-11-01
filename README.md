# 🇲🇾 Dashboard Ramalan Politik Malaysia PRU16

Aplikasi Streamlit interaktif untuk unjuran kerusi Parlimen Malaysia menuju
**PRU16** — 100% teks UI dalam Bahasa Melayu.

Unjuran dikira secara langsung dari **data keputusan PRU15 sebenar (222 kerusi)**
ditambah andaian swing yang anda laraskan melalui slider. Model adalah
**ilustratif** — bukan ramalan muktamad atau nasihat pengundian. Lihat tab
📐 Metodologi dalam aplikasi untuk formula penuh dan batasan.

## Ciri utama

| Tab | Kandungan |
|---|---|
| 🏠 Ramalan Kerusi | Kad KPI setiap gabungan (unjuran + julat Monte Carlo), carta parlimen hemicycle 222 kerusi, jadual 222 kerusi (carian + tapis negeri), kerusi marginal & super marginal |
| 📊 Pecahan Negeri | Carta bar bertindan mengufuk: unjuran kerusi mengikut negeri |
| 🧭 Halatuju Politik | Kad neutral 3 blok (Kerajaan Perpaduan / PN / Blok Borneo), garis masa ke PRU16, countdown ke bubar automatik 19 Dis 2027, jadual Janji vs Pelaksanaan (label editorial) |
| 💬 Sentimen Awam | Peranan setiap platform media sosial + trend kata kunci dari tajuk berita (dilabel proksi, bukan data media sosial) |
| 📰 Berita Terkini | Google News RSS — politik Malaysia, PRU16, pilihan raya Malaysia, PRN Melaka (refresh 30 minit) |
| 📐 Metodologi | Formula model, sumber data, 4 batasan, nota KDM/PBM |

Fakta latar: Parlimen bubar automatik **19 Dis 2027**; PRU16 selewat-lewatnya
**17 Feb 2028**.

## Struktur projek

```
pru16-dashboard/
├── app.py                 # Aplikasi Streamlit
├── forecast_model.py      # Model ramalan (fungsi pure, tiada data direka)
├── requirements.txt
├── README.md
└── data/
    └── pru15_results.csv  # Keputusan PRU15 sebenar, 222 kerusi (sumber: Wikipedia)
```

## Cara run secara lokal

```bash
cd pru16-dashboard
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi dibuka di `http://localhost:8501`.

## Deploy ke Streamlit Cloud

1. Push folder ini ke repo GitHub (cth. `hijazimaju/pru16-dashboard`).
2. Pergi ke [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pilih repo → branch `main` → main file path: `app.py` → **Deploy**.

(Pattern sama seperti dashboard bencana sedia ada.)

## Sumber data

- **Keputusan PRU15:** Wikipedia — *Results of the 2022 Malaysian general
  election by parliamentary constituency*
  (https://en.wikipedia.org/wiki/Results_of_the_2022_Malaysian_general_election_by_parliamentary_constituency)
- **Berita:** Google News RSS (`hl=ms`, `gl=MY`), dikemas kini setiap 30 minit.

## Batasan

- Model berasaskan penyandang PRU15 + swing nasional yang ditetapkan pengguna;
  ia tidak tahu tentang persempadanan semula, pertandingan berbilang penjuru
  sebenar, atau calon spesifik.
- Nilai swing, kekuatan penyandang dan hingar simulasi adalah **ilustratif**,
  bukan anggaran statistik.
- Kekuatan penyandang dianggar linear dari margin PRU15.
- Output adalah distribusi kebarangkalian (Monte Carlo 2,000 ulangan),
  bukan ramalan pasti.
- P168 Kota Marudu (KDM) dan P209 Julau (PBM) dimodelkan sebagai Bebas;
  label asal dikekalkan dalam jadual.
- Tiada andaian lompat parti — baseline ialah keputusan PRU15.
