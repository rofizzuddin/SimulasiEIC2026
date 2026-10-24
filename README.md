# Simulasi Indeks Ketersediaan Terkoreksi Cuaca (EIC 2026)

Kode dan data pendukung esai *"Model Bisnis PLTS Desa Off-Grid Berkelanjutan melalui Skema
Pembayaran Ketersediaan Layanan Berbasis Indeks Terkoreksi Cuaca"* (Izzuddin Ar Rofi', ITB).

## Isi
| Berkas | Keterangan |
|---|---|
| `sim_plts_desa.py` | Simulasi per jam PLTS desa off-grid berbaterai: indeks usulan vs. tiga skema pembanding |
| `nasa.csv` | Data per jam NASA POWER (iradiasi `ALLSKY_SFC_SW_DWN`, suhu `T2M`, waktu LST), 2024–2025, titik 0,683° LU, 104,284° BT (Pulau Sembur Laut) |

## Menjalankan
```bash
pip install numpy pandas matplotlib
python sim_plts_desa.py --weather nasa.csv --month 11 --stress 200
```
Keluaran (±20 detik): Tabel 2, pemeriksaan kriteria kelulusan, `gambar2.png`,
tabel sensitivitas galat sensor, dan hasil uji tekanan 200 kombinasi acak.

## Catatan
- Profil beban adalah **ilustratif** (asumsi penulis, ±700 kWh/hari, puncak malam), bukan data terukur.
- Parameter berlabel `[ASUMSI]` di kode belum divalidasi lapangan; validasi direncanakan pada tahap percontohan.
- Sumber data: NASA POWER Project, https://power.larc.nasa.gov/
