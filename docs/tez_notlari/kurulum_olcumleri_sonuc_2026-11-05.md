# Kurulum ölçümleri: M1 ve M2 sonuçları (5 Kasım 2026)

**Ön kayıt:** `kurulum_olcumleri_on_kayit_2026-11-05.md` (commit `e4031e4`, sonuçlardan önce yazıldı).
**Ham sonuçlar:** `paper/ieee_access/kanit/kurulum_20261105/` (`far_*.json`, `dataeff_*.json`, `logs/`).
**Betik:** `vad_bench/vast/runs_r2/deploy_measures.py`.
**Sistem:** Tur 2 `base` yapılandırması (YOLO11n, YOLO11n-pose, DIS, MobileCLIP2-S0, NIW, Fisher, toplam, nedensel medyan 9, warmup 4). Yalnız nesne dalı ölçüldü.

## M1. Etiketsiz alarm eşiği

**Yöntem:** Eşik, çapraz uydurulmuş normal eğitim referansının (1 − α) kantili olarak, kamera başına bir tane seçilir. Tablodaki değerler testteki normal karelerde gerçekleşen yanlış alarm oranıdır (FAR); parantez içinde FAR / α var. Köşeli parantezler, video bootstrap'iyle hesaplanan %95 güven aralığıdır.

| Veri | Test normal / anomali kare | α = %10 | α = %5 | α = %1 | α = %0.1 |
|---|---|---|---|---|---|
| Ped1 | 3059 / 3997 | %13.2 (1.3) | %7.6 (1.5) [4.5, 11.8] | %4.0 (4.0) [1.9, 7.1] | %2.6 (26) |
| Ped2 | 342 / 1620 | %17.0 (1.7) | %8.8 (1.8) [1.5, 24.3] | %3.8 (3.8) [0, 14.3] | %2.1 (20) |
| Avenue | 11540 / 3700 | %39.3 (3.9) | %29.8 (6.0) [12.1, 51.1] | %13.1 (13) [1.2, 27.0] | %0.55 (5.5) |
| ShT (12 kamera, ikincil) | 23079 / 17284 | %7.5 (0.75) | %3.6 (0.72) [2.1, 5.3] | %1.4 (1.4) [0.7, 2.4] | %0.24 (2.4) |

**Aynı eşikte tespit oranı (α = %5):** Ped1 0.34, Ped2 0.82, Avenue 0.82, ShT 0.30.

**Ön kayıt ölçütü:** α = %5 ve α = %1 için FAR / α oranının 0.5 ile 2 arasında kalması. Hipotez, bunun ana kümenin (Ped1, Ped2, Avenue) en az 2/3'ünde tutmasıydı.

**Sonuç:** Hipotez **desteklenmedi.**
- α = %5'te Ped1 ve Ped2 bandın içinde, Avenue değil.
- α = %1'de ana kümenin hiçbiri bandın içinde değil.

**İkincil bulgu:** ShT'de eşik sahne başına (kamera başına) seçildiğinde iki α değeri de bandın içinde kalıyor (0.72 ve 1.42). Ancak ShT, eğitim referansı alt örneklendiği için ön kayıtta ikincil olarak işaretlenmişti.

**Analitik Gamma(k, 1) eşiği:** Nesne dalında ampirik eşiğe yakın sonuç veriyor. Izgara + nesne dalında ise ciddi biçimde kayıyor: α = %5'te FAR / α oranı Ped2'de 2.3, ShT'de 5.3, Avenue'da daha yüksek. Kanallar bağımsız olmadığı için bu beklenen bir sonuç.

**Olası nedenler (sınanmadı):**
- Avenue'da test normal kareleri ile eğitim normal kareleri arasında dağılım kayması var (kamera sarsıntısı, farklı kalabalık).
- Anomali aralıklarının hemen bitişindeki kareler, nedensel medyan yüzünden yüksek skor taşıyor.
- Ped2'de test normal karesi çok az (342), bu yüzden güven aralığı çok geniş.

**Makale için:** "Etiketsiz eşik" manşet iddiası olamaz. Sınırlılık olarak raporlanacak. ShT'deki kamera başına sonuç bir ipucu, ama kanıt değil.

## M2. Veri verimliliği

**Yöntem:** Eğitim videolarının f oranı, en az 2 video olacak şekilde alındı (ShT'de sahne başına). Her oran için 5 tohum kullanıldı, f = 1.0 tek koşu. Tablodaki değerler nedensel mikro AUC'nin ortalama ± std'sidir; parantez içinde kullanılan normal video süresi var.

| Veri | f = 0.1 | f = 0.25 | f = 0.5 | f = 1.0 |
|---|---|---|---|---|
| Ped1 | 0.688 ± 0.010 (1.3 dk) | 0.699 ± 0.005 (3.0 dk) | 0.706 ± 0.004 (5.7 dk) | 0.709 (11.3 dk) |
| Ped2 | 0.937 ± 0.005 (0.5 dk) | 0.944 ± 0.006 (1.0 dk) | 0.938 ± 0.003 (2.0 dk) | 0.939 (4.2 dk) |
| Avenue | 0.840 ± 0.023 (1.2–1.9 dk) | 0.833 ± 0.015 (1.9–3.7 dk) | 0.799 ± 0.037 (4.5–5.9 dk) | 0.828 (10.2 dk) |
| ShT (mikro) | 0.742 ± 0.016 (22–26 dk) | 0.726 ± 0.013 (48–55 dk) | 0.766 ± 0.016 (97–102 dk) | 0.795 (191 dk) |
| ShT (makro) | 0.809 ± 0.010 | 0.810 ± 0.006 | 0.813 ± 0.008 | 0.819 |

**Ön kayıt hipotezi:** f = 0.25 ile mikro AUC, tam verinin en fazla 2 puan altında kalır; bunun en az 3/4 veri setinde tutması beklenir.

**Sonuç:** Hipotez **desteklendi (3/4).**
- Ped1: −1.0 puan
- Ped2: +0.5 puan
- Avenue: +0.5 puan
- ShT: −6.9 puan (tutmadı)

ShT'nin makro AUC'si ise yalnız −0.9 puan düştü. Bu da düşüşün sahneler arası kalibrasyondan geldiğini düşündürüyor; ızgara dalının ShT sorunuyla aynı belirti. Sınanmış bir açıklama değil.

**Makale için:** UCSD ve Avenue'da 1–4 dakikalık normal video, tam verinin sonucunu veriyor. Bu, "kolay kurulum" iddiasının ölçülmüş kanıtı.

**Uydurma süreleri** (`fit_seconds` alanı) bu koşuda **geçersiz.** Makine aşırı yüklüydü: tur 2'nin 16 işçisi ve bu koşunun 20 süreci aynı anda çalışıyordu, yük ortalaması 36–47. Ön kayıtta yazıldığı gibi, süreler tur 2 bittikten sonra sessiz makinede yeniden ölçülecek.
