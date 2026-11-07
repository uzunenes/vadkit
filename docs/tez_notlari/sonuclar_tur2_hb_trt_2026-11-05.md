# Sonuçlar: tur 2, HB-POT ve TensorRT (5 Kasım 2026)

Üç çalışmanın üçü de ön kayıtlıydı. Bütün sayılar `paper/ieee_access/kanit/` altında.

## 1. Tur 2: bileşen çalışması ve bütçe kuralı

**Ön kayıt:** `tur2_on_kayit_2026-11-05.md`.
**Sonuçlar:** `kanit/r2_20261105/` (`res/`, `bench/`, `boot/`).

**Gecikme** (RTX 3090, batch 1, sıralı çalışma): 8 yapılandırma 71–75 ms aralığında; MobileCLIP2-S2 90 ms. Süreyi bileşenin boyutu değil Python yükü belirliyor: YOLO11n ile YOLO11m arasındaki fark yalnız 1.7 ms.

**Bütçe kuralı:**
- B1 (≤ 15 ms) ve B2 (≤ 40 ms): hiçbir yapılandırma sığmadı.
- B3 (≤ 100 ms): kural, yuvaları sırayla şu seçeneklerle doldurdu: YOLO11m, DIS, MobileCLIP2-S2, YOLO11n-pose. Poz için iki model eşit gecikmedeydi, kurala göre küçük olan seçildi. Ölçülen gecikme 91.8 ms.

**B3 ile base karşılaştırması** (nedensel mikro AUC):

| | Ped1 | Ped2 | Avenue | ShT |
|---|---|---|---|---|
| base, nesne | 70.9 | 93.9 | 82.8 | 79.5 |
| B3, nesne | 71.1 | 93.4 | 84.5 | 77.2 |
| base, ızgara + nesne | 76.4 | 96.5 | 87.4 | 70.2 |
| B3, ızgara + nesne | 76.7 | 96.3 | 88.5 | 69.3 |

**Hipotezler:**
- **H1** (görünüm farkı daha büyük kodlayıcıyla en az yarı yarıya kapanır): **desteklenmedi.** Avenue'da en iyi kodlayıcı MobileCLIP2-B: 0.748. Farkın kapanan kısmı yaklaşık 0.41 (hafif 0.704, ağır 0.812).
- **H2** (ızgara dalı B2'de 2/3 veri setinde anlamlı katkı verir): **sınanamadı**, çünkü B2'ye sığan yapılandırma yok. Bunun yerine base'de: Ped1'de anlamlı (+5.6); Ped2 ve Avenue'da olumlu ama anlamlı değil; ShT'de anlamlı biçimde olumsuz (−9.3).
- **H3** (B3, AccI-VAD'ın nedensel sonucunun en fazla 1 puan altında kalır): **desteklenmedi.** (Rakibin bizim protokolümüzdeki sonuçları bu açık kopyada yer almaz; makale rakipleri yalnız yayımlanmış değerleriyle karşılaştırır.)

**Okuma:** Daha büyük bileşenler doğruluğu kayda değer artırmıyor. Hafif `base` yapılandırması ana sistem olarak kalıyor. Bu da makaledeki "basit ve hafif" mesajını destekliyor.

## 2. HB-POT: ızgara dalı için hiyerarşik Bayesçi kalibrasyon, Aşama 1

**Ön kayıt:** `hiyerarsik_kalibrasyon_on_kayit_2026-11-05.md`.
**Sonuçlar:** `kanit/hb_20261105/` (`gate.json`, `results.json`).

**Kapı** (yalnız normal veri; test değerlendirmesinden önce commit edildi): G1–G4'ün dördü de tutmadı.
- REML, Fd'de kuyruk biçimi ξ'nin sahneler arası yayılımını sıfır buldu. Böylece bütün sahnelere ortak ξ = 0.405 atandı.
- Küçük sahne 07'nin kuyruk ölçeği veri setinin ortalamasına çekildi; bu sahnede yanlış alarm arttı.
- Havuzlamasız GPD (T1) Fd'de 10'un üstündeki z sayısını 190'dan 22'ye indirdi.

**Test** (tek koşu; T0 değerleri dört veri setinde de yeniden üretildi):

| Hipotez | Sonuç |
|---|---|
| H1: ShT, T2 − T0 | +0.0015 [−0.0005, +0.0039], **tutmadı** |
| H1b: T2 − T1 | +0.0002, tutmadı |
| H2: ShT, T2 − nesne | −0.092 [−0.138, −0.047], ızgara zararı sürüyor |
| H3a: hata kontrolü | geçti; en büyük makro farkı 3e-6 |
| H4: tek sahne koruması | geçti (en kötü Ped1, −0.0008) |
| H5: makro koruması | geçti (0.823) |

**Karar:** T2 benimsenmiyor, mevcut Fisher kalibratörü kalıyor. Aşama 1 olumsuz sonuç olarak raporlanacak.

**Yorum:** ShT'deki kayıp kalibrasyondan değil, kanalların kendisinden geliyor. I kanalının ShT'de tek başına mikro AUC'si 0.558. Aşama 2'nin hedefi bu: hücre → sahne → veri seti NIW hiyerarşisi (ayrı ön kayıtla).

## 3. TensorRT (hız, C2)

**Ön kayıt:** `hiz_on_kayit_2026-11-05.md`.
**Sonuçlar:** `kanit/hiz_20261105/c2*/`.

**Adım 1, hız** (RTX 3090, Avenue):
- YOLO11n TensorRT FP16'da 8.4 ms'den 2.8 ms'ye, poz modeli 12.3 ms'den 2.9 ms'ye indi.
- MobileCLIP2-S0'ın TensorRT motoru ONNX çıktısından sapıyor: kosinüs yaklaşık 0.94. Bu FP32'de, TF32 kapalıyken ve FP16'da aynı. Sapmanın PyTorch TF32'den gelmediği kontrol edildi (gerçek kırpmalarda kosinüs 0.99998). Bu yüzden kodlayıcı PyTorch'ta kaldı.
- TensorRT YOLO + PyTorch kodlayıcı + paralel akış: **24.8–26.7 FPS** (medyan 37.5–40.4 ms). PyTorch'ta paralel akışla 18–20 FPS ölçülmüştü, kazanç yaklaşık %30. Ön kayıttaki %20 eşiği geçildi.

**Adım 2, doğruluk** (Ped1 geliştirme seti yeniden çıkarıldı):

| | PyTorch | TensorRT | Fark |
|---|---|---|---|
| Nesne, mikro | 0.7086 | 0.7105 | +0.0019 |
| Izgara + nesne, mikro | 0.7641 | 0.7647 | +0.0005 |

Fark ±0.005 sınırının içinde. Ön kayda göre TensorRT arka ucu **eşdeğer** sayılıyor ve makaledeki hız bu arka uçla verilebilir.

Ön kayıt, eşdeğerlik çıkarsa Ped2, Avenue ve ShT'nin de TensorRT ile yeniden koşulup raporlanmasını istiyor. Bu, bir sonraki GPU turunda yapılacak.

**Adım 2'nin devamı:** Ön kayda göre üç veri seti TensorRT ile yeniden koşuldu (`kanit/r2_20261105/res/base_trt/`). Nedensel mikro AUC, PyTorch → TensorRT:

| | Ped2 | Avenue | ShT |
|---|---|---|---|
| Nesne | 0.9385 → 0.9381 | 0.8278 → 0.8324 | 0.7954 → 0.7923 |
| Izgara + nesne | 0.9653 → 0.9653 | 0.8738 → 0.8800 | 0.7020 → 0.7012 |

- Farkların hepsi ±0.62 puanın içinde, ama işaretleri sabit değil.
- Sıralı çalışmada TensorRT gecikmesi 60.6 ms; aşamalara göre: dedektör 5.2, poz 3.8, akış 27.9 (CPU), gömme 20.4 ms.
- Akış paralel çalıştırıldığında 24.8–26.7 FPS ölçülmüştü (`kanit/hiz_20261105/c2/trtyolo`).
