# Hakem düzeltmeleri: ön kayıt (6 Kasım 2026)

**Amaç:** `hakem_degerlendirmesi_2026-11-06.md`'deki eksikleri kapatacak ölçümleri, sonuçları görmeden tanımlamak. Bu dosya ölçümler koşmadan önce commit'lenir ve bağlayıcıdır.

**Genel kurallar:**
- Ana sistem bu sonuçlara göre **değişmez**: `base` yapılandırması kalır; ShT'de yalnız nesne dalı kullanılır (†).
- Hangi yönde çıkarsa çıksın her satır raporlanır. Lehimize olmayan sonuç da yazılır.
- Test etiketine bakılarak hiçbir seçim yapılmaz. Hiperparametreler yalnız normal eğitim verisinde seçilir, mevcut sistemle aynı yolla.
- Daha önce ölçülmüş ve makalede kullanılmamış kanıtlar da kullanılacak (aşağıda K bölümü).

## S1. Skorlama ablation'ı (nesne dalı, CPU)

**Veri:** Arşivlenmiş `base` özellikleri (`data/r2_feat/base`, Tablo 1'deki sayıların özellikleri). Ped1, Ped2, Avenue ve ShT'nin dördü de kullanılır. ShT'de modeller sahne başına kurulur, mevcut sistemdeki gibi.

**Protokol:**
- Online micro AUC: nedensel medyan 9, warmup 4. Ayrıca macro AUC ve standart protokol.
- Her satır için NIW satırına (D1) karşı eşleştirilmiş video bootstrap uygulanır: 2000 tekrar, seed 0, `vadkit.eval.protocol.paired_bootstrap`.

**Satırlar.** Kalibrasyon (Fisher) ve füzyon (toplam), aksi belirtilmedikçe sabit kalır. Aynı skorlayıcı üç niteliğin üçüne de uygulanır.

| Kod | Değişiklik |
|---|---|
| D1 | NIW karışımı, normal veride seçilmiş (K, κ0, ν0): mevcut sistem. **Yeniden üretim kontrolü:** `kanit/r2_20261105/res/base/<ds>_obj.json`'daki füzyon online AUC'si 1e-6 içinde tutmazsa S1 durur. |
| D2 | ML-GMM: aynı K, tam kovaryans, `reg_covar` 1e-6, k-means başlangıcı, `random_state` 0 (sklearn `GaussianMixture`). Skor −log p. |
| D2b | Prior'sız ikiz: `aean.niw.PluginMixture`. NIW ile aynı K ve aynı k-means bileşenleri (seed 0); her bileşen ML Gaussian + `NIWCells` ridge'i; ağırlıklar n_k/N. Yalnız NIW prior'ının etkisini ayırır. *Ek (6 Kasım, ilk commit'ten sonra, hiçbir sonuç görülmeden).* |
| D3 | Tek Gaussian (K = 1, ML, `reg_covar` 1e-6). |
| D4 | kNN: eğitim satırlarına kesin 1-NN Öklid uzaklığı (faiss `IndexFlatL2`). Skor uzaklıktır; AccI-VAD'ın skorlayıcısına karşılık gelir. Referans skorlar, NIW'deki gibi 2-katlı çapraz uydurmayla elde edilir. Ek olarak bellek (MB) ve CPU'da 4 thread ile ms/kare ölçülür; aynı ölçüm D1 için de yapılır. |
| C1 | Füzyon: Fisher yerine z-skor kalibrasyonu, toplam. |
| C2 | Füzyon: Fisher kalibreli kanalların maksimumu. |
| T1 | Zamansal düzleştirme yok (medyan k = 1). |
| L1–L3 | Tek nitelik çıkarılmış: hız, poz, görünüm. |
| F1 | f = 0.25 (veri verimliliği ölçümündeki 5 çekiliş, `default_rng(1000 + seed)`, seed 0–4): D1, D2 ve D4. Satırlar f = 1'deki kendi değerleriyle karşılaştırılır. |

Tek nitelikli satırlar zaten `res/base` içinde; tabloda bunlar yeniden üretilmiş hâlleriyle verilir.

**Hiperparametreler:** Seçilen K, κ0 ve ν0, veri seti ve nitelik başına raporlanır; ShT için sahneler arası aralık verilir.

**Yorum kuralı:** "NIW daha iyi" ifadesi yalnız fark güven aralığı sıfırı NIW lehine dışlarsa kullanılır. Dışlamazsa "fark yok" denir; ters yönde dışlarsa "D2 veya D4 önde" denir. Makalenin konumlandırması buna uyar: NIW kazandırmazsa kazanç bellek ve hız olarak anlatılır.

## S2. GPU ölçümleri (vast.ai, RTX 3090)

**Ön koşul:** Her ölçümden önce GPU'nun boşta olduğu `nvidia-smi` ile kontrol edilir; AED-MAE hız protokolündeki gibi. Bu ölçümlerden doğruluk iddiası çıkarılmaz.

- **G1, ızgara dalı gecikmesi.**
  - Araç: `tools/bench_pipeline.py`. Ped2 ve Avenue'da; GPU'da (`--device cuda`) ve CPU'da 4 thread ile; değerlendirmedeki stride ile.
  - Ayar: 3 tekrar × 100 kare; medyan ve p95.
  - Ped2'nin CPU ölçümü önceden yapılmıştı (`kanit/public_20261102/latency_ped2_cpu*.json`); aynı makinede yeniden ölçülür.
  - Rapor: ızgara dalının kare başına süresi. Bu, nesne dalıyla birlikte koşan tam sistemin gecikmesi değildir; birleşik bir streaming uygulaması yok ve bu açıkça yazılır.
- **G2, kurulum süresi.**
  - Ölçülen: eğitim bölümünün `vadkit extract` duvar saati süresi. Tek süreç, GPU, `base` yapılandırması, makaledeki `train_every` (1; ShT'de 5), dört veri seti.
  - Ek olarak aynı makinede `vadkit fit` süresi.
  - Rapor: dakika cinsinden özellik çıkarma, uydurma ve ikisinin toplamı.
  - Izgara autoencoder'ının eğitim süresi önceki loglardan alınır (Ped1/Ped2: 3.7–3.8 dk); Avenue için ölçüm yok, "ölçülmedi" yazılır.
- **G3, gecikme dağılımı.**
  - Araç: `vadkit bench`, dört veri setinin test kareleri. İki arka uç: TensorRT YOLO + paralel akış (makaledeki hız yapılandırması) ve PyTorch + paralel akış.
  - Ayar: warm-up 20, 3 × 100 kare.
  - Rapor: p50, p95 ve p99 toplam gecikme; kare başına nesne sayısı (ortalama ve maksimum); toplam süre ile nesne sayısı arasında Spearman korelasyonu.
  - Kod değişikliği: `bench` çıktısına kare başına nesne sayısı ve p99 eklenir. Zamanlama kodu değişmez.

## K. Mevcut, makaleye alınacak kanıt (yeni ölçüm değil)

- **AccI-VAD'ın kendi özellikleri üzerinde orijinal skorlama ile NIW karşılaştırması** (online protokol). Kaynak: `kanit/vast_20261104/sonuclar/accivad_bayes/`, v2 `gen/tab_bayes.tex`.
- **Skorlama maliyeti** (Avenue görünüm niteliği, 113,751 eğitim nesnesi). Kesin NN (FAISS): 222.2 MiB ve 39.6 ms/kare. NIW: 4.3 MiB ve 0.60 ms/kare. Kaynak: `kanit/vast_20261104/sonuclar/scoring_cost_avenue_clean*.json`.

## M. Metin düzeltmeleri (yeni sayı içermez)

- **Online (madde 2):**
  - Online, sistemin bir özelliği olarak sunulur.
  - Tablo 1'deki boş online sütunları kaldırılır; bizim online sayılarımız ayrı satırda verilir.
  - MOVAD (Doshi & Yilmaz, WACV 2022), "Look at Adjacent Frames" (Ouyang ve ark., ECCVW 2022) ve Doshi & Yilmaz (WACV 2022, continual) atıf alır.
  - Rakipler online'a çevrilmez (danışman kararı, 5 Kasım).
- **Gönderime hazırlık (madde 3):**
  - Metindeki iç dosya yolları ve "Generated by" kuyrukları kaldırılır; yerine tek bir *Code and Data Availability* bölümü konur.
  - Ön kayıtlar commit hash ve tarihleriyle anılır.
  - Seçilen hiperparametreler ve seed'ler eklenir; float yerleşimi düzeltilir.
  - Ford adresi, fon beyanı ve kod deposu adresi kullanıcıdan gelecek; o zamana kadar TODO kalır.
- **Yapılandırmalar (madde 4):**
  - Tablo 1'de iki adlı satır olur: nesne dalı ve nesne + ızgara.
  - Her sayının yanında yapılandırma ve arka uç belirtilir.
- **Rakiplere karşı iddialar (madde 5):**
  - Tablo 2'ye rakiplerin yayımlanmış hızı eklenir; değer yoksa "raporlanmamış" yazılır.
  - AED-MAE ve Fast-AED'nin daha küçük olduğu açıkça yazılır.
  - İddia daraltılır: "eğitimsiz ve tek sınıf nesne-merkezli yöntemler arasında".
  - Ped2/Avenue'daki autoencoder eğitimi belirtilir; FastAno atıf alır.
- **Diğer:**
  - "Aynı yapılandırma" cümlesine ShT istisnası eklenir.
  - Nedensel medyanın 4 karelik gecikmesi yazılır.
  - Başlıktaki "Bayesian" ifadesi danışmana sorulacak; başlık değişmez.
