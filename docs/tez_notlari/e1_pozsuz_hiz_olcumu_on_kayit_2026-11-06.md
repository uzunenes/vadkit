# E1 eki: pozsuz yapılandırmanın hız ve kurulum ölçümü, ön kayıt (6 Kasım 2026)

**Bağlam:** E1 ön kaydının karar maddesi şunu söylüyor: "Poz kapalı veri setlerinde ölçülen parametre sayısı ve hız da raporlanır." Bu dosya, o ölçümün nasıl yapılacağını ölçüm koşmadan önce sabitler. Ölçüm doğruluk iddiası içermez.

**Yapılandırmalar:** `vadkit/configs/r2/base_nopose*.yaml` (dört dosya). Bunlar `base*.yaml` ile aynıdır; yalnız `pose` bloğu ve `opose` niteliği çıkarılmıştır. Poz modeli hiç yüklenmez ve çalıştırılmaz.
- Laptop kontrolü: Ped2'de `base_nopose.yaml` ile uydurulan nesne dalının online AUC'si, E1'in `obj_gated` değerine 1e-6 içinde eşit olmalı.

**Ölçüm (tek GPU oturumu, vast.ai, tercihen RTX 3090):**
- **Hız.** Araç ve ayar G3 ile aynıdır: `vadkit bench`, `--parallel-flow`, warm-up 20, 3 × 100 kare; TensorRT YOLO ve PyTorch arka uçları.
  - Veri setleri: Ped1, Ped2 ve ShanghaiTech test kareleri.
  - Her veri setinde **aynı oturumda** iki yapılandırma ölçülür: pozlu (`base*`) ve pozsuz (`base_nopose*`). Fark böylece aynı makinede okunur.
  - Rapor: p50, p95 ve p99 toplam gecikme, FPS (tekrar medyanlarının medyanı), kare başına nesne sayısı. Pozlu ile pozsuz arasındaki p50 farkı ayrıca verilir.
  - G3 ile aynı biçimde, her ölçümden önce GPU'nun boşta olduğu ve hostun yük ortalaması kaydedilir.
- **Kurulum.** `vadkit fit` süresi, GPU'da, üç veri setinde pozlu ve pozsuz. Özellik çıkarma yeniden ölçülmez: pozsuz çıkarma ayrı bir ölçüm ister ve bu turda yapılmaz.
- **Parametre sayısı.** `kanit/boyut_20261105/size_ours.json`'dan hesaplanır: YOLO11n + MobileCLIP2-S0 görüntü kulesi, poz modeli olmadan.

**Raporlama:** Sonuç ne çıkarsa çıksın yazılır. Pozsuz yapılandırmanın daha yavaş çıkması da dahil.
