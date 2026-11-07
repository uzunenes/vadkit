# Tur 2 ön kaydı: bileşen Pareto çalışması (5 Kasım 2026, sonuç görülmeden yazıldı)

## Neden bu tasarım

Pilotta en iyi hız özelliğinin veri setine göre değiştiği görüldü. Bileşenleri test AUC'sine bakarak seçmek, test setine fazla uyum demek olur. Bu turda **seçim hesaplama bütçesiyle yapılır**:
- Her bileşenin hem doğruluğu hem gecikmesi ölçülür.
- Sistem yapılandırmaları, aşağıda önceden yazılmış süre bütçelerine sığan en pahalı seçenekten oluşur.
- Bütün sonuçlar, olumsuz olanlar dahil, raporlanır.

## Bileşen yuvaları ve seçenekleri

| Yuva | Seçenekler |
|---|---|
| Dedektör (+ hareket konturu kutuları) | YOLO11n, YOLO11s, YOLO11m |
| Poz | YOLO11n-pose, YOLO11s-pose |
| Akış (hız niteliği, 8 yön kutusu) | DIS (CPU), NeuFlow v2, RAFT-large |
| Ek hız kanalı | ByteTrack kutu yörüngesi (akışsız) |
| Görünüm kodlayıcı | MobileCLIP2-S0, MobileCLIP2-S2, MobileCLIP2-B, DINOv2 ViT-S/14, CLIP ViT-B/16 |
| Puanlayıcı (sabit) | NIW karışımı, normal veride seçim, Fisher kalibrasyonu, toplam füzyon, nedensel medyan 9 |
| Izgara dalı (hocanın hattı) | yok / I + Fs + Fd hücre kanalları |

## Ölçümler

- **Bileşen başına:** tek başına nitelik AUC'si (Ped1, Ped2, Avenue, ShT; nedensel, mikro ve makro) ve RTX 3090'da batch 1 gecikmesi. DIS için ayrıca 4 çekirdek CPU. Ölçüm 20 ısınma + 3×100 kare, medyan ve p95.
- **Sistem başına:** üç ya da dört kanallı füzyon AUC'si, ızgara dalıyla ve dalsız; video bootstrap ile güven aralığı.

## Önceden yazılan bütçeler (RTX 3090, kare başına, algılama + nitelik çıkarımı)

| Nokta | Bütçe | Amaç |
|---|---|---|
| B1 | ≤ 15 ms | gömülüye taşınabilir en hafif nokta (tez) |
| B2 | ≤ 40 ms | "hafif" makale noktası (≥ 25 FPS, T2 hedefi) |
| B3 | ≤ 100 ms | hafif çizginin üst sınırı; ağır hatla karşılaştırma |

**Seçim kuralı:** Her bütçe için yuvalar tek tek doldurulur. Sıra dedektör, akış, kodlayıcı, poz; her yuvaya bütçeye sığan en pahalı seçenek konur.

- Gecikmeler eşitse küçük model seçilir.
- Bütçe kuralı birden fazla yapılandırma bırakırsa, kalan seçim **Ped1 geliştirme setinde** yapılır.
- Ped2, Avenue ve ShT hiçbir seçimde kullanılmaz.

## Hipotezler (önceden yazıldı)

- **H1 (görünüm):** Avenue'da hafif görünüm (MobileCLIP2-S0, 0.716) ile ağır görünüm (0.812) arasındaki fark, daha büyük kodlayıcılarla en az yarı yarıya kapanır.
- **H2 (hocanın hattı):** Izgara dalı, B2 noktasında füzyonu en az 2/3 veri setinde anlamlı olarak iyileştirir. Pilottaki veri ShT'de tersini gösterdi; bu da raporlanacak.
- **H3 (T1):** B3 noktası AccI-VAD'ın nedensel sonucunun (0.983 / 0.915 / 0.842) en fazla 1 puan altında kalır.

Hipotezler doğrulanmazsa sonuç olduğu gibi raporlanır.
