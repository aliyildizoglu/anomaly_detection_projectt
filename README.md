# Anomaly Detection Project

Bu proje, farklı veri setlerinde anormal (anomaly) verileri tespit etmek için Python tabanlı makine öğrenmesi ve derin öğrenme yöntemleri kullanan kapsamlı bir anormallik tespit sistemidir.

---

## Özellikler

- Veri temizleme ve ön işleme  
- Çoklu anormallik tespiti algoritmaları:
  - Isolation Forest  
  - One-Class SVM  
  - Autoencoder (derin öğrenme)  
- Model eğitimi, doğrulama ve test süreçleri  
- Sonuçların grafiklerle görselleştirilmesi (matplotlib, seaborn)  
- Performans metrikleri: Precision, Recall, F1-Score, ROC-AUC  
- Modüler ve kolay genişletilebilir yapı  
- Farklı veri setlerine uygulanabilirlik  

---

## Kullanılan Veri Setleri

- **KDD Cup 1999:** Ağ saldırıları ve anormallik tespiti için yaygın kullanılan benchmark veri seti.  


(*Not: Veri setleri projede yer almamakta olup, kullanıcı tarafından temin edilmelidir.*)

---

## Kullanılan Teknolojiler ve Kütüphaneler

Projenin çalışması için aşağıdaki Python paketlerine ihtiyacınız vardır:

- **Python 3.8+**  
- `pandas` — Veri işlemleri  
- `numpy` — Sayısal hesaplamalar  
- `scikit-learn` — Makine öğrenmesi algoritmaları (Isolation Forest, One-Class SVM)  
- `tensorflow` ve `keras` — Derin öğrenme (Autoencoder)  
- `matplotlib` ve `seaborn` — Veri görselleştirme  
- `jupyter` — Prototip geliştirme ve analiz  

Bu paketleri yüklemek için terminal veya komut satırına şu komutları yazabilirsiniz:

```bash
pip install pandas numpy scikit-learn tensorflow keras matplotlib seaborn jupyter
