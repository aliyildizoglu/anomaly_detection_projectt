import pyshark
import pandas as pd

pcap_path = "anomaly_detection_project/uploads/kendi_trafik.pcap"
#pcap_path = "anomaly_detection_project/uploads/kendi_trafik_2.pcap"

# Tam içerikle oku (no summary)
cap = pyshark.FileCapture(pcap_path, use_json=True)

packets = []
for i, pkt in enumerate(cap):
    if i >= 300:
        break
    try:
        layer = pkt.highest_layer
        packets.append({
            'time': pkt.sniff_time,
            'protocol': layer,
            'src_ip': pkt.ip.src if hasattr(pkt, 'ip') else None,
            'dst_ip': pkt.ip.dst if hasattr(pkt, 'ip') else None,
            'src_port': pkt[pkt.transport_layer].srcport if hasattr(pkt, 'transport_layer') else None,
            'dst_port': pkt[pkt.transport_layer].dstport if hasattr(pkt, 'transport_layer') else None,
            'length': pkt.length
        })
    except Exception:
        continue

cap.close()
df = pd.DataFrame(packets)

df['src_ip'] = df['src_ip'].apply(lambda x: "HIDDEN_SRC" if pd.notnull(x) else x)
df['dst_ip'] = df['dst_ip'].apply(lambda x: "HIDDEN_DST" if pd.notnull(x) else x)

df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M')

print(df.head())
print(f"\nToplam alınan paket sayısı: {len(df)}")

#Veri Analizi Kodları
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Anonimleştirilmiş dataframe burada varsayılıyor: df

# 🎯 Protokol dağılımı
protocol_counts = df['protocol'].value_counts()
print("🔌 Protokol Dağılımı:\n", protocol_counts)

plt.figure(figsize=(6,4))
sns.countplot(data=df, x='protocol')
plt.title("Protokol Dağılımı")
plt.ylabel("Paket Sayısı")
plt.xlabel("Protokol")
plt.tight_layout()
plt.show()

# 🛠️ En çok kullanılan portlar (ilk 10)
print("\n🔧 En Sık Kullanılan 10 Hedef Port:")
print(df['dst_port'].value_counts().head(10))

# 📦 Paket uzunluğu histogramı
df['length'] = pd.to_numeric(df['length'], errors='coerce')

plt.figure(figsize=(6,4))
sns.histplot(df['length'], bins=20, kde=True)
plt.title("Paket Uzunluğu Dağılımı")
plt.xlabel("Uzunluk (byte)")
plt.ylabel("Paket Sayısı")
plt.tight_layout()
plt.show()

#Isolation Forest ile Anomali Tespiti
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 🎯 Girdi: df (senin mevcut anonimleştirilmiş DataFrame'in)

# 1. Gerekli sütunları seç ve hazırlık yap
df_model = df[['protocol', 'dst_port', 'length']].copy()

# 2. Protokol kategorisini sayısal hale getir
le = LabelEncoder()
df_model['protocol'] = le.fit_transform(df_model['protocol'])

# 3. dst_port ve length sayıya çevrilsin
df_model['dst_port'] = pd.to_numeric(df_model['dst_port'], errors='coerce')
df_model['length'] = pd.to_numeric(df_model['length'], errors='coerce')
df_model.dropna(inplace=True)

# 4. Isolation Forest modeli ile anomalileri tespit et
iso_model = IsolationForest(contamination=0.05, random_state=42)
df_model['anomaly'] = iso_model.fit_predict(df_model)

# -1 = anomali, 1 = normal
df_model['anomaly'] = df_model['anomaly'].map({1: 0, -1: 1})

# 5. Orijinal veriyle birleştir
df_result = df.copy()
df_result = df_result.loc[df_model.index]
df_result['anomaly'] = df_model['anomaly']

# 6. Anomali dağılımını göster
print("\n🔍 Anomali Dağılımı:")
print(df_result['anomaly'].value_counts())

# 7. Anomalileri görselleştir (paket uzunluğu vs. port)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_result, x='dst_port', y='length', hue='anomaly', palette={0: 'blue', 1: 'red'})
plt.title("📊 Anomali Tespiti: Port vs. Paket Uzunluğu")
plt.xlabel("Hedef Port")
plt.ylabel("Paket Uzunluğu (byte)")
plt.legend(title="Anomali")
plt.tight_layout()
plt.show()

#Protokol Dağılımı
plt.figure(figsize=(6,4))
sns.countplot(data=df, x='protocol')
plt.title("Protokol Dağılımı")
plt.ylabel("Paket Sayısı")
plt.xlabel("Protokol")
plt.tight_layout()
plt.savefig("static/plots/protocol_distribution.png")
plt.close()

#Paket Uzunluğu Histogramı
plt.figure(figsize=(6,4))
sns.histplot(df['length'], bins=20, kde=True)
plt.title("Paket Uzunluğu Dağılımı")
plt.xlabel("Uzunluk (byte)")
plt.ylabel("Paket Sayısı")
plt.tight_layout()
plt.savefig("static/plots/packet_length_hist.png")
plt.close()

#Anomali Tespiti Scatter Grafiği
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_result, x='dst_port', y='length', hue='anomaly', palette={0: 'blue', 1: 'red'})
plt.title("Anomali Tespiti: Port vs. Paket Uzunluğu")
plt.xlabel("Hedef Port")
plt.ylabel("Paket Uzunluğu (byte)")
plt.legend(title="Anomali")
plt.tight_layout()
plt.savefig("static/plots/anomaly_scatter.png")
plt.close()






