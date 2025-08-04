from flask import Flask, render_template,make_response
import pandas as pd
from xhtml2pdf import pisa
from io import BytesIO

from flask import request, redirect, url_for
import asyncio
import pyshark
import os

import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/results')
def results():
    return render_template("results.html")

@app.route('/traffic-analysis')
def traffic():
    return render_template("traffic_analysis.html")

@app.route('/download-pdf')
def download_pdf():
    # Her veri seti için tablo html’ini oku
    cic_table = pd.read_html("templates/partials/results_cicids.html")[0].to_html(index=False)
    kdd_table = pd.read_html("templates/partials/results_kdd.html")[0].to_html(index=False)
    unsw_table = pd.read_html("templates/partials/results_unsw.html")[0].to_html(index=False)

    # HTML şablonunu render et
    html = render_template("pdf_template.html",
                           cic_table=cic_table,
                           kdd_table=kdd_table,
                           unsw_table=unsw_table)

    # PDF üret
    pdf_buffer = BytesIO()
    pisa.CreatePDF(html, dest=pdf_buffer)

    response = make_response(pdf_buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=anomaly_detection_report.pdf'
    return response




UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload-traffic', methods=['POST'])
def upload_traffic():
    file = request.files['file']
    filename = file.filename

    # Dosyayı kaydet
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # CSV mi PCAP mı kontrol et
    if filename.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif filename.endswith('.pcap'):
        # Asyncio event loop oluştur ve ayarla (thread içinde)
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        cap = pyshark.FileCapture(file_path, use_json=True)



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
        df['length'] = pd.to_numeric(df['length'], errors='coerce')
    else:
        return " Geçersiz dosya türü (sadece .csv veya .pcap destekleniyor)."

    # Trafik analizlerini yap ve grafikleri kaydet
    return process_uploaded_traffic(df)

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

def process_uploaded_traffic(df):
    # Grafik klasörü
    os.makedirs('static/plots', exist_ok=True)

    # Protokol dağılımı
    if 'protocol' in df.columns:
        plt.figure(figsize=(6,4))
        sns.countplot(data=df, x='protocol')
        plt.title("Protokol Dağılımı")
        plt.ylabel("Paket Sayısı")
        plt.xlabel("Protokol")
        plt.tight_layout()
        plt.savefig("static/plots/user_protocol_distribution.png")
        plt.close()

    # Paket uzunluğu histogramı
    if 'length' in df.columns:
        plt.figure(figsize=(6,4))
        sns.histplot(df['length'], bins=20, kde=True)
        plt.title("Paket Uzunluğu Dağılımı")
        plt.xlabel("Uzunluk (byte)")
        plt.ylabel("Paket Sayısı")
        plt.tight_layout()
        plt.savefig("static/plots/user_packet_length_hist.png")
        plt.close()

    # Anomali tespiti
    if {'protocol', 'dst_port', 'length'}.issubset(df.columns):
        df_model = df[['protocol', 'dst_port', 'length']].copy()

        # Sayısallaştır
        le = LabelEncoder()
        df_model['protocol'] = le.fit_transform(df_model['protocol'].astype(str))
        df_model['dst_port'] = pd.to_numeric(df_model['dst_port'], errors='coerce')
        df_model['length'] = pd.to_numeric(df_model['length'], errors='coerce')
        df_model.dropna(inplace=True)

        iso_model = IsolationForest(contamination=0.05, random_state=42)
        df_model['anomaly'] = iso_model.fit_predict(df_model)
        df_model['anomaly'] = df_model['anomaly'].map({1: 0, -1: 1})

        df_result = df.loc[df_model.index]
        df_result['anomaly'] = df_model['anomaly']

        # Scatter plot
        plt.figure(figsize=(8,5))
        sns.scatterplot(data=df_result, x='dst_port', y='length', hue='anomaly', palette={0: 'blue', 1: 'red'})
        plt.title("Kullanıcı Trafiği Anomali Tespiti")
        plt.xlabel("Hedef Port")
        plt.ylabel("Paket Uzunluğu (byte)")
        plt.legend(title="Anomali")
        plt.tight_layout()
        plt.savefig("static/plots/user_anomaly_scatter.png")
        plt.close()

    # Yönlendir -> Sonuçları gösteren sayfa
    return redirect(url_for('user_traffic_results'))

@app.route('/user-traffic-results')
def user_traffic_results():
    return render_template("user_traffic_results.html")


@app.route('/download-user-traffic-pdf')
def download_user_traffic_pdf():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 30px;
                text-align: center;
            }
            h1 {
                margin-bottom: 30px;
            }
            img {
                display: block;
                margin: 20px auto;
                width: 80%;
            }
        </style>
    </head>
    <body>
        <h1>🌐 Kullanıcı Trafik Analizi Sonucu</h1>

        <img src="static/plots/user_protocol_distribution.png" alt="Protokol Dağılımı">
        <img src="static/plots/user_packet_length_hist.png" alt="Paket Uzunluğu Histogramı">
        <img src="static/plots/user_anomaly_scatter.png" alt="Anomali Scatter Grafiği">

    </body>
    </html>
    """

    pdf_buffer = BytesIO()
    pisa.CreatePDF(html, dest=pdf_buffer)

    response = make_response(pdf_buffer.getvalue())
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=user_traffic_report.pdf'
    return response







if __name__ == '__main__':
    app.run(debug=True)
