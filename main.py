import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Veri seti işlemleri
# CICIDS2017 verisini yükle
file_path = r'C:\Users\aliyi\OneDrive\Masaüstü\anomaly_detection_project\uploads\CICIDS2017.csv'
df = pd.read_csv(file_path, low_memory=False)
df = df.sample(n=30000, random_state=42)

print("Başlangıç veri şekli:", df.shape)

# 1. Çok fazla boş olan sütunları (örneğin %50'den fazla boş) çıkar
threshold = 0.5
df = df.loc[:, df.isnull().mean() < threshold]

# 2. Kalan eksik verileri satır bazında temizle
df.dropna(inplace=True)

# 3. Etiket sütununun adını kontrol et
label_column = None
for col in df.columns:
    if "label" in col.lower():
        label_column = col
        break

if label_column is None:
    raise Exception("Etiket sütunu bulunamadı!")

# 4. Etiketleri 0 = Normal (BENIGN), 1 = Anomali olarak ayarla
df[label_column] = df[label_column].apply(lambda x: 0 if 'BENIGN' in str(x).upper() else 1)

# 5. İlgisiz sütunları temizle (IP, Timestamp, vb.)
drop_cols = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp', 'Src IP', 'Dst IP', 'Protocol']
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# 6. Özellik ve etiketleri ayır
X = df.drop(columns=[label_column])
y = df[label_column]

# 7. Kategorik sütunları sayısallaştır
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

# 7.5 Sonsuz değerleri temizle
X.replace([np.inf, -np.inf], np.nan, inplace=True)
X.dropna(inplace=True)
y = y[X.index]  # Etiketleri de senkronize et

# 8. Özellikleri normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("İşlenmiş veri şekli:", X_scaled.shape)
print("Etiket dağılımı:\n", pd.Series(y).value_counts())


# RANDOM FOREST

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Veriyi eğitim ve test olarak ayır (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Modeli oluştur
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Tahmin yap
y_pred = rf_model.predict(X_test)

# Sonuçları yazdır
print("\n🎯 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print("\n✅ Accuracy:", accuracy_score(y_test, y_pred))


#  Grafiklerin "static/plots" klasörüne kaydedilmesi için gerekli kodlar.

import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd

# Grafiklerin kaydedileceği dizini oluştur
os.makedirs("static/plots", exist_ok=True)

# 1. Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.tight_layout()
plt.savefig("static/plots/confusion_matrix.png")
plt.close()

#2. ROC Eğrisi
y_prob = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC (AUC = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("static/plots/roc_curve.png")
plt.close()

# 3. Özellik Önem Dereceleri
importances = rf_model.feature_importances_
features = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=features[:15], y=features.index[:15])
plt.title("Özellik Önem Dereceleri (Top 15)")
plt.xlabel("Önem Skoru")
plt.ylabel("Özellikler")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("static/plots/feature_importance.png")
plt.close()

#Aşağıdaki kodla Confusion Matrix, ROC Curve ve Feature Importances grafiğini çiziyoruz

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# Confusion Matrix Görselleştirme
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

# ROC Eğrisi
y_prob = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC eğrisi (AUC = %0.4f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi')
plt.legend(loc="lower right")
plt.show()

# Özellik Önem Dereceleri
importances = rf_model.feature_importances_
features = pd.Series(importances, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(x=features[:15], y=features.index[:15])
plt.title("Özellik Önem Dereceleri (Top 15)")
plt.xlabel("Önem Skoru")
plt.ylabel("Özellikler")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()



#Aşağıdaki kod, 3 yeni modeli eğitir ve karşılaştırma tablosu oluşturur

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Model listesi
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Sonuçları saklamak için liste
results = []

# Her model için eğit ve değerlendir
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

# Random Forest'ı da ekleyelim
results.append({
    'Model': 'Random Forest',
    'Accuracy': accuracy_score(y_test, rf_model.predict(X_test)),
    'Precision': precision_score(y_test, rf_model.predict(X_test)),
    'Recall': recall_score(y_test, rf_model.predict(X_test)),
    'F1 Score': f1_score(y_test, rf_model.predict(X_test))
})

# Sonuçları DataFrame olarak göster
results_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)
print("\n📊 Model Karşılaştırması:\n")
print(results_df)

#Isolation Forest ve One-Class
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np

# Sadece normal verileri (etiketi 0 olanları) eğitim için al
X_normal = X_scaled[y == 0]

# Tüm test verisini kullanacağız (normal + anomali)
X_test_all = X_scaled
y_test_all = y

# Isolation Forest
iso_model = IsolationForest(contamination='auto', random_state=42)
iso_model.fit(X_normal)
y_pred_iso = iso_model.predict(X_test_all)
y_pred_iso = np.where(y_pred_iso == 1, 0, 1)  # 1: normal, -1: anomali → 0: normal, 1: anomali

print("\n🌲 Isolation Forest")
print("Confusion Matrix:\n", confusion_matrix(y_test_all, y_pred_iso))
print("Classification Report:\n", classification_report(y_test_all, y_pred_iso))

# One-Class SVM
svm_model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
svm_model.fit(X_normal)
y_pred_svm = svm_model.predict(X_test_all)
y_pred_svm = np.where(y_pred_svm == 1, 0, 1)  # 1: normal, -1: anomali → 0: normal, 1: anomali

print("\n🌀 One-Class SVM")
print("Confusion Matrix:\n", confusion_matrix(y_test_all, y_pred_svm))
print("Classification Report:\n", classification_report(y_test_all, y_pred_svm))

#KDD99
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Dosyayı oku
df = pd.read_csv(r"C:\Users\aliyi\OneDrive\Masaüstü\anomaly_detection_project\uploads\KDD99.csv")

df = df.sample(n=30000, random_state=42)

# Etiketi ikili hale getir (normal → 0, diğerleri → 1)
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Kategorik sütunları dönüştür
cat_cols = ['protocol_type', 'service', 'flag']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Özellik ve etiket ayır
X = df.drop('label', axis=1)
y = df['label']

# Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim/test ayır
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Modeller
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Sonuçları topla
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    })

# Sonuçları yazdır
results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
print("\n📊 Model Karşılaştırması:\n")
print(results_df)

#KDD99 – Tablo + Grafikleri kaydet
# Skor tablosunu HTML olarak kaydet
results_df.to_html("templates/partials/results_kdd.html", index=False, classes='table table-striped')

# En iyi modelin confusion matrix vs. ROC grafiğini çiz
best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("KDD99 Confusion Matrix")
plt.tight_layout()
plt.savefig("static/plots/kdd_confusion_matrix.png")
plt.close()

# ROC Curve
if y_prob_best is not None:
    fpr, tpr, _ = roc_curve(y_test, y_prob_best)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title("KDD99 ROC Eğrisi")
    plt.tight_layout()
    plt.savefig("static/plots/kdd_roc_curve.png")
    plt.close()

# Feature Importances (varsa)
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    features = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=features[:15], y=features.index[:15])
    plt.title("KDD99 Özellik Önem Dereceleri")
    plt.tight_layout()
    plt.savefig("static/plots/kdd_feature_importance.png")
    plt.close()


#Anomaly Detection (Unsupervised) Uygula (Isolation Forest + One-Class SVM)

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix

# Eğitim için sadece "normal" örnekler (label == 0)
X_normal = X_scaled[y == 0]

# Test seti: tüm veri
X_test_all = X_scaled
y_test_all = y

# Isolation Forest
iso_model = IsolationForest(contamination='auto', random_state=42)
iso_model.fit(X_normal)
y_pred_iso = iso_model.predict(X_test_all)
y_pred_iso = np.where(y_pred_iso == 1, 0, 1)  # 1 → normal, -1 → anomaly → 0-1'e dönüştür

print("\n🌲 Isolation Forest")
print("Confusion Matrix:\n", confusion_matrix(y_test_all, y_pred_iso))
print("Classification Report:\n", classification_report(y_test_all, y_pred_iso))

# One-Class SVM
svm_model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
svm_model.fit(X_normal)
y_pred_svm = svm_model.predict(X_test_all)
y_pred_svm = np.where(y_pred_svm == 1, 0, 1)

print("\n🌀 One-Class SVM")
print("Confusion Matrix:\n", confusion_matrix(y_test_all, y_pred_svm))
print("Classification Report:\n", classification_report(y_test_all, y_pred_svm))

#UNSW-NB15
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

# 🔹 1. Dosya ve sütun adları
file_path = r'C:\Users\aliyi\OneDrive\Masaüstü\anomaly_detection_project\uploads\UNSW-NB15_1.csv'


column_names = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes',
    'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts',
    'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth',
    'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt',
    'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm',
    'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
    'attack_cat', 'label'
]

df = pd.read_csv(file_path, header=None, names=column_names, low_memory=False)

df = df.sample(n=30000, random_state=42)

#2. Gerekli sütunları seç
df.drop(columns=['srcip', 'dstip', 'Stime', 'Ltime', 'attack_cat'], inplace=True)

#3. Kategorik verileri sayısala çevir
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

#4. Özellik/etiket ayır
X = df.drop(columns=['label'])
y = df['label']

#5. Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#6. Eğitim/Test bölme
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

#7. Supervised modeller
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    })

#8. Isolation Forest (unsupervised)
X_normal = X_scaled[y == 0]
iso_model = IsolationForest(contamination='auto', random_state=42)
iso_model.fit(X_normal)
y_pred_iso = iso_model.predict(X_scaled)
y_pred_iso = np.where(y_pred_iso == 1, 0, 1)

results.append({
    'Model': 'Isolation Forest',
    'Accuracy': accuracy_score(y, y_pred_iso),
    'Precision': precision_score(y, y_pred_iso),
    'Recall': recall_score(y, y_pred_iso),
    'F1 Score': f1_score(y, y_pred_iso)
})

#9. One-Class SVM (unsupervised)
svm_model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.01)
svm_model.fit(X_normal)
y_pred_svm = svm_model.predict(X_scaled)
y_pred_svm = np.where(y_pred_svm == 1, 0, 1)

results.append({
    'Model': 'One-Class SVM',
    'Accuracy': accuracy_score(y, y_pred_svm),
    'Precision': precision_score(y, y_pred_svm),
    'Recall': recall_score(y, y_pred_svm),
    'F1 Score': f1_score(y, y_pred_svm)
})

#10. Sonuçları göster
results_df = pd.DataFrame(results).sort_values(by='F1 Score', ascending=False)
print("\n📊 UNSW-NB15 Model Karşılaştırması:\n")
print(results_df)

# Model skorlarını HTML'e dönüştür
results_df.to_html("templates/partials/results_cicids.html", index=False, classes='table table-striped')

#UNSW-NB15 Tablo + Grafikleri kaydet

results_df.to_html("templates/partials/results_unsw.html", index=False, classes='table table-striped')

best_model_name = results_df.iloc[0]['Model']
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)
y_prob_best = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

conf_matrix = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("UNSW Confusion Matrix")
plt.tight_layout()
plt.savefig("static/plots/unsw_confusion_matrix.png")
plt.close()

if y_prob_best is not None:
    fpr, tpr, _ = roc_curve(y_test, y_prob_best)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC (AUC = %0.4f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title("UNSW ROC Eğrisi")
    plt.tight_layout()
    plt.savefig("static/plots/unsw_roc_curve.png")
    plt.close()

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    features = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=features[:15], y=features.index[:15])
    plt.title("UNSW Özellik Önem Dereceleri")
    plt.tight_layout()
    plt.savefig("static/plots/unsw_feature_importance.png")
    plt.close()


