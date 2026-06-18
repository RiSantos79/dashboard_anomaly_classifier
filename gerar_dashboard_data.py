"""
Gerar dashboard_data.parquet — Dashboard Interativo CICIDS2017

Script autónomo que replica o pipeline do anomaly_classifier.ipynb
(carregamento, limpeza, pseudo-labelling IF+LOF, treino RF+LR, avaliação
no dataset completo) e exporta dashboard_data.parquet para o dashboard
Streamlit com filtros dinâmicos.

Uso:
    uv run python gerar_dashboard_data.py

Requisitos: pandas, numpy, scikit-learn, pyarrow
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

t0 = time.time()

# ── 1. CONFIGURAÇÃO ──────────────────────────────────────────────
# Ajusta DATA_PATH se a pasta Dataset não estiver um nível acima
DATA_PATH = "../anomaly_classifier/Dataset/"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "../Dataset/"
if not os.path.exists(DATA_PATH):
    DATA_PATH = "./Dataset/"

FILES = {
    'Monday'    : 'Monday-WorkingHours.pcap_ISCX.csv',
    'Wednesday' : 'Wednesday-workingHours.pcap_ISCX.csv',
    'Thursday'  : 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv',
    'Friday'    : 'Friday-WorkingHours-Morning.pcap_ISCX.csv'
}

OUTPUT_FILE = "dashboard_data.parquet"

print(f"DATA_PATH resolvido para: {os.path.abspath(DATA_PATH)}")
for nome, ficheiro in FILES.items():
    caminho = os.path.join(DATA_PATH, ficheiro)
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Ficheiro não encontrado: {caminho}\n"
            f"Ajusta a variável DATA_PATH no topo do script para o caminho correto da pasta Dataset."
        )

# ── 2. CARREGAMENTO ───────────────────────────────────────────────
print("\n=== 1. CARREGAMENTO DOS DADOS ===")
dfs = []
for nome, ficheiro in FILES.items():
    path = os.path.join(DATA_PATH, ficheiro)
    df_temp = pd.read_csv(path, encoding='utf-8', low_memory=False)
    df_temp['day'] = nome
    dfs.append(df_temp)
    print(f"  {nome:<12} {df_temp.shape[0]:>9,} registos | {df_temp.shape[1]} colunas")

df = pd.concat(dfs, ignore_index=True)
print(f"\n  Total bruto: {df.shape[0]:,} | Colunas: {df.shape[1]}")

# ── 3. LIMPEZA ────────────────────────────────────────────────────
print("\n=== 2. LIMPEZA DOS DADOS ===")
df.columns = df.columns.str.strip()
total_antes = df.shape[0]

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"  Registos após limpeza : {df.shape[0]:,}")
print(f"  Registos eliminados   : {total_antes - df.shape[0]:,}")

# ── 4. ENCODING DA LABEL + METADADOS PARA O DASHBOARD ─────────────
print("\n=== 3. ENCODING DA LABEL E METADADOS ===")
df['label'] = (df['Label'] != 'BENIGN').astype(int)
print(f"  % Anómalo : {df['label'].mean()*100:.2f}%")

# Guardar metadados ANTES de remover 'Label'
df_meta = df[['day', 'Label', 'Destination Port']].copy()
df_meta.columns = ['day', 'attack_type', 'dst_port']

print("\n  Tipos de ataque (attack_type):")
print(df_meta['attack_type'].value_counts().to_string())

# Remover Label original e colunas de variância zero (replica notebook original)
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
zero_var = [col for col in numeric_cols if df[col].var() == 0]
cols_remover = ['Label'] + zero_var
df.drop(columns=cols_remover, inplace=True)
print(f"\n  Colunas removidas: {cols_remover}")

# ── 5. FEATURES ───────────────────────────────────────────────────
features_pl = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Flow Bytes/s', 'Flow Packets/s', 'Fwd Packets/s',
    'Bwd Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Packet Length Mean', 'Packet Length Std', 'Min Packet Length', 'Max Packet Length',
]

features_modelo = [
    'Destination Port', 'Down/Up Ratio', 'Fwd Header Length', 'Bwd Header Length', 'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std', 'Bwd Packet Length Max',
    'Bwd Packet Length Min', 'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Average Packet Size', 'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Packet Length Variance', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'URG Flag Count', 'ECE Flag Count', 'Fwd PSH Flags', 'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'Subflow Fwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'act_data_pkt_fwd',
    'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
]

# ── 6. PSEUDO-LABELLING — CONSENSO IF + LOF ──────────────────────
print("\n=== 4. PSEUDO-LABELLING (IF + LOF) ===")
SAMPLE_SIZE = 200_000

normal_sample = df[df['label'] == 0].sample(int(SAMPLE_SIZE * 0.84), random_state=42)
anomalo_sample = df[df['label'] == 1].sample(int(SAMPLE_SIZE * 0.16), random_state=42)

df_sample = pd.concat([normal_sample, anomalo_sample])

X_pl = df_sample[features_pl].copy()
scaler_pl = StandardScaler()
X_pl_scaled = scaler_pl.fit_transform(X_pl)

print("  A treinar Isolation Forest...")
iso = IsolationForest(n_estimators=100, contamination=0.20, random_state=42, n_jobs=-1)
iso.fit_predict(X_pl_scaled)

print("  A treinar Local Outlier Factor...")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.20, n_jobs=-1)
lof.fit_predict(X_pl_scaled)

score_if = -iso.score_samples(X_pl_scaled)
score_if = (score_if - score_if.min()) / (score_if.max() - score_if.min())

score_lof = -lof.negative_outlier_factor_
score_lof = (score_lof - score_lof.min()) / (score_lof.max() - score_lof.min())

score_combined = (score_if + score_lof) / 2
threshold = np.percentile(score_combined, 80)
df_sample['pseudo_label'] = (score_combined >= threshold).astype(int)

concordancia = (df_sample['pseudo_label'] == df_sample['label']).mean() * 100
print(f"  Threshold percentil 80 : {threshold:.4f}")
print(f"  Concordância c/ ground truth: {concordancia:.2f}%")

# ── 7. TRAIN/TEST SPLIT TEMPORAL ──────────────────────────────────
print("\n=== 5. TRAIN/TEST SPLIT (TEMPORAL) ===")
ordem_dias = ['Monday', 'Wednesday', 'Thursday', 'Friday']
df_sample['day_order'] = df_sample['day'].map({d: i for i, d in enumerate(ordem_dias)})
df_sample = df_sample.sort_values('day_order').drop(columns='day_order')

X = df_sample[features_modelo].values
y = df_sample['pseudo_label'].values

split_idx = int(len(df_sample) * 0.80)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler_modelo = StandardScaler()
X_train_scaled = scaler_modelo.fit_transform(X_train)
X_test_scaled  = scaler_modelo.transform(X_test)

print(f"  Treino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,}")

# ── 8. MODELAÇÃO — RF + LR ────────────────────────────────────────
print("\n=== 6. MODELAÇÃO (Random Forest + Logistic Regression) ===")
print("  A treinar Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100, max_depth=15, min_samples_leaf=10,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train_scaled, y_train)

print("  A treinar Logistic Regression...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1)
lr.fit(X_train_scaled, y_train)
print("  ✓ Treino concluído")

# ── 9. AVALIAÇÃO NO DATASET COMPLETO ──────────────────────────────
print("\n=== 7. AVALIAÇÃO NO DATASET COMPLETO ===")
X_full = df[features_modelo].values
X_full_scaled = scaler_modelo.transform(X_full)
y_full_real = df['label'].values

y_pred_rf_full = rf.predict(X_full_scaled)
y_prob_rf_full = rf.predict_proba(X_full_scaled)[:, 1]
y_pred_lr_full = lr.predict(X_full_scaled)
y_prob_lr_full = lr.predict_proba(X_full_scaled)[:, 1]

print(f"  ROC-AUC Random Forest       : {roc_auc_score(y_full_real, y_prob_rf_full):.4f}")
print(f"  ROC-AUC Logistic Regression : {roc_auc_score(y_full_real, y_prob_lr_full):.4f}")
print("\n--- Random Forest ---")
print(classification_report(y_full_real, y_pred_rf_full, target_names=['Normal', 'Anómalo']))
print("\n--- Logistic Regression ---")
print(classification_report(y_full_real, y_pred_lr_full, target_names=['Normal', 'Anómalo']))

# ── 10. EXPORTAR PARQUET ──────────────────────────────────────────
print("\n=== 8. EXPORTAR dashboard_data.parquet ===")

def classificar_risco(prob):
    if prob < 0.30:
        return 'Baixo'
    elif prob < 0.60:
        return 'Médio'
    elif prob < 0.80:
        return 'Alto'
    else:
        return 'Crítico'

# ── Construir coluna split (Treino / Teste / Fora da Amostra) ─────
# Os índices do df_sample (após ordenação temporal) permitem mapear
# quais registos do df completo pertencem ao treino/teste da amostra
idx_sample_ordered = df_sample.index.tolist()
idx_train = set(idx_sample_ordered[:split_idx])
idx_test  = set(idx_sample_ordered[split_idx:])

def get_split(idx):
    if idx in idx_train:
        return 'Treino'
    elif idx in idx_test:
        return 'Teste'
    else:
        return 'Fora da Amostra'

print("  A calcular coluna split (Treino/Teste/Fora da Amostra)...")
split_col = [get_split(i) for i in df.index]
print(f"  Treino             : {split_col.count('Treino'):,}")
print(f"  Teste              : {split_col.count('Teste'):,}")
print(f"  Fora da Amostra    : {split_col.count('Fora da Amostra'):,}")

df_dashboard = pd.DataFrame({
    'day': df_meta['day'].values,
    'attack_type': df_meta['attack_type'].values,
    'dst_port': df_meta['dst_port'].values,
    'label_real': y_full_real,
    'pred_rf': y_pred_rf_full,
    'prob_rf': y_prob_rf_full,
    'pred_lr': y_pred_lr_full,
    'prob_lr': y_prob_lr_full,
    'split': split_col,
})

df_dashboard['risco_rf'] = df_dashboard['prob_rf'].apply(classificar_risco)
df_dashboard['risco_lr'] = df_dashboard['prob_lr'].apply(classificar_risco)

# Otimizar tipos para reduzir tamanho do parquet
df_dashboard['day'] = df_dashboard['day'].astype('category')
df_dashboard['attack_type'] = df_dashboard['attack_type'].astype('category')
df_dashboard['risco_rf'] = df_dashboard['risco_rf'].astype('category')
df_dashboard['risco_lr'] = df_dashboard['risco_lr'].astype('category')
df_dashboard['split'] = df_dashboard['split'].astype('category')
df_dashboard['dst_port'] = df_dashboard['dst_port'].astype('int32')
df_dashboard['label_real'] = df_dashboard['label_real'].astype('int8')
df_dashboard['pred_rf'] = df_dashboard['pred_rf'].astype('int8')
df_dashboard['pred_lr'] = df_dashboard['pred_lr'].astype('int8')
df_dashboard['prob_rf'] = df_dashboard['prob_rf'].astype('float32')
df_dashboard['prob_lr'] = df_dashboard['prob_lr'].astype('float32')

df_dashboard.to_parquet(OUTPUT_FILE, index=False, compression='snappy')

size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
elapsed = time.time() - t0

print(f"\n✓ {OUTPUT_FILE} exportado")
print(f"  Registos : {len(df_dashboard):,}")
print(f"  Tamanho  : {size_mb:.2f} MB")
print(f"  Tempo total: {elapsed:.1f}s")
print(f"\nPreview:")
print(df_dashboard.head().to_string())