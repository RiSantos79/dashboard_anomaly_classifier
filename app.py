"""
Dashboard Interativo — Deteção de Anomalias CICIDS2017

Reestruturado em 3 abas por audiência/decisão (storytelling, slide 22 da Aula 4):
  - Operacional / SOC: o que investigar agora (analista SOC)
  - Tendências / Gestão: escalar recursos ou não (gestor SOC)
  - Comparação de Modelos: qual modelo produtizar (data scientist)

Filtros na sidebar são dinâmicos por aba (só os relevantes aparecem),
e filtros partilhados entre abas mantêm o valor selecionado via session_state.

Dados: dashboard_data.parquet (gerado por gerar_dashboard_data.py)
"""

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    roc_auc_score, accuracy_score, roc_curve,
)

# ── Configuração da página ────────────────────────────────────────
st.set_page_config(
    page_title="Anomaly Classifier — Dashboard Interativo",
    page_icon="🛡️",
    layout="wide",
)

# ── Paleta de cores (consistente com o notebook/dashboard narrativo) ──
BG         = '#1a2233'
GRID       = '#2e3f5c'
TEXT       = '#d0d8e8'
TEXT_MUTED = '#8899aa'
BLUE       = '#5b8dd9'
RED        = '#c0504d'
GREEN      = '#2ea84f'
ORANGE     = '#e07b1a'
YELLOW     = '#e8c547'
TEAL       = '#7ED8B3'
CARD_BG    = '#243353'

CORES_RISCO = {
    'Baixo'   : BLUE,
    'Médio'   : YELLOW,
    'Alto'    : ORANGE,
    'Crítico' : RED,
}

ORDEM_DIAS   = ['Monday', 'Wednesday', 'Thursday', 'Friday']
ORDEM_RISCO  = ['Baixo', 'Médio', 'Alto', 'Crítico']
ORDEM_SPLIT  = ['Treino', 'Teste', 'Fora da Amostra']

PORTOS_CONHECIDOS = {
    20: "FTP-Data", 21: "FTP", 22: "SSH", 23: "Telnet", 25: "SMTP",
    53: "DNS", 67: "DHCP", 68: "DHCP", 69: "TFTP", 80: "HTTP",
    110: "POP3", 119: "NNTP", 123: "NTP", 135: "MSRPC", 137: "NetBIOS",
    138: "NetBIOS", 139: "NetBIOS", 143: "IMAP", 161: "SNMP", 162: "SNMP-Trap",
    179: "BGP", 389: "LDAP", 443: "HTTPS", 445: "SMB", 465: "SMTPS",
    514: "Syslog", 515: "LPD", 587: "SMTP-TLS", 636: "LDAPS", 873: "Rsync",
    993: "IMAPS", 995: "POP3S", 1080: "SOCKS", 1433: "MSSQL", 1434: "MSSQL-Monitor",
    1521: "Oracle DB", 1723: "PPTP", 2049: "NFS", 2082: "cPanel", 2083: "cPanel-SSL",
    2483: "Oracle DB", 2484: "Oracle DB-SSL", 3268: "LDAP-GC", 3306: "MySQL",
    3389: "RDP", 4443: "HTTPS-Alt", 5060: "SIP", 5061: "SIPS", 5432: "PostgreSQL",
    5900: "VNC", 5985: "WinRM", 5986: "WinRM-SSL", 6379: "Redis", 6667: "IRC",
    8000: "HTTP-Alt", 8008: "HTTP-Alt", 8080: "HTTP-Proxy", 8081: "HTTP-Alt",
    8443: "HTTPS-Alt", 8888: "HTTP-Alt", 9000: "HTTP-Alt", 9090: "WebSM",
    9200: "Elasticsearch", 11211: "Memcached", 27017: "MongoDB",
}

def formatar_porto(porto):
    """Retorna 'porto (Serviço)' para portos conhecidos, ou só 'porto'."""
    servico = PORTOS_CONHECIDOS.get(int(porto))
    return f"{porto} ({servico})" if servico else f"{porto}"

def beeswarm_offsets(values, n_bins=30, scale=0.38):
    """Calcula deslocamentos verticais (jitter) por densidade, para simular um beeswarm plot
    em Plotly (que não tem este tipo de gráfico nativamente). Pontos em zonas mais densas do
    eixo X ficam mais espalhados verticalmente; pontos isolados ficam perto do centro da linha."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return np.array([])
    if values.max() == values.min():
        return np.zeros(n)
    bins = np.linspace(values.min(), values.max(), n_bins + 1)
    bin_idx = np.clip(np.digitize(values, bins) - 1, 0, n_bins - 1)
    offsets = np.zeros(n)
    for b in np.unique(bin_idx):
        mask = bin_idx == b
        count = mask.sum()
        ordem = np.arange(count)
        lado = np.where(ordem % 2 == 0, 1, -1)
        magnitude = (ordem + 1) // 2
        offsets[mask] = lado * magnitude
    max_abs = np.abs(offsets).max()
    if max_abs > 0:
        offsets = offsets / max_abs * scale
    return offsets

PLOTLY_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family='sans-serif'),
    margin=dict(l=16, r=16, t=44, b=16),
    xaxis=dict(showgrid=False, zeroline=False, showline=False, tickcolor=TEXT, color=TEXT),
    yaxis=dict(gridcolor=GRID, gridwidth=0.7, zeroline=False, showline=False, tickcolor=TEXT, color=TEXT),
    legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0, font=dict(color=TEXT)),
    hoverlabel=dict(bgcolor=CARD_BG, bordercolor=GRID, font=dict(color=TEXT, size=12)),
)

LEGEND_TOPO = dict(
    orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0,
    bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(color=TEXT, size=10),
)
MARGIN_TOPO = dict(l=16, r=16, t=72, b=16)

# Para gráficos com legendas longas (muitos itens) que colidem com o título
LEGEND_TOPO_XL = dict(
    orientation="h", yanchor="bottom", y=1.15, xanchor="left", x=0,
    bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(color=TEXT, size=10),
)
MARGIN_TOPO_XL = dict(l=16, r=16, t=110, b=16)

# ── CSS global ────────────────────────────────────────────────────
st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; }}
    section[data-testid="stSidebar"] {{ background-color: #111827; border-right: 1px solid {GRID}; }}
    .block-container {{ padding-left: 0 !important; padding-right: 0 !important; padding-top: 0 !important; max-width: 100% !important; }}
    .kpi-card {{
        background-color: #243353;
        border: 1px solid {GRID};
        border-radius: 8px;
        padding: 16px 20px;
        text-align: center;
    }}
    .kpi-label {{ font-size: 11px; color: {TEXT_MUTED}; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; white-space: nowrap; }}
    .kpi-value {{ font-size: 24px; font-weight: 700; color: {TEXT}; }}
    .kpi-value.blue {{ color: {BLUE}; }}
    .kpi-value.red {{ color: {RED}; }}
    .kpi-value.green {{ color: {GREEN}; }}
    .kpi-value.orange {{ color: {ORANGE}; }}
    .section-title {{ font-size: 15px; font-weight: 600; color: {TEXT}; margin: 16px 0 0 24px; border-left: 3px solid {BLUE}; padding-left: 10px; }}
    div[data-testid="stMetric"] {{ display: none; }}

    div[data-testid="stStatusWidget"] {{ display: none; }}
    div[data-testid="stDecoration"] {{ display: none !important; height: 0 !important; }}

    .crit-bar {{ height: 4px; border-radius: 2px; margin: 8px 24px 16px; }}
    .crit-bar.green  {{ background: linear-gradient(90deg, {GREEN} 0%, transparent 100%); }}
    .crit-bar.yellow {{ background: linear-gradient(90deg, {ORANGE} 0%, transparent 100%); }}
    .crit-bar.red    {{ background: linear-gradient(90deg, {RED} 0%, transparent 100%); }}

    .chart-card {{
        background-color: #243353;
        border: 1px solid {GRID};
        border-radius: 8px;
        padding: 8px 4px 0;
        margin: 0 8px 16px;
    }}

    .audience-banner {{
        margin: 0 24px 8px;
        padding: 10px 16px;
        border-radius: 6px;
        background: rgba(91,141,217,0.08);
        border-left: 3px solid {BLUE};
        font-size: 12px;
        color: {TEXT_MUTED};
    }}
    .audience-banner b {{ color: {TEXT}; }}

    .footer {{
        padding: 48px 80px 32px;
        border-top: 1px solid #1e2d45;
        background: linear-gradient(180deg, {BG} 0%, #0d1117 100%);
        margin-top: 32px;
    }}
    .footer-grid {{ display: grid; grid-template-columns: 2fr 1fr 1fr; gap: 32px; margin-bottom: 24px; }}
    .footer-title {{ font-size: 13px; font-weight: 700; color: #d0d8e8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }}
    .footer-text {{ font-size: 13px; color: #6b7a8d; line-height: 1.7; }}
    .footer-text a {{ color: {BLUE}; text-decoration: none; }}
    .footer-text a:hover {{ text-decoration: underline; }}
    .footer-bottom {{
        font-size: 11px; color: #3d5278; border-top: 1px solid #1e2d45;
        padding-top: 16px; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;
    }}

    button[data-baseweb="tab"] {{ font-size: 14px; font-weight: 600; }}
    div[data-baseweb="tab-list"] {{ gap: 4px; padding: 0 24px; }}
</style>
""", unsafe_allow_html=True)

# ── Carregamento e pré-agregação de dados ────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet("dashboard_data.parquet")
    df["attack_type"] = (
        df["attack_type"].astype(str)
        .str.replace("\ufffd", "-", regex=False)
        .str.strip()
    )
    return df

@st.cache_data
def load_feature_importance():
    """Carrega a importância das features (Random Forest), se o ficheiro tiver sido gerado.
    Devolve None se ainda não existir, para o dashboard não rebentar nesse caso."""
    try:
        return pd.read_parquet("feature_importance.parquet")
    except FileNotFoundError:
        return None

@st.cache_data
def load_shap_values():
    """Carrega os valores SHAP (Random Forest, amostra de referência), se gerados.
    Devolve None se ainda não existir."""
    try:
        return pd.read_parquet("shap_values.parquet")
    except FileNotFoundError:
        return None

@st.cache_data
def build_aggregates(df):
    """Pré-calcula todos os agregados necessários para os gráficos."""
    agg_dia = {}
    for suf in ["rf", "lr"]:
        col_pred  = f"pred_{suf}"
        col_risco = f"risco_{suf}"
        g = df.groupby("day", observed=True).agg(
            n=("label_real", "count"),
            n_real=("label_real", "sum"),
            n_pred=(col_pred, "sum"),
        ).reindex(ORDEM_DIAS).fillna(0)
        g["pct_real"] = (g["n_real"] / g["n"] * 100).round(2)
        g["pct_pred"] = (g["n_pred"] / g["n"] * 100).round(2)

        risco_dia = (
            df.groupby(["day", col_risco, "split"], observed=True)
            .size().reset_index(name="n_r")
        )
        risco_total = risco_dia.groupby("day")["n_r"].transform("sum")
        risco_dia["pct"] = (risco_dia["n_r"] / risco_total * 100).round(2)
        risco_dia["day"] = pd.Categorical(risco_dia["day"], categories=ORDEM_DIAS, ordered=True)
        risco_dia = risco_dia.sort_values("day")

        agg_dia[suf] = {"por_dia": g, "risco_dia": risco_dia, "col_risco": col_risco}

    agg_ataque = {}
    for suf in ["rf", "lr"]:
        col_pred = f"pred_{suf}"
        g = df.groupby(["attack_type", "label_real", "day", f"risco_{suf}", "split"], observed=True).agg(
            n=(col_pred, "count"),
            n_pred=(col_pred, "sum"),
        ).reset_index()
        g["n_real"] = g["n"] * g["label_real"]
        agg_ataque[suf] = g

    agg_porto = {}
    for suf in ["rf", "lr"]:
        col_pred = f"pred_{suf}"
        col_risco = f"risco_{suf}"
        g = df.groupby(["dst_port", "day", "attack_type", col_risco, "split"], observed=True).agg(
            n_pred=(col_pred, "sum"),
            n=("label_real", "count"),
            n_real=("label_real", "sum"),
        ).reset_index()
        agg_porto[suf] = g

    sample = df.sample(min(50_000, len(df)), random_state=42)[
        ["attack_type", "day", "label_real", "prob_rf", "prob_lr",
         "risco_rf", "risco_lr", "pred_rf", "pred_lr", "split"]
    ].copy()

    return agg_dia, agg_ataque, agg_porto, sample

df = load_data()
df_feature_importance = load_feature_importance()
df_shap = load_shap_values()
HAS_SPLIT = "split" in df.columns
if not HAS_SPLIT:
    df["split"] = "Fora da Amostra"
agg_dia, agg_ataque, agg_porto, df_sample = build_aggregates(df)

# ── Score de Saúde Global (RF, amostra completa, sem filtros) ──────
cm_global = confusion_matrix(df_sample["label_real"], df_sample["pred_rf"], labels=[0, 1])
tn_g, fp_g, fn_g, tp_g = cm_global[0][0], cm_global[0][1], cm_global[1][0], cm_global[1][1]
fnr_global = fn_g / (fn_g + tp_g) if (fn_g + tp_g) > 0 else 0
fpr_global = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0
f1_global = f1_score(df_sample["label_real"], df_sample["pred_rf"], zero_division=0)

if fnr_global < 0.25 and fpr_global < 0.15:
    icone_saude, status_saude, cor_saude = "🔵", "Desempenho Dentro do Esperado", BLUE
elif fnr_global < 0.40 and fpr_global < 0.25:
    icone_saude, status_saude, cor_saude = "🟠", "Desempenho a Monitorizar", ORANGE
else:
    icone_saude, status_saude, cor_saude = "🔴", "Desempenho Abaixo do Esperado", RED

# ── Header global (comum às 3 abas) ──────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(90deg,#0d1320,{BG});padding:24px 32px 16px;border-bottom:1px solid {GRID};margin-bottom:0;margin-top:38px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:16px'>
    <div>
        <div style='font-size:11px;letter-spacing:3px;color:{BLUE};text-transform:uppercase;margin-bottom:6px'>🛡️ ISCTE — Data Science & Business Analytics · 2026</div>
        <div style='font-size:28px;font-weight:900;color:{TEXT};line-height:1.1'>Deteção de Anomalias em Tráfego de Rede</div>
        <div style='font-size:13px;color:{TEXT_MUTED};margin-top:6px'>Pipeline: Isolation Forest + LOF → Random Forest / Logistic Regression · Dataset: CICIDS2017</div>
    </div>
    <div style='background:{cor_saude}22;border:1px solid {cor_saude};border-radius:8px;padding:10px 18px;text-align:right;min-width:230px'>
        <div style='font-size:10px;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>Desempenho do Modelo · Random Forest</div>
        <div style='font-size:16px;font-weight:700;color:{cor_saude}'>{icone_saude} {status_saude}</div>
        <div style='font-size:11px;color:{TEXT_MUTED};margin-top:4px'>F1 {f1_global:.2f} · FNR {fnr_global*100:.1f}% · FPR {fpr_global*100:.1f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

with st.expander("ℹ️ Como é calculado este indicador?"):
    st.markdown(f"""
Este indicador resume o desempenho do **Random Forest** avaliado sobre toda a amostra de referência ({len(df_sample):,} registos), **sem filtros aplicados** — é um valor fixo, igual em todas as abas, que não muda com a seleção da sidebar.

**Classificação:**
- 🔵 **Dentro do Esperado** — FNR < 25% e FPR < 15%
- 🟠 **A Monitorizar** — FNR < 40% e FPR < 25%
- 🔴 **Abaixo do Esperado** — acima destes valores

Os limiares são deliberadamente mais permissivos do que os usados nos KPIs por aba (ver "O que significam estes termos?"), porque refletem a dificuldade real de deteção de anomalias num dataset fortemente desbalanceado (~16% de tráfego anómalo no CICIDS2017). Um FNR de 30-35% não significa que o modelo falhou — é um resultado conhecido e já discutido no relatório do projeto, decorrente do trade-off entre Precision e Recall ao threshold de decisão atual (0.5).
    """)

st.markdown("""
<style>
    /* Seletor de aba de navegação (radio no conteúdo principal, não na sidebar) */
    section.main div[data-testid="stRadio"] > div { gap: 4px; padding: 0 24px; }
    section.main div[data-testid="stRadio"] label {
        background-color: #243353; border: 1px solid #2e3f5c; border-radius: 6px 6px 0 0;
        padding: 10px 20px; font-weight: 600; font-size: 14px;
    }
    section.main div[data-testid="stRadio"] label:has(input:checked) {
        background-color: #5b8dd9; border-color: #5b8dd9;
    }
    section.main div[data-testid="stRadio"] svg { display: none; }

    /* Restaurar aspeto normal dos radio buttons na sidebar */
    section[data-testid="stSidebar"] div[data-testid="stRadio"] > div { gap: 0; padding: 0; }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label {
        background-color: transparent; border: none; border-radius: 0;
        padding: 4px 0; font-weight: 400; font-size: 14px;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] label:has(input:checked) {
        background-color: transparent; border-color: transparent;
    }
    section[data-testid="stSidebar"] div[data-testid="stRadio"] svg { display: inline; }
</style>
""", unsafe_allow_html=True)

if "aba_ativa" not in st.session_state:
    st.session_state["aba_ativa"] = "🚨 Operacional / SOC"

aba_ativa = st.radio(
    "Navegação", ["🚨 Operacional / SOC", "📈 Tendências / Gestão", "⚖️ Comparação de Modelos"],
    key="aba_ativa", horizontal=True, label_visibility="collapsed",
)

# ──────────────────────────────────────────────────────────────────
# ABA 1 — OPERACIONAL / SOC
# ──────────────────────────────────────────────────────────────────
if aba_ativa == "🚨 Operacional / SOC":
    st.sidebar.markdown(f"<div style='color:{BLUE};font-size:18px;font-weight:700;padding:8px 0'>🚨 Filtros — Operacional</div>", unsafe_allow_html=True)

    st.sidebar.markdown(f"<div style='font-size:12px;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:1px;margin-bottom:6px'>Modelo</div>", unsafe_allow_html=True)
    modelo_label = st.sidebar.radio("Modelo", ["Random Forest", "Logistic Regression"], key="modelo_soc")
    suf = "rf" if modelo_label == "Random Forest" else "lr"

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<div style='font-size:12px;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>Dia</div>", unsafe_allow_html=True)
    dias_sel = [d for d in ORDEM_DIAS if st.sidebar.checkbox(d, value=True, key=f"dia_{d}")]
    if not dias_sel:
        dias_sel = ORDEM_DIAS

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<div style='font-size:12px;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>Nível de Risco</div>", unsafe_allow_html=True)
    riscos_sel = [r for r in ORDEM_RISCO if st.sidebar.checkbox(r, value=True, key=f"risco_{r}")]
    if not riscos_sel:
        riscos_sel = ORDEM_RISCO

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<div style='font-size:12px;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>Tipo de Tráfego</div>", unsafe_allow_html=True)
    tipos_ataque = sorted(df["attack_type"].unique().tolist())
    tipos_sel = st.sidebar.multiselect("Tipo de tráfego", tipos_ataque, default=tipos_ataque, key="tipos_soc", label_visibility="collapsed")
    if not tipos_sel:
        tipos_sel = tipos_ataque

    st.sidebar.markdown("---")
    top_n = st.sidebar.slider("Top N portos", 3, 20, 10, key="top_n_soc")

    split_sel_soc = ORDEM_SPLIT

    col_risco = f"risco_{suf}"
    col_pred  = f"pred_{suf}"
    col_prob  = f"prob_{suf}"

    a_atq    = agg_ataque[suf]
    a_atq_filt = a_atq[
        a_atq["day"].isin(dias_sel) &
        a_atq[col_risco].isin(riscos_sel) &
        a_atq["attack_type"].isin(tipos_sel) &
        a_atq["split"].isin(split_sel_soc)
    ]
    a_port   = agg_porto[suf]
    a_port_filt = a_port[
        a_port["day"].isin(dias_sel) &
        a_port[col_risco].isin(riscos_sel) &
        a_port["attack_type"].isin(tipos_sel) &
        a_port["split"].isin(split_sel_soc)
    ]
    sample_filt = df_sample[
        df_sample["day"].isin(dias_sel) &
        df_sample[col_risco].isin(riscos_sel) &
        df_sample["attack_type"].isin(tipos_sel) &
        df_sample["split"].isin(split_sel_soc)
    ]

    n_total  = int(a_atq_filt["n"].sum())
    n_real   = int(a_atq_filt["n_real"].sum()) if "n_real" in a_atq_filt.columns else 0
    n_pred_pos = int(a_atq_filt["n_pred"].sum())
    pct_r    = round(n_real / n_total * 100, 1) if n_total > 0 else 0
    pct_p    = round(n_pred_pos / n_total * 100, 1) if n_total > 0 else 0

    if n_total == 0:
        st.warning("Nenhum registo corresponde aos filtros selecionados.")
    else:
        cm = confusion_matrix(sample_filt["label_real"], sample_filt[col_pred], labels=[0, 1])
        tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
        prec = precision_score(sample_filt["label_real"], sample_filt[col_pred], zero_division=0)
        rec  = recall_score(sample_filt["label_real"], sample_filt[col_pred], zero_division=0)
        f1   = f1_score(sample_filt["label_real"], sample_filt[col_pred], zero_division=0)
        risco_counts = a_atq_filt.groupby(col_risco, observed=True)["n"].sum()
        pct_crit = round(risco_counts.get("Crítico", 0) / n_total * 100, 1) if n_total > 0 else 0

        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0          # taxa de falsos negativos (ataques não detetados)
        fpr_rate = fp / (fp + tn) if (fp + tn) > 0 else 0      # taxa de falsos positivos (falsos alarmes)
        n_nao_detetados = int(round(fnr * (fn + tp) * (n_real / max(fn + tp, 1)))) if (fn + tp) > 0 else 0
        # Estimativa proporcional do nº absoluto de ataques não detetados no total filtrado (a amostra é de 50k)
        n_nao_detetados_total = int(round(fnr * n_real))

        def cor_fnr(v):
            if v < 0.10: return "blue"
            elif v < 0.25: return "orange"
            else: return "red"

        def cor_fpr(v):
            if v < 0.05: return "blue"
            elif v < 0.15: return "orange"
            else: return "red"

        st.markdown(
            f"<div class='audience-banner'><b>Audiência:</b> Analista SOC &nbsp;·&nbsp; "
            f"<b>Decisão:</b> o que investigar agora &nbsp;·&nbsp; <b>Janela:</b> imediata/diária</div>",
            unsafe_allow_html=True,
        )

        with st.expander("ℹ️ O que significam estes termos?"):
            st.markdown("""
- **Precision** — das anomalias que o modelo identificou, quantas eram mesmo anomalias reais.
- **Recall** — das anomalias reais que existiam, quantas o modelo conseguiu identificar.
- **F1-score** — equilíbrio entre Precision e Recall numa só métrica (quanto mais perto de 1, melhor).
- **FNR (Taxa de Falsos Negativos)** — % de ataques reais que o modelo **não detetou**. É a métrica mais crítica em segurança: cada falso negativo é um ataque que passou despercebido.
- **FPR (Taxa de Falsos Positivos)** — % de tráfego normal que o modelo classificou incorretamente como anómalo (falso alarme), gerando ruído operacional.
- **Fluxo Crítico** — registo de tráfego classificado no nível de risco mais alto, prioritário para investigação.
            """)

        with st.expander("📦 De onde vêm estes dados?"):
            st.markdown("""
Os dados apresentados combinam três origens, todas presentes no dataset filtrado:
- **Treino** — dados usados para treinar o Random Forest e a Logistic Regression.
- **Teste** — dados separados antes do treino, usados para validar o desempenho do modelo (split temporal 80/20).
- **Fora da Amostra** — a maioria dos registos (~87%), nunca usados em treino nem teste, avaliados aqui para verificar a generalização do modelo a dados completamente novos.

Esta separação evita *data leakage*: as métricas de desempenho apresentadas no dashboard não estão "infladas" por o modelo já ter visto estes dados durante o treino.
            """)

        k1, k2, k3, k4 = st.columns(4)
        for col_ui, label, value, cls in [
            (k1, "Registos",           f"{n_total:,}",          "blue"),
            (k2, "% Anómalo Real",     f"{pct_r:.1f}%",         "red"),
            (k3, "% Anómalo Previsto", f"{pct_p:.1f}%",         "orange"),
            (k4, "Fluxos Críticos",    f"{pct_crit:.1f}%",      "red"),
        ]:
            col_ui.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {cls}">{value}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin:10px 0 0'></div>", unsafe_allow_html=True)
        k5, k6, k7, k8 = st.columns(4)
        for col_ui, label, value, cls in [
            (k5, "Precision / Recall / F1", f"{prec:.2f} / {rec:.2f} / {f1:.2f}", "blue"),
            (k6, "Taxa de Falsos Negativos (FNR)", f"{fnr*100:.1f}%", cor_fnr(fnr)),
            (k7, "Ataques Não Detetados (nº)", f"{n_nao_detetados_total:,}", cor_fnr(fnr)),
            (k8, "Taxa de Falsos Positivos (FPR)", f"{fpr_rate*100:.1f}%", cor_fpr(fpr_rate)),
        ]:
            col_ui.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {cls}">{value}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin:24px 0 0'></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Análise de Detecção</div>", unsafe_allow_html=True)
        st.markdown("<div class='crit-bar red'></div>", unsafe_allow_html=True)
        c5, c6 = st.columns(2)

        with c5:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            top_p = a_port_filt.groupby("dst_port", observed=True)["n_pred"].sum().nlargest(top_n).reset_index()
            if len(top_p) > 0:
                top_p.columns = ["porto", "n"]
                top_p["porto_label"] = top_p["porto"].apply(formatar_porto)
                top_p = top_p.sort_values("n")
                fig_portos = px.bar(
                    top_p, x="n", y="porto_label", orientation="h",
                    color_discrete_sequence=[RED],
                    labels={"n": "Anomalias Previstas", "porto_label": "Porto"},
                    custom_data=["porto"],
                )
                fig_portos.update_traces(
                    hovertemplate="<b>Porto %{customdata[0]}</b><br>%{y}<br>Anomalias previstas: %{x:,}<extra></extra>"
                )
                fig_portos.update_layout(
                    **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
                    title=dict(text="Portos com Maior Volume de Anomalias (Prioridades de Investigação)",
                                x=0.01, xanchor="left", y=0.97, yanchor="top", font=dict(size=20, color=TEXT)),
                    height=460, margin=dict(l=16, r=16, t=60, b=16),
                )
                fig_portos.update_yaxes(type="category")
                st.plotly_chart(fig_portos, use_container_width=True, config={"displayModeBar": False})
                st.caption("Identifica os portos mais associados a comportamento anómalo, permitindo priorizar investigação no SOC.")
            else:
                st.info("Nenhuma anomalia prevista com os filtros atuais.")
            st.markdown('</div>', unsafe_allow_html=True)

        with c6:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            cm_pct = (cm / cm.sum(axis=1, keepdims=True) * 100).round(1)
            siglas = [["TN", "FP"], ["FN", "TP"]]
            anotacoes = [[f"<b>{siglas[i][j]}</b><br>{cm[i][j]:,}<br>({cm_pct[i][j]}%)" for j in range(2)] for i in range(2)]
            z_cor = [[1, 0], [0, 1]]
            fig_cm = go.Figure(go.Heatmap(
                z=z_cor, x=["Normal", "Anómalo"], y=["Normal", "Anómalo"],
                colorscale=[[0, RED], [1, BLUE]], showscale=False, zmin=0, zmax=1,
                text=anotacoes, texttemplate="%{text}", textfont=dict(color=TEXT, size=13),
            ))
            fig_cm.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
                title=dict(text=f"Qualidade de Deteção de Anomalias — {modelo_label}",
                            x=0.01, xanchor="left", y=0.97, yanchor="top", font=dict(size=20, color=TEXT)),
                xaxis_title="Previsto", yaxis_title="Real", height=460, margin=dict(l=16, r=16, t=60, b=16),
            )
            st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
            st.caption(f"Mostra a capacidade do modelo em identificar tráfego anómalo e normal, destacando falsos negativos e falsos positivos. **Precision: {prec:.3f} · Recall: {rec:.3f} · F1: {f1:.3f}**")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Tipos de Ataque — Detecção por Classe</div>", unsafe_allow_html=True)
        st.markdown("<div class='crit-bar red'></div>", unsafe_allow_html=True)
        c7, c8 = st.columns(2)

        top8 = a_atq_filt.groupby("attack_type", observed=True)["n"].sum().nlargest(8).index.tolist()

        with c7:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            contagem_real = a_atq_filt[a_atq_filt["attack_type"].isin(top8)].copy()
            contagem_real = contagem_real.groupby(["attack_type", "label_real"], observed=True)["n"].sum().reset_index()
            contagem_real["classe"] = contagem_real["label_real"].map({0: "Normal", 1: "Anómalo"})
            fig_tipos = px.bar(
                contagem_real, x="n", y="attack_type", color="classe", orientation="h",
                color_discrete_map={"Normal": BLUE, "Anómalo": RED},
                category_orders={"attack_type": top8},
                labels={"n": "Nº de Registos", "attack_type": "Tipo de Tráfego", "classe": "Classe (real)"},
            )
            fig_tipos.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("legend", "margin")},
                title=dict(
                    text="Composição do Tráfego por Tipo (Base para Avaliação do Modelo)",
                    x=0.01,
                    xanchor="left",
                    y=0.97,
                    yanchor="top",
                    font=dict(size=20, color=TEXT),
                ),
                barmode="stack", height=460,
                legend_title_text="",
                legend=dict(
                    orientation="h",
                    yanchor="top", y=1.20,
                    xanchor="left", x=-0.20,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color=TEXT, size=13),
                ),
                margin=dict(l=16, r=16, t=100, b=16),
            )
            st.plotly_chart(fig_tipos, use_container_width=True, config={"displayModeBar": False})
            st.caption("Apresenta a distribuição real de tráfego, evidenciando o desbalanceamento entre classes.")
            st.markdown('</div>', unsafe_allow_html=True)

        with c8:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            recall_agg = (
                a_atq_filt[a_atq_filt["attack_type"].isin(top8) & (a_atq_filt["label_real"] == 1)]
                .groupby("attack_type", observed=True)
                .agg(n_real=("n", "sum"), n_pred=("n_pred", "sum"))
                .reset_index()
            )
            recall_agg = recall_agg[recall_agg["n_real"] > 0].copy()
            recall_agg["taxa_deteccao"] = (recall_agg["n_pred"] / recall_agg["n_real"] * 100).round(1)
            recall_agg = recall_agg.sort_values("taxa_deteccao", ascending=False)

            def categoria_eficacia(pct):
                if pct > 80: return "Forte (>80%)"
                elif pct >= 50: return "Moderada (50–80%)"
                else: return "Fraca (<50%)"

            if len(recall_agg) > 0:
                recall_agg["categoria"] = recall_agg["taxa_deteccao"].apply(categoria_eficacia)
                fig_recall = px.bar(
                    recall_agg, x="taxa_deteccao", y="attack_type", orientation="h",
                    color="categoria",
                    color_discrete_map={"Forte (>80%)": BLUE, "Moderada (50–80%)": ORANGE, "Fraca (<50%)": RED},
                    category_orders={
                        "categoria": ["Forte (>80%)", "Moderada (50–80%)", "Fraca (<50%)"],
                        "attack_type": recall_agg["attack_type"].tolist(),
                    },
                    labels={"taxa_deteccao": "Taxa de Detecção (%)", "attack_type": "Tipo de Ataque", "categoria": "Eficácia"},
                    text="taxa_deteccao",
                )
                fig_recall.update_traces(
                    texttemplate="%{text:.1f}%", textposition="outside",
                    textfont=dict(color=TEXT, size=12), cliponaxis=False,
                )
                fig_recall.update_layout(
                    **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("legend", "margin")},
                    title=dict(
                        text="Capacidade de Deteção por Tipo de Ataque (Pontos Fortes vs Falhas)",
                        x=0.01,
                        xanchor="left",
                        y=0.97,
                        yanchor="top",
                        font=dict(size=20, color=TEXT),
                    ),
                    height=460,
                    legend_title_text="",
                    legend=dict(
                        orientation="h",
                        yanchor="top", y=1.20,
                        xanchor="left", x=-0.20,
                        bgcolor="rgba(0,0,0,0)", borderwidth=0,
                        font=dict(color=TEXT, size=13),
                    ),
                    margin=dict(l=16, r=16, t=100, b=16),
                )
                fig_recall.update_xaxes(range=[0, 122], showgrid=False, zeroline=False, showline=False, tickcolor=TEXT, color=TEXT)
                st.plotly_chart(fig_recall, use_container_width=True, config={"displayModeBar": False})
                st.caption("Permite identificar em que tipos de ataque o modelo tem melhor e pior desempenho.")
            else:
                st.info("Sem anomalias reais no subset filtrado para calcular taxa de detecção.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Decision Layer — Insights e Recomendações ──────────────
        st.markdown("<div class='section-title'>📌 Insights e Recomendações</div>", unsafe_allow_html=True)
        st.markdown("<div class='crit-bar red'></div>", unsafe_allow_html=True)

        with st.expander("ℹ️ Como são gerados estes insights?"):
            st.markdown("""
Os insights abaixo são gerados automaticamente por um conjunto de regras fixas, aplicadas aos dados filtrados:
- **Porto com mais anomalias** → identifica sempre o porto com maior nº absoluto de ocorrências previstas.
- **Tipo de ataque com pior deteção** → sinalizado se algum tipo de ataque tiver taxa de deteção **< 50%**.
- **Falsos positivos elevados** → alerta se a Taxa de Falsos Positivos (FPR) for **≥ 5%**.
- **Falsos negativos elevados** → alerta de prioridade máxima se a Taxa de Falsos Negativos (FNR) for **≥ 10%**.

Se nenhuma destas condições se verificar, é mostrada a mensagem "sem alertas críticos". Os limiares (50%, 5%, 10%) foram definidos com base nos valores de referência usados nos KPIs de cor (verde/amarelo/vermelho) do dashboard.
            """)

        insights_soc = []
        if len(top_p) > 0:
            porto_top = top_p.sort_values("n", ascending=False).iloc[0]
            insights_soc.append(
                f"🔴 Elevada concentração de anomalias no porto **{formatar_porto(porto_top['porto'])}** "
                f"({int(porto_top['n']):,} ocorrências previstas) → priorizar investigação deste tráfego."
            )
        if len(recall_agg) > 0:
            piores = recall_agg[recall_agg["taxa_deteccao"] < 50]
            if len(piores) > 0:
                pior = piores.sort_values("taxa_deteccao").iloc[0]
                insights_soc.append(
                    f"🟠 Modelo apresenta baixa eficácia em **{pior['attack_type']}** "
                    f"({pior['taxa_deteccao']:.1f}% de detecção) → considerar melhorar features ou re-treinar o modelo para esta classe."
                )
        if fpr_rate >= 0.05:
            insights_soc.append(
                f"🟡 Taxa de falsos positivos em **{fpr_rate*100:.1f}%** → "
                f"considerar ajustar o threshold de decisão para reduzir ruído operacional."
            )
        if fnr >= 0.10:
            insights_soc.append(
                f"🔴 Taxa de falsos negativos em **{fnr*100:.1f}%** (~{n_nao_detetados_total:,} ataques não detetados) → "
                f"risco de segurança a tratar com prioridade."
            )
        if not insights_soc:
            insights_soc.append("🟢 Sem alertas críticos identificados nos dados filtrados atualmente.")

        for ins in insights_soc:
            st.markdown(f"<div class='audience-banner' style='margin:6px 24px'>{ins}</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# ABA 2 — TENDÊNCIAS / GESTÃO
# ──────────────────────────────────────────────────────────────────
elif aba_ativa == "📈 Tendências / Gestão":
    st.sidebar.markdown(f"<div style='color:{ORANGE};font-size:18px;font-weight:700;padding:8px 0'>📈 Filtros — Tendências</div>", unsafe_allow_html=True)

    modelo_label_t = st.sidebar.radio("Modelo", ["Random Forest", "Logistic Regression"], key="modelo_tendencias")
    suf_t = "rf" if modelo_label_t == "Random Forest" else "lr"

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<div style='font-size:12px;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>Dia</div>", unsafe_allow_html=True)
    dias_sel_t = [d for d in ORDEM_DIAS if st.sidebar.checkbox(d, value=True, key=f"dia_t_{d}")]
    if not dias_sel_t:
        dias_sel_t = ORDEM_DIAS

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"<div style='font-size:12px;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>Nível de Risco</div>", unsafe_allow_html=True)
    riscos_sel_t = [r for r in ORDEM_RISCO if st.sidebar.checkbox(r, value=True, key=f"risco_t_{r}")]
    if not riscos_sel_t:
        riscos_sel_t = ORDEM_RISCO

    col_risco_t = f"risco_{suf_t}"
    col_prob_t  = f"prob_{suf_t}"
    split_sel_t = ORDEM_SPLIT

    a_dia_t = agg_dia[suf_t]
    risco_dia_filt_t = a_dia_t["risco_dia"][
        a_dia_t["risco_dia"]["day"].isin(dias_sel_t) &
        a_dia_t["risco_dia"][col_risco_t].isin(riscos_sel_t) &
        a_dia_t["risco_dia"]["split"].isin(split_sel_t)
    ]
    sample_filt_t = df_sample[
        df_sample["day"].isin(dias_sel_t) &
        df_sample[col_risco_t].isin(riscos_sel_t) &
        df_sample["split"].isin(split_sel_t)
    ]
    a_atq_t = agg_ataque[suf_t]
    a_atq_filt_t = a_atq_t[
        a_atq_t["day"].isin(dias_sel_t) &
        a_atq_t[col_risco_t].isin(riscos_sel_t) &
        a_atq_t["split"].isin(split_sel_t)
    ]

    n_total_t = int(a_atq_filt_t["n"].sum())

    if n_total_t == 0:
        st.warning("Nenhum registo corresponde aos filtros selecionados.")
    else:
        pct_real_geral = round(a_atq_filt_t["n_real"].sum() / n_total_t * 100, 1)
        dia_pico = a_dia_t["por_dia"].loc[lambda x: x.index.isin(dias_sel_t), "pct_real"].idxmax() if len(dias_sel_t) > 0 else "—"
        pct_pico = a_dia_t["por_dia"].loc[lambda x: x.index.isin(dias_sel_t), "pct_real"].max() if len(dias_sel_t) > 0 else 0
        risco_counts_t = a_atq_filt_t.groupby(col_risco_t, observed=True)["n"].sum()
        pct_crit_t = round(risco_counts_t.get("Crítico", 0) / n_total_t * 100, 1)

        # Comparação com o dia anterior (dentro dos dias filtrados, em ordem cronológica)
        dias_ord_t = [d for d in ORDEM_DIAS if d in dias_sel_t]
        delta_str = ""
        delta_cls = "blue"
        if len(dias_ord_t) >= 2:
            pct_ultimo = a_dia_t["por_dia"].loc[dias_ord_t[-1], "pct_real"]
            pct_anterior = a_dia_t["por_dia"].loc[dias_ord_t[-2], "pct_real"]
            delta = pct_ultimo - pct_anterior
            seta = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            delta_cls = "red" if delta > 0 else ("blue" if delta < 0 else "blue")
            delta_str = f"{pct_ultimo:.1f}% {seta} {delta:+.1f}pp vs {dias_ord_t[-2]}"
        else:
            delta_str = f"{pct_real_geral:.1f}%"

        st.markdown(
            f"<div class='audience-banner'><b>Audiência:</b> Gestor SOC / Operações &nbsp;·&nbsp; "
            f"<b>Decisão:</b> escalar recursos ou não &nbsp;·&nbsp; <b>Janela:</b> diária/semanal</div>",
            unsafe_allow_html=True,
        )

        with st.expander("ℹ️ O que significam estes termos?"):
            st.markdown("""
- **% Anómalo Real** — proporção de tráfego que é efetivamente anómalo, segundo o ground truth (rótulo real, não previsão do modelo).
- **Dia de Pico** — dia da semana com maior percentagem de tráfego anómalo no período filtrado.
- **Fluxos Críticos** — % de tráfego no nível de risco mais alto (Crítico), que exige resposta prioritária.
- **Tendência + Projeção** — reta de regressão linear ajustada aos dias filtrados, usada para estimar o comportamento esperado no dia seguinte (não é uma previsão de modelo, é uma extrapolação estatística simples).
- **Intervalo de Incerteza (±3%)** — margem de erro assumida para a projeção; quanto mais distante no tempo, maior a incerteza real.
            """)

        with st.expander("📦 De onde vêm estes dados?"):
            st.markdown("""
Os valores apresentados nesta aba combinam dados de **Treino**, **Teste** e **Fora da Amostra** (a maioria, ~87% dos registos, nunca usados para treinar os modelos). Isto garante que as tendências mostradas refletem o comportamento real do tráfego, e não um efeito de o modelo "já ter visto" estes dados antes.
            """)

        k1, k2, k3, k4, k5 = st.columns(5)
        for col_ui, label, value, cls in [
            (k1, "Registos",            f"{n_total_t:,}",      "blue"),
            (k2, "% Anómalo Real (médio)", f"{pct_real_geral:.1f}%", "orange"),
            (k3, "Último Dia vs Anterior", delta_str, delta_cls),
            (k4, "Dia de Pico",          f"{dia_pico} ({pct_pico:.1f}%)", "red"),
            (k5, "Fluxos Críticos",      f"{pct_crit_t:.1f}%",  "red"),
        ]:
            col_ui.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {cls}">{value}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin:24px 0 0'></div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Evolução Temporal</div>", unsafe_allow_html=True)
        st.markdown("<div class='crit-bar yellow'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            df_evo_rf = agg_dia["rf"]["por_dia"].loc[lambda x: x.index.isin(dias_sel_t), ["pct_real", "pct_pred"]].reset_index()
            df_evo_lr = agg_dia["lr"]["por_dia"].loc[lambda x: x.index.isin(dias_sel_t), ["pct_pred"]].reset_index()

            fig_linha = go.Figure()
            fig_linha.add_trace(go.Scatter(x=df_evo_rf["day"], y=df_evo_rf["pct_real"], mode="lines+markers", name="Real",
                                             line=dict(color=TEXT, width=2.5, dash="dot"), marker=dict(size=8)))
            fig_linha.add_trace(go.Scatter(x=df_evo_rf["day"], y=df_evo_rf["pct_pred"], mode="lines+markers", name="Random Forest",
                                             line=dict(color=BLUE, width=2.5), marker=dict(size=8)))
            fig_linha.add_trace(go.Scatter(x=df_evo_lr["day"], y=df_evo_lr["pct_pred"], mode="lines+markers", name="Logistic Regression",
                                             line=dict(color=TEAL, width=2.5, dash="dash"), marker=dict(size=8)))

            dias_ord_filt = [d for d in ORDEM_DIAS if d in dias_sel_t]
            if len(dias_ord_filt) >= 2:
                x_num = np.arange(len(dias_ord_filt))
                y_vals = df_evo_rf.set_index("day").reindex(dias_ord_filt)["pct_real"].values
                coef = np.polyfit(x_num, y_vals, 1)
                tendencia = np.poly1d(coef)
                x_proj = np.append(x_num, len(dias_ord_filt))
                y_proj = tendencia(x_proj)
                eixo_x_proj = dias_ord_filt + ["Próximo Dia (Projeção)"]
                margem = 3.0

                fig_linha.add_trace(go.Scatter(x=eixo_x_proj, y=y_proj.round(2), mode="lines", name="Tendência + Projeção",
                                                 line=dict(color=ORANGE, width=2, dash="dash")))
                fig_linha.add_trace(go.Scatter(x=eixo_x_proj[-1:], y=[round(y_proj[-1], 2)], mode="markers",
                                                 name=f"Projeção: {y_proj[-1]:.1f}%", marker=dict(color=RED, size=12)))
                fig_linha.add_trace(go.Scatter(
                    x=eixo_x_proj[-2:] + eixo_x_proj[-2:][::-1],
                    y=[y_proj[-2] - margem, y_proj[-1] - margem, y_proj[-1] + margem, y_proj[-2] + margem],
                    fill="toself", fillcolor="rgba(192,80,77,0.15)", line=dict(color="rgba(0,0,0,0)"),
                    name=f"Incerteza ±{margem}%", showlegend=True, hoverinfo="skip",
                ))

            # Destaque visual do dia de pico de anomalias
            if dia_pico in dias_ord_filt:
                idx_pico = dias_ord_filt.index(dia_pico)
                fig_linha.add_vrect(
                    x0=idx_pico - 0.4, x1=idx_pico + 0.4,
                    fillcolor=ORANGE, opacity=0.12, line_width=0, layer="below",
                    annotation_text=f"Pico: {dia_pico}", annotation_position="top",
                    annotation_font=dict(color=ORANGE, size=10),
                )

            layout_linha = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("legend", "margin")}
            fig_linha.update_layout(
                **layout_linha,
                title=dict(
                    text="Evolução Temporal de Anomalias e Projeção de Risco",
                    x=0.01,
                    xanchor="left",
                    y=0.97,
                    yanchor="top",
                    font=dict(size=20, color=TEXT),
                ),
                yaxis_title="% Anomalias", height=460,
                legend_title_text="",
                legend=dict(
                    orientation="h",
                    yanchor="top", y=1.20,
                    xanchor="left", x=-0.05,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color=TEXT, size=13),
                ),
                margin=dict(l=16, r=16, t=100, b=16),
            )
            st.plotly_chart(fig_linha, use_container_width=True, config={"displayModeBar": False})
            st.caption("Mostra a tendência ao longo dos dias e a projeção para o próximo período (regressão linear sobre os dias filtrados, com intervalo de incerteza de ±3%). A zona sombreada destaca o dia de pico de atividade anómala.")
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            rd = risco_dia_filt_t.copy()
            rd[col_risco_t] = pd.Categorical(rd[col_risco_t], categories=ORDEM_RISCO, ordered=True)
            rd = rd.groupby(["day", col_risco_t], observed=True, as_index=False)["pct"].sum()
            rd["day"] = pd.Categorical(rd["day"], categories=ORDEM_DIAS, ordered=True)
            rd = rd.sort_values("day")
            fig_risco_dia = go.Figure()
            for risco in ORDEM_RISCO:
                sub = rd[rd[col_risco_t] == risco]
                texto = sub["pct"].apply(lambda v: f"{v:.0f}%" if v >= 4 else "")
                fig_risco_dia.add_trace(go.Bar(
                    x=sub["day"], y=sub["pct"], name=risco, marker_color=CORES_RISCO[risco],
                    text=texto, textposition="inside",
                    textfont=dict(color=BG, size=11),
                    customdata=sub["pct"], hovertemplate=f"{risco}: " + "%{customdata:.1f}%<extra></extra>",
                ))

            # Destaque visual do dia de pico de anomalias (mesmo dia identificado no gráfico de evolução)
            dias_presentes_risco = [d for d in ORDEM_DIAS if d in rd["day"].unique()]
            if dia_pico in dias_presentes_risco:
                idx_pico_rd = dias_presentes_risco.index(dia_pico)
                fig_risco_dia.add_vrect(
                    x0=idx_pico_rd - 0.4, x1=idx_pico_rd + 0.4,
                    fillcolor=ORANGE, opacity=0.12, line_width=0, layer="below",
                    annotation_text=f"Pico: {dia_pico}", annotation_position="top",
                    annotation_font=dict(color=ORANGE, size=10),
                )

            layout_risco_dia = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("legend", "margin")}
            fig_risco_dia.update_layout(
                **layout_risco_dia,
                title=dict(
                    text="Distribuição de Risco por Dia (Impacto Operacional)",
                    x=0.01,
                    xanchor="left",
                    y=0.97,
                    yanchor="top",
                    font=dict(size=20, color=TEXT),
                ),
                barmode="stack",
                yaxis_title="% de Fluxos", height=460,
                legend_title_text="",
                legend=dict(
                    orientation="h",
                    yanchor="top", y=1.20,
                    xanchor="left", x=-0.05,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color=TEXT, size=13),
                ),
                margin=dict(l=16, r=16, t=100, b=16),
                uniformtext=dict(mode="show", minsize=11),
            )
            st.plotly_chart(fig_risco_dia, use_container_width=True, config={"displayModeBar": False})
            st.caption("Quantifica a percentagem de tráfego classificado por nível de risco ao longo dos dias. Valores ≤4% não são rotulados para evitar sobreposição — passar o rato sobre a barra para ver o valor exato. A zona sombreada destaca o dia de pico de atividade anómala.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div class='section-title'>Distribuição e Risco</div>", unsafe_allow_html=True)
        st.markdown("<div class='crit-bar yellow'></div>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)

        with c3:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            top_tipos = sample_filt_t["attack_type"].value_counts().head(8).index.tolist()
            df_box = sample_filt_t[sample_filt_t["attack_type"].isin(top_tipos)]

            # Nível de risco predominante (moda) por tipo de ataque, para colorir cada caixa
            risco_predominante = (
                df_box.groupby("attack_type", observed=True)[col_risco_t]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else "Baixo")
            )
            cor_por_tipo = {t: CORES_RISCO[risco_predominante.get(t, "Baixo")] for t in top_tipos}

            fig_box = px.box(
                df_box, x="attack_type", y=col_prob_t, color="attack_type",
                color_discrete_map=cor_por_tipo,
                labels={"attack_type": "Tipo de Tráfego", col_prob_t: "Probabilidade de Anomalia"},
                category_orders={"attack_type": top_tipos},
            )
            fig_box.add_hline(y=0.5, line_dash="dash", line_color=TEXT_MUTED, line_width=1.5,
                                annotation_text="Threshold de decisão (0.5)", annotation_position="top left",
                                annotation_font=dict(color=TEXT_MUTED, size=10))
            fig_box.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
                title=dict(text="Distribuição de Confiança do Modelo por Tipo de Tráfego (cor = risco predominante)",
                            x=0.01, xanchor="left", y=0.97, yanchor="top", font=dict(size=20, color=TEXT)),
                showlegend=False, height=460, xaxis_tickangle=-30, xaxis_title=None,
                margin=dict(l=16, r=16, t=60, b=16),
            )
            fig_box.update_traces(marker_size=3)
            st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})
            st.caption("Analisa como o modelo atribui probabilidades às diferentes classes. A linha tracejada marca o threshold de decisão (0.5).")
            st.markdown('</div>', unsafe_allow_html=True)

        with c4:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            risco_agg = a_atq_filt_t.groupby(col_risco_t, observed=True)["n"].sum().reindex(ORDEM_RISCO).fillna(0)
            total_risco = risco_agg.sum()
            pcts_risco = (risco_agg / total_risco * 100).round(0) if total_risco > 0 else risco_agg

            import math

            def arco_svg_path(cx, cy, r_outer, r_inner, pct, n_segmentos=60):
                """
                Path de um anel de progresso completo (0-100%), início no topo, sentido horário.
                NOTA: o Plotly não suporta o comando de arco elíptico 'A' em shapes do tipo path
                (é aceite sem erro mas renderiza de forma degenerada/invisível). Por isso o arco
                é aproximado por um polígono de pequenos segmentos de linha 'L', que é suportado.
                """
                pct = max(0.5, min(99.5, pct))
                angulo_total = pct / 100 * 360
                angulo_inicio = 90  # topo, em coordenadas Plotly (y crescente para cima)
                angulo_fim = angulo_inicio - angulo_total  # sinal negativo = sentido horário no ecrã

                angulos = [angulo_inicio + (angulo_fim - angulo_inicio) * t / n_segmentos for t in range(n_segmentos + 1)]
                pontos_out = [(cx + r_outer * math.cos(math.radians(a)), cy + r_outer * math.sin(math.radians(a))) for a in angulos]
                pontos_in  = [(cx + r_inner * math.cos(math.radians(a)), cy + r_inner * math.sin(math.radians(a))) for a in reversed(angulos)]
                todos_pontos = pontos_out + pontos_in

                path = f"M {todos_pontos[0][0]:.6f},{todos_pontos[0][1]:.6f} "
                for x, y in todos_pontos[1:]:
                    path += f"L {x:.6f},{y:.6f} "
                path += "Z"
                return path

            fig_gauges = go.Figure()

            n_gauges = len(ORDEM_RISCO)
            R_OUT, R_IN = 0.85, 0.62
            ESPACO = 2.4
            Y_CENTRO_GAUGE = 1.55
            Y_TIMELINE = 0.0
            Y_LEGENDA = -0.55

            centros_x = [i * ESPACO for i in range(n_gauges)]

            for i, risco in enumerate(ORDEM_RISCO):
                pct = pcts_risco[risco]
                cx = centros_x[i]

                # Anel de fundo (trilho cinza, círculo completo)
                fig_gauges.add_shape(
                    type="path",
                    path=arco_svg_path(cx, Y_CENTRO_GAUGE, R_OUT, R_IN, 100),
                    fillcolor=GRID, line=dict(width=0),
                )
                # Anel de progresso (cor do risco, proporcional ao %)
                fig_gauges.add_shape(
                    type="path",
                    path=arco_svg_path(cx, Y_CENTRO_GAUGE, R_OUT, R_IN, pct),
                    fillcolor=CORES_RISCO[risco], line=dict(width=0),
                )
                # Percentagem ao centro
                fig_gauges.add_annotation(
                    x=cx, y=Y_CENTRO_GAUGE, text=f"<b>{pct:.0f}%</b>", showarrow=False,
                    font=dict(color=CORES_RISCO[risco], size=22),
                )

            # Linha horizontal da timeline
            fig_gauges.add_shape(
                type="line", x0=centros_x[0], x1=centros_x[-1], y0=Y_TIMELINE, y1=Y_TIMELINE,
                line=dict(color=GRID, width=2),
            )

            for i, risco in enumerate(ORDEM_RISCO):
                cx = centros_x[i]
                # Linha conectora ponto → anel
                fig_gauges.add_shape(
                    type="line", x0=cx, x1=cx, y0=Y_TIMELINE, y1=Y_CENTRO_GAUGE - R_OUT,
                    line=dict(color=CORES_RISCO[risco], width=1.5),
                )
                # Ponto na timeline
                fig_gauges.add_shape(
                    type="circle", x0=cx - 0.10, x1=cx + 0.10, y0=Y_TIMELINE - 0.10, y1=Y_TIMELINE + 0.10,
                    line=dict(color=CORES_RISCO[risco], width=2), fillcolor=CARD_BG,
                )
                # Legenda
                fig_gauges.add_annotation(
                    x=cx, y=Y_LEGENDA, text=f"Risco {risco}", showarrow=False,
                    font=dict(color=TEXT_MUTED, size=11),
                )

            fig_gauges.update_xaxes(visible=False, range=[centros_x[0] - 1.2, centros_x[-1] + 1.2])
            fig_gauges.update_yaxes(visible=False, range=[Y_LEGENDA - 0.3, Y_CENTRO_GAUGE + R_OUT + 0.2], scaleanchor="x", scaleratio=1)
            fig_gauges.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
                title=dict(text=f"Distribuição de Risco — {modelo_label_t}",
                            x=0.01, xanchor="left", y=0.97, yanchor="top", font=dict(size=20, color=TEXT)),
                height=460, margin=dict(l=16, r=16, t=60, b=16),
            )
            st.plotly_chart(fig_gauges, use_container_width=True, config={"displayModeBar": False})
            st.caption("Percentagem de fluxos classificados por cada nível de risco. O arco colorido mostra a proporção de cada categoria no total filtrado.")
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📌 Insights e Recomendações</div>", unsafe_allow_html=True)
        st.markdown("<div class='crit-bar yellow'></div>", unsafe_allow_html=True)

        with st.expander("ℹ️ Como são gerados estes insights?"):
            st.markdown("""
Os insights abaixo são gerados automaticamente por um conjunto de regras fixas, aplicadas aos dados filtrados:
- **Dia de pico** → identifica sempre o dia da semana com maior % de tráfego anómalo real no período filtrado.
- **Tendência dia-a-dia** → compara o último dia filtrado (cronologicamente) com o anterior; sinaliza subida ou descida.
- **Fluxos críticos** → alerta de impacto operacional elevado se a % de fluxos classificados como risco Crítico for **≥ 10%**.

Se nenhuma destas condições se verificar, é mostrada a mensagem "sem alertas críticos".
            """)

        insights_tend = []
        if dia_pico != "—":
            insights_tend.append(
                f"🟠 Pico de atividade anómala em **{dia_pico}** ({pct_pico:.1f}%) → "
                f"considerar reforço operacional nesse dia da semana."
            )
        if len(dias_ord_t) >= 2 and delta > 0:
            insights_tend.append(
                f"🔴 Tendência a subir: **{dias_ord_t[-1]}** registou +{delta:.1f}pp face a **{dias_ord_t[-2]}** → "
                f"vigilância reforçada recomendada para o próximo período."
            )
        elif len(dias_ord_t) >= 2 and delta < 0:
            insights_tend.append(
                f"🟢 Tendência a descer: **{dias_ord_t[-1]}** registou {delta:.1f}pp face a **{dias_ord_t[-2]}** → "
                f"sem necessidade de reforço imediato."
            )
        if pct_crit_t >= 10:
            insights_tend.append(
                f"🔴 {pct_crit_t:.1f}% dos fluxos classificados como risco Crítico → "
                f"impacto operacional elevado, recomenda-se priorização de recursos de resposta."
            )
        if not insights_tend:
            insights_tend.append("🟢 Sem alertas críticos identificados nos dados filtrados atualmente.")

        for ins in insights_tend:
            st.markdown(f"<div class='audience-banner' style='margin:6px 24px'>{ins}</div>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# ABA 3 — COMPARAÇÃO DE MODELOS
# ──────────────────────────────────────────────────────────────────
elif aba_ativa == "⚖️ Comparação de Modelos":
    st.sidebar.markdown(f"<div style='color:{GREEN};font-size:18px;font-weight:700;padding:8px 0'>⚖️ Filtros — Comparação</div>", unsafe_allow_html=True)

    st.sidebar.markdown(f"<div style='font-size:12px;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>Dia</div>", unsafe_allow_html=True)
    dias_sel_m = [d for d in ORDEM_DIAS if st.sidebar.checkbox(d, value=True, key=f"dia_m_{d}")]
    if not dias_sel_m:
        dias_sel_m = ORDEM_DIAS

    st.sidebar.markdown("---")
    if HAS_SPLIT:
        st.sidebar.markdown(f"<div style='font-size:12px;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:1px;margin-bottom:4px'>Split</div>", unsafe_allow_html=True)
        split_sel_m = [s for s in ORDEM_SPLIT if st.sidebar.checkbox(s, value=True, key=f"split_m_{s}")]
        if not split_sel_m:
            split_sel_m = ORDEM_SPLIT
    else:
        split_sel_m = ORDEM_SPLIT
        st.sidebar.markdown(f"<div style='font-size:11px;color:{TEXT_MUTED};font-style:italic'>Filtro Split disponível após regenerar o parquet.</div>", unsafe_allow_html=True)

    sample_filt_m = df_sample[
        df_sample["day"].isin(dias_sel_m) &
        df_sample["split"].isin(split_sel_m)
    ]

    if len(sample_filt_m) == 0:
        st.warning("Nenhum registo corresponde aos filtros selecionados.")
    else:
        st.markdown(
            f"<div class='audience-banner'><b>Audiência:</b> Data Scientist / Eng. responsável pelo pipeline &nbsp;·&nbsp; "
            f"<b>Decisão:</b> qual modelo produtizar &nbsp;·&nbsp; <b>Janela:</b> pontual (deployment)</div>",
            unsafe_allow_html=True,
        )

        with st.expander("ℹ️ O que significam estes termos?"):
            st.markdown("""
- **Precision** — das previsões positivas (anómalo) do modelo, quantas estavam corretas.
- **Recall** — das anomalias reais existentes, quantas o modelo conseguiu identificar.
- **F1-score** — média harmónica entre Precision e Recall; penaliza fortemente um desequilíbrio entre as duas.
- **ROC-AUC** — área sob a curva ROC; mede a capacidade do modelo separar as duas classes, independentemente do threshold escolhido (1.0 = separação perfeita, 0.5 = aleatório).
- **Threshold de decisão (0.5)** — ponto de corte na probabilidade prevista a partir do qual um registo é classificado como anómalo. Pode ser ajustado para trocar Precision por Recall (ou vice-versa).
- **TN / FP / FN / TP** — Verdadeiro Negativo, Falso Positivo, Falso Negativo, Verdadeiro Positivo (as 4 células da matriz de confusão).
            """)

        with st.expander("📦 De onde vêm estes dados?"):
            st.markdown("""
O pipeline usa um **split temporal 80/20**: os primeiros registos cronologicamente são usados para Treino, os seguintes para Teste — em vez de uma divisão aleatória, para evitar que o modelo "veja o futuro" durante o treino.

| Split | Registos | Papel |
|---|---|---|
| **Treino** | 160.000 | Usados para ajustar os parâmetros do Random Forest e da Logistic Regression |
| **Teste** | 40.000 | Reservados antes do treino, usados para a validação inicial do modelo |
| **Fora da Amostra** | 1.382.029 (~87%) | Nunca vistos em treino nem teste — avaliados aqui para confirmar a generalização real do modelo |

Os filtros de Split nesta aba permitem isolar cada grupo e confirmar que o desempenho do modelo não está inflacionado por *data leakage* (ex: comparar as métricas em "Teste" vs "Fora da Amostra" — devem ser semelhantes).
            """)

        eixos = ["Precision", "Recall", "F1-score", "ROC-AUC", "Accuracy"]
        metricas_modelo = {}
        roc_data = {}
        for m_suf, m_nome in [("rf", "Random Forest"), ("lr", "Logistic Regression")]:
            y_true = sample_filt_m["label_real"]
            y_pred = sample_filt_m[f"pred_{m_suf}"]
            y_prob = sample_filt_m[f"prob_{m_suf}"]
            if y_true.nunique() > 1:
                auc = roc_auc_score(y_true, y_prob)
                fpr, tpr, _ = roc_curve(y_true, y_prob)
            else:
                auc = float("nan")
                fpr, tpr = np.array([0, 1]), np.array([0, 1])
            metricas_modelo[m_nome] = [
                round(precision_score(y_true, y_pred, zero_division=0), 3),
                round(recall_score(y_true, y_pred, zero_division=0), 3),
                round(f1_score(y_true, y_pred, zero_division=0), 3),
                round(auc, 3),
                round(accuracy_score(y_true, y_pred), 3),
            ]
            roc_data[m_nome] = (fpr, tpr, auc)

        k1, k2, k3, k4 = st.columns(4)
        for col_ui, label, value, cls in [
            (k1, "Registos (amostra)", f"{len(sample_filt_m):,}", "blue"),
            (k2, "F1 — Random Forest", f"{metricas_modelo['Random Forest'][2]:.3f}", "blue"),
            (k3, "F1 — Logistic Regr.", f"{metricas_modelo['Logistic Regression'][2]:.3f}", "blue"),
            (k4, "Diferença F1", f"{abs(metricas_modelo['Random Forest'][2] - metricas_modelo['Logistic Regression'][2]):.3f}", "orange"),
        ]:
            col_ui.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value {cls}">{value}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div style='margin:24px 0 0'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Comparação de Modelos</div>", unsafe_allow_html=True)
        st.markdown("<div class='crit-bar green'></div>", unsafe_allow_html=True)
        c9, c10 = st.columns(2)

        with c9:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            valores_rf = metricas_modelo["Random Forest"]
            valores_lr = metricas_modelo["Logistic Regression"]
            fig_dumb = go.Figure()
            for i, eixo in enumerate(eixos):
                fig_dumb.add_trace(go.Scatter(x=[valores_rf[i], valores_lr[i]], y=[eixo, eixo], mode="lines",
                                                line=dict(color=GRID, width=3), showlegend=False, hoverinfo="skip"))
            fig_dumb.add_trace(go.Scatter(x=valores_rf, y=eixos, mode="markers+text", name="Random Forest",
                                            marker=dict(color=BLUE, size=16, line=dict(color=CARD_BG, width=2)),
                                            text=[f"{v:.3f}" for v in valores_rf], textposition="top center",
                                            textfont=dict(color=BLUE, size=11)))
            fig_dumb.add_trace(go.Scatter(x=valores_lr, y=eixos, mode="markers+text", name="Logistic Regression",
                                            marker=dict(color=TEAL, size=16, line=dict(color=CARD_BG, width=2)),
                                            text=[f"{v:.3f}" for v in valores_lr], textposition="bottom center",
                                            textfont=dict(color=TEAL, size=11)))
            layout_dumb = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("legend", "margin", "xaxis")}
            fig_dumb.update_layout(
                **layout_dumb,
                title=dict(
                    text="Comparação de Modelos (Decisão de Produção)",
                    x=0.01,
                    xanchor="left",
                    y=0.97,
                    yanchor="top",
                    font=dict(size=20, color=TEXT),
                ),
                xaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=0.7, zeroline=False,
                            showline=False, tickcolor=TEXT, color=TEXT, range=[0, 1.1],
                            title="Valor da Métrica"),
                height=460,
                legend_title_text="",
                legend=dict(
                    orientation="h",
                    yanchor="top", y=1.20,
                    xanchor="left", x=-0.05,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color=TEXT, size=13),
                ),
                margin=dict(l=16, r=16, t=100, b=16),
            )
            st.plotly_chart(fig_dumb, use_container_width=True, config={"displayModeBar": False})
            st.caption("Compara o desempenho dos modelos para apoiar a decisão de deployment.")
            st.markdown('</div>', unsafe_allow_html=True)

        with c10:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig_roc = go.Figure()
            for m_nome, cor in [("Random Forest", BLUE), ("Logistic Regression", TEAL)]:
                fpr, tpr, auc = roc_data[m_nome]
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{m_nome} (AUC={auc:.3f})",
                                               line=dict(color=cor, width=2.5)))
                # Marcar o ponto correspondente ao threshold de decisão atual (0.5)
                y_pred_atual = sample_filt_m[f"pred_{'rf' if m_nome == 'Random Forest' else 'lr'}"]
                y_true_atual = sample_filt_m["label_real"]
                fpr_atual = ((y_pred_atual == 1) & (y_true_atual == 0)).sum() / max((y_true_atual == 0).sum(), 1)
                tpr_atual = ((y_pred_atual == 1) & (y_true_atual == 1)).sum() / max((y_true_atual == 1).sum(), 1)
                fig_roc.add_trace(go.Scatter(x=[fpr_atual], y=[tpr_atual], mode="markers",
                                               name=f"{m_nome} — threshold 0.5", marker=dict(color=cor, size=11, symbol="x")))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatório",
                                           line=dict(color=TEXT_MUTED, width=1.5, dash="dot")))
            layout_roc = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("legend", "margin")}
            fig_roc.update_layout(
                **layout_roc,
                title=dict(
                    text="Capacidade de Separação entre Classes (ROC-AUC)",
                    x=0.01,
                    xanchor="left",
                    y=0.97,
                    yanchor="top",
                    font=dict(size=20, color=TEXT),
                ),
                xaxis_title="Falsos Positivos (FPR)", yaxis_title="Verdadeiros Positivos (TPR)",
                height=460,
                legend_title_text="",
                legend=dict(
                    orientation="h",
                    yanchor="top", y=1.20,
                    xanchor="left", x=-0.05,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color=TEXT, size=13),
                ),
                margin=dict(l=16, r=16, t=100, b=16),
            )
            st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})
            st.caption("Avalia a capacidade do modelo separar tráfego normal de anómalo. O 'x' marca o ponto correspondente ao threshold de decisão atual (0.5).")
            st.markdown('</div>', unsafe_allow_html=True)

        c11, c12 = st.columns(2)

        with c11:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig_violin = go.Figure()
            for risco in ORDEM_RISCO:
                sub = sample_filt_m[sample_filt_m["risco_rf"] == risco]
                if len(sub) > 0:
                    fig_violin.add_trace(go.Violin(y=sub["prob_rf"], name=risco, box_visible=True, meanline_visible=True,
                                                     line_color=CORES_RISCO[risco], fillcolor=CORES_RISCO[risco], opacity=0.6))
            fig_violin.add_hline(y=0.5, line_dash="dash", line_color=TEXT_MUTED, line_width=1.5,
                                   annotation_text="Threshold de decisão (0.5)", annotation_position="top left",
                                   annotation_font=dict(color=TEXT_MUTED, size=10))
            fig_violin.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("legend", "margin")},
                title=dict(
                    text="Distribuição de Probabilidade por Risco — Random Forest",
                    x=0.01,
                    xanchor="left",
                    y=0.97,
                    yanchor="top",
                    font=dict(size=20, color=TEXT),
                ),
                yaxis_title="Probabilidade de Anomalia", showlegend=True, height=460,
                legend_title_text="",
                legend=dict(
                    orientation="h",
                    yanchor="top", y=1.20,
                    xanchor="left", x=-0.05,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color=TEXT, size=13),
                ),
                margin=dict(l=16, r=16, t=100, b=16),
            )
            st.plotly_chart(fig_violin, use_container_width=True, config={"displayModeBar": False})
            st.caption("Distribuição das probabilidades de anomalia atribuídas pelo Random Forest, por nível de risco. A linha horizontal representa a mediana; a caixa central o intervalo interquartil. A linha tracejada marca o threshold de decisão (0.5).")
            st.markdown('</div>', unsafe_allow_html=True)

        with c12:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig_violin_lr = go.Figure()
            for risco in ORDEM_RISCO:
                sub = sample_filt_m[sample_filt_m["risco_lr"] == risco]
                if len(sub) > 0:
                    fig_violin_lr.add_trace(go.Violin(y=sub["prob_lr"], name=risco, box_visible=True, meanline_visible=True,
                                                        line_color=CORES_RISCO[risco], fillcolor=CORES_RISCO[risco], opacity=0.6))
            fig_violin_lr.add_hline(y=0.5, line_dash="dash", line_color=TEXT_MUTED, line_width=1.5,
                                      annotation_text="Threshold de decisão (0.5)", annotation_position="top left",
                                      annotation_font=dict(color=TEXT_MUTED, size=10))
            fig_violin_lr.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("legend", "margin")},
                title=dict(
                    text="Distribuição de Probabilidade por Risco — Logistic Regression",
                    x=0.01,
                    xanchor="left",
                    y=0.97,
                    yanchor="top",
                    font=dict(size=20, color=TEXT),
                ),
                yaxis_title="Probabilidade de Anomalia", showlegend=True, height=460,
                legend_title_text="",
                legend=dict(
                    orientation="h",
                    yanchor="top", y=1.17,
                    xanchor="left", x=-0.05,
                    bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color=TEXT, size=13),
                ),
                margin=dict(l=16, r=16, t=100, b=16),
            )
            st.plotly_chart(fig_violin_lr, use_container_width=True, config={"displayModeBar": False})
            st.caption("Distribuição das probabilidades de anomalia atribuídas pela Logistic Regression, por nível de risco. A linha horizontal representa a mediana; a caixa central o intervalo interquartil. A linha tracejada marca o threshold de decisão (0.5).")
            st.markdown('</div>', unsafe_allow_html=True)

        def construir_sankey(y_true, y_pred, modelo_nome):
            cm_s = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_pct_s = (cm_s / cm_s.sum(axis=1, keepdims=True) * 100).round(1)
            tn, fp, fn, tp = cm_s[0][0], cm_s[0][1], cm_s[1][0], cm_s[1][1]
            tn_pct, fp_pct, fn_pct, tp_pct = cm_pct_s[0][0], cm_pct_s[0][1], cm_pct_s[1][0], cm_pct_s[1][1]

            labels = ["Real: Normal", "Real: Anómalo", "Previsto: Normal", "Previsto: Anómalo"]
            fig_sankey = go.Figure(go.Sankey(
                node=dict(
                    label=labels, pad=20, thickness=18,
                    color=[BLUE, RED, BLUE, RED],
                    line=dict(color=CARD_BG, width=0),
                ),
                link=dict(
                    source=[0, 0, 1, 1],
                    target=[2, 3, 2, 3],
                    value=[tn_pct, fp_pct, fn_pct, tp_pct],
                    color=[
                        "rgba(91,141,217,0.45)",  # TN — azul
                        "rgba(192,80,77,0.45)",   # FP — vermelho
                        "rgba(192,80,77,0.45)",   # FN — vermelho
                        "rgba(91,141,217,0.45)",  # TP — azul
                    ],
                    customdata=[
                        f"TN: {tn_pct}% ({tn:,})", f"FP: {fp_pct}% ({fp:,})",
                        f"FN: {fn_pct}% ({fn:,})", f"TP: {tp_pct}% ({tp:,})",
                    ],
                    hovertemplate="%{customdata}<extra></extra>",
                ),
                textfont=dict(color=TEXT, size=12),
            ))
            fig_sankey.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "margin")},
                title=dict(text=f"Fluxo de Classificação (Erros Críticos e Acertos) — {modelo_nome}",
                            x=0.01, xanchor="left", y=0.97, yanchor="top", font=dict(size=20, color=TEXT)),
                height=460, margin=dict(l=16, r=16, t=60, b=16),
            )
            return fig_sankey

        c13, c14 = st.columns(2)
        with c13:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig_sankey_rf = construir_sankey(sample_filt_m["label_real"], sample_filt_m["pred_rf"], "Random Forest")
            st.plotly_chart(fig_sankey_rf, use_container_width=True, config={"displayModeBar": False})
            st.caption("Fluxo de classificação do Random Forest: azul = acertos (TN e TP), vermelho = erros (FP = falsos alarmes, FN = ataques não detetados). Espessura proporcional à % por linha.")
            st.markdown('</div>', unsafe_allow_html=True)

        with c14:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig_sankey_lr = construir_sankey(sample_filt_m["label_real"], sample_filt_m["pred_lr"], "Logistic Regression")
            st.plotly_chart(fig_sankey_lr, use_container_width=True, config={"displayModeBar": False})
            st.caption("Fluxo de classificação da Logistic Regression: azul = acertos (TN e TP), vermelho = erros (FP = falsos alarmes, FN = ataques não detetados). Espessura proporcional à % por linha.")
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Explicabilidade do Modelo ────────────────────────────────
        st.markdown("<div class='section-title'>🔍 Fatores Mais Importantes para Deteção</div>", unsafe_allow_html=True)
        st.markdown("<div class='crit-bar green'></div>", unsafe_allow_html=True)

        if df_feature_importance is not None:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            top_features = df_feature_importance.sort_values("importance", ascending=False).head(15)
            top_features = top_features.sort_values("importance", ascending=True)  # maior no topo do gráfico horizontal

            fig_importance = px.bar(
                top_features, x="importance", y="feature", orientation="h",
                color_discrete_sequence=[BLUE],
                labels={"importance": "Importância (Gini)", "feature": "Feature"},
                text="importance",
            )
            fig_importance.update_traces(
                texttemplate="%{text:.3f}", textposition="outside",
                textfont=dict(color=TEXT, size=11), cliponaxis=False,
            )
            fig_importance.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k != "margin"},
                title=dict(text="Fatores Mais Importantes para Deteção (Random Forest)",
                            x=0.01, xanchor="left", y=0.97, yanchor="top", font=dict(size=20, color=TEXT)),
                height=500, margin=dict(l=16, r=16, t=60, b=16),
            )
            fig_importance.update_xaxes(range=[0, top_features["importance"].max() * 1.25])
            st.plotly_chart(fig_importance, use_container_width=True, config={"displayModeBar": False})
            st.caption("Estas variáveis têm maior influência na deteção de anomalias pelo modelo Random Forest. "
                       "Importância calculada pela redução média de impureza (Gini) ao longo das árvores do modelo.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.info(
                "Feature importance ainda não disponível — execute `gerar_dashboard_data.py` "
                "(com o bloco de exportação de feature importance) para gerar `feature_importance.parquet`."
            )
            st.markdown('</div>', unsafe_allow_html=True)

        if df_shap is not None:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)

            ordem_importancia_shap = (
                df_shap.groupby("feature")["shap_value"]
                .apply(lambda x: x.abs().mean())
                .sort_values(ascending=True)  # menor primeiro -> fica em baixo no gráfico horizontal
                .index.tolist()
            )

            shap_plot = df_shap.copy()
            shap_plot["valor_norm"] = shap_plot.groupby("feature")["feature_value"].transform(
                lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else 0.5
            )
            shap_plot["y_base"] = shap_plot["feature"].map({f: i for i, f in enumerate(ordem_importancia_shap)})
            shap_plot["y_jitter"] = shap_plot["y_base"].astype(float)
            for feat in ordem_importancia_shap:
                mask = shap_plot["feature"] == feat
                shap_plot.loc[mask, "y_jitter"] += beeswarm_offsets(shap_plot.loc[mask, "shap_value"].values)

            fig_shap = go.Figure()
            fig_shap.add_trace(go.Scatter(
                x=shap_plot["shap_value"], y=shap_plot["y_jitter"], mode="markers",
                marker=dict(
                    color=shap_plot["valor_norm"], colorscale=[[0, BLUE], [0.5, YELLOW], [1, RED]],
                    size=5, opacity=0.65, line=dict(width=0),
                    colorbar=dict(
                        title=dict(text="Valor da Feature", font=dict(color=TEXT, size=10)),
                        tickvals=[0, 1], ticktext=["Baixo", "Alto"],
                        tickfont=dict(color=TEXT, size=10), thickness=12, len=0.7,
                        outlinewidth=0,
                    ),
                ),
                customdata=shap_plot["feature"],
                hovertemplate="<b>%{customdata}</b><br>Impacto (SHAP): %{x:.4f}<extra></extra>",
            ))
            fig_shap.add_vline(x=0, line_color=TEXT_MUTED, line_width=1, line_dash="dot")
            fig_shap.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("margin", "yaxis")},
                title=dict(text="Impacto Individual das Features na Previsão (SHAP)",
                            x=0.01, xanchor="left", y=0.97, yanchor="top", font=dict(size=20, color=TEXT)),
                xaxis_title="Impacto na previsão (← reduz risco · aumenta risco →)",
                yaxis=dict(
                    tickmode="array", tickvals=list(range(len(ordem_importancia_shap))),
                    ticktext=ordem_importancia_shap, gridcolor=GRID, gridwidth=0.7,
                    zeroline=False, showline=False, tickcolor=TEXT, color=TEXT,
                ),
                height=560, margin=dict(l=16, r=16, t=60, b=16),
            )
            st.plotly_chart(fig_shap, use_container_width=True, config={"displayModeBar": False})
            st.caption(
                "Cada ponto representa um registo da amostra de referência (3.000 registos). A posição horizontal mostra "
                "o impacto dessa feature na previsão para esse registo específico (à direita = empurra para 'anómalo'; "
                "à esquerda = empurra para 'normal'). A cor mostra se o valor da feature, nesse registo, era baixo (azul) "
                "ou alto (vermelho) — permite perceber, por exemplo, se valores altos de uma feature tendem a aumentar o risco previsto."
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.info(
                "Análise SHAP ainda não disponível — execute `gerar_dashboard_data.py` "
                "(com o bloco de exportação SHAP) para gerar `shap_values.parquet`."
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Decision Layer — Insights e Recomendações ──────────────
        st.markdown("<div class='section-title'>📌 Insights e Recomendações</div>", unsafe_allow_html=True)
        st.markdown("<div class='crit-bar green'></div>", unsafe_allow_html=True)

        with st.expander("ℹ️ Como são gerados estes insights?"):
            st.markdown("""
Os insights abaixo são gerados automaticamente por um conjunto de regras fixas, aplicadas aos dados filtrados:
- **Modelo recomendado** → o modelo com F1-score mais alto nos dados filtrados (sem nenhuma margem mínima — qualquer diferença, por pequena que seja, gera uma recomendação).
- **ROC-AUC equivalente** → sinalizado se a diferença entre os ROC-AUC dos dois modelos for **< 0.02**, indicando que a capacidade de separação é praticamente igual.
- **Estabilidade Precision/Recall** → identifica qual modelo tem menor diferença absoluta entre Precision e Recall (maior equilíbrio entre as duas métricas).

Estes insights refletem apenas os dados e dias atualmente filtrados — mudam se ajustares os filtros na sidebar.
            """)

        f1_rf, f1_lr = metricas_modelo["Random Forest"][2], metricas_modelo["Logistic Regression"][2]
        auc_rf, auc_lr = metricas_modelo["Random Forest"][3], metricas_modelo["Logistic Regression"][3]
        prec_rf, rec_rf = metricas_modelo["Random Forest"][0], metricas_modelo["Random Forest"][1]
        prec_lr, rec_lr = metricas_modelo["Logistic Regression"][0], metricas_modelo["Logistic Regression"][1]
        equilibrio_rf = abs(prec_rf - rec_rf)
        equilibrio_lr = abs(prec_lr - rec_lr)

        insights_modelos = []
        if f1_lr > f1_rf:
            modelo_rec, f1_rec, f1_outro = "Logistic Regression", f1_lr, f1_rf
        elif f1_rf > f1_lr:
            modelo_rec, f1_rec, f1_outro = "Random Forest", f1_rf, f1_lr
        else:
            modelo_rec, f1_rec, f1_outro = None, f1_rf, f1_lr

        if modelo_rec:
            insights_modelos.append(
                f"🟢 **{modelo_rec}** recomendado para produção (F1 {f1_rec:.3f} vs {f1_outro:.3f}) → "
                f"melhor equilíbrio entre Precision e Recall nos dados filtrados."
            )
        else:
            insights_modelos.append("🟡 Ambos os modelos apresentam F1 equivalente — considerar critérios adicionais (custo computacional, interpretabilidade).")

        if abs(auc_rf - auc_lr) < 0.02:
            insights_modelos.append(
                f"🟡 ROC-AUC muito próximo entre modelos (RF {auc_rf:.3f} vs LR {auc_lr:.3f}) → "
                f"a capacidade de separação entre classes é semelhante; a decisão pode pesar mais noutros fatores (latência, explicabilidade)."
            )

        if equilibrio_rf < equilibrio_lr:
            insights_modelos.append(f"🟢 Random Forest mostra maior estabilidade entre Precision ({prec_rf:.3f}) e Recall ({rec_rf:.3f}).")
        elif equilibrio_lr < equilibrio_rf:
            insights_modelos.append(f"🟢 Logistic Regression mostra maior estabilidade entre Precision ({prec_lr:.3f}) e Recall ({rec_lr:.3f}).")

        for ins in insights_modelos:
            st.markdown(f"<div class='audience-banner' style='margin:6px 24px'>{ins}</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-grid">
        <div>
            <div class="footer-title">Âmbito do Projeto</div>
            <div class="footer-text">
                Pipeline de deteção de anomalias em tráfego de rede combinando
                pseudo-labelling não supervisionado (Isolation Forest + Local Outlier Factor,
                consenso ponderado) com classificação supervisionada
                (Random Forest + Logistic Regression), avaliado contra ground truth
                real com split temporal 80/20. Dashboard interativo organizado por audiência
                e decisão (Operacional/SOC, Tendências/Gestão, Comparação de Modelos),
                com Decision Layer de insights automáticos e explicabilidade do modelo
                (Feature Importance e SHAP).
            </div>
        </div>
        <div>
            <div class="footer-title">Projecto</div>
            <div class="footer-text">
                Projeto Aplicado em Ciência dos Dados e Analítica de Negócio<br>
                ISCTE — Instituto Universitário de Lisboa<br>
                Prof. Ricardo Ferreira<br>
                Ano letivo 2025/2026
            </div>
        </div>
        <div>
            <div class="footer-title">Recursos</div>
            <div class="footer-text">
                Repositório:<br>
                <a href="https://github.com/RiSantos79/anomaly_classifier" target="_blank">
                    Repositório do Projeto ⇗
                </a><br><br>
                Dataset:<br>
                <a href="https://www.unb.ca/cic/datasets/ids-2017.html" target="_blank">
                    CICIDS2017 (CIC, University of New Brunswick)
                </a>
            </div>
        </div>
    </div>
    <div class="footer-bottom">
        <span>Ricardo Santos · Cibersegurança & Ciência de Dados · Lisboa, Portugal</span>
        <span>Streamlit · pandas · scikit-learn · plotly</span>
    </div>
</div>
""", unsafe_allow_html=True)