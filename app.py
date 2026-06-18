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
    'Baixo'   : GREEN,
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

PLOTLY_LAYOUT = dict(
    paper_bgcolor=CARD_BG,
    plot_bgcolor=CARD_BG,
    font=dict(color=TEXT, family='sans-serif'),
    margin=dict(l=16, r=16, t=40, b=16),
    xaxis=dict(showgrid=False, zeroline=False, showline=False, tickcolor=TEXT, color=TEXT),
    yaxis=dict(gridcolor=GRID, gridwidth=0.7, zeroline=False, showline=False, tickcolor=TEXT, color=TEXT),
    legend=dict(bgcolor='rgba(0,0,0,0)', borderwidth=0, font=dict(color=TEXT)),
)

LEGEND_TOPO = dict(
    orientation="h", yanchor="bottom", y=1.08, xanchor="left", x=0,
    bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(color=TEXT, size=10),
)
MARGIN_TOPO = dict(l=16, r=16, t=90, b=16)

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
HAS_SPLIT = "split" in df.columns
if not HAS_SPLIT:
    df["split"] = "Fora da Amostra"
agg_dia, agg_ataque, agg_porto, df_sample = build_aggregates(df)

# ── Header global (comum às 3 abas) ──────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(90deg,#0d1320,{BG});padding:24px 32px 16px;border-bottom:1px solid {GRID};margin-bottom:0'>
    <div style='font-size:11px;letter-spacing:3px;color:{BLUE};text-transform:uppercase;margin-bottom:6px'>🛡️ ISCTE — Data Science & Business Analytics · 2026</div>
    <div style='font-size:28px;font-weight:900;color:{TEXT};line-height:1.1'>Deteção de Anomalias em Tráfego de Rede</div>
    <div style='font-size:13px;color:{TEXT_MUTED};margin-top:6px'>Pipeline: Isolation Forest + LOF → Random Forest / Logistic Regression · Dataset: CICIDS2017</div>
</div>
""", unsafe_allow_html=True)

tab_soc, tab_tendencias, tab_modelos = st.tabs([
    "🚨 Operacional / SOC", "📈 Tendências / Gestão", "⚖️ Comparação de Modelos",
])

# ──────────────────────────────────────────────────────────────────
# ABA 1 — OPERACIONAL / SOC
# ──────────────────────────────────────────────────────────────────
with tab_soc:
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
        prec = precision_score(sample_filt["label_real"], sample_filt[col_pred], zero_division=0)
        rec  = recall_score(sample_filt["label_real"], sample_filt[col_pred], zero_division=0)
        f1   = f1_score(sample_filt["label_real"], sample_filt[col_pred], zero_division=0)
        risco_counts = a_atq_filt.groupby(col_risco, observed=True)["n"].sum()
        pct_crit = round(risco_counts.get("Crítico", 0) / n_total * 100, 1) if n_total > 0 else 0

        st.markdown(
            f"<div class='audience-banner'><b>Audiência:</b> Analista SOC &nbsp;·&nbsp; "
            f"<b>Decisão:</b> o que investigar agora &nbsp;·&nbsp; <b>Janela:</b> imediata/diária</div>",
            unsafe_allow_html=True,
        )

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        for col_ui, label, value, cls in [
            (k1, "Registos",           f"{n_total:,}",          "blue"),
            (k2, "% Anómalo Real",     f"{pct_r:.1f}%",         "red"),
            (k3, "% Anómalo Previsto", f"{pct_p:.1f}%",         "orange"),
            (k4, "Precision",          f"{prec:.3f}",            "green"),
            (k5, "Recall / F1",        f"{rec:.2f} / {f1:.2f}", "blue"),
            (k6, "Fluxos Críticos",    f"{pct_crit:.1f}%",      "red"),
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
                fig_portos.update_layout(**PLOTLY_LAYOUT, title=f"Top {top_n} Portos — Anomalias Previstas", height=360)
                fig_portos.update_yaxes(type="category")
                st.plotly_chart(fig_portos, use_container_width=True, config={"displayModeBar": False})
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
                colorscale=[[0, RED], [1, GREEN]], showscale=False, zmin=0, zmax=1,
                text=anotacoes, texttemplate="%{text}", textfont=dict(color=TEXT, size=13),
            ))
            fig_cm.update_layout(**PLOTLY_LAYOUT, title=f"Matriz de Confusão — {modelo_label} (amostra 50k)",
                                   xaxis_title="Previsto", yaxis_title="Real", height=360)
            st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
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
            fig_tipos.update_layout(**PLOTLY_LAYOUT, title="Distribuição Real por Tipo de Tráfego", barmode="stack", height=380)
            st.plotly_chart(fig_tipos, use_container_width=True, config={"displayModeBar": False})
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
            recall_agg = recall_agg.sort_values("taxa_deteccao")

            if len(recall_agg) > 0:
                fig_recall = px.bar(
                    recall_agg, x="taxa_deteccao", y="attack_type", orientation="h",
                    color="taxa_deteccao",
                    color_continuous_scale=[[0, GREEN], [0.5, ORANGE], [1, RED]],
                    labels={"taxa_deteccao": "Taxa de Detecção (%)", "attack_type": "Tipo de Ataque"},
                    text="taxa_deteccao",
                )
                fig_recall.update_traces(texttemplate="%{text:.1f}%", textposition="outside", textfont_color=TEXT)
                fig_recall.update_layout(**PLOTLY_LAYOUT, title="Eficácia de Detecção por Tipo de Ataque (%)",
                                           coloraxis_showscale=False, height=380)
                fig_recall.update_xaxes(range=[0, 115], showgrid=False, zeroline=False, showline=False, tickcolor=TEXT, color=TEXT)
                st.plotly_chart(fig_recall, use_container_width=True, config={"displayModeBar": False})
            else:
                st.info("Sem anomalias reais no subset filtrado para calcular taxa de detecção.")
            st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# ABA 2 — TENDÊNCIAS / GESTÃO
# ──────────────────────────────────────────────────────────────────
with tab_tendencias:
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

        st.markdown(
            f"<div class='audience-banner'><b>Audiência:</b> Gestor SOC / Operações &nbsp;·&nbsp; "
            f"<b>Decisão:</b> escalar recursos ou não &nbsp;·&nbsp; <b>Janela:</b> diária/semanal</div>",
            unsafe_allow_html=True,
        )

        k1, k2, k3, k4 = st.columns(4)
        for col_ui, label, value, cls in [
            (k1, "Registos",            f"{n_total_t:,}",      "blue"),
            (k2, "% Anómalo Real (médio)", f"{pct_real_geral:.1f}%", "orange"),
            (k3, "Dia de Pico",          f"{dia_pico} ({pct_pico:.1f}%)", "red"),
            (k4, "Fluxos Críticos",      f"{pct_crit_t:.1f}%",  "red"),
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

            layout_linha = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("legend", "margin")}
            fig_linha.update_layout(**layout_linha, title="% Anomalias Detectadas por Dia — Tendência e Projeção",
                                      yaxis_title="% Anomalias", height=400, legend=LEGEND_TOPO, margin=MARGIN_TOPO)
            st.plotly_chart(fig_linha, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            rd = risco_dia_filt_t.copy()
            rd[col_risco_t] = pd.Categorical(rd[col_risco_t], categories=ORDEM_RISCO, ordered=True)
            fig_risco_dia = go.Figure()
            for risco in ORDEM_RISCO:
                sub = rd[rd[col_risco_t] == risco]
                fig_risco_dia.add_trace(go.Bar(x=sub["day"], y=sub["pct"], name=risco, marker_color=CORES_RISCO[risco]))
            layout_risco_dia = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("legend", "margin")}
            fig_risco_dia.update_layout(**layout_risco_dia, title="Tendência de Risco por Dia (%)", barmode="stack",
                                          yaxis_title="% de Fluxos", height=400, legend=LEGEND_TOPO, margin=MARGIN_TOPO)
            st.plotly_chart(fig_risco_dia, use_container_width=True, config={"displayModeBar": False})
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
            fig_box.update_layout(**PLOTLY_LAYOUT, title="Distribuição de Probabilidade por Tipo de Tráfego (cor = risco predominante)",
                                    showlegend=False, height=360, xaxis_tickangle=-30)
            fig_box.update_traces(marker_size=3)
            st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})
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
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
                title=f"Distribuição de Risco — {modelo_label_t}",
                height=360,
            )
            st.plotly_chart(fig_gauges, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# ABA 3 — COMPARAÇÃO DE MODELOS
# ──────────────────────────────────────────────────────────────────
with tab_modelos:
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
            (k2, "F1 — Random Forest", f"{metricas_modelo['Random Forest'][2]:.3f}", "green"),
            (k3, "F1 — Logistic Regr.", f"{metricas_modelo['Logistic Regression'][2]:.3f}", "green"),
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
            fig_dumb.update_layout(**layout_dumb, title="Comparação de Métricas — RF vs LR (amostra 50k)",
                                     xaxis=dict(showgrid=True, gridcolor=GRID, gridwidth=0.7, zeroline=False,
                                                 showline=False, tickcolor=TEXT, color=TEXT, range=[0, 1.1]),
                                     height=400, legend=LEGEND_TOPO, margin=MARGIN_TOPO)
            st.plotly_chart(fig_dumb, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with c10:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig_roc = go.Figure()
            for m_nome, cor in [("Random Forest", BLUE), ("Logistic Regression", TEAL)]:
                fpr, tpr, auc = roc_data[m_nome]
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{m_nome} (AUC={auc:.3f})",
                                               line=dict(color=cor, width=2.5)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatório",
                                           line=dict(color=TEXT_MUTED, width=1.5, dash="dot")))
            layout_roc = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("legend", "margin")}
            fig_roc.update_layout(**layout_roc, title="Curva ROC — RF vs LR (amostra 50k)",
                                    xaxis_title="Falsos Positivos (FPR)", yaxis_title="Verdadeiros Positivos (TPR)",
                                    height=400, legend=LEGEND_TOPO, margin=MARGIN_TOPO)
            st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})
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
            fig_violin.update_layout(**PLOTLY_LAYOUT, title="Distribuição de Probabilidade por Risco — Random Forest",
                                       yaxis_title="Probabilidade de Anomalia", showlegend=False, height=380)
            st.plotly_chart(fig_violin, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with c12:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig_violin_lr = go.Figure()
            for risco in ORDEM_RISCO:
                sub = sample_filt_m[sample_filt_m["risco_lr"] == risco]
                if len(sub) > 0:
                    fig_violin_lr.add_trace(go.Violin(y=sub["prob_lr"], name=risco, box_visible=True, meanline_visible=True,
                                                        line_color=CORES_RISCO[risco], fillcolor=CORES_RISCO[risco], opacity=0.6))
            fig_violin_lr.update_layout(**PLOTLY_LAYOUT, title="Distribuição de Probabilidade por Risco — Logistic Regression",
                                          yaxis_title="Probabilidade de Anomalia", showlegend=False, height=380)
            st.plotly_chart(fig_violin_lr, use_container_width=True, config={"displayModeBar": False})
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
                        "rgba(46,168,79,0.45)",   # TN — verde
                        "rgba(192,80,77,0.45)",   # FP — vermelho
                        "rgba(192,80,77,0.45)",   # FN — vermelho
                        "rgba(46,168,79,0.45)",   # TP — verde
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
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis")},
                title=f"Fluxo Real → Previsto (% por linha) — {modelo_nome}",
                height=340,
            )
            return fig_sankey

        c13, c14 = st.columns(2)
        with c13:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig_sankey_rf = construir_sankey(sample_filt_m["label_real"], sample_filt_m["pred_rf"], "Random Forest")
            st.plotly_chart(fig_sankey_rf, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with c14:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig_sankey_lr = construir_sankey(sample_filt_m["label_real"], sample_filt_m["pred_lr"], "Logistic Regression")
            st.plotly_chart(fig_sankey_lr, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-grid">
        <div>
            <div class="footer-title">
                <a href="https://github.com/RiSantos79/anomaly_classifier" target="_blank" style="color: inherit; text-decoration: none;">
                    Repositório do Projeto ⇗
                </a>
            </div>
            <div class="footer-text">
                Pipeline de deteção de anomalias em tráfego de rede combinando
                pseudo-labelling não supervisionado (Isolation Forest + Local Outlier Factor,
                consenso ponderado) com classificação supervisionada
                (Random Forest + Logistic Regression), avaliado contra ground truth
                real com split temporal 80/20.
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