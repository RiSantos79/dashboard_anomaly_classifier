import streamlit as st
import pandas as pd
#import base64
#from pathlib import Path

# --------------------------------------------------
# Configuração da página
# --------------------------------------------------
st.set_page_config(
    page_title="Deteção de Anomalias em Tráfego de Rede",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CSS global + barra de scan animada
# --------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0d1117;
    color: #d0d8e8;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.scan-bar {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, transparent, #5b8dd9, #c0504d, transparent);
    background-size: 200% 100%;
    animation: scan 3s linear infinite;
    z-index: 9999;
}
@keyframes scan {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}
</style>

<div class="scan-bar"></div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HERO — narrativa principal e número central
# --------------------------------------------------
st.markdown("""
<style>
.hero {
    background: linear-gradient(180deg, #0a0f1a 0%, #1a2233 100%);
    padding: 60px 80px 50px;
    border-bottom: 1px solid #2e3f5c;
    position: relative;
    overflow: hidden;
}
.hero-eyebrow {
    font-size: 11px; font-weight: 600; letter-spacing: 3px;
    color: #5b8dd9; text-transform: uppercase; margin-bottom: 16px;
}
.hero-title {
    font-size: 52px; font-weight: 900;
    color: #f0f4ff; line-height: 1.1; margin-bottom: 12px;
}
.hero-subtitle {
    font-size: 14px; color: #6b7a8d;
    margin-bottom: 40px; line-height: 1.6;
}
.hero-number {
    font-size: 96px; font-weight: 900;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #5b8dd9, #7aaae8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1; margin-bottom: 8px;
}
.hero-number-label {
    font-size: 13px; color: #6b7a8d;
    letter-spacing: 1px; text-transform: uppercase;
}
</style>

<div class="hero">
    <div class="hero-eyebrow">🛡️ ISCTE — Data Science & Business Analytics · Jun/2026</div>
    <div class="hero-title">Deteção de Anomalias<br>em Tráfego de Rede</div>
    <div class="hero-subtitle">
        Pipeline ML Supervisionado com Pseudo-Labelling · IF+LOF · Random Forest · Logistic Regression · CICIDS2017
    </div>
    <div style="display:flex; gap:64px; align-items:flex-end; flex-wrap:wrap;">
        <div>
            <div class="hero-number">1.582.029</div>
            <div class="hero-number-label">fluxos de rede analisados</div>
        </div>
        <div style="padding-bottom:12px; max-width:420px;">
            <div style="font-size:15px; color:#8899aa; line-height:1.8;">
                Imagina que és um analista SOC. Todas as manhãs tens 
                <strong style="color:#d0d8e8;">1,5 milhões de eventos</strong> 
                para analisar — impossível fazer manualmente.<br><br>
                Este sistema reduz esse volume em 
                <strong style="color:#2ea84f;">90,7%</strong>. 
                O analista foca-se apenas no que importa.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# KPI STRIP — métricas chave do projecto
# --------------------------------------------------
st.markdown("""
<style>
.kpi-strip {
    background: #111827;
    padding: 20px 80px;
    display: flex; gap: 0;
    border-bottom: 1px solid #1e2d45;
}
.kpi-item {
    flex: 1; padding: 0 32px;
    border-right: 1px solid #1e2d45;
}
.kpi-item:first-child { padding-left: 0; }
.kpi-item:last-child  { border-right: none; }
.kpi-value {
    font-size: 28px; font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1; margin-bottom: 4px;
}
.kpi-label {
    font-size: 11px; color: #6b7a8d;
    text-transform: uppercase; letter-spacing: 1px;
}
.blue   { color: #5b8dd9; }
.red    { color: #c0504d; }
.green  { color: #2ea84f; }
.orange { color: #f0a500; }
.white  { color: #f0f4ff; }
</style>

<div class="kpi-strip">
    <div class="kpi-item">
        <div class="kpi-value blue">0,8951</div>
        <div class="kpi-label">ROC-AUC Random Forest</div>
    </div>
    <div class="kpi-item">
        <div class="kpi-value green">90,7%</div>
        <div class="kpi-label">Redução volume investigação</div>
    </div>
    <div class="kpi-item">
        <div class="kpi-value red">16,17%</div>
        <div class="kpi-label">Taxa de anomalias real</div>
    </div>
    <div class="kpi-item">
        <div class="kpi-value orange">85,46%</div>
        <div class="kpi-label">Concordância pseudo-labels</div>
    </div>
    <div class="kpi-item">
        <div class="kpi-value white">3</div>
        <div class="kpi-label">Iterações do projecto</div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CAPÍTULO 1 — O Problema
# --------------------------------------------------
st.markdown("""
<style>
.section {
    padding: 56px 80px;
    border-bottom: 1px solid #1e2d45;
}
.section-alt { background: #0f1623; }
.chapter-label {
    font-size: 10px; font-weight: 700; letter-spacing: 3px;
    color: #3d5278; text-transform: uppercase; margin-bottom: 8px;
}
.section-title {
    font-size: 32px; font-weight: 800;
    color: #f0f4ff; margin-bottom: 8px; line-height: 1.2;
}
.section-body {
    font-size: 14px; color: #8899aa; line-height: 1.7;
    max-width: 680px; margin-bottom: 36px;
}
.card-grid { display: grid; gap: 16px; }
.card-grid-3 { grid-template-columns: 1fr 1fr 1fr; }
.card {
    background: #1a2233;
    border: 1px solid #2e3f5c;
    border-radius: 8px;
    padding: 24px;
}
.card-accent-blue   { border-left: 3px solid #5b8dd9; }
.card-accent-red    { border-left: 3px solid #c0504d; }
.card-accent-orange { border-left: 3px solid #f0a500; }
.card-accent-green  { border-left: 3px solid #2ea84f; }
.card-title {
    font-size: 11px; font-weight: 600; color: #6b7a8d;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;
}
.card-value {
    font-size: 26px; font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
}
.card-desc { font-size: 12px; color: #6b7a8d; margin-top: 6px; line-height: 1.5; }
.insight {
    background: #111827; border-left: 3px solid #5b8dd9;
    border-radius: 0 6px 6px 0; padding: 14px 18px;
    margin: 12px 0; font-size: 13px; color: #8899aa; line-height: 1.6;
}
.insight strong { color: #d0d8e8; }
</style>

<div class="section">
    <div class="chapter-label">Capítulo 1</div>
    <div class="section-title">O Problema</div>
    <div class="section-body">
        Os ambientes SOC enfrentam três desafios estruturais que nenhuma regra manual resolve sozinha.
        O volume é avassalador, os ataques novos escapam às regras existentes, e em produção real
        não existem labels — não sabemos à partida o que é ataque.
    </div>
    <div class="card-grid card-grid-3">
        <div class="card card-accent-red">
            <div class="card-title">Volume massivo</div>
            <div class="card-value red">1,5M</div>
            <div class="card-desc">eventos diários impossíveis de analisar manualmente por um analista humano</div>
        </div>
        <div class="card card-accent-orange">
            <div class="card-title">Regras SIEM limitadas</div>
            <div class="card-value orange">0%</div>
            <div class="card-desc">de detecção para ataques novos ou variantes não catalogadas nas regras existentes</div>
        </div>
        <div class="card card-accent-blue">
            <div class="card-title">Ausência de labels</div>
            <div class="card-value blue">SOC Real</div>
            <div class="card-desc">não existem ground truth em produção — não se sabe à partida quais os eventos são ataques</div>
        </div>
    </div>
    <div class="insight">
        <strong>A solução:</strong> um pipeline de ML que gera labels automaticamente (pseudo-labelling)
        e treina um classificador supervisionado — sem intervenção humana — reduzindo o volume
        a investigar em <strong>90,7%</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CAPÍTULO 2 — O Dataset
# --------------------------------------------------
st.markdown("""
<style>
.card-grid-4 { grid-template-columns: repeat(4, 1fr); }
.chart-wrap {
    background: #111827; border: 1px solid #1e2d45;
    border-radius: 8px; overflow: hidden; margin: 24px 0;
}
.chart-caption {
    font-size: 11px; color: #6b7a8d; text-align: center;
    padding: 10px 16px; border-top: 1px solid #1e2d45;
    font-style: italic;
}
</style>

<div class="section section-alt">
    <div class="chapter-label">Capítulo 2</div>
    <div class="section-title">O Dataset — CICIDS2017</div>
    <div class="section-body">
        Dataset público do Canadian Institute for Cybersecurity com tráfego real de rede
        capturado durante 4 dias, incluindo ataques reais lançados em ambiente controlado
        — com ground truth real.
    </div>
    <div class="card-grid card-grid-4" style="margin-bottom:24px;">
        <div class="card">
            <div class="card-title">Registos após limpeza</div>
            <div class="card-value blue">1.582.029</div>
        </div>
        <div class="card">
            <div class="card-title">Features seleccionadas</div>
            <div class="card-value white">70</div>
        </div>
        <div class="card">
            <div class="card-title">Taxa de anomalias</div>
            <div class="card-value red">16,17%</div>
        </div>
        <div class="card">
            <div class="card-title">Dias de captura</div>
            <div class="card-value orange">4 dias</div>
        </div>
    </div>
    <div class="insight">
        <strong>Wednesday</strong> é o dia mais crítico — 36,4% de anomalias (DoS/DDoS massivo).
        <strong>Thursday e Friday</strong> têm menos de 1,3% — Web Attacks e Botnet são mais subtis:
        cirúrgicos, de baixo volume, difíceis de detectar por regras de volume.
    </div>
</div>
""", unsafe_allow_html=True)

import base64
from pathlib import Path
def load_img(filename):
    path = Path(__file__).parent.parent / filename
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return None

col1, col2 = st.columns(2)

fig01 = load_img("fig01.png")
fig08 = load_img("fig08.png")

with col1:
    if fig01:
        st.markdown(f"""
        <div style="padding:0 8px 32px 80px;">
        <div class="chart-wrap">
            <img src="data:image/png;base64,{fig01}" style="width:100%;display:block;">
            <div class="chart-caption">Figura 1 — Distribuição de classes: 83,83% Normal vs 16,17% Anómalo</div>
        </div></div>""", unsafe_allow_html=True)

with col2:
    if fig08:
        st.markdown(f"""
        <div style="padding:0 80px 32px 8px;">
        <div class="chart-wrap">
            <img src="data:image/png;base64,{fig08}" style="width:100%;display:block;">
            <div class="chart-caption">Figura 8 — Wednesday concentra 36,4% de anomalias (DoS/DDoS massivo)</div>
        </div></div>""", unsafe_allow_html=True)

# --------------------------------------------------
# CAPÍTULO 3 — Como Detectámos (pipeline + consenso)
# --------------------------------------------------
st.markdown("""
<style>
.pipeline {
    display: flex; gap: 0; align-items: stretch; margin: 32px 0;
}
.pipeline-step {
    flex: 1; background: #1a2233;
    border: 1px solid #2e3f5c; border-right: none;
    padding: 20px 16px; position: relative;
}
.pipeline-step:first-child { border-radius: 8px 0 0 8px; }
.pipeline-step:last-child  { border-right: 1px solid #2e3f5c; border-radius: 0 8px 8px 0; }
.pipeline-step::after {
    content: '→'; position: absolute; right: -12px; top: 50%;
    transform: translateY(-50%); color: #3d5278; font-size: 16px; z-index: 1;
}
.pipeline-step:last-child::after { display: none; }
.step-num {
    font-size: 10px; font-weight: 700; color: #3d5278;
    text-transform: uppercase; letter-spacing: 2px; margin-bottom: 6px;
}
.step-title { font-size: 13px; font-weight: 700; color: #d0d8e8; margin-bottom: 4px; }
.step-desc  { font-size: 11px; color: #6b7a8d; line-height: 1.5; }
.step-highlight { color: #5b8dd9; font-weight: 600; }
.consensus-row { display: flex; gap: 16px; margin: 24px 0; align-items: center; }
.consensus-item {
    flex: 1; background: #1a2233; border: 1px solid #2e3f5c;
    border-radius: 8px; padding: 20px; text-align: center;
}
.consensus-strategy { font-size: 13px; font-weight: 700; margin-bottom: 8px; }
.consensus-metric   { font-size: 22px; font-weight: 800;
                      font-family: 'JetBrains Mono', monospace; }
.consensus-desc     { font-size: 11px; color: #6b7a8d; margin-top: 6px; }
.consensus-arrow    { font-size: 24px; color: #3d5278; }
</style>

<div class="section">
    <div class="chapter-label">Capítulo 3</div>
    <div class="section-title">Como Detectámos</div>
    <div class="section-body">
        Um pipeline de 5 fases com separação explícita entre pseudo-labelling e classificação
        supervisionada — o elemento metodológico mais crítico para evitar leakage.
    </div>
    <div class="pipeline">
        <div class="pipeline-step">
            <div class="step-num">Fase 01</div>
            <div class="step-title">Limpeza</div>
            <div class="step-desc">4 CSVs → <span class="step-highlight">1.582.029</span> registos<br>Infinity→NaN, 10 cols removidas</div>
        </div>
        <div class="pipeline-step">
            <div class="step-num">Fase 02</div>
            <div class="step-title">Separação de Features</div>
            <div class="step-desc"><span class="step-highlight">17</span> volumétricas → PL<br><span class="step-highlight">50</span> sessão TCP → Modelo<br>Sem sobreposição</div>
        </div>
        <div class="pipeline-step" style="border-left:3px solid #f0a500;">
            <div class="step-num">Fase 03</div>
            <div class="step-title">Pseudo-Labelling</div>
            <div class="step-desc">IF + LOF · 200k registos<br>Score combinado · Percentil 80<br><span class="step-highlight">F1=0,60</span> · Conc. 85,46%</div>
        </div>
        <div class="pipeline-step">
            <div class="step-num">Fase 04</div>
            <div class="step-title">Split Temporal</div>
            <div class="step-desc">80% treino / 20% teste<br>Ordenação cronológica<br>Simula produção real</div>
        </div>
        <div class="pipeline-step">
            <div class="step-num">Fase 05</div>
            <div class="step-title">Modelos</div>
            <div class="step-desc">Random Forest + LR<br>class_weight=balanced<br><span class="step-highlight">ROC-AUC 0,8951</span></div>
        </div>
    </div>
    <div class="insight">
        <strong>Nota sobre iterações:</strong> O projecto passou por 3 iterações —
        BOTSv1+DBSCAN (leakage), CICIDS2017+DBSCAN (MemoryError), CICIDS2017+LOF (versão final).
        O LOF foi seleccionado por ser mais eficiente em memória mantendo a lógica de detecção
        por densidade local.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Evolução do consenso + gráfico pseudo-labelling
# --------------------------------------------------
fig09 = load_img("fig09.png")

st.markdown("""
<div class="section section-alt">
    <div class="chapter-label">Capítulo 3 · Detalhe</div>
    <div class="section-title">Evolução do Consenso de Pseudo-Labelling</div>
    <div class="section-body">
        A estratégia de consenso evoluiu em três passos — cada um com um trade-off
        diferente entre Precision e Recall.
    </div>
    <div class="consensus-row">
        <div class="consensus-item" style="border-color:#c0504d;">
            <div class="consensus-strategy red">AND (intersecção)</div>
            <div class="consensus-metric red">Recall 9%</div>
            <div class="consensus-desc">Demasiado conservador — detecta muito poucos ataques</div>
        </div>
        <div class="consensus-arrow">→</div>
        <div class="consensus-item" style="border-color:#f0a500;">
            <div class="consensus-strategy orange">OR (união)</div>
            <div class="consensus-metric orange">Recall 71%</div>
            <div class="consensus-desc">Precision de 32% — demasiados falsos alarmes</div>
        </div>
        <div class="consensus-arrow">→</div>
        <div class="consensus-item" style="border-color:#2ea84f;">
            <div class="consensus-strategy green">Score Combinado ✓</div>
            <div class="consensus-metric green">F1 = 0,60</div>
            <div class="consensus-desc">Precision 54% · Recall 67% · Equilíbrio óptimo</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if fig09:
    st.markdown(f"""
    <div style="padding:0 80px 32px;">
    <div class="chart-wrap">
        <img src="data:image/png;base64,{fig09}" style="width:100%;display:block;">
        <div class="chart-caption">Figura 9 — Pseudo-Labelling: 160.000 Normal (80%) e 40.000 Anómalo (20%) · F1=0,60 · Concordância 85,46%</div>
    </div></div>""", unsafe_allow_html=True)

# --------------------------------------------------
# CAPÍTULO 4 — O Que Descobrimos (resultados)
# --------------------------------------------------
st.markdown("""
<style>
.card-grid-2 { grid-template-columns: 1fr 1fr; }
.card-grid-4 { grid-template-columns: repeat(4, 1fr); }
</style>

<div class="section">
    <div class="chapter-label">Capítulo 4</div>
    <div class="section-title">O Que Descobrimos</div>
    <div class="section-body">
        Dois modelos complementares avaliados contra o ground truth real de 1.582.029 registos —
        treinados exclusivamente com pseudo-labels geradas automaticamente.
    </div>
    <div class="card-grid card-grid-4" style="margin-bottom:24px;">
        <div class="card card-accent-blue">
            <div class="card-title">ROC-AUC · Random Forest</div>
            <div class="card-value blue">0,8951</div>
            <div class="card-desc">vs classificador aleatório: 0,5000</div>
        </div>
        <div class="card card-accent-blue">
            <div class="card-title">ROC-AUC · Logistic Reg.</div>
            <div class="card-value blue">0,8751</div>
            <div class="card-desc">Supera RF em F1-score (0,602 vs 0,593)</div>
        </div>
        <div class="card card-accent-green">
            <div class="card-title">Recall · Classe Anómalo</div>
            <div class="card-value green">67%</div>
            <div class="card-desc">2 em cada 3 ataques reais detectados</div>
        </div>
        <div class="card card-accent-orange">
            <div class="card-title">Precision · Classe Anómalo</div>
            <div class="card-value orange">55%</div>
            <div class="card-desc">Mais de metade dos alertas são reais</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Curva ROC + Feature Importance
# --------------------------------------------------
fig12 = load_img("fig12.png")
fig13 = load_img("fig13.png")

col1, col2 = st.columns(2)

with col1:
    if fig12:
        st.markdown(f"""
        <div style="padding:0 8px 0 80px;">
        <div class="chart-wrap">
            <img src="data:image/png;base64,{fig12}" style="width:100%;display:block;">
            <div class="chart-caption">Figura 12 — Curva ROC: RF AUC=0,8951 · LR AUC=0,8751 · muito acima do aleatório (0,50)</div>
        </div></div>""", unsafe_allow_html=True)

with col2:
    if fig13:
        st.markdown(f"""
        <div style="padding:0 80px 0 8px;">
        <div class="chart-wrap">
            <img src="data:image/png;base64,{fig13}" style="width:100%;display:block;">
            <div class="chart-caption">Figura 13 — Top 20 features: Subflow Bwd Bytes (0,1066) e Fwd IAT Total (0,1032) dominam</div>
        </div></div>""", unsafe_allow_html=True)

# --------------------------------------------------
# Matrizes de confusão
# --------------------------------------------------
fig11 = load_img("fig11.png")

if fig11:
    st.markdown(f"""
    <div style="padding:24px 80px 0;">
    <div class="chart-wrap">
        <img src="data:image/png;base64,{fig11}" style="width:100%;display:block;">
        <div class="chart-caption">Figura 11 — Matrizes de Confusão: FN (Grave) é o indicador mais crítico para SOC</div>
    </div></div>""", unsafe_allow_html=True)

# --------------------------------------------------
# Dot plot comparação de modelos
# --------------------------------------------------
fig14 = load_img("fig14.png")

if fig14:
    st.markdown(f"""
    <div style="padding:24px 80px 32px;">
    <div class="chart-wrap">
        <img src="data:image/png;base64,{fig14}" style="width:100%;display:block;">
        <div class="chart-caption">Figura 14 — Comparação de modelos: RF lidera em ROC-AUC · LR lidera em F1-score e Recall</div>
    </div></div>""", unsafe_allow_html=True)

st.markdown("""
<div style="padding:0 80px 32px;">
    <div class="insight">
        <strong>Train vs Test:</strong> RF F1=0,945 no treino vs F1=0,593 no teste. Não é overfitting —
        treino e teste são avaliados contra referências diferentes (pseudo-labels vs ground truth real).
        A diferença é proporcional à qualidade das pseudo-labels (F1=0,60) e representa o
        <strong>limite teórico atingível sem labels reais</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CAPÍTULO 5 — Deployment (níveis de risco + projecção)
# --------------------------------------------------
st.markdown("""
<div class="section">
    <div class="chapter-label">Capítulo 5</div>
    <div class="section-title">Deployment</div>
    <div class="section-body">
        O modelo Random Forest foi aplicado a cenários de monitorização contínua e
        a tráfego "futuro" (últimos 20% do dataset, ordenado cronologicamente),
        simulando a operação real num SOC.
    </div>
    <div class="card-grid card-grid-4" style="margin-bottom:24px;">
        <div class="card card-accent-red">
            <div class="card-title">Porto Mais Atacado</div>
            <div class="card-value red">80 (HTTP)</div>
            <div class="card-desc">182.964 anomalias — DoS e Web Attacks</div>
        </div>
        <div class="card card-accent-red">
            <div class="card-title">Pico de Anomalias</div>
            <div class="card-value red">Wednesday · 30,9%</div>
            <div class="card-desc">DoS/DDoS massivo correctamente detectado</div>
        </div>
        <div class="card card-accent-orange">
            <div class="card-title">Fluxos Críticos</div>
            <div class="card-value orange">9,3%</div>
            <div class="card-desc">29.496 fluxos com p ≥ 0,80 — alerta imediato</div>
        </div>
        <div class="card card-accent-blue">
            <div class="card-title">Projecção Dia 5</div>
            <div class="card-value blue">10,9% ± 3,0%</div>
            <div class="card-desc">Tendência estabilizada após o pico</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Top Portos + Previsão por dia
# --------------------------------------------------
fig16 = load_img("fig16.png")
fig17 = load_img("fig17.png")

col1, col2 = st.columns(2)

with col1:
    if fig16:
        st.markdown(f"""
        <div style="padding:0 8px 32px 80px;">
        <div class="chart-wrap">
            <img src="data:image/png;base64,{fig16}" style="width:100%;display:block;">
            <div class="chart-caption">Figura 16 — Top 10 portos de destino: 80 (HTTP) e 443 (HTTPS) dominam as anomalias</div>
        </div></div>""", unsafe_allow_html=True)

with col2:
    if fig17:
        st.markdown(f"""
        <div style="padding:0 80px 32px 8px;">
        <div class="chart-wrap">
            <img src="data:image/png;base64,{fig17}" style="width:100%;display:block;">
            <div class="chart-caption">Figura 17 — Monitorização contínua: Wednesday concentra 213.977 anomalias (30,9%)</div>
        </div></div>""", unsafe_allow_html=True)

# --------------------------------------------------
# Distribuição de risco + Tendência/Projecção
# --------------------------------------------------
fig18 = load_img("fig18.png")
fig19 = load_img("fig19.png")

col1, col2 = st.columns(2)

with col1:
    if fig18:
        st.markdown(f"""
        <div style="padding:0 8px 32px 80px;">
        <div class="chart-wrap">
            <img src="data:image/png;base64,{fig18}" style="width:100%;display:block;">
            <div class="chart-caption">Figura 18 — Distribuição de risco em tráfego futuro: 84,9% Baixo · 9,3% Crítico</div>
        </div></div>""", unsafe_allow_html=True)

with col2:
    if fig19:
        st.markdown(f"""
        <div style="padding:0 80px 32px 8px;">
        <div class="chart-wrap">
            <img src="data:image/png;base64,{fig19}" style="width:100%;display:block;">
            <div class="chart-caption">Figura 19 — Tendência de risco e projecção: estabilização em ~10,9% para o Dia 5</div>
        </div></div>""", unsafe_allow_html=True)

st.markdown("""
<div style="padding:0 80px 32px;">
    <div class="insight">
        <strong>Triagem por nível de risco:</strong> a separação clara entre fluxos de baixo risco (84,9%)
        e críticos (9,3%) permite priorizar a atenção dos analistas L1 nos casos com maior
        probabilidade de anomalia, em vez de uma triagem indiferenciada de todo o tráfego —
        <strong>redução directa da carga operacional do SOC</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CAPÍTULO 6 — EDA (bónus)
# --------------------------------------------------
st.markdown("""
<div class="section">
    <div class="chapter-label">Capítulo 6</div>
    <div class="section-title">EDA — Análise Exploratória (Bónus)</div>
    <div class="section-body">
        Antes da modelação, a análise exploratória das 17 features de pseudo-labelling
        já revelava diferenças claras entre tráfego Normal e Anómalo — a base que tornou
        possível todo o pipeline de detecção.
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# CDF por classe
# --------------------------------------------------
fig07 = load_img("fig07.png")

if fig07:
    st.markdown(f"""
    <div style="padding:0 80px 32px;">
    <div class="chart-wrap">
        <img src="data:image/png;base64,{fig07}" style="width:100%;display:block;">
        <div class="chart-caption">Figura 7 — Distribuição Acumulada (CDF): curvas afastadas indicam features discriminativas</div>
    </div></div>""", unsafe_allow_html=True)

# --------------------------------------------------
# Boxplots + Violin Plot
# --------------------------------------------------
fig03 = load_img("fig03.png")
fig04 = load_img("fig04.png")

col1, col2 = st.columns(2)

with col1:
    if fig03:
        st.markdown(f"""
        <div style="padding:0 8px 32px 80px;">
        <div class="chart-wrap">
            <img src="data:image/png;base64,{fig03}" style="width:100%;display:block;">
            <div class="chart-caption">Figura 3 — Boxplots por classe: outliers e dispersão evidenciam padrões de anomalia</div>
        </div></div>""", unsafe_allow_html=True)

with col2:
    if fig04:
        st.markdown(f"""
        <div style="padding:0 80px 32px 8px;">
        <div class="chart-wrap">
            <img src="data:image/png;base64,{fig04}" style="width:100%;display:block;">
            <div class="chart-caption">Figura 4 — Violin Plot: densidade de probabilidade revela diferenças de distribuição entre classes</div>
        </div></div>""", unsafe_allow_html=True)

st.markdown("""
<div style="padding:0 80px 32px;">
    <div class="insight">
        <strong>EDA como fundamento:</strong> a separação visível entre classes nas distribuições de
        Flow Duration, Flow IAT e Packet Length — já identificável nesta fase exploratória —
        antecipou as features que viriam a dominar a Feature Importance do modelo final
        (Subflow Bwd Bytes, Fwd IAT Total, Average Packet Size).
    </div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown("""
<style>
.footer {
    background: linear-gradient(180deg, #1a2233 0%, #0a0f1a 100%);
    padding: 50px 80px 40px;
    border-top: 1px solid #2e3f5c;
}
.footer-eyebrow {
    font-size: 11px; font-weight: 600; letter-spacing: 3px;
    color: #5b8dd9; text-transform: uppercase; margin-bottom: 16px;
}
.footer-grid {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr;
    gap: 32px;
    margin-bottom: 32px;
}
.footer-title {
    font-size: 11px; font-weight: 700; color: #d0d8e8;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;
}
.footer-text {
    font-size: 13px; color: #6b7a8d; line-height: 1.7;
}
.footer-text a { color: #5b8dd9; text-decoration: none; }
.footer-text a:hover { text-decoration: underline; }
.footer-mono {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: #8899aa;
}
.footer-bottom {
    font-size: 11px; color: #3d5278;
    border-top: 1px solid #2e3f5c;
    padding-top: 16px;
    display: flex; justify-content: space-between;
    flex-wrap: wrap; gap: 8px;
}
</style>

<div class="footer">
    <div class="footer-eyebrow">🛡️ ISCTE — Data Science & Business Analytics · Jun/2026</div>
    <div class="footer-grid">
        <div>
            <div class="footer-title">Anomaly Classifier — CICIDS2017</div>
            <div class="footer-text">
                Pipeline de detecção de anomalias em tráfego de rede combinando
                pseudo-labelling não supervisionado (Isolation Forest + Local Outlier Factor,
                consenso ponderado) com classificação supervisionada
                (Random Forest + Logistic Regression), avaliado contra ground truth
                real com split temporal 80/20.
            </div>
        </div>
        <div>
            <div class="footer-title">Projecto</div>
            <div class="footer-text">
                Projeto Aplicado em Ciência dos Dados<br>e Analítica de Negócio<br>
                ISCTE — Instituto Universitário de Lisboa<br>
                Prof. Ricardo Ferreira
            </div>
        </div>
        <div>
            <div class="footer-title">Recursos</div>
            <div class="footer-text footer-mono">
                <a href="https://github.com/RiSantos79/anomaly_classifier" target="_blank">
                    github.com/RiSantos79/<br>anomaly_classifier
                </a>
            </div>
            <div class="footer-text" style="margin-top:10px;">
                Dataset: CICIDS2017<br>(CIC, University of New Brunswick)
            </div>
        </div>
    </div>
    <div class="footer-bottom">
        <span>Ricardo Santos · Cibersegurança & Ciência de Dados · Lisboa, Portugal</span>
        <span class="footer-mono">Streamlit · pandas · scikit-learn · matplotlib</span>
    </div>
</div>
""", unsafe_allow_html=True)