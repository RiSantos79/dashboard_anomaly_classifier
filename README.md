# Dashboard Interativo — Anomaly Classifier (CICIDS2017)

Dashboard Streamlit interativo para visualização e análise dos resultados do pipeline de deteção de anomalias em tráfego de rede, 
desenvolvido como parte do **Projeto Aplicado em Ciência dos Dados e Analítica de Negócio** (ISCTE — Instituto Universitário de Lisboa).

## Estrutura

O dashboard está organizado em 3 abas, cada uma desenhada para uma audiência e decisão específicas:

- **🚨 Operacional / SOC** — para o analista SOC decidir o que investigar agora (KPIs, top portos, matriz de confusão, eficácia de detecção por tipo de ataque).
- **📈 Tendências / Gestão** — para o gestor SOC decidir se escala recursos (evolução temporal com tendência/projeção, distribuição de risco, boxplot por tipo de tráfego).
- **⚖️ Comparação de Modelos** — para decidir qual modelo (Random Forest vs Logistic Regression) produtizar (dumbbell chart, curva ROC, violin plots, diagramas de fluxo Sankey).

Os filtros na barra lateral adaptam-se à aba ativa.

## Pipeline de origem

Pseudo-labelling não supervisionado (Isolation Forest + Local Outlier Factor, consenso ponderado) seguido de classificação supervisionada (Random Forest e Logistic Regression), 
avaliado contra ground truth real com split temporal 80/20, sobre o dataset CICIDS2017.

## Executar localmente

```bash
uv run streamlit run app.py
```

Requer `dashboard_data.parquet` na mesma pasta (gerado pelo script `gerar_dashboard_data.py`, disponível no repositório principal do projeto).

## Repositório principal do projeto

[github.com/RiSantos79/anomaly_classifier](https://github.com/RiSantos79/anomaly_classifier) — notebook completo, relatório e apresentação.

## Dataset

[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (Canadian Institute for Cybersecurity, University of New Brunswick).

---
Ricardo Santos · Cibersegurança & Ciência de Dados · ISCTE, 2025/2026

