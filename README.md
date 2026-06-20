# Dashboard Interativo — Anomaly Classifier (CICIDS2017)

Dashboard Streamlit interativo, organizado como um **sistema de apoio à decisão** para deteção de anomalias em tráfego de rede,
desenvolvido como parte do **Projeto Aplicado em Ciência dos Dados e Analítica de Negócio** (ISCTE — Instituto Universitário de Lisboa).

## Estrutura

O dashboard está organizado em 3 abas, cada uma desenhada para uma audiência e decisão específicas. Todas incluem
KPIs com cores semânticas (verde/amarelo/vermelho), um glossário de termos (`ℹ️`), uma nota sobre a proveniência
dos dados (Treino/Teste/Fora da Amostra) e uma secção de **Insights e Recomendações** com lógica de regras transparente.

- **🚨 Operacional / SOC** — o que investigar agora. KPIs (Registos, % Anómalo, Fluxos Críticos, FNR, FPR, ataques não
  detetados), top portos com anomalias, matriz de confusão, composição do tráfego e capacidade de deteção por tipo de ataque.
- **📈 Tendências / Gestão** — escalar recursos ou não. Evolução temporal com tendência e projeção (regressão linear, com
  destaque visual do dia de pico), distribuição de risco por dia, distribuição de confiança do modelo por tipo de tráfego,
  anéis de progresso por nível de risco.
- **⚖️ Comparação de Modelos** — qual modelo produtizar. Dumbbell chart de métricas, curva ROC com threshold marcado,
  violin plots de probabilidade por risco, diagramas de fluxo Sankey, e uma secção de **Explicabilidade do Modelo**
  (Feature Importance + SHAP summary plot) para o Random Forest.

Um indicador de **Desempenho do Modelo** (Random Forest, amostra completa, sem filtros) está sempre visível no
cabeçalho, antes de entrar em qualquer aba.

Os filtros na barra lateral adaptam-se à aba ativa.

## Acessibilidade

A paleta de cores foi revista e validada para daltonismo (deuteranopia/protanopia) — os pares "bom vs mau" usam
azul/vermelho em vez de verde/vermelho, em todos os gráficos e KPIs.

## Pipeline de origem

Pseudo-labelling não supervisionado (Isolation Forest + Local Outlier Factor, consenso ponderado) seguido de
classificação supervisionada (Random Forest e Logistic Regression), avaliado contra ground truth real com split
temporal 80/20, sobre o dataset CICIDS2017.

## Executar localmente

```bash
uv run streamlit run app.py
```

Requer, na mesma pasta:
- `dashboard_data.parquet` — dados pré-processados (obrigatório)
- `feature_importance.parquet` — importância das features do Random Forest (opcional; sem ele, a secção de
  Explicabilidade mostra um aviso em vez de rebentar)
- `shap_values.parquet` — valores SHAP de uma amostra de referência (opcional, mesma lógica)

Todos gerados por `gerar_dashboard_data.py`, disponível no repositório principal do projeto. Gerar o SHAP requer
adicionalmente `pip install shap` no ambiente de geração (não é necessário para correr o dashboard em si).

## Repositório principal do projeto

[github.com/RiSantos79/anomaly_classifier](https://github.com/RiSantos79/anomaly_classifier) — notebook completo, relatório e apresentação.

## Dataset

[CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) (Canadian Institute for Cybersecurity, University of New Brunswick).

---
Ricardo Santos · Cibersegurança & Ciência de Dados · ISCTE, 2025/2026

