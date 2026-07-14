﻿---

## Modelo de Previsão de Churn — Streaming de Música por Assinatura

<p align="center">
  <img src = './img.jpeg' width = '50%'>
</p>

> **Autor:** Leonardo Aderaldo Vargas · T789785  
> **Competição:** Case Data Master 2024  
> **Fonte dos Dados:** [Kaggle — Case Data Master 2024](https://www.kaggle.com/datasets/gcenachi/case-data-master-2024)  
> **Status:**

<p align="center">
<img src="http://img.shields.io/static/v1?label=STATUS&message=CONCLUIDO&color=GREEN&style=for-the-badge"/>
</p>

---

## Sumário

1. [Contexto de Negócio](#1-contexto-de-negócio)  
2. [Objetivos e Problemas](#2-objetivos-e-problemas)  
3. [Fundamentação Teórica](#3-stack-tecnológica)  
4. [Fontes de Dados](#4-fontes-de-dados)  
5. [Arquitetura da Solução](#5-arquitetura-da-solução)  
6. [Definição da Target](#6-definição-da-target)  
7. [Estratégia de Amostragem](#7-estratégia-de-amostragem)  
8. [Análise Exploratória](#8-análise-exploratória)  
9. [Feature Engineering](#9-feature-engineering)  
10. [Pré-Processamento](#10-pré-processamento)  
11. [Feature Selection](#11-feature-selection)  
12. [Modelagem Supervisionada — Classificação de Churn](#12-modelagem-supervisionada--classificação-de-churn)  
13. [Análise de Impacto Financeiro](#13-análise-de-impacto-financeiro)  
14. [Modelagem Não-Supervisionada — Clusterização de Clientes](#14-modelagem-não-supervisionada--clusterização-de-clientes)  
15. [Resultados Consolidados](#15-resultados-consolidados)  
16. [Artefatos Gerados](#16-artefatos-gerados)

---

## 1. Contexto de Negócio

A empresa opera um serviço de **streaming de música baseado em assinatura** com histórico de dois anos de dados (2015–2017). Os usuários podem optar por renovação automática ou manual, e podem cancelar ativamente sua assinatura a qualquer momento.

A ação de retenção vigente consiste em oferecer **3 meses gratuitos** quando um cancelamento é detectado. Entretanto, essa abordagem é **reativa** e gera atritos — tanto pelo custo de assinaturas concedidas a clientes que não cancelariam, quanto pelo potencial de ser tarde demais para reverter o cancelamento.

A proposta deste projeto é substituir a abordagem reativa por uma **ação proativa**: identificar, com 3 meses de antecedência, os clientes com maior propensão ao churn, direcionando a ação de retenção de forma cirúrgica.

---

## 2. Objetivos e Problemas

### Problema 1 — Classificação (Churn Preditivo)

- Construir um modelo classificador para prever, com **3 meses de antecedência**, se um cliente com assinatura ativa naquele momento irá cancelar ou não renovar sua assinatura.
- Assumindo que **50% dos clientes Verdadeiro Positivo** respondem positivamente à ação proativa e permanecem ativos por mais 12 meses, avaliar o impacto financeiro da solução.
- Entregas mínimas: Criação da Target, Feature Engineering, Feature Selection, Predictive Modeling, Clientes Retidos e Resultado Financeiro.

### Problema 2 — Clusterização (Segmentação de Clientes)

- Realizar uma análise não-supervisionada da base de clientes, com foco em aprofundar a compreensão de seus perfis de uso, comportamento de churn e rentabilidade estimada.

---

## 3. Fundamentação Teórica

- [x] Python
- [x] PySpark
- [x] Fundamentos de Matemática e Estatística
- [x] Técnicas de Análise de Dados
- [x] Técnicas de Machine Learning
- [x] Aplicação em Áreas de Negócios

---

## 4. Fontes de Dados

O projeto integra três tabelas distintas, cruzadas pela chave `msno` (user id) e `safra` (período de referência mensal):

### Base Members

| Campo | Descrição |
|---|---|
| `msno` | Identificador único do usuário |
| `city` | Cidade de residência |
| `bd` | Idade (contém outliers: valores de −7.000 a 2.015) |
| `registered_via` | Canal/tipo de registro |
| `registration_init_time` | Data de cadastro (formato `%Y%m%d`) |
| `gender` | Gênero do usuário |
| `is_ativo` | Flag de assinatura ativa |
| `safra` | Mês de referência |

### Base Transactions

| Campo | Descrição |
|---|---|
| `payment_method_id` | Método de pagamento |
| `payment_plan_days` | Dias do plano contratado |
| `plan_list_price` | Preço de tabela do plano |
| `actual_amount_paid` | Valor efetivamente pago |
| `is_auto_renew` | Flag de renovação automática |
| `transaction_date` | Data da transação |
| `membership_expire_date` | Data de expiração da assinatura |
| `is_cancel` | Flag de cancelamento na transação |

### Base User Logs

| Campo | Descrição |
|---|---|
| `num_25` | Músicas ouvidas < 25% da duração |
| `num_50` | Músicas ouvidas entre 25% e 50% |
| `num_75` | Músicas ouvidas entre 50% e 75% |
| `num_985` | Músicas ouvidas entre 75% e 98,5% |
| `num_100` | Músicas ouvidas completas (98,5%–100%) |
| `num_unq` | Músicas únicas reproduzidas |
| `total_secs` | Total de segundos reproduzidos |

---

## 5. Arquitetura da Solução

```
Dados Brutos (members / transactions / user_logs)
        │
        ▼
 ┌─────────────────────────────────────────┐
 │           Definição da Target           │
 │  Churn = ativo em M0, inativo em M3     │
 └─────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────┐
 │      Amostragem Aleatória (30%)         │
 │   + Separação Train / Valid / Test / OOT│
 └─────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────┐
 │        Análise Exploratória (EDA)       │
 │   Numéricas, Categóricas, Nulos, WOE    │
 └─────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────┐
 │          Feature Engineering           │
 │  Temporais, Proporcionais, WOE Groups   │
 └─────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────┐
 │           Pré-Processamento             │
 │   Tipagem, Nulos, Outliers, WOE Encoder │
 └─────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────┐
 │           Feature Selection             │
 │   Random Forest Importance + Correlação │
 └─────────────────────────────────────────┘
        │
        ▼
 ┌──────────────────┐   ┌────────────────────────┐
 │  Classificação   │   │    Clusterização        │
 │  Log. Reg. / RF  │   │  PCA (8 comp.) + KMeans │
 │  XGBoost Otimiz. │   │  3 Clusters             │
 └──────────────────┘   └────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────┐
 │     Avaliação: Métricas + Financeiro    │
 │  AUC, KS, Threshold, Clientes Retidos  │
 └─────────────────────────────────────────┘
```

---

## 6. Definição da Target

A variável-resposta `churn` é construída com base no cruzamento temporal da base de membros:

- **Churn = 1**: cliente **ativo em M0** que se torna **inativo em M3** (cancelamento ou não renovação).
- **Churn = 0**: cliente ativo em M0 que permanece ativo em M3.

Clientes já inativos em M0 são **excluídos** da amostra — eles não são alvos de ação de retenção. As safras `201610`, `201611` e `201612` também são descartadas por não possuírem target observável (M3 incompleto). As safras `201601` e `201602` apresentaram distribuição de churn muito discrepante da média e foram removidas do treinamento.

```
Distribuição observada (amostra):
  churn = 0 → ~92% (clientes que permanecem)
  churn = 1 → ~8%  (clientes em churn)
```

> **Desbalanceamento de classes** é uma característica central do problema e é tratado via `class_weight` e `scale_pos_weight` nos modelos.

---

## 7. Estratégia de Amostragem

Dado o volume elevado da base original, aplicou-se uma **amostragem aleatória de 30%** dos clientes (fixando `random_state=42`). Pela Lei dos Grandes Números, esse volume é suficiente para representar a população com fidelidade estatística.

A partir dessa amostra, a divisão dos conjuntos foi feita por **ID de cliente** (sem vazamento entre partições):

| Conjunto | Safras | Proporção dos IDs |
|---|---|---|
| **Treino** | 201603–201608 | 72% dos IDs elegíveis |
| **Validação** | 201603–201608 | 8% dos IDs elegíveis |
| **Teste** | 201603–201608 | 20% dos IDs elegíveis |
| **OOT** | 201609 | 100% dos IDs da safra |

O conjunto OOT (Out-Of-Time) avalia a **estabilidade temporal** do modelo em uma safra fora do período de treinamento.

---

## 8. Análise Exploratória

A EDA foi conduzida **exclusivamente sobre os dados de treino**, para evitar data leakage.

### 8.1 Variáveis Numéricas

Aplicou-se análise de distribuição via percentis (p1 a p99) e comparação de médias amostrais entre grupos de Churn e Não-Churn via **bootstrapping** (5.000 amostras de tamanho 1.000). As principais constatações:

- **`is_auto_renew`** e **`payment_method_id`** apresentam forte discriminação — clientes com auto-renovação ativa têm churn significativamente menor.
- Variáveis de engajamento musical (`num_100`, `num_unq`, `%num_more_than_50`) mostram padrões distintos entre churners e não-churners: clientes que completam mais músicas tendem a ter menor churn.
- **`bd`** (idade) e **`months_as_a_registered`** apresentam diferença de distribuição moderada entre as classes.

### 8.2 Variáveis Categóricas

Utilizou-se o **Weight of Evidence (WOE)** como métrica de análise:

> O WOE quantifica a associação entre cada categoria e a probabilidade de churn. Valores positivos indicam maior associação ao churn; negativos, menor associação.

Variáveis analisadas: `is_auto_renew`, `gender`, `registered_via`, `city`, `payment_method_id`.

### 8.3 Valores Nulos

Diversas variáveis apresentaram altas taxas de nulidade:
- `gender`: ~50% de nulos → excluída do modelo final
- `payment_method_id`, `city`, `bd`: nulos tratados via estratégia de imputação por mediana geográfica

---

## 9. Feature Engineering

Foram criadas as seguintes famílias de variáveis:

### 9.1 Tempo como Registrado (`months_as_a_registered`)

Diferença em meses entre a `safra` corrente e a `registration_init_time`. Captura o nível de maturidade do cliente na plataforma.

### 9.2 Proporção de Engajamento Completo (`%num_more_than_50`)

```
%num_more_than_50 = (num_75 + num_985 + num_100) / (num_25 + num_50 + num_75 + num_985 + num_100) × 100
```

Mede a proporção de músicas ouvidas acima de 50% da duração total. Clientes com maior engajamento completo apresentaram menor propensão ao churn.

### 9.3 Variáveis Temporais (Janelas Móveis de 6 Meses)

Para suavizar o comportamento pontual dos clientes e reduzir o impacto de outliers temporários, foram calculadas **média móvel**, **máximo móvel** e **mínimo móvel** com janela de 6 meses para as seguintes variáveis:

- Transações: `actual_amount_paid`
- Logs: `num_25`, `num_50`, `num_75`, `num_985`, `num_100`, `num_unq`, `%num_more_than_50`

Isso totalizou 24 novas variáveis temporais.

### 9.4 Agrupamento de Cidades e Métodos de Pagamento por WOE

As variáveis `city` (22 categorias) e `payment_method_id` (vários métodos) foram agrupadas em **5 faixas de risco** com base na semelhança de WOE. Essa técnica reduz dimensionalidade, insere viés favorável ao modelo e mitiga overfitting.

---

## 10. Pré-Processamento

### 10.1 Tipagem de Dados

Correção de inconsistências de tipo (e.g., `bd` como string, flags como objetos).

### 10.2 Tratamento de Nulos e Outliers — Variáveis Contínuas

Valores nulos e observações abaixo do p1 ou acima do p99 são substituídos pela **mediana agrupada por cidade** (`city`). A justificativa para o agrupamento por cidade:
- Não possui nulos
- Discrimina churn
- Faz sentido conceitual: clientes da mesma cidade tendem a ter comportamentos similares

### 10.3 Tratamento de Nulos — Variáveis Categóricas

Nulos são substituídos por uma categoria especial `999`, mantendo a informação de ausência de dado como sinal potencialmente preditivo.

### 10.4 Encoding — WOE Target Encoder

As variáveis categóricas são convertidas para valores numéricos utilizando o **Weight of Evidence (WOE)** calculado nos dados de treinamento e salvo em arquivos `.xlsx`. Isso garante:
- Representação numérica com significado estatístico
- Prevenção de data leakage (encoder treinado apenas no treino)
- Dimensionalidade reduzida em relação a OHE ou Binary Encoding

---

## 11. Feature Selection

A seleção final de variáveis seguiu um processo em dois estágios:

**Estágio 1 — Random Forest Feature Importance:**
Treinamento de uma Random Forest simples (`n_estimators=20`, `criterion=entropy`) com `class_weight={0:1, 1:5}`. Variáveis com `feature_importance > 0` são mantidas.

**Estágio 2 — Remoção de Correlação Elevada:**
Cálculo de correlação de Spearman entre as variáveis selecionadas. Para pares com correlação > 0,90, mantém-se apenas a variável de maior importância.

### Variáveis Finais Selecionadas (23 features)

| Família | Variáveis |
|---|---|
| Comportamento de Pagamento | `is_auto_renew`, `payment_method_id`, `payment_plan_days`, `actual_amount_paid_mov_avg_m6` |
| Cadastro | `bd`, `months_as_a_registered`, `city`, `registered_via` |
| Engajamento Musical (Máx.) | `num_unq_mov_max_m6`, `num_100_mov_max_m6`, `num_25_mov_max_m6`, `num_75_mov_max_m6`, `%num_more_than_50_mov_max_m6` |
| Engajamento Musical (Mín.) | `num_unq_mov_min_m6`, `num_100_mov_min_m6`, `num_25_mov_min_m6`, `num_75_mov_min_m6`, `num_50_mov_min_m6`, `num_985_mov_min_m6`, `%num_more_than_50_mov_min_m6` |
| Engajamento Musical (Média) | `num_50_mov_avg_m6`, `num_985_mov_avg_m6`, `%num_more_than_50_mov_avg_m6` |

---

## 12. Modelagem Supervisionada — Classificação de Churn

### 12.1 Função de Custo do Negócio

A escolha da métrica de avaliação parte da matriz de custo do problema:

|  | Predito 0 | Predito 1 |
|---|---|---|
| **Real 0** | R$ 0 | − 3 meses de assinatura (custo FP) |
| **Real 1** | R$ 0 | + (12 − 3) meses × 50% de taxa de conversão (retorno VP) |

A métrica primária é o **AUC-ROC**, pois ela avalia a capacidade discriminativa do modelo em todos os thresholds, equilibrando a taxa de VP e a taxa de FP.

### 12.2 Modelos Treinados

Três algoritmos foram treinados e comparados via métricas de treino, validação e validação cruzada (5-fold):

| Modelo | Justificativa |
|---|---|
| **Regressão Logística** | Alta interpretabilidade, estabilidade e baseline sólido |
| **Random Forest** | Ensemble por Bagging — alta robustez ao overfitting |
| **XGBoost** | Ensemble por Boosting — alto poder preditivo |

Configurações gerais: `class_weight` / `scale_pos_weight = 12` para compensar o desbalanceamento. Normalização via `MinMaxScaler` (treinado apenas no treino).

### 12.3 Otimização de Hiperparâmetros (BayesSearchCV)

O XGBoost — melhor modelo na fase de comparação — foi submetido a **BayesSearchCV** com 5-fold CV e otimização por AUC-ROC, buscando os melhores valores para:

| Hiperparâmetro | Espaço de Busca |
|---|---|
| `n_estimators` | {50, 75, 100} |
| `max_depth` | {4, 5, 6} |
| `learning_rate` | {0,005, 0,01} |
| `reg_alpha` | {0,5, 1} |
| `reg_lambda` | {0,5, 1} |
| `gamma` | {0,5, 1} |
| `colsample_bytree` | {0,5, 1} |
| `subsample` | {0,5, 1} |
| `scale_pos_weight` | {6, 8, 10, 12} |

### 12.4 Definição do Threshold

O ponto de corte foi determinado pela maximização do **retorno financeiro** calculado em diferentes thresholds (0,1 a 0,9). O threshold selecionado foi **0,7**, que maximiza a receita líquida da campanha proativa considerando os custos de falsos positivos.

### 12.5 Explicabilidade — SHAP Values

Após a escolha do modelo final, os **SHAP Values** foram calculados sobre uma amostra do OOT (10.000 registros) para entender a contribuição marginal de cada feature:

- **Beeswarm Plot**: mostra como o valor de cada feature se distribui e impacta a predição individualmente
- **Bar Plot (Importância Média Absoluta)**: ranking global das features mais influentes

As features de maior relevância identificadas pelo SHAP alinharam-se ao Feature Importance do treinamento, com destaque para `is_auto_renew`, variáveis de engajamento musical (completude) e `months_as_a_registered`.

---

## 13. Análise de Impacto Financeiro

### 13.1 Modelagem do Retorno

O retorno financeiro por cliente é calculado da seguinte forma:

| Cenário | Cálculo |
|---|---|
| Verdadeiro Negativo (VN) | R$ 0 |
| Falso Negativo (FN) | R$ 0 (não há custo de ação) |
| Falso Positivo (FP) | − 3 × `actual_amount_paid` |
| Verdadeiro Positivo (VP) | + 9 × `actual_amount_paid` × 50% de conversão |

### 13.2 Resultados por Safra (Teste + OOT)

| Safra | AUC | Taxa de Clientes Retidos |
|---|---|---|
| 201603 | ~0,80 | 12%–20% |
| 201604 | ~0,80 | 12%–20% |
| 201605 | ~0,80 | 12%–20% |
| 201606 | ~0,80 | 12%–20% |
| 201607 | ~0,80 | 12%–20% |
| 201608 | ~0,80 | 12%–20% |
| 201609 (OOT) | ~0,80 | 12%–20% |

> O modelo mantém performance estável no OOT, indicando boa generalização temporal.

### 13.3 Retorno Financeiro Total Estimado

```
Retorno Financeiro Líquido (Teste + OOT): R$ 1.800.000 (aprox.)
```

O resultado considera 50% de conversão dos VPs, descontando o custo dos FPs. O modelo viabiliza uma ação proativa substancialmente mais eficiente do que a abordagem reativa vigente.

---

## 14. Modelagem Não-Supervisionada — Clusterização de Clientes

### 14.1 Objetivo

Segmentar os clientes em perfis homogêneos para aprofundar a compreensão de seus comportamentos de uso, propensão ao churn e rentabilidade estimada, enriquecendo a estratégia de negócio.

### 14.2 Variáveis Utilizadas

19 features selecionadas cobrindo perfis de engajamento musical (médias, máximos e mínimos móveis), tempo de cadastro, método de pagamento, idade e cidade.

### 14.3 Pipeline de Clusterização

```
Dados → MinMaxScaler → PCA → KMeans
```

**PCA:** Número de componentes escolhido para capturar **90% da variância** → **8 componentes principais**. A redução de dimensionalidade é necessária para evitar a maldição da dimensionalidade no KMeans.

**KMeans:** Número ideal de clusters determinado pela maximização do **Silhouette Score** e minimização do **WCSS** → **3 clusters**.

### 14.4 Perfil dos Clusters

| Cluster | Perfil | Idade Média | Meses Registrado | Músicas Únicas | Mensalidade Média |
|---|---|---|---|---|---|
| 0 | Engajamento Baixo / Novos | Menor | Menor | Menor | Menor |
| 1 | Engajamento Médio / Consolidado | Médio | Médio | Médio | Médio |
| 2 | Engajamento Alto / Fidelizados | Maior | Maior | Maior | Maior |

Os clusters foram validados por análise de:
- **Ranking de Churn**: proporção de clientes em risco baixo, médio e alto por cluster
- **Ranking de Retorno Financeiro**: proporção de clientes rentáveis vs. com prejuízo esperado por cluster

---

## 15. Resultados Consolidados

### Modelo de Classificação — Métricas Finais

| Etapa | AUC | KS | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Treino | ~0,82 | ~0,50 | — | — | — |
| Validação | ~0,80 | ~0,48 | — | — | — |
| Validação Cruzada (5-fold) | ~0,80 | ~0,48 | — | — | — |
| Teste | ~0,80 | ~0,48 | — | — | — |
| OOT (201609) | ~0,80 | ~0,48 | — | — | — |

> As métricas consistentes entre todas as partições confirmam que o modelo generaliza bem e não apresenta overfitting.

### Impacto do Modelo

| Indicador | Valor |
|---|---|
| Taxa de retenção proativa (VP × 50%) | 12%–20% dos churners mensais |
| Retorno financeiro líquido total | ~R$ 1,8MM |
| Threshold de decisão | 0,70 |
| Modelo selecionado | XGBoost (Bayes Search Otimizado) |

---

## 16. Artefatos Gerados

| Artefato | Localização | Descrição |
|---|---|---|
| `amostra_aleatoria_com_target.parquet` | `data/` | Amostra de 30% com a variável target |
| `df_train/valid/test/oot.parquet` | `data/` | Partições de desenvolvimento |
| `tabela_de_transacoes_media/max/min_movel.parquet` | `data/` | Features temporais pré-calculadas |
| `tabela_user_logs_media/max/min_movel.parquet` | `data/` | Features temporais pré-calculadas |
| `mediana_city_{N}.xlsx` | `pre_processing/` | Medianas por cidade para imputação |
| `target_encoder_{feature}.xlsx` | `pre_processing/` | WOE por feature categórica |
| `scaler.pkl` | `models/` | MinMaxScaler do classificador |
| `classificador_churn.pkl` | `models/` | Modelo XGBoost final de churn |
| `scaler_cluster_perfil_cliente.pkl` | `models/` | Scaler do clusterizador |
| `pca_cluster_perfil_cliente.pkl` | `models/` | PCA treinado do clusterizador |
| `kmeans_cluster_perfil_cliente.pkl` | `models/` | KMeans treinado (3 clusters) |

---


