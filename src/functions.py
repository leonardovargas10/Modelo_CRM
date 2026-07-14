"""Funções reutilizáveis extraídas do notebook do case Data Masters."""

import builtins
import pickle
import random

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from IPython.display import display
from joblib import Parallel, delayed
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

try:
    from skopt import BayesSearchCV
except ImportError:  # Necessário apenas nas rotinas de otimização.
    BayesSearchCV = None


def plota_barras(lista_variaveis, df, titulo, rotation=0):        
    k = 0
    # Ordena os dados para garantir que as labels correspondam corretamente às barras
    df_sorted = df[lista_variaveis[k]].value_counts().index
    ax = sns.countplot(x=lista_variaveis[k], data=df, order=df_sorted, color='#1FB3E5')
    
    ax.set_title(f'{titulo}')
    ax.set_xlabel(f'{lista_variaveis[k]}', fontsize=14)
    ax.set_ylabel('Quantidade', fontsize=14)
    
    # Calcular o total para obter os percentuais
    total = sum([p.get_height() for p in ax.patches])
    
    sizes = []
    for bar in ax.patches:
        height = bar.get_height()
        sizes.append(height)
        ax.text(bar.get_x() + bar.get_width()/2,
                height,
                f'{builtins.round((height/total)*100, 2)}%',
                ha='center',
                fontsize=12
        )
    
    ax.set_ylim(0, max(sizes) * 1.1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation, ha='right', fontsize=10)
    ax.set_yticklabels(['{:,.0f}'.format(y) for y in ax.get_yticks()], fontsize=10)

    plt.tight_layout()
    plt.show()


def plota_grafico_linhas(df, x, y, nao_calcula_media, title):

    if nao_calcula_media:
        # Criando o gráfico de linha
        plt.figure(figsize=(18, 6))
        plt.plot(df[x], df[y], marker='o', linestyle='-', color='#1FB3E5')

        # Adicionando títulos e rótulos aos eixos
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)

        for i, txt in enumerate(df[y]):
            plt.annotate(f'{txt:.1f}', (df[x][i], df[y][i]), textcoords="offset points", xytext=(0,1), ha='center')

        # Exibindo o gráfico
        plt.grid(True)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()
    else:
        media = df[y].mean()
        # Criando o gráfico de linha
        plt.figure(figsize=(18, 6))
        plt.plot(df[x], df[y], marker='o', linestyle='-', color='#1FB3E5')

        # Adicionando linha da média
        plt.axhline(y=media, color='r', linestyle='--', linewidth=1, label=f'Média: {media:.2f}')
        plt.legend()

        # Adicionando títulos e rótulos aos eixos
        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)

        for i, txt in enumerate(df[y]):
            plt.annotate(f'{txt:.1f}', (df[x][i], df[y][i]), textcoords="offset points", xytext=(0,1), ha='center')

        # Exibindo o gráfico
        plt.grid(True)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


def muda_tipagem_variavel(df, feature, type):

    if type == "int":
        df[feature] = df[feature].apply(lambda x: int(x) if pd.notnull(x) else 999999)
    else:
        df[feature] = df[feature].apply(lambda x: float(x) if pd.notnull(x) else 999999)

    df.replace(999999, np.nan, inplace=True)

    return df[feature]


def analisa_distribuicao_via_percentis(df, variaveis):
    def sublinha_percentis(s):
        is_1_percentile = s.name == '1%'
        is_99_8_percentile = s.name == '99.8%'
        if is_1_percentile or is_99_8_percentile:
            return ['background-color: blue'] * len(s)
        else:
            return [''] * len(s)

    percentis = df[variaveis].describe(percentiles = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.998]).style.apply(sublinha_percentis, axis=1)    

    return percentis


def compara_medias_amostras(df, variaveis_continuas):  
    num_variaveis = len(variaveis_continuas)
    num_pares = (num_variaveis + 1) // 2  # Número de pares de variáveis para subplots
    fig, axes = plt.subplots(num_pares, 2, figsize=(14, 4 * num_pares))

    # Ajusta para o caso onde há apenas uma variável
    if num_pares == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i in range(num_pares):
        if 2 * i < num_variaveis:
            variavel1 = variaveis_continuas[2 * i]
            percentis1 = df[variavel1].describe(percentiles=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
            p1_1 = percentis1['1%']
            p99_1 = percentis1['99%']
            df_raw1 = df.loc[(df[variavel1] > p1_1) & (df[variavel1] < p99_1)].copy()
            df_com_churn1 = df_raw1.loc[df_raw1["churn"] == 1]
            df_sem_churn1 = df_raw1.loc[df_raw1["churn"] == 0]
            
            medias_amostrais_com_churn1 = []
            medias_amostrais_sem_churn1 = []
            
            for j in range(5000):
                amostra_churn1 = random.choices(df_com_churn1[variavel1].values, k=1000)
                media_amostra_churn1 = np.mean(amostra_churn1)
                medias_amostrais_com_churn1.append(media_amostra_churn1)

                amostra_sem_churn1 = random.choices(df_sem_churn1[variavel1].values, k=1000)
                media_amostra_sem_churn1 = np.mean(amostra_sem_churn1)
                medias_amostrais_sem_churn1.append(media_amostra_sem_churn1)

            ax_hist1 = axes[i, 0]
            ax_hist1.hist(medias_amostrais_com_churn1, bins=30, alpha=0.5, label='Churn', linewidth=5, color="red")
            ax_hist1.hist(medias_amostrais_sem_churn1, bins=30, alpha=0.5, label='Sem Churn', linewidth=5, color="green")
            ax_hist1.legend(loc='upper right')
            ax_hist1.set_xlabel('Valores')
            ax_hist1.set_ylabel('Frequência')
            ax_hist1.set_title(f'Distribuição das Médias Amostrais de "{variavel1}" ')
            ax_hist1.grid(True)
        
        if 2 * i + 1 < num_variaveis:
            variavel2 = variaveis_continuas[2 * i + 1]
            percentis2 = df[variavel2].describe(percentiles=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
            p1_2 = percentis2['1%']
            p99_2 = percentis2['99%']
            df_raw2 = df.loc[(df[variavel2] > p1_2) & (df[variavel2] < p99_2)].copy()
            df_com_churn2 = df_raw2.loc[df_raw2["churn"] == 1]
            df_sem_churn2 = df_raw2.loc[df_raw2["churn"] == 0]
            
            medias_amostrais_com_churn2 = []
            medias_amostrais_sem_churn2 = []
            
            for j in range(5000):
                amostra_churn2 = random.choices(df_com_churn2[variavel2].values, k=1000)
                media_amostra_churn2 = np.mean(amostra_churn2)
                medias_amostrais_com_churn2.append(media_amostra_churn2)

                amostra_sem_churn2 = random.choices(df_sem_churn2[variavel2].values, k=1000)
                media_amostra_sem_churn2 = np.mean(amostra_sem_churn2)
                medias_amostrais_sem_churn2.append(media_amostra_sem_churn2)

            ax_hist2 = axes[i, 1]
            ax_hist2.hist(medias_amostrais_com_churn2, bins=30, alpha=0.5, label='Churn', linewidth=5, color="red")
            ax_hist2.hist(medias_amostrais_sem_churn2, bins=30, alpha=0.5, label='Sem Churn', linewidth=5, color="green")
            ax_hist2.legend(loc='upper right')
            ax_hist2.set_xlabel('Valores')
            ax_hist2.set_ylabel('Frequência')
            ax_hist2.set_title(f'Distribuição das Médias Amostrais de "{variavel2}" ')
            ax_hist2.grid(True)

    plt.tight_layout()
    plt.show()


def woe(df, feature, target):
    churn = df.loc[df[target] == 1].groupby(feature, as_index = False)[target].count().rename({target:'churn'}, axis = 1)
    sem_churn = df.loc[df[target] == 0].groupby(feature, as_index = False)[target].count().rename({target:'sem_churn'}, axis = 1)

    woe = churn.merge(sem_churn, on = feature, how = 'left')
    woe['percent_churn'] = woe['churn']/woe['churn'].sum()
    woe['percent_sem_churn'] = woe['sem_churn']/woe['sem_churn'].sum()
    woe['woe'] = round(np.log(woe['percent_churn']/woe['percent_sem_churn']), 3)
    woe.sort_values(by = 'woe', ascending = True, inplace = True)
    
    weight_of_evidence = woe['woe'].unique()


    x = list(woe[feature])
    y = list(woe['woe'])

    plt.figure(figsize=(10, 4))
    plt.plot(x, y, marker='o', linestyle='--', linewidth=2, color='#1FB3E5')

    for label, value in zip(x, y):
        plt.text(x=label, y=value, s=str(value), fontsize=10, color='red', ha='left', va='center', rotation=45)

    plt.title(f'Weight of Evidence da variável "{feature}"', fontsize=14)
    plt.xlabel('Classes', fontsize=14)
    plt.ylabel('Weight of Evidence', fontsize=14)
    plt.xticks(ha='right', fontsize=10, rotation=45)
    plt.show()


def months_as_a_registered(df):

    df["registration_init_time"] = df["registration_init_time"].apply(lambda x:str(x)[:6])

    # Converter as colunas para objetos datetime
    df['registration_init_time'] = pd.to_datetime(df['registration_init_time'], format='%Y%m')
    df['safra'] = pd.to_datetime(df['safra'], format='%Y%m')

    # Calcular a diferença de meses
    df['months_as_a_registered'] = (df['safra'].dt.year - df['registration_init_time'].dt.year) * 12 + (df['safra'].dt.month - df['registration_init_time'].dt.month)
    df["safra"] = df["safra"].apply(lambda x:str(x)[:7].replace("-", ""))

    return df['months_as_a_registered']


def num_more_than_50(df):

    df["num_less_than_50"] = df["num_25"] + df["num_50"]
    df["num_more_than_50"] = df["num_75"] + df["num_985"] + df["num_100"]
    df["%num_more_than_50"] = round(df["num_more_than_50"]/(df["num_more_than_50"]+df["num_less_than_50"])*100, 2)

    return df["%num_more_than_50"]


def media_movel(df, feature, window):

    df[f"{feature}_mov_avg_m{window}"] = df.groupby('msno')[feature].rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
                                           
    return df[f"{feature}_mov_avg_m{window}"]


def max_movel(df, feature, window):

    df[f"{feature}_mov_max_m{window}"] = df.groupby('msno')[feature].rolling(window=window, min_periods=1).max().reset_index(level=0, drop=True)
    
    return df[f"{feature}_mov_max_m{window}"]


def min_movel(df, feature, window):

    df[f"{feature}_mov_min_m{window}"] = df.groupby('msno')[feature].rolling(window=window, min_periods=1).min().reset_index(level=0, drop=True)
    
    return df[f"{feature}_mov_min_m{window}"]


def get_temporal_features(df):

    tabela_de_transacoes_media_movel = pd.read_parquet("../00_DataMaster/data/tabela_de_transacoes_media_movel.parquet")
    tabela_user_logs_media_movel = pd.read_parquet("../00_DataMaster/data/tabela_user_logs_media_movel.parquet")

    tabela_de_transacoes_max_movel = pd.read_parquet("../00_DataMaster/data/tabela_de_transacoes_max_movel.parquet")
    tabela_user_logs_max_movel = pd.read_parquet("../00_DataMaster/data/tabela_user_logs_max_movel.parquet")

    tabela_de_transacoes_min_movel = pd.read_parquet("../00_DataMaster/data/tabela_de_transacoes_min_movel.parquet")
    tabela_user_logs_min_movel = pd.read_parquet("../00_DataMaster/data/tabela_user_logs_min_movel.parquet")

    df = df.merge(tabela_de_transacoes_media_movel, on = ["msno", "safra"], how = "left")
    df = df.merge(tabela_user_logs_media_movel, on = ["msno", "safra"], how = "left")

    df = df.merge(tabela_de_transacoes_max_movel, on = ["msno", "safra"], how = "left")
    df = df.merge(tabela_user_logs_max_movel, on = ["msno", "safra"], how = "left")

    df = df.merge(tabela_de_transacoes_min_movel, on = ["msno", "safra"], how = "left")
    df = df.merge(tabela_user_logs_min_movel, on = ["msno", "safra"], how = "left")

    return df


def transform_to_percentiles(df, n, variavel_continua):
    # Calcula os limites dos percentiles
    percentile_limits = [i / n for i in range(n+1)] 
    
    # Aplica a função qcut para transformar a variável em percentiles
    percentiles = pd.qcut(df[variavel_continua], q=n, labels=False, duplicates='drop')
    
    return percentiles


def agrupa_categorias_cidade_pelo_woe(df):

    df['city'] = (
                np.where(df['city'].isin(['13', '14', '16', '7', '20']), 0, 
                np.where(df['city'].isin(['17', '4', '5', '18']), 1, 
                np.where(df['city'].isin(['19', '3', '15', '10']), 2, 
                np.where(df['city'].isin(['12', '11', '6', '22']), 3, 
                np.where(df['city'].isin(['8', '21', '9', '1']), 4, 
                np.nan)))))
    )

    return df['city']


def agrupa_categorias_metodo_pagamento_pelo_woe(df):

    df['payment_method_id'] = (
                np.where(df['payment_method_id'].isin(['32', '14', '10', '19', '31']), 0, 
                np.where(df['payment_method_id'].isin(['34', '41', '18', '37', '21']), 1, 
                np.where(df['payment_method_id'].isin(['23', '33', '39', '27']), 2, 
                np.where(df['payment_method_id'].isin(['40', '30', '16', '36', '26']), 3, 
                np.where(df['payment_method_id'].isin(['38', '29', '28', '35', '17']), 4, 
                np.nan)))))
    )

    return df['payment_method_id']


def pre_processamento_categoricas_nulas(df):
    variaveis_categoricas = ['is_auto_renew', 'gender', 'registered_via', 'city', 'payment_method_id']

    def aplica_tratamento_categorica_nulas(df, feature):
        df[feature] = np.where(df[feature].isnull(), 999, df[feature])
        return df[feature]

    for categorica in variaveis_categoricas:
        df[categorica] = aplica_tratamento_categorica_nulas(df, categorica)

    return df


def pre_processamento_continuas_nulas(df, calcula_mediana=False):
    variaveis_continuas = [
        'bd',
        'months_as_a_registered',
        'payment_plan_days', 
        'actual_amount_paid', 'actual_amount_paid_mov_avg_m6', 'actual_amount_paid_mov_max_m6', 'actual_amount_paid_mov_min_m6', 
        'num_25', 'num_25_mov_avg_m6', 'num_25_mov_max_m6', 'num_25_mov_min_m6',
        'num_50', 'num_50_mov_avg_m6', 'num_50_mov_max_m6', 'num_50_mov_min_m6', 
        'num_75', 'num_75_mov_avg_m6', 'num_75_mov_max_m6', 'num_75_mov_min_m6', 
        'num_985', 'num_985_mov_avg_m6', 'num_985_mov_max_m6', 'num_985_mov_min_m6', 
        'num_100', 'num_100_mov_avg_m6', 'num_100_mov_max_m6', 'num_100_mov_min_m6',
        'num_unq', 'num_unq_mov_avg_m6','num_unq_mov_max_m6', 'num_unq_mov_min_m6',  
        '%num_more_than_50', '%num_more_than_50_mov_avg_m6', '%num_more_than_50_mov_max_m6', '%num_more_than_50_mov_min_m6'
    ]

    def calcular_e_salvar_mediana_por_cidade(df):
        variaveis_continuas = [
            'bd',
            'months_as_a_registered',
            'payment_plan_days', 
            'actual_amount_paid', 'actual_amount_paid_mov_avg_m6', 'actual_amount_paid_mov_max_m6', 'actual_amount_paid_mov_min_m6',
            'num_25', 'num_25_mov_avg_m6', 'num_25_mov_max_m6', 'num_25_mov_min_m6',
            'num_50', 'num_50_mov_avg_m6', 'num_50_mov_max_m6', 'num_50_mov_min_m6', 
            'num_75', 'num_75_mov_avg_m6', 'num_75_mov_max_m6', 'num_75_mov_min_m6', 
            'num_985', 'num_985_mov_avg_m6', 'num_985_mov_max_m6', 'num_985_mov_min_m6', 
            'num_100', 'num_100_mov_avg_m6', 'num_100_mov_max_m6', 'num_100_mov_min_m6',
            'num_unq', 'num_unq_mov_avg_m6','num_unq_mov_max_m6', 'num_unq_mov_min_m6',  
            '%num_more_than_50', '%num_more_than_50_mov_avg_m6', '%num_more_than_50_mov_max_m6', '%num_more_than_50_mov_min_m6'
        ]
        
        cidades = df['city'].unique()
        for cidade in cidades:
            mediana_df = df[df['city'] == cidade][variaveis_continuas].median().reset_index()
            mediana_df.columns = ['variavel', 'mediana']
            mediana_df.to_excel(f"../00_DataMaster/pre_processing/mediana_city_{cidade}.xlsx", index=False)

    if calcula_mediana:
        calcular_e_salvar_mediana_por_cidade(df)
        print(f'Salvando a Mediana com dados de treinamento...')

    for cidade in df['city'].unique():
        mediana_dict = pd.read_excel(f"../00_DataMaster/pre_processing/mediana_city_{cidade}.xlsx").set_index('variavel')['mediana'].to_dict()
        for feature in variaveis_continuas:
            percentis = df[feature].describe(percentiles=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
            p1 = percentis['1%']
            p99 = percentis['99%']
            df[feature] = np.where((df[feature].isnull() | (df[feature] <= p1) | (df[feature] >= p99)), mediana_dict[feature], df[feature])
            
        return df


def woe_training(df, feature):
    churn = df.dropna().loc[df.dropna()["churn"] == 1].groupby(feature, as_index = False)["churn"].count().rename({"churn":'churn'}, axis = 1)
    sem_churn = df.dropna().loc[df.dropna()["churn"] == 0].groupby(feature, as_index = False)["churn"].count().rename({"churn":'sem_churn'}, axis = 1)

    woe = churn.merge(sem_churn, on = feature, how = 'left')
    woe['percent_churn'] = woe['churn']/woe['churn'].sum()
    woe['percent_sem_churn'] = woe['sem_churn']/woe['sem_churn'].sum()
    woe['woe'] = round(np.log(woe['percent_churn']/woe['percent_sem_churn']), 3)
    woe = woe[[feature, "woe"]]
    woe.rename({f"{feature}":"variavel"}, axis = 1)
    woe.sort_values(by = "woe", ascending = True, inplace = True)
    woe.to_excel(f"../00_DataMaster/pre_processing/target_encoder_{feature}.xlsx", index = False)
    print(f"Excel com o WOE da variável {feature} Salvo!!")


def target_encoder_woe(df):

    woe_is_auto_renew = pd.read_excel(f"../00_DataMaster/pre_processing/target_encoder_is_auto_renew.xlsx").set_index('is_auto_renew')['woe'].to_dict()
    woe_gender = pd.read_excel(f"../00_DataMaster/pre_processing/target_encoder_gender.xlsx").set_index('gender')['woe'].to_dict()
    woe_registered_via = pd.read_excel(f"../00_DataMaster/pre_processing/target_encoder_registered_via.xlsx").set_index('registered_via')['woe'].to_dict()
    woe_city = pd.read_excel(f"../00_DataMaster/pre_processing/target_encoder_city.xlsx").set_index('city')['woe'].to_dict()
    woe_payment_method_id = pd.read_excel(f"../00_DataMaster/pre_processing/target_encoder_payment_method_id.xlsx").set_index('payment_method_id')['woe'].to_dict()

    df["is_auto_renew"] = df['is_auto_renew'].map(woe_is_auto_renew)
    df["gender"] = df['gender'].map(woe_gender)
    df["registered_via"] = df['registered_via'].map(woe_registered_via).fillna(0) # CONTINGÊNCIA POIS UMA DAS CATEGORIAS TINHA 20 REGISTROS E NÃO TINHA CHURN, ENTÃO O WOE SERÁ 0 COMO CONTINGÊNCIA
    df["city"] = df['city'].map(woe_city)
    df["payment_method_id"] = df['payment_method_id'].map(woe_payment_method_id)

    return df


def analisa_correlacao(metodo, df):
    plt.figure(figsize=(30, 15))
    mask = np.triu(np.ones_like(df.corr(method=metodo), dtype=bool))
    heatmap = sns.heatmap(df.corr(method=metodo), vmin=-1, vmax=1, cmap='magma', annot=True, fmt='.1f', cbar_kws={"shrink": .8}, mask=mask)
    heatmap.set_title(f"Analisando Correlação de {metodo}")
    plt.grid(False)
    plt.box(False)
    plt.tight_layout()
    plt.grid(False)
    plt.show()


def aplica_feature_selection(df):
    def separa_feature_target(target, dados):
        x = dados.drop(target, axis = 1)
        y = dados[[target]]

        return x, y

    def remove_features_feature_importance(target, df, class_weight, threshold):
        # Separa entre Features e Target
        x, y = separa_feature_target(target, df)
        
        # Criar o modelo de Random Forest
        model = RandomForestClassifier(random_state=42, criterion='entropy', n_estimators=20, class_weight={0:1, 1:class_weight})
        
        # Treinar o modelo
        model.fit(x, y)
        
        # Obter as importâncias das features
        feature_importances = model.feature_importances_
        
        # Selecionar as features com importância maior que zero
        selected_features = list(x.columns[feature_importances > threshold])
        selected_features.append(target)
        
        feature_importance_df = pd.DataFrame({
            'feature': x.columns,
            'importance': feature_importances
        }).sort_values(by='importance', ascending=False)
        feature_importance_df = feature_importance_df.loc[feature_importance_df['importance'] > 0]
        feature_importance_df['importance'] = feature_importance_df['importance'] * 100
        
        return selected_features, feature_importance_df

    def remove_features_altamente_correlacionadas(df, variaveis_importantes_df, threshold_correlacao=0.9):
        # Filtrar variáveis com alta importância
        alta_importancia_features = variaveis_importantes_df['feature'].tolist()
        
        # Selecionar as colunas do DataFrame com as variáveis de interesse
        df_reduzido = df[alta_importancia_features]
        
        # Calcular a matriz de correlação de Spearman
        correlacoes = df_reduzido.corr(method='spearman')
        
        # Encontrar variáveis altamente correlacionadas
        alta_correlacao = np.abs(correlacoes) > threshold_correlacao
        features_para_remover = set()
        
        for i in range(len(alta_correlacao.columns)):
            for j in range(i):
                if alta_correlacao.iloc[i, j] and correlacoes.columns[j] not in features_para_remover:
                    features_para_remover.add(correlacoes.columns[i])
        
        variaveis_filtradas = [col for col in alta_importancia_features if col not in features_para_remover]
        
        return variaveis_filtradas

    # Aplicando Random Forest e selecionado feature com importância > 0
    features, feature_importances = remove_features_feature_importance('churn', df.drop(['msno', 'safra'], axis=1).copy(), 5, 0)
    feature_importances = feature_importances.loc[feature_importances['importance'] > 0]
    
    # Filtrar variáveis altamente correlacionadas e mantendo a que possui maior importância com a target dentre as correlacionadas
    variaveis_selecionadas = remove_features_altamente_correlacionadas(df, feature_importances)
    feature_importances_final = feature_importances[feature_importances['feature'].isin(variaveis_selecionadas)]

    return feature_importances_final


def aplica_pre_processamento_feature_eng_feature_selection(df):

    # Aplicando Feature Engineering
    df['months_as_a_registered'] = months_as_a_registered(df)
    df['%num_more_than_50'] = num_more_than_50(df)
    df['city'] = agrupa_categorias_cidade_pelo_woe(df)
    df['payment_method_id'] = agrupa_categorias_metodo_pagamento_pelo_woe(df)
    df = get_temporal_features(df)

    # Aplicando Pré-Processamento
    df["is_auto_renew"] = muda_tipagem_variavel(df, "is_auto_renew", "int")
    df["bd"] = muda_tipagem_variavel(df, "bd", "int")
    df["registered_via"] = muda_tipagem_variavel(df, "registered_via", "int")
    df["payment_plan_days"] = muda_tipagem_variavel(df, "payment_plan_days", "float")
    df["actual_amount_paid"] = muda_tipagem_variavel(df, "actual_amount_paid", "float")

    df = pre_processamento_categoricas_nulas(df)
    df = pre_processamento_continuas_nulas(df)

    df = target_encoder_woe(df)

    # Organizando
    variaveis_selecionadas =  [
        'is_auto_renew', 'payment_method_id', 'months_as_a_registered',
        'num_unq_mov_max_m6', 'num_100_mov_max_m6', 'num_unq_mov_min_m6',
        'num_100_mov_min_m6', '%num_more_than_50_mov_max_m6',
        '%num_more_than_50_mov_avg_m6', '%num_more_than_50_mov_min_m6',
        'num_25_mov_max_m6', 'bd', 'num_50_mov_avg_m6',
        'num_985_mov_avg_m6', 'actual_amount_paid_mov_avg_m6',
        'num_25_mov_min_m6', 'num_75_mov_max_m6', 'num_50_mov_min_m6',
        'num_985_mov_min_m6', 'city', 'num_75_mov_min_m6',
        'registered_via', 'payment_plan_days',
        ]
    mensalidade = ['actual_amount_paid']
    safra = ["safra"]
    target = ["churn"]
    user_id = ["msno"]

    df = df[target + user_id + safra + mensalidade + variaveis_selecionadas]

    return df


def separa_feature_target(target, dados):
    x = dados.drop(target, axis = 1)
    y = dados[[target]]

    return x, y


def train_min_max_scaler(df):

    cols = list(df.drop(['churn', 'msno', 'safra', 'actual_amount_paid'], axis = 1).columns)

    df_scaler = df[cols].copy()

    scaler = MinMaxScaler()
    scaler.fit(df_scaler)
    joblib.dump(scaler, "../00_DataMaster/models/scaler.pkl")
    print('Scaler Treinado e Salvo com sucesso!')


def Classificador(classificador, x_train, y_train, x_test, y_test, class_weight):

    # Puxa o Scaler Treinado com os dados de Treino
    scaler = joblib.load("../00_DataMaster/models/scaler.pkl")
    
    cols = list(x_train.drop(['msno', 'safra', 'actual_amount_paid'], axis = 1).columns)

    x_train = x_train[cols]
    x_test = x_test[cols]

    # Define as colunas categóricas e numéricas
    models = {
        'Regressão Logística': make_pipeline(
            ColumnTransformer([
                ('scaler', make_pipeline(scaler), cols)
            ]),
            LogisticRegression(
                random_state=42, # Semente aleatória para reproducibilidade dos resultados
                class_weight={0: 1, 1: class_weight}, # Peso atribuído às classes. Pode ser útil para lidar com conjuntos de dados desbalanceados.
                C=1, # Parâmetro de regularização inversa. Controla a força da regularização.
                penalty='l2', # Tipo de regularização. 'l1', 'l2', 'elasticnet', ou 'none'.
                max_iter=50, # Número máximo de iterações para a convergência do otimizador.
                solver='liblinear' # Algoritmo de otimização. 'newton-cg', 'lbfgs', 'liblinear' (gradiente descendente), 'sag' (Stochastic gradient descent), 'saga' (Stochastic gradient descent que suporta reg L1).
                )
        ),
        'Random Forest': make_pipeline(
        RandomForestClassifier(
            random_state=42,            # Semente aleatória para reproducibilidade dos resultados
            criterion='entropy',       # Critério usado para medir a qualidade de uma divisão
            n_estimators=50,           # Número de árvores na floresta (equivalente ao n_estimators no XGBoost)
            max_depth = 6,                # Profundidade máxima de cada árvore
            class_weight={0:1, 1:class_weight},  # Peso das classes em casos desequilibrados
            bootstrap=True               # Se deve ou não amostrar com substituição ao construir árvores
            )
        ),
        'XGBoost': make_pipeline(
        XGBClassifier(
            random_state=42,            # Semente aleatória para reproducibilidade dos resultados
            tree_method = 'gpu_hist',
            n_estimators=50,           # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
            max_depth = 4,                # Profundidade máxima de cada árvore
            learning_rate = 0.01,         # Taxa de aprendizado - controla a contribuição de cada árvore
            eval_metric='logloss',      # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
            objective='binary:logistic',# Define o objetivo do modelo, 'binary:logistic' para classificação binária
            scale_pos_weight=class_weight,  # Peso das classes positivas em casos desequilibrados
            reg_alpha=1,                # Termo de regularização L1 (penalidade nos pesos)
            reg_lambda=1,               # Termo de regularização L2 (penalidade nos quadrados dos pesos)
            gamma=1,                    # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
            colsample_bytree=0.5,       # Fração de características a serem consideradas ao construir cada árvore --> 0.5 significa que 50% das features (seleção aleatória) será considerada
            subsample=0.5,              # Fração de amostras a serem usadas para treinar cada árvore --> 0.5 significa que 50% da amostra de treino (seleção aleatória) será considerada
            )
        )
    }

    if classificador in models:
        model = models[classificador]
    else:
        print('Utilize Regressão Logística, Random Forest ou XGBoost como opções de Classificadores!')

    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    y_proba_train = model.predict_proba(x_train)
    y_proba_test = model.predict_proba(x_test)

    return model, y_pred_train, y_pred_test, y_proba_train, y_proba_test


def validacao_cruzada_classificacao(classificador, df, target_column, n_splits, class_weight):

    columns_selected = [
       'is_auto_renew', 'payment_method_id', 'months_as_a_registered',
       'num_unq_mov_max_m6', 'num_100_mov_max_m6', 'num_unq_mov_min_m6',
       'num_100_mov_min_m6', '%num_more_than_50_mov_max_m6',
       '%num_more_than_50_mov_avg_m6', '%num_more_than_50_mov_min_m6',
       'num_25_mov_max_m6', 'bd', 'num_50_mov_avg_m6',
       'num_985_mov_avg_m6', 'actual_amount_paid_mov_avg_m6',
       'num_25_mov_min_m6', 'num_75_mov_max_m6', 'num_50_mov_min_m6',
       'num_985_mov_min_m6', 'city', 'num_75_mov_min_m6',
       'registered_via', 'payment_plan_days',
       'churn'
        ]
    

    df_raw = df[columns_selected].copy()

    # Inicializar o KFold para dividir os dados
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Listas para armazenar as métricas para cada fold
    accuracy_scores = [] # Lista para armazenar os valores de ACURÁCIA
    precision_scores = [] # Lista para armazenar os valores de PRECISION
    recall_scores = [] # Lista para armazenar os valores de RECALL
    f1_scores = [] # Lista para armazenar os valores de F1
    auc_scores = []  # Lista para armazenar os valores de AUC
    ks_scores = []   # Lista para armazenar os valores de KS
    logloss_scores = [] # Lista para armazenar os valores de LogLoss
    cv_results = []  # Lista para armazenar os resultados da VALIDAÇÃO CRUZADA

    # Loop pelos folds
    for train_idx, test_idx in kfold.split(df_raw):
        # Criar DataFrames de treino e teste
        df_train = df_raw.iloc[train_idx]
        df_test = df_raw.iloc[test_idx]

        # Filtragem das Features que passaram no Feature Selection
        df_train = df_train[columns_selected]
        df_test = df_test[columns_selected]

        # Separação Feature e Target
        x_train, y_train = separa_feature_target('churn', df_train)
        x_test, y_test = separa_feature_target('churn', df_test)

    # Roda Modelos
        models = {
            'Regressão Logística': make_pipeline(
                LogisticRegression(
                    random_state=42, # Semente aleatória para reproducibilidade dos resultados
                    class_weight={0: 1, 1: class_weight}, # Peso atribuído às classes. Pode ser útil para lidar com conjuntos de dados desbalanceados.
                    C=1, # Parâmetro de regularização inversa. Controla a força da regularização.
                    penalty='l2', # Tipo de regularização. 'l1', 'l2', 'elasticnet', ou 'none'.
                    max_iter=50, # Número máximo de iterações para a convergência do otimizador.
                    solver='liblinear' # Algoritmo de otimização. 'newton-cg', 'lbfgs', 'liblinear' (gradiente descendente), 'sag' (Stochastic gradient descent), 'saga' (Stochastic gradient descent que suporta reg L1).
                    )
            ),
            'Random Forest': make_pipeline(
            RandomForestClassifier(
                random_state=42,            # Semente aleatória para reproducibilidade dos resultados
                criterion='entropy',       # Critério usado para medir a qualidade de uma divisão
                n_estimators=50,           # Número de árvores na floresta (equivalente ao n_estimators no XGBoost)
                max_depth = 4,                # Profundidade máxima de cada árvore
                class_weight={0:1, 1:class_weight},  # Peso das classes em casos desequilibrados
                bootstrap=True               # Se deve ou não amostrar com substituição ao construir árvores
                )
            ),
            'XGBoost': make_pipeline(
            XGBClassifier(
                random_state=42,            # Semente aleatória para reproducibilidade dos resultados
                tree_method = 'gpu_hist',
                n_estimators=50,           # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
                max_depth = 4,                # Profundidade máxima de cada árvore
                learning_rate = 0.005,         # Taxa de aprendizado - controla a contribuição de cada árvore
                eval_metric='logloss',      # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
                objective='binary:logistic',# Define o objetivo do modelo, 'binary:logistic' para classificação binária
                scale_pos_weight=class_weight,  # Peso das classes positivas em casos desequilibrados
                reg_alpha=1,                # Termo de regularização L1 (penalidade nos pesos)
                reg_lambda=1,               # Termo de regularização L2 (penalidade nos quadrados dos pesos)
                gamma=1,                    # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
                colsample_bytree=0.5,       # Fração de características a serem consideradas ao construir cada árvore --> 0.5 significa que 50% das features (seleção aleatória) será considerada
                subsample=0.5,              # Fração de amostras a serem usadas para treinar cada árvore --> 0.5 significa que 50% da amostra de treino (seleção aleatória) será considerada
                )
            )
        }

        if classificador in models:
            model = models[classificador]
        else:
            print('Utilize Regressão Logística, Random Forest ou XGBoost como opções de Classificadores!')

        # Treinar o modelo usando os dados de treinamento
        model.fit(x_train, y_train)

        # Obter as probabilidades previstas para ambas as classes
        y_proba = model.predict_proba(x_test)

        # Fazer as previsões usando o modelo nos dados de teste
        y_pred = model.predict(x_test)

        # Calcular as métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        y_proba = model.predict_proba(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ks = max(tpr - fpr)
        logloss = log_loss(y_test, y_proba[:, 1])

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(roc_auc)
        ks_scores.append(ks)
        logloss_scores.append(logloss)

        # Adicionar resultados de validação cruzada ao DataFrame
        fold_results = pd.DataFrame({
            'churn': y_test['churn'].values,
            'y_predict': y_pred,
            'predict_proba_0': y_proba[:, 0],  # Probabilidade da classe 0
            'predict_proba_1': y_proba[:, 1]  # Probabilidade da classe 1
        })
        cv_results.append(fold_results)


    # Calcular a média das métricas para todos os folds
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    mean_auc = np.mean(auc_scores),
    mean_ks = np.mean(ks_scores)
    mean_logloss = np.mean(logloss_scores)

    # Criar um DataFrame com as métricas
    metricas_finais = pd.DataFrame({
        'Acuracia': mean_accuracy,
        'Precisao': mean_precision,
        'Recall': mean_recall,
        'F1-Score': mean_f1,
        'AUC':mean_auc,
        'KS': mean_ks,
        'LogLoss': mean_logloss,
        'Etapa': 'validacao_cruzada',
        'Classificador': classificador
    }, index=[1])

    return metricas_finais, cv_results


def metricas_classificacao(classificador, y_train, y_predict_train, y_test, y_predict_test, y_predict_proba_train, y_predict_proba_test, etapa_1, etapa_2):

    predict_proba_train = pd.DataFrame(y_predict_proba_train.tolist(), columns=['predict_proba_0', 'predict_proba_1'])
    predict_proba_test = pd.DataFrame(y_predict_proba_test.tolist(), columns=['predict_proba_0', 'predict_proba_1'])

    # Treino
    accuracy_train = accuracy_score(y_train, y_predict_train)
    precision_train = precision_score(y_train, y_predict_train)
    recall_train = recall_score(y_train, y_predict_train)
    f1_train = f1_score(y_train, y_predict_train)
    roc_auc_train = roc_auc_score(y_train['churn'], predict_proba_train['predict_proba_1'])
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train['churn'], predict_proba_train['predict_proba_1'])
    ks_train = max(tpr_train - fpr_train)
    logloss_train = log_loss(y_train['churn'], predict_proba_train['predict_proba_1'])
    metricas_treino = pd.DataFrame(
        {
            'Acuracia': accuracy_train, 
            'Precisao': precision_train, 
            'Recall': recall_train, 
            'F1-Score': f1_train, 
            'AUC': roc_auc_train, 
            'KS': ks_train, 
            'LogLoss':logloss_train,
            'Etapa': etapa_1, 
            'Classificador': classificador
        }, 
        index=[0]
    )
    
    # Teste
    accuracy_test = accuracy_score(y_test, y_predict_test)
    precision_test = precision_score(y_test, y_predict_test)
    recall_test = recall_score(y_test, y_predict_test)
    f1_test = f1_score(y_test, y_predict_test)
    roc_auc_test = roc_auc_score(y_test['churn'], predict_proba_test['predict_proba_1'])
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test['churn'], predict_proba_test['predict_proba_1'])
    ks_test = max(tpr_test - fpr_test)
    logloss_test = log_loss(y_test['churn'], predict_proba_test['predict_proba_1'])
    metricas_teste = pd.DataFrame(
        {
            'Acuracia': accuracy_test, 
            'Precisao': precision_test, 
            'Recall': recall_test, 
            'F1-Score': f1_test, 
            'AUC': roc_auc_test, 
            'KS': ks_test, 
            'LogLoss':logloss_test,
            'Etapa': etapa_2, 
            'Classificador': classificador
        }, 
        index=[0]
    )
    
    # Consolidando
    metricas_finais = pd.concat([metricas_treino, metricas_teste])

    return metricas_finais


def metricas_classificacao_modelos_juntos(lista_modelos):
    if len(lista_modelos) > 0:
        metricas_modelos = pd.concat(lista_modelos)#.set_index('Classificador')
    else:
        metricas_modelos = lista_modelos[0]
    # Redefina o índice para torná-lo exclusivo
    df = metricas_modelos.reset_index(drop=True)
    df = df.round(2)

    # Função para formatar as células com base na Etapa
    def color_etapa(val):
        color = 'black'
        if val == 'treino':
            color = 'blue'
        elif val == 'teste':
            color = 'red'
        return f'color: {color}; font-weight: bold;'

    # Função para formatar os valores com até duas casas decimais
    def format_values(val):
        if isinstance(val, (int, float)):
            return f'{val:.2f}'
        return val

    # Estilizando o DataFrame
    styled_df = df.style\
        .format(format_values)\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px', subset=pd.IndexSlice[:, :])\
        .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px', subset=pd.IndexSlice[:, 'Acuracia':'F1-Score'])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px', subset=pd.IndexSlice[:, 'Etapa'])\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    # Mostrando o DataFrame estilizado
    styled_df
    return styled_df


def otimizacao(classificador, x_train, y_train, x_test, y_test):
    cols = list(x_train.drop(['msno', 'safra', 'actual_amount_paid'], axis = 1).columns)

    # O FILLNA(0) É UMA CONTINGÊNCIA --> A VARIÁVEL 'REGISTERED_VIA' VEIO COM WOE ZERADO POIS NA AMOSTRA DE TREINO NÃO HAVIAM INADIMPLENTES, 
    # SENDO ASSIM, O REGISTED_VIA = 10 FICOU WOE NULO, MAS FORAM APENAS 2 REGISTROS, ENTÃO SEM PROBLEMAS!!!!!
    # O FILLNA(0) NÃO PREJUDICA MEU TREINAMENTO, POIS UM WOE = 0 SIGNIFICA QUE A VARIÁVEL NÃO TEM NENHUMA ASSOCIAÇÃO ENTRE A CLASSE 0 E 1, OU SEJA, ELA É NEUTRA E NÃO AFETA A DECISÃO DO MODELO
    
    x_train = x_train[cols].copy()
    x_test = x_test[cols].copy()

    # Define o modelo de XGBoost com a otimização de hiperparâmetros via BayesSearch
    model = make_pipeline(
        BayesSearchCV(
            XGBClassifier(random_state=42, tree_method = 'gpu_hist', eval_metric='logloss', objective='binary:logistic'),
            {
                'n_estimators': (50, 75, 100), # Número de Árvores construídas
                'max_depth': (4, 5, 6), # Profundidade Máxima de cada Árvore
                'learning_rate': (0.005, 0.01), # Tamanho do passo utilizado no Método do Gradiente Descendente
                'reg_alpha':(0.5, 1), # Valor do Alpha aplicado durante a Regularização Lasso L1 
                'reg_lambda':(0.5, 1), # Valor do Lambda aplicado durante a Regularização Ridge L2
                'gamma':(0.5, 1), # Valor mínimo permitido para um Nó de Árvore ser aceito. Ajuda a controlar o crescimento das Árvores, evitando divisões insignificantes
                'colsample_bytree':(0.5, 1), # Porcentagem de Colunas utilizada para a amostragem aleatória durante a criação das Árvores
                'subsample':(0.5, 1), # Porcentagem de Linhas utilizada para a amostragem aleatória durante a criação das Árvores
                'scale_pos_weight':(6, 8, 10, 12), # Peso atribuído a classe positiva, aumentando a importância da classe minoritária
            },
            n_iter=10,
            random_state=42,
            n_jobs=-1,
            scoring='roc_auc', #precision, recall, f1, roc_auc, neg_log_loss
            cv=5
        )
    )

    np.int = int # CORREÇÃO POIS O MÉTODO .fit() DA CLASSE SKOPT ESTAVA COM PROBLEMAS DEVIDO A ATUALIZAÇÃO DO NUMPY

    # Treina o modelo
    model.fit(x_train, y_train)

    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    y_proba_train = model.predict_proba(x_train)
    y_proba_test = model.predict_proba(x_test)

    melhores_hiperparametros = model.named_steps['bayessearchcv'].best_params_
    hiperparametros = pd.DataFrame([melhores_hiperparametros])

    best_hiperpams = []
    for chave, valor in melhores_hiperparametros.items():
        best_hiperpams.append([chave, valor])

    pivot = pd.DataFrame(best_hiperpams).T
    pivot.columns = pivot.iloc[0]
    pivot = pivot.drop(0)

    # Crie um DataFrame a partir dos hiperparâmetros
    df = hiperparametros.reset_index(drop=True)
    df = df.round(2)

    def color_etapa(val):
        color = 'black'
        if val == 'treino':
            color = 'blue'
        elif val == 'teste':
            color = 'red'
        return f'color: {color}; font-weight: bold;'

    # Função para formatar os valores com até duas casas decimais
    def format_values(val):
        if isinstance(val, (int, float)):
            return f'{val:.2f}'
        return val

    # Estilizando o DataFrame
    styled_df = df.style\
        .format(format_values)\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px')\
        .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px')\
        .set_table_styles([
            {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
        ])

    return model, y_pred_train, y_pred_test, y_proba_train, y_proba_test, styled_df, pivot


def validacao_cruzada_classificacao_otimizada(classificador, df, target_column, n_splits, best_hiperpams):

    columns_selected = [
            'is_auto_renew', 'payment_method_id', 'months_as_a_registered',
            'num_unq_mov_max_m6', 'num_100_mov_max_m6', 'num_unq_mov_min_m6',
            'num_100_mov_min_m6', '%num_more_than_50_mov_max_m6',
            '%num_more_than_50_mov_avg_m6', '%num_more_than_50_mov_min_m6',
            'num_25_mov_max_m6', 'bd', 'num_50_mov_avg_m6',
            'num_985_mov_avg_m6', 'actual_amount_paid_mov_avg_m6',
            'num_25_mov_min_m6', 'num_75_mov_max_m6', 'num_50_mov_min_m6',
            'num_985_mov_min_m6', 'city', 'num_75_mov_min_m6',
            'registered_via', 'payment_plan_days',
            'churn'
        ]
    

    df_raw = df[columns_selected].copy()

    # Inicializar o KFold para dividir os dados
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Listas para armazenar as métricas para cada fold
    accuracy_scores = [] # Lista para armazenar os valores de ACURÁCIA
    precision_scores = [] # Lista para armazenar os valores de PRECISION
    recall_scores = [] # Lista para armazenar os valores de RECALL
    f1_scores = [] # Lista para armazenar os valores de F1
    auc_scores = []  # Lista para armazenar os valores de AUC
    ks_scores = []   # Lista para armazenar os valores de KS
    logloss_scores = [] # Lista para armazenar os valores de LogLoss
    cv_results = []  # Lista para armazenar os resultados da VALIDAÇÃO CRUZADA

    # Loop pelos folds
    for train_idx, test_idx in kfold.split(df_raw):
        # Criar DataFrames de treino e teste
        df_train = df_raw.iloc[train_idx]
        df_test = df_raw.iloc[test_idx]

        # Filtragem das Features que passaram no Feature Selection
        df_train = df_train[columns_selected]
        df_test = df_test[columns_selected]

        # Separação Feature e Target
        x_train, y_train = separa_feature_target('churn', df_train)
        x_test, y_test = separa_feature_target('churn', df_test)

        # Melhores Hiperparâmetros
        melhores_hiperparametros = best_hiperpams
        colsample_bytree = round(melhores_hiperparametros['colsample_bytree'][1], 2)
        gamma = round(melhores_hiperparametros['gamma'][1], 2)
        learning_rate = round(melhores_hiperparametros['learning_rate'][1], 2)
        max_depth = int(round(melhores_hiperparametros['max_depth'][1], 2))
        n_estimators = int(round(melhores_hiperparametros['n_estimators'][1], 2))
        reg_alpha = round(melhores_hiperparametros['reg_alpha'][1], 2)
        reg_lambda = round(melhores_hiperparametros['reg_lambda'][1], 2)
        scale_pos_weight = int(round(melhores_hiperparametros['scale_pos_weight'][1], 2))
        subsample = round(melhores_hiperparametros['subsample'][1], 2)

        # Roda Modelo
        model = make_pipeline(
            XGBClassifier(
                random_state=42,            # Semente aleatória para reproducibilidade dos resultados
                tree_method = 'gpu_hist',
                n_estimators=n_estimators,           # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
                max_depth = max_depth,                # Profundidade máxima de cada árvore
                learning_rate = learning_rate,         # Taxa de aprendizado - controla a contribuição de cada árvore
                eval_metric='logloss',      # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
                objective='binary:logistic',# Define o objetivo do modelo, 'binary:logistic' para classificação binária
                scale_pos_weight=scale_pos_weight,  # Peso das classes positivas em casos desequilibrados
                reg_alpha=reg_alpha,                # Termo de regularização L1 (penalidade nos pesos)
                reg_lambda=reg_lambda,               # Termo de regularização L2 (penalidade nos quadrados dos pesos)
                gamma=gamma,                    # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
                colsample_bytree=colsample_bytree,       # Fração de características a serem consideradas ao construir cada árvore
                subsample=subsample,              # Fração de amostras a serem usadas para treinar cada árvore
                )
            )

        # Treinar o modelo usando os dados de treinamento
        model.fit(x_train, y_train)

        # Obter as probabilidades previstas para ambas as classes
        y_proba = model.predict_proba(x_test)

        # Fazer as previsões usando o modelo nos dados de teste
        y_pred = model.predict(x_test)

        # Calcular as métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        y_proba = model.predict_proba(x_test)
        fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        ks = max(tpr - fpr)
        logloss = log_loss(y_test, y_proba[:, 1])

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        auc_scores.append(roc_auc)
        ks_scores.append(ks)
        logloss_scores.append(logloss)

        # Adicionar resultados de validação cruzada ao DataFrame
        fold_results = pd.DataFrame({
            'churn': y_test['churn'].values,
            'y_predict': y_pred,
            'predict_proba_0': y_proba[:, 0],  # Probabilidade da classe 0
            'predict_proba_1': y_proba[:, 1]  # Probabilidade da classe 1
        })
        cv_results.append(fold_results)


    # Calcular a média das métricas para todos os folds
    mean_accuracy = np.mean(accuracy_scores)
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    mean_auc = np.mean(auc_scores),
    mean_ks = np.mean(ks_scores)
    mean_logloss = np.mean(logloss_scores)

    # Criar um DataFrame com as métricas
    metricas_finais = pd.DataFrame({
        'Acuracia': mean_accuracy,
        'Precisao': mean_precision,
        'Recall': mean_recall,
        'F1-Score': mean_f1,
        'AUC':mean_auc,
        'KS': mean_ks,
        'LogLoss': mean_logloss,
        'Etapa': 'validacao_cruzada',
        'Classificador': classificador
    }, index=[1])

    return metricas_finais, cv_results


def define_ponto_de_corte(x_train, y_train, x_test, y_test, best_hiperpams):

    df_threshold = pd.concat([x_train, y_train], axis=1).copy()
    cols = list(df_threshold.drop(['churn', 'msno', 'safra', 'actual_amount_paid'], axis=1).columns)

    x = df_threshold[cols].copy()
    y = df_threshold['churn'].copy()

    # Melhores Hiperparâmetros
    melhores_hiperparametros = best_hiperpams
    colsample_bytree = round(melhores_hiperparametros['colsample_bytree'][1], 2)
    gamma = round(melhores_hiperparametros['gamma'][1], 2)
    learning_rate = round(melhores_hiperparametros['learning_rate'][1], 2)
    max_depth = int(round(melhores_hiperparametros['max_depth'][1], 2))
    n_estimators = int(round(melhores_hiperparametros['n_estimators'][1], 2))
    reg_alpha = round(melhores_hiperparametros['reg_alpha'][1], 2)
    reg_lambda = round(melhores_hiperparametros['reg_lambda'][1], 2)
    scale_pos_weight = int(round(melhores_hiperparametros['scale_pos_weight'][1], 2))
    subsample = round(melhores_hiperparametros['subsample'][1], 2)


    model = make_pipeline(
        XGBClassifier(
            random_state=42,            # Semente aleatória para reproducibilidade dos resultados
            tree_method = 'gpu_hist',
            n_estimators=n_estimators,           # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
            max_depth = max_depth,                # Profundidade máxima de cada árvore
            learning_rate = learning_rate,         # Taxa de aprendizado - controla a contribuição de cada árvore
            eval_metric='logloss',      # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
            objective='binary:logistic',# Define o objetivo do modelo, 'binary:logistic' para classificação binária
            scale_pos_weight=scale_pos_weight,  # Peso das classes positivas em casos desequilibrados
            reg_alpha=reg_alpha,                # Termo de regularização L1 (penalidade nos pesos)
            reg_lambda=reg_lambda,               # Termo de regularização L2 (penalidade nos quadrados dos pesos)
            gamma=gamma,                    # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
            colsample_bytree=colsample_bytree,       # Fração de características a serem consideradas ao construir cada árvore
            subsample=subsample,              # Fração de amostras a serem usadas para treinar cada árvore
        )
    )

    # Treina o modelo de classificação
    model.fit(x, y)

    def calculate_metrics(x, y, model):

        def retorno_financeiro(df_modelo, y_predict):

            df_aux = df_modelo.loc[df_modelo['safra'].isin(['201603', '201604', '201605', '201606','201607', '201608', '201609'])].copy()
            df_aux['y_predict'] = y_predict

            TN = df_aux.loc[(df_aux['churn'] == 0) & (df_aux['y_predict'] == 0)].shape[0] # O CARA NÃO É CHURN E MEU MODELO FALA QUE ELE NÃO É CHURN
            FN = df_aux.loc[(df_aux['churn'] == 1) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É CHURN E MEU MODELO FALA QUE ELE NÃO É CHURN
            FP = df_aux.loc[(df_aux['churn'] == 0) & (df_aux['y_predict'] == 1)].shape[0] # O CARA NÃO É CHURN E MEU MODELO FALA QUE ELE É CHURN
            TP = df_aux.loc[(df_aux['churn'] == 1) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É CHURN E O MEU MODELO FALA QUE ELE É CHURN
        

            df_aux['retorno_financeiro'] = (
            np.where((df_aux['churn'] == 0) & (df_aux['y_predict'] == 0), 0, # Não sofre nenhuma medida, então não temos retorno nem custo
            np.where((df_aux['churn'] == 1) & (df_aux['y_predict'] == 0), 0, # Embora não tenhamos identificado que era CHURN, não oferecemos nenhum serviço e portanto não houve custo
            np.where((df_aux['churn'] == 0) & (df_aux['y_predict'] == 1), 3*df_aux['actual_amount_paid'], # Implementamos a ação incorretamente, logo, estaremos fornecendo 3 meses de assinatura grátis e tendo custo
            np.where((df_aux['churn'] == 1) & (df_aux['y_predict'] == 1), 9*df_aux['actual_amount_paid'], # Implementamos a ação corretamente, logo, estaremos retendo 50% desses casos e garantindo a assinatura deles por mais 3 meses
            0 # Não ganho nada
            )))))

            quantidade_de_clientes_retidos = 0.5*TP
            taxa_de_clientes_retidos = round((0.5*TP)/(FN+TP)*100, 2) # O RECALL FORNECE A QUANTIDADE DE CLIENTES RETIDOS, MAS PRECISAMOS DIVIDIR O VALOR POR 2 POR CONTA DO ENUNCIADO
            retorno_financeiro_acao_correta = (
                df_aux.loc[
                    (df_aux['churn'] == 1) & (df_aux['y_predict'] == 1)
                ]
                ['retorno_financeiro'].sum()
            )*0.5

            retorno_financeiro_acao_incorreta = (
                df_aux.loc[
                    (df_aux['churn'] == 0) & (df_aux['y_predict'] == 1)
                ]
                ['retorno_financeiro'].sum()
            )

            retorno_financeiro = round(retorno_financeiro_acao_correta - retorno_financeiro_acao_incorreta, 0)


            return quantidade_de_clientes_retidos, taxa_de_clientes_retidos, retorno_financeiro

        df_threshold = pd.concat([x, y], axis=1).copy()
        cols = list(df_threshold.drop(['churn', 'msno', 'safra', 'actual_amount_paid'], axis=1).columns)

        x = df_threshold[cols].copy()
        y = df_threshold['churn'].copy()

        y_pred = model.predict(x)
        y_predict_proba = model.predict_proba(x)[:, 1]

        df_threshold = df_threshold[['churn', 'msno', 'safra', 'actual_amount_paid']]
        df_threshold['Proba Churn'] = y_predict_proba

        list_threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        clientes_retidos_scores = []
        taxa_clientes_retidos_scores = []
        retorno_financeiro_scores = []

        for threshold in list_threshold:
            df_threshold['y_predict_threshold'] = np.where(df_threshold['Proba Churn'] <= threshold, 0, 1)

            quantidade_de_clientes_retidos, taxa_de_clientes_retidos, retorno_financeiro_threshold= retorno_financeiro(df_threshold, df_threshold['y_predict_threshold'])

            clientes_retidos_scores.append(quantidade_de_clientes_retidos)
            taxa_clientes_retidos_scores.append(taxa_de_clientes_retidos)
            retorno_financeiro_scores.append(retorno_financeiro_threshold)

        metrics_df = pd.DataFrame({
            'Threshold': list_threshold,
            'Clientes Retidos': clientes_retidos_scores,
            'Taxa de Clientes Retidos': taxa_clientes_retidos_scores,
            'Retorno Financeiro': retorno_financeiro_scores
        })

        return metrics_df

    metrics_train = calculate_metrics(x_train, y_train, model)
    metrics_test = calculate_metrics(x_test, y_test, model)

    best_threshold_train = metrics_train.loc[metrics_train['Retorno Financeiro'].idxmax(), 'Threshold']
    best_return_train = metrics_train['Retorno Financeiro'].max()
    
    best_threshold_test = metrics_test.loc[metrics_test['Retorno Financeiro'].idxmax(), 'Threshold']
    best_return_test = metrics_test['Retorno Financeiro'].max()

    sns.set(style="whitegrid", font_scale=1.2)
    plt.figure(figsize=(20, 6))

    plt.subplot(1, 2, 1)
    plt.plot(metrics_train['Threshold'], metrics_train['Retorno Financeiro'], marker='o', label='Retorno Financeiro', color='green')
    plt.annotate(f'Melhor Retorno: R${int(best_return_train)}', 
                xy=(best_threshold_train, best_return_train), 
                xytext=(best_threshold_train + 0.1, best_return_train + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12,
                color='black')
    plt.title("Métricas vs Thresholds (Treino)", fontsize=16)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Métricas', fontsize=14)
    plt.xticks(metrics_train['Threshold'], rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(metrics_test['Threshold'], metrics_test['Retorno Financeiro'], marker='o', label='Retorno Financeiro', color='green')
    plt.annotate(f'Melhor Retorno: R${int(best_return_test)}', 
                xy=(best_threshold_test, best_return_test), 
                xytext=(best_threshold_test + 0.1, best_return_test + 0.1),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=12,
                color='black')
    plt.title("Métricas vs Thresholds (Validação)", fontsize=16)
    plt.xlabel('Threshold', fontsize=14)
    plt.ylabel('Métricas', fontsize=14)
    plt.xticks(metrics_test['Threshold'], rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()

    plt.tight_layout()
    plt.show()


def modelo_classificador_churn_oficial(df_train, df_test, df_oot, opcao, best_hiperpams):

    base_train = df_train.copy()
    base_test = df_test.copy()
    base_oot = df_oot.copy()

    # Prepara as Amostras
    base_train['tipo_amostra'] = 'train'
    base_test['tipo_amostra'] = 'test'
    base_oot['tipo_amostra'] = 'oot'
    df = pd.concat([base_train, base_test, base_oot])

    # Prepara DataFrame para Treinamento ou Escoragem
    cols = list(df.drop(['churn', 'msno', 'safra', 'actual_amount_paid', 'tipo_amostra'], axis=1).columns)

    # Treina e Salva o Modelo
    if opcao == 'salvar':

        df_model = df.loc[(df['tipo_amostra'] == 'train')].copy()

        x_model = df_model[cols].copy()
        y_model = df_model['churn'].copy()

        # Define o modelo de XGBoost com a otimização de hiperparâmetros via BayesSearch + Calibração de Probabilidade
        # Melhores Hiperparâmetros
        melhores_hiperparametros = best_hiperpams
        colsample_bytree = round(melhores_hiperparametros['colsample_bytree'][1], 2)
        gamma = round(melhores_hiperparametros['gamma'][1], 2)
        learning_rate = round(melhores_hiperparametros['learning_rate'][1], 2)
        max_depth = int(round(melhores_hiperparametros['max_depth'][1], 2))
        n_estimators = int(round(melhores_hiperparametros['n_estimators'][1], 2))
        reg_alpha = round(melhores_hiperparametros['reg_alpha'][1], 2)
        reg_lambda = round(melhores_hiperparametros['reg_lambda'][1], 2)
        scale_pos_weight = int(round(melhores_hiperparametros['scale_pos_weight'][1], 2))
        subsample = round(melhores_hiperparametros['subsample'][1], 2)

        model = make_pipeline(
            XGBClassifier(
                random_state=42,            # Semente aleatória para reproducibilidade dos resultados
                tree_method = 'gpu_hist',
                n_estimators=n_estimators,           # Número de árvores no modelo (equivalente ao n_estimators na Random Forest)
                max_depth = max_depth,                # Profundidade máxima de cada árvore
                learning_rate = learning_rate,         # Taxa de aprendizado - controla a contribuição de cada árvore
                eval_metric='logloss',      # Métrica de avaliação durante o treinamento, 'logloss' é comum para problemas de classificação binária
                objective='binary:logistic',# Define o objetivo do modelo, 'binary:logistic' para classificação binária
                scale_pos_weight=scale_pos_weight,  # Peso das classes positivas em casos desequilibrados
                reg_alpha=reg_alpha,                # Termo de regularização L1 (penalidade nos pesos)
                reg_lambda=reg_lambda,               # Termo de regularização L2 (penalidade nos quadrados dos pesos)
                gamma=gamma,                    # Controle de poda da árvore, maior gamma leva a menos crescimento da árvore
                colsample_bytree=colsample_bytree,       # Fração de características a serem consideradas ao construir cada árvore
                subsample=subsample,              # Fração de amostras a serem usadas para treinar cada árvore
            )
        )

        # Treina o modelo de classificação
        model.fit(x_model, y_model)

        joblib.dump(model, "../00_DataMaster/models/classificador_churn.pkl")

        return print('Modelo de Churn Treinado e Salvo com Sucesso!')

    else:
        # Carrega o Classificador e Escora para as bases de Teste e OOT
        classificador_churn = joblib.load("../00_DataMaster/models/classificador_churn.pkl")
        df_scoring = df.loc[df['tipo_amostra'].isin(['test', 'oot'])].copy()
        df_scoring['churn_predict'] = classificador_churn.predict(df_scoring[cols])
        df_scoring['churn_predict_proba_0'] = classificador_churn.predict_proba(df_scoring[cols])[:, 0]
        df_scoring['churn_predict_proba_1'] = classificador_churn.predict_proba(df_scoring[cols])[:, 1]
        df_scoring['churn_predict_calib'] = np.where(df_scoring['churn_predict_proba_1'] <= 0.7, 0, 1)
        df_scoring = df_scoring[['tipo_amostra', 'msno', 'safra', 'actual_amount_paid', 'churn', 'churn_predict', 'churn_predict_calib', 'churn_predict_proba_0', 'churn_predict_proba_1']]

        return df_scoring


def metricas_estabilidade_final(classificador, df):

    def metricas_classificacao_modelos_juntos(lista_modelos):
        if len(lista_modelos) > 0:
            metricas_modelos = pd.concat(lista_modelos)#.set_index('Classificador')
        else:
            metricas_modelos = lista_modelos[0]
        # Redefina o índice para torná-lo exclusivo
        df = metricas_modelos.reset_index(drop=True)
        df = df.round(2)

        # Função para formatar as células com base na Etapa
        def color_etapa(val):
            color = 'black'
            if val == 'treino':
                color = 'blue'
            elif val == 'teste':
                color = 'red'
            return f'color: {color}; font-weight: bold;'

        # Função para formatar os valores com até duas casas decimais
        def format_values(val):
            if isinstance(val, (int, float)):
                return f'{val:.2f}'
            return val

        # Estilizando o DataFrame
        styled_df = df.style\
            .format(format_values)\
            .applymap(lambda x: 'color: black; font-weight: bold; background-color: white; font-size: 14px', subset=pd.IndexSlice[:, :])\
            .applymap(color_etapa, subset=pd.IndexSlice[:, :])\
            .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px', subset=pd.IndexSlice[:, 'Acuracia':'F1-Score'])\
            .applymap(lambda x: 'color: black; font-weight: bold; background-color: #white; font-size: 14px', subset=pd.IndexSlice[:, 'Etapa'])\
            .set_table_styles([
                {'selector': 'thead', 'props': [('color', 'black'), ('font-weight', 'bold'), ('background-color', 'lightgray')]}
            ])

        # Mostrando o DataFrame estilizado
        styled_df
        return styled_df

    def retorno_financeiro(df_modelo, y_predict):
        df_aux = df_modelo.loc[df_modelo['safra'].isin(['201603', '201604', '201605', '201606', '201607', '201608', '201609'])].copy()
        df_aux['y_predict'] = y_predict

        TN = df_aux.loc[(df_aux['churn'] == 0) & (df_aux['y_predict'] == 0)].shape[0] # O CARA NÃO É CHURN E MEU MODELO FALA QUE ELE NÃO É CHURN
        FN = df_aux.loc[(df_aux['churn'] == 1) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É CHURN E MEU MODELO FALA QUE ELE NÃO É CHURN
        FP = df_aux.loc[(df_aux['churn'] == 0) & (df_aux['y_predict'] == 1)].shape[0] # O CARA NÃO É CHURN E MEU MODELO FALA QUE ELE É CHURN
        TP = df_aux.loc[(df_aux['churn'] == 1) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É CHURN E O MEU MODELO FALA QUE ELE É CHURN

        df_aux['retorno_financeiro'] = (
            np.where((df_aux['churn'] == 0) & (df_aux['y_predict'] == 0), 0, # Não sofre nenhuma medida, então não temos retorno nem custo
            np.where((df_aux['churn'] == 1) & (df_aux['y_predict'] == 0), 0, # Embora não tenhamos identificado que era CHURN, não oferecemos nenhum serviço e portanto não houve custo
            np.where((df_aux['churn'] == 0) & (df_aux['y_predict'] == 1), 3*df_aux['actual_amount_paid'], # Implementamos a ação incorretamente, logo, estaremos fornecendo 3 meses de assinatura grátis e tendo custo
            np.where((df_aux['churn'] == 1) & (df_aux['y_predict'] == 1), 9*df_aux['actual_amount_paid'], # Implementamos a ação corretamente, logo, estaremos retendo 50% desses casos e garantindo a assinatura deles por mais 3 meses
            0 # Não ganho nada
        )))))

        quantidade_de_clientes_retidos = 0.5*TP
        taxa_de_clientes_retidos = round((0.5*TP)/(FN+TP)*100, 2) # O RECALL FORNECE A QUANTIDADE DE CLIENTES RETIDOS, MAS PRECISAMOS DIVIDIR O VALOR POR 2 POR CONTA DO ENUNCIADO
        retorno_financeiro_acao_correta = (
            df_aux.loc[
                (df_aux['churn'] == 1) & (df_aux['y_predict'] == 1)
            ]
            ['retorno_financeiro'].sum()
        )*0.5

        retorno_financeiro_acao_incorreta = (
            df_aux.loc[
                (df_aux['churn'] == 0) & (df_aux['y_predict'] == 1)
            ]
            ['retorno_financeiro'].sum()
        )

        retorno_financeiro = round(retorno_financeiro_acao_correta - retorno_financeiro_acao_incorreta, 0)

        return quantidade_de_clientes_retidos, taxa_de_clientes_retidos, retorno_financeiro

    df_estabilidade = df.copy()

    # Teste
    df_estabilidade_teste = df_estabilidade.loc[df_estabilidade['safra'] != '201609'].copy()
    accuracy = accuracy_score(df_estabilidade['churn'], df_estabilidade['churn_predict_calib'])
    precision = precision_score(df_estabilidade['churn'], df_estabilidade['churn_predict_calib'])
    recall = recall_score(df_estabilidade['churn'], df_estabilidade['churn_predict_calib'])
    f1 = f1_score(df_estabilidade['churn'], df_estabilidade['churn_predict_calib'])
    roc_auc = roc_auc_score(df_estabilidade['churn'], df_estabilidade['churn_predict_proba_1'])
    fpr, tpr, thresholds = roc_curve(df_estabilidade['churn'], df_estabilidade['churn_predict_proba_1'])
    ks = max(tpr - fpr)
    logloss = log_loss(df_estabilidade['churn'], df_estabilidade['churn_predict_proba_1'])
    metricas_teste = pd.DataFrame(
        {
            'Acuracia': accuracy, 
            'Precisao': precision, 
            'Recall': recall, 
            'F1-Score': f1, 
            'AUC': roc_auc, 
            'KS': ks, 
            'LogLoss':logloss,
            'Etapa': 'Teste', 
            'Classificador': 'Modelo Final'
        }, 
        index=[0]
    )

    # OOT
    df_estabilidade_oot = df_estabilidade.loc[df_estabilidade['safra'] == '201609'].copy()
    accuracy = accuracy_score(df_estabilidade['churn'], df_estabilidade['churn_predict_calib'])
    precision = precision_score(df_estabilidade['churn'], df_estabilidade['churn_predict_calib'])
    recall = recall_score(df_estabilidade['churn'], df_estabilidade['churn_predict_calib'])
    f1 = f1_score(df_estabilidade['churn'], df_estabilidade['churn_predict_calib'])
    roc_auc = roc_auc_score(df_estabilidade['churn'], df_estabilidade['churn_predict_proba_1'])
    fpr, tpr, thresholds = roc_curve(df_estabilidade['churn'], df_estabilidade['churn_predict_proba_1'])
    ks = max(tpr - fpr)
    logloss = log_loss(df_estabilidade['churn'], df_estabilidade['churn_predict_proba_1'])
    metricas_oot = pd.DataFrame(
        {
            'Acuracia': accuracy, 
            'Precisao': precision, 
            'Recall': recall, 
            'F1-Score': f1, 
            'AUC': roc_auc, 
            'KS': ks, 
            'LogLoss':logloss,
            'Etapa': 'OOT', 
            'Classificador': 'Modelo Final'
        }, 
        index=[0]
    )

    metricas_teste_oot = metricas_classificacao_modelos_juntos([metricas_teste, metricas_oot])

    display(metricas_teste_oot)
    
    # Estabilidade
    safras = ['201603', '201604', '201605', '201606', '201607', '201608', '201609']
    metrics = {'safra': [], 'AUC': [], 'Clientes Churn Retidos': []}

    retorno_financeiro_total = 0  # Inicializa o total do retorno financeiro

    for safra in safras:
        df_safras = df.loc[df['safra'] == safra].copy()
        y_true = df_safras['churn']
        y_predict = df_safras['churn_predict_calib']
        y_predict_proba_1 = df_safras['churn_predict_proba_1'].values

        auc = roc_auc_score(y_true, y_predict_proba_1)
        fpr, tpr, _ = roc_curve(y_true, y_predict_proba_1)
        ks = max(tpr - fpr)
        precision = precision_score(y_true, y_predict)

        quantidade_de_clientes_retidos, taxa_clientes_retidos, retorno_financeiro_calculado = retorno_financeiro(df_safras, y_predict)

        metrics['safra'].append(safra)
        metrics['AUC'].append(auc)
        metrics['Clientes Churn Retidos'].append(taxa_clientes_retidos)

        retorno_financeiro_total += retorno_financeiro_calculado

    metrics_df = pd.DataFrame(metrics)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18, 6))

    # Gráfico das métricas
    ax1.plot(metrics_df['safra'], metrics_df['AUC'] * 100, marker='o', linestyle='-', color='blue', label='AUC')
    ax1.plot(metrics_df['safra'], metrics_df['Clientes Churn Retidos'], marker='o', linestyle='-', color='green', label='Clientes Churn Retidos (%)')

    for i in range(len(metrics_df)):
        ax1.annotate(f'{metrics_df["AUC"].iloc[i] * 100:.2f}', 
                     (metrics_df['safra'].iloc[i], metrics_df['AUC'].iloc[i] * 100), 
                     textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, color='blue')

        ax1.annotate(f'{metrics_df["Clientes Churn Retidos"].iloc[i]:.2f}%', 
                     (metrics_df['safra'].iloc[i], metrics_df['Clientes Churn Retidos'].iloc[i]), 
                     textcoords="offset points", xytext=(0,5), ha='center', fontsize=9, color='green')

    ax1.set_title(f'AUC e Taxa de Clientes Churn Retidos por Safra ({classificador})')
    ax1.set_xlabel('Safra')
    ax1.set_ylabel('Valor')
    ax1.set_ylim(0, 100)
    ax1.legend(loc='lower right', fontsize='small')
    ax1.grid(True)
    ax1.set_xticks(metrics_df['safra'])
    ax1.set_xticklabels(metrics_df['safra'], rotation=45)

    # Gráfico do retorno financeiro total
    retorno_financeiro_total = round(retorno_financeiro_total, 0)
    ax2.text(0.5, 0.5, f'Retorno Financeiro (50% dos VP):\nR${retorno_financeiro_total:.0f}', 
             fontsize=20, ha='center', va='center', color='black', bbox=dict(facecolor='lightgray', alpha=0.5))
    ax2.set_title('Retorno Financeiro Total')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def plot_shap(model, X, titulo):
    # Pega o modelo dentro do pipeline
    model_lgbm = model.named_steps['xgbclassifier']
    
    # Cria o objeto explainer
    explainer = shap.Explainer(model_lgbm, X)
    
    # Calcula os valores SHAP
    shap_values = explainer(X)
    
    # Cria os subplots
    fig, axes = plt.subplots(1, 2, figsize=(16,6))
    
    # Beeswarm
    plt.sca(axes[0])
    shap.plots.beeswarm(shap_values, show=False)
    #axes[0].set_title("SHAP Beeswarm", fontsize=14)
    axes[0].tick_params(axis='y', labelsize=8)
    axes[0].grid(False)
    
    # Bar plot
    plt.sca(axes[1])
    shap.plots.bar(shap_values, show=False)
    #axes[1].set_title("Importância Média Absoluta", fontsize=14)
    axes[1].grid(False)
    axes[1].set_yticklabels([])
    # Título geral
    fig.suptitle(titulo, fontsize=16)
    
    plt.tight_layout()
    plt.show()


def retorno_financeiro(df_modelo, y_predict):
    df_aux = df_modelo.loc[df_modelo['safra'].isin(['201603', '201604', '201605', '201606', '201607', '201608', '201609'])].copy()
    df_aux['y_predict'] = y_predict

    TN = df_aux.loc[(df_aux['churn'] == 0) & (df_aux['y_predict'] == 0)].shape[0] # O CARA NÃO É CHURN E MEU MODELO FALA QUE ELE NÃO É CHURN
    FN = df_aux.loc[(df_aux['churn'] == 1) & (df_aux['y_predict'] == 0)].shape[0] # O CARA É CHURN E MEU MODELO FALA QUE ELE NÃO É CHURN
    FP = df_aux.loc[(df_aux['churn'] == 0) & (df_aux['y_predict'] == 1)].shape[0] # O CARA NÃO É CHURN E MEU MODELO FALA QUE ELE É CHURN
    TP = df_aux.loc[(df_aux['churn'] == 1) & (df_aux['y_predict'] == 1)].shape[0] # O CARA É CHURN E O MEU MODELO FALA QUE ELE É CHURN

    df_aux['retorno_financeiro'] = (
        np.where((df_aux['churn'] == 0) & (df_aux['y_predict'] == 0), 0, # Não sofre nenhuma medida, então não temos retorno nem custo
        np.where((df_aux['churn'] == 1) & (df_aux['y_predict'] == 0), 0, # Embora não tenhamos identificado que era CHURN, não oferecemos nenhum serviço e portanto não houve custo
        np.where((df_aux['churn'] == 0) & (df_aux['y_predict'] == 1), 3*df_aux['actual_amount_paid'], # Implementamos a ação incorretamente, logo, estaremos fornecendo 3 meses de assinatura grátis e tendo custo
        np.where((df_aux['churn'] == 1) & (df_aux['y_predict'] == 1), 12*df_aux['actual_amount_paid'], # Implementamos a ação corretamente, logo, estaremos retendo 50% desses casos e garantindo a assinatura deles por mais 3 meses
        0 # Não ganho nada
    )))))

    quantidade_de_clientes_retidos = 0.5*TP
    taxa_de_clientes_retidos = round((0.5*TP)/(FN+TP)*100, 2) # O RECALL FORNECE A QUANTIDADE DE CLIENTES RETIDOS, MAS PRECISAMOS DIVIDIR O VALOR POR 2 POR CONTA DO ENUNCIADO
    retorno_financeiro_acao_correta = (
        df_aux.loc[
            (df_aux['churn'] == 1) & (df_aux['y_predict'] == 1)
        ]
        ['retorno_financeiro'].sum()
    )*0.5

    retorno_financeiro_acao_incorreta = (
        df_aux.loc[
            (df_aux['churn'] == 0) & (df_aux['y_predict'] == 1)
        ]
        ['retorno_financeiro'].sum()
    )

    retorno_financeiro = round(retorno_financeiro_acao_correta - retorno_financeiro_acao_incorreta, 0)


def escoragem(df):

    # Prepara DataFrame para Treinamento ou Escoragem
    cols = list(df.drop(['churn', 'msno', 'safra', 'actual_amount_paid'], axis=1).columns)

    # Carrega o Classificador e Escora para as bases de Teste e OOT
    classificador_churn = joblib.load("../00_DataMaster/models/classificador_churn.pkl")
    df_scoring = df.copy()
    df_scoring['churn_predict'] = classificador_churn.predict(df_scoring[cols])
    df_scoring['churn_predict_proba_0'] = classificador_churn.predict_proba(df_scoring[cols])[:, 0]
    df_scoring['churn_predict_proba_1'] = classificador_churn.predict_proba(df_scoring[cols])[:, 1]
    df_scoring['churn_predict_calib'] = np.where(df_scoring['churn_predict_proba_1'] <= 0.7, 0, 1)
    df_scoring = df_scoring[['msno', 'safra', 'churn', 'actual_amount_paid', 'churn_predict_calib', 'churn_predict_proba_0', 'churn_predict_proba_1'] + cols]

    return df_scoring


def analise_cluster(df):

    features = [
       'num_unq_mov_max_m6',
       'num_100_mov_max_m6', 'num_unq_mov_min_m6', 'num_100_mov_min_m6',
       '%num_more_than_50_mov_max_m6', '%num_more_than_50_mov_avg_m6',
       '%num_more_than_50_mov_min_m6', 'num_25_mov_max_m6', 'num_50_mov_avg_m6', 'num_985_mov_avg_m6', 'num_25_mov_min_m6',
       'num_75_mov_max_m6', 'num_50_mov_min_m6', 'num_985_mov_min_m6', 'num_75_mov_min_m6', 'months_as_a_registered', 'payment_method_id', 'bd', 'city'
    ]

    # Padronização dos dados com MinMaxScaler
    scaler = MinMaxScaler()
    padronizado = scaler.fit_transform(df[features])
    
    # Aplicando o PCA
    pca = PCA()
    pca.fit(padronizado)
    
    # Variância explicada acumulada
    variancia_explicada_acumulada = np.cumsum(pca.explained_variance_ratio_)
    
    # Aplicando o PCA e selecionando as primeiras componentes principais
    principais_componentes = pca.transform(padronizado)
    
    # Listas para armazenar os scores do Silhouette e WCSS
    silhouette_scores = []
    wcss = []
    
    for n_clusters in np.arange(2, 11):  # Começa em 2 porque não faz sentido calcular para 1 cluster
        kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42, max_iter=100)
        kmeans.fit(principais_componentes)
    
        score = silhouette_score(principais_componentes, kmeans.labels_)
        silhouette_scores.append(score)
        wcss.append(kmeans.inertia_)
    
    # Plotagem dos gráficos
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    # Gráfico de variância acumulada
    ax[0].plot(np.arange(1, len(variancia_explicada_acumulada) + 1), variancia_explicada_acumulada, marker='o', color='orange', label='Variância Explicada Acumulada')
    for i, valor in enumerate(variancia_explicada_acumulada):
        ax[0].text(i + 1, valor - 0.02, f'{valor:.2f}', ha='center', va='bottom', color='orange')
    ax[0].set_title('PCA - Variância Explicada Acumulada')
    ax[0].set_xlabel('Número de Componentes Principais')
    ax[0].set_ylabel('Proporção da Variância Explicada Acumulada')
    ax[0].legend(loc='center right', bbox_to_anchor=(0.9, 0.5), frameon=False)
    ax[0].grid(True)

    # Gráfico do Silhouette Score e WCSS
    ax[1].plot(np.arange(2, 11), silhouette_scores, marker='o', color='blue', label='Silhouette Score')
    ax[1].set_title('Definição do Melhor Número de Clusters')
    ax[1].set_xlabel('Número de Clusters')
    ax[1].set_ylabel('Silhouette Score')
    ax[1].grid(True)

    ax2 = ax[1].twinx()
    ax2.plot(np.arange(2, 11), wcss, marker='x', color='red', linestyle='--', label='WCSS')
    ax2.set_ylabel('WCSS')

    ax2.spines['right'].set_color('none')
    ax2.yaxis.set_ticks([])

    ax[1].legend(loc='center right', bbox_to_anchor=(0.9, 0.8), frameon=False)
    ax2.legend(loc='center right', bbox_to_anchor=(0.9, 0.7), frameon=False)

    plt.tight_layout()
    plt.show()


def train_min_max_scaler_cluster(df, tipo):

    features = [
       'num_unq_mov_max_m6',
       'num_100_mov_max_m6', 'num_unq_mov_min_m6', 'num_100_mov_min_m6',
       '%num_more_than_50_mov_max_m6', '%num_more_than_50_mov_avg_m6',
       '%num_more_than_50_mov_min_m6', 'num_25_mov_max_m6', 'num_50_mov_avg_m6', 'num_985_mov_avg_m6', 'num_25_mov_min_m6',
       'num_75_mov_max_m6', 'num_50_mov_min_m6', 'num_985_mov_min_m6', 'num_75_mov_min_m6', 'months_as_a_registered', 'payment_method_id', 'bd', 'city'
    ]

    df_scaler = df[features].copy()
    scaler = MinMaxScaler()
    scaler.fit(df_scaler)

    joblib.dump(scaler, f"../00_DataMaster/models/scaler_cluster_{tipo}.pkl")

    print('Scaler Treinado e Salvo com sucesso!')


def train_PCA(df, tipo):

    features = [
       'num_unq_mov_max_m6',
       'num_100_mov_max_m6', 'num_unq_mov_min_m6', 'num_100_mov_min_m6',
       '%num_more_than_50_mov_max_m6', '%num_more_than_50_mov_avg_m6',
       '%num_more_than_50_mov_min_m6', 'num_25_mov_max_m6', 'num_50_mov_avg_m6', 'num_985_mov_avg_m6', 'num_25_mov_min_m6',
       'num_75_mov_max_m6', 'num_50_mov_min_m6', 'num_985_mov_min_m6', 'num_75_mov_min_m6', 'months_as_a_registered', 'payment_method_id', 'bd', 'city'
    ]

    # Padronização dos dados
    scaler = joblib.load(f"../00_DataMaster/models/scaler_cluster_{tipo}.pkl")
    padronizado = scaler.transform(df[features])
    
    # Aplicando o PCA
    pca = PCA()
    pca.fit(padronizado)
    
    # Salvando o PCA
    joblib.dump(pca, f"../00_DataMaster/models/pca_cluster_{tipo}.pkl")
    print('PCA Treinado e Salvo com sucesso!')


def Clusterizador(df, tipo):
    scaler = joblib.load(f"../00_DataMaster/models/scaler_cluster_{tipo}.pkl")
    pca = joblib.load(f"../00_DataMaster/models/pca_cluster_{tipo}.pkl")

    features = [
       'num_unq_mov_max_m6',
       'num_100_mov_max_m6', 'num_unq_mov_min_m6', 'num_100_mov_min_m6',
       '%num_more_than_50_mov_max_m6', '%num_more_than_50_mov_avg_m6',
       '%num_more_than_50_mov_min_m6', 'num_25_mov_max_m6', 'num_50_mov_avg_m6', 'num_985_mov_avg_m6', 'num_25_mov_min_m6',
       'num_75_mov_max_m6', 'num_50_mov_min_m6', 'num_985_mov_min_m6', 'num_75_mov_min_m6', 'months_as_a_registered', 'payment_method_id', 'bd', 'city'
    ]

    # Aplicando o Min Max Scaler, PCA e selecionando as primeiras componentes principais
    padronizado = scaler.transform(df[features])
    principais_componentes = pca.transform(padronizado)[:, :8]
    kmeans = KMeans(n_clusters=3, init='random', random_state=42, max_iter=100)
    kmeans.fit(principais_componentes)

    joblib.dump(kmeans, f"../00_DataMaster/models/kmeans_cluster_{tipo}.pkl")

    print('KMeans Treinado e Salvo com sucesso!')


def modelo_clusterizador_churn_oficial(df, tipo):

    scaler = joblib.load(f"../00_DataMaster/models/scaler_cluster_{tipo}.pkl")
    pca = joblib.load(f"../00_DataMaster/models/pca_cluster_{tipo}.pkl")
    kmeans = joblib.load(f"../00_DataMaster/models/kmeans_cluster_{tipo}.pkl")

    features = [
       'num_unq_mov_max_m6',
       'num_100_mov_max_m6', 'num_unq_mov_min_m6', 'num_100_mov_min_m6',
       '%num_more_than_50_mov_max_m6', '%num_more_than_50_mov_avg_m6',
       '%num_more_than_50_mov_min_m6', 'num_25_mov_max_m6', 'num_50_mov_avg_m6', 'num_985_mov_avg_m6', 'num_25_mov_min_m6',
       'num_75_mov_max_m6', 'num_50_mov_min_m6', 'num_985_mov_min_m6', 'num_75_mov_min_m6', 'months_as_a_registered', 'payment_method_id', 'bd', 'city'
    ]

    df_cluster = df[features].copy()
    df_features = scaler.transform(df[features])
    df_features = pca.transform(df_features)[:, :8]
    clusters = kmeans.predict(df_features)

    return clusters
