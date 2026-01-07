import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.metrics import (
    r2_score, 
    mean_squared_error, 
    mean_absolute_error,    
    median_absolute_error   
)
import lightgbm as lgb
import xgboost as xgb
import multiprocessing
from functools import partial
from tqdm import tqdm
import sys
import os
import warnings
import gc
import math
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import requests
import zipfile
import io
import logging
import time

# ==============================================================================
# CONFIGURAÇÃO DE LOGS E AMBIENTE
# ==============================================================================
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("execution_log_no_checkpoint.txt", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==============================================================================
# 0. UTILITÁRIOS DE SISTEMA E DOWNLOAD
# ==============================================================================

def baixar_e_extrair_inep(anos, diretorio_base):
    url_base = "https://download.inep.gov.br/dados_abertos/microdados_censo_escolar_{}.zip"
    
    dir_dados = os.path.join(diretorio_base, "dados_brutos")
    if not os.path.exists(dir_dados):
        os.makedirs(dir_dados)

    logger.info(f"Verificando dados locais em: {dir_dados}")

    for ano in anos:
        caminho_ano = os.path.join(dir_dados, str(ano))
        if not os.path.exists(caminho_ano):
            os.makedirs(caminho_ano)
        
        csvs = []
        for ext in ["*.csv", "*.CSV"]:
            csvs.extend(glob.glob(os.path.join(caminho_ano, "**", ext), recursive=True))
            
        if csvs:
            continue

        logger.info(f"Ano {ano}: Baixando do INEP...")
        try:
            url = url_base.format(ano)
            headers = {'User-Agent': 'Mozilla/5.0'}
            
            response = requests.get(url, headers=headers, stream=True, verify=False)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024 
            
            buffer = io.BytesIO()
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"Download {ano}") as t:
                for data in response.iter_content(block_size):
                    t.update(len(data))
                    buffer.write(data)
            
            logger.info(f"Ano {ano}: Extraindo...")
            with zipfile.ZipFile(buffer) as z:
                z.extractall(caminho_ano)
            
        except Exception as e:
            logger.error(f"Falha ao baixar {ano}: {e}")

# ==============================================================================
# 0.2 UTILITÁRIOS GERAIS
# ==============================================================================

def gerar_shap_analysis(modelo_pipeline, X_treino, feature_names_originais, etapa, diretorio_saida):
    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        if not hasattr(modelo_pipeline, 'named_steps'): return
        
        model_obj = modelo_pipeline.named_steps['model']
        target_model = model_obj
        
        if isinstance(model_obj, VotingRegressor):
            target_model = model_obj.estimators_[0] 
        
        if 'xgb' not in str(type(target_model)).lower() and 'lgbm' not in str(type(target_model)).lower(): 
            return

        
        X_transformed = modelo_pipeline.named_steps['preprocessor'].transform(X_treino)
        X_transformed = modelo_pipeline.named_steps['imputer'].transform(X_transformed)
        
        feature_names_out = modelo_pipeline.named_steps['preprocessor'].get_feature_names_out()

        if 'variance' in modelo_pipeline.named_steps:
            variance_step = modelo_pipeline.named_steps['variance']
            X_transformed = variance_step.transform(X_transformed)
            
            mask = variance_step.get_support()
            feature_names_out = feature_names_out[mask]

        clean_feature_names = [f.split('__')[-1] for f in feature_names_out]

        X_sample = pd.DataFrame(
            X_transformed, 
            columns=clean_feature_names
        ).sample(n=min(500, len(X_transformed)), random_state=42)
        
        explainer = shap.TreeExplainer(target_model)
        shap_values = explainer.shap_values(X_sample)
        
        plt.figure()
    
        shap.summary_plot(shap_values, X_sample, show=False)
        
        if not os.path.exists(diretorio_saida): os.makedirs(diretorio_saida, exist_ok=True)
        plt.savefig(os.path.join(diretorio_saida, f'SHAP_summary_{etapa}.png'), bbox_inches='tight')
        plt.close()
        logger.info(f"Gráfico SHAP gerado para {etapa}")
    except Exception as e:
        logger.warning(f"Erro SHAP (ignorado) - {etapa}: {e}")
        import traceback
        traceback.print_exc() 

# ==============================================================================
# 1. CARGA E PREPROCESSAMENTO
# ==============================================================================

def otimizar_memoria(df):
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and col_type.name != 'category':
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max: df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max: df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max: df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max: df[col] = df[col].astype(np.float32)
    
    cols_object = df.select_dtypes(include=['object']).columns
    for col in cols_object:
        if df[col].nunique() / len(df) < 0.5: df[col] = df[col].astype('category')
        else: df[col] = df[col].fillna("").astype(str)
    
    return df

def _ler_um_ano_flexivel(ano, diretorio_base, colunas=None):
    path_ano = os.path.join(diretorio_base, 'dados_brutos', str(ano))
    csvs = []
    for ext in ["**/*.csv", "**/*.CSV"]:
        csvs.extend(glob.glob(os.path.join(path_ano, ext), recursive=True))
    
    if not csvs: return None
        
    csv_principal = max(csvs, key=os.path.getsize)
    for sep in [';', '|']:
        for enc in ['latin1', 'utf-8']:
            try:
                df = pd.read_csv(csv_principal, sep=sep, encoding=enc, low_memory=False, usecols=colunas)
                if colunas is None: df['NU_ANO_CENSO'] = ano
                return df
            except: continue
    return None

def carregar_dados_paralelo(caminho_base, anos, amostra=None):
    logger.info(f"Lendo CSVs para {len(anos)} anos...")
    pool_size = max(1, multiprocessing.cpu_count() - 2)
    with multiprocessing.Pool(pool_size) as pool:
        func = partial(_ler_um_ano_flexivel, diretorio_base=caminho_base)
        dfs = [res for res in tqdm(pool.imap_unordered(func, anos), total=len(anos)) if res is not None]
    
    if not dfs: sys.exit("Erro: Nenhum dado carregado.")
    df = pd.concat(dfs, ignore_index=True)
    if amostra is not None:
        logger.info("Filtrando amostra de entidades...")
        df = df[df['CO_ENTIDADE'].isin(amostra)]
    df = otimizar_memoria(df)
    return df

def preprocessar(df):
    logger.info("Iniciando Preprocessamento (SEM CHECKPOINT)...")
    
    sufixos = ['CRE', 'PRE', 'AI', 'AF', 'MED', 'FUND', 'TEC', 'CE', 'EJA_FUND', 'EJA_MED']
    cols_etapas = [c for c in df.columns if c.startswith('IN_') and any(s in c for s in sufixos)]
    for c in cols_etapas: df[c] = df[c].fillna(0).astype(np.int8)

    cols_cat = [c for c in df.select_dtypes(include=['object', 'category']).columns if c != 'CO_ENTIDADE']
    for c in cols_cat: 
        if not pd.api.types.is_categorical_dtype(df[c]):
            df[c] = df[c].astype('category')

    cols_nan = [c for c in df.select_dtypes(include=np.number).columns 
                if df[c].isnull().any() and c not in ['CO_ENTIDADE', 'NU_ANO_CENSO']]
    
    if cols_nan:
        logger.info(f"Tratando {len(cols_nan)} colunas com Forward Fill (apenas para criação de features)...")
        df = df.sort_values(['CO_ENTIDADE', 'NU_ANO_CENSO'])
        df[cols_nan] = df.groupby('CO_ENTIDADE')[cols_nan].ffill()
        df[cols_nan] = df[cols_nan].fillna(0)

    df = otimizar_memoria(df)
    return df, cols_etapas

# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================

def _chunk_features(df_chunk):
    df_chunk = df_chunk.sort_values(['CO_ENTIDADE', 'NU_ANO_CENSO'])
    cols_base = [c for c in df_chunk.columns if c.startswith(('QT_MAT_', 'QT_DOC_', 'QT_TUR_'))]
    grp = df_chunk.groupby('CO_ENTIDADE')
    
    for c in cols_base:
        df_chunk[f'{c}_lag1'] = grp[c].shift(1)
        df_chunk[f'{c}_lag2'] = grp[c].shift(2)
        df_chunk[f'{c}_roll_mean3'] = grp[c].shift(1).rolling(3, min_periods=1).mean()
        df_chunk[f'{c}_diff1'] = df_chunk[f'{c}_lag1'] - df_chunk[f'{c}_lag2']

    epsilon = 1e-6
    sufixos_comuns = ['MED', 'FUND', 'INF_CRE', 'INF_PRE', 'EJA_MED', 'EJA_FUND', 'PROF_TEC']
    
    for suf in sufixos_comuns:
        mat_lag = f'QT_MAT_{suf}_lag1'
        doc_lag = f'QT_DOC_{suf}_lag1'
        tur_lag = f'QT_TUR_{suf}_lag1'
        
        if mat_lag in df_chunk.columns and doc_lag in df_chunk.columns:
            df_chunk[f'RATIO_ALUNO_DOC_{suf}'] = df_chunk[mat_lag] / (df_chunk[doc_lag] + epsilon)
        if mat_lag in df_chunk.columns and tur_lag in df_chunk.columns:
            df_chunk[f'RATIO_ALUNO_TUR_{suf}'] = df_chunk[mat_lag] / (df_chunk[tur_lag] + epsilon)
        
        diff_col = f'QT_MAT_{suf}_diff1'
        if diff_col in df_chunk.columns:
            df_chunk[f'ACCEL_{suf}'] = df_chunk[diff_col].diff() 

    return df_chunk

def features_temporais_paralelo(df):
    logger.info("Criando Features Temporais (CPU Multiprocessing)...")
    codigos = df['CO_ENTIDADE'].unique()
    chunks = np.array_split(codigos, min(multiprocessing.cpu_count() * 2, len(codigos)))
    gen = (df[df['CO_ENTIDADE'].isin(c)].copy() for c in chunks)
    
    with multiprocessing.Pool(max(1, multiprocessing.cpu_count() - 2)) as pool:
        dfs = [res for res in tqdm(pool.imap_unordered(_chunk_features, gen), total=len(chunks))]
        
    df = pd.concat(dfs, ignore_index=True)
    
    cols_lag = [c for c in df.columns if '_lag1' in c]
    if cols_lag: df.dropna(subset=cols_lag, inplace=True)
        
    cols_numericas = df.select_dtypes(include=[np.number]).columns
    df[cols_numericas] = df[cols_numericas].fillna(0)

    df = otimizar_memoria(df)
    return df

# ==============================================================================
# 3. MODELAGEM OTIMIZADA (SEM LEAKAGE)
# ==============================================================================

def treinar_etapa_gpu(col_etapa, df, top_n, output_dir):
    start_time = time.time()
    logger.info(f"--- Processando Etapa: {col_etapa} ---")
    
    df_etapa = df[df[col_etapa] == 1].copy()
    if len(df_etapa) < 100: 
        logger.warning(f"Amostra insuficiente para {col_etapa} ({len(df_etapa)})")
        return []

    df_etapa = df_etapa.sort_values(['NU_ANO_CENSO', 'CO_ENTIDADE'])
    
    sufixo = col_etapa.replace('IN_', '')
    alvo = f'QT_MAT_{sufixo}'
    if alvo not in df_etapa.columns or df_etapa[alvo].sum() == 0: 
        return []

    y = np.log1p(df_etapa[alvo])
    
    colunas_proibidas = []
    for col in df_etapa.columns:
        if col.startswith('IN_') or col in ['CO_ENTIDADE', 'NU_ANO_CENSO', 'NO_ENTIDADE']:
            colunas_proibidas.append(col)
            continue
        if col.startswith('QT_'):
            if not any(x in col for x in ['_lag', '_diff', '_roll', 'RATIO_', 'ACCEL_']):
                colunas_proibidas.append(col)
    if alvo not in colunas_proibidas: colunas_proibidas.append(alvo)

    X_cols = [c for c in df_etapa.columns if c not in colunas_proibidas]
    
    cols_cat = [c for c in X_cols if pd.api.types.is_categorical_dtype(df_etapa[c]) or df_etapa[c].dtype == 'object']
    cols_num = [c for c in X_cols if c not in cols_cat]
    
    if cols_cat:
        for col in cols_cat:
            df_etapa[col] = df_etapa[col].astype(str).fillna("missing")

    df_etapa = df_etapa[X_cols + [alvo] + ['NU_ANO_CENSO']]
    X_full = df_etapa[X_cols]
    
    logger.info(f"Shape: {X_full.shape}. Iniciando Validação Walk-Forward (Métricas em Escala Real)...")

    anos_unicos = sorted(df_etapa['NU_ANO_CENSO'].unique())
    
    lgbm_gpu = lgb.LGBMRegressor(
        random_state=42, n_jobs=1, verbosity=-1, 
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        device='gpu'
    )
    
    xgb_gpu = xgb.XGBRegressor(
        random_state=42, n_jobs=1, verbosity=0,
        n_estimators=200, learning_rate=0.05, subsample=0.8,
        tree_method='gpu_hist', device='cuda'
    )

    voting_ensemble = VotingRegressor(
        estimators=[('lgbm', lgbm_gpu), ('xgb', xgb_gpu)],
        n_jobs=1
    )
    
    modelos = {
        "Voting_Ensemble": voting_ensemble,
        "LightGBM": lgbm_gpu,
        "XGBoost": xgb_gpu
    }
    
    resultados_gerais = []
    melhor_modelo_final = None
    melhor_score_medio = -np.inf 
    melhor_nome_modelo = ""

    for nome_modelo, modelo_base in modelos.items():
        scores_r2_real = []
        scores_rmse_real = []
        scores_mae_real = []
        scores_medae_real = []
        
        for i in range(2, len(anos_unicos)):
            ano_teste = anos_unicos[i]
            anos_treino = anos_unicos[:i] 
            
            mask_treino = df_etapa['NU_ANO_CENSO'].isin(anos_treino)
            mask_teste = df_etapa['NU_ANO_CENSO'] == ano_teste
            
            X_train = X_full[mask_treino]
            y_train = y[mask_treino] 
            X_test = X_full[mask_teste]
            y_test = y[mask_teste]   
            
            if len(X_train) == 0 or len(X_test) == 0: continue

            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cols_cat),
                    ('num', 'passthrough', cols_num)
                ]
            )
            
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('imputer', SimpleImputer(strategy='median')),
                ('variance', VarianceThreshold(threshold=0.0)),
                ('model', modelo_base)
            ])
            
            try:
                pipeline.fit(X_train, y_train)
                
                y_pred_log = pipeline.predict(X_test)
                
                y_test_real = np.expm1(y_test)
                y_pred_real = np.expm1(y_pred_log)
                
                y_pred_real = np.maximum(0, y_pred_real) 
                
                r2_real = r2_score(y_test_real, y_pred_real)
                rmse_real = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
                mae_real = mean_absolute_error(y_test_real, y_pred_real)
                medae_real = median_absolute_error(y_test_real, y_pred_real)
                
                scores_r2_real.append(r2_real)
                scores_rmse_real.append(rmse_real)
                scores_mae_real.append(mae_real)
                scores_medae_real.append(medae_real)
                
            except Exception as e:
                logger.warning(f"Erro fold {ano_teste} ({nome_modelo}): {e}")
                continue
        
        if scores_r2_real:
            r2_mean = np.mean(scores_r2_real)
            
            resultados_gerais.append({
                'etapa': col_etapa, 
                'modelo': nome_modelo,
                'R2_Media_CV': r2_mean, 
                'R2_Std_CV': np.std(scores_r2_real),
                'RMSE_Media_CV': np.mean(scores_rmse_real),
                'MAE_Media_CV': np.mean(scores_mae_real),
                'MedAE_Media_CV': np.mean(scores_medae_real),
                'num_amostras': len(X_full),
                'num_folds': len(scores_r2_real)
            })
            
            if r2_mean > melhor_score_medio:
                melhor_score_medio = r2_mean
                melhor_modelo_final = pipeline
                melhor_nome_modelo = nome_modelo
            
            logger.info(f"Modelo {nome_modelo}: R2_Real={r2_mean:.4f} | MAE_Real={np.mean(scores_mae_real):.2f} alunos")

    if melhor_modelo_final:
        logger.info(f"Retreinando modelo final ({melhor_nome_modelo}) para geração de SHAP...")
        try:
            cutoff_idx = int(len(X_full) * 0.8)
            X_shap = X_full.iloc[:cutoff_idx]
            y_shap = y.iloc[:cutoff_idx] 
            
            melhor_modelo_final.fit(X_shap, y_shap)
            
            feature_names_in = X_cols 
            gerar_shap_analysis(melhor_modelo_final, X_shap, feature_names_in, f"{col_etapa}_{melhor_nome_modelo}", output_dir)
        except Exception as e: 
            logger.warning(f"Erro ao gerar final fit/SHAP: {e}")

    logger.info(f"Fim etapa {col_etapa}. Tempo: {time.time() - start_time:.1f}s")
    return resultados_gerais

def orquestrador_modelagem(df, cols_etapas, output_dir):
    logger.info("INICIANDO MODELAGEM (MODO ANTI-LEAKAGE)")
    for f in glob.glob(os.path.join(output_dir, "parcial_*.csv")):
        try: os.remove(f)
        except: pass
    
    todos_resultados = []
    for i, etapa in enumerate(cols_etapas):
        logger.info(f"Progresso: {i+1}/{len(cols_etapas)} etapas.")
        res = treinar_etapa_gpu(etapa, df, top_n=30, output_dir=output_dir)
        if res:
            todos_resultados.extend(res)
            pd.DataFrame(res).to_csv(os.path.join(output_dir, f"parcial_{etapa}.csv"), index=False)
            
    return todos_resultados

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    HOME_USER = os.path.expanduser('~')
    DIRETORIO_PROJETO = os.path.join(HOME_USER, 'projeto_censo_escolar_v3')
    CAMINHO_SAIDA = os.path.join(DIRETORIO_PROJETO, 'resultados_finais.csv')
    
    if not os.path.exists(DIRETORIO_PROJETO):
        os.makedirs(DIRETORIO_PROJETO)
    
    ANOS = range(2011, 2022) 
    FRACAO_AMOSTRA = 1.00 

    logger.info(f"--- START SCRIPT (NO CHECKPOINT VERSION) ---")
    if torch.cuda.is_available():
        logger.info(f"GPU DETECTADA: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("GPU NÃO DETECTADA! Isso será lento.")

    baixar_e_extrair_inep(ANOS, DIRETORIO_PROJETO)

    codigos_unicos = set()
    for ano in ANOS:
        df_tmp = _ler_um_ano_flexivel(ano, DIRETORIO_PROJETO, ['CO_ENTIDADE'])
        if df_tmp is not None: codigos_unicos.update(df_tmp['CO_ENTIDADE'].unique())
    
    codigos_unicos = list(codigos_unicos)
    if not codigos_unicos:
        sys.exit("[ERRO] Nenhum dado encontrado.")

    np.random.seed(42)
    amostra = np.random.choice(codigos_unicos, int(len(codigos_unicos)*FRACAO_AMOSTRA), replace=False)
    
    df = carregar_dados_paralelo(DIRETORIO_PROJETO, ANOS, amostra)
    df, cols_etapas = preprocessar(df)
    df = features_temporais_paralelo(df)
    
    resultados = orquestrador_modelagem(df, cols_etapas, DIRETORIO_PROJETO)
    
    if resultados:
        df_res = pd.DataFrame(resultados)
        df_res.to_csv(CAMINHO_SAIDA, index=False)
        logger.info(f"SUCESSO! Resultados salvos em: {CAMINHO_SAIDA}")
        logger.info(f"R2 Médio Global: {df_res['R2_Media_CV'].mean():.4f}")
        
        for f in glob.glob(os.path.join(DIRETORIO_PROJETO, "parcial_*.csv")): os.remove(f)
    else:
        logger.warning("Nenhum resultado gerado.")

if __name__ == "__main__":
    main()
