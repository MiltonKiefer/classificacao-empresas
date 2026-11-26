from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src import config
from src.model.evaluate import metricas, recall_por_fold, render_avaliacao
from src.preprocessing.build_features import feature_columns, make_preprocessor

N_SPLITS = 3
PROFUNDIDADES = (2, 3)

XGB_FIXOS = {
    "objective": "binary:logistic",
    "n_estimators": 80,
    "learning_rate": 0.1,
    "min_child_weight": 2,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_lambda": 1.0,
    "random_state": config.RANDOM_STATE,
    "eval_metric": "logloss",
    "verbosity": 0,
    "n_jobs": 1,
}


def carregar_treino(path: Path | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Lê só as linhas de treino da matriz da Etapa 4."""
    origem = Path(path or config.FEATURES_PARQUET)
    if not origem.is_file():
        raise FileNotFoundError(
            f"Matriz da Etapa 4 não encontrada: {origem}. Execute antes: python -m src.pipeline --etapa 4"
        )
    frame = pd.read_parquet(origem)
    treino = frame.loc[frame["conjunto"] == "treino"].reset_index(drop=True)
    if treino.empty:
        raise ValueError("A matriz não tem linhas de treino.")
    x = treino.loc[:, feature_columns()]
    y = treino["rotulo"].astype(int)
    identificadores = treino["cnpj_raiz"].astype(str)
    return x, y, identificadores


def peso_grande(y) -> float:
    """Peso da classe Grande (1) a partir das linhas de treino deste fold."""
    rotulos = np.asarray(y).astype(int)
    n_grande = int((rotulos == config.CLASS_GRANDE).sum())
    n_medio = int((rotulos == config.CLASS_MEDIO).sum())
    if n_grande == 0:
        return 1.0
    return n_medio / n_grande


def _pipeline_xgb(profundidade: int, y_treino, ponderar: bool) -> Pipeline:
    params = dict(XGB_FIXOS)
    params["max_depth"] = profundidade
    params["scale_pos_weight"] = peso_grande(y_treino) if ponderar else 1.0
    return Pipeline([
        ("prep", make_preprocessor()),
        ("modelo", XGBClassifier(**params)),
    ])


def _pipeline_baseline(nome: str) -> Pipeline:
    if nome == "maioria":
        modelo = DummyClassifier(strategy="most_frequent")
    elif nome == "arvore":
        modelo = DecisionTreeClassifier(max_depth=3, random_state=config.RANDOM_STATE)
    else:
        raise ValueError(nome)
    return Pipeline([
        ("prep", make_preprocessor()),
        ("modelo", modelo),
    ])


def _folds(y, n_splits: int) -> StratifiedKFold:
    menor = int(pd.Series(y).value_counts().min())
    if menor < n_splits:
        raise ValueError(
            f"A menor classe tem {menor} empresas, menos que {n_splits} folds. "
            "A validação estratificada não fecha."
        )
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)


def _escolher_profundidade(x_treino: pd.DataFrame, y_treino: pd.Series, ponderar: bool) -> int:
    """Escolhe a profundidade só com as linhas de treino do fold, nunca com o teste."""
    y_treino = pd.Series(y_treino).reset_index(drop=True)
    x_treino = x_treino.reset_index(drop=True)
    if int(y_treino.value_counts().min()) < 2:
        return PROFUNDIDADES[0]
    interno = StratifiedKFold(n_splits=2, shuffle=True, random_state=config.RANDOM_STATE)
    melhor_profundidade = PROFUNDIDADES[0]
    melhor_nota = -1.0
    for profundidade in PROFUNDIDADES:
        notas = []
        for treino_idx, valida_idx in interno.split(x_treino, y_treino):
            pipe = _pipeline_xgb(profundidade, y_treino.iloc[treino_idx], ponderar)
            pipe.fit(x_treino.iloc[treino_idx], y_treino.iloc[treino_idx])
            previsto = pipe.predict(x_treino.iloc[valida_idx])
            notas.append(f1_score(y_treino.iloc[valida_idx], previsto, average="macro", zero_division=0))
        nota = float(np.mean(notas))
        if nota > melhor_nota:
            melhor_nota = nota
            melhor_profundidade = profundidade
    return melhor_profundidade


def _prever_folds(x, y, fabrica, n_splits: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ajusta um modelo novo em cada fold e devolve previsão e probabilidade fora do fold."""
    y = pd.Series(y).reset_index(drop=True).astype(int)
    x = x.reset_index(drop=True)
    previsto = np.zeros(len(y), dtype=int)
    probabilidade = np.zeros(len(y), dtype=float)
    folds = np.zeros(len(y), dtype=int)
    for numero, (treino_idx, teste_idx) in enumerate(_folds(y, n_splits).split(x, y)):
        modelo = fabrica(x.iloc[treino_idx], y.iloc[treino_idx])
        modelo.fit(x.iloc[treino_idx], y.iloc[treino_idx])
        previsto[teste_idx] = modelo.predict(x.iloc[teste_idx]).astype(int)
        classes = list(modelo.named_steps["modelo"].classes_)
        proba = modelo.predict_proba(x.iloc[teste_idx])
        probabilidade[teste_idx] = proba[:, classes.index(config.CLASS_GRANDE)]
        folds[teste_idx] = numero
    return previsto, probabilidade, folds


def validar(x: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS) -> dict:
    """Compara maioria, árvore e XGBoost com e sem peso de classe."""
    y = pd.Series(y).reset_index(drop=True).astype(int)
    x = x.reset_index(drop=True)
    contagem = {
        config.CLASS_NAMES[config.CLASS_MEDIO]: int((y == config.CLASS_MEDIO).sum()),
        config.CLASS_NAMES[config.CLASS_GRANDE]: int((y == config.CLASS_GRANDE).sum()),
    }
    variantes = {
        "maioria": lambda xt, yt: _pipeline_baseline("maioria"),
        "arvore": lambda xt, yt: _pipeline_baseline("arvore"),
        "xgb": lambda xt, yt: _pipeline_xgb(_escolher_profundidade(xt, yt, False), yt, False),
        "xgb_ponderado": lambda xt, yt: _pipeline_xgb(_escolher_profundidade(xt, yt, True), yt, True),
    }
    modelos = {}
    previsoes = {}
    for nome, fabrica in variantes.items():
        previsto, probabilidade, folds = _prever_folds(x, y, fabrica, n_splits)
        bloco = metricas(y, previsto, probabilidade)
        bloco["recall_grande_por_fold"] = recall_por_fold(y, previsto, folds)
        modelos[nome] = bloco
        previsoes[nome] = previsto
    candidatos = ("xgb", "xgb_ponderado")
    escolhido = max(candidatos, key=lambda nome: modelos[nome]["f1_macro"])
    return {
        "n_treino": int(len(y)),
        "contagem": contagem,
        "validacao": f"StratifiedKFold com {n_splits} folds, embaralhado, semente {config.RANDOM_STATE}",
        "calibrador_aplicado": False,
        "calibracao": (
            "Não há amostra suficiente para avaliar ou ajustar um calibrador de probabilidade. "
            "O Brier é só um registro; a probabilidade não deve ser chamada de confiança."
        ),
        "modelo_salvo": escolhido,
        "modelos": modelos,
        "previsoes_fora_do_fold": {nome: previsoes[nome].tolist() for nome in previsoes},
        "limitacao": (
            "Amostra de treino pequena, em especial a classe Grande. "
            "As métricas são exploratórias e não sustentam uma conclusão forte de generalização."
        ),
        "hiperparametros": (
            "XGBoost com objetivo binary:logistic, 80 árvores, learning_rate 0,1 e min_child_weight 2. "
            "A profundidade (2 ou 3) é escolhida por uma validação estratificada interna, "
            "só com as linhas de treino de cada fold. O peso da classe Grande, quando usado, "
            "também é contado só nesse treino. Não houve oversampling."
        ),
    }


def ajustar_modelo_final(x: pd.DataFrame, y: pd.Series, ponderar: bool) -> Pipeline:
    """Reajusta o modelo escolhido em todas as empresas rotuladas, depois da validação."""
    y = pd.Series(y).reset_index(drop=True).astype(int)
    x = x.reset_index(drop=True)
    profundidade = _escolher_profundidade(x, y, ponderar)
    modelo = _pipeline_xgb(profundidade, y, ponderar)
    modelo.fit(x, y)
    modelo.profundidade_escolhida_ = profundidade
    return modelo


def executar_treinamento(
    x: pd.DataFrame | None = None,
    y: pd.Series | None = None,
    identificadores: pd.Series | None = None,
    n_splits: int = N_SPLITS,
    caminho_modelo: Path | None = None,
    caminho_json: Path | None = None,
    caminho_md: Path | None = None,
) -> dict:
    """Valida, grava o relatório e salva o XGBoost reajustado nas empresas rotuladas."""
    if x is None or y is None:
        x, y, identificadores = carregar_treino()
    relatorio = validar(x, y, n_splits=n_splits)
    ponderar = relatorio["modelo_salvo"] == "xgb_ponderado"
    modelo = ajustar_modelo_final(x, y, ponderar)
    relatorio["profundidade_final"] = int(modelo.profundidade_escolhida_)
    relatorio["features"] = feature_columns()
    relatorio["classes"] = {str(k): v for k, v in config.CLASS_NAMES.items()}
    relatorio["regra_capital_social_zero"] = config.CAPITAL_SOCIAL_ZERO_RULE
    relatorio["referencia_temporal"] = config.REFERENCE_DATE.isoformat()
    if identificadores is not None:
        relatorio["cnpj_raiz_treino"] = identificadores.astype(str).tolist()

    caminho_modelo = Path(caminho_modelo or config.MODEL_PATH)
    caminho_json = Path(caminho_json or config.REPORT_AVALIACAO_JSON)
    caminho_md = Path(caminho_md or config.REPORT_AVALIACAO_MD)
    caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
    caminho_json.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": modelo,
            "features": feature_columns(),
            "classes": config.CLASS_NAMES,
            "variante": relatorio["modelo_salvo"],
            "profundidade": relatorio["profundidade_final"],
            "random_state": config.RANDOM_STATE,
            "referencia_temporal": config.REFERENCE_DATE.isoformat(),
            "n_treino": relatorio["n_treino"],
            "contagem": relatorio["contagem"],
        },
        caminho_modelo,
    )
    gravar = {chave: valor for chave, valor in relatorio.items() if chave != "previsoes_fora_do_fold"}
    caminho_json.write_text(json.dumps(gravar, ensure_ascii=False, indent=2), encoding="utf-8")
    caminho_md.write_text(render_avaliacao(relatorio), encoding="utf-8")
    relatorio["caminhos"] = {
        "modelo": str(caminho_modelo),
        "json": str(caminho_json),
        "markdown": str(caminho_md),
    }
    return relatorio
