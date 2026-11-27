"""Etapa 6: inferência das empresas sem rótulo Médio/Grande."""
from __future__ import annotations

from datetime import datetime
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.labeling.prepare_labels import CONJUNTO_INFERENCIA
from src.preprocessing.build_features import feature_columns

MOTIVO_REVISAO = (
    "Amostra rotulada pequena e probabilidades sem calibração; "
    "revisão manual recomendada."
)

COLUNAS_RESULTADO = [
    "cnpj_raiz",
    "porte_original",
    "porte_estimado",
    "prob_medio",
    "prob_grande",
    "status_classificacao",
    "revisao_recomendada",
    "motivo_revisao",
    "modelo_versao",
    "modelo_variante",
    "dados_versao",
    "data_execucao",
]


def _sha256_arquivo(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            digest.update(bloco)
    return f"sha256:{digest.hexdigest()}"


def _validar_pacote_modelo(pacote: object) -> tuple[object, list[str], list[int], str]:
    if not isinstance(pacote, dict):
        raise ValueError("O arquivo do modelo não contém um pacote válido da Etapa 5.")

    pipeline = pacote.get("pipeline")
    features = pacote.get("features")
    classes = pacote.get("classes")
    variante = pacote.get("variante")

    if pipeline is None or not hasattr(pipeline, "predict") or not hasattr(pipeline, "predict_proba"):
        raise ValueError("O pacote do modelo não contém um pipeline com predict e predict_proba.")
    if not isinstance(features, list) or not all(isinstance(nome, str) for nome in features):
        raise ValueError("O pacote do modelo não contém uma lista válida de features.")
    if features != feature_columns():
        raise ValueError(
            "As features do modelo não correspondem às features atuais. "
            "Execute novamente as Etapas 4 e 5 antes da inferência."
        )
    if classes != config.CLASS_NAMES:
        raise ValueError("O mapeamento de classes do modelo não corresponde à configuração atual.")

    named_steps = getattr(pipeline, "named_steps", None)
    modelo_final = named_steps.get("modelo") if named_steps is not None else None
    classes_modelo = getattr(modelo_final, "classes_", None)
    if classes_modelo is None:
        raise ValueError("O pipeline salvo não expõe as classes do classificador final.")

    classes_ordenadas = [int(classe) for classe in classes_modelo]
    classes_esperadas = [config.CLASS_MEDIO, config.CLASS_GRANDE]
    if set(classes_ordenadas) != set(classes_esperadas) or len(classes_ordenadas) != 2:
        raise ValueError(
            f"O modelo precisa conter as classes {classes_esperadas}; "
            f"classes encontradas: {classes_ordenadas}."
        )
    if not isinstance(variante, str) or not variante:
        raise ValueError("O pacote do modelo não identifica a variante treinada.")
    return pipeline, features, classes_ordenadas, variante


def gerar_previsoes(
    features: pd.DataFrame,
    empresas: pd.DataFrame,
    pacote_modelo: object,
    *,
    modelo_versao: str,
    dados_versao: str,
    data_execucao: str | None = None,
) -> pd.DataFrame:
    """Gera resultados somente para raízes elegíveis à inferência."""
    pipeline, colunas_features, classes_modelo, variante = _validar_pacote_modelo(pacote_modelo)

    colunas_features_necessarias = {"cnpj_raiz", "conjunto", "rotulo", *colunas_features}
    ausentes = sorted(colunas_features_necessarias - set(features.columns))
    if ausentes:
        raise ValueError("A matriz de features está incompleta: " + ", ".join(ausentes))
    colunas_empresas_necessarias = {"cnpj_raiz", "conjunto", "porte_oficial"}
    ausentes = sorted(colunas_empresas_necessarias - set(empresas.columns))
    if ausentes:
        raise ValueError("A base consolidada está incompleta: " + ", ".join(ausentes))

    inferencia = features.loc[features["conjunto"] == CONJUNTO_INFERENCIA].copy().reset_index(drop=True)
    origem = empresas.loc[empresas["conjunto"] == CONJUNTO_INFERENCIA, [
        "cnpj_raiz", "porte_oficial",
    ]].copy()
    if inferencia.empty:
        raise ValueError("A matriz não contém empresas elegíveis à inferência.")
    if inferencia["rotulo"].notna().any():
        raise ValueError("Há rótulo conhecido no conjunto de inferência; a previsão foi interrompida.")
    if inferencia["cnpj_raiz"].isna().any() or origem["cnpj_raiz"].isna().any():
        raise ValueError("Há CNPJ raiz ausente no conjunto de inferência.")

    inferencia["cnpj_raiz"] = inferencia["cnpj_raiz"].astype(str)
    origem["cnpj_raiz"] = origem["cnpj_raiz"].astype(str)
    if inferencia["cnpj_raiz"].duplicated().any():
        raise ValueError("A matriz contém mais de uma linha de inferência para o mesmo CNPJ raiz.")
    if origem["cnpj_raiz"].duplicated().any():
        raise ValueError("A base consolidada contém CNPJ raiz duplicado no conjunto de inferência.")

    ids_features = set(inferencia["cnpj_raiz"])
    ids_origem = set(origem["cnpj_raiz"])
    if ids_features != ids_origem:
        raise ValueError(
            "A matriz de features e a base consolidada não têm o mesmo conjunto de CNPJs raiz "
            "para inferência."
        )

    x = inferencia.loc[:, colunas_features]
    classes_proba = [int(classe) for classe in pipeline.named_steps["modelo"].classes_]
    previsoes = np.asarray(pipeline.predict(x))
    probabilidades = np.asarray(pipeline.predict_proba(x), dtype="float64")
    if previsoes.shape != (len(inferencia),):
        raise ValueError("O modelo retornou uma quantidade inesperada de classes previstas.")
    if probabilidades.shape != (len(inferencia), len(classes_proba)):
        raise ValueError("O modelo retornou uma matriz de probabilidades com formato inesperado.")
    if not np.isfinite(probabilidades).all() or (probabilidades < 0).any() or (probabilidades > 1).any():
        raise ValueError("O modelo retornou probabilidades inválidas.")
    if not np.allclose(probabilidades.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("As probabilidades por empresa não somam 1.")

    previsoes = previsoes.astype(int)
    if not set(previsoes).issubset(set(classes_modelo)):
        raise ValueError("O modelo retornou uma classe que não consta no pacote.")

    indice_medio = classes_proba.index(config.CLASS_MEDIO)
    indice_grande = classes_proba.index(config.CLASS_GRANDE)
    resultado = pd.DataFrame({
        "cnpj_raiz": inferencia["cnpj_raiz"].to_numpy(),
        "porte_original": inferencia["cnpj_raiz"].map(
            origem.set_index("cnpj_raiz")["porte_oficial"]
        ).to_numpy(),
        "porte_estimado": pd.Series(previsoes).map(config.CLASS_NAMES).to_numpy(),
        "prob_medio": probabilidades[:, indice_medio],
        "prob_grande": probabilidades[:, indice_grande],
        "status_classificacao": "estimativa_modelo",
        "revisao_recomendada": True,
        "motivo_revisao": MOTIVO_REVISAO,
        "modelo_versao": modelo_versao,
        "modelo_variante": variante,
        "dados_versao": dados_versao,
        "data_execucao": data_execucao or datetime.now().astimezone().isoformat(timespec="seconds"),
    }, columns=COLUNAS_RESULTADO)
    return resultado


def executar_inferencia(
    caminho_features: Path | None = None,
    caminho_empresas: Path | None = None,
    caminho_modelo: Path | None = None,
    caminho_saida: Path | None = None,
) -> pd.DataFrame:
    """Carrega artefatos, prevê as empresas elegíveis e grava o Parquet de resultados."""
    caminho_features = Path(caminho_features or config.FEATURES_PARQUET)
    caminho_empresas = Path(caminho_empresas or config.PROCESSED_EMPRESAS)
    caminho_modelo = Path(caminho_modelo or config.MODEL_PATH)
    caminho_saida = Path(caminho_saida or config.PREDICTIONS_PARQUET)

    for caminho, descricao, comando in (
        (caminho_features, "Matriz da Etapa 4", "python -m src.pipeline --etapa 4"),
        (caminho_empresas, "Base consolidada da Etapa 3", "python -m src.pipeline --etapa 3"),
        (caminho_modelo, "Modelo da Etapa 5", "python -m src.pipeline --etapa 5"),
    ):
        if not caminho.is_file():
            raise FileNotFoundError(
                f"{descricao} não encontrado: {caminho}. Execute antes: {comando}"
            )

    pacote_modelo = joblib.load(caminho_modelo)
    features = pd.read_parquet(caminho_features)
    empresas = pd.read_parquet(caminho_empresas)
    resultado = gerar_previsoes(
        features,
        empresas,
        pacote_modelo,
        modelo_versao=_sha256_arquivo(caminho_modelo),
        dados_versao=(
            "sha256:"
            + hashlib.sha256(
                (
                    _sha256_arquivo(caminho_features)
                    + _sha256_arquivo(caminho_empresas)
                ).encode("ascii")
            ).hexdigest()
        ),
    )

    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_parquet(caminho_saida, index=False)
    return resultado
