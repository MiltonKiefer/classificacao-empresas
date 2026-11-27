import numpy as np
import pandas as pd
import pytest

from src import config
from src.labeling.prepare_labels import CONJUNTO_INFERENCIA, CONJUNTO_TREINO
from src.model.predict import gerar_previsoes
from src.preprocessing.build_features import feature_columns


class _ModeloFinal:
    classes_ = np.array([1, 0])


class _PipelineFalso:
    named_steps = {"modelo": _ModeloFinal()}

    def predict(self, x):
        return np.where(x["capital_social_log"].to_numpy() > 0, 1, 0)

    def predict_proba(self, x):
        resultados = []
        for valor in x["capital_social_log"]:
            resultados.append([0.8, 0.2] if valor > 0 else [0.1, 0.9])
        return np.asarray(resultados)


def _features(cnpj_raiz, conjunto, rotulo, capital):
    linha = {coluna: 0.0 for coluna in feature_columns()}
    linha.update({
        "cnpj_raiz": cnpj_raiz,
        "conjunto": conjunto,
        "rotulo": rotulo,
        "capital_social_log": capital,
        "natureza_juridica": "2062",
        "divisao_cnae": "47",
        "situacao_matriz": "ATIVA",
    })
    return linha


def _pacote_modelo():
    return {
        "pipeline": _PipelineFalso(),
        "features": feature_columns(),
        "classes": config.CLASS_NAMES,
        "variante": "xgb_ponderado",
    }


def test_gera_estimativas_apenas_para_inferencia_e_mapeia_classes():
    features = pd.DataFrame([
        _features("00000001", CONJUNTO_TREINO, 0, 0.0),
        _features("00000002", CONJUNTO_INFERENCIA, pd.NA, 1.0),
        _features("00000003", CONJUNTO_INFERENCIA, pd.NA, 0.0),
    ])
    empresas = pd.DataFrame([
        {"cnpj_raiz": "00000001", "conjunto": CONJUNTO_TREINO, "porte_oficial": "DEMAIS"},
        {"cnpj_raiz": "00000002", "conjunto": CONJUNTO_INFERENCIA, "porte_oficial": "DEMAIS"},
        {"cnpj_raiz": "00000003", "conjunto": CONJUNTO_INFERENCIA, "porte_oficial": "DEMAIS"},
    ])

    resultado = gerar_previsoes(
        features,
        empresas,
        _pacote_modelo(),
        modelo_versao="sha256:modelo",
        dados_versao="sha256:dados",
        data_execucao="2026-11-26T19:00:00-03:00",
    )

    assert resultado["cnpj_raiz"].tolist() == ["00000002", "00000003"]
    assert resultado["porte_estimado"].tolist() == ["Grande", "Médio"]
    assert resultado["prob_grande"].tolist() == [0.8, 0.1]
    assert resultado["prob_medio"].tolist() == [0.2, 0.9]
    assert resultado["porte_original"].tolist() == ["DEMAIS", "DEMAIS"]
    assert set(resultado["status_classificacao"]) == {"estimativa_modelo"}
    assert resultado["revisao_recomendada"].all()
    assert set(resultado["modelo_versao"]) == {"sha256:modelo"}
    assert set(resultado["dados_versao"]) == {"sha256:dados"}


def test_rejeita_rotulo_conhecido_no_conjunto_de_inferencia():
    features = pd.DataFrame([
        _features("00000002", CONJUNTO_INFERENCIA, 1, 1.0),
    ])
    empresas = pd.DataFrame([
        {"cnpj_raiz": "00000002", "conjunto": CONJUNTO_INFERENCIA, "porte_oficial": "DEMAIS"},
    ])

    with pytest.raises(ValueError, match="rótulo conhecido"):
        gerar_previsoes(
            features,
            empresas,
            _pacote_modelo(),
            modelo_versao="sha256:modelo",
            dados_versao="sha256:dados",
        )


def test_rejeita_cnpjs_raiz_diferentes_entre_features_e_base():
    features = pd.DataFrame([
        _features("00000002", CONJUNTO_INFERENCIA, pd.NA, 1.0),
    ])
    empresas = pd.DataFrame([
        {"cnpj_raiz": "00000003", "conjunto": CONJUNTO_INFERENCIA, "porte_oficial": "DEMAIS"},
    ])

    with pytest.raises(ValueError, match="mesmo conjunto de CNPJs raiz"):
        gerar_previsoes(
            features,
            empresas,
            _pacote_modelo(),
            modelo_versao="sha256:modelo",
            dados_versao="sha256:dados",
        )
