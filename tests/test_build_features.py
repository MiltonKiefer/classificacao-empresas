from datetime import date

import numpy as np
import pandas as pd
import pytest

from src import config
from src.labeling.prepare_labels import CONJUNTO_FORA, CONJUNTO_INFERENCIA, CONJUNTO_TREINO
from src.preprocessing.build_features import (
    build_feature_frame,
    feature_columns,
    make_preprocessor,
    matrix_xy,
)


def _empresa(**kwargs):
    base = {
        "cnpj_raiz": "00000001",
        "conjunto": CONJUNTO_TREINO,
        "rotulo": 0,
        "capital_social": 1000.0,
        "qtd_estabelecimentos": 1,
        "qtd_cnaes_distintos": 1,
        "qtd_divisoes_cnae": 1,
        "qtd_situacao_ativa": 1,
        "qtd_situacao_baixada": 0,
        "qtd_situacao_inapta": 0,
        "qtd_situacao_suspensa": 0,
        "qtd_situacao_nula": 0,
        "qtd_situacao_outra": 0,
        "data_inicio_empresa": pd.Timestamp("2000-10-06"),
        "tem_matriz": True,
        "natureza_juridica": "2062",
        "cnae_principal": "4711302",
        "situacao_matriz": "ATIVA",
        "funcionarios": 10,
        "porte_oficial": "DEMAIS",
        "razao_social": "EMPRESA",
        "rotulos_observados": "Média",
    }
    base.update(kwargs)
    return base


def test_capital_zero_nao_vira_ausencia_nem_some():
    frame = build_feature_frame(pd.DataFrame([
        _empresa(capital_social=0.0),
        _empresa(cnpj_raiz="00000002", capital_social=np.nan, conjunto=CONJUNTO_INFERENCIA, rotulo=pd.NA),
    ]))
    zero = frame.iloc[0]
    assert zero["capital_social_log"] == 0.0
    assert zero["capital_social_zero"] == 1.0
    assert pd.isna(frame.iloc[1]["capital_social_log"])
    assert pd.isna(frame.iloc[1]["capital_social_zero"])


def test_idade_usa_a_data_de_referencia_fixa():
    frame = build_feature_frame(
        pd.DataFrame([_empresa(data_inicio_empresa=pd.Timestamp("2016-10-06"))]),
        referencia=date(2026, 10, 6),
    )
    assert frame["tempo_atividade_anos"].item() == pytest.approx(10.0, abs=0.01)
    assert config.REFERENCE_DATE == date(2026, 10, 6)


def test_fora_de_escopo_nao_entra_e_rotulo_nao_e_feature():
    frame = build_feature_frame(pd.DataFrame([
        _empresa(conjunto=CONJUNTO_TREINO, rotulo=1),
        _empresa(cnpj_raiz="00000002", conjunto=CONJUNTO_FORA, rotulo=pd.NA),
        _empresa(cnpj_raiz="00000003", conjunto=CONJUNTO_INFERENCIA, rotulo=pd.NA),
    ]))
    assert set(frame["conjunto"]) == {CONJUNTO_TREINO, CONJUNTO_INFERENCIA}
    assert "funcionarios" not in feature_columns()
    assert "rotulo" not in feature_columns()
    assert "porte_oficial" not in feature_columns()
    assert "cnpj_raiz" not in feature_columns()
    assert "razao_social" not in feature_columns()
    x, y = matrix_xy(frame)
    assert list(x.columns) == feature_columns()
    assert list(y) == [1]
    assert frame.loc[frame["conjunto"] == CONJUNTO_TREINO, "divisao_cnae"].item() == "47"


def test_imputer_ajusta_so_no_treino():
    linhas = [
        _empresa(cnpj_raiz="00000001", capital_social=0.0, natureza_juridica="2062"),
        _empresa(cnpj_raiz="00000002", capital_social=0.0, natureza_juridica="2062", rotulo=1),
        _empresa(cnpj_raiz="00000003", conjunto=CONJUNTO_INFERENCIA, rotulo=pd.NA, capital_social=1_000_000_000.0),
    ]
    frame = build_feature_frame(pd.DataFrame(linhas))
    x, _ = matrix_xy(frame)
    treino = x.loc[frame["conjunto"] == CONJUNTO_TREINO]
    pre = make_preprocessor()
    pre.fit(treino)
    mediana = pre.named_transformers_["numeric"].named_steps["imputer"].statistics_[0]
    assert mediana == 0.0
    novo = treino.iloc[[0]].copy()
    novo["natureza_juridica"] = "9999"
    novo["capital_social_log"] = np.nan
    transformado = pre.transform(novo)
    assert transformado.shape[0] == 1
    assert np.isfinite(transformado).all()
