import pandas as pd

from src import config
from src.model.evaluate import metricas
from src.model.train import executar_treinamento, peso_grande, validar
from src.preprocessing.build_features import build_feature_frame, feature_columns
from tests.test_build_features import _empresa


def _treino_sintetico():
    linhas = []
    for i in range(6):
        linhas.append(_empresa(
            cnpj_raiz=f"{i + 1:08d}",
            rotulo=0,
            capital_social=0.0 if i % 2 == 0 else 1000.0,
            natureza_juridica="2062" if i < 4 else "2135",
        ))
    for i in range(3):
        linhas.append(_empresa(
            cnpj_raiz=f"{i + 20:08d}",
            rotulo=1,
            capital_social=5_000_000.0,
            natureza_juridica="2062",
            qtd_estabelecimentos=4,
        ))
    frame = build_feature_frame(pd.DataFrame(linhas))
    return frame.loc[:, feature_columns()], frame["rotulo"].astype(int)


def test_metricas_separam_recall_de_medio_e_grande():
    bloco = metricas([0, 0, 1, 1], [0, 1, 1, 1], [0.2, 0.6, 0.7, 0.9])
    assert bloco["recall"]["Médio"] == 0.5
    assert bloco["recall"]["Grande"] == 1.0
    assert 0.0 <= bloco["f1_macro"] <= 1.0


def test_peso_da_classe_grande_usa_so_o_treino_do_fold():
    assert peso_grande([0, 0, 0, 1]) == 3.0


def test_validacao_estratificada_nao_calibra_e_a_maioria_erra_grande():
    x, y = _treino_sintetico()
    relatorio = validar(x, y, n_splits=3)
    assert relatorio["contagem"]["Médio"] == 6
    assert relatorio["contagem"]["Grande"] == 3
    assert relatorio["calibrador_aplicado"] is False
    assert set(relatorio["modelos"]) == {"maioria", "arvore", "xgb", "xgb_ponderado"}
    maioria = relatorio["modelos"]["maioria"]
    assert maioria["recall"]["Grande"] == 0.0
    assert sum(sum(linha) for linha in maioria["matriz_confusao"]) == 9
    assert "rotulo" not in x.columns


def test_execucao_grava_modelo_e_relatorio(tmp_path):
    x, y = _treino_sintetico()
    relatorio = executar_treinamento(
        x, y,
        identificadores=pd.Series([f"{i:08d}" for i in range(len(y))]),
        n_splits=3,
        caminho_modelo=tmp_path / "modelo.joblib",
        caminho_json=tmp_path / "avaliacao.json",
        caminho_md=tmp_path / "avaliacao.md",
    )
    assert (tmp_path / "modelo.joblib").is_file()
    texto = (tmp_path / "avaliacao.md").read_text(encoding="utf-8")
    assert "exploratórias" in texto
    assert relatorio["modelo_salvo"] in {"xgb", "xgb_ponderado"}
    assert relatorio["profundidade_final"] in (2, 3)
