import pandas as pd
import pytest

from src.labeling.prepare_labels import (
    CONJUNTO_FORA,
    CONJUNTO_INFERENCIA,
    CONJUNTO_REVISAO,
    CONJUNTO_TREINO,
)
from src.reporting.export_results import (
    construir_relatorio,
    exportar_resultados,
    montar_resultado_final,
)


def _empresas():
    empresas = pd.DataFrame([
        {
            "cnpj_raiz": "00000001",
            "conjunto": CONJUNTO_TREINO,
            "porte_oficial": "DEMAIS",
            "rotulo": 0,
            "rotulo_nome": "Médio",
        },
        {
            "cnpj_raiz": "00000002",
            "conjunto": CONJUNTO_TREINO,
            "porte_oficial": "DEMAIS",
            "rotulo": 1,
            "rotulo_nome": "Grande",
        },
        {
            "cnpj_raiz": "00000003",
            "conjunto": CONJUNTO_INFERENCIA,
            "porte_oficial": "DEMAIS",
            "rotulo": pd.NA,
            "rotulo_nome": pd.NA,
        },
        {
            "cnpj_raiz": "00000004",
            "conjunto": CONJUNTO_REVISAO,
            "porte_oficial": "DEMAIS",
            "rotulo": pd.NA,
            "rotulo_nome": pd.NA,
        },
        {
            "cnpj_raiz": "00000005",
            "conjunto": CONJUNTO_FORA,
            "porte_oficial": "MICRO",
            "rotulo": pd.NA,
            "rotulo_nome": pd.NA,
        },
    ])
    empresas["rotulo"] = empresas["rotulo"].astype("Int64")
    empresas["rotulo_nome"] = empresas["rotulo_nome"].astype("string")
    return empresas


def _previsoes():
    return pd.DataFrame([{
        "cnpj_raiz": "00000003",
        "porte_original": "DEMAIS",
        "porte_estimado": "Grande",
        "prob_medio": 0.2,
        "prob_grande": 0.8,
        "status_classificacao": "estimativa_modelo",
        "revisao_recomendada": True,
        "motivo_revisao": "revisão manual",
        "modelo_versao": "sha256:modelo",
        "modelo_variante": "xgb_ponderado",
        "dados_versao": "sha256:dados",
        "data_execucao": "2026-11-26T19:00:00-03:00",
    }])


def test_export_preserva_rotulos_conhecidos_separados_de_estimativas():
    empresas = _empresas()
    resultado = montar_resultado_final(empresas, _previsoes())

    conhecidos = resultado.loc[resultado["status_classificacao"] == "rotulo_conhecido"]
    estimativas = resultado.loc[resultado["status_classificacao"] == "estimativa_modelo"]
    assert resultado["cnpj_raiz"].tolist() == ["00000001", "00000002", "00000003"]
    assert conhecidos["rotulo_conhecido"].tolist() == ["Médio", "Grande"]
    assert conhecidos["porte_estimado"].isna().all()
    assert conhecidos["prob_grande"].isna().all()
    assert estimativas["rotulo_conhecido"].isna().all()
    assert estimativas["porte_estimado"].tolist() == ["Grande"]
    assert estimativas["porte_original"].tolist() == ["DEMAIS"]

    relatorio = construir_relatorio(empresas, resultado)
    assert relatorio["rotulos_conhecidos"] == 2
    assert relatorio["estimativas_modelo"] == 1
    assert relatorio["estimativas_por_classe"]["Grande"] == 1
    assert relatorio["revisao_recomendada"] == 1
    assert relatorio["excluidas_revisao"] == 1
    assert relatorio["excluidas_fora_do_escopo"] == 1


def test_export_rejeita_previsoes_que_nao_cobrem_todo_conjunto_de_inferencia():
    previsoes = _previsoes()
    previsoes.loc[0, "cnpj_raiz"] = "00000009"

    with pytest.raises(ValueError, match="não correspondem exatamente"):
        montar_resultado_final(_empresas(), previsoes)


def test_export_rejeita_probabilidades_invalidas():
    previsoes = _previsoes()
    previsoes.loc[0, "prob_grande"] = 0.7

    with pytest.raises(ValueError, match="probabilidades inválidas"):
        montar_resultado_final(_empresas(), previsoes)


def test_exportar_resultados_grava_csv_parquet_e_relatorios(tmp_path):
    empresas_path = tmp_path / "empresas.parquet"
    previsoes_path = tmp_path / "previsoes.parquet"
    csv_path = tmp_path / "saida" / "resultado.csv"
    parquet_path = tmp_path / "saida" / "resultado.parquet"
    json_path = tmp_path / "relatorios" / "resultado.json"
    md_path = tmp_path / "relatorios" / "resultado.md"
    _empresas().to_parquet(empresas_path, index=False)
    _previsoes().to_parquet(previsoes_path, index=False)

    relatorio = exportar_resultados(
        caminho_empresas=empresas_path,
        caminho_previsoes=previsoes_path,
        caminho_csv=csv_path,
        caminho_parquet=parquet_path,
        caminho_json=json_path,
        caminho_md=md_path,
    )

    assert csv_path.is_file()
    assert parquet_path.is_file()
    assert json_path.is_file()
    assert md_path.is_file()
    assert len(pd.read_parquet(parquet_path)) == 3
    assert relatorio["caminhos"]["csv"] == str(csv_path)
