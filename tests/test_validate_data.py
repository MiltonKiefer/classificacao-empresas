import json

import pandas as pd
import pytest

from src.data import load_data as ld
from src.data import validate_data as vd


def test_esquema_sem_coluna_obrigatoria_falha(sample_df, tmp_path):
    path = tmp_path / "x.csv"
    sample_df.drop(columns=["Obs Porte Sa3"]).to_csv(path, index=False)
    df, info = ld.load_dataset(path=path)
    with pytest.raises(vd.SchemaError):
        vd.validate_schema(df, info, strict=True)
    assert any(i["nivel"] == "erro" for i in vd.validate_schema(df, info, strict=False))


def test_esquema_valido_sem_erros(sample_xlsx):
    df, info = ld.load_dataset(path=sample_xlsx)
    assert not [i for i in vd.validate_schema(df, info) if i["nivel"] == "erro"]


def test_relatorio_de_qualidade(sample_xlsx, sample_df):
    df, info = ld.load_dataset(path=sample_xlsx)
    rep = vd.build_quality_report(df, info)
    assert rep["geral"]["linhas"] == len(sample_df)
    assert rep["chaves"]["cnpj_raiz_distintos"] == 60
    assert rep["chaves"]["raizes_com_mais_de_um_estabelecimento"] > 0
    assert rep["chaves"]["cnpj_com_tamanho_invalido"] == 0
    cap = rep["capital_social"]
    assert cap["zeros"] > 0 and cap["regra_zero_adotada"] == "zero_nao_define_porte"
    assert "por_grupo_de_rotulo" in cap["evidencias_semantica_zero"]
    assert 0 < rep["funcionarios"]["cobertura_pct"] < 100
    assert rep["rotulos_porte_obs"]["registros_por_grupo"]["medio"] > 0
    assert rep["rotulos_porte_obs"]["registros_por_grupo"]["grande"] > 0
    assert not any("capital social = 0" in p for p in rep["pendencias"])
    json.dumps(rep)  # deve ser serializável


def test_deteccao_de_duplicidades(sample_xlsx):
    df, info = ld.load_dataset(path=sample_xlsx)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    rep = vd.build_quality_report(dup, info)
    assert rep["chaves"]["linhas_totalmente_duplicadas"] == 1
    assert rep["chaves"]["linhas_com_cnpj_duplicado"] == 2


def test_escrita_dos_relatorios(sample_xlsx, tmp_path):
    df, info = ld.load_dataset(path=sample_xlsx)
    rep = vd.build_quality_report(df, info)
    j, m = vd.write_report(rep, tmp_path / "r.json", tmp_path / "r.md")
    assert json.loads(j.read_text(encoding="utf-8"))["geral"]["linhas"] == len(df)
    assert "Relatório de qualidade" in m.read_text(encoding="utf-8")
