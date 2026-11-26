import pandas as pd
import pytest

from src.data import load_data as ld


def test_normalize_text():
    assert ld.normalize_text("  Obs Porte  SA3 ") == "obs porte sa3"
    assert ld.normalize_text("Média") == "media"
    assert ld.normalize_text(None) is None
    assert ld.normalize_text("---") is None


@pytest.mark.parametrize("raw,expected", [
    ("1.234,56", 1234.56), ("1234.56", 1234.56), ("R$ 10,00", 10.0),
    ("0", 0.0), ("abc", None), (None, None), ("", None),
])
def test_parse_number(raw, expected):
    out = ld.parse_number(raw)
    assert (pd.isna(out) and expected is None) or out == expected


def test_find_raw_file_erros(tmp_path):
    with pytest.raises(FileNotFoundError):
        ld.find_raw_file(raw_dir=tmp_path)
    (tmp_path / "a.csv").write_text("x")
    (tmp_path / "b.csv").write_text("x")
    with pytest.raises(ValueError):
        ld.find_raw_file(raw_dir=tmp_path)


def test_cabecalhos_da_planilha_real_mapeiam_cnae_e_telefone():
    df = pd.DataFrame({
        "Codigo Cnae Fiscal Principal": ["0111301"],
        "Telefone1": ["7133334444"],
        "CNPJ": ["00000001000191"],
        "Obs Porte Sa3": ["Grande"],
    })
    mapped, info = ld.map_columns(df)
    assert "cnae_principal" in mapped.columns
    assert "telefone" in mapped.columns
    assert "cnae_principal" not in info["campos_logicos_ausentes"]
    assert "telefone" not in info["campos_logicos_ausentes"]


def test_leitura_xlsx_mapeia_colunas_e_preserva_zeros(sample_xlsx):
    df, info = ld.load_dataset(path=sample_xlsx)
    assert info["aba"] == "Resultado da consulta"
    for field in ("cnpj", "cnpj_raiz", "capital_social", "natureza_juridica", "cnae_principal",
                  "matriz_filial", "situacao_cadastral", "data_inicio", "funcionarios",
                  "porte_oficial", "porte_obs", "cep"):
        assert field in df.columns, field
    assert info["nao_mapeadas"] == ["Coluna Extra"]
    # zeros à esquerda preservados
    assert df["cnpj_raiz"].iloc[0] == "00000001"
    assert (df["cnpj"].str.len() == 14).all()
    assert (df["cnae_principal"].str.len() == 7).all()
    assert df["natureza_juridica"].eq("0000").any()
    assert pd.api.types.is_datetime64_any_dtype(df["data_inicio"])
    assert df["capital_social"].dtype == "float64"


def test_leitura_csv_equivale_ao_xlsx(sample_xlsx, sample_csv):
    a, _ = ld.load_dataset(path=sample_xlsx)
    b, _ = ld.load_dataset(path=sample_csv)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)


def test_raiz_derivada_do_cnpj_quando_ausente(tmp_path, sample_df):
    path = tmp_path / "sem_raiz.csv"
    sample_df.drop(columns=["CNPJ Raiz"]).to_csv(path, index=False)
    df, _ = ld.load_dataset(path=path)
    assert (df["cnpj_raiz"] == df["cnpj"].str[:8]).all()


def test_cnpj_numerico_do_excel_recupera_zeros(tmp_path):
    path = tmp_path / "num.csv"
    pd.DataFrame({"CNPJ": ["123456780001.0"], "Obs Porte Sa3": ["Grande"]}).to_csv(path, index=False)
    df, _ = ld.load_dataset(path=path)
    assert df["cnpj"].iloc[0] == "00123456780001"
    assert df["cnpj_raiz"].iloc[0] == "00123456"


def test_valores_invalidos_sao_contados(tmp_path):
    path = tmp_path / "ruim.csv"
    pd.DataFrame({
        "CNPJ": ["12345678000190", "12345678000271"],
        "Capital Social": ["abc", "1.000,50"],
        "Data Início Atividade": ["00000000", "15/03/2010"],
        "Obs Porte Sa3": [None, None],
    }).to_csv(path, index=False)
    df, info = ld.load_dataset(path=path)
    assert info["problemas_conversao"]["capital_social_nao_numerico"] == 1
    assert info["problemas_conversao"]["data_inicio_invalida"] == 1
    assert df["capital_social"].iloc[1] == 1000.5
    assert df["data_inicio"].iloc[1] == pd.Timestamp("2010-03-15")


def test_roundtrip_parquet(sample_xlsx, tmp_path):
    df, _ = ld.load_dataset(path=sample_xlsx)
    out = ld.save_interim(df, tmp_path / "x.parquet")
    back = ld.load_interim(out)
    assert back["cnpj_raiz"].iloc[0] == "00000001"
    assert len(back) == len(df)
