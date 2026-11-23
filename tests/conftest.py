import pytest

from tests.helpers import make_sample_dataframe


@pytest.fixture
def sample_df():
    return make_sample_dataframe()


@pytest.fixture
def sample_xlsx(tmp_path, sample_df):
    path = tmp_path / "amostra.xlsx"
    sample_df.to_excel(path, index=False, sheet_name="Resultado da consulta")
    return path


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    path = tmp_path / "amostra.csv"
    sample_df.to_csv(path, index=False, sep=";", encoding="utf-8-sig")
    return path
