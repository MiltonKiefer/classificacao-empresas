import pandas as pd

from src import config
from src.data import load_data as ld
from src.entities.aggregate_root import aggregate_by_root, build_consolidation_report
from src.labeling.prepare_labels import (
    CONJUNTO_FORA,
    CONJUNTO_INFERENCIA,
    CONJUNTO_REVISAO,
    CONJUNTO_TREINO,
    map_porte_obs,
)


def _linha(**kwargs):
    base = {
        "cnpj": "00000001000100",
        "cnpj_raiz": "00000001",
        "razao_social": "EMPRESA",
        "matriz_filial": "MATRIZ",
        "capital_social": 1000.0,
        "natureza_juridica": "2062",
        "cnae_principal": "0111301",
        "situacao_cadastral": "ATIVA",
        "data_inicio": pd.Timestamp("2000-01-01"),
        "funcionarios": None,
        "porte_oficial": "DEMAIS",
        "porte_obs": None,
    }
    base.update(kwargs)
    return base


def _empresas(linhas):
    empresas, _ = aggregate_by_root(pd.DataFrame(linhas))
    return empresas


def test_mapeamento_de_grafias():
    bruto = pd.Series(["Média", "médio", "Grande", "Pequena", "Micro", "Sem informação", None, "Foo"])
    obtido = list(map_porte_obs(bruto))
    assert obtido[:6] == ["medio", "medio", "grande", "pequena", "micro", "sem_informacao"]
    assert pd.isna(obtido[6])
    assert obtido[7] == "desconhecido"


def test_uma_linha_por_raiz_e_contagem_de_estabelecimentos():
    empresas = _empresas([
        _linha(cnpj="00000001000100", matriz_filial="MATRIZ"),
        _linha(cnpj="00000001000200", matriz_filial="FILIAL", situacao_cadastral="BAIXADA"),
        _linha(cnpj="00000002000100", cnpj_raiz="00000002", porte_obs="Grande"),
    ])
    assert list(empresas["cnpj_raiz"]) == ["00000001", "00000002"]
    assert empresas.loc[empresas["cnpj_raiz"] == "00000001", "qtd_estabelecimentos"].item() == 2
    assert empresas.loc[empresas["cnpj_raiz"] == "00000001", "qtd_situacao_ativa"].item() == 1
    assert empresas.loc[empresas["cnpj_raiz"] == "00000001", "qtd_situacao_baixada"].item() == 1


def test_capital_zero_e_preservado_e_nao_somado():
    empresas = _empresas([
        _linha(capital_social=0.0),
        _linha(cnpj="00000001000200", matriz_filial="FILIAL", capital_social=0.0),
    ])
    assert empresas["capital_social"].item() == 0.0
    assert not empresas["capital_divergente"].item()


def test_capital_divergente_usa_a_matriz():
    empresas = _empresas([
        _linha(capital_social=10.0, matriz_filial="MATRIZ"),
        _linha(cnpj="00000001000200", matriz_filial="FILIAL", capital_social=99.0),
    ])
    assert empresas["capital_social"].item() == 10.0
    assert empresas["capital_divergente"].item()


def test_capital_divergente_sem_matriz_fica_ausente():
    empresas = _empresas([
        _linha(matriz_filial="FILIAL", capital_social=10.0, data_inicio=pd.Timestamp("2010-01-01")),
        _linha(cnpj="00000001000200", matriz_filial="FILIAL", capital_social=99.0),
    ])
    assert pd.isna(empresas["capital_social"].item())
    assert empresas["capital_divergente"].item()
    assert not empresas["tem_matriz"].item()


def test_cnae_da_matriz_e_contagem_de_divergencia():
    empresas = _empresas([
        _linha(cnae_principal="0111301", matriz_filial="MATRIZ"),
        _linha(cnpj="00000001000200", matriz_filial="FILIAL", cnae_principal="4711302"),
    ])
    assert empresas["cnae_principal"].item() == "0111301"
    assert empresas["cnae_origem"].item() == "matriz"
    assert empresas["qtd_cnaes_distintos"].item() == 2
    assert empresas["cnae_divergente"].item()
    assert empresas["qtd_divisoes_cnae"].item() == 2


def test_sem_matriz_cnae_vem_do_estabelecimento_mais_antigo():
    empresas = _empresas([
        _linha(matriz_filial="FILIAL", cnae_principal="1111111", cnpj="00000001000200",
               data_inicio=pd.Timestamp("2015-01-01")),
        _linha(matriz_filial="FILIAL", cnae_principal="2222222", cnpj="00000001000100",
               data_inicio=pd.Timestamp("1999-01-01")),
    ])
    assert empresas["cnae_principal"].item() == "2222222"
    assert empresas["cnae_origem"].item() == "estabelecimento_mais_antigo"
    assert pd.isna(empresas["situacao_matriz"].item())
    assert pd.isna(empresas["data_inicio_matriz"].item())


def test_idade_usa_a_menor_data_e_guarda_a_da_matriz():
    empresas = _empresas([
        _linha(matriz_filial="MATRIZ", data_inicio=pd.Timestamp("2005-01-01")),
        _linha(cnpj="00000001000200", matriz_filial="FILIAL", data_inicio=pd.Timestamp("1998-06-01")),
    ])
    assert empresas["data_inicio_empresa"].item() == pd.Timestamp("1998-06-01")
    assert empresas["data_inicio_matriz"].item() == pd.Timestamp("2005-01-01")
    assert empresas["filial_mais_antiga_que_matriz"].item()


def test_funcionarios_ausentes_nao_viram_zero():
    empresas = _empresas([
        _linha(funcionarios=None),
        _linha(cnpj="00000001000200", matriz_filial="FILIAL", funcionarios=None),
    ])
    assert pd.isna(empresas["funcionarios"].item())
    assert not empresas["funcionarios_divergente"].item()

    zero = _empresas([_linha(funcionarios=0)])
    assert zero["funcionarios"].item() == 0


def test_rotulos_medio_grande_pequena_e_inferencia():
    empresas = _empresas([
        _linha(cnpj_raiz="00000001", porte_obs="Média"),
        _linha(cnpj_raiz="00000002", cnpj="00000002000100", porte_obs="Grande"),
        _linha(cnpj_raiz="00000003", cnpj="00000003000100", porte_obs="Pequena"),
        _linha(cnpj_raiz="00000004", cnpj="00000004000100", porte_obs="Micro"),
        _linha(cnpj_raiz="00000005", cnpj="00000005000100", porte_obs="Sem informação"),
        _linha(cnpj_raiz="00000006", cnpj="00000006000100", porte_obs=None),
    ])
    por_raiz = empresas.set_index("cnpj_raiz")
    assert por_raiz.loc["00000001", "conjunto"] == CONJUNTO_TREINO
    assert por_raiz.loc["00000001", "rotulo"] == config.CLASS_MEDIO
    assert por_raiz.loc["00000002", "rotulo"] == config.CLASS_GRANDE
    assert por_raiz.loc["00000002", "rotulo_nome"] == "Grande"
    assert por_raiz.loc["00000003", "conjunto"] == CONJUNTO_FORA
    assert pd.isna(por_raiz.loc["00000003", "rotulo"])
    assert por_raiz.loc["00000004", "conjunto"] == CONJUNTO_FORA
    assert por_raiz.loc["00000005", "conjunto"] == CONJUNTO_INFERENCIA
    assert por_raiz.loc["00000006", "conjunto"] == CONJUNTO_INFERENCIA


def test_conflito_medio_e_grande_vai_para_revisao_sem_maioria():
    empresas = _empresas([
        _linha(porte_obs="Média", matriz_filial="MATRIZ"),
        _linha(cnpj="00000001000200", matriz_filial="FILIAL", porte_obs="Grande"),
        _linha(cnpj="00000001000300", matriz_filial="FILIAL", porte_obs="Grande"),
    ])
    assert empresas["conjunto"].item() == CONJUNTO_REVISAO
    assert pd.isna(empresas["rotulo"].item())


def test_sem_informacao_nao_anula_rotulo_consistente():
    empresas = _empresas([
        _linha(porte_obs="Grande"),
        _linha(cnpj="00000001000200", matriz_filial="FILIAL", porte_obs="Sem informação"),
    ])
    assert empresas["conjunto"].item() == CONJUNTO_TREINO
    assert empresas["rotulo"].item() == config.CLASS_GRANDE


def test_rotulo_desconhecido_vai_para_revisao():
    empresas = _empresas([_linha(porte_obs="Holding")])
    assert empresas["conjunto"].item() == CONJUNTO_REVISAO


def test_amostra_sintetica_separa_conjuntos(sample_xlsx):
    df, _ = ld.load_dataset(path=sample_xlsx)
    empresas, meta = aggregate_by_root(df)
    relatorio = build_consolidation_report(empresas, meta)
    assert relatorio["raizes"] == df["cnpj_raiz"].nunique()
    assert set(empresas["conjunto"]) <= {CONJUNTO_TREINO, CONJUNTO_INFERENCIA, CONJUNTO_FORA, CONJUNTO_REVISAO}
    treino = empresas[empresas["conjunto"] == CONJUNTO_TREINO]
    assert set(treino["rotulo"]).issubset({0, 1})
    assert (empresas["capital_social"] == 0).any()
    assert empresas["qtd_estabelecimentos"].sum() == df["cnpj"].nunique()
