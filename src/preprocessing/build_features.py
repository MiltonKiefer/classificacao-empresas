"""Etapa 4: atributos da empresa e pré-processamento sem vazamento de rótulo.

A matriz salva ainda está *antes* da imputação e da codificação. O
``ColumnTransformer`` é criado sem ajuste. Quem treina (Etapa 5) deve chamar
``fit`` só nas linhas de treino de cada partição.

Capital social igual a zero permanece zero (``log1p(0) = 0``) e ganha um
indicador próprio. Esse indicador não classifica o porte sozinho: entra junto
com as outras colunas. Funcionários ficam de fora: estão preenchidos nas 63
empresas rotuladas e quase vazios nas empresas que serão classificadas.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.labeling.prepare_labels import CONJUNTO_INFERENCIA, CONJUNTO_TREINO

NUMERIC_FEATURES: tuple[str, ...] = (
    "capital_social_log",
    "capital_social_zero",
    "qtd_estabelecimentos",
    "qtd_cnaes_distintos",
    "qtd_divisoes_cnae",
    "qtd_situacao_ativa",
    "qtd_situacao_baixada",
    "qtd_situacao_inapta",
    "qtd_situacao_suspensa",
    "qtd_situacao_nula",
    "qtd_situacao_outra",
    "tempo_atividade_anos",
    "tem_matriz",
)
CATEGORICAL_FEATURES: tuple[str, ...] = (
    "natureza_juridica",
    "divisao_cnae",
    "situacao_matriz",
)
FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES
MODELING_CONJUNTOS = (CONJUNTO_TREINO, CONJUNTO_INFERENCIA)

# Colunas que não podem aparecer na matriz, mesmo que existam na base por raiz.
_BLOQUEADAS = set(config.NEVER_FEATURES) | {
    "rotulo", "rotulo_nome", "conjunto", "motivo_conjunto", "rotulos_observados",
    "porte_obs", "funcionarios", "funcionarios_divergente",
}

_EXCLUIDAS: tuple[tuple[str, str], ...] = (
    ("cnpj_raiz", "Chave da empresa. Fica na tabela só para rastrear a linha, não entra no modelo."),
    ("razao_social", "Identificador textual."),
    ("rotulo", "Resposta do treino. Vira o vetor y, nunca uma coluna de X."),
    ("rotulo_nome", "O mesmo rótulo, em texto."),
    ("conjunto", "Define quem é treino e quem é inferência. Não descreve a empresa."),
    ("motivo_conjunto", "Explica a separação dos rótulos."),
    ("rotulos_observados", "Texto original de Obs Porte Sa3."),
    ("porte_oficial", "Porte oficial. Na base atual é constante (DEMAIS) e não pode revelar a resposta."),
    ("funcionarios", "Preenchido nas 63 empresas rotuladas e em quase nenhuma das demais. Usar esse campo treinaria o modelo com um dado que a inferência não tem."),
    ("cnae_principal", "Mais de mil códigos para 63 exemplos. No lugar entra a divisão (dois primeiros dígitos)."),
    ("capital_social", "Substituído por capital_social_log e capital_social_zero, para a escala não dominar o zero."),
    ("capital_divergente", "Na planilha real nenhuma raiz diverge. Coluna de auditoria."),
    ("natureza_divergente", "Na planilha real nenhuma raiz diverge. Coluna de auditoria."),
    ("data_inicio_empresa", "Vira tempo_atividade_anos até a data de referência."),
    ("data_inicio_matriz", "Auditoria. A idade usa a menor data da raiz."),
    ("filial_mais_antiga_que_matriz", "Só 30 raízes. Quase constante."),
    ("mais_de_uma_matriz", "Nenhuma raiz tem mais de uma matriz."),
    ("cnae_origem", "Diz de onde veio o CNAE. tem_matriz já cobre a ausência de matriz."),
    ("cnae_divergente", "Resumo de qtd_cnaes_distintos, que já entra como número."),
    ("porte_oficial_divergente", "Auditoria do porte oficial."),
)


def feature_columns() -> list[str]:
    """Nomes das colunas de X, na ordem consumida pelo pré-processador."""
    return list(FEATURE_COLUMNS)


def _checar_bloqueadas(colunas: list[str]) -> None:
    vazadas = sorted(set(colunas) & _BLOQUEADAS)
    if vazadas:
        raise ValueError("Coluna de identificador ou de rótulo entrou nas features: " + ", ".join(vazadas))


def _capital(empresas: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """log1p preserva o zero. Ausência continua ausência e não vira zero."""
    capital = pd.to_numeric(empresas["capital_social"], errors="coerce")
    logaritmo = pd.Series(np.nan, index=empresas.index, dtype="float64")
    validos = capital.notna() & capital.ge(0)
    logaritmo.loc[validos] = np.log1p(capital.loc[validos].to_numpy())
    indicador = pd.Series(np.nan, index=empresas.index, dtype="float64")
    indicador.loc[capital.notna() & capital.eq(0)] = 1.0
    indicador.loc[capital.notna() & capital.ne(0)] = 0.0
    return logaritmo, indicador


def _tempo_atividade(empresas: pd.DataFrame, referencia: date) -> pd.Series:
    inicio = pd.to_datetime(empresas["data_inicio_empresa"], errors="coerce")
    dias = (pd.Timestamp(referencia) - inicio).dt.days
    return (dias / 365.25).astype("float64")


def _divisao_cnae(empresas: pd.DataFrame) -> pd.Series:
    cnae = empresas["cnae_principal"]
    return cnae.map(lambda valor: valor[:2] if isinstance(valor, str) and len(valor) >= 2 else pd.NA)


def build_feature_frame(empresas: pd.DataFrame, referencia: date | None = None) -> pd.DataFrame:
    """Monta uma linha por empresa de treino ou inferência, com as colunas de X e o rótulo.

    Revisão e fora de escopo não entram: não treinam e não serão classificadas.
    """
    if "conjunto" not in empresas.columns or "cnpj_raiz" not in empresas.columns:
        raise ValueError("A base precisa das colunas conjunto e cnpj_raiz. Execute a Etapa 3 antes.")
    referencia = referencia or config.REFERENCE_DATE
    uso = empresas[empresas["conjunto"].isin(MODELING_CONJUNTOS)].copy()
    logaritmo, indicador = _capital(uso)
    frame = pd.DataFrame({
        "cnpj_raiz": uso["cnpj_raiz"].astype(str).to_numpy(),
        "conjunto": uso["conjunto"].astype(str).to_numpy(),
        "rotulo": uso["rotulo"].astype("Int64").to_numpy() if "rotulo" in uso.columns else pd.NA,
        "capital_social_log": logaritmo.to_numpy(),
        "capital_social_zero": indicador.to_numpy(),
        "qtd_estabelecimentos": pd.to_numeric(uso["qtd_estabelecimentos"], errors="coerce").to_numpy(),
        "qtd_cnaes_distintos": pd.to_numeric(uso["qtd_cnaes_distintos"], errors="coerce").to_numpy(),
        "qtd_divisoes_cnae": pd.to_numeric(uso["qtd_divisoes_cnae"], errors="coerce").to_numpy(),
        "qtd_situacao_ativa": pd.to_numeric(uso["qtd_situacao_ativa"], errors="coerce").to_numpy(),
        "qtd_situacao_baixada": pd.to_numeric(uso["qtd_situacao_baixada"], errors="coerce").to_numpy(),
        "qtd_situacao_inapta": pd.to_numeric(uso["qtd_situacao_inapta"], errors="coerce").to_numpy(),
        "qtd_situacao_suspensa": pd.to_numeric(uso["qtd_situacao_suspensa"], errors="coerce").to_numpy(),
        "qtd_situacao_nula": pd.to_numeric(uso["qtd_situacao_nula"], errors="coerce").to_numpy(),
        "qtd_situacao_outra": pd.to_numeric(uso["qtd_situacao_outra"], errors="coerce").to_numpy(),
        "tempo_atividade_anos": _tempo_atividade(uso, referencia).to_numpy(),
        "tem_matriz": uso["tem_matriz"].astype(float).to_numpy(),
        "natureza_juridica": uso["natureza_juridica"].astype("object").to_numpy(),
        "divisao_cnae": _divisao_cnae(uso).to_numpy(),
        "situacao_matriz": uso["situacao_matriz"].astype("object").to_numpy(),
    })
    _checar_bloqueadas(feature_columns())
    return frame.reset_index(drop=True)


def matrix_xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa X (treino e inferência) e y (somente treino, índice alinhado a X)."""
    x = frame.loc[:, feature_columns()].copy()
    y = frame.loc[frame["conjunto"] == CONJUNTO_TREINO, "rotulo"]
    if y.isna().any():
        raise ValueError("Há empresa de treino sem rótulo.")
    return x, y.astype(int)


def make_preprocessor() -> ColumnTransformer:
    """Pré-processador ainda sem ``fit``. Ajuste só no treino de cada partição."""
    numericos = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categoricos = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2, sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("numeric", numericos, list(NUMERIC_FEATURES)),
            ("categorical", categoricos, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def build_feature_report(frame: pd.DataFrame, n_colunas_codificadas: int | None = None) -> dict:
    treino = frame[frame["conjunto"] == CONJUNTO_TREINO]
    return {
        "referencia_temporal": config.REFERENCE_DATE.isoformat(),
        "regra_capital_social_zero": config.CAPITAL_SOCIAL_ZERO_RULE,
        "empresas": int(len(frame)),
        "treino": int(len(treino)),
        "inferencia": int((frame["conjunto"] == CONJUNTO_INFERENCIA).sum()),
        "treino_por_classe": {str(k): int(v) for k, v in treino["rotulo"].value_counts().sort_index().items()},
        "features": list(FEATURE_COLUMNS),
        "numericas": list(NUMERIC_FEATURES),
        "categoricas": list(CATEGORICAL_FEATURES),
        "excluidas": [{"coluna": nome, "motivo": motivo} for nome, motivo in _EXCLUIDAS],
        "colunas_codificadas_no_treino": n_colunas_codificadas,
        "preprocessamento": (
            "SimpleImputer e OneHotEncoder dentro de ColumnTransformer. "
            "O fit ocorre só nas linhas de treino de cada partição da Etapa 5. "
            "Esta etapa não grava um transformador já ajustado."
        ),
    }


def render_feature_markdown(relatorio: dict) -> str:
    linhas = [
        "# Dicionário de atributos",
        "",
        f"- Data de referência da idade da empresa: **{relatorio['referencia_temporal']}**",
        f"- Regra do capital zero: **{relatorio['regra_capital_social_zero']}**",
        f"- Empresas na matriz: **{relatorio['empresas']}** (treino {relatorio['treino']}, inferência {relatorio['inferencia']})",
        f"- Colunas de X, antes da codificação: **{len(relatorio['features'])}**",
        "",
        "Capital igual a zero vira `log1p` 0 e o indicador `capital_social_zero` = 1. "
        "Esse indicador entra junto com as outras colunas e não define o porte sozinho. "
        "Ausência de capital continua nula até a imputação, que só vê o treino.",
        "",
        "## Colunas numéricas",
        "",
    ]
    linhas += [f"- `{nome}`" for nome in relatorio["numericas"]]
    linhas += ["", "## Colunas categóricas", ""]
    linhas += [f"- `{nome}`" for nome in relatorio["categoricas"]]
    if relatorio.get("colunas_codificadas_no_treino") is not None:
        linhas += [
            "",
            f"Depois do ajuste só nas {relatorio['treino']} linhas de treino, "
            f"a matriz codificada tem **{relatorio['colunas_codificadas_no_treino']}** colunas.",
        ]
    linhas += ["", "## Fora do modelo", "", "| Coluna | Motivo |", "|---|---|"]
    for item in relatorio["excluidas"]:
        linhas.append(f"| `{item['coluna']}` | {item['motivo']} |")
    linhas += ["", relatorio["preprocessamento"], ""]
    return "\n".join(linhas)


def write_features(
    frame: pd.DataFrame,
    relatorio: dict,
    parquet_path: Path | None = None,
    json_path: Path | None = None,
    md_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    parquet_path = Path(parquet_path or config.FEATURES_PARQUET)
    json_path = Path(json_path or config.REPORT_FEATURES_JSON)
    md_path = Path(md_path or config.REPORT_FEATURES_MD)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(parquet_path, index=False)
    json_path.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_feature_markdown(relatorio), encoding="utf-8")
    return parquet_path, json_path, md_path
