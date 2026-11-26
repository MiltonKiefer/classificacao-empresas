"""Etapa 3: rótulo binário Médio/Grande a partir de ``Obs Porte Sa3``.

A grafia é normalizada (minúsculas, sem acento). A convenção numérica fica em
``config.CLASS_MEDIO`` (0) e ``config.CLASS_GRANDE`` (1).

Uma raiz entra no treino somente quando há um único porte conhecido Médio ou Grande.
Médio e Grande na mesma raiz, ou um desses junto de Micro/Pequena, ou um texto
desconhecido, vão para revisão e ficam sem rótulo. Micro e Pequena ficam fora do
treino e não recebem estimativa. Vazio e "sem informação" não são rótulo: a raiz
segue para inferência se o porte oficial for DEMAIS.
"""
from __future__ import annotations

import pandas as pd

from src import config
from src.data.load_data import normalize_text

LABEL_MEDIO = "medio"
LABEL_GRANDE = "grande"
LABEL_PEQUENA = "pequena"
LABEL_MICRO = "micro"
LABEL_SEM = "sem_informacao"
LABEL_DESCONHECIDO = "desconhecido"

CONJUNTO_TREINO = "treino"
CONJUNTO_INFERENCIA = "inferencia"
CONJUNTO_REVISAO = "revisao"
CONJUNTO_FORA = "fora_do_escopo"

_TOKENS = {
    "media": LABEL_MEDIO,
    "medio": LABEL_MEDIO,
    "grande": LABEL_GRANDE,
    "pequena": LABEL_PEQUENA,
    "micro": LABEL_MICRO,
    "sem informacao": LABEL_SEM,
    "nao informado": LABEL_SEM,
    "nao informada": LABEL_SEM,
}

_SIZES = (LABEL_MEDIO, LABEL_GRANDE, LABEL_PEQUENA, LABEL_MICRO)


def map_porte_obs(values: pd.Series) -> pd.Series:
    """Mapeia o texto de ``porte_obs`` para um token canônico.

    Nulo permanece nulo. Texto fora da tabela vira ``desconhecido``.
    """
    norm = values.map(normalize_text)

    def one(value: str | None) -> str | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return _TOKENS.get(value, LABEL_DESCONHECIDO)

    return norm.map(one)


def assign_root_labels(token_counts: pd.DataFrame, porte_demais: pd.Series, porte_divergente: pd.Series) -> pd.DataFrame:
    """Define ``conjunto``, ``rotulo`` e ``motivo_conjunto`` por CNPJ raiz.

    ``token_counts`` tem uma coluna por token, com a contagem de estabelecimentos.
    ``porte_demais`` é verdadeiro quando todo porte oficial não nulo da raiz é DEMAIS.
    """
    idx = token_counts.index

    def present(name: str) -> pd.Series:
        if name not in token_counts.columns:
            return pd.Series(False, index=idx)
        return token_counts[name].fillna(0).astype(int) > 0

    medio = present(LABEL_MEDIO)
    grande = present(LABEL_GRANDE)
    pequena = present(LABEL_PEQUENA)
    micro = present(LABEL_MICRO)
    desconhecido = present(LABEL_DESCONHECIDO)
    n_tamanhos = medio.astype(int) + grande.astype(int) + pequena.astype(int) + micro.astype(int)

    revisao = desconhecido | (n_tamanhos > 1)
    treino = ~revisao & (n_tamanhos == 1) & (medio | grande)
    fora_tamanho = ~revisao & ~treino & (n_tamanhos >= 1)

    demais = porte_demais.reindex(idx)
    if demais.isna().any():
        demais = demais.fillna(False)
    divergente = porte_divergente.reindex(idx).fillna(False).astype(bool)
    fora_porte = ~revisao & ~treino & ~fora_tamanho & (~demais | divergente)

    conjunto = pd.Series(CONJUNTO_INFERENCIA, index=idx, dtype="object")
    conjunto = conjunto.mask(fora_tamanho | fora_porte, CONJUNTO_FORA)
    conjunto = conjunto.mask(revisao, CONJUNTO_REVISAO)
    conjunto = conjunto.mask(treino, CONJUNTO_TREINO)

    motivo = pd.Series("sem rotulo medio/grande", index=idx, dtype="object")
    motivo = motivo.mask(fora_porte & divergente, "porte oficial divergente na raiz")
    motivo = motivo.mask(fora_porte & ~divergente, "porte oficial diferente de DEMAIS")
    motivo = motivo.mask(fora_tamanho, "porte pequeno ou micro, fora do treino binario")
    motivo = motivo.mask(revisao & ~desconhecido, "rotulos de porte contraditorios na mesma raiz")
    motivo = motivo.mask(revisao & desconhecido, "rotulo desconhecido; revisar antes de rotular")
    motivo = motivo.mask(treino & grande, "rotulo grande consistente")
    motivo = motivo.mask(treino & medio, "rotulo medio consistente")

    rotulo = pd.Series(pd.NA, index=idx, dtype="Int64")
    rotulo = rotulo.mask(treino & medio, config.CLASS_MEDIO)
    rotulo = rotulo.mask(treino & grande, config.CLASS_GRANDE)
    nome = pd.Series(pd.NA, index=idx, dtype="object")
    nome = nome.mask(treino & medio, config.CLASS_NAMES[config.CLASS_MEDIO])
    nome = nome.mask(treino & grande, config.CLASS_NAMES[config.CLASS_GRANDE])

    return pd.DataFrame({
        "conjunto": conjunto,
        "motivo_conjunto": motivo,
        "rotulo": rotulo,
        "rotulo_nome": nome,
    })
