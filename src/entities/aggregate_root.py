"""Etapa 3: uma linha por CNPJ raiz, com a agregação definida após a planilha real.

Regras (inspeção de 26/11/2026, 180.628 estabelecimentos e 163.658 raízes):

* A chave é ``cnpj_raiz``. Ela não é feature.
* ``qtd_estabelecimentos`` conta CNPJs completos da raiz.
* Capital social não é somado. Na planilha ele não varia dentro da raiz; o valor
  único é mantido, inclusive o zero. Se houver valores diferentes, usa-se o da
  matriz e a divergência fica marcada. Sem matriz e sem valor único, o campo
  fica ausente e a divergência continua marcada.
* Natureza jurídica segue a mesma regra do capital.
* CNAE principal é o da matriz. Filiais podem divergir: guardamos a quantidade
  de CNAEs e de divisões (dois primeiros dígitos). Sem matriz, ou com mais de
  uma, usa-se o CNAE do estabelecimento com data de início mais antiga
  (desempate pelo menor CNPJ).
* Situação cadastral da matriz fica em coluna própria. Não se escolhe uma filial
  no lugar dela. Contagens de ativas, baixadas, inaptas, suspensas e nulas
  acompanham a raiz. Há raízes sem matriz (8.227 na planilha real).
* A data da matriz coincide com a menor data da raiz em quase todos os casos,
  mas 30 raízes têm filial mais antiga. A idade da empresa usa a menor data de
  início válida; a data da matriz é preservada à parte.
* Funcionários: ausência continua ausência. O valor só é copiado quando é único
  na raiz. A cobertura é baixa fora das classes já rotuladas; a Etapa 4 decide
  se a coluna entra no modelo.
* Rótulos Médio/Grande contraditórios não são resolvidos por maioria.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src import config
from src.data.load_data import normalize_text
from src.labeling.prepare_labels import assign_root_labels, map_porte_obs

REGRAS_AGREGACAO: tuple[tuple[str, str], ...] = (
    ("cnpj_raiz", "Chave da empresa. Não entra como feature."),
    ("qtd_estabelecimentos", "Quantidade de CNPJs completos distintos da raiz."),
    ("capital_social", "Valor único da raiz, ou o da matriz se houver divergência. Nunca é somado. Zero é preservado."),
    ("natureza_juridica", "Valor único da raiz, ou o da matriz se houver divergência."),
    ("cnae_principal", "CNAE da matriz quando há exatamente uma. Senão, o do estabelecimento com início mais antigo."),
    ("qtd_cnaes_distintos", "CNAEs principais diferentes entre os estabelecimentos da raiz."),
    ("qtd_divisoes_cnae", "Divisões (dois primeiros dígitos) distintas na raiz."),
    ("situacao_matriz", "Situação cadastral da matriz. Filial não substitui matriz ausente."),
    ("qtd_situacao_*", "Contagem de estabelecimentos por situação cadastral."),
    ("data_inicio_empresa", "Menor data de início válida da raiz. É a referência de idade."),
    ("data_inicio_matriz", "Data de início da matriz, preservada para auditoria."),
    ("funcionarios", "Valor único observado. Ausência permanece nula e não vira zero."),
    ("rotulo", "0 = Médio e 1 = Grande, só quando o porte é único na raiz."),
    ("conjunto", "treino, inferencia, revisao ou fora_do_escopo."),
)

_SITUACOES = {
    "ativa": "ativa",
    "baixada": "baixada",
    "inapta": "inapta",
    "suspensa": "suspensa",
    "nula": "nula",
}


def _eh_matriz(values: pd.Series) -> pd.Series:
    norm = values.map(normalize_text)
    return norm.isin({"matriz", "1"}).fillna(False)


def _valor_consistente(df: pd.DataFrame, coluna: str, matriz: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Retorna (valor, divergente). Divergência com uma única matriz usa o valor da matriz."""
    raizes = pd.Index(df["cnpj_raiz"].unique())
    if coluna not in df.columns:
        return pd.Series(pd.NA, index=raizes, dtype="object"), pd.Series(False, index=raizes)
    distintos = df.groupby("cnpj_raiz", sort=False)[coluna].nunique(dropna=True)
    observado = df.dropna(subset=[coluna]).groupby("cnpj_raiz", sort=False)[coluna].first()
    valor = observado.where(distintos.reindex(observado.index).fillna(0).le(1)).reindex(distintos.index)
    divergente = distintos.gt(1)

    uma_matriz = matriz.groupby(df["cnpj_raiz"], sort=False).sum().eq(1)
    if matriz.any():
        da_matriz = df.loc[matriz].drop_duplicates("cnpj_raiz").set_index("cnpj_raiz")[coluna]
        usar = divergente & uma_matriz.reindex(divergente.index).fillna(False)
        valor = valor.mask(usar, da_matriz.reindex(valor.index))
    return valor, divergente


def _mais_antigo(df: pd.DataFrame, preferir_matriz: pd.Series, exigir: str | None = None) -> pd.DataFrame:
    """Uma linha por raiz. Matriz marcada tem prioridade; o desempate é a data e o CNPJ."""
    work = df
    pref = preferir_matriz.reindex(df.index).fillna(False)
    if exigir is not None and exigir in df.columns:
        work = df[df[exigir].notna()]
        pref = pref.reindex(work.index).fillna(False)
    if work.empty:
        return work
    ordem = work.assign(_matriz=pref.to_numpy())
    chaves = ["cnpj_raiz", "_matriz"]
    asc = [True, False]
    if "data_inicio" in ordem.columns:
        chaves.append("data_inicio")
        asc.append(True)
    if "cnpj" in ordem.columns:
        chaves.append("cnpj")
        asc.append(True)
    ordem = ordem.sort_values(chaves, ascending=asc, na_position="last")
    return ordem.groupby("cnpj_raiz", sort=False).head(1).set_index("cnpj_raiz")


def _contagens_situacao(df: pd.DataFrame, raizes: pd.Index) -> pd.DataFrame:
    colunas = [
        "qtd_situacao_ativa", "qtd_situacao_baixada", "qtd_situacao_inapta",
        "qtd_situacao_suspensa", "qtd_situacao_nula", "qtd_situacao_outra",
    ]
    if "situacao_cadastral" not in df.columns:
        return pd.DataFrame(0, index=raizes, columns=colunas)
    bruto = df["situacao_cadastral"].map(normalize_text)
    token = bruto.map(lambda v: _SITUACOES.get(v) if v else None)
    tabela = pd.crosstab(df.loc[token.notna(), "cnpj_raiz"], token.dropna())
    tabela = tabela.reindex(index=raizes, columns=["ativa", "baixada", "inapta", "suspensa", "nula", "outra"], fill_value=0)
    tabela.columns = colunas
    return tabela.astype(int)


def _porte_oficial(df: pd.DataFrame, raizes: pd.Index) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Retorna (valor, eh_demais, divergente) por raiz."""
    if "porte_oficial" not in df.columns:
        return (
            pd.Series(pd.NA, index=raizes, dtype="object"),
            pd.Series(True, index=raizes),
            pd.Series(False, index=raizes),
        )
    norm = df["porte_oficial"].map(normalize_text)
    distintos = norm.groupby(df["cnpj_raiz"], sort=False).nunique(dropna=True).reindex(raizes).fillna(0)
    valor = (
        df.dropna(subset=["porte_oficial"])
        .groupby("cnpj_raiz", sort=False)["porte_oficial"]
        .first()
        .reindex(raizes)
    )
    divergente = distintos.gt(1)
    unico = (
        df.assign(_p=norm)
        .dropna(subset=["_p"])
        .groupby("cnpj_raiz", sort=False)["_p"]
        .first()
        .reindex(raizes)
    )
    eh_demais = distintos.eq(0) | (distintos.eq(1) & unico.eq("demais"))
    return valor, eh_demais.fillna(False), divergente


def aggregate_by_root(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Consolida estabelecimentos em uma linha por CNPJ raiz.

    Retorna (empresas, metadados da entrada). Linhas sem ``cnpj_raiz`` são excluídas
    e contadas nos metadados.
    """
    if "cnpj_raiz" not in df.columns:
        raise ValueError("A coluna cnpj_raiz é necessária para a Etapa 3. Execute a Etapa 2 antes.")
    meta = {
        "linhas_entrada": int(len(df)),
        "linhas_sem_cnpj_raiz": int(df["cnpj_raiz"].isna().sum()),
    }
    base = df[df["cnpj_raiz"].notna()].copy()
    if base.empty:
        raise ValueError("Nenhuma linha com cnpj_raiz para consolidar.")
    if "cnpj" not in base.columns:
        base["cnpj"] = pd.NA
    if "data_inicio" not in base.columns:
        base["data_inicio"] = pd.NaT

    matriz = _eh_matriz(base["matriz_filial"]) if "matriz_filial" in base.columns else pd.Series(False, index=base.index)
    n_matrizes = matriz.groupby(base["cnpj_raiz"], sort=False).sum().astype(int)
    raizes = n_matrizes.index
    uma_matriz = n_matrizes.reindex(base["cnpj_raiz"]).eq(1).to_numpy()
    preferir = pd.Series(matriz.to_numpy() & uma_matriz, index=base.index)

    qtd_cnpj = base.groupby("cnpj_raiz", sort=False)["cnpj"].nunique(dropna=True)
    qtd_linhas = base.groupby("cnpj_raiz", sort=False).size()
    qtd = qtd_cnpj.where(qtd_cnpj > 0, qtd_linhas).astype(int).reindex(raizes)

    capital, capital_div = _valor_consistente(base, "capital_social", preferir)
    natureza, natureza_div = _valor_consistente(base, "natureza_juridica", preferir)

    if "cnae_principal" in base.columns:
        qtd_cnaes = base.groupby("cnpj_raiz", sort=False)["cnae_principal"].nunique(dropna=True).astype(int)
        divisao = base["cnae_principal"].map(lambda v: v[:2] if isinstance(v, str) and len(v) >= 2 else pd.NA)
        qtd_divisoes = divisao.groupby(base["cnpj_raiz"], sort=False).nunique(dropna=True).astype(int)
        fallback = _mais_antigo(base, pd.Series(False, index=base.index), exigir="cnae_principal")
        cnae = fallback["cnae_principal"].reindex(raizes) if len(fallback) else pd.Series(pd.NA, index=raizes)
        da_matriz = (
            base.loc[preferir & base["cnae_principal"].notna()]
            .drop_duplicates("cnpj_raiz")
            .set_index("cnpj_raiz")["cnae_principal"]
        )
        origem = pd.Series("ausente", index=raizes, dtype="object")
        origem = origem.mask(cnae.notna(), "estabelecimento_mais_antigo")
        origem = origem.mask(da_matriz.reindex(raizes).notna(), "matriz")
        cnae = cnae.mask(da_matriz.reindex(raizes).notna(), da_matriz.reindex(raizes))
    else:
        qtd_cnaes = pd.Series(0, index=raizes, dtype="int64")
        qtd_divisoes = pd.Series(0, index=raizes, dtype="int64")
        cnae = pd.Series(pd.NA, index=raizes, dtype="object")
        origem = pd.Series("ausente", index=raizes, dtype="object")

    ident = _mais_antigo(base, preferir)
    tem_uma = n_matrizes.eq(1)
    razao = ident["razao_social"].reindex(raizes) if "razao_social" in ident.columns else pd.Series(pd.NA, index=raizes)
    situacao_matriz = (
        ident["situacao_cadastral"].reindex(raizes).where(tem_uma)
        if "situacao_cadastral" in ident.columns else pd.Series(pd.NA, index=raizes)
    )
    data_matriz = ident["data_inicio"].reindex(raizes).where(tem_uma) if "data_inicio" in ident.columns else pd.Series(pd.NaT, index=raizes)
    data_empresa = base.groupby("cnpj_raiz", sort=False)["data_inicio"].min().reindex(raizes)
    filial_mais_antiga = data_matriz.notna() & data_empresa.notna() & (data_matriz > data_empresa)

    if "funcionarios" in base.columns:
        distintos_f = base.groupby("cnpj_raiz", sort=False)["funcionarios"].nunique(dropna=True)
        funcionarios = (
            base.dropna(subset=["funcionarios"]).groupby("cnpj_raiz", sort=False)["funcionarios"].first()
        )
        funcionarios = funcionarios.where(distintos_f.reindex(funcionarios.index).fillna(0).le(1)).reindex(raizes)
        funcionarios_div = distintos_f.gt(1).reindex(raizes).fillna(False)
    else:
        funcionarios = pd.Series(pd.NA, index=raizes, dtype="Float64")
        funcionarios_div = pd.Series(False, index=raizes)

    porte_valor, porte_demais, porte_div = _porte_oficial(base, raizes)
    if "porte_obs" in base.columns:
        tokens = map_porte_obs(base["porte_obs"]).fillna("__nenhum__")
        contagem = pd.crosstab(base["cnpj_raiz"], tokens).reindex(raizes, fill_value=0)
        if "__nenhum__" in contagem.columns:
            contagem = contagem.drop(columns="__nenhum__")
        observados = (
            base.dropna(subset=["porte_obs"])
            .groupby("cnpj_raiz", sort=False)["porte_obs"]
            .agg(lambda s: " | ".join(sorted({str(v) for v in s})))
        )
    else:
        contagem = pd.DataFrame(index=raizes)
        observados = pd.Series(dtype="object")
    rotulos = assign_root_labels(contagem, porte_demais, porte_div)
    situacao = _contagens_situacao(base, raizes)

    empresas = pd.DataFrame({
        "cnpj_raiz": raizes.astype(str),
        "conjunto": rotulos["conjunto"].reindex(raizes).to_numpy(),
        "motivo_conjunto": rotulos["motivo_conjunto"].reindex(raizes).to_numpy(),
        "rotulo": rotulos["rotulo"].reindex(raizes).astype("Int64").to_numpy(),
        "rotulo_nome": rotulos["rotulo_nome"].reindex(raizes).to_numpy(),
        "qtd_estabelecimentos": qtd.astype(int).to_numpy(),
        "tem_matriz": n_matrizes.ge(1).to_numpy(),
        "mais_de_uma_matriz": n_matrizes.gt(1).to_numpy(),
        "capital_social": pd.to_numeric(capital.reindex(raizes), errors="coerce").to_numpy(),
        "capital_divergente": capital_div.reindex(raizes).fillna(False).astype(bool).to_numpy(),
        "natureza_juridica": natureza.reindex(raizes).to_numpy(),
        "natureza_divergente": natureza_div.reindex(raizes).fillna(False).astype(bool).to_numpy(),
        "cnae_principal": cnae.reindex(raizes).to_numpy(),
        "cnae_origem": origem.reindex(raizes).to_numpy(),
        "qtd_cnaes_distintos": qtd_cnaes.reindex(raizes).fillna(0).astype(int).to_numpy(),
        "qtd_divisoes_cnae": qtd_divisoes.reindex(raizes).fillna(0).astype(int).to_numpy(),
        "cnae_divergente": qtd_cnaes.reindex(raizes).fillna(0).gt(1).to_numpy(),
        "situacao_matriz": situacao_matriz.to_numpy(),
        "data_inicio_empresa": data_empresa.to_numpy(),
        "data_inicio_matriz": data_matriz.to_numpy(),
        "filial_mais_antiga_que_matriz": filial_mais_antiga.fillna(False).astype(bool).to_numpy(),
        "funcionarios": pd.to_numeric(funcionarios.reindex(raizes), errors="coerce").to_numpy(),
        "funcionarios_divergente": funcionarios_div.fillna(False).astype(bool).to_numpy(),
        "porte_oficial": porte_valor.to_numpy(),
        "porte_oficial_divergente": porte_div.fillna(False).astype(bool).to_numpy(),
        "rotulos_observados": observados.reindex(raizes).to_numpy(),
        "razao_social": razao.to_numpy(),
    })
    empresas = pd.concat([empresas.reset_index(drop=True), situacao.reset_index(drop=True)], axis=1)
    empresas = empresas.sort_values("cnpj_raiz").reset_index(drop=True)
    empresas["rotulo"] = empresas["rotulo"].astype("Int64")
    meta["raizes"] = int(len(empresas))
    return empresas, meta


def build_consolidation_report(empresas: pd.DataFrame, meta: dict) -> dict:
    """Contagens por conjunto e por classe, mais o registro das divergências."""
    conjunto = empresas["conjunto"].value_counts().to_dict()
    treino = empresas[empresas["conjunto"] == "treino"]
    por_classe = treino["rotulo_nome"].value_counts(dropna=False).to_dict()
    return {
        "entrada": meta,
        "raizes": int(len(empresas)),
        "conjuntos": {str(k): int(v) for k, v in conjunto.items()},
        "treino_por_classe": {str(k): int(v) for k, v in por_classe.items()},
        "revisao": int((empresas["conjunto"] == "revisao").sum()),
        "sem_matriz": int((~empresas["tem_matriz"].astype(bool)).sum()),
        "mais_de_uma_matriz": int(empresas["mais_de_uma_matriz"].sum()),
        "capital_divergente": int(empresas["capital_divergente"].sum()),
        "capital_zero": int((empresas["capital_social"] == 0).sum()),
        "natureza_divergente": int(empresas["natureza_divergente"].sum()),
        "cnae_divergente": int(empresas["cnae_divergente"].sum()),
        "filial_mais_antiga_que_matriz": int(empresas["filial_mais_antiga_que_matriz"].sum()),
        "funcionarios_preenchidos": int(empresas["funcionarios"].notna().sum()),
        "funcionarios_preenchidos_no_treino": int(treino["funcionarios"].notna().sum()),
        "funcionarios_no_treino": int(len(treino)),
        "regras": [{"campo": campo, "regra": regra} for campo, regra in REGRAS_AGREGACAO],
    }


def render_consolidation_markdown(relatorio: dict) -> str:
    entrada = relatorio["entrada"]
    linhas = [
        "# Consolidação por CNPJ raiz",
        "",
        f"- Estabelecimentos lidos: **{entrada['linhas_entrada']}**",
        f"- Linhas sem CNPJ raiz, excluídas: **{entrada['linhas_sem_cnpj_raiz']}**",
        f"- Empresas (CNPJs raiz): **{relatorio['raizes']}**",
        "",
        "## Conjuntos",
        "",
        "| Conjunto | Empresas |",
        "|---|---|",
    ]
    for nome in ("treino", "inferencia", "revisao", "fora_do_escopo"):
        linhas.append(f"| {nome} | {relatorio['conjuntos'].get(nome, 0)} |")
    linhas += ["", "## Treino por classe", "", "| Classe | Empresas |", "|---|---|"]
    if relatorio["treino_por_classe"]:
        for nome, qtd in relatorio["treino_por_classe"].items():
            linhas.append(f"| {nome} | {qtd} |")
    else:
        linhas.append("| — | 0 |")
    linhas += [
        "",
        "## Conferências da agregação",
        "",
        f"- Raízes sem matriz: **{relatorio['sem_matriz']}**",
        f"- Raízes com mais de uma matriz: **{relatorio['mais_de_uma_matriz']}**",
        f"- Capital social divergente: **{relatorio['capital_divergente']}**",
        f"- Capital social igual a zero (valor preservado): **{relatorio['capital_zero']}**",
        f"- Natureza jurídica divergente: **{relatorio['natureza_divergente']}**",
        f"- CNAE divergente entre estabelecimentos: **{relatorio['cnae_divergente']}**",
        f"- Filial mais antiga que a matriz: **{relatorio['filial_mais_antiga_que_matriz']}**",
        f"- Funcionários preenchidos: **{relatorio['funcionarios_preenchidos']}** "
        f"({relatorio['funcionarios_preenchidos_no_treino']} de {relatorio['funcionarios_no_treino']} no treino)",
        "",
        "Ausência de funcionários não foi convertida em zero. Capital zero não classifica o porte.",
        "",
        "## Regras de agregação",
        "",
        "| Campo | Regra |",
        "|---|---|",
    ]
    for item in relatorio["regras"]:
        linhas.append(f"| `{item['campo']}` | {item['regra']} |")
    linhas.append("")
    return "\n".join(linhas)


def write_consolidation(
    empresas: pd.DataFrame,
    relatorio: dict,
    parquet_path: Path | None = None,
    json_path: Path | None = None,
    md_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    """Grava a base por raiz e o relatório de rótulos."""
    parquet_path = Path(parquet_path or config.PROCESSED_EMPRESAS)
    json_path = Path(json_path or config.REPORT_ROTULOS_JSON)
    md_path = Path(md_path or config.REPORT_ROTULOS_MD)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    empresas.to_parquet(parquet_path, index=False)
    json_path.write_text(json.dumps(relatorio, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    md_path.write_text(render_consolidation_markdown(relatorio), encoding="utf-8")
    return parquet_path, json_path, md_path
