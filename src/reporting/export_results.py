"""Etapa 7: exportação de resultados e relatório de distribuição."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import pandas as pd

from src import config
from src.labeling.prepare_labels import (
    CONJUNTO_FORA,
    CONJUNTO_INFERENCIA,
    CONJUNTO_REVISAO,
    CONJUNTO_TREINO,
)

COLUNAS_RESULTADO = [
    "cnpj_raiz",
    "porte_original",
    "rotulo_conhecido",
    "porte_estimado",
    "prob_medio",
    "prob_grande",
    "status_classificacao",
    "revisao_recomendada",
    "motivo_revisao",
    "modelo_versao",
    "modelo_variante",
    "dados_versao",
    "data_execucao",
]

COLUNAS_PREVISAO = {
    "cnpj_raiz",
    "porte_original",
    "porte_estimado",
    "prob_medio",
    "prob_grande",
    "status_classificacao",
    "revisao_recomendada",
    "motivo_revisao",
    "modelo_versao",
    "modelo_variante",
    "dados_versao",
    "data_execucao",
}


def _validar_ids(frame: pd.DataFrame, descricao: str) -> pd.Series:
    if "cnpj_raiz" not in frame.columns:
        raise ValueError(f"{descricao} não contém a coluna cnpj_raiz.")
    if frame["cnpj_raiz"].isna().any():
        raise ValueError(f"{descricao} contém CNPJ raiz ausente.")
    ids = frame["cnpj_raiz"].astype(str)
    if ids.duplicated().any():
        raise ValueError(f"{descricao} contém CNPJ raiz duplicado.")
    return ids


def _valor_unico(frame: pd.DataFrame, coluna: str) -> str:
    if frame[coluna].isna().any():
        raise ValueError(f"Todas as previsões devem conter {coluna}.")
    valores = frame[coluna].dropna().astype(str).unique()
    if len(valores) != 1:
        raise ValueError(f"As previsões devem compartilhar um único valor em {coluna}.")
    return valores[0]


def montar_resultado_final(empresas: pd.DataFrame, previsoes: pd.DataFrame) -> pd.DataFrame:
    """Combina rótulos observados e previsões sem confundi-los ou incluir outros conjuntos."""
    colunas_empresas = {"cnpj_raiz", "conjunto", "porte_oficial", "rotulo", "rotulo_nome"}
    ausentes = sorted(colunas_empresas - set(empresas.columns))
    if ausentes:
        raise ValueError("A base consolidada está incompleta: " + ", ".join(ausentes))
    ausentes = sorted(COLUNAS_PREVISAO - set(previsoes.columns))
    if ausentes:
        raise ValueError("O arquivo de previsões está incompleto: " + ", ".join(ausentes))

    em_escopo = empresas.loc[
        empresas["conjunto"].isin([CONJUNTO_TREINO, CONJUNTO_INFERENCIA])
    ].copy()
    ids_escopo = _validar_ids(em_escopo, "Base consolidada em escopo")
    em_escopo["cnpj_raiz"] = ids_escopo

    treino = em_escopo.loc[em_escopo["conjunto"] == CONJUNTO_TREINO].copy()
    inferencia = em_escopo.loc[em_escopo["conjunto"] == CONJUNTO_INFERENCIA].copy()
    if treino.empty:
        raise ValueError("A base consolidada não contém rótulos conhecidos para exportar.")
    if inferencia.empty:
        raise ValueError("A base consolidada não contém empresas elegíveis à inferência.")

    rotulos = treino["rotulo"].astype("Int64")
    if rotulos.isna().any() or not rotulos.isin(config.CLASS_NAMES).all():
        raise ValueError("O conjunto de treino contém rótulos ausentes ou fora das classes Médio/Grande.")
    nomes_rotulos = rotulos.map(config.CLASS_NAMES).reset_index(drop=True).astype(str)
    nomes_origem = treino["rotulo_nome"].reset_index(drop=True).astype(str)
    if nomes_rotulos.tolist() != nomes_origem.tolist():
        raise ValueError("Os nomes dos rótulos conhecidos não correspondem aos rótulos numéricos.")

    previsoes = previsoes.copy()
    ids_previsoes = _validar_ids(previsoes, "Arquivo de previsões")
    previsoes["cnpj_raiz"] = ids_previsoes
    ids_inferencia = set(inferencia["cnpj_raiz"].astype(str))
    if set(ids_previsoes) != ids_inferencia:
        raise ValueError(
            "Os CNPJs raiz das previsões não correspondem exatamente ao conjunto de inferência."
        )
    if (
        previsoes["status_classificacao"].isna().any()
        or not previsoes["status_classificacao"].eq("estimativa_modelo").all()
    ):
        raise ValueError("O arquivo de previsões contém status diferente de estimativa_modelo.")
    if not previsoes["porte_estimado"].isin(list(config.CLASS_NAMES.values())).all():
        raise ValueError("O arquivo de previsões contém porte estimado fora das classes Médio/Grande.")
    probabilidades = previsoes[["prob_medio", "prob_grande"]].to_numpy(dtype="float64")
    if (
        not pd.notna(probabilidades).all()
        or (probabilidades < 0).any()
        or (probabilidades > 1).any()
        or not (abs(probabilidades.sum(axis=1) - 1.0) <= 1e-6).all()
    ):
        raise ValueError("O arquivo de previsões contém probabilidades inválidas.")

    porte_fonte = inferencia.set_index("cnpj_raiz")["porte_oficial"]
    porte_previsao = previsoes.set_index("cnpj_raiz")["porte_original"]
    porte_fonte = porte_fonte.reindex(porte_previsao.index)
    iguais = porte_previsao.eq(porte_fonte) | (porte_previsao.isna() & porte_fonte.isna())
    if not iguais.fillna(False).all():
        raise ValueError("O porte original das previsões diverge da base consolidada.")

    conhecidos = pd.DataFrame({
        "cnpj_raiz": treino["cnpj_raiz"].astype(str).to_numpy(),
        "porte_original": treino["porte_oficial"].to_numpy(),
        "rotulo_conhecido": nomes_rotulos.reset_index(drop=True).to_numpy(),
        "porte_estimado": pd.NA,
        "prob_medio": float("nan"),
        "prob_grande": float("nan"),
        "status_classificacao": "rotulo_conhecido",
        "revisao_recomendada": False,
        "motivo_revisao": "Rótulo conhecido na base; não foi feita inferência.",
        "modelo_versao": pd.NA,
        "modelo_variante": pd.NA,
        "dados_versao": pd.NA,
        "data_execucao": pd.NA,
    })
    estimativas = previsoes.copy()
    estimativas["rotulo_conhecido"] = pd.NA
    resultado = pd.concat([conhecidos, estimativas], ignore_index=True)
    resultado = resultado.loc[:, COLUNAS_RESULTADO]
    for coluna in (
        "cnpj_raiz",
        "porte_original",
        "rotulo_conhecido",
        "porte_estimado",
        "status_classificacao",
        "motivo_revisao",
        "modelo_versao",
        "modelo_variante",
        "dados_versao",
        "data_execucao",
    ):
        resultado[coluna] = resultado[coluna].astype("string")
    resultado = resultado.sort_values("cnpj_raiz", kind="stable").reset_index(drop=True)

    ids_resultado = _validar_ids(resultado, "Resultado final")
    if set(ids_resultado) != set(ids_escopo):
        raise ValueError("O resultado final não preservou exatamente as empresas em escopo.")
    return resultado


def construir_relatorio(empresas: pd.DataFrame, resultado: pd.DataFrame) -> dict:
    conhecidos = resultado.loc[resultado["status_classificacao"] == "rotulo_conhecido"]
    estimativas = resultado.loc[resultado["status_classificacao"] == "estimativa_modelo"]
    previsoes = estimativas
    metadados = {
        coluna: _valor_unico(previsoes, coluna)
        for coluna in ("modelo_versao", "modelo_variante", "dados_versao", "data_execucao")
    }
    return {
        "data_exportacao": datetime.now().astimezone().isoformat(timespec="seconds"),
        "total_registros": int(len(resultado)),
        "rotulos_conhecidos": int(len(conhecidos)),
        "estimativas_modelo": int(len(estimativas)),
        "rotulos_conhecidos_por_classe": {
            nome: int(conhecidos["rotulo_conhecido"].eq(nome).sum())
            for nome in config.CLASS_NAMES.values()
        },
        "estimativas_por_classe": {
            nome: int(estimativas["porte_estimado"].eq(nome).sum())
            for nome in config.CLASS_NAMES.values()
        },
        "revisao_recomendada": int(estimativas["revisao_recomendada"].sum()),
        "excluidas_revisao": int((empresas["conjunto"] == CONJUNTO_REVISAO).sum()),
        "excluidas_fora_do_escopo": int((empresas["conjunto"] == CONJUNTO_FORA).sum()),
        "modelo_versao": metadados["modelo_versao"],
        "modelo_variante": metadados["modelo_variante"],
        "dados_versao": metadados["dados_versao"],
        "data_execucao_inferencia": metadados["data_execucao"],
        "probabilidades_calibradas": False,
        "observacao": (
            "As probabilidades são exploratórias. Todas as estimativas estão sinalizadas para revisão "
            "manual porque não há calibração nem limiar de revisão validado."
        ),
    }


def render_relatorio(relatorio: dict) -> str:
    linhas = [
        "# Relatório de resultados da classificação de porte",
        "",
        f"- Exportado em: **{relatorio['data_exportacao']}**",
        f"- Registros exportados: **{relatorio['total_registros']}**",
        f"- Rótulos conhecidos: **{relatorio['rotulos_conhecidos']}**",
        f"- Estimativas do modelo: **{relatorio['estimativas_modelo']}**",
        f"- Estimativas recomendadas para revisão: **{relatorio['revisao_recomendada']}**",
        f"- Raízes excluídas para revisão: **{relatorio['excluidas_revisao']}**",
        f"- Raízes fora do escopo: **{relatorio['excluidas_fora_do_escopo']}**",
        "",
        "## Distribuição por classe",
        "",
        "| Classe | Rótulos conhecidos | Estimativas do modelo |",
        "|---|---:|---:|",
    ]
    for nome in config.CLASS_NAMES.values():
        linhas.append(
            f"| {nome} | {relatorio['rotulos_conhecidos_por_classe'][nome]} "
            f"| {relatorio['estimativas_por_classe'][nome]} |"
        )
    linhas += [
        "",
        "## Rastreabilidade",
        "",
        f"- Variante do modelo: `{relatorio['modelo_variante']}`",
        f"- Versão do modelo: `{relatorio['modelo_versao']}`",
        f"- Versão dos dados: `{relatorio['dados_versao']}`",
        f"- Execução da inferência: `{relatorio['data_execucao_inferencia']}`",
        "",
        relatorio["observacao"],
        "",
        "Rótulos conhecidos e estimativas permanecem em colunas e status distintos. "
        "O porte original não é sobrescrito.",
        "",
    ]
    return "\n".join(linhas)


def exportar_resultados(
    caminho_empresas: Path | None = None,
    caminho_previsoes: Path | None = None,
    caminho_csv: Path | None = None,
    caminho_parquet: Path | None = None,
    caminho_json: Path | None = None,
    caminho_md: Path | None = None,
) -> dict:
    """Valida e exporta o conjunto final, nos formatos tabular e relatório."""
    caminho_empresas = Path(caminho_empresas or config.PROCESSED_EMPRESAS)
    caminho_previsoes = Path(caminho_previsoes or config.PREDICTIONS_PARQUET)
    caminho_csv = Path(caminho_csv or config.RESULTADOS_CSV)
    caminho_parquet = Path(caminho_parquet or config.RESULTADOS_PARQUET)
    caminho_json = Path(caminho_json or config.REPORT_RESULTADOS_JSON)
    caminho_md = Path(caminho_md or config.REPORT_RESULTADOS_MD)

    for caminho, descricao, comando in (
        (caminho_empresas, "Base consolidada da Etapa 3", "python -m src.pipeline --etapa 3"),
        (caminho_previsoes, "Previsões da Etapa 6", "python -m src.pipeline --etapa 6"),
    ):
        if not caminho.is_file():
            raise FileNotFoundError(f"{descricao} não encontrado: {caminho}. Execute antes: {comando}")

    empresas = pd.read_parquet(caminho_empresas)
    previsoes = pd.read_parquet(caminho_previsoes)
    resultado = montar_resultado_final(empresas, previsoes)
    relatorio = construir_relatorio(empresas, resultado)

    for caminho in (caminho_csv, caminho_parquet, caminho_json, caminho_md):
        caminho.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(caminho_csv, index=False, encoding="utf-8-sig")
    resultado.to_parquet(caminho_parquet, index=False)
    caminho_json.write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    caminho_md.write_text(render_relatorio(relatorio), encoding="utf-8")
    relatorio["caminhos"] = {
        "csv": str(caminho_csv),
        "parquet": str(caminho_parquet),
        "json": str(caminho_json),
        "markdown": str(caminho_md),
    }
    return relatorio
