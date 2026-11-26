"""Coordenação das etapas do projeto.

Uso (a partir da raiz do projeto, com o ambiente virtual ativado):

    python -m src.pipeline --etapa 2                      # usa o único arquivo de data/raw/
    python -m src.pipeline --etapa 2 --arquivo caminho.xlsx
    python -m src.pipeline --etapa 3                      # consolida o Parquet da Etapa 2

Etapas implementadas: 2 (ingestão + relatório de qualidade) e 3 (CNPJ raiz e rótulos).
Etapas 4 a 7: pendentes (ver README.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.data.load_data import load_dataset, load_interim, save_interim
from src.data.validate_data import build_quality_report, validate_schema, write_report
from src.entities.aggregate_root import (
    aggregate_by_root,
    build_consolidation_report,
    write_consolidation,
)


def run_ingestao(arquivo: str | Path | None = None) -> dict:
    """Etapa 2: lê a planilha, valida, gera relatório de qualidade e salva Parquet intermediário."""
    print("[1/4] Lendo planilha (pode levar alguns minutos para ~180 mil linhas)...")
    df, info = load_dataset(path=arquivo)
    print(f"      {info['linhas_lidas']:,} linhas x {info['colunas_lidas']} colunas ({info['arquivo']})".replace(",", "."))

    print("[2/4] Validando esquema...")
    for issue in validate_schema(df, info, strict=True):  # levanta SchemaError se faltar o essencial
        print(f"      {issue['nivel'].upper()}: {issue['mensagem']}")

    print("[3/4] Gerando relatório de qualidade...")
    report = build_quality_report(df, info)
    json_path, md_path = write_report(report)
    print(f"      {md_path}")

    print("[4/4] Salvando dados intermediários em Parquet...")
    parquet_path = save_interim(df)
    print(f"      {parquet_path}")

    if report["pendencias"]:
        print("\nPendências sinalizadas no relatório:")
        for p in report["pendencias"]:
            print(f"  - {p}")
    return report


def run_consolidacao() -> dict:
    """Etapa 3: lê o Parquet da Etapa 2 e grava uma linha por CNPJ raiz, com os rótulos."""
    if not config.INTERIM_ESTABELECIMENTOS.is_file():
        raise SystemExit(
            f"Parquet da Etapa 2 não encontrado: {config.INTERIM_ESTABELECIMENTOS}. "
            "Execute antes: python -m src.pipeline --etapa 2"
        )
    print("[1/3] Lendo estabelecimentos...")
    estabelecimentos = load_interim()
    print(f"      {len(estabelecimentos):,} linhas".replace(",", "."))

    print("[2/3] Consolidando por CNPJ raiz e preparando rótulos...")
    empresas, meta = aggregate_by_root(estabelecimentos)
    relatorio = build_consolidation_report(empresas, meta)

    print("[3/3] Gravando base e relatório...")
    parquet_path, _, md_path = write_consolidation(empresas, relatorio)
    print(f"      {parquet_path}")
    print(f"      {md_path}")
    print("\nConjuntos:")
    for nome in ("treino", "inferencia", "revisao", "fora_do_escopo"):
        print(f"  - {nome}: {relatorio['conjuntos'].get(nome, 0)}")
    if relatorio["treino_por_classe"]:
        print("Treino por classe:")
        for nome, qtd in relatorio["treino_por_classe"].items():
            print(f"  - {nome}: {qtd}")
    return relatorio


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de classificação de porte empresarial")
    parser.add_argument("--etapa", type=int, default=2, help="Etapa a executar (2 ou 3)")
    parser.add_argument("--arquivo", type=str, default=None, help="Caminho da planilha (opcional, Etapa 2)")
    args = parser.parse_args()

    if args.etapa == 2:
        run_ingestao(args.arquivo)
    elif args.etapa == 3:
        run_consolidacao()
    else:
        raise SystemExit(f"Etapa {args.etapa} ainda não implementada (ver README.md).")


if __name__ == "__main__":
    main()
