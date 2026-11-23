"""Coordenação das etapas do projeto.

Uso (a partir da raiz do projeto, com o ambiente virtual ativado):

    python -m src.pipeline --etapa 2                      # usa o único arquivo de data/raw/
    python -m src.pipeline --etapa 2 --arquivo caminho.xlsx

Etapas implementadas: 2 (ingestão + relatório de qualidade).
Etapas 3 a 7: pendentes (ver README.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src import config
from src.data.load_data import load_dataset, save_interim
from src.data.validate_data import build_quality_report, validate_schema, write_report


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de classificação de porte empresarial")
    parser.add_argument("--etapa", type=int, default=2, help="Etapa a executar (hoje: apenas 2)")
    parser.add_argument("--arquivo", type=str, default=None, help="Caminho da planilha (opcional)")
    args = parser.parse_args()

    if args.etapa == 2:
        run_ingestao(args.arquivo)
    else:
        raise SystemExit(f"Etapa {args.etapa} ainda não implementada (ver README.md).")


if __name__ == "__main__":
    main()
