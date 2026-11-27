from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src import config
from src.data.load_data import load_dataset, load_interim, save_interim
from src.data.validate_data import build_quality_report, validate_schema, write_report
from src.entities.aggregate_root import (
    aggregate_by_root,
    build_consolidation_report,
    write_consolidation,
)
from src.model.predict import executar_inferencia
from src.model.train import executar_treinamento
from src.preprocessing.build_features import (
    build_feature_frame,
    build_feature_report,
    feature_columns,
    make_preprocessor,
    write_features,
)
from src.reporting.export_results import exportar_resultados


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


def run_features() -> dict:
    """Etapa 4: monta X e y a partir da base por CNPJ raiz, sem ajustar o pré-processador."""
    if not config.PROCESSED_EMPRESAS.is_file():
        raise SystemExit(
            f"Base da Etapa 3 não encontrada: {config.PROCESSED_EMPRESAS}. "
            "Execute antes: python -m src.pipeline --etapa 3"
        )
    print("[1/3] Lendo empresas por CNPJ raiz...")
    empresas = pd.read_parquet(config.PROCESSED_EMPRESAS)
    print(f"      {len(empresas):,} empresas".replace(",", "."))

    print("[2/3] Montando atributos...")
    frame = build_feature_frame(empresas)
    treino = frame.loc[frame["conjunto"] == "treino", feature_columns()]
    preprocessador = make_preprocessor()
    preprocessador.fit(treino)
    relatorio = build_feature_report(frame, n_colunas_codificadas=len(preprocessador.get_feature_names_out()))

    print("[3/3] Gravando matriz e dicionário...")
    parquet_path, _, md_path = write_features(frame, relatorio)
    print(f"      {parquet_path}")
    print(f"      {md_path}")
    print(f"      treino: {relatorio['treino']} | inferência: {relatorio['inferencia']}")
    print(f"      colunas de X: {len(relatorio['features'])} | codificadas no treino: {relatorio['colunas_codificadas_no_treino']}")
    return relatorio


def run_treino() -> dict:
    """Etapa 5: valida o XGBoost nas empresas rotuladas e grava o modelo."""
    if not config.FEATURES_PARQUET.is_file():
        raise SystemExit(
            f"Matriz da Etapa 4 não encontrada: {config.FEATURES_PARQUET}. "
            "Execute antes: python -m src.pipeline --etapa 4"
        )
    print("[1/2] Validando com 3 folds estratificados (pode levar um minuto)...")
    relatorio = executar_treinamento()
    print("[2/2] Modelo e relatório gravados.")
    print(f"      {relatorio['caminhos']['markdown']}")
    print(f"      {relatorio['caminhos']['modelo']}")
    print(f"      modelo salvo: {relatorio['modelo_salvo']} (profundidade {relatorio['profundidade_final']})")
    for nome, bloco in relatorio["modelos"].items():
        print(
            f"      {nome}: F1-macro {bloco['f1_macro']:.3f} | "
            f"recall Médio {bloco['recall']['Médio']:.3f} | recall Grande {bloco['recall']['Grande']:.3f}"
        )
    return relatorio


def run_inferencia() -> dict:
    """Etapa 6: estima o porte das empresas sem rótulo conhecido."""
    print("[1/2] Carregando modelo e preparando empresas elegíveis...")
    resultado = executar_inferencia()
    print("[2/2] Previsões gravadas.")
    print(f"      {config.PREDICTIONS_PARQUET}")
    print(f"      empresas: {len(resultado):,}".replace(",", "."))
    print(
        f"      Médio: {(resultado['porte_estimado'] == config.CLASS_NAMES[config.CLASS_MEDIO]).sum():,} | "
        f"Grande: {(resultado['porte_estimado'] == config.CLASS_NAMES[config.CLASS_GRANDE]).sum():,}"
    )
    print("      todas as estimativas estão sinalizadas para revisão manual.")
    return {
        "empresas": int(len(resultado)),
        "por_classe": resultado["porte_estimado"].value_counts().to_dict(),
        "caminho": str(config.PREDICTIONS_PARQUET),
    }


def run_exportacao() -> dict:
    """Etapa 7: exporta rótulos conhecidos e estimativas em saídas distintas."""
    print("[1/2] Validando e consolidando os resultados...")
    relatorio = exportar_resultados()
    print("[2/2] CSV, Parquet e relatórios gravados.")
    print(f"      {relatorio['caminhos']['csv']}")
    print(f"      {relatorio['caminhos']['parquet']}")
    print(f"      {relatorio['caminhos']['markdown']}")
    print(
        f"      rótulos conhecidos: {relatorio['rotulos_conhecidos']:,} | "
        f"estimativas: {relatorio['estimativas_modelo']:,} | "
        f"revisão recomendada: {relatorio['revisao_recomendada']:,}"
    )
    return relatorio


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de classificação de porte empresarial")
    parser.add_argument("--etapa", type=int, default=2, help="Etapa a executar (2 a 7)")
    parser.add_argument("--arquivo", type=str, default=None, help="Caminho da planilha (opcional, Etapa 2)")
    args = parser.parse_args()

    if args.etapa == 2:
        run_ingestao(args.arquivo)
    elif args.etapa == 3:
        run_consolidacao()
    elif args.etapa == 4:
        run_features()
    elif args.etapa == 5:
        run_treino()
    elif args.etapa == 6:
        run_inferencia()
    elif args.etapa == 7:
        run_exportacao()
    else:
        raise SystemExit(f"Etapa {args.etapa} ainda não implementada (ver README.md).")


if __name__ == "__main__":
    main()
