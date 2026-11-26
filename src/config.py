"""Configuração central do projeto: caminhos, nomes de colunas, classes e parâmetros gerais.

Todo nome de coluna da planilha original é referenciado por um *nome lógico* (chave de
``COLUMN_ALIASES``). O restante do código só usa nomes lógicos; se a planilha mudar de
cabeçalho, basta ajustar os aliases aqui.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_PREDICTIONS = PROJECT_ROOT / "outputs" / "predictions"
OUTPUT_REPORTS = PROJECT_ROOT / "outputs" / "reports"

# Arquivos gerados na Etapa 2
INTERIM_ESTABELECIMENTOS = DATA_INTERIM / "estabelecimentos.parquet"
REPORT_QUALIDADE_JSON = OUTPUT_REPORTS / "relatorio_qualidade.json"
REPORT_QUALIDADE_MD = OUTPUT_REPORTS / "relatorio_qualidade.md"

# Arquivos gerados na Etapa 3
PROCESSED_EMPRESAS = DATA_PROCESSED / "empresas_raiz.parquet"
REPORT_ROTULOS_JSON = OUTPUT_REPORTS / "relatorio_rotulos.json"
REPORT_ROTULOS_MD = OUTPUT_REPORTS / "relatorio_rotulos.md"

# Arquivos gerados na Etapa 4
FEATURES_PARQUET = DATA_PROCESSED / "features.parquet"
REPORT_FEATURES_JSON = OUTPUT_REPORTS / "dicionario_features.json"
REPORT_FEATURES_MD = OUTPUT_REPORTS / "dicionario_features.md"

# ---------------------------------------------------------------------------
# Fonte de dados
# ---------------------------------------------------------------------------
RAW_FILE_EXTENSIONS = (".xlsx", ".xls", ".csv")
EXCEL_SHEET = "Resultado da consulta"  # se não existir, usa-se a primeira aba

# ---------------------------------------------------------------------------
# Reprodutibilidade e classes
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
# Data do extrato OLAP (no nome do arquivo). A idade da empresa é contada até esta data,
# não até o dia em que o código é executado.
REFERENCE_DATE = date(2026, 10, 6)

CLASS_MEDIO = 0
CLASS_GRANDE = 1
CLASS_NAMES = {CLASS_MEDIO: "Médio", CLASS_GRANDE: "Grande"}

# ---------------------------------------------------------------------------
# Colunas da planilha
# ---------------------------------------------------------------------------
# nome lógico -> possíveis cabeçalhos (comparados após normalização: sem acentos,
# minúsculas, pontuação -> espaço). CNAE e telefone conferidos na planilha real.
COLUMN_ALIASES: dict[str, list[str]] = {
    "cnpj": ["cnpj", "cnpj completo", "cnpj estabelecimento"],
    "cnpj_raiz": ["cnpj raiz", "cnpj basico", "raiz cnpj", "raiz"],
    "razao_social": ["razao social", "nome empresarial", "razao"],
    "capital_social": ["capital social"],
    "natureza_juridica": ["natureza juridica", "cod natureza juridica", "codigo natureza juridica"],
    "cnae_principal": [
        "codigo cnae fiscal principal", "cnae fiscal principal", "cnae principal", "cnae fiscal", "cnae",
    ],
    "matriz_filial": ["matriz filial", "identificador matriz filial", "tipo estabelecimento"],
    "situacao_cadastral": ["situacao cadastral", "situacao"],
    "data_inicio": [
        "data inicio atividade", "data de inicio da atividade", "data inicio", "inicio atividade",
    ],
    "funcionarios": [
        "obs quantidade de funcionarios", "obs qtd funcionarios", "obs funcionarios",
        "quantidade de funcionarios", "qtd funcionarios",
    ],
    "porte_oficial": ["porte", "porte empresa", "porte da empresa", "porte oficial"],
    "porte_obs": ["obs porte sa3"],  # fonte do rótulo Médio/Grande
    "cep": ["cep"],
    "email": ["email", "e mail", "correio eletronico"],
    "telefone": ["telefone1", "telefone", "telefone 1", "ddd telefone 1"],
}

# Campos sem os quais a Etapa 2 não faz sentido
REQUIRED_FIELDS = ["porte_obs"]
REQUIRED_ONE_OF = [["cnpj", "cnpj_raiz"]]  # ao menos um destes
# Campos esperados para as etapas seguintes (ausência gera apenas aviso)
EXPECTED_FIELDS = [
    "capital_social", "natureza_juridica", "cnae_principal", "matriz_filial",
    "situacao_cadastral", "data_inicio", "funcionarios", "porte_oficial",
]

# Campos que são identificadores/códigos: sempre texto, preservando zeros à esquerda
ID_WIDTHS: dict[str, int] = {
    "cnpj": 14,
    "cnpj_raiz": 8,
    "cnae_principal": 7,
    "natureza_juridica": 4,
    "cep": 8,
}
NUMERIC_FIELDS = ["capital_social", "funcionarios"]
DATE_FIELDS = ["data_inicio"]

# Colunas que nunca podem virar feature (guia, seção 3.4)
NEVER_FEATURES = ["cnpj", "cnpj_raiz", "razao_social", "email", "telefone", "porte_obs", "porte_oficial"]

# ---------------------------------------------------------------------------
# Capital social igual a zero
# ---------------------------------------------------------------------------
# Regra adotada em 26/11/2026.
# Um valor 0 em uma métrica — capital social é o caso confirmado na planilha —
# não é evidência de pequeno porte e não classifica a empresa sozinho.
# O zero é preservado na base (não vira ausência nem NaN). A estimativa de
# Médio/Grande considera o conjunto das métricas disponíveis.
# "pendente" reabre o aviso no relatório de qualidade.
CAPITAL_SOCIAL_ZERO_RULE = "zero_nao_define_porte"
