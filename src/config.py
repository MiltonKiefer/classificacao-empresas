"""Configuração central do projeto: caminhos, nomes de colunas, classes e parâmetros gerais.

Todo nome de coluna da planilha original é referenciado por um *nome lógico* (chave de
``COLUMN_ALIASES``). O restante do código só usa nomes lógicos; se a planilha mudar de
cabeçalho, basta ajustar os aliases aqui.
"""
from __future__ import annotations

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

# ---------------------------------------------------------------------------
# Fonte de dados
# ---------------------------------------------------------------------------
RAW_FILE_EXTENSIONS = (".xlsx", ".xls", ".csv")
EXCEL_SHEET = "Resultado da consulta"  # se não existir, usa-se a primeira aba

# ---------------------------------------------------------------------------
# Reprodutibilidade e classes
# ---------------------------------------------------------------------------
RANDOM_STATE = 42

CLASS_MEDIO = 0
CLASS_GRANDE = 1
CLASS_NAMES = {CLASS_MEDIO: "Médio", CLASS_GRANDE: "Grande"}

# ---------------------------------------------------------------------------
# Colunas da planilha
# ---------------------------------------------------------------------------
# nome lógico -> possíveis cabeçalhos (comparados após normalização: sem acentos,
# minúsculas, pontuação -> espaço). ⚠ PENDENTE: confirmar com a planilha real.
COLUMN_ALIASES: dict[str, list[str]] = {
    "cnpj": ["cnpj", "cnpj completo", "cnpj estabelecimento"],
    "cnpj_raiz": ["cnpj raiz", "cnpj basico", "raiz cnpj", "raiz"],
    "razao_social": ["razao social", "nome empresarial", "razao"],
    "capital_social": ["capital social"],
    "natureza_juridica": ["natureza juridica", "cod natureza juridica", "codigo natureza juridica"],
    "cnae_principal": ["cnae fiscal principal", "cnae principal", "cnae fiscal", "cnae"],
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
    "telefone": ["telefone", "telefone 1", "ddd telefone 1"],
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
# Decisões ainda pendentes (ver README / guia)
# ---------------------------------------------------------------------------
# Semântica do capital social igual a zero: "pendente" até confirmação com a fonte.
# Valores possíveis quando confirmado: "nao_informado" | "valor_real".
CAPITAL_SOCIAL_ZERO_RULE = "pendente"
