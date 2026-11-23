"""Leitura reproduzível da planilha de empresas e padronização de tipos básicos.

Princípios:
* Tudo é lido como texto (``dtype=str``) para que nenhum código perca zeros à esquerda.
* Os cabeçalhos são mapeados para *nomes lógicos* (ver ``config.COLUMN_ALIASES``).
* Identificadores/códigos (CNPJ, CNPJ raiz, CNAE, CEP, natureza jurídica) permanecem texto.
* Valores numéricos e datas são convertidos; o que não puder ser convertido vira nulo e é
  contabilizado em ``info["problemas_conversao"]`` para o relatório de qualidade.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from src import config


# ---------------------------------------------------------------------------
# Utilitários de texto
# ---------------------------------------------------------------------------
def normalize_text(value: object) -> str | None:
    """Minúsculas, sem acentos, pontuação -> espaço, espaços colapsados. Nulo -> None."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return text or None


def parse_number(value: object) -> float:
    """Converte texto numérico (formato BR ``1.234,56`` ou US ``1234.56``) em float."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return float("nan")
    text = str(value).strip().replace("R$", "").replace(" ", "")
    if not text:
        return float("nan")
    if "," in text:  # formato brasileiro: ponto = milhar, vírgula = decimal
        text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return float("nan")


# ---------------------------------------------------------------------------
# Localização e leitura do arquivo
# ---------------------------------------------------------------------------
def find_raw_file(raw_dir: Path | None = None, explicit: str | Path | None = None) -> Path:
    """Retorna o arquivo de entrada. Se ``explicit`` não for dado, exige um único arquivo
    em ``data/raw/`` (evita escolher silenciosamente o arquivo errado)."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"Arquivo não encontrado: {path}")
        return path

    raw_dir = Path(raw_dir or config.DATA_RAW)
    candidates = sorted(
        p for p in raw_dir.glob("*") if p.suffix.lower() in config.RAW_FILE_EXTENSIONS
        and not p.name.startswith("~$")
    )
    if not candidates:
        raise FileNotFoundError(
            f"Nenhuma planilha ({', '.join(config.RAW_FILE_EXTENSIONS)}) encontrada em {raw_dir}. "
            "Copie o arquivo original para data/raw/ ou use --arquivo."
        )
    if len(candidates) > 1:
        nomes = ", ".join(p.name for p in candidates)
        raise ValueError(
            f"Mais de um arquivo em {raw_dir}: {nomes}. Deixe apenas um ou use --arquivo."
        )
    return candidates[0]


def read_raw(path: Path, sheet: str | None = None) -> tuple[pd.DataFrame, str | None]:
    """Lê o arquivo bruto como texto. Retorna (DataFrame, nome da aba usada ou None)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        sheet = sheet or config.EXCEL_SHEET
        with pd.ExcelFile(path) as xls:
            used = sheet if sheet in xls.sheet_names else xls.sheet_names[0]
            df = pd.read_excel(xls, sheet_name=used, dtype=str)
        return df, used
    if suffix == ".csv":
        for encoding in ("utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(path, dtype=str, sep=None, engine="python", encoding=encoding)
                return df, None
            except UnicodeDecodeError:
                continue
        raise ValueError(f"Não foi possível decodificar {path}")
    raise ValueError(f"Extensão não suportada: {suffix}")


# ---------------------------------------------------------------------------
# Mapeamento de colunas e limpeza de tipos
# ---------------------------------------------------------------------------
def map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Renomeia cabeçalhos para nomes lógicos. Colunas não reconhecidas são mantidas
    com o nome original (úteis para auditoria) e listadas em ``nao_mapeadas``."""
    normalized = {col: normalize_text(col) for col in df.columns}
    rename: dict[str, str] = {}
    used_logical: set[str] = set()

    for logical, aliases in config.COLUMN_ALIASES.items():
        alias_set = {normalize_text(a) for a in aliases}
        for original, norm in normalized.items():
            if original in rename or norm is None:
                continue
            if norm in alias_set and logical not in used_logical:
                rename[original] = logical
                used_logical.add(logical)
                break

    mapped = df.rename(columns=rename)
    info = {
        "mapeamento": {orig: logical for orig, logical in rename.items()},
        "nao_mapeadas": [c for c in df.columns if c not in rename],
        "campos_logicos_ausentes": [f for f in config.COLUMN_ALIASES if f not in used_logical],
    }
    return mapped, info


def _clean_digits(series: pd.Series, width: int) -> pd.Series:
    """Mantém só dígitos, remove sufixo '.0' de números lidos como float e completa zeros."""
    s = series.astype("object").where(series.notna(), None)
    s = s.map(lambda v: re.sub(r"\.0+$", "", str(v).strip()) if v is not None else None)
    s = s.map(lambda v: re.sub(r"\D", "", v) if v is not None else None)
    s = s.map(lambda v: v.zfill(width) if v else None)
    return s.astype("object")


def _parse_dates(series: pd.Series) -> pd.Series:
    """Converte datas ISO (padrão do Excel/pandas); o que falhar tenta o formato dia/mês/ano."""
    s = series.astype("object").where(series.notna(), None)
    parsed = pd.to_datetime(s, format="ISO8601", errors="coerce")
    retry = parsed.isna() & s.notna()
    if retry.any():
        parsed.loc[retry] = pd.to_datetime(s[retry], dayfirst=True, errors="coerce", format="mixed")
    return parsed


def clean_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Padroniza tipos. Retorna (df, contagem de valores não convertidos)."""
    df = df.copy()
    problems: dict[str, int] = {}

    # Texto: remove espaços e transforma vazio em nulo
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype).startswith(("str", "string")):
            cleaned = df[col].astype("object").map(
                lambda v: v.strip() if isinstance(v, str) else v
            )
            df[col] = cleaned.map(lambda v: None if (isinstance(v, str) and v == "") else v)

    # Identificadores: sempre texto com zeros à esquerda
    for field, width in config.ID_WIDTHS.items():
        if field in df.columns:
            df[field] = _clean_digits(df[field], width)

    # CNPJ raiz derivado do CNPJ completo, quando não vier na planilha
    if "cnpj_raiz" not in df.columns and "cnpj" in df.columns:
        df["cnpj_raiz"] = df["cnpj"].map(lambda v: v[:8] if isinstance(v, str) else None)
    elif "cnpj_raiz" in df.columns and "cnpj" in df.columns:
        df["cnpj_raiz"] = df["cnpj_raiz"].where(
            df["cnpj_raiz"].notna(), df["cnpj"].map(lambda v: v[:8] if isinstance(v, str) else None)
        )

    # Numéricos
    for field in config.NUMERIC_FIELDS:
        if field in df.columns:
            raw_present = df[field].notna()
            df[field] = df[field].map(parse_number).astype("float64")
            problems[f"{field}_nao_numerico"] = int((raw_present & df[field].isna()).sum())

    # Datas
    for field in config.DATE_FIELDS:
        if field in df.columns:
            raw_present = df[field].notna()
            df[field] = _parse_dates(df[field])
            problems[f"{field}_invalida"] = int((raw_present & df[field].isna()).sum())

    return df, problems


# ---------------------------------------------------------------------------
# API principal
# ---------------------------------------------------------------------------
def load_dataset(
    path: str | Path | None = None, raw_dir: Path | None = None, sheet: str | None = None
) -> tuple[pd.DataFrame, dict]:
    """Lê, mapeia colunas e padroniza tipos. Retorna (df por estabelecimento, info)."""
    file_path = find_raw_file(raw_dir=raw_dir, explicit=path)
    df_raw, used_sheet = read_raw(file_path, sheet=sheet)
    df, mapping_info = map_columns(df_raw)
    df, problems = clean_types(df)
    info = {
        "arquivo": file_path.name,
        "aba": used_sheet,
        "linhas_lidas": int(len(df_raw)),
        "colunas_lidas": int(df_raw.shape[1]),
        "problemas_conversao": problems,
        **mapping_info,
    }
    return df, info


def save_interim(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Salva o DataFrame padronizado em Parquet (``data/interim/``)."""
    path = Path(path or config.INTERIM_ESTABELECIMENTOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_interim(path: Path | None = None) -> pd.DataFrame:
    """Lê o Parquet gerado por ``save_interim``."""
    return pd.read_parquet(Path(path or config.INTERIM_ESTABELECIMENTOS))
