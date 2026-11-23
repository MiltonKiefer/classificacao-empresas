"""Validação estrutural e relatório de qualidade dos dados por estabelecimento.

Duas camadas:
* ``validate_schema``: confere se as colunas indispensáveis existem (erro) e se as esperadas
  estão presentes (aviso).
* ``build_quality_report``: estatísticas de nulos, zeros, duplicidades, datas, categorias,
  cobertura de capital social e funcionários e evidências sobre o significado de capital = 0.

Nada aqui decide a semântica do capital social igual a zero: o relatório apenas reúne
evidências. A decisão é registrada em ``config.CAPITAL_SOCIAL_ZERO_RULE``.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.data.load_data import normalize_text

TOP_N = 10
# Agrupamento *apenas informativo* do texto bruto de `porte_obs` para o relatório.
# O mapeamento oficial do rótulo binário é feito na Etapa 3 (labeling/prepare_labels.py).
_GRUPO_MEDIO = {"media", "medio"}
_GRUPO_GRANDE = {"grande"}


class SchemaError(ValueError):
    """Estrutura da planilha incompatível com o esperado."""


# ---------------------------------------------------------------------------
# Validação de esquema
# ---------------------------------------------------------------------------
def validate_schema(df: pd.DataFrame, info: dict | None = None, strict: bool = True) -> list[dict]:
    """Retorna lista de problemas ``{"nivel": "erro"|"aviso", "mensagem": str}``.
    Com ``strict=True``, levanta ``SchemaError`` se houver algum erro."""
    issues: list[dict] = []

    for field in config.REQUIRED_FIELDS:
        if field not in df.columns:
            issues.append({"nivel": "erro", "mensagem": f"Coluna obrigatória ausente: '{field}'."})
    for group in config.REQUIRED_ONE_OF:
        if not any(f in df.columns for f in group):
            issues.append({
                "nivel": "erro",
                "mensagem": f"É necessário ao menos uma das colunas: {', '.join(group)}.",
            })
    for field in config.EXPECTED_FIELDS:
        if field not in df.columns:
            issues.append({
                "nivel": "aviso",
                "mensagem": f"Coluna esperada não encontrada: '{field}' "
                            "(ajuste COLUMN_ALIASES em src/config.py).",
            })
    if len(df) == 0:
        issues.append({"nivel": "erro", "mensagem": "A planilha não possui linhas."})

    errors = [i for i in issues if i["nivel"] == "erro"]
    if strict and errors:
        raise SchemaError(" | ".join(i["mensagem"] for i in errors))
    return issues


# ---------------------------------------------------------------------------
# Helpers de estatística
# ---------------------------------------------------------------------------
def _native(obj):
    """Converte tipos numpy/pandas em tipos nativos serializáveis em JSON."""
    if isinstance(obj, dict):
        return {str(k): _native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (pd.Timestamp, date)):
        return obj.isoformat()
    return obj


def _pct(part: int | float, total: int | float) -> float:
    return round(100.0 * part / total, 2) if total else 0.0


def _value_counts(series: pd.Series, top: int = TOP_N) -> dict:
    counts = series.value_counts(dropna=True).head(top)
    return {str(k): int(v) for k, v in counts.items()}


def _quantiles(series: pd.Series) -> dict:
    s = series.dropna()
    if s.empty:
        return {}
    qs = s.quantile([0.01, 0.25, 0.5, 0.75, 0.95, 0.99])
    out = {f"p{int(q * 100)}": float(v) for q, v in qs.items()}
    out.update({"min": float(s.min()), "max": float(s.max()), "media": float(s.mean())})
    return out


def _grupo_rotulo(series: pd.Series) -> pd.Series:
    norm = series.map(normalize_text)

    def classify(v):
        if v is None or pd.isna(v):
            return "sem_informacao"
        if v in _GRUPO_MEDIO:
            return "medio"
        if v in _GRUPO_GRANDE:
            return "grande"
        return "outros_portes"

    return norm.map(classify)


# ---------------------------------------------------------------------------
# Seções do relatório
# ---------------------------------------------------------------------------
def _section_chaves(df: pd.DataFrame) -> dict:
    out: dict = {}
    if "cnpj" in df.columns:
        out["cnpj_nulos"] = int(df["cnpj"].isna().sum())
        out["cnpj_com_tamanho_invalido"] = int(
            (df["cnpj"].dropna().str.len() != config.ID_WIDTHS["cnpj"]).sum()
        )
        out["linhas_com_cnpj_duplicado"] = int(df["cnpj"].dropna().duplicated(keep=False).sum())
        out["cnpj_distintos"] = int(df["cnpj"].nunique())
    out["linhas_totalmente_duplicadas"] = int(df.duplicated().sum())

    if "cnpj_raiz" in df.columns:
        out["cnpj_raiz_nulos"] = int(df["cnpj_raiz"].isna().sum())
        out["cnpj_raiz_distintos"] = int(df["cnpj_raiz"].nunique())
        per_root = df["cnpj_raiz"].value_counts()
        out["raizes_com_mais_de_um_estabelecimento"] = int((per_root > 1).sum())
        out["max_estabelecimentos_por_raiz"] = int(per_root.max()) if len(per_root) else 0
        if "cnpj" in df.columns:
            both = df.dropna(subset=["cnpj", "cnpj_raiz"])
            out["raiz_diferente_do_prefixo_do_cnpj"] = int((both["cnpj"].str[:8] != both["cnpj_raiz"]).sum())
    return out


def _section_capital(df: pd.DataFrame) -> dict:
    if "capital_social" not in df.columns:
        return {"disponivel": False}
    cap = df["capital_social"]
    total = len(cap)
    n_null, n_zero = int(cap.isna().sum()), int((cap == 0).sum())
    n_pos, n_neg = int((cap > 0).sum()), int((cap < 0).sum())
    out = {
        "disponivel": True,
        "nulos": n_null, "pct_nulos": _pct(n_null, total),
        "zeros": n_zero, "pct_zeros": _pct(n_zero, total),
        "positivos": n_pos, "pct_positivos": _pct(n_pos, total),
        "negativos": n_neg,
        "quantis_positivos": _quantiles(cap[cap > 0]),
        "maiores_valores": [float(v) for v in cap.dropna().nlargest(5)],
        "regra_zero_adotada": config.CAPITAL_SOCIAL_ZERO_RULE,
    }

    evid: dict = {}
    if "porte_obs" in df.columns:
        grupo = _grupo_rotulo(df["porte_obs"])
        tab = {}
        for g, sub in cap.groupby(grupo):
            tab[str(g)] = {
                "registros": int(len(sub)),
                "pct_zero": _pct(int((sub == 0).sum()), len(sub)),
                "pct_nulo": _pct(int(sub.isna().sum()), len(sub)),
                "mediana_positivos": float(sub[sub > 0].median()) if (sub > 0).any() else None,
            }
        evid["por_grupo_de_rotulo"] = tab
    if "natureza_juridica" in df.columns:
        by_nat = (
            df.assign(_zero=(cap == 0))
            .groupby("natureza_juridica")["_zero"]
            .agg(["size", "mean"])
            .query("size >= 20")
            .sort_values("mean", ascending=False)
            .head(TOP_N)
        )
        evid["naturezas_com_maior_pct_zero(n>=20)"] = {
            str(k): {"registros": int(r["size"]), "pct_zero": round(100 * float(r["mean"]), 2)}
            for k, r in by_nat.iterrows()
        }
    out["evidencias_semantica_zero"] = evid
    return out


def _section_funcionarios(df: pd.DataFrame) -> dict:
    if "funcionarios" not in df.columns:
        return {"disponivel": False}
    f = df["funcionarios"]
    total = len(f)
    n_val = int(f.notna().sum())
    out = {
        "disponivel": True,
        "preenchidos": n_val, "cobertura_pct": _pct(n_val, total),
        "nulos": total - n_val,
        "zeros": int((f == 0).sum()),
        "quantis": _quantiles(f),
    }
    if "porte_obs" in df.columns:
        grupo = _grupo_rotulo(df["porte_obs"])
        out["cobertura_por_grupo_de_rotulo"] = {
            str(g): {"registros": int(len(sub)), "cobertura_pct": _pct(int(sub.notna().sum()), len(sub))}
            for g, sub in f.groupby(grupo)
        }
    return out


def _section_datas(df: pd.DataFrame, info: dict | None) -> dict:
    if "data_inicio" not in df.columns:
        return {"disponivel": False}
    d = df["data_inicio"]
    valid = d.dropna()
    today = pd.Timestamp.today().normalize()
    return {
        "disponivel": True,
        "nulas": int(d.isna().sum()),
        "invalidas_na_conversao": int((info or {}).get("problemas_conversao", {}).get("data_inicio_invalida", 0)),
        "futuras": int((valid > today).sum()),
        "anteriores_a_1900": int((valid < pd.Timestamp("1900-01-01")).sum()),
        "menor": valid.min().date().isoformat() if len(valid) else None,
        "maior": valid.max().date().isoformat() if len(valid) else None,
    }


def _section_categorias(df: pd.DataFrame) -> dict:
    out = {}
    for field in ["natureza_juridica", "cnae_principal", "matriz_filial", "situacao_cadastral", "porte_oficial"]:
        if field in df.columns:
            out[field] = {
                "valores_distintos": int(df[field].nunique()),
                "nulos": int(df[field].isna().sum()),
                f"top_{TOP_N}": _value_counts(df[field]),
            }
    return out


def _section_rotulos(df: pd.DataFrame) -> dict:
    s = df["porte_obs"]
    norm = s.map(normalize_text)
    grupo = _grupo_rotulo(s)
    out = {
        "preenchidos": int(s.notna().sum()),
        "valores_brutos_normalizados": _value_counts(norm, top=20),
        "registros_por_grupo": {k: int(v) for k, v in grupo.value_counts().items()},
        "nota": "Contagem por estabelecimento. O mapeamento oficial e a contagem por CNPJ raiz "
                "(com tratamento de conflitos) serão feitos na Etapa 3.",
    }
    if "cnpj_raiz" in df.columns:
        out["raizes_distintas_por_grupo"] = {
            str(g): int(sub["cnpj_raiz"].nunique())
            for g, sub in df.groupby(grupo)
            if g in ("medio", "grande")
        }
    return out


# ---------------------------------------------------------------------------
# Relatório completo
# ---------------------------------------------------------------------------
def build_quality_report(df: pd.DataFrame, info: dict | None = None) -> dict:
    """Monta o relatório de qualidade (dicionário serializável)."""
    info = info or {}
    issues = validate_schema(df, info, strict=False)
    nulls = df.isna().sum()

    report = {
        "geral": {
            "arquivo": info.get("arquivo"),
            "aba": info.get("aba"),
            "linhas": int(len(df)),
            "colunas": int(df.shape[1]),
            "gerado_em": pd.Timestamp.now().isoformat(timespec="seconds"),
        },
        "colunas": {
            "mapeamento": info.get("mapeamento", {}),
            "nao_mapeadas": info.get("nao_mapeadas", []),
            "campos_logicos_ausentes": info.get("campos_logicos_ausentes", []),
        },
        "problemas_esquema": issues,
        "problemas_conversao": info.get("problemas_conversao", {}),
        "chaves": _section_chaves(df),
        "nulos_por_coluna": {
            str(c): {"nulos": int(n), "pct": _pct(int(n), len(df))}
            for c, n in nulls.sort_values(ascending=False).items() if n > 0
        },
        "capital_social": _section_capital(df),
        "funcionarios": _section_funcionarios(df),
        "datas": _section_datas(df, info),
        "categorias": _section_categorias(df),
        "rotulos_porte_obs": _section_rotulos(df) if "porte_obs" in df.columns else {"disponivel": False},
    }
    report["pendencias"] = _auto_pendencias(report)
    return _native(report)


def _auto_pendencias(report: dict) -> list[str]:
    pend: list[str] = []
    if report["colunas"]["campos_logicos_ausentes"]:
        pend.append(
            "Campos lógicos não encontrados na planilha: "
            + ", ".join(report["colunas"]["campos_logicos_ausentes"])
            + ". Conferir cabeçalhos e ajustar COLUMN_ALIASES em src/config.py."
        )
    if config.CAPITAL_SOCIAL_ZERO_RULE == "pendente":
        pend.append(
            "Confirmar com a fonte (Receita/OLAP) a semântica de capital social = 0 e registrar a "
            "regra em config.CAPITAL_SOCIAL_ZERO_RULE. Até lá, zeros são preservados e NÃO tratados como ausência."
        )
    func = report["funcionarios"]
    if func.get("disponivel") and func.get("cobertura_pct", 0) < 50:
        pend.append(
            f"Cobertura baixa de funcionários ({func['cobertura_pct']}%). Só usar como feature se a "
            "cobertura nas classes Médio/Grande for suficiente."
        )
    return pend


# ---------------------------------------------------------------------------
# Escrita do relatório
# ---------------------------------------------------------------------------
def _md_table(rows: list[tuple], headers: tuple) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render_markdown(report: dict) -> str:
    g = report["geral"]
    md = [
        "# Relatório de qualidade dos dados",
        "",
        f"- Arquivo: `{g['arquivo']}` (aba: `{g['aba']}`)",
        f"- Linhas: **{g['linhas']:,}** | Colunas: **{g['colunas']}** | Gerado em: {g['gerado_em']}".replace(",", "."),
        "",
        "## Pendências identificadas automaticamente",
        "",
    ]
    md += [f"- {p}" for p in report["pendencias"]] or ["- Nenhuma."]

    md += ["", "## Problemas de esquema", ""]
    md += [f"- **{i['nivel']}**: {i['mensagem']}" for i in report["problemas_esquema"]] or ["- Nenhum."]

    md += ["", "## Colunas", "",
           f"- Mapeadas para nomes lógicos: {len(report['colunas']['mapeamento'])}",
           f"- Não mapeadas (mantidas com nome original): {len(report['colunas']['nao_mapeadas'])}"]
    if report["colunas"]["nao_mapeadas"]:
        md.append("  - " + ", ".join(f"`{c}`" for c in report["colunas"]["nao_mapeadas"]))

    md += ["", "## Chaves e duplicidades", ""]
    md += [f"- {k.replace('_', ' ')}: **{v}**" for k, v in report["chaves"].items()]

    if report["problemas_conversao"]:
        md += ["", "## Valores não convertidos", ""]
        md += [f"- {k.replace('_', ' ')}: **{v}**" for k, v in report["problemas_conversao"].items()]

    md += ["", "## Capital social", ""]
    cap = report["capital_social"]
    if cap.get("disponivel"):
        md.append(_md_table(
            [("Nulos", cap["nulos"], f"{cap['pct_nulos']}%"),
             ("Zeros", cap["zeros"], f"{cap['pct_zeros']}%"),
             ("Positivos", cap["positivos"], f"{cap['pct_positivos']}%"),
             ("Negativos", cap["negativos"], "")],
            ("Categoria", "Registros", "%"),
        ))
        md.append(f"\nRegra adotada para zero: **{cap['regra_zero_adotada']}**")
        grupo = cap["evidencias_semantica_zero"].get("por_grupo_de_rotulo")
        if grupo:
            md += ["", "Evidência (por grupo de rótulo `Obs Porte Sa3`):", ""]
            md.append(_md_table(
                [(k, v["registros"], f"{v['pct_zero']}%", f"{v['pct_nulo']}%", v["mediana_positivos"])
                 for k, v in grupo.items()],
                ("Grupo", "Registros", "% zero", "% nulo", "Mediana (>0)"),
            ))
    else:
        md.append("Coluna não disponível.")

    md += ["", "## Quantidade de funcionários", ""]
    fun = report["funcionarios"]
    if fun.get("disponivel"):
        md.append(f"- Cobertura: **{fun['cobertura_pct']}%** ({fun['preenchidos']} preenchidos, {fun['zeros']} zeros)")
        for g_, v in fun.get("cobertura_por_grupo_de_rotulo", {}).items():
            md.append(f"  - {g_}: {v['cobertura_pct']}% de {v['registros']} registros")
    else:
        md.append("Coluna não disponível.")

    md += ["", "## Datas de início", ""]
    dt = report["datas"]
    md += [f"- {k.replace('_', ' ')}: **{v}**" for k, v in dt.items() if k != "disponivel"] if dt.get("disponivel") else ["Coluna não disponível."]

    md += ["", "## Rótulos (`Obs Porte Sa3`) — por estabelecimento", ""]
    rot = report["rotulos_porte_obs"]
    if "registros_por_grupo" in rot:
        md.append(_md_table(list(rot["registros_por_grupo"].items()), ("Grupo", "Registros")))
        if "raizes_distintas_por_grupo" in rot:
            md += ["", "CNPJs raiz distintos: " + ", ".join(f"{k}={v}" for k, v in rot["raizes_distintas_por_grupo"].items())]
        md += ["", f"_{rot['nota']}_"]

    md += ["", "## Categorias (top 10)", ""]
    for field, v in report["categorias"].items():
        md += [f"**{field}** — {v['valores_distintos']} valores distintos, {v['nulos']} nulos", ""]
        md.append(_md_table(list(v[f"top_{TOP_N}"].items()), ("Valor", "Registros")))
        md.append("")

    md += ["## Nulos por coluna", ""]
    md.append(_md_table([(c, v["nulos"], f"{v['pct']}%") for c, v in report["nulos_por_coluna"].items()],
                        ("Coluna", "Nulos", "%")) if report["nulos_por_coluna"] else "Nenhum nulo.")
    return "\n".join(md) + "\n"


def write_report(report: dict, json_path: Path | None = None, md_path: Path | None = None) -> tuple[Path, Path]:
    json_path = Path(json_path or config.REPORT_QUALIDADE_JSON)
    md_path = Path(md_path or config.REPORT_QUALIDADE_MD)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path
