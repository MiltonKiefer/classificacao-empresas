"""Gera uma planilha sintética no formato presumido da planilha real (apenas para testes)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_sample_dataframe(n_roots: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_roots):
        raiz = f"{i + 1:08d}"  # começa com zeros à esquerda de propósito
        n_est = int(rng.choice([1, 1, 1, 2, 3]))
        porte_obs = None
        if i % 15 == 0:
            porte_obs = "Média"
        elif i % 20 == 1:
            porte_obs = "Grande"
        elif i % 7 == 0:
            porte_obs = "Pequena"
        capital = 0 if i % 3 == 0 else float(rng.integers(1_000, 5_000_000))
        for j in range(n_est):
            rows.append({
                "CNPJ": f"{raiz}{j + 1:04d}{i % 100:02d}",
                "CNPJ Raiz": raiz,
                "Razão Social": f"EMPRESA {i}",
                "Capital Social": capital,
                "Natureza Jurídica": "2062" if i % 2 else "0000",
                "CNAE Fiscal Principal": "0111301" if i % 4 else "4711302",
                "Matriz/Filial": "1" if j == 0 else "2",
                "Situação Cadastral": "02" if i % 10 else "08",
                "Data Início Atividade": f"{1995 + i % 25}-0{1 + i % 9}-15",
                "Obs Quantidade de Funcionários": (int(rng.integers(1, 500)) if i % 4 == 0 else None),
                "Porte": "DEMAIS",
                "Obs Porte Sa3": porte_obs,
                "CEP": f"{i % 100:08d}",
                "Coluna Extra": "x",
            })
    return pd.DataFrame(rows)
