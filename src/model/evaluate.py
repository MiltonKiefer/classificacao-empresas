from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from src import config


def metricas(y_true, y_pred, y_prob) -> dict:
    """F1-macro, acurácia balanceada, precisão/recall por classe, matriz e Brier."""
    verdade = np.asarray(y_true).astype(int)
    previsto = np.asarray(y_pred).astype(int)
    probabilidade = np.asarray(y_prob, dtype=float)
    precisao, recall, f1, suporte = precision_recall_fscore_support(
        verdade, previsto, labels=[config.CLASS_MEDIO, config.CLASS_GRANDE], zero_division=0,
    )
    nomes = [config.CLASS_NAMES[config.CLASS_MEDIO], config.CLASS_NAMES[config.CLASS_GRANDE]]
    matriz = confusion_matrix(verdade, previsto, labels=[config.CLASS_MEDIO, config.CLASS_GRANDE])
    return {
        "f1_macro": float(f1_score(verdade, previsto, average="macro", zero_division=0)),
        "acuracia_balanceada": float(balanced_accuracy_score(verdade, previsto)),
        "precisao": {nome: float(valor) for nome, valor in zip(nomes, precisao)},
        "recall": {nome: float(valor) for nome, valor in zip(nomes, recall)},
        "f1": {nome: float(valor) for nome, valor in zip(nomes, f1)},
        "suporte": {nome: int(valor) for nome, valor in zip(nomes, suporte)},
        "matriz_confusao": matriz.astype(int).tolist(),
        "brier": float(brier_score_loss(verdade, probabilidade)),
    }


def recall_por_fold(y_true, y_pred, folds: np.ndarray) -> list[float]:
    """Recall da classe Grande em cada fold, para mostrar a instabilidade."""
    verdade = np.asarray(y_true).astype(int)
    previsto = np.asarray(y_pred).astype(int)
    saida = []
    for fold in sorted(set(folds)):
        mascara = folds == fold
        reais = verdade[mascara] == config.CLASS_GRANDE
        if not reais.any():
            saida.append(0.0)
            continue
        saida.append(float((previsto[mascara][reais] == config.CLASS_GRANDE).mean()))
    return saida


def render_avaliacao(relatorio: dict) -> str:
    linhas = [
        "# Avaliação do modelo",
        "",
        relatorio["limitacao"],
        "",
        f"- Validação: **{relatorio['validacao']}**",
        f"- Empresas de treino: **{relatorio['n_treino']}** "
        f"(Médio {relatorio['contagem']['Médio']}, Grande {relatorio['contagem']['Grande']})",
        f"- Calibrador aplicado: **não**. {relatorio['calibracao']}",
        f"- Modelo gravado para a Etapa 6: **{relatorio['modelo_salvo']}**",
        "",
        "As métricas abaixo usam as previsões fora de cada fold (cada empresa entra uma vez no teste). "
        "Capital zero não é uma regra de porte: ele é só uma das colunas.",
        "",
        "## Comparação",
        "",
        "| Modelo | F1-macro | Acurácia balanceada | Recall Médio | Recall Grande | Brier |",
        "|---|---|---|---|---|---|",
    ]
    for nome, bloco in relatorio["modelos"].items():
        linhas.append(
            f"| {nome} | {bloco['f1_macro']:.3f} | {bloco['acuracia_balanceada']:.3f} | "
            f"{bloco['recall']['Médio']:.3f} | {bloco['recall']['Grande']:.3f} | {bloco['brier']:.3f} |"
        )
    salvo = relatorio["modelos"][relatorio["modelo_salvo"]]
    matriz = salvo["matriz_confusao"]
    linhas += [
        "",
        f"## Matriz de confusão — {relatorio['modelo_salvo']}",
        "",
        "|  | Previsto Médio | Previsto Grande |",
        "|---|---|---|",
        f"| Verdadeiro Médio | {matriz[0][0]} | {matriz[0][1]} |",
        f"| Verdadeiro Grande | {matriz[1][0]} | {matriz[1][1]} |",
        "",
        f"Recall de Grande por fold: {', '.join(f'{v:.2f}' for v in salvo['recall_grande_por_fold'])}.",
        f"Recall de Médio no agregado: {salvo['recall']['Médio']:.3f}.",
        "",
        "## Hiperparâmetros",
        "",
        relatorio["hiperparametros"],
        "",
    ]
    return "\n".join(linhas)
