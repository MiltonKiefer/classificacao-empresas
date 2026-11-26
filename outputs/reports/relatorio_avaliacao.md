# Avaliação do modelo

Amostra de treino pequena, em especial a classe Grande. As métricas são exploratórias e não sustentam uma conclusão forte de generalização.

- Validação: **StratifiedKFold com 3 folds, embaralhado, semente 42**
- Empresas de treino: **63** (Médio 45, Grande 18)
- Calibrador aplicado: **não**. Não há amostra suficiente para avaliar ou ajustar um calibrador de probabilidade. O Brier é só um registro; a probabilidade não deve ser chamada de confiança.
- Modelo gravado para a Etapa 6: **xgb_ponderado**

As métricas abaixo usam as previsões fora de cada fold (cada empresa entra uma vez no teste). Capital zero não é uma regra de porte: ele é só uma das colunas.

## Comparação

| Modelo | F1-macro | Acurácia balanceada | Recall Médio | Recall Grande | Brier |
|---|---|---|---|---|---|
| maioria | 0.417 | 0.500 | 1.000 | 0.000 | 0.286 |
| arvore | 0.449 | 0.494 | 0.933 | 0.056 | 0.266 |
| xgb | 0.508 | 0.511 | 0.800 | 0.222 | 0.247 |
| xgb_ponderado | 0.524 | 0.528 | 0.667 | 0.389 | 0.261 |

## Matriz de confusão — xgb_ponderado

|  | Previsto Médio | Previsto Grande |
|---|---|---|
| Verdadeiro Médio | 30 | 15 |
| Verdadeiro Grande | 11 | 7 |

Recall de Grande por fold: 0.33, 0.50, 0.33.
Recall de Médio no agregado: 0.667.

## Hiperparâmetros

XGBoost com objetivo binary:logistic, 80 árvores, learning_rate 0,1 e min_child_weight 2. A profundidade (2 ou 3) é escolhida por uma validação estratificada interna, só com as linhas de treino de cada fold. O peso da classe Grande, quando usado, também é contado só nesse treino. Não houve oversampling.
