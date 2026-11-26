# Classificação de porte empresarial (Médio × Grande)

Projeto de Mineração de Dados — SENAI CIMATEC Universidade (Salvador, BA, 2026).
Orientador: Márcio Renê Brandão Soussa.

**O que o projeto faz:** estima se uma empresa de Salvador classificada como `DEMAIS` na base OLAP do Setor de
Inteligência de Mercado é de porte **Médio** ou **Grande**, usando XGBoost treinado com as empresas que já têm
essa informação na coluna `Obs Porte Sa3`. A unidade de análise é o **CNPJ raiz** (uma linha por empresa).
As previsões são sempre *estimativas*, nunca substituem o porte oficial.

> Documento de referência completo: [`docs/GUIA_DESENVOLVIMENTO_CLASSIFICACAO_PORTE.md`](docs/GUIA_DESENVOLVIMENTO_CLASSIFICACAO_PORTE.md)
> (arquitetura, regras de dados e checklist detalhado de cada etapa).

---

## 📍 Onde paramos (atualizado em 26/11/2026)

| Etapa | Descrição | Status |
|---|---|---|
| 0 | Definição do escopo | ✅ Concluída (falta só registrar divisão de tarefas da equipe) |
| 1 | Ambiente e estrutura | ✅ Concluída |
| 2 | Ingestão e análise exploratória | ✅ Concluída na planilha real (26/11/2026) |
| 3 | Consolidação por CNPJ raiz e rótulos | ⏳ Não iniciada |
| 4 | Engenharia de atributos | ⏳ Não iniciada |
| 5 | Treinamento e avaliação | ⏳ Não iniciada |
| 6 | Inferência das empresas sem rótulo | ⏳ Não iniciada |
| 7 | Exportação, testes e documentação final | ⏳ Não iniciada |

### ⚠️ Pendências abertas (ler antes de continuar)

1. **Ambiente virtual:** cada integrante precisa criar o seu `.venv` (ver "Como começar").
2. **Divisão de tarefas da equipe:** não definida (ver seção Equipe).

Aviso para a Etapa 4: funcionários têm cobertura de 0,08% no arquivo inteiro e 100% nas classes Médio e Grande já rotuladas. Só entra como feature se essa cobertura nas classes de treino for considerada suficiente.

### ▶️ Próximo passo
**Etapa 3** — implementar `src/entities/aggregate_root.py` e `src/labeling/prepare_labels.py`:
consolidar por CNPJ raiz, mapear `Obs Porte Sa3` → Médio = 0 / Grande = 1, separar raízes com rótulos
conflitantes (excluí-las do treino, sem decidir por maioria) e contar quantas empresas rotuladas sobram por classe.
Essa contagem define a estratégia de validação da Etapa 5 (a amostra é pequena: ~45 Média e ~18 Grande antes
da consolidação).

---

## Como começar

Requer Python 3.10+ (validado com 3.12).

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
.venv\Scripts\activate             # Windows (PowerShell/CMD)

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Rodar os testes
python -m pytest

# 4. Executar a Etapa 2 (com a planilha em data/raw/)
python -m src.pipeline --etapa 2
# ou apontando o arquivo:
python -m src.pipeline --etapa 2 --arquivo "C:/caminho/planilha.xlsx"
```

A leitura do Excel completo pode levar alguns minutos. Saídas da Etapa 2:

- `data/interim/estabelecimentos.parquet` — dados padronizados, uma linha por estabelecimento.
- `outputs/reports/relatorio_qualidade.md` e `.json` — qualidade dos dados, cobertura de capital social e
  funcionários, evidências sobre capital = 0, distribuição de rótulos e pendências automáticas.

O notebook `notebooks/analise_exploratoria.ipynb` consome essas saídas para visualização (abrir no VS Code
depois de rodar o pipeline).

## Estrutura

```text
src/
  config.py                 # caminhos, aliases de colunas, classes, semente   [pronto]
  data/load_data.py         # leitura Excel/CSV, mapeamento e tipos            [pronto]
  data/validate_data.py     # validação de esquema e relatório de qualidade    [pronto]
  pipeline.py               # coordenação (hoje só a Etapa 2)                  [parcial]
  entities/aggregate_root.py        # Etapa 3 — esqueleto
  labeling/prepare_labels.py        # Etapa 3 — esqueleto
  preprocessing/build_features.py   # Etapa 4 — esqueleto
  model/{train,evaluate,predict}.py # Etapas 5 e 6 — esqueletos
  reporting/export_results.py       # Etapa 7 — esqueleto
tests/                      # pytest; dados sintéticos em tests/helpers.py
docs/                       # guia de desenvolvimento (checklist por etapa)
data/{raw,interim,processed}/  # dados (NÃO versionados)
models/  outputs/              # modelos e resultados gerados (NÃO versionados, exceto relatórios)
```

Arquivos marcados como "esqueleto" contêm só uma docstring; a implementação está nas etapas seguintes.

## Decisões já tomadas (resumo)

- Classificação binária **Médio (0) × Grande (1)**; Micro e Pequena ficam fora do modelo.
- Um valor 0 em uma métrica (capital social é o caso medido) não é evidência de pequeno porte. A estimativa considera o conjunto das métricas. Zeros são preservados. Regra: `CAPITAL_SOCIAL_ZERO_RULE = "zero_nao_define_porte"`.
- Unidade: **CNPJ raiz**; o CNPJ raiz é chave, nunca feature.
- Rótulo vem de `Obs Porte Sa3`; essa coluna e o porte oficial **nunca** entram como features.
- Rótulos conflitantes na mesma raiz: registrar e excluir do treino (sem escolher por maioria).
- Algoritmo: `XGBClassifier` (`binary:logistic`); transformações dentro de pipeline scikit-learn para evitar vazamento.
- Códigos (CNPJ, CNAE, CEP, natureza jurídica) são lidos como texto para não perder zeros à esquerda.
- Previsões ficam em colunas separadas (`porte_estimado`, `prob_*`, `status_classificacao`), nunca sobrescrevem o porte original.

## Limitações a ter em mente

Amostra rotulada muito pequena (especialmente `Grande`). Métricas futuras devem ser apresentadas como
exploratórias e probabilidades não devem ser chamadas de "confiança" sem calibração avaliada
(ver seção 5 do guia).

## Equipe

| Integrante | Responsabilidade |
|---|---|
| João Pereira Paulo | _a definir_ |
| João Vitor Cardoso dos Santos | _a definir_ |
| Milton Kiefer de Albuquerque Mello | _a definir_ |
| Pedro Henrique Araújo Fonseca | _a definir_ |
| Pedro Boaventura Ferraz Oliveira | _a definir_ |

## Convenções para continuar o trabalho

- Ao concluir um item, marque-o em `docs/GUIA_DESENVOLVIMENTO_CLASSIFICACAO_PORTE.md` **e** atualize a tabela
  "Onde paramos" acima.
- Lógica reutilizável vai em `src/`; notebooks servem só para exploração.
- Cada módulo novo deve vir com testes em `tests/`.
- Não versionar dados, modelos ou previsões (já coberto pelo `.gitignore`).
