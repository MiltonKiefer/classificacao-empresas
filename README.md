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
| 2 | Ingestão e análise exploratória | ✅ Concluída na planilha real |
| 3 | Consolidação por CNPJ raiz e rótulos | ✅ Concluída (45 Médio e 18 Grande, sem conflito) |
| 4 | Engenharia de atributos | ✅ Concluída (16 colunas; funcionários ficou de fora) |
| 5 | Treinamento e avaliação | ✅ Concluída (métricas exploratórias; F1-macro 0,524) |
| 6 | Inferência das empresas sem rótulo | ✅ Concluída (execução real validada) |
| 7 | Exportação, testes e documentação final | ✅ Concluída (execução real validada; 4 testes automatizados passaram) |

### ⚠️ Pendências abertas (ler antes de continuar)

1. **Ambiente virtual:** cada integrante precisa criar o seu `.venv` (ver "Como começar").
2. **Divisão de tarefas da equipe:** não definida (ver seção Equipe).

Aviso para a Etapa 4: funcionários têm cobertura de 0,08% no arquivo inteiro e 100% nas classes Médio e Grande já rotuladas. A decisão foi não usar essa coluna: ela existe nas empresas já anotadas e quase não existe nas que serão classificadas.

### ▶️ Status e observações
As Etapas 6 e 7 foram executadas na base real, e os quatro testes automatizados da exportação passaram.
As estimativas permanecem sinalizadas para revisão manual porque a amostra é pequena e as probabilidades
não foram calibradas; nenhum limiar de confiança foi inventado.

---

## Como começar

Requer Python 3.10+ (validado com 3.12).

No Windows, instale o [Python Install Manager](https://www.python.org/downloads/latest/pymanager/) (ou use a
[Microsoft Store](https://apps.microsoft.com/detail/9NQ7512CXL7T)). Python 3.12.16 é distribuído como código-fonte;
Python 3.12.10 foi a última versão 3.12 com instalador tradicional para Windows. No terminal do VS Code, confira
as versões 3.12 que o gerenciador oferece e instale 3.12.10:

```powershell
py list --online 3.12
py install 3.12.10
py -V:3.12.10 --version
```

Crie e use o ambiente do projeto sem depender da política de ativação do PowerShell:

```powershell
py -V:3.12.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest
```

No VS Code, selecione `.venv\Scripts\python.exe` com **Python: Select Interpreter**. Se `py install 3.12.10`
não estiver disponível, instale a versão 3.12 indicada por `py list --online 3.12` e use o mesmo seletor de versão
em `py -V:<versão>`.

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

# 5. Consolidar por CNPJ raiz (usa o Parquet da Etapa 2)
python -m src.pipeline --etapa 3

# 6. Montar a matriz de atributos (usa a base da Etapa 3)
python -m src.pipeline --etapa 4

# 7. Treinar e avaliar (usa a matriz da Etapa 4)
python -m src.pipeline --etapa 5

# 8. Inferir empresas sem rótulo Médio/Grande (usa o modelo da Etapa 5)
python -m src.pipeline --etapa 6

# 9. Exportar resultados e gerar relatório (usa as Etapas 3 e 6)
python -m src.pipeline --etapa 7
```

A leitura do Excel completo pode levar alguns minutos. Saídas da Etapa 2:

- `data/interim/estabelecimentos.parquet` — dados padronizados, uma linha por estabelecimento.
- `outputs/reports/relatorio_qualidade.md` e `.json` — qualidade dos dados, cobertura de capital social e
  funcionários, evidências sobre capital = 0, distribuição de rótulos e pendências automáticas.

Saídas da Etapa 3:

- `data/processed/empresas_raiz.parquet` — uma linha por CNPJ raiz, com o conjunto (`treino`, `inferencia`,
  `revisao` ou `fora_do_escopo`).
- `outputs/reports/relatorio_rotulos.md` e `.json` — contagem por classe e regras de agregação.

Saídas da Etapa 4:

- `data/processed/features.parquet` — atributos de treino e inferência, antes da imputação e da codificação.
- `outputs/reports/dicionario_features.md` e `.json` — lista do que entra no modelo e do que ficou de fora.

Saídas da Etapa 5:

- `models/porte_xgb.joblib` — XGBoost reajustado nas 63 empresas rotuladas (não versionado).
- `outputs/reports/relatorio_avaliacao.md` e `.json` — comparação com a classe majoritária e com uma árvore, métricas e limitação da amostra.

Saída da Etapa 6:

- `outputs/predictions/predicoes.parquet` — estimativas por CNPJ raiz, com probabilidades, porte original,
  sinalização de revisão manual, versão do modelo, fingerprint dos dados e data da execução.

Saídas da Etapa 7:

- `outputs/predictions/resultado_classificacao.csv` e `.parquet` — rótulos conhecidos e estimativas em
  registros distintos, com o porte original preservado.
- `outputs/reports/relatorio_resultados.md` e `.json` — distribuição por classe, registros para revisão
  e metadados da execução.

O notebook `notebooks/analise_exploratoria.ipynb` consome essas saídas para visualização (abrir no VS Code
depois de rodar o pipeline).

## Estrutura

```text
src/
  config.py                 # caminhos, aliases de colunas, classes, semente   [pronto]
  data/load_data.py         # leitura Excel/CSV, mapeamento e tipos            [pronto]
  data/validate_data.py     # validação de esquema e relatório de qualidade    [pronto]
  pipeline.py               # coordenação das Etapas 2 a 7                  [pronto]
  entities/aggregate_root.py        # Etapa 3 — uma linha por CNPJ raiz          [pronto]
  labeling/prepare_labels.py        # Etapa 3 — rótulo Médio/Grande              [pronto]
  preprocessing/build_features.py   # Etapa 4 — matriz de atributos              [pronto]
  model/train.py                    # Etapa 5 — XGBoost e baselines              [pronto]
  model/evaluate.py                 # Etapa 5 — métricas da validação            [pronto]
  model/predict.py                  # Etapa 6 — inferência                    [implementada]
  reporting/export_results.py       # Etapa 7 — exportação e relatório       [implementada]
tests/                      # pytest; dados sintéticos em tests/helpers.py
docs/                       # guia de desenvolvimento (checklist por etapa)
data/{raw,interim,processed}/  # dados (NÃO versionados)
models/  outputs/              # modelos e resultados gerados (NÃO versionados, exceto relatórios)
```

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
