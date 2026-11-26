# Dicionário de atributos

- Data de referência da idade da empresa: **2026-10-06**
- Regra do capital zero: **zero_nao_define_porte**
- Empresas na matriz: **163584** (treino 63, inferência 163521)
- Colunas de X, antes da codificação: **16**

Capital igual a zero vira `log1p` 0 e o indicador `capital_social_zero` = 1. Esse indicador entra junto com as outras colunas e não define o porte sozinho. Ausência de capital continua nula até a imputação, que só vê o treino.

## Colunas numéricas

- `capital_social_log`
- `capital_social_zero`
- `qtd_estabelecimentos`
- `qtd_cnaes_distintos`
- `qtd_divisoes_cnae`
- `qtd_situacao_ativa`
- `qtd_situacao_baixada`
- `qtd_situacao_inapta`
- `qtd_situacao_suspensa`
- `qtd_situacao_nula`
- `qtd_situacao_outra`
- `tempo_atividade_anos`
- `tem_matriz`

## Colunas categóricas

- `natureza_juridica`
- `divisao_cnae`
- `situacao_matriz`

Depois do ajuste só nas 63 linhas de treino, a matriz codificada tem **29** colunas.

## Fora do modelo

| Coluna | Motivo |
|---|---|
| `cnpj_raiz` | Chave da empresa. Fica na tabela só para rastrear a linha, não entra no modelo. |
| `razao_social` | Identificador textual. |
| `rotulo` | Resposta do treino. Vira o vetor y, nunca uma coluna de X. |
| `rotulo_nome` | O mesmo rótulo, em texto. |
| `conjunto` | Define quem é treino e quem é inferência. Não descreve a empresa. |
| `motivo_conjunto` | Explica a separação dos rótulos. |
| `rotulos_observados` | Texto original de Obs Porte Sa3. |
| `porte_oficial` | Porte oficial. Na base atual é constante (DEMAIS) e não pode revelar a resposta. |
| `funcionarios` | Preenchido nas 63 empresas rotuladas e em quase nenhuma das demais. Usar esse campo treinaria o modelo com um dado que a inferência não tem. |
| `cnae_principal` | Mais de mil códigos para 63 exemplos. No lugar entra a divisão (dois primeiros dígitos). |
| `capital_social` | Substituído por capital_social_log e capital_social_zero, para a escala não dominar o zero. |
| `capital_divergente` | Na planilha real nenhuma raiz diverge. Coluna de auditoria. |
| `natureza_divergente` | Na planilha real nenhuma raiz diverge. Coluna de auditoria. |
| `data_inicio_empresa` | Vira tempo_atividade_anos até a data de referência. |
| `data_inicio_matriz` | Auditoria. A idade usa a menor data da raiz. |
| `filial_mais_antiga_que_matriz` | Só 30 raízes. Quase constante. |
| `mais_de_uma_matriz` | Nenhuma raiz tem mais de uma matriz. |
| `cnae_origem` | Diz de onde veio o CNAE. tem_matriz já cobre a ausência de matriz. |
| `cnae_divergente` | Resumo de qtd_cnaes_distintos, que já entra como número. |
| `porte_oficial_divergente` | Auditoria do porte oficial. |

SimpleImputer e OneHotEncoder dentro de ColumnTransformer. O fit ocorre só nas linhas de treino de cada partição da Etapa 5. Esta etapa não grava um transformador já ajustado.
