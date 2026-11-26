# Relatório de qualidade dos dados

- Arquivo: `query_result_2026-10-06T08_52_19.110898753-03_00(1).xlsx` (aba: `Resultado da consulta`)
- Linhas: **180.628** | Colunas: **46** | Gerado em: 2026-11-26T14:50:08

## Pendências identificadas automaticamente

- Cobertura baixa de funcionários (0.08%). Só usar como feature se a cobertura nas classes Médio/Grande for suficiente.

## Problemas de esquema

- Nenhum.

## Colunas

- Mapeadas para nomes lógicos: 15
- Não mapeadas (mantidas com nome original): 31
  - `Tipo Conta`, `Nome Fantasia`, `Empresa Sediada`, `Ente Federativo`, `Origem`, `Codigo Porte`, `Codigo Situacao Cadastral`, `Data Situacao Cadastral`, `Descricao Cnae Fiscal Principal`, `Codigo Cnae Fiscal Secundario`, `Descricao Cnae Fiscal Secundario`, `Descricao Natureza Juridica`, `ID Municipio`, `Nome Municipio`, `Tipo Logradouro`, `Logradouro`, `Bairro`, `Complemento`, `Numero Endereco`, `Uf`, `Pais Codigo`, `Pais Nome`, `Telefone2`, `Fax`, `ID Sindicato`, `Nome Sindicato`, `Inscricao Estadual`, `Situacao Especial`, `Data Situacao Especial`, `Search Term`, `Sigla Sindicato`

## Chaves e duplicidades

- cnpj nulos: **0**
- cnpj com tamanho invalido: **0**
- linhas com cnpj duplicado: **0**
- cnpj distintos: **180628**
- linhas totalmente duplicadas: **0**
- cnpj raiz nulos: **0**
- cnpj raiz distintos: **163658**
- raizes com mais de um estabelecimento: **6380**
- max estabelecimentos por raiz: **179**
- raiz diferente do prefixo do cnpj: **0**

## Valores não convertidos

- capital social nao numerico: **0**
- funcionarios nao numerico: **0**
- data inicio invalida: **0**

## Capital social

| Categoria | Registros | % |
|---|---|---|
| Nulos | 0 | 0.0% |
| Zeros | 146878 | 81.32% |
| Positivos | 33750 | 18.68% |
| Negativos | 0 |  |

Regra adotada para zero: **zero_nao_define_porte**

Evidência (por grupo de rótulo `Obs Porte Sa3`):

| Grupo | Registros | % zero | % nulo | Mediana (>0) |
|---|---|---|---|---|
| grande | 18 | 5.56% | 0.0% | 2025000000.0 |
| medio | 45 | 8.89% | 0.0% | 1038500000.0 |
| outros_portes | 176 | 7.39% | 0.0% | 100000000.0 |
| sem_informacao | 180389 | 81.41% | 0.0% | 20000000.0 |

## Quantidade de funcionários

- Cobertura: **0.08%** (139 preenchidos, 2 zeros)
  - grande: 100.0% de 18 registros
  - medio: 100.0% de 45 registros
  - outros_portes: 43.18% de 176 registros
  - sem_informacao: 0.0% de 180389 registros

## Datas de início

- nulas: **0**
- invalidas na conversao: **0**
- futuras: **0**
- anteriores a 1900: **0**
- menor: **1901-01-01**
- maior: **2026-09-11**

## Rótulos (`Obs Porte Sa3`) — por estabelecimento

| Grupo | Registros |
|---|---|
| sem_informacao | 180389 |
| outros_portes | 176 |
| medio | 45 |
| grande | 18 |

CNPJs raiz distintos: grande=18, medio=45

_Contagem por estabelecimento. O mapeamento oficial e a contagem por CNPJ raiz (com tratamento de conflitos) serão feitos na Etapa 3._

## Categorias (top 10)

**natureza_juridica** — 67 valores distintos, 0 nulos

| Valor | Registros |
|---|---|
| 2062 | 80957 |
| 2135 | 45737 |
| 3999 | 13084 |
| 4090 | 12273 |
| 3085 | 6591 |
| 2054 | 5333 |
| 2321 | 2992 |
| 3220 | 2210 |
| 4014 | 1910 |
| 2232 | 1733 |

**cnae_principal** — 1076 valores distintos, 0 nulos

| Valor | Registros |
|---|---|
| 8888888 | 22797 |
| 9492800 | 12783 |
| 4781400 | 8092 |
| 8112500 | 6572 |
| 5611203 | 6372 |
| 9430800 | 5383 |
| 4774100 | 4705 |
| 6911701 | 4594 |
| 5611201 | 4474 |
| 4712100 | 4086 |

**matriz_filial** — 2 valores distintos, 0 nulos

| Valor | Registros |
|---|---|
| MATRIZ | 155431 |
| FILIAL | 25197 |

**situacao_cadastral** — 5 valores distintos, 0 nulos

| Valor | Registros |
|---|---|
| BAIXADA | 133384 |
| ATIVA | 33484 |
| INAPTA | 11558 |
| SUSPENSA | 1512 |
| NULA | 690 |

**porte_oficial** — 1 valores distintos, 0 nulos

| Valor | Registros |
|---|---|
| DEMAIS | 180628 |

## Nulos por coluna

| Coluna | Nulos | % |
|---|---|---|
| Search Term | 180628 | 100.0% |
| Descricao Cnae Fiscal Secundario | 180628 | 100.0% |
| Inscricao Estadual | 180628 | 100.0% |
| funcionarios | 180489 | 99.92% |
| Data Situacao Especial | 180465 | 99.91% |
| Situacao Especial | 180465 | 99.91% |
| Nome Sindicato | 180389 | 99.87% |
| ID Sindicato | 180389 | 99.87% |
| Sigla Sindicato | 180389 | 99.87% |
| porte_obs | 180389 | 99.87% |
| Ente Federativo | 180047 | 99.68% |
| Telefone2 | 165261 | 91.49% |
| Fax | 151155 | 83.68% |
| Codigo Cnae Fiscal Secundario | 144698 | 80.11% |
| email | 126929 | 70.27% |
| telefone | 111342 | 61.64% |
| Complemento | 53592 | 29.67% |
| Tipo Logradouro | 5529 | 3.06% |
| Data Situacao Cadastral | 349 | 0.19% |
