# Consolidação por CNPJ raiz

- Estabelecimentos lidos: **180628**
- Linhas sem CNPJ raiz, excluídas: **0**
- Empresas (CNPJs raiz): **163658**

## Conjuntos

| Conjunto | Empresas |
|---|---|
| treino | 63 |
| inferencia | 163521 |
| revisao | 0 |
| fora_do_escopo | 74 |

## Treino por classe

| Classe | Empresas |
|---|---|
| Médio | 45 |
| Grande | 18 |

## Conferências da agregação

- Raízes sem matriz: **8227**
- Raízes com mais de uma matriz: **0**
- Capital social divergente: **0**
- Capital social igual a zero (valor preservado): **139166**
- Natureza jurídica divergente: **0**
- CNAE divergente entre estabelecimentos: **2708**
- Filial mais antiga que a matriz: **30**
- Funcionários preenchidos: **139** (63 de 63 no treino)

Ausência de funcionários não foi convertida em zero. Capital zero não classifica o porte.

## Regras de agregação

| Campo | Regra |
|---|---|
| `cnpj_raiz` | Chave da empresa. Não entra como feature. |
| `qtd_estabelecimentos` | Quantidade de CNPJs completos distintos da raiz. |
| `capital_social` | Valor único da raiz, ou o da matriz se houver divergência. Nunca é somado. Zero é preservado. |
| `natureza_juridica` | Valor único da raiz, ou o da matriz se houver divergência. |
| `cnae_principal` | CNAE da matriz quando há exatamente uma. Senão, o do estabelecimento com início mais antigo. |
| `qtd_cnaes_distintos` | CNAEs principais diferentes entre os estabelecimentos da raiz. |
| `qtd_divisoes_cnae` | Divisões (dois primeiros dígitos) distintas na raiz. |
| `situacao_matriz` | Situação cadastral da matriz. Filial não substitui matriz ausente. |
| `qtd_situacao_*` | Contagem de estabelecimentos por situação cadastral. |
| `data_inicio_empresa` | Menor data de início válida da raiz. É a referência de idade. |
| `data_inicio_matriz` | Data de início da matriz, preservada para auditoria. |
| `funcionarios` | Valor único observado. Ausência permanece nula e não vira zero. |
| `rotulo` | 0 = Médio e 1 = Grande, só quando o porte é único na raiz. |
| `conjunto` | treino, inferencia, revisao ou fora_do_escopo. |
