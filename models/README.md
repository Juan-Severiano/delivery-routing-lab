# Modelos do delivery-routing-lab

**Ficha local preliminar. A publicação destes modelos no Hugging Face Hub ainda não foi realizada.**

Este diretório contém quatro checkpoints treinados no laboratório, com pesos e metadados suficientes para a inferência usando o código do projeto. São modelos supervisionados que recebem dados numéricos: prever preparo, estimar tempo de viagem e pontuar paradas candidatas. O mapa público exibe replays já calculados; ele não executa o treino nem os modelos no navegador.

## Quais modelos foram gerados

| Checkpoint | O que aprende | Arquitetura | Parâmetros treináveis | Tamanho do arquivo |
|---|---|---|---:|---:|
| [`prep.npz`](https://github.com/Juan-Severiano/delivery-routing-lab/blob/main/models/prep.npz) | Distribuição do tempo de preparo | Embedding de 600 restaurantes × 8; 14 features + embedding; MLP 22 → 48 → 48 → 2 | 8.354 | 33.721 bytes |
| [`eta.npz`](https://github.com/Juan-Severiano/delivery-routing-lab/blob/main/models/eta.npz) | Multiplicador sobre o tempo livre de um trecho | MLP 14 → 64 → 64 → 1 | 5.185 | 21.138 bytes |
| [`jev-linear.npz`](https://github.com/Juan-Severiano/delivery-routing-lab/blob/main/models/jev-linear.npz) | Pontuação da próxima parada, imitando a escolha do B2 | Linear 9 → 1; softmax entre as candidatas | 10 | 1.345 bytes |
| [`jev-mlp.npz`](https://github.com/Juan-Severiano/delivery-routing-lab/blob/main/models/jev-mlp.npz) | A mesma tarefa de imitação, com uma rede pequena | MLP 9 → 32 → 16 → 1; softmax entre as candidatas | 865 | 5.365 bytes |

Os quatro NPZ somam **61.569 bytes**, incluindo metadados e compressão. A contagem de parâmetros soma pesos, vieses e embedding; não conta médias, desvios ou os tipos dos restaurantes.

As camadas ocultas usam `tanh`. Inicialização, perdas, Adam e laço de treino foram implementados no projeto. PyTorch fornece tensores e autograd; o forward usado na inferência é NumPy. As classes de treino e de inferência são próprias, e não subclasses de `torch.nn.Module`.

Referência da implementação inspecionada: [commit b7f4cb7](https://github.com/Juan-Severiano/delivery-routing-lab/tree/b7f4cb71040e573c81a785a378fac4996b13bac8), com [rede e inferência NumPy](https://github.com/Juan-Severiano/delivery-routing-lab/blob/b7f4cb71040e573c81a785a378fac4996b13bac8/src/delivery_lab/learn/nn.py), [treino](https://github.com/Juan-Severiano/delivery-routing-lab/blob/b7f4cb71040e573c81a785a378fac4996b13bac8/src/delivery_lab/learn/fit.py) e [Adam](https://github.com/Juan-Severiano/delivery-routing-lab/blob/b7f4cb71040e573c81a785a378fac4996b13bac8/src/delivery_lab/learn/optim.py).

## Entradas, saídas e pré-processamento

### Preparo

Entradas: índice do restaurante no mundo sintético, tipo do restaurante, instante do pedido em segundos desde meia-noite e indicador de chuva. A hora vira seis features seno/cosseno, com três harmônicos; o tipo vira one-hot e o índice seleciona uma linha do embedding.

O alvo de treino é o log do preparo observado, em segundos, centralizado pela média do treino. A rede retorna dois valores que formam a média e o desvio do log do preparo. A perda gaussiana nesse espaço assume uma família lognormal, também usada pelo gerador do laboratório.

- `predict(...)` retorna `(mu, sigma)` do **log do tempo em segundos**, não minutos.
- `quantile_s(..., q)` retorna o quantil de preparo em **segundos**. Divida por 60 para obter minutos.
- O planejador B1+ usa o quantil `q=0.6`; o erro absoluto reportado abaixo usa a mediana `q=0.5`.

O checkpoint guarda pesos, embedding, vetor `restaurant_kind` com 600 entradas, `y_mean` e metadados de treino. [Implementação de preparo](https://github.com/Juan-Severiano/delivery-routing-lab/blob/b7f4cb71040e573c81a785a378fac4996b13bac8/src/delivery_lab/learn/prep.py).

### ETA dos trechos

Entradas: coordenadas projetadas da origem e do destino em metros, tempo livre do trecho em segundos, instante em segundos e chuva. A transformação usa três harmônicos da hora, posições relativas ao centro salvas no checkpoint, escala espacial de 5.000 m e transformações do tempo livre e da distância em linha reta.

O alvo é `log(tempo observado / tempo livre)`, centralizado pela média do treino. A perda é erro quadrático nesse espaço. `ratio(...)` devolve um **multiplicador sem unidade**; o tempo previsto em segundos é `tempo_livre_s * ratio(...)`.

O checkpoint guarda pesos, centro das coordenadas, `y_mean` e metadados. Coordenadas latitude/longitude não podem ser usadas diretamente no lugar das coordenadas projetadas. [Implementação de ETA](https://github.com/Juan-Severiano/delivery-routing-lab/blob/b7f4cb71040e573c81a785a378fac4996b13bac8/src/delivery_lab/learn/eta.py).

### Jev linear e rede

Cada candidata recebe nove features, nesta ordem:

1. Tempo estimado até a parada, em minutos.
2. Indicador de coleta.
3. Espera prevista no restaurante, em minutos.
4. Folga até o prazo, em minutos.
5. Idade do pedido, em minutos.
6. Indicador de atraso previsto (folga estimada < 0).
7. Distância às outras candidatas, expressa em minutos pela função do projeto.
8. Quantidade de pedidos na mochila depois da parada.
9. Tempo previsto que falta para ficar pronto, em minutos.

As features são normalizadas pelas médias e desvios do treino, guardados nos checkpoints. O modelo produz uma pontuação por candidata. O softmax é aplicado ao **conjunto de candidatas daquela decisão**, de modo que as probabilidades somam 1. O alvo é o índice da escolha do planejador B2; a perda é `-log(probabilidade da escolha do B2)`.

`decide(features, candidatas)` retorna a parada escolhida, probabilidades, confiança, margem sobre a segunda e, no modelo linear, contribuição de cada fator em relação à média das alternativas. As contribuições explicam a pontuação do modelo, não relações causais no mundo real.

O Jev escolhe a próxima parada; **a atribuição de pedidos continua com o planejador B1+**, usando os modelos de preparo e ETA. Para reproduzir a política completa, são necessários esses modelos, o planejador, o estado da simulação e a construção das candidatas viáveis. Só o NPZ do Jev não implementa a operação inteira. [Implementação do Jev](https://github.com/Juan-Severiano/delivery-routing-lab/blob/b7f4cb71040e573c81a785a378fac4996b13bac8/src/delivery_lab/learn/jev.py).

## Treino, validação e teste

Mundo principal: preset `plataforma`, semente 7, 600 restaurantes cadastrados, frota de 400 entregadores e mochila de até três pedidos. O planejador foi ajustado nos dias 1 a 10.

| Modelo | Dias de treino | Amostras de treino | Dias de validação | Amostras de validação | Dias de teste | Amostras de teste | Melhor época |
|---|---|---:|---|---:|---|---:|---:|
| Preparo | 21–60 | 212.389 pedidos | 61–70 | 52.882 | 101–120 | 106.401 | 26 |
| ETA | 21–60 | 390.904 trechos | 61–70 | 97.147 | 101–120 | 195.369 | 4 |
| Jev linear | 21–50 | 193.190 decisões | 61–65 | 32.524 | 101–120 | 129.013 | 29 |
| Jev rede | 21–50 | 193.190 decisões | 61–65 | 32.524 | 101–120 | 129.013 | 21 |

Preparo treina com rótulos gerados pelo simulador. ETA treina com trechos percorridos pelo B0. O conjunto do Jev é coletado com o B2 escolhendo paradas, mas as features entregues ao modelo usam a visão disponível à plataforma. O teste offline do Jev contém decisões com duas ou três candidatas. O treino usa parada antecipada pela validação, Adam e redução cossenoidal da taxa de aprendizado.

Separar dias mede novos sorteios do mesmo mundo. Não testa generalização entre cidades, restaurantes novos ou vários geradores. [Comandos e divisão na CLI](https://github.com/Juan-Severiano/delivery-routing-lab/blob/b7f4cb71040e573c81a785a378fac4996b13bac8/src/delivery_lab/cli.py).

## Resultados registrados

| Avaliação | Resultado do modelo | Referência |
|---|---|---|
| Preparo: erro absoluto médio da mediana | **4,77 min** | Regra fixa de 700 s: 11,60 min; tabela tipo × hora: 5,44 min; mediana verdadeira do simulador: 4,74 min |
| Preparo: erro médio de cobertura dos quantis | **0,0031** | Quantis previstos de 10% a 90% confrontados com a fração observada |
| ETA: erro absoluto médio por trecho | **23,38 s** | Tempo livre: 44,06 s; tabela por hora: 31,85 s |
| Jev linear: mesma escolha que B2 | **95,34%** | 129.013 decisões; ECE 0,0074 |
| Jev rede: mesma escolha que B2 | **96,51%** | 129.013 decisões; ECE 0,00072 |

Relatórios de modelo: [`prep.json`](https://github.com/Juan-Severiano/delivery-routing-lab/blob/main/models/prep.json), [`eta.json`](https://github.com/Juan-Severiano/delivery-routing-lab/blob/main/models/eta.json) e [`jev.json`](https://github.com/Juan-Severiano/delivery-routing-lab/blob/main/models/jev.json).

Com as políticas dirigindo a frota nos mesmos 20 dias de teste, as médias de entregas no prazo foram **91,21% no B1+**, **91,26% no Jev linear**, **90,96% no Jev rede** e **93,83% no B2**. A proximidade de médias não demonstra superioridade nem equivalência estatística. O maior acerto offline da rede não trouxe melhora operacional neste experimento.

A pontualidade considera os pedidos entregues: cada política recebeu 106.401 pedidos, entregou 105.511 e teve 890 sem entrega. Os valores por dia, configuração e hashes estão em [`docs/resultados.json`](https://github.com/Juan-Severiano/delivery-routing-lab/blob/main/docs/resultados.json). A comparação B1 → B1+ inclui também o ajuste da margem de saída de 150 para 30 s; não isola o modelo mantendo essa margem fixa.

## Usar um checkpoint sem treinar de novo

Execute da raiz do checkout, com Python 3.12+. A implementação de inferência calcula com NumPy, mas os módulos atuais ainda importam PyTorch; por isso o ambiente precisa do extra `train`.

```bash
uv sync --extra train
```

```python
from pathlib import Path
import numpy as np
from delivery_lab.learn.prep import PrepModel

model = PrepModel.load(Path("models/prep.npz"))
minutes = model.quantile_s(
    restaurant=np.array([1]),  # restaurante ativo do mundo sintético publicado
    t_s=np.array([12 * 3600]),
    rain=np.array([False]),
    q=0.6,
) / 60

print(minutes[0])  # aproximadamente 12.73077078
```

Esse exemplo foi executado com o checkpoint existente, sem retreino, API ou novo download de dados. O índice `1` pertence ao mundo de semente 7; não é um identificador de estabelecimento real.

Os NPZ foram inspecionados com `np.load(path, allow_pickle=False)`: todos abriram sem pickle e continham apenas arrays numéricos e strings Unicode, incluindo metadados JSON, sem arrays de objetos. Os métodos `load` do projeto chamam `np.load(path)` sem passar explicitamente esse argumento.

## Limites dos checkpoints

- Os modelos aprenderam o gerador sintético. Só as ruas são reais; não há avaliação em uma operação comercial.
- Os 600 índices de restaurante, seus tipos, o embedding e o centro de coordenadas estão ligados ao mundo usado no treino. Novos IDs ou outro mundo exigem rever os dados, o treinamento e a avaliação.
- A perda de preparo usa a mesma família lognormal do simulador, com rótulos limpos. Dados reais podem ter observações censuradas ou faltantes.
- O B2 usa informação privilegiada e uma heurística; não calcula o ótimo global. Imitá-lo não define um limite matemático de desempenho.
- A qualidade do Jev foi avaliada com mochila de três pedidos. As bancadas com planos maiores mediram latência, sem validar a qualidade nessas condições.
- Os arquivos são checkpoints para inferência e reavaliação. **Não guardam o estado do Adam, o estado aleatório ou o histórico completo do treinamento**, portanto não permitem retomá-lo exatamente do ponto em que parou.
- Pesos e metadados não substituem o código de pré-processamento. A ordem das features, escalas e construção das candidatas precisam acompanhar a versão dos pesos.

## Pode ir para o Hugging Face?

**Sim, como um repositório de modelos customizados.** O Hub aceita checkpoints, configurações e código de modelos próprios, sem exigir compatibilidade com Transformers. Um único repositório pode documentar esta família de quatro checkpoints e a dependência do Jev em preparo/ETA. [Documentação oficial de upload de modelos](https://huggingface.co/docs/hub/models-uploading).

Antes da publicação, o pacote deve reunir:

- Os quatro NPZ, os relatórios JSON e hashes dos artefatos.
- A revisão exata do código, as dependências e um exemplo de carregamento e inferência executável.
- A model card final com arquitetura, entradas/unidades, transformações, origem dos dados, divisão temporal, métricas e limites. Este arquivo é a ficha local de partida, não uma publicação no Hub.
- A identificação das licenças de cada componente do pacote. O código do projeto está sob MIT; a malha e o pacote público de dados possuem atribuição OpenStreetMap e aviso ODbL. A ficha e os arquivos a enviar devem registrar essa separação, em vez de declarar todo o conteúdo como MIT por padrão.

Os modelos foram inicializados no próprio laboratório; não são ajustes de um modelo de linguagem. Não devem ser apresentados como um `AutoModel` de Transformers ou como uma pipeline de texto. Para baixar um NPZ publicado no futuro, pode-se usar `hf_hub_download()` e então a classe `load` do projeto. As classes atuais também não herdam de `torch.nn.Module`, portanto `PyTorchModelHubMixin` não se aplica diretamente sem uma adaptação.

A publicação dos pesos cria uma página para descobrir e baixar os modelos. **Não cria automaticamente um servidor de inferência ou uma demonstração interativa.** Um Space é uma publicação separada; o mapa já está disponível no [GitHub Pages](https://juan-severiano.github.io/delivery-routing-lab/).

Referências oficiais para essa etapa futura: [Model cards](https://huggingface.co/docs/hub/model-cards), [download de arquivos](https://huggingface.co/docs/huggingface_hub/guides/download), [upload de arquivos](https://huggingface.co/docs/huggingface_hub/guides/upload) e [Spaces](https://huggingface.co/docs/hub/spaces-overview). Nenhum repositório, Space ou serviço de inferência foi criado no Hugging Face nesta etapa.

## Integridade dos pesos inspecionados

```text
prep.npz        e9ddd57e1a347aad060a26d3d7ce21a758fb68bd7aca8e9817945e4f1978ecf4
eta.npz         b32d6e4236ec94473c4a6602a1f9a0b36e4c519e030b08fe7deaddec33a62cb5
jev-linear.npz  e37aa8babe631dea970c2a9662d13e6255b2eb940dbf5da1e5a90c3527f34b05
jev-mlp.npz     6064e4caf431f8dee8056ebb65c81a4562d621af9a9be3eddcb6a25f5346ba7a
```

Hashes SHA-256 obtidos dos arquivos versionados existentes. Os pesos não foram alterados ao escrever esta ficha.
