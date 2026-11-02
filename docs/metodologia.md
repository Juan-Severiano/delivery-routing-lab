# Metodologia e resultados detalhados

[Demo publicada](https://juan-severiano.github.io/delivery-routing-lab/) · [Mapa interativo](https://juan-severiano.github.io/delivery-routing-lab/mapa/) · [Voltar ao projeto](../README.md) · [Apresentação visual](delivery-routing-lab.html) · [Resultados por dia](resultados.json)

Este documento descreve o mundo sintético, as políticas, os modelos e as medições do `delivery-routing-lab`. **Só a malha viária de Itapipoca (CE) vem de uma fonte real, o OpenStreetMap.** Pedidos, restaurantes, entregadores, turnos, preparo e trânsito são simulados. As porcentagens medem desempenho nesse ambiente; não representam impacto em uma operação comercial.

## Protocolo de comparação

- **Ambiente principal:** preset `plataforma`, semente 7, 600 restaurantes cadastrados, frota de 400 entregadores e capacidade de 3 pedidos por mochila. O volume varia por dia: aproximadamente 5,3 mil pedidos com restaurante disponível.
- **Unidade da avaliação operacional:** um dia simulado. As políticas usam os mesmos dias 101 a 120, pedidos, preparo, turnos e campo de trânsito gerados por semente. Rotas, estados posteriores e a associação do ruído de cada viagem aos trechos percorridos dependem da política. O quadro de resultados apresenta média e desvio padrão populacional entre os 20 dias (`numpy.std`, `ddof=0`).
- **Divisão temporal:** dias 1 a 10 para afinar parâmetros do planejador; preparo e ETA treinam nos dias 21 a 60 e validam nos 61 a 70; Jev treina nos 21 a 50 e valida nos 61 a 65. O teste de todos usa os dias 101 a 120. Separar dias testa novos sorteios da mesma simulação, sem demonstrar generalização para outra cidade ou outro gerador.
- **Métricas de operação:** taxa de entregas no prazo, atraso P90, tempo de pedido, espera no restaurante e quilômetros por pedido. Pedidos recusados por ausência de entregador de turno são contados à parte e ficam fora das métricas de entrega. Pedidos sem restaurante disponível no raio são descartados no gerador e também contados separadamente.
- **Métricas de modelo:** erro do preparo, erro do ETA, acerto de imitação do B2 e calibração do Jev. Acertar a escolha do B2 não equivale a entregar no prazo. A avaliação operacional deixa cada política dirigir os estados que ela mesma produz.
- **B2 é uma referência com informação privilegiada:** usa a mesma heurística de inserção e 2-opt, mas enxerga preparo verdadeiro e trânsito corrente do simulador. Não conhece o ruído de cada viagem, não resolve o ótimo global e não é um limite matemático para qualquer política possível.
- **Incerteza:** o desvio entre dias descreve variação, não um intervalo de confiança. Proximidade de médias e sobreposição de desvios não demonstram equivalência estatística. Há um único mundo principal (semente 7); não foi medida generalização entre várias cidades ou sementes de mundo.

Os [resultados por dia](resultados.json) e o [gerador de gráficos](../scripts/plot_results.py) permitem inspecionar a comparação. Os tempos de treino, de interface e das bancadas de latência abaixo são medições locais registradas no desenvolvimento; dependem do hardware e da carga. Uma nova execução pode mudar os tempos e o último dígito arredondado, sem mudar o protocolo.

A coleta versionada contém 106.401 pedidos por política: 105.511 entregues e 890 sem entrega (0,84%). A pontualidade é calculada sobre os entregues. Os totais coincidem entre as oito configurações nesta execução.

**A margem de saída também muda na ablação de preparo.** B1 e a variante só com ETA usam 150 s; variantes com preparo aprendido usam quantil 0,6 e margem de 30 s, ajustados nos dias 1 a 10. A comparação mede essas configurações completas. Não foi executada aqui uma ablação adicional mantendo a margem fixa para atribuir todo o ganho exclusivamente ao modelo de preparo.

## Arquitetura e treino

```text
OpenStreetMap → grafo dirigido → caminhos mínimos em tempo livre
                                  ↓
gerador de mundo → dias sintéticos → simulador por eventos → métricas + replay
                                  ↑                          ↓
                       políticas de despacho            mapa React
                          B0 · B1 · B1+ · Jev · B2
```

A biblioteca própria em [`src/delivery_lab/learn/`](../src/delivery_lab/learn/) implementa rede, inicialização, perdas, Adam, loop de treino e inferência em NumPy. PyTorch fornece tensores e autograd no treino. O código não usa `torch.nn`, `torch.optim`, `torch.utils.data`, scikit-learn, pesos prontos ou LLM. Os testes comparam gradientes com diferenças finitas, inferência NumPy com o forward de treino e Adam com uma atualização calculada à mão. Os modelos treinados ficam em [`models/`](../models/).

## A cidade

Medido sobre o recorte baixado do OSM (município inteiro de Itapipoca, só vias dirigíveis):

| | |
|---|---|
| Vias lidas / trechos entre cruzamentos | 4.187 / 10.170 |
| Cruzamentos antes e depois da maior componente fortemente conectada | 7.345 → 7.288 (99,2%) |
| Arestas dirigidas | 19.320 |
| Extensão (cada rua contada uma vez) | 2.594 km |
| Semáforos marcados no OSM | 23 |

Por classe: residencial 706 km, sem classe (`unclassified`) 1.501 km, terciária 118 km, secundária 68 km, principal 176 km, rodovia 25 km. A maior parte do `unclassified` é estrada rural; a demanda simulada se concentra no miolo urbano, mas o grafo cobre o município todo.

### Suposições e limites

- **A velocidade livre é suposição, não medição.** Só 99 das 4.187 vias têm `maxspeed` no OSM. Sem ele, o laboratório usa um padrão por classe (de 25 km/h em rua residencial a 60 km/h em rodovia), pensando em moto. O trânsito do simulador é outra função, que o modelo não vê diretamente.
- O OSM pode ter ruas faltando, mão única mal marcada e classes imprecisas. O grafo é tão bom quanto o mapa.
- Rampas e alças (`*_link`) andam a 70% da velocidade da via principal; é uma regra do laboratório.
- Calçadas, trilhas, vias de serviço e a pista de corrida ficam de fora.

## Restaurantes e demanda

O mundo sintético é gerado por semente (`delivery-lab world build`). Medido com a semente 7 no preset `plataforma`, em 30 dias simulados, neste Mac (Apple Silicon, 16 GB):

| | |
|---|---|
| Restaurantes cadastrados / ativos / pico online | 600 / 347 / 205 (das 18h às 21h) |
| Pedidos por dia (atendidos) | média 5.337, de 5.189 a 5.485 |
| Pedidos sem restaurante online no raio de 4 km | 3,5% (191 por dia), quase todos de madrugada |
| Hora de pico | 19h, 745 pedidos por hora (14,0% do dia); às 12h, 568 |
| Distância cliente-restaurante, em linha reta | mediana 1.478 m, P90 3.196 m, máximo 4.000 m |
| Restaurantes a menos de 2 km / 4 km do centro urbano | 72% / 93% |
| Preparo médio no pico contra a madrugada | 1,44 vezes |
| Tempo para gerar o mundo / amostrar um dia de pedidos | 0,023 s / 0,057 s |

O preset `dev` (40 restaurantes, ~900 pedidos por dia) existe para testes rápidos. Os dois presets são só parâmetros: `--restaurants`, `--orders` e `--radius` sobrescrevem.

Como o mundo é montado:

- **Restaurantes** ficam em cruzamentos do miolo urbano, com mais chance perto do centro e das avenidas. Cada um tem um tipo (marmita, restaurante, lanchonete, pizzaria, açaí e sorvete, espetinho, dia todo), uma janela online e um perfil de preparo **escondido**: tempo médio e dispersão, que crescem no pico de demanda. O modelo nunca recebe esse perfil, só os horários de criado e pronto.
- **Pedidos** seguem uma curva por minuto (almoço, lanche da tarde, jantar e noite) e um endereço sorteado pela densidade de cruzamentos. O restaurante é escolhido entre os online dentro do raio, com peso de popularidade vezes `exp(-distância / 2 km)`.
- **Mapa de calor** é a média de 30 dias simulados por célula de 250 m e por hora, não uma fórmula.

### Suposições e limites

- **Todos os parâmetros do mundo são suposições do laboratório**: parcelas de cada tipo, janelas online, tempos de preparo, curva horária, popularidade e a escala de 600 cadastrados e ~5.500 pedidos por dia. Nada disso foi medido em operação real.
- A densidade de cruzamentos é um proxy da população; não é um censo.
- As distâncias deste passo são em linha reta. O caminho pelas ruas entra no simulador.
- Pedidos que não acham restaurante online no raio são descartados e contados à parte.

## Simulador e o baseline B0

Um dia simulado por semente: pedidos do gerador de demanda, entregadores com turno e mochila de 3 pedidos, trânsito escondido e preparo escondido. O B0 é o **piso** da escada: o pedido vai para o entregador **livre** mais próximo (se ninguém está livre, o mais próximo com espaço) e cada entregador faz sempre a parada mais próxima. Distâncias em linha reta. Ele não olha trânsito nem preparo, então manda o entregador no instante em que o pedido nasce e ele espera no restaurante.

**Todas as medições deste README são de 20 dias de teste (dias 101 a 120)**, nunca usados para escolher parâmetro. Média ± desvio entre dias, preset `plataforma` (~5.300 pedidos por dia):

| B0 com N entregadores | 300 | 400 | 550 |
|---|---|---|---|
| Entregues no prazo (%) | 65,2 ± 3,4 | **78,6 ± 4,1** | 91,5 ± 2,7 |
| Atraso médio (min) | 11,4 ± 1,4 | 3,7 ± 1,0 | 1,0 ± 0,4 |
| Atraso P90 (min) | 40,1 ± 4,2 | 12,5 ± 4,3 | 0,6 ± 1,2 |
| Tempo do pedido, média (min) | 44,0 ± 2,1 | 34,1 ± 2,1 | 28,9 ± 1,4 |
| Tempo do pedido, P90 (min) | 86,4 ± 4,0 | 61,4 ± 4,1 | 47,8 ± 2,6 |
| Espera no restaurante (min) | 11,7 | 14,5 | 17,6 |
| Km rodados por pedido | 4,96 | 4,88 | 4,43 |
| Utilização dos entregadores (%) | 80,1 | 68,7 | 54,3 |

A frota de **400 é a usada nas medições**: com 300 o piso fica saturado (80% de utilização) e com 550 sobra gente; em 400 há folga para um planejador melhorar. A **espera no restaurante cresce com a frota** (11,7, 14,5, 17,6 min): é o desperdício que o B0 não evita. Pedidos criados quando **nenhum entregador está de turno** são recusados e ficam fora das métricas: 890 dos 106.401 pedidos (0,84%, ~45 por dia, todos de madrugada); os outros 105.511 foram entregues.

Como o dia é montado:

- **Caminho** é o de menor tempo livre (comprimento / velocidade livre da classe), calculado uma vez para todos os pares de cruzamentos e guardado. O entregador não escolhe outro caminho por causa do trânsito.
- **Tempo real** de cada aresta = tempo livre × `(1 + 1,0 × pico(hora) × centralidade × sensibilidade da classe × chuva)` × ruído por zona de 1 km (AR(1) ao longo das horas) × fator do dia, mais um ruído de 8% por viagem. O pico tem três picos (manhã fraca, almoço, fim de tarde), a centralidade cai com `exp(-distância ao centro / 2,5 km)` e chove em 20% dos dias.
- **Preparo** de cada pedido: média do restaurante naquela hora (mais lento no pico) × lognormal, mais 10% com chuva.
- **Prazo prometido** = criado + 10 min + preparo médio esperado + 6 min por km em linha reta restaurante-cliente. É a regra do laboratório; ela afeta o nível de "no prazo" de todos os degraus, não a comparação entre eles.
- **Turnos:** 25% almoço (10h a 14h30), 25% dia todo (10h a 23h), 50% noite (17h30 à 1h), com ±30 min de variação por entregador.
- Retirada leva 60 s, entrega 60 s, e o pedido conta como entregue 45 s depois da chegada ao cliente.

### Desempenho do simulador

Medido neste Mac (Apple Silicon, 16 GB), Python 3.14, sem biblioteca de grafos:

| | |
|---|---|
| Caminhos mínimos, 7.288 origens (Dijkstra com `heapq`) | 18,1 s, 2,5 ms por origem (uma vez; 212 MB de tempos e 106 MB de predecessoras em disco) |
| Um dia simulado com o B0, ~5.300 pedidos e 400 entregadores | 0,23 s (~120 mil eventos por segundo) |
| O dobro da escala (~10.700 pedidos, 800 entregadores) | 0,46 s |
| Mapa: posição dos 400 entregadores + pedidos em aberto por quadro | 0,03 ms |

### Suposições e limites

- **Todo o trânsito, o preparo, os turnos e o prazo são suposições do laboratório**; nada foi medido em operação real. O modelo aprende o que o simulador gera.
- O tempo real de uma aresta, medido em 20 dias, vai de 0,73 a 2,49 vezes o tempo livre (1% das arestas-hora abaixo de 0,88): o ruído de zona e do dia pode deixá-la mais rápida que o "livre", que é uma referência e não um limite físico.
- O B0 mede a posição do entregador no fim do trecho em curso, não a posição exata naquele instante.
- Só um dia (o dia 101) aparece no replay do mapa; as métricas são de 20 dias.

## Planejadores B1 e B2

Os dois usam a mesma máquina: cada entregador tem um plano (as paradas que ainda não começou). Quando um pedido nasce, o planejador testa, para os **40 entregadores elegíveis mais próximos**, todas as posições em que a coleta e a entrega cabem no plano (a entrega depois da coleta, até 3 pedidos) e fica com a que menos piora o custo. Depois o **2-opt** (inverter um trecho do plano) melhora a ordem das paradas daquele entregador. O custo de um plano é a soma, sobre os pedidos dele, do tempo desde a criação até a entrega mais **3 vezes** o atraso sobre o prazo prometido. Os dois também **seguram a saída** do entregador até "chegar quando o pedido fica pronto", e um pedido novo e perto pode entrar antes de uma coleta que ainda está cozinhando.

O que separa os degraus é só o que cada um enxerga:

- **B1** estima tempos pelo tempo livre das ruas e supõe ~12 min de preparo para qualquer restaurante (uma regra de bolso). Não vê trânsito nem a hora real em que o pedido fica pronto.
- **B2**, o referência privilegiada, usa o tempo real das arestas na hora e a hora exata em que cada pedido fica pronto. Só não vê o ruído de 8% de cada viagem, sorteado quando ele despacha.

`delivery-lab sim run --policy b0|b1|b2 --days 20 --first-day 101`, 400 entregadores:

| | B0 (piso) | B1 | B2 (referência privilegiada) |
|---|---|---|---|
| Entregues no prazo (%) | 78,6 ± 4,1 | 80,6 ± 3,2 | **93,8 ± 1,8** |
| Atraso médio (min) | 3,7 ± 1,0 | 2,0 ± 0,4 | 0,4 ± 0,1 |
| Atraso P90 (min) | 12,5 ± 4,3 | 6,7 ± 1,8 | 0,0 ± 0,0 |
| Tempo do pedido, média (min) | 34,1 ± 2,1 | 32,8 ± 1,3 | 28,0 ± 1,0 |
| Tempo do pedido, P90 (min) | 61,4 ± 4,1 | 52,2 ± 2,1 | 45,5 ± 1,8 |
| Espera no restaurante (min) | 14,5 | 7,1 | 0,5 |
| Km rodados por pedido | 4,88 | 3,89 | 4,17 |
| Planejar um pedido (média / P99) | | 0,54 / 1,21 ms | 1,04 / 2,00 ms |
| Tempo por dia simulado | 0,23 s | 3,15 s | 6,44 s |

A leitura: o B1 já corta a espera no restaurante pela metade e 20% dos km, mas **a taxa no prazo quase não sobe** (+2,0 pontos). O B2 sobe 15,2 pontos sobre o piso. Como comparação de escala dentro da simulação: o B0 com **550 entregadores** (~37% a mais) ainda fica abaixo do B2 com 400 (91,5% contra 93,8%).

### Quanto vale saber cada coisa

Mesmos 20 dias, mesma frota, mesmo planejador; só muda o que ele enxerga:

| | B1 | B1 + preparo real | B1 + trânsito real | B2 (os dois) |
|---|---|---|---|---|
| Entregues no prazo (%) | 80,6 ± 3,2 | **92,9 ± 2,2** | 81,2 ± 3,2 | 93,8 ± 1,8 |
| Atraso P90 (min) | 6,7 | 0,2 | 6,5 | 0,0 |
| Tempo do pedido, média (min) | 32,8 | 28,4 | 32,6 | 28,0 |
| Espera no restaurante (min) | 7,1 | 0,2 | 7,6 | 0,5 |

**Saber a hora em que o pedido fica pronto vale quase todo o ganho do referência privilegiada** (12,3 dos 13,2 pontos); saber o trânsito vale 0,6 ponto. Neste simulador, portanto, o modelo de **preparo** é o que mais pode render, e o ETA tem pouco a ganhar. Isso depende da força do trânsito sintético, que é um parâmetro nosso (o tempo dobra no pico, no centro, numa avenida); com trânsito mais forte o ETA valeria mais.

### O quanto o planejador é bom

- **Contra a força bruta:** em 1.000 instâncias aleatórias de 3 pedidos (cidade de brinquedo em grade, metade com os tempos do B1, metade com os do B2), a inserção mais barata seguida de 2-opt acha o **plano ótimo em 85%** dos casos; em média custa **0,92% a mais**, no pior caso 30%. Reprodutível com `OPTIMALITY_INSTANCES=1000 uv run pytest -s -k close_to_the_optimum`.
- **Parâmetros afinados só nos dias 1 a 10**, nunca nos 101 a 120. Candidatos: com 10, 20, 40 e 80 o B1 fez 76,7%, 78,2%, 79,2% e 80,0% no prazo, a 0,16, 0,28, 0,53 e 1,06 ms por decisão (medido antes de corrigir os turnos da abertura; a tendência se repetiu depois); fiquei com 40. Regra de bolso do B1: de 600 a 800 s o resultado fica dentro de 0,3 ponto (80,8%, 81,0%, 80,7%), 900 s perde 0,8 e 1.200 s perde 4,8; fiquei com 700 s. Margem de saída: 150 s no B1 (60 s e 300 s deram 0,1 e 0,5 ponto a menos) e 30 s no B2 (de 0 a 90 s ele muda menos de 0,2 ponto). Peso do atraso: 3 (com 1 ou 8 o B2 muda menos de 0,2 ponto).

### Desempenho e a decisão sobre Go

Na medição de perfil, o planejador foi o principal gargalo: `assign` consome ~90% do tempo do B1 e ~85% do B2 (105.511 decisões em 20 dias; 0,54 e 1,04 ms em média). Ainda assim, a medição original de 20 dias de B2 levou cerca de 2 minutos. O projeto permaneceu em Python; a coleta posterior de 55 dias de decisões do B2 levou 369 s. Uma migração de linguagem precisaria justificar o custo com uma carga maior e uma nova medição.

### Suposições e limites

- O B2 é um **referência privilegiada no sentido de "vê tudo, menos o ruído da viagem"**, não um ótimo: ele usa heurística (inserção + 2-opt, só 40 candidatos) e custo próprio, e o ótimo global não foi calculado.
- O B2 mede o trânsito na hora em que decide, não na hora em que cada trecho de fato acontece.
- A regra de bolso do B1 é uma suposição; uma plataforma de verdade teria algo melhor que "12 min para todos", e o modelo de preparo aprendido é a alternativa avaliada aqui.
- O resultado "preparo vale mais que trânsito" vale para este simulador e para os parâmetros que escolhi.

## Modelos próprios: preparo e ETA

A ablação do planejador clássico mostrou o que valia aprender: o **preparo** primeiro, o ETA depois. Os dois modelos treinam em dias simulados e são avaliados em outros: **treino nos dias 21 a 60, validação (parada antecipada) nos 61 a 70, teste nos 101 a 120**. Os dias 1 a 10 ficam para afinar o planejador.

### Preparo por restaurante e hora

A rede recebe o restaurante (embedding de 8 números), o tipo, a hora do dia (3 harmônicos) e se chove, e devolve a **média e o desvio do log do preparo**, treinada com a perda gaussiana. A perda gaussiana no log equivale a assumir preparo lognormal, a mesma família usada pelo simulador; essa compatibilidade facilita o aprendizado neste ambiente. Ter uma distribuição permite ao planejador escolher um quantil e permite medir se as probabilidades estão calibradas. O que ele vê é o que uma plataforma tem quando o pedido nasce: o restaurante, a hora e a previsão do tempo. Treina em 7 s no total, contando a geração dos dados (212.389 pedidos).

Erro absoluto médio do preparo nos 106.401 pedidos de teste:

| | Erro (min) |
|---|---|
| Regra de bolso do B1 (700 s para todos) | 11,60 |
| Média global do treino | 9,87 |
| Tabela tipo de restaurante x hora (mediana do treino) | 5,44 |
| **Modelo (mediana prevista)** | **4,77** |
| Piso: a mediana verdadeira do simulador | 4,74 |

O modelo fica a 0,03 min do piso, que é só o ruído do sorteio de cada pedido. A **calibração** também fecha: quando ele prevê o quantil de 10%, 20%, ..., 90%, o preparo observado ficou abaixo em 10,1%, 20,1%, 29,9%, 39,7%, 49,7%, 59,6%, 69,4%, 79,5% e 89,6% dos pedidos (erro médio de 0,003). O desvio do log que ele estima (0,281) bate com o do simulador (0,281).

### ETA dos trechos

A rede prevê o log da razão `tempo real / tempo livre` de um trecho com features baratas, que o planejador monta para milhares de pares por decisão sem reconstruir o caminho: hora, chuva, posições da origem e do destino, tempo livre e distância em linha reta. Treina nos trechos que os entregadores percorreram (390.904 trechos do B0, com o ruído de 8% de cada viagem), em 20 s no total, contando a coleta dos trechos.

Erro do tempo de um trecho nos 195.369 trechos de teste (49% deles em horas de pico):

| | Todos os trechos | Só horas de pico |
|---|---|---|
| Tempo livre (o que o B1 usa) | 44,1 s (15,2%, viés -12,2%) | 62,6 s (19,1%, viés -18,0%) |
| Tabela por hora (razão média do treino) | 31,8 s (10,1%) | 44,3 s (11,3%) |
| **Modelo** | **23,4 s (8,8%, viés +1,6%)** | **28,3 s (8,9%, viés +0,9%)** |
| Piso: o tempo real esperado sem o ruído de cada viagem | 17,9 s (6,4%) | 21,6 s (6,4%) |

### B1+: o B1 com os modelos no lugar das regras

O B1+ é o planejador do planejador clássico com o **preparo previsto** (o quantil 0,6, afinado nos dias 1 a 10) no lugar dos 12 min de bolso e o **ETA aprendido** no lugar do tempo livre. Nunca vê o preparo nem o trânsito verdadeiros (um teste confere que as estimativas dele não dependem do preparo verdadeiro). `delivery-lab sim run --policy b1+ --days 20 --first-day 101`, 400 entregadores, nos mesmos 20 dias de teste:

| | B1 | B1 + só preparo | B1 + só ETA | **B1+** | B2 (referência privilegiada) |
|---|---|---|---|---|---|
| Entregues no prazo (%) | 80,6 ± 3,2 | 90,2 ± 3,0 | 81,1 ± 3,1 | **91,2 ± 2,4** | 93,8 ± 1,8 |
| Atraso médio (min) | 2,0 | 0,6 | 2,0 | 0,6 | 0,4 |
| Atraso P90 (min) | 6,7 ± 1,8 | 0,5 ± 1,1 | 6,5 ± 1,8 | 0,4 ± 0,8 | 0,0 ± 0,0 |
| Tempo do pedido, média (min) | 32,8 | 31,8 | 32,6 | 31,2 | 28,0 |
| Tempo do pedido, P90 (min) | 52,2 | 50,6 | 52,5 | 50,0 | 45,5 |
| Espera no restaurante (min) | 7,1 | 1,9 | 7,7 | 2,1 | 0,5 |
| Km rodados por pedido | 3,89 | 4,37 | 3,72 | 4,25 | 4,17 |
| Planejar um pedido (média) | 0,51 ms | 0,47 ms | 0,69 ms | 0,70 ms | 0,98 ms |

O B1+ **fecha 80% da distância entre o B1 e o referência privilegiada** (10,6 dos 13,2 pontos). A comparação é pareada, nos mesmos dias: o B1+ ganha do B1 em **20 dos 20 dias** (+10,6 pontos em média, entre +9,4 e +12,6) e fica atrás do B2 nos 20 dias (2,6 pontos em média, entre 1,9 e 4,2). O preparo aprendido responde por quase tudo (9,6 pontos); o ETA sozinho dá só 0,5 e soma 1,0 por cima do preparo, como a ablação do planejador clássico já indicava (o trânsito vale pouco neste simulador). O B1+ roda mais km por pedido que o B1 (4,25 contra 3,89); não investiguei a causa.

Parâmetros afinados **só nos dias 1 a 10**: quantil do preparo (0,5, 0,6 e 0,75) e margem de saída (30, 90 e 150 s), 9 combinações, todas entre 91,4% e 91,9% no prazo; ficaram 0,6 e 30 s.

### Suposições e limites

- **Os modelos aprendem o que o nosso simulador gera.** O preparo é lognormal com média por tipo e hora, e o trânsito tem uma forma simples; chegar no piso mostra que o treino funciona, não que preparo ou trânsito reais sejam tão previsíveis. Com operação real haveria mudança de comportamento, dados faltantes e muito mais ruído.
- O rótulo do preparo é **limpo**: supomos que a plataforma saiba exatamente quando o pedido ficou pronto. Na vida real ele seria censurado pela chegada do entregador.
- O treino de preparo inclui os pedidos que o aplicativo recusaria por falta de entregador de turno (0,8%).
- A previsão do tempo (chuva) é tratada como pública: os dois modelos e o B1+ a usam.
- O ETA é treinado com trechos do B0; um planejador diferente passa por pares de nós em outra distribuição.
- O modelo de preparo depende do mundo com que foi treinado (600 restaurantes, semente 7): outro mundo exige treinar de novo (o planejador confere).

## O Jev: a próxima parada com probabilidades

Quando um entregador fica livre, o Jev escolhe a **próxima parada** entre as candidatas viáveis (coletas pendentes e entregas de pedidos já coletados) e diz por quê. É um contrato, `decide(features, candidatas) -> Decision`, que devolve a probabilidade de cada candidata, a confiança (a probabilidade da escolhida), a margem sobre a segunda e a contribuição de cada fator. Cada pedido contribui com no máximo uma parada viável (a coleta ou a entrega), então com mochila de 3 pedidos há **2 ou 3 candidatas** (média de 2,31).

A pontuação de cada candidata é um modelo próprio e a probabilidade é um softmax entre as candidatas, treinado para **imitar a escolha do B2**, o planejador que vê o preparo e o trânsito verdadeiros. O Jev só usa o que a plataforma enxerga, com o ETA e o preparo aprendidos do treino de preparo e ETA. Os 9 fatores: tempo até a parada, se é coleta, espera prevista no restaurante, folga até o prazo, idade do pedido, se já está atrasado, distância às outras paradas, pedidos na mochila depois e quanto falta para o pedido ficar pronto.

Dois modelos: um **linear com softmax**, em que a explicação é exata (a contribuição do fator j é `w_j × (z_j da candidata − média das candidatas)`, e a soma dá a pontuação da candidata menos a média), e uma **rede pequena** (32 e 16 unidades), só para comparar. Treinam nas decisões do B2 em dias simulados: 193.190 decisões (dias 21 a 50), validação com 32.524 (dias 61 a 65), teste com **129.013 decisões dos dias 101 a 120**. Coletar os dados custou 369 s de simulação do B2 (55 dias: 30 de treino, 5 de validação e 20 de teste); treinar leva 1 s o linear e 4 s a rede.

### Quanto o Jev acerta o que o B2 faz

Top-1 nas 129.013 decisões de teste (acertar a mesma parada que o B2 escolheu):

| | Acerto |
|---|---|
| Sorteio uniforme entre as candidatas | 44,9% |
| Sempre a parada mais próxima (a regra do B0) | 31,3% |
| Sempre a de menor folga até o prazo | 47,2% |
| Planejador re-otimizando agora, com a mesma visão da plataforma | 93,5% |
| **Jev linear** | **95,3%** |
| **Jev rede** | **96,5%** |

Os dois Jev ficam acima do planejador que re-otimiza com as mesmas estimativas. Uma explicação plausível, que não testei: imitar o B2 incorpora um pouco do que o preparo e o trânsito verdadeiros mudam, que estimativas pontuais perdem. O acerto é 95,2% com 2 candidatas e 95,7% com 3 (linear).

**Calibração** (linear): a probabilidade que ele dá à escolhida tem erro de calibração esperado de **0,007** (rede: 0,001) e a probabilidade média que ele dá à escolha do B2 é 0,925 (rede: 0,950). Por faixa de confiança, no linear:

| Confiança | Decisões | Confiança média | Acerto |
|---|---|---|---|
| 0,3 a 0,5 | 217 | 0,46 | 0,44 |
| 0,5 a 0,6 | 3.712 | 0,55 | 0,58 |
| 0,6 a 0,7 | 3.912 | 0,65 | 0,69 |
| 0,7 a 0,8 | 5.090 | 0,75 | 0,79 |
| 0,8 a 0,9 | 8.332 | 0,86 | 0,88 |
| 0,9 a 1,0 | 107.750 | 0,99 | 0,99 |

Ele é um pouco **subconfiante** nas faixas do meio (acerta mais do que diz). 83,5% das decisões têm confiança acima de 0,9: a maioria é fácil.

Os pesos do linear, por desvio padrão de cada fator (todos negativos: a pontuação sobe quando o fator é menor): espera prevista no restaurante -1,79, falta para o pedido ficar pronto -1,35, pedidos na mochila depois -0,90, é uma coleta -0,78, tempo até a parada -0,55, folga até o prazo -0,41, idade do pedido -0,38, já atrasado -0,34 e distância às outras paradas -0,29. Os fatores andam juntos (folga, atraso e idade, por exemplo), então os pesos individuais não se leem sozinhos: no mapa, leia o conjunto.

### O Jev dirigindo a frota

No painel do mapa, ao clicar num entregador, aparecem as candidatas da decisão que gerou o trecho em curso, a probabilidade de cada uma, as barras de quanto cada fator pesou na escolhida (para cima em verde, para baixo em vermelho) e anéis nas paradas candidatas, com o tamanho dado pela probabilidade.

A política **Jev** atribui como o B1+ (inserção mais barata com os modelos aprendidos) e usa o Jev para escolher a próxima parada. Mesmos 20 dias de teste, 400 entregadores (`delivery-lab sim run --policy jev --days 20 --first-day 101`):

| | B1+ | **Jev linear** | Jev rede | B2 (referência privilegiada) |
|---|---|---|---|---|
| Entregues no prazo (%) | 91,2 ± 2,4 | **91,3 ± 2,5** | 91,0 ± 2,5 | 93,8 ± 1,8 |
| Atraso P90 (min) | 0,4 ± 0,8 | 0,4 ± 0,8 | 0,5 ± 0,9 | 0,0 ± 0,0 |
| Tempo do pedido, média (min) | 31,2 | 31,3 | 31,3 | 28,0 |
| Espera no restaurante (min) | 2,1 | 2,1 | 2,1 | 0,5 |
| Km rodados por pedido | 4,25 | 4,27 | 4,26 | 4,17 |

**O Jev linear tem resultado próximo ao B1+** (91,3% contra 91,2% no prazo), acrescentando probabilidades e a explicação dos fatores. A diferença observada não demonstra superioridade operacional. **A rede acerta mais as escolhas do B2 (96,5% contra 95,3%), mas fica com 91,0% no prazo.** Melhor imitação offline não se converteu em melhora na frota. O desvio entre dias não é um intervalo de confiança nem um teste de equivalência; a comparação precisa considerar diferenças pareadas entre os mesmos dias.

### Velocidade e a decisão sobre Go

- No simulador, uma decisão do Jev (montar as features e pontuar) leva em média **50 µs** (P99 75 µs, 258.285 decisões).
- **Comparação justa nos mesmos estados** (13.165 decisões, planos de 4 paradas em média, no máximo 6): o planejador re-otimizando agora leva 75 µs (P99 320 µs) e o Jev 51 µs (P99 70 µs). Ou seja, **só 1,5 vez mais rápido na média**: com mochila de 3 pedidos o planejador exaustivo já é rápido, e a latência em microssegundos, por si só, não é o principal diferencial neste tamanho de mochila.
- A vantagem aparece quando o plano cresce. Planejar do zero, com todas as paradas, em uma bancada com pedidos aleatórios:

| Pedidos no plano | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| Planejador exaustivo | 0,8 ms | 1,3 ms | 13,8 ms | 1,2 s | 157 s |
| Jev (features + modelo) | 0,12 ms | 0,07 ms | 0,13 ms | 0,24 ms | 0,87 ms |

  Isto mede só a **latência**: o Jev foi treinado com no máximo 3 candidatas, e a qualidade dele com mais pedidos não foi avaliada.
- **Go não foi necessário.** Gerar os dados de 55 dias do B2 levou 369 s; o treino leva segundos.

### Suposições e limites

- **O Jev foi treinado para imitar o B2.** Isso define a tarefa supervisionada; não estabelece um limite matemático para a operação. O que ele entrega neste experimento é explicação, calibração e uma decisão que não depende de enumerar sequências. Não foi demonstrada vantagem operacional sobre o B1+.
- O acerto de 95% vem em boa parte de decisões fáceis (83,5% com confiança acima de 0,9). A avaliação da frota mede a política nos próprios estados que ela produz, além do acerto offline de imitação.
- A explicação é do **modelo**, não do mundo: diz o que pesou na pontuação do Jev, não o que causaria atraso. Os fatores são correlacionados.
- Na frota o Jev só decide a sequência; a atribuição de cada pedido continua pela inserção do B1+, e o plano da inserção segue existindo só para avaliar pedidos novos.
- Treinado e testado só com mochila de 3 pedidos; só o linear tem a explicação exata.

## Reprodução

Execute os comandos a partir da raiz do repositório. O [início rápido](../README.md#executar-o-mapa) prepara grafo, mundo, caminhos e replays. Para usar os pesos publicados, mantenha `--preset plataforma --seed 7` e a frota de 400 nas comparações principais. Mudar o mundo requer treinar novos modelos de preparo, ETA e Jev.

### Demo congelada e publicação

A [demo pública](https://juan-severiano.github.io/delivery-routing-lab/) contém a apresentação e o [mapa em `/mapa/`](https://juan-severiano.github.io/delivery-routing-lab/mapa/). O mapa reproduz o dia 101 já calculado para B0, B1, B1+, Jev e B2. Seu snapshot é separado da coleta de 20 dias em [resultados.json](resultados.json): navegar pela demo não reexecuta o experimento nem treina os modelos.

O diretório [`demo/`](../demo/README.md) versiona oito arquivos gzip: os sete JSONs de cidade, mundo e replays, mais o [recorte OSM original](../demo/itapipoca-roads.osm.json.gz). O [manifesto](../demo/manifest.json) identifica os arquivos e registra seus hashes. A fonte original também é disponibilizada no site em [`dados/itapipoca-roads.osm.json.gz`](https://juan-severiano.github.io/delivery-routing-lab/dados/itapipoca-roads.osm.json.gz). Os snapshots permitem publicar o mesmo replay sem consultar o Overpass ou depender de um novo treino.

Para construir o site, instale as dependências da interface e execute o [script de publicação](../scripts/build_pages.py):

```bash
npm --prefix web ci
python3 scripts/build_pages.py
python3 -m http.server --directory _site
```

O script compila o Vite e monta `_site/` com a apresentação, os arquivos de apoio e o mapa. Os JSONs da demo são restaurados diretamente em `_site/mapa/data/`; `web/public/data/` permanece intacto. A montagem usa Python 3 e Node, sem PyTorch, simulação ou chamada a API. A instalação inicial de dependências precisa de rede; a montagem posterior usa os arquivos locais.

Abra `http://localhost:8000/` e `http://localhost:8000/mapa/`. No GitHub, o workflow de Pages faz a mesma montagem e publica a cada push na `main`. Os dados distribuídos ficam sob [ODbL 1.0](../demo/LICENSE-data.txt), com atribuição ao OpenStreetMap; o código continua sob MIT.

### Comparação operacional nos 20 dias

Depois de preparar os dados pelo início rápido:

```bash
for policy in b0 b1 'b1+' jev b2; do
  uv run delivery-lab sim run --policy "$policy" \
    --preset plataforma --seed 7 --couriers 400 --days 20 --first-day 101
done
```

A CLI imprime cada métrica, média e desvio entre dias. O mapa mostra o replay de um dia e não deve ser usado como substituto dessas médias.

Para a ablação com previsões aprendidas, compare apenas o componente trocado:

```bash
for policy in b1 'b1+preparo' 'b1+eta' 'b1+' b2; do
  uv run delivery-lab sim run --policy "$policy" \
    --preset plataforma --seed 7 --couriers 400 --days 20 --first-day 101
done
```

Para a ablação com informação privilegiada, as opções são `b1-preparo` (pronto verdadeiro), `b1-transito` (trânsito verdadeiro) e `b2` (os dois). Elas não usam os modelos de previsão.

Os gráficos do projeto usam [resultados.json](resultados.json). Para redesenhá-los a partir desse arquivo, ou recolher as métricas novamente:

```bash
# Só redesenha os gráficos a partir dos resultados salvos.
uv run --with matplotlib python scripts/plot_results.py

# Reexecuta as políticas e atualiza o arquivo de resultados e os gráficos.
uv run --extra train --with matplotlib python scripts/plot_results.py --collect --workers 3
```

### Treinar de novo

Os pesos existentes permitem abrir o mapa sem treinar. Os comandos abaixo **substituem** os pesos e métricas correspondentes em `models/`. Use `--out` para gravar preparo/ETA em outro lugar; no Jev, `--models` seleciona a pasta que contém as entradas de preparo/ETA e recebe suas saídas.

```bash
uv sync --extra dev --extra train
uv run delivery-lab train prep --preset plataforma --seed 7
uv run delivery-lab train eta --preset plataforma --seed 7
uv run delivery-lab train jev --preset plataforma --seed 7
```

A CLI de treino fixa as divisões temporais descritas no protocolo. O treino de preparo e ETA precisa do grafo; o ETA e o Jev também precisam dos caminhos mínimos. Na medição original, o preparo levou 7 s, o ETA 20 s e a coleta do Jev aproximadamente 6 min, mais poucos segundos para ajustar os dois modelos. Não são estimativas garantidas para outra máquina.

### Testes e bancada do planejador

```bash
uv sync --extra dev --extra train
uv run pytest
OPTIMALITY_INSTANCES=1000 uv run pytest -s -k close_to_the_optimum
npm --prefix web ci
npm --prefix web run build
```

Os testes abrangem grafo dirigido e conectividade, caminhos mínimos, conservação dos eventos e pedidos, capacidade da mochila, precedência entre coleta e entrega, turnos, determinismo, métricas calculadas à mão, gradientes, calibração, persistência dos modelos e ausência de acesso indevido ao preparo verdadeiro por B1+/Jev. A bancada de ótimo compara heurística e enumeração apenas em instâncias pequenas de 3 pedidos; não prova a otimalidade da frota.

### Fonte do mapa e limites de reprodução

`city fetch` baixa um recorte atual do Overpass. Se `data/raw/itapipoca-roads.json` já existe, o comando se recusa a substituí-lo sem `--force`. Um download novo pode alterar nós, arestas, restaurantes sorteados e resultados.

Os arquivos de trabalho em `data/raw/`, `data/out/` e `web/public/data/` não são versionados. Para congelar a fonte usada neste experimento, o repositório distribui [`demo/itapipoca-roads.osm.json.gz`](../demo/itapipoca-roads.osm.json.gz), além dos sete JSONs compactados usados no mapa público. O recorte original informa timestamp OSM `2026-11-01T17:26:28Z`; seus hashes e os dos arquivos da demo ficam no [manifesto](../demo/manifest.json).

Para reconstruir o mesmo grafo, restaure o conteúdo desse gzip em `data/raw/itapipoca-roads.json`, preservando antes qualquer recorte existente, e então execute `city build` e `paths build`. Não use um download novo como substituto silencioso da fonte congelada. O pacote da demo abre o replay pronto; reconstruir o grafo e reexecutar a simulação continuam sendo etapas separadas.

Para reproduzir os resultados, use também semente 7, os mesmos pesos e as versões de dependências dos arquivos de lock. Os [resultados versionados](resultados.json) registram hashes dos dados de entrada e do código usados na coleta. O snapshot OSM fixa a fonte geográfica; os locks fixam as dependências, sem garantir tempos de execução idênticos entre máquinas.

Não há chamada a API de IA, serviço de cobrança ou credencial no experimento. O início rápido precisa de rede para instalar dependências e buscar o mapa; depois disso, geração de pedidos, simulação, treino, inferência e interface rodam localmente.

## Escopo e próximos experimentos

O laboratório publicado cobre as cinco políticas, modelos de preparo e ETA, comparação operacional, Jev com calibração e explicação, e replay no mapa. Oferta e aceite de despacho ainda não foram implementados. Acrescentá-los exige declarar uma hipótese, uma regra de referência e métricas próprias, porque rejeição de ofertas muda o processo de decisão.

Outras perguntas abertas: robustez a trânsito mais forte, rótulos de preparo atrasados ou censurados, mudanças na distribuição de pedidos e generalização para novas sementes ou cidades. O teste de latência com mais de 3 pedidos não avaliou a qualidade do Jev fora do tamanho de mochila usado no treino.

## Licenças e créditos

- Código: [MIT](../LICENSE).
- [Pacote demo](../demo/README.md), incluindo fonte OSM e dados derivados: [ODbL 1.0](../demo/LICENSE-data.txt). Os arquivos compactados e o manifesto são versionados; suas cópias geradas nos diretórios de trabalho continuam ignoradas.
- Ruas e mapas derivados: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright). Pedidos, restaurantes e entregadores são sintéticos.
