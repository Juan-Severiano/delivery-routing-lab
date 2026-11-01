# delivery-routing-lab

Laboratório de roteamento de entregadores sobre as **ruas reais de Itapipoca (CE)**: um simulador de frota, baselines clássicos e um modelo próprio medido contra eles, com um mapa para ver onde cada entregador está, qual rota ele segue e por que o modelo escolheu a próxima parada.

> **Os dados são sintéticos.** Só as ruas vêm do OpenStreetMap. Restaurantes, pedidos, entregadores, trânsito e tempos de preparo serão gerados pelo simulador; nada aqui vem de operação real nem de cardápio ou cliente de lojista.

A pergunta que guia o projeto: **quanto um modelo treinado consegue se aproximar de um planejador caro que conhece o trânsito de verdade, respondendo em microssegundos?**

## Estado

Projeto no **Passo 5** (modelos próprios de preparo e ETA dentro do planejador B1+). Falta a decisão da próxima parada (Jev) e a explicação no mapa.

| | Estado |
|---|---|
| Ruas de Itapipoca (baixar do OSM, montar o grafo, maior componente conectada) | feito |
| Mapa em Vite/React com pan, zoom e camadas por classe de via | feito |
| Restaurantes sintéticos, curva de demanda por hora e pedidos do dia (por semente) | feito |
| Mapa de calor de demanda e restaurantes online por hora no mapa | feito |
| Caminhos mínimos entre todos os cruzamentos (Python puro) | feito |
| Trânsito escondido, ciclo do pedido, entregadores com turno e mochila, simulador por eventos | feito |
| Baseline B0 (entregador livre mais próximo, parada mais próxima) e métricas por dia | feito |
| Replay do dia e mapa com entregadores andando, pedidos em aberto e rota do entregador | feito |
| Planejadores B1 (inserção mais barata + 2-opt, sem ver trânsito nem preparo) e B2 (o teto) | feito |
| Mapa com a escolha da política (B0, B1, B1+, B2) e a comparação do dia | feito |
| Biblioteca própria de treino sobre o autograd (rede, Adam, loop, inferência em numpy) | feito |
| Modelo de preparo por restaurante e hora, como distribuição calibrada | feito |
| Modelo de ETA dos trechos | feito |
| B1+ (B1 com os dois modelos no lugar das regras) | feito |
| Modelos próprios: próxima parada (`decide`) e aceite de oferta | não feito |
| Explicação da escolha no mapa | não feito |

## O que é medido (plano)

Uma escada, não um número solto: o **piso** é o vizinho mais próximo, o **teto** é o planejador com os tempos reais, e o modelo próprio tem de ficar no meio.

- Manchete: **% de entregas no prazo** e **atraso P90**.
- Junto: tempo total do pedido, distância da frota, erro do ETA em minutos, calibração das probabilidades e latência do `decide()`.
- Semente fixa, dias de treino separados dos de teste e variação entre sementes.
- O modelo imita o teto, então provavelmente **não o supera**. O que se mede é quanto chega perto e quanto mais rápido responde.

## "Modelo próprio"

PyTorch entra **só como autograd**: `Tensor`, operações de tensor (produto de matrizes, `tanh`, `exp`) e `backward()`. A rede (com embedding), a inicialização, as perdas, o Adam, o loop de treino e a inferência em numpy estão em `src/delivery_lab/learn/`. Não usa `torch.nn`, `torch.optim`, `torch.utils.data`, scikit-learn, pesos prontos nem LLM. O torch é um extra opcional (`uv sync --extra train`): os pesos treinados ficam em `models/` (~55 KB, versionados) e o simulador e os planejadores rodam só com numpy. Os testes conferem o gradiente do autograd contra diferenças finitas, a inferência em numpy contra o forward do torch e o Adam contra a conta feita à mão.

## A cidade (Passo 1)

Medido sobre o recorte baixado do OSM (município inteiro de Itapipoca, só vias dirigíveis):

| | |
|---|---|
| Vias lidas / trechos entre cruzamentos | 4.187 / 10.170 |
| Cruzamentos antes e depois da maior componente fortemente conectada | 7.345 → 7.288 (99,2%) |
| Arestas dirigidas | 19.320 |
| Extensão (cada rua contada uma vez) | 2.594 km |
| Semáforos marcados no OSM | 23 |

Por classe: residencial 706 km, sem classe (`unclassified`) 1.501 km, terciária 118 km, secundária 68 km, principal 176 km, rodovia 25 km. A maior parte do `unclassified` é estrada rural; a demanda simulada vai se concentrar no miolo urbano, mas o grafo cobre o município todo.

### Suposições e limites

- **A velocidade livre é suposição, não medição.** Só 99 das 4.187 vias têm `maxspeed` no OSM. Sem ele, o laboratório usa um padrão por classe (de 25 km/h em rua residencial a 60 km/h em rodovia), pensando em moto. O trânsito real do simulador será outra função, que o modelo não vê diretamente.
- O OSM pode ter ruas faltando, mão única mal marcada e classes imprecisas. O grafo é tão bom quanto o mapa.
- Rampas e alças (`*_link`) andam a 70% da velocidade da via principal; é uma regra do laboratório.
- Calçadas, trilhas, vias de serviço e a pista de corrida ficam de fora.

## Restaurantes e demanda (Passo 2)

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

- **Restaurantes** ficam em cruzamentos do miolo urbano, com mais chance perto do centro e das avenidas. Cada um tem um tipo (marmita, restaurante, lanchonete, pizzaria, açaí e sorvete, espetinho, dia todo), uma janela online e um perfil de preparo **escondido**: tempo médio e dispersão, que crescem no pico de demanda. O modelo nunca verá esse perfil, só os horários de criado e pronto.
- **Pedidos** seguem uma curva por minuto (almoço, lanche da tarde, jantar e noite) e um endereço sorteado pela densidade de cruzamentos. O restaurante é escolhido entre os online dentro do raio, com peso de popularidade vezes `exp(-distância / 2 km)`.
- **Mapa de calor** é a média de 30 dias simulados por célula de 250 m e por hora, não uma fórmula.

### Suposições e limites

- **Todos os parâmetros do mundo são suposições do laboratório**: parcelas de cada tipo, janelas online, tempos de preparo, curva horária, popularidade e a escala de 600 cadastrados e ~5.500 pedidos por dia. Nada disso foi medido em operação real.
- A densidade de cruzamentos é um proxy da população; não é um censo.
- As distâncias deste passo são em linha reta. O caminho pelas ruas entra no simulador (Passo 3).
- Pedidos que não acham restaurante online no raio são descartados e contados à parte.

## Simulador e o baseline B0 (Passo 3)

Um dia simulado por semente: pedidos do Passo 2, entregadores com turno e mochila de 3 pedidos, trânsito escondido e preparo escondido. O B0 é o **piso** da escada: o pedido vai para o entregador **livre** mais próximo (se ninguém está livre, o mais próximo com espaço) e cada entregador faz sempre a parada mais próxima. Distâncias em linha reta. Ele não olha trânsito nem preparo, então manda o entregador no instante em que o pedido nasce e ele espera no restaurante.

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

### Suposições e limites do Passo 3

- **Todo o trânsito, o preparo, os turnos e o prazo são suposições do laboratório**; nada foi medido em operação real. O modelo vai aprender o que o simulador gera.
- O tempo real de uma aresta, medido em 20 dias, vai de 0,73 a 2,49 vezes o tempo livre (1% das arestas-hora abaixo de 0,88): o ruído de zona e do dia pode deixá-la mais rápida que o "livre", que é uma referência e não um teto físico.
- O B0 mede a posição do entregador no fim do trecho em curso, não a posição exata naquele instante.
- Só um dia (o dia 101) vai para o replay do mapa; as métricas são de 20 dias.

## Planejadores B1 e B2 (Passo 4)

Os dois usam a mesma máquina: cada entregador tem um plano (as paradas que ainda não começou). Quando um pedido nasce, o planejador testa, para os **40 entregadores elegíveis mais próximos**, todas as posições em que a coleta e a entrega cabem no plano (a entrega depois da coleta, até 3 pedidos) e fica com a que menos piora o custo. Depois o **2-opt** (inverter um trecho do plano) melhora a ordem das paradas daquele entregador. O custo de um plano é a soma, sobre os pedidos dele, do tempo desde a criação até a entrega mais **3 vezes** o atraso sobre o prazo prometido. Os dois também **seguram a saída** do entregador até "chegar quando o pedido fica pronto", e um pedido novo e perto pode entrar antes de uma coleta que ainda está cozinhando.

O que separa os degraus é só o que cada um enxerga:

- **B1** estima tempos pelo tempo livre das ruas e supõe ~12 min de preparo para qualquer restaurante (uma regra de bolso). Não vê trânsito nem a hora real em que o pedido fica pronto.
- **B2**, o teto, usa o tempo real das arestas na hora e a hora exata em que cada pedido fica pronto. Só não vê o ruído de 8% de cada viagem, sorteado quando ele despacha.

`delivery-lab sim run --policy b0|b1|b2 --days 20 --first-day 101`, 400 entregadores:

| | B0 (piso) | B1 | B2 (teto) |
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

A leitura: o B1 já corta a espera no restaurante pela metade e 20% dos km, mas **a taxa no prazo quase não sobe** (+2,0 pontos). O B2 sobe 15,2 pontos sobre o piso. Para dar uma régua de negócio: o B0 com **550 entregadores** (~37% a mais) ainda fica abaixo do B2 com 400 (91,5% contra 93,8%).

### Quanto vale saber cada coisa

Mesmos 20 dias, mesma frota, mesmo planejador; só muda o que ele enxerga:

| | B1 | B1 + preparo real | B1 + trânsito real | B2 (os dois) |
|---|---|---|---|---|
| Entregues no prazo (%) | 80,6 ± 3,2 | **92,9 ± 2,2** | 81,2 ± 3,2 | 93,8 ± 1,8 |
| Atraso P90 (min) | 6,7 | 0,2 | 6,5 | 0,0 |
| Tempo do pedido, média (min) | 32,8 | 28,4 | 32,6 | 28,0 |
| Espera no restaurante (min) | 7,1 | 0,2 | 7,6 | 0,5 |

**Saber a hora em que o pedido fica pronto vale quase todo o ganho do teto** (12,3 dos 13,2 pontos); saber o trânsito vale 0,6 ponto. Neste simulador, portanto, o modelo de **preparo** é o que mais pode render, e o ETA tem pouco a ganhar. Isso depende da força do trânsito sintético, que é um parâmetro nosso (o tempo dobra no pico, no centro, numa avenida); com trânsito mais forte o ETA valeria mais.

### O quanto o planejador é bom

- **Contra a força bruta:** em 1.000 instâncias aleatórias de 3 pedidos (cidade de brinquedo em grade, metade com os tempos do B1, metade com os do B2), a inserção mais barata seguida de 2-opt acha o **plano ótimo em 85%** dos casos; em média custa **0,92% a mais**, no pior caso 30%. Reprodutível com `OPTIMALITY_INSTANCES=1000 uv run pytest -s -k close_to_the_optimum`.
- **Parâmetros afinados só nos dias 1 a 10**, nunca nos 101 a 120. Candidatos: com 10, 20, 40 e 80 o B1 fez 76,7%, 78,2%, 79,2% e 80,0% no prazo, a 0,16, 0,28, 0,53 e 1,06 ms por decisão (medido antes de corrigir os turnos da abertura; a tendência se repetiu depois); fiquei com 40. Regra de bolso do B1: de 600 a 800 s o resultado fica dentro de 0,3 ponto (80,8%, 81,0%, 80,7%), 900 s perde 0,8 e 1.200 s perde 4,8; fiquei com 700 s. Margem de saída: 150 s no B1 (60 s e 300 s deram 0,1 e 0,5 ponto a menos) e 30 s no B2 (de 0 a 90 s ele muda menos de 0,2 ponto). Peso do atraso: 3 (com 1 ou 8 o B2 muda menos de 0,2 ponto).

### Desempenho e a decisão sobre Go

O planejador **é** o gargalo agora: `assign` consome ~90% do tempo do B1 e ~85% do B2 (105.511 decisões em 20 dias; 0,54 e 1,04 ms em média). Mesmo assim, 20 dias de B2 rodam em ~2 minutos, então **Go continua sem justificativa**. A pergunta volta se o Passo 6 precisar de muitas centenas de dias de decisões do B2 como dado de treino; antes de Go, o próximo passo seria vetorizar mais em numpy.

### Suposições e limites do Passo 4

- O B2 é um **teto no sentido de "vê tudo, menos o ruído da viagem"**, não um ótimo: ele usa heurística (inserção + 2-opt, só 40 candidatos) e custo próprio, e o ótimo global não foi calculado.
- O B2 mede o trânsito na hora em que decide, não na hora em que cada trecho de fato acontece.
- A regra de bolso do B1 é uma suposição; uma plataforma de verdade teria algo melhor que "12 min para todos", e é exatamente o que o modelo de preparo vai tentar ser.
- O resultado "preparo vale mais que trânsito" vale para este simulador e para os parâmetros que escolhi.

## Modelos próprios: preparo e ETA (Passo 5)

A ablação do Passo 4 mostrou o que valia aprender: o **preparo** primeiro, o ETA depois. Os dois modelos treinam em dias simulados e são avaliados em outros: **treino nos dias 21 a 60, validação (parada antecipada) nos 61 a 70, teste nos 101 a 120**. Os dias 1 a 10 ficam para afinar o planejador.

### Preparo por restaurante e hora

A rede recebe o restaurante (embedding de 8 números), o tipo, a hora do dia (3 harmônicos) e se chove, e devolve a **média e o desvio do log do preparo**, treinada com a perda gaussiana. O modelo não sabe que o preparo do simulador é lognormal. Ter uma distribuição permite ao planejador escolher um quantil e permite medir se as probabilidades estão calibradas. O que ele vê é o que uma plataforma tem quando o pedido nasce: o restaurante, a hora e a previsão do tempo. Treina em 7 s no total, contando a geração dos dados (212.389 pedidos).

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

O B1+ é o planejador do Passo 4 com o **preparo previsto** (o quantil 0,6, afinado nos dias 1 a 10) no lugar dos 12 min de bolso e o **ETA aprendido** no lugar do tempo livre. Nunca vê o preparo nem o trânsito verdadeiros (um teste confere que as estimativas dele não dependem do preparo verdadeiro). `delivery-lab sim run --policy b1+ --days 20 --first-day 101`, 400 entregadores, nos mesmos 20 dias de teste:

| | B1 | B1 + só preparo | B1 + só ETA | **B1+** | B2 (teto) |
|---|---|---|---|---|---|
| Entregues no prazo (%) | 80,6 ± 3,2 | 90,2 ± 3,0 | 81,1 ± 3,1 | **91,2 ± 2,4** | 93,8 ± 1,8 |
| Atraso médio (min) | 2,0 | 0,6 | 2,0 | 0,6 | 0,4 |
| Atraso P90 (min) | 6,7 ± 1,8 | 0,5 ± 1,1 | 6,5 ± 1,8 | 0,4 ± 0,8 | 0,0 ± 0,0 |
| Tempo do pedido, média (min) | 32,8 | 31,8 | 32,6 | 31,2 | 28,0 |
| Tempo do pedido, P90 (min) | 52,2 | 50,6 | 52,5 | 50,0 | 45,5 |
| Espera no restaurante (min) | 7,1 | 1,9 | 7,7 | 2,1 | 0,5 |
| Km rodados por pedido | 3,89 | 4,37 | 3,72 | 4,25 | 4,17 |
| Planejar um pedido (média) | 0,51 ms | 0,47 ms | 0,69 ms | 0,70 ms | 0,98 ms |

O B1+ **fecha 80% da distância entre o B1 e o teto** (10,6 dos 13,2 pontos). A comparação é pareada, nos mesmos dias: o B1+ ganha do B1 em **20 dos 20 dias** (+10,6 pontos em média, entre +9,4 e +12,6) e fica atrás do B2 nos 20 dias (2,6 pontos em média, entre 1,9 e 4,2). O preparo aprendido responde por quase tudo (9,6 pontos); o ETA sozinho dá só 0,5 e soma 1,0 por cima do preparo, como a ablação do Passo 4 já indicava (o trânsito vale pouco neste simulador). O B1+ roda mais km por pedido que o B1 (4,25 contra 3,89); não investiguei a causa.

Parâmetros afinados **só nos dias 1 a 10**: quantil do preparo (0,5, 0,6 e 0,75) e margem de saída (30, 90 e 150 s), 9 combinações, todas entre 91,4% e 91,9% no prazo; ficaram 0,6 e 30 s.

### Suposições e limites do Passo 5

- **Os modelos aprendem o que o nosso simulador gera.** O preparo é lognormal com média por tipo e hora, e o trânsito tem uma forma simples; chegar no piso mostra que o treino funciona, não que preparo ou trânsito reais sejam tão previsíveis. Com operação real haveria mudança de comportamento, dados faltantes e muito mais ruído.
- O rótulo do preparo é **limpo**: supomos que a plataforma saiba exatamente quando o pedido ficou pronto. Na vida real ele seria censurado pela chegada do entregador.
- O treino de preparo inclui os pedidos que o aplicativo recusaria por falta de entregador de turno (0,8%).
- A previsão do tempo (chuva) é tratada como pública: os dois modelos e o B1+ a usam.
- O ETA é treinado com trechos do B0; um planejador diferente passa por pares de nós em outra distribuição.
- O modelo de preparo depende do mundo com que foi treinado (600 restaurantes, semente 7): outro mundo exige treinar de novo (o planejador confere).

## Como rodar

Requer [uv](https://docs.astral.sh/uv/) e Node 20.19 ou mais novo.

```bash
uv sync --extra dev --extra train   # --extra train só é preciso para treinar (instala o torch, ~580 MB)
uv run delivery-lab city fetch      # baixa as ruas do Overpass (um pedido, ~7 MB)
uv run delivery-lab city build      # grafo + JSON do mapa
uv run delivery-lab world build     # restaurantes, demanda e JSON do mapa (--preset dev para o pequeno)
uv run delivery-lab paths build     # caminhos mínimos entre todos os cruzamentos (~20 s, ~320 MB, não versionado)
uv run delivery-lab train prep      # modelo de preparo (7 s) -> models/prep.npz
uv run delivery-lab train eta       # modelo de ETA (20 s) -> models/eta.npz
uv run delivery-lab sim run --policy b2 --days 20 --first-day 101   # métricas de 20 dias de teste (b0, b1, b1+, b2)
uv run delivery-lab sim run --policy b1 --days 1 --first-day 101 --render-day 101 --render 'web/public/data/replay-{policy}.json'   # replay para o mapa (repita com b0, b1+ e b2)
uv run pytest
npm --prefix web install
npm --prefix web run dev            # http://localhost:5173
```

`city fetch` faz um único pedido ao Overpass API e se recusa a baixar de novo se o arquivo existir.

## Licenças e créditos

- Código: MIT (ver `LICENSE`).
- Ruas e mapa: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), licença ODbL. O recorte baixado e o JSON do mapa derivado dele **não são versionados**; os scripts os recriam.
