# delivery-routing-lab

Laboratório de roteamento de entregadores sobre as **ruas reais de Itapipoca (CE)**: um simulador de frota, baselines clássicos e um modelo próprio medido contra eles, com um mapa para ver onde cada entregador está, qual rota ele segue e por que o modelo escolheu a próxima parada.

> **Os dados são sintéticos.** Só as ruas vêm do OpenStreetMap. Restaurantes, pedidos, entregadores, trânsito e tempos de preparo serão gerados pelo simulador; nada aqui vem de operação real nem de cardápio ou cliente de lojista.

A pergunta que guia o projeto: **quanto um modelo treinado consegue se aproximar de um planejador caro que conhece o trânsito de verdade, respondendo em microssegundos?**

## Estado

Projeto no **Passo 3** (simulador de frota com o baseline B0). Ainda não há planejador clássico (B1 e B2) nem modelo.

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
| Baselines B1 (inserção + 2-opt, tempo livre) e B2 (o mesmo com tempo real, o teto) | não feito |
| Modelos próprios: ETA, preparo, próxima parada (`decide`), aceite de oferta | não feito |
| Explicação da escolha no mapa | não feito |

## O que é medido (plano)

Uma escada, não um número solto: o **piso** é o vizinho mais próximo, o **teto** é o planejador com os tempos reais, e o modelo próprio tem de ficar no meio.

- Manchete: **% de entregas no prazo** e **atraso P90**.
- Junto: tempo total do pedido, distância da frota, erro do ETA em minutos, calibração das probabilidades e latência do `decide()`.
- Semente fixa, dias de treino separados dos de teste e variação entre sementes.
- O modelo imita o teto, então provavelmente **não o supera**. O que se mede é quanto chega perto e quanto mais rápido responde.

## "Modelo próprio"

PyTorch entra **só como autograd** (`Tensor` e gradiente automático). Camadas, perda, otimizador e loop de treino são escritos aqui. Não usa `torch.nn`, `torch.optim`, scikit-learn, pesos prontos nem LLM. A inferência roda em numpy com os pesos exportados.

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

Medido com `delivery-lab sim run --days 20` no preset `plataforma` (~5.300 pedidos por dia), mesmos 20 dias para as três frotas, média ± desvio entre dias:

| Entregadores | 300 | 400 | 550 |
|---|---|---|---|
| Entregues no prazo (%) | 63,1 ± 3,6 | **77,0 ± 4,2** | 90,0 ± 2,6 |
| Atraso médio (min) | 16,2 ± 1,5 | 8,6 ± 1,0 | 5,7 ± 0,7 |
| Atraso P90 (min) | 44,0 ± 4,0 | 15,1 ± 3,9 | 0,9 ± 1,7 |
| Tempo do pedido, média (min) | 49,1 ± 2,2 | 39,2 ± 1,9 | 34,0 ± 1,4 |
| Tempo do pedido, P90 (min) | 89,9 ± 3,9 | 64,1 ± 4,0 | 49,6 ± 2,5 |
| Espera por entregador (min) | 6,5 | 5,0 | 4,9 |
| Espera no restaurante (min) | 11,5 | 14,3 | 17,4 |
| Km rodados por pedido | 4,94 | 4,91 | 4,49 |
| Utilização dos entregadores (%) | 84,1 | 72,5 | 57,5 |

A frota de **400 é a usada nas medições**: com 300 o piso fica saturado (84% de utilização) e com 550 sobra gente; em 400 há folga para um planejador melhorar. Nenhum dos 106.617 pedidos ficou sem entrega. A **espera no restaurante cresce com a frota** (11,5, 14,3, 17,4 min): é o desperdício que o B0 não evita, e é onde o preparo aprendido deve ajudar.

Como o dia é montado:

- **Caminho** é o de menor tempo livre (comprimento / velocidade livre da classe), calculado uma vez para todos os pares de cruzamentos e guardado. O entregador não escolhe outro caminho por causa do trânsito.
- **Tempo real** de cada aresta = tempo livre × `(1 + 1,0 × pico(hora) × centralidade × sensibilidade da classe × chuva)` × ruído por zona de 1 km (AR(1) ao longo das horas) × fator do dia, mais um ruído de 8% por viagem. O pico tem três picos (manhã fraca, almoço, fim de tarde), a centralidade cai com `exp(-distância ao centro / 2,5 km)` e chove em 20% dos dias.
- **Preparo** de cada pedido: média do restaurante naquela hora (mais lento no pico) × lognormal, mais 10% com chuva.
- **Prazo prometido** = criado + 10 min + preparo médio esperado + 6 min por km em linha reta restaurante-cliente. É a regra do laboratório; ela afeta o nível de "no prazo" de todos os degraus, não a comparação entre eles.
- **Turnos:** 25% almoço (11h a 15h), 25% dia todo (11h a 23h), 50% noite (17h30 à 1h), com ±30 min de variação por entregador.
- Retirada leva 60 s, entrega 60 s, e o pedido conta como entregue 45 s depois da chegada ao cliente.

### Desempenho (e a decisão sobre Go)

Medido neste Mac (Apple Silicon, 16 GB), Python 3.14, sem biblioteca de grafos:

| | |
|---|---|
| Caminhos mínimos, 7.288 origens (Dijkstra com `heapq`) | 18,1 s, 2,5 ms por origem (uma vez; 212 MB de tempos e 106 MB de predecessoras em disco) |
| Um dia simulado, ~5.300 pedidos e 400 entregadores | 0,21 s (~130 mil eventos por segundo) |
| O dobro da escala (~10.700 pedidos, 800 entregadores) | 0,46 s |
| Onde o tempo vai (profile de um dia) | reconstruir caminhos 45%, despacho e atribuição o resto |
| Mapa: posição dos 400 entregadores + pedidos em aberto por quadro | 0,03 ms |

**Go não se justifica neste passo.** O plano era portar só o que o profile mostrasse lento, e o caminho mínimo e o simulador ficam longe disso. O ponto a medir de novo é o planejador do Passo 4 (inserção + 2-opt a cada decisão).

### Suposições e limites do Passo 3

- **Todo o trânsito, o preparo, os turnos e o prazo são suposições do laboratório**; nada foi medido em operação real. O modelo vai aprender o que o simulador gera.
- O tempo real de uma aresta, medido em 20 dias, vai de 0,73 a 2,49 vezes o tempo livre (1% das arestas-hora abaixo de 0,88): o ruído de zona e do dia pode deixá-la mais rápida que o "livre", que é uma referência e não um teto físico.
- O B0 mede a posição do entregador no fim do trecho em curso, não a posição exata naquele instante.
- Só um dia (o dia 1) vai para o replay do mapa; as métricas acima são de 20 dias.

## Como rodar

Requer [uv](https://docs.astral.sh/uv/) e Node 20.19 ou mais novo.

```bash
uv sync --extra dev
uv run delivery-lab city fetch      # baixa as ruas do Overpass (um pedido, ~7 MB)
uv run delivery-lab city build      # grafo + JSON do mapa
uv run delivery-lab world build     # restaurantes, demanda e JSON do mapa (--preset dev para o pequeno)
uv run delivery-lab paths build     # caminhos mínimos entre todos os cruzamentos (~20 s, ~320 MB, não versionado)
uv run delivery-lab sim run --days 20 --render web/public/data/replay.json   # métricas do B0 e replay do dia 1
uv run pytest
npm --prefix web install
npm --prefix web run dev            # http://localhost:5173
```

`city fetch` faz um único pedido ao Overpass API e se recusa a baixar de novo se o arquivo existir.

## Licenças e créditos

- Código: MIT (ver `LICENSE`).
- Ruas e mapa: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), licença ODbL. O recorte baixado e o JSON do mapa derivado dele **não são versionados**; os scripts os recriam.
