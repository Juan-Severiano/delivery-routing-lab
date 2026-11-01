# delivery-routing-lab

Laboratório de roteamento de entregadores sobre as **ruas reais de Itapipoca (CE)**: um simulador de frota, baselines clássicos e um modelo próprio medido contra eles, com um mapa para ver onde cada entregador está, qual rota ele segue e por que o modelo escolheu a próxima parada.

> **Os dados são sintéticos.** Só as ruas vêm do OpenStreetMap. Restaurantes, pedidos, entregadores, trânsito e tempos de preparo serão gerados pelo simulador; nada aqui vem de operação real nem de cardápio ou cliente de lojista.

A pergunta que guia o projeto: **quanto um modelo treinado consegue se aproximar de um planejador caro que conhece o trânsito de verdade, respondendo em microssegundos?**

## Estado

Projeto no **Passo 2** (restaurantes e demanda sobre a cidade). Ainda não há entregadores, baseline nem modelo.

| | Estado |
|---|---|
| Ruas de Itapipoca (baixar do OSM, montar o grafo, maior componente conectada) | feito |
| Mapa em Vite/React com pan, zoom e camadas por classe de via | feito |
| Restaurantes sintéticos, curva de demanda por hora e pedidos do dia (por semente) | feito |
| Mapa de calor de demanda, restaurantes online por hora e pedidos do dia no mapa | feito |
| Entregadores e pedidos andando pelas ruas | não feito |
| Simulador de frota com trânsito escondido | não feito |
| Baselines: vizinho mais próximo, inserção + 2-opt, e o mesmo com tempo real (teto) | não feito |
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

## Como rodar

Requer [uv](https://docs.astral.sh/uv/) e Node 20.19 ou mais novo.

```bash
uv sync --extra dev
uv run delivery-lab city fetch      # baixa as ruas do Overpass (um pedido, ~7 MB)
uv run delivery-lab city build      # grafo + JSON do mapa
uv run delivery-lab world build     # restaurantes, demanda e JSON do mapa (--preset dev para o pequeno)
uv run pytest
npm --prefix web install
npm --prefix web run dev            # http://localhost:5173
```

`city fetch` faz um único pedido ao Overpass API e se recusa a baixar de novo se o arquivo existir.

## Licenças e créditos

- Código: MIT (ver `LICENSE`).
- Ruas e mapa: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), licença ODbL. O recorte baixado e o JSON do mapa derivado dele **não são versionados**; os scripts os recriam.
