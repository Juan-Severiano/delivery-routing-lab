# delivery-routing-lab

Laboratório de roteamento de entregadores sobre as **ruas reais de Itapipoca (CE)**: um simulador de frota, baselines clássicos e um modelo próprio medido contra eles, com um mapa para ver onde cada entregador está, qual rota ele segue e por que o modelo escolheu a próxima parada.

> **Os dados são sintéticos.** Só as ruas vêm do OpenStreetMap. Restaurantes, pedidos, entregadores, trânsito e tempos de preparo serão gerados pelo simulador; nada aqui vem de operação real nem de cardápio ou cliente de lojista.

A pergunta que guia o projeto: **quanto um modelo treinado consegue se aproximar de um planejador caro que conhece o trânsito de verdade, respondendo em microssegundos?**

## Estado

Projeto no **Passo 1** (a cidade). Nada de simulação, baseline ou modelo existe ainda.

| | Estado |
|---|---|
| Ruas de Itapipoca (baixar do OSM, montar o grafo, maior componente conectada) | feito |
| Mapa em Vite/React com pan, zoom e camadas por classe de via | feito |
| Restaurantes, pedidos, entregadores e mapa de calor de demanda | não feito |
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

## Como rodar

Requer [uv](https://docs.astral.sh/uv/) e Node 20.19 ou mais novo.

```bash
uv sync --extra dev
uv run delivery-lab city fetch      # baixa as ruas do Overpass (um pedido, ~7 MB)
uv run delivery-lab city build      # grafo + JSON do mapa
uv run pytest
npm --prefix web install
npm --prefix web run dev            # http://localhost:5173
```

`city fetch` faz um único pedido ao Overpass API e se recusa a baixar de novo se o arquivo existir.

## Licenças e créditos

- Código: MIT (ver `LICENSE`).
- Ruas e mapa: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), licença ODbL. O recorte baixado e o JSON do mapa derivado dele **não são versionados**; os scripts os recriam.
