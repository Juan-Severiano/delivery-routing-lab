# Capturas do mapa

[Voltar ao projeto](../README.md) · [Apresentação visual](delivery-routing-lab.html)

As três imagens foram capturadas da aplicação React em execução, sem montagem ou alteração dos dados exibidos. Os arquivos são JPEGs originais do navegador. As ruas são do OpenStreetMap; toda a operação é sintética.

## Preparação comum

1. Siga o [quickstart](../README.md#executar-o-mapa) e gere os cinco replays do dia **101**, com preset `plataforma`, semente **7**, **400 entregadores** e modelos de `models/`.
2. Abra o mapa no navegador, mantenha o relógio pausado em **12:30** e escolha **Jev**.
3. Selecione **Centro urbano**. O tema das capturas é escuro, seguindo a preferência de aparência do navegador. O enquadramento se adapta ao tamanho da janela.
4. Capture o viewport da aplicação, preservando a atribuição ao OpenStreetMap no canto inferior direito.

## Visão geral

Arquivo: [`assets/mapa-visao-geral.jpg`](assets/mapa-visao-geral.jpg), **1280 × 720**.

Camadas de entregadores, pedidos e restaurantes ligadas; calor desligado. Nenhum entregador selecionado. A barra lateral está no topo, com a identificação do projeto, o aviso de dados sintéticos, o relógio e a política Jev.

## Decisão do Jev

Arquivo: [`assets/mapa-decisao-jev.jpg`](assets/mapa-decisao-jev.jpg), **954 × 943**.

No mesmo replay às **12:30**, selecione o **entregador 162** (índice 161 nos dados). A última decisão registrada para ele ocorreu às **12:26**: duas coletas candidatas, com probabilidades arredondadas de **86% e 14%**. Role a barra lateral até mostrar o entregador, as candidatas e os seis fatores de maior contribuição. Os anéis no mapa marcam as alternativas.

Na imagem, o entregador já está no restaurante: a explicação é da decisão que originou seu último trecho, não de uma nova decisão às 12:30. As contribuições descrevem a pontuação do modelo; não estabelecem relações causais sobre uma operação real.

## Demanda média

Arquivo: [`assets/mapa-calor.jpg`](assets/mapa-calor.jpg), **954 × 943**.

Feche a seleção do entregador, desligue as camadas de entregadores, pedidos e restaurantes e ligue **Calor da demanda**. Expanda **Mundo simulado** e role a barra lateral para mostrar a legenda numérica.

O mapa exibe a demanda média da faixa das **12h**, em células de **250 m**, calculada em **30 dias simulados**, semente 7. A escala é compartilhada entre as horas e vai de 0 a **9,3 pedidos por hora**. Esse calor não é a contagem de pedidos abertos às 12:30 nem a média dos 20 dias de avaliação das políticas.

## Reprodução e licença

O tamanho da janela e a rolagem alteram apenas o enquadramento. Novos downloads do OSM podem mudar as ruas e a localização visual dos marcadores; preserve os dados e consulte os hashes do [snapshot de resultados](resultados.json) para comparar entradas. Estas capturas não substituem os resultados agregados do experimento.

Ruas e mapas derivados: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), ODbL. Nenhum pedido, restaurante ou entregador de uma operação comercial foi utilizado.
