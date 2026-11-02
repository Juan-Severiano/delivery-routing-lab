export type FlowGraphId = 'visao-geral' | 'ciclo-pedido' | 'treinamento' | 'avaliacao'

export type FlowNodeKind =
  | 'process'
  | 'decision'
  | 'model'
  | 'result'
  | 'policy'
  | 'teacher'
  | 'note'
  | 'rejected'

export interface FlowNode {
  id: string
  label: string
  kind: FlowNodeKind
}

export interface FlowEdge {
  id: string
  source: string
  target: string
  label?: string
  dashed?: boolean
}

export interface FlowGraph {
  id: FlowGraphId
  title: string
  description: string
  direction: 'TB'
  nodes: FlowNode[]
  edges: FlowEdge[]
}

function node(id: string, label: string, kind: FlowNodeKind = 'process'): FlowNode {
  return { id, label, kind }
}

function edge(source: string, target: string, label?: string, dashed?: boolean): FlowEdge {
  return { id: `${source}-${target}`, source, target, label, dashed }
}

// Conteúdo dos quatro fluxogramas em docs/fluxos/*.mmd.
// IDs e ramos são mantidos para permitir comparar a versão interativa com a fonte.
export const flowGraphs: Record<FlowGraphId, FlowGraph> = {
  'visao-geral': {
    id: 'visao-geral',
    title: 'Da malha de Itapipoca ao replay público',
    description:
      'As ruas viram grafo e caminhos mínimos. O mundo sintético e a política alimentam o simulador por eventos. Modelos treinados offline são usados pela política. A simulação produz métricas e replay; o navegador apenas reproduz os arquivos.',
    direction: 'TB',
    nodes: [
      node('OSM', 'Ruas do OpenStreetMap'),
      node('G', 'Grafo dirigido\ne caminhos mínimos'),
      node('W', 'Mundo por semente\nPedidos, preparo, trânsito e frota'),
      node('M', 'Treino offline\nQuatro modelos salvos em .npz', 'model'),
      node('P', 'Política escolhida\nB0, B1, B1+, Jev ou B2', 'policy'),
      node('S', 'Simulador por eventos\nExecuta o dia sintético'),
      node('E', 'Avaliação\nMétricas por dia e gráficos', 'result'),
      node('R', 'Replay JSON\nPosições, paradas e decisões', 'result'),
      node('UI', 'Mapa no navegador\nLê e reproduz os arquivos'),
    ],
    edges: [
      edge('OSM', 'G'),
      edge('W', 'S'),
      edge('G', 'S'),
      edge('M', 'P'),
      edge('P', 'S'),
      edge('S', 'E'),
      edge('S', 'R'),
      edge('R', 'UI'),
    ],
  },
  'ciclo-pedido': {
    id: 'ciclo-pedido',
    title: 'O ciclo de um pedido no simulador',
    description:
      'Sem entregadores de turno, o pedido é recusado. Com turno, espera capacidade na fila. Após atribuir, um entregador ocupado conclui o trecho ou atendimento até FREE. Quando livre, escolhe parada e saída, podendo agendar WAKE. Coleta e entrega atualizam a mochila e geram novas decisões.',
    direction: 'TB',
    nodes: [
      node('N', 'Pedido criado'),
      node('T', 'Há entregador\nde turno?', 'decision'),
      node('R', 'Recusa registrada\nFora das métricas de entrega', 'rejected'),
      node('Q', 'Fila de pedidos pendentes'),
      node('A', 'assign encontra\ncapacidade?', 'decision'),
      node('F', 'Aguardar FREE, SHIFT_START\nou chegada de novo pedido'),
      node('B', 'Atribui ao entregador\ne inclui na mochila'),
      node('BUSY', 'Está ocupado?', 'decision'),
      node('WAIT', 'Conclui trecho ou atendimento\naté o evento FREE'),
      node('C', 'next_stop escolhe\nparada e hora de saída', 'policy'),
      node('IDLE', 'Fica livre\naté um novo pedido'),
      node('D', 'Sair agora?', 'decision'),
      node('WA', 'Agenda WAKE\nReavalia o plano ao despertar'),
      node('P', 'Segue caminho mínimo\ncom tempo real simulado'),
      node('H', 'Chegou para\ncoletar ou entregar?', 'decision'),
      node('CO', 'Espera ficar pronto\ne faz a retirada'),
      node('EN', 'Registra entrega\ne libera capacidade'),
      node('FR', 'Evento FREE\nTenta próxima parada\ne pedidos pendentes'),
    ],
    edges: [
      edge('N', 'T'),
      edge('T', 'R', 'não'),
      edge('T', 'Q', 'sim'),
      edge('Q', 'A'),
      edge('A', 'F', 'não'),
      edge('F', 'Q'),
      edge('A', 'B', 'sim'),
      edge('B', 'BUSY'),
      edge('BUSY', 'WAIT', 'sim'),
      edge('WAIT', 'FR'),
      edge('BUSY', 'C', 'não'),
      edge('C', 'IDLE', 'sem parada viável'),
      edge('C', 'D', 'com parada'),
      edge('D', 'WA', 'não'),
      edge('WA', 'C'),
      edge('D', 'P', 'sim'),
      edge('P', 'H'),
      edge('H', 'CO', 'coletar'),
      edge('H', 'EN', 'entregar'),
      edge('CO', 'FR'),
      edge('EN', 'FR'),
      edge('FR', 'C'),
      edge('FR', 'Q'),
    ],
  },
  treinamento: {
    id: 'treinamento',
    title: 'Como os quatro modelos são treinados',
    description:
      'Preparo aprende a distribuição do log do tempo observado. ETA aprende o log da razão entre duração real e tempo livre em trechos percorridos pelo B0. Os dois fornecem previsões para as nove features do Jev; B2 fornece a escolha rotulada. O treino usa autograd, Adam próprio e a melhor época da validação, sem ajuste por temperatura.',
    direction: 'TB',
    nodes: [
      node('PD', 'Pedidos sintéticos\nRestaurante, tipo, hora e chuva\nAlvo: log do preparo observado'),
      node('PT', 'Rede de preparo\nPerda gaussiana no log\nAprende média e dispersão'),
      node('ED', 'Trechos percorridos pelo B0\nHora, posições, chuva e tempo livre\nAlvo: log do tempo real / livre'),
      node('ET', 'Rede de ETA\nErro quadrático no log da razão'),
      node('PM', 'prep.npz', 'model'),
      node('EM', 'eta.npz', 'model'),
      node('B2', 'B2 como professor\nVê preparo e trânsito verdadeiros\nRótulo: a próxima parada escolhida', 'teacher'),
      node('JV', 'Visão da plataforma\nNove fatores por candidata\ncom preparo e ETA previstos'),
      node('JT', 'Jev linear e Jev rede\nPontuação por candidata + softmax\nEntropia cruzada contra a escolha B2'),
      node('JM', 'jev-linear.npz\njev-mlp.npz', 'model'),
      node('LOOP', 'Tensores e autograd do PyTorch\nAdam e laço próprios\nMelhor época na validação', 'note'),
    ],
    edges: [
      edge('PD', 'PT'),
      edge('ED', 'ET'),
      edge('PT', 'PM'),
      edge('ET', 'EM'),
      edge('PM', 'JV'),
      edge('EM', 'JV'),
      edge('B2', 'JV'),
      edge('JV', 'JT'),
      edge('JT', 'JM'),
      edge('JT', 'LOOP', 'regra comum do treino', true),
    ],
  },
  avaliacao: {
    id: 'avaliacao',
    title: 'Duas avaliações: o modelo e a frota',
    description:
      'Dias 101 a 120 ficam separados do ajuste e do treino. Uma avaliação mede erros, imitação e calibração; a outra deixa cada política conduzir a frota. Resultados por dia viram JSON e gráficos. Melhor imitação não garante ganho operacional, e o replay de um dia é uma exibição separada.',
    direction: 'TB',
    nodes: [
      node('TEST', 'Teste separado\nDias 101 a 120\nMesmo mundo, pedidos e frota'),
      node('OFF', 'Avaliação dos modelos\nEm exemplos reservados'),
      node('SIM', 'Avaliação operacional\nCada política conduz a frota'),
      node('ERR', 'Erros de preparo e ETA\nPreparo: calibração dos quantis'),
      node('IMI', 'Jev contra o B2\nAcerto de imitação, NLL e ECE'),
      node('MO', 'models/*.json\nMétricas offline dos modelos', 'result'),
      node('OP', 'Prazo, atraso, tempo e distância\nPedidos sem entrega contados à parte'),
      node('SNAP', 'docs/resultados.json\nMétricas de cada dia', 'result'),
      node('PLOT', 'Gráficos reproduzíveis\nMédia e desvio entre dias', 'result'),
      node('DEMO', 'Replay do dia 101\nMapa público para inspecionar'),
    ],
    edges: [
      edge('TEST', 'OFF'),
      edge('TEST', 'SIM'),
      edge('OFF', 'ERR'),
      edge('OFF', 'IMI'),
      edge('ERR', 'MO'),
      edge('IMI', 'MO'),
      edge('SIM', 'OP'),
      edge('OP', 'SNAP'),
      edge('SNAP', 'PLOT'),
      edge('IMI', 'OP', 'não implica ganho', true),
      edge('SIM', 'DEMO', 'exibição separada', true),
    ],
  },
}
