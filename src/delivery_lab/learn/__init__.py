"""Modelos próprios do laboratório.

Regra do projeto: o PyTorch entra só como `Tensor` e `backward()` (autograd). Camadas, inicialização,
perdas, otimizador, loop de treino e a inferência em numpy são escritos aqui. Nada de `torch.nn`,
`torch.optim`, `torch.utils.data`, scikit-learn nem pesos prontos.
"""
