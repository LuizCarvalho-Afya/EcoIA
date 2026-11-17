"""Conjuntos de características comparados no experimento e montagem do vetor de entrada."""
from __future__ import annotations

import numpy as np

CLASSICO = "classico"      # cor + forma/textura (OpenCV)
PROFUNDO = "profundo"      # rede pré-treinada (transferência de aprendizado)
COMBINADO = "combinado"    # clássico + profundo concatenados

DESCRICAO = {
    CLASSICO: "Características clássicas (cor HSV, HOG, bordas)",
    PROFUNDO: "Rede pré-treinada (MobileNetV2)",
    COMBINADO: "Clássicas + rede pré-treinada",
}


def montar(conjunto: str, classico: np.ndarray, profundo: np.ndarray | None) -> np.ndarray:
    """Seleciona/combina as características conforme o conjunto. Funciona com vetor ou matriz."""
    if conjunto == CLASSICO:
        return classico
    if profundo is None:
        raise ValueError(f"O conjunto '{conjunto}' exige características profundas.")
    if conjunto == PROFUNDO:
        return profundo
    if conjunto == COMBINADO:
        return np.concatenate([classico, profundo], axis=-1)
    raise ValueError(f"Conjunto de características desconhecido: {conjunto}")
