"""Leitura do dataset em pastas e extração das matrizes de características."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from src.constantes import CLASSES, MAPA_TRASHNET
from src.visao_computacional.extracao import carregar_imagem, processar_imagem

EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp"}


def listar_imagens(raiz, mapa: dict | None = None):
    """Percorre `raiz/<pasta>/*.jpg` e devolve (caminhos, rótulos).

    Cada pasta é mapeada para uma classe do projeto (ver MAPA_TRASHNET). Pastas já
    nomeadas com uma classe do projeto (plastico, papel, ...) também são aceitas; outras
    pastas (como __MACOSX) são ignoradas.
    """
    mapa = mapa or MAPA_TRASHNET
    raiz = Path(raiz)
    if not raiz.is_dir():
        raise FileNotFoundError(f"Pasta do dataset não encontrada: {raiz}")
    caminhos, rotulos = [], []
    for pasta in sorted(p for p in raiz.iterdir() if p.is_dir()):
        nome = pasta.name.lower()
        classe = mapa.get(nome, nome if nome in CLASSES else None)
        if classe is None:
            continue
        for arq in sorted(pasta.iterdir()):
            if arq.suffix.lower() in EXTENSOES:
                caminhos.append(str(arq))
                rotulos.append(classe)
    if not caminhos:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {raiz}")
    return caminhos, np.array(rotulos)


def extrair_matrizes(caminhos, rotulos, extrator=None, transformacao=None):
    """Extrai as características de cada imagem.

    Retorna (X_classico, X_profundo ou None, y, caminhos_mantidos). Imagens ilegíveis são
    ignoradas, com aviso. `extrator` (ExtratorProfundo) é opcional; `transformacao` é uma
    função imagem->imagem aplicada antes da extração (teste de robustez). As características
    profundas são calculadas sobre o MESMO recorte usado pelas clássicas.
    """
    Xc, Xp, y, mantidos = [], [], [], []
    for caminho, rotulo in zip(caminhos, rotulos):
        try:
            img = carregar_imagem(caminho)
            if transformacao is not None:
                img = transformacao(img)
            vetor, recorte, _ = processar_imagem(img)
            if extrator is not None:
                Xp.append(extrator.extrair(recorte))
            Xc.append(vetor)
            y.append(rotulo)
            mantidos.append(caminho)
        except ValueError as e:
            warnings.warn(str(e))
            if extrator is not None and len(Xp) > len(Xc):
                Xp.pop()
    return (np.vstack(Xc), np.vstack(Xp) if extrator is not None else None,
            np.array(y), mantidos)
