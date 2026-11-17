"""Características profundas: transferência de aprendizado com uma rede pré-treinada.

Uma rede convolucional treinada no ImageNet (MobileNetV2, formato ONNX) é executada com o
módulo `cv2.dnn` do OpenCV, sem PyTorch ou TensorFlow. A saída da rede (1000 pontuações de
classes do ImageNet, como "garrafa de água", "lata", "sacola plástica") é usada como vetor de
características de entrada para o classificador do projeto, que continua sendo treinado e
avaliado por nós sobre o TrashNet. A rede em si NÃO é reajustada (apenas extrai atributos).

Pré-processamento exigido pela rede: imagem RGB 224x224, valores em [0, 1], normalização por
canal com média/desvio do ImageNet. O objeto é colocado em um quadrado (com margem da cor do
fundo) antes de redimensionar, para não deformar objetos altos como garrafas.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

TAMANHO_REDE = 224
MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DESVIO = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CAMINHO_PADRAO = "modelos/mobilenetv2.onnx"


def quadrado_com_margem(img: np.ndarray) -> np.ndarray:
    """Centraliza a imagem em um quadrado, preenchendo com a cor mediana das bordas."""
    h, w = img.shape[:2]
    lado = max(h, w)
    borda = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]]).reshape(-1, 3)
    cor = tuple(int(c) for c in np.median(borda, axis=0))
    cima, esq = (lado - h) // 2, (lado - w) // 2
    return cv2.copyMakeBorder(img, cima, lado - h - cima, esq, lado - w - esq,
                              cv2.BORDER_CONSTANT, value=cor)


class ExtratorProfundo:
    """Carrega a rede ONNX e converte imagens BGR em vetores de características."""

    def __init__(self, caminho: str = CAMINHO_PADRAO):
        self.caminho = str(caminho)
        if not Path(self.caminho).exists():
            raise FileNotFoundError(
                f"Modelo pré-treinado não encontrado: {self.caminho}. "
                "Rode: python -m src.machine_learning.baixar_modelo")
        self.rede = cv2.dnn.readNetFromONNX(self.caminho)

    @staticmethod
    def disponivel(caminho: str = CAMINHO_PADRAO) -> bool:
        return Path(caminho).exists() and Path(caminho).stat().st_size > 0

    def extrair(self, img_bgr: np.ndarray) -> np.ndarray:
        quadrada = quadrado_com_margem(img_bgr)
        r = cv2.resize(quadrada, (TAMANHO_REDE, TAMANHO_REDE), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(r, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - MEDIA) / DESVIO
        blob = np.ascontiguousarray(np.transpose(rgb, (2, 0, 1))[None], dtype=np.float32)
        self.rede.setInput(blob)
        return self.rede.forward().flatten().astype(np.float32)
