"""Fixtures dos testes.

ATENÇÃO: as imagens geradas aqui são SINTÉTICAS (formas coloridas) e servem apenas para
verificar que o código executa de ponta a ponta. Nenhuma métrica obtida com elas é um
resultado do projeto.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# cor base (BGR) de cada classe sintética
_CORES = {"plastic": (60, 60, 200), "paper": (200, 120, 60), "glass": (60, 180, 60),
          "metal": (60, 200, 220), "trash": (130, 130, 130)}


def gerar_imagem(classe: str, rng: np.random.Generator) -> np.ndarray:
    img = np.full((160, 160, 3), 235, np.uint8)
    cor = np.clip(np.array(_CORES[classe]) + rng.integers(-20, 20, 3), 0, 255).tolist()
    cx, cy = 80 + int(rng.integers(-10, 10)), 80 + int(rng.integers(-10, 10))
    if classe in ("plastic", "glass"):
        cv2.ellipse(img, (cx, cy), (35, 55), 0, 0, 360, cor, -1)
    else:
        cv2.rectangle(img, (cx - 45, cy - 40), (cx + 45, cy + 40), cor, -1)
    return img


@pytest.fixture(scope="session")
def dataset_sintetico(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("sintetico")
    rng = np.random.default_rng(0)
    for classe in _CORES:
        (raiz / classe).mkdir()
        for i in range(24):
            cv2.imwrite(str(raiz / classe / f"{i}.jpg"), gerar_imagem(classe, rng))
    return raiz


def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _campo_varint(num, v):
    return _varint(num << 3) + _varint(v)


def _campo_bytes(num, b):
    return _varint((num << 3) | 2) + _varint(len(b)) + b


def _campo_str(num, s):
    return _campo_bytes(num, s.encode())


def _value_info(nome, dims):
    shape = b"".join(_campo_bytes(1, _campo_varint(1, d)) for d in dims)
    tensor = _campo_varint(1, 1) + _campo_bytes(2, shape)
    return _campo_str(1, nome) + _campo_bytes(2, _campo_bytes(1, tensor))


def _no(entrada, saida, nome, op):
    return _campo_str(1, entrada) + _campo_str(2, saida) + _campo_str(3, nome) + _campo_str(4, op)


@pytest.fixture(scope="session")
def onnx_minimo(tmp_path_factory):
    """ONNX de brinquedo (média global por canal -> 3 números), escrito à mão em protobuf.

    Serve só para testar o código que carrega e executa uma rede via cv2.dnn. NÃO é a rede
    pré-treinada real (MobileNetV2), que precisa ser baixada com baixar_modelo.py.
    """
    grafo = (_campo_bytes(1, _no("input", "gap", "gap", "GlobalAveragePool"))
             + _campo_bytes(1, _no("gap", "output", "flat", "Flatten"))
             + _campo_str(2, "mini")
             + _campo_bytes(11, _value_info("input", [1, 3, 224, 224]))
             + _campo_bytes(12, _value_info("output", [1, 3])))
    modelo = _campo_varint(1, 8) + _campo_bytes(7, grafo) + _campo_bytes(8, _campo_varint(2, 13))
    caminho = tmp_path_factory.mktemp("onnx") / "mini.onnx"
    caminho.write_bytes(modelo)
    return str(caminho)
