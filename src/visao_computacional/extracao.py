"""Visão Computacional: segmentação simples e extração de características (OpenCV).

Fluxo:
    imagem BGR -> recorte do objeto (Otsu + morfologia + maior contorno)
               -> redimensionamento -> vetor de características
O vetor combina cor (histogramas HSV e estatísticas BGR) com forma/textura
(HOG, densidade de bordas, variância do Laplaciano, proporção de brilho especular).
"""
from __future__ import annotations

import cv2
import numpy as np

TAMANHO = 128
BINS_H, BINS_S, BINS_V = 16, 8, 8


def calcular_hog(cinza64: np.ndarray) -> np.ndarray:
    """HOG próprio (numpy + Sobel), independente da versão do OpenCV.

    Imagem 64x64 -> células de 8x8 px com 9 faixas de orientação (0-180 graus, gradiente sem
    sinal), blocos de 2x2 células com passo de 1 célula e normalização L2 por bloco.
    Resultado: 7 x 7 blocos x 36 = 1764 atributos.
    """
    g = cinza64.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.degrees(np.arctan2(gy, gx)) % 180.0
    faixa = np.minimum((ang / 20.0).astype(np.int32), 8)
    hist = np.zeros((8, 8, 9), np.float32)
    for b in range(9):
        hist[:, :, b] = (mag * (faixa == b)).reshape(8, 8, 8, 8).sum(axis=(1, 3))
    blocos = []
    for by in range(7):
        for bx in range(7):
            v = hist[by : by + 2, bx : bx + 2].ravel()
            blocos.append(v / np.sqrt(float((v * v).sum()) + 1e-6))
    return np.concatenate(blocos).astype(np.float32)


def decodificar_bytes(dados: bytes) -> np.ndarray:
    """Converte bytes (upload do Streamlit) em imagem BGR."""
    arr = np.frombuffer(dados, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Não foi possível decodificar a imagem enviada.")
    return img


def carregar_imagem(caminho) -> np.ndarray:
    img = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Não foi possível ler a imagem: {caminho}")
    return img


def recortar_objeto(img: np.ndarray, area_minima: float = 0.05):
    """Isola o objeto principal.

    Estratégia: limiarização de Otsu sobre a imagem em tons de cinza; se as bordas
    da máscara forem majoritariamente claras, a máscara é invertida (fundo claro).
    Após fechamento/abertura morfológica, recorta-se o maior contorno.

    Retorna (recorte, máscara, sucesso). Se nenhum contorno com área suficiente for
    encontrado, devolve a imagem inteira e sucesso=False.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mascara = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    borda = np.concatenate([mascara[0, :], mascara[-1, :], mascara[:, 0], mascara[:, -1]])
    if borda.mean() > 127:
        mascara = cv2.bitwise_not(mascara)
    kernel = np.ones((5, 5), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return img, mascara, False
    maior = max(contornos, key=cv2.contourArea)
    h, w = gray.shape
    if cv2.contourArea(maior) < area_minima * h * w:
        return img, mascara, False
    x, y, bw, bh = cv2.boundingRect(maior)
    return img[y : y + bh, x : x + bw], mascara, True


def _normalizar_hist(h: np.ndarray) -> np.ndarray:
    h = h.flatten().astype(np.float32)
    s = h.sum()
    return h / s if s > 0 else h


def _caracteristicas_cor(r: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(r, cv2.COLOR_BGR2HSV)
    hist_h = _normalizar_hist(cv2.calcHist([hsv], [0], None, [BINS_H], [0, 180]))
    hist_s = _normalizar_hist(cv2.calcHist([hsv], [1], None, [BINS_S], [0, 256]))
    hist_v = _normalizar_hist(cv2.calcHist([hsv], [2], None, [BINS_V], [0, 256]))
    bgr = r.reshape(-1, 3).astype(np.float32) / 255.0
    return np.concatenate([hist_h, hist_s, hist_v, bgr.mean(axis=0), bgr.std(axis=0)])


def _caracteristicas_forma(r: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)
    brilho = float((gray >= 240).mean())  # reflexos especulares (vidro/metal)
    bordas = float((cv2.Canny(gray, 100, 200) > 0).mean())
    lap = float(cv2.Laplacian(gray, cv2.CV_64F).var() / 1e4)
    hog = calcular_hog(cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA))
    return np.concatenate([np.array([brilho, bordas, lap], dtype=np.float32), hog])


# Tamanhos reais, calculados a partir das funções (evita números mágicos).
_amostra = np.zeros((TAMANHO, TAMANHO, 3), dtype=np.uint8)
N_COR = int(_caracteristicas_cor(_amostra).size)
N_FORMA = int(_caracteristicas_forma(_amostra).size)
N_TOTAL = N_COR + N_FORMA
GRUPOS = {
    "cor": slice(0, N_COR),
    "forma_textura": slice(N_COR, N_TOTAL),
    "todas": slice(0, N_TOTAL),
}


def processar_imagem(img: np.ndarray):
    """Retorna (vetor de características, recorte, sucesso_da_segmentação)."""
    recorte, _, ok = recortar_objeto(img)
    r = cv2.resize(recorte, (TAMANHO, TAMANHO), interpolation=cv2.INTER_AREA)
    vetor = np.concatenate([_caracteristicas_cor(r), _caracteristicas_forma(r)])
    return vetor, recorte, ok


def extrair_caracteristicas(img: np.ndarray) -> np.ndarray:
    return processar_imagem(img)[0]
