"""Gráficos e experimentos de avaliação do modelo de visão."""
from __future__ import annotations

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from src.visao_computacional.extracao import carregar_imagem


def matriz_confusao(y_true, y_pred, classes, caminho):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de confusão (conjunto de teste)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def barras(rotulos, valores, titulo, ylabel, caminho, ylim=(0, 1), figsize=(6, 3.8)):
    fig, ax = plt.subplots(figsize=figsize)
    barras_ = ax.bar(rotulos, valores, color="#3b7a57")
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo)
    for b, v in zip(barras_, valores):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def exemplos_predicoes(caminhos, y_true, y_pred, confiancas, caminho, n=8, apenas_erros=False):
    idx = np.arange(len(caminhos))
    if apenas_erros:
        idx = idx[np.array(y_true) != np.array(y_pred)]
    idx = idx[:n]
    if len(idx) == 0:
        return False
    cols = min(4, len(idx))
    linhas = int(np.ceil(len(idx) / cols))
    fig, eixos = plt.subplots(linhas, cols, figsize=(3 * cols, 3.2 * linhas), squeeze=False)
    for ax in eixos.flat:
        ax.axis("off")
    for ax, i in zip(eixos.flat, idx):
        img = cv2.cvtColor(carregar_imagem(caminhos[i]), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        cor = "green" if y_true[i] == y_pred[i] else "red"
        ax.set_title(f"real: {y_true[i]}\nprev: {y_pred[i]} ({confiancas[i]:.2f})", fontsize=8, color=cor)
    fig.tight_layout()
    fig.savefig(caminho, dpi=150)
    plt.close(fig)
    return True


# Perturbações usadas no teste de robustez.
def _mais_claro(img):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=50)


def _mais_escuro(img):
    return cv2.convertScaleAbs(img, alpha=1.0, beta=-50)


def _ruido(img):
    rng = np.random.default_rng(42)
    ruido = rng.normal(0, 15, img.shape)
    return np.clip(img.astype(np.float32) + ruido, 0, 255).astype(np.uint8)


def _desfoque(img):
    return cv2.GaussianBlur(img, (9, 9), 0)


PERTURBACOES = {
    "original": None,
    "mais_claro": _mais_claro,
    "mais_escuro": _mais_escuro,
    "ruido_gaussiano": _ruido,
    "desfoque": _desfoque,
}
