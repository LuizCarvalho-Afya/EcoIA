"""Baixa o modelo pré-treinado MobileNetV2 (ONNX, ImageNet) para modelos/mobilenetv2.onnx.

O arquivo vem do ONNX Model Zoo (https://github.com/onnx/models). O script tenta alguns
endereços em sequência e só aceita um arquivo que o OpenCV consiga carregar e executar.

Uso:
    python -m src.machine_learning.baixar_modelo

Se todos os endereços falharem, baixe manualmente o arquivo `mobilenetv2-12.onnx` (ou
`mobilenetv2-7.onnx`) na pasta vision/classification/mobilenet do repositório onnx/models,
renomeie para `mobilenetv2.onnx` e coloque na pasta `modelos/`. Confira a licença do modelo
(Apache 2.0 no repositório) antes de redistribuir.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

BASE = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model"
URLS = [
    f"{BASE}/mobilenetv2-12.onnx",
    f"{BASE}/mobilenetv2-10.onnx",
    f"{BASE}/mobilenetv2-7.onnx",
    "https://huggingface.co/onnxmodelzoo/mobilenetv2-12/resolve/main/mobilenetv2-12.onnx",
]
TAMANHO_MINIMO = 5_000_000  # bytes; um ponteiro Git LFS ou página de erro é bem menor


def _valido(caminho: Path) -> bool:
    """O arquivo precisa ser grande o bastante, carregar no OpenCV e gerar saída finita."""
    try:
        if caminho.stat().st_size < TAMANHO_MINIMO:
            return False
        rede = cv2.dnn.readNetFromONNX(str(caminho))
        rede.setInput(np.zeros((1, 3, 224, 224), dtype=np.float32))
        saida = rede.forward()
        return saida.size > 1 and bool(np.isfinite(saida).all())
    except Exception as e:  # noqa: BLE001 - qualquer falha significa "arquivo inutilizável"
        print(f"  não foi possível usar o arquivo: {e}")
        return False


def baixar(destino: str = "modelos/mobilenetv2.onnx") -> Path:
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and _valido(destino):
        print(f"Modelo já presente e válido: {destino}")
        return destino
    temporario = destino.with_suffix(".baixando")
    for url in URLS:
        print(f"Tentando {url} ...")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "EcoIA/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(temporario, "wb") as f:
                f.write(r.read())
        except Exception as e:  # noqa: BLE001
            print(f"  falhou: {e}")
            continue
        if _valido(temporario):
            temporario.replace(destino)
            print(f"Modelo salvo em {destino} ({destino.stat().st_size / 1e6:.1f} MB)")
            return destino
        print("  arquivo baixado não é um modelo válido; tentando o próximo endereço.")
    temporario.unlink(missing_ok=True)
    raise SystemExit("Não foi possível baixar o modelo. Veja as instruções de download manual no topo "
                     "de src/machine_learning/baixar_modelo.py.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Baixa o MobileNetV2 (ONNX) para o EcoIA.")
    ap.add_argument("--destino", default="modelos/mobilenetv2.onnx")
    baixar(ap.parse_args().destino)
