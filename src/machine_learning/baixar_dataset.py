"""Baixa e extrai o dataset TrashNet (versão redimensionada) para dados/trashnet.

O TrashNet foi publicado por G. Thung e M. Yang (Stanford, 2016) em
https://github.com/garythung/trashnet. Confira a licença e os termos de uso no
repositório original antes de redistribuir as imagens: este projeto NÃO inclui as imagens.

Uso:
    python -m src.machine_learning.baixar_dataset
Se o download automático falhar, baixe `dataset-resized.zip` manualmente (pasta data/ do
repositório), extraia e deixe as subpastas (glass, paper, cardboard, plastic, metal, trash)
dentro de dados/trashnet/.
"""
from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"


def baixar(destino: str = "dados/trashnet") -> Path:
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    zip_path = destino / "dataset-resized.zip"
    print(f"Baixando {URL} ...")
    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(destino)
    zip_path.unlink()
    # O zip costuma conter a pasta "dataset-resized/"; sobe as subpastas um nível.
    interna = destino / "dataset-resized"
    if interna.is_dir():
        for sub in interna.iterdir():
            alvo = destino / sub.name
            if not alvo.exists():
                shutil.move(str(sub), str(alvo))
        shutil.rmtree(interna, ignore_errors=True)
    print("Pastas encontradas:", sorted(p.name for p in destino.iterdir() if p.is_dir()))
    return destino


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--destino", default="dados/trashnet")
    baixar(ap.parse_args().destino)
