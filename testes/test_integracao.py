"""Teste de fluxo completo com dados SINTÉTICOS (verifica execução, não desempenho real)."""
from pathlib import Path

import numpy as np

from src.machine_learning import treino
from src.pipeline import EcoIA
from src.pln import treino_pln
from testes.conftest import gerar_imagem

RAIZ = Path(__file__).resolve().parents[1]


def test_fluxo_completo(dataset_sintetico, tmp_path):
    mv, mp = str(tmp_path / "visao.joblib"), str(tmp_path / "pln.joblib")
    m = treino.treinar(str(dataset_sintetico), str(tmp_path / "res"), mv, onnx=None)
    treino_pln.treinar(str(RAIZ / "dados" / "pln" / "perguntas.csv"), str(tmp_path / "res"), mp)
    assert (tmp_path / "res" / "graficos" / "matriz_confusao.png").exists()
    assert m["acuracia_teste"] > 0.6  # sanidade em dados sintéticos fáceis

    sistema = EcoIA(mv, mp)
    assert sistema.pronto == {"visao": True, "pln": True}
    img = gerar_imagem("glass", np.random.default_rng(7))

    r = sistema.processar(img, "Qual a cor da lixeira?")
    assert r["visual"] is not None and r["texto"] is not None
    assert r["decisao"].material in {"plastico", "papel", "vidro", "metal", "outros"}

    # Só texto, sem imagem: material vem da pergunta.
    r2 = sistema.processar(None, "Onde jogo uma lata de atum?")
    assert r2["decisao"].material == "metal" and r2["decisao"].fonte == "texto"

    # Pergunta fora do escopo.
    r3 = sistema.processar(img, "Qual a capital do Brasil?")
    assert r3["decisao"].acao in {"fora_escopo", "responder", "responder_com_aviso", "pedir_nova_imagem"}
