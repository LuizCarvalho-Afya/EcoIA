import numpy as np

from src.visao_computacional.extracao import (GRUPOS, N_TOTAL, extrair_caracteristicas, processar_imagem,
                                              recortar_objeto)
from testes.conftest import gerar_imagem


def test_vetor_tem_tamanho_fixo_e_e_finito():
    img = gerar_imagem("plastic", np.random.default_rng(1))
    v = extrair_caracteristicas(img)
    assert v.shape == (N_TOTAL,) and np.isfinite(v).all()
    assert GRUPOS["cor"].stop == GRUPOS["forma_textura"].start


def test_recorte_isola_objeto_central():
    img = gerar_imagem("glass", np.random.default_rng(2))
    recorte, mascara, ok = recortar_objeto(img)
    assert ok and recorte.shape[0] < img.shape[0] and recorte.shape[1] < img.shape[1]


def test_imagem_uniforme_nao_quebra():
    img = np.full((100, 100, 3), 128, np.uint8)
    vetor, recorte, ok = processar_imagem(img)
    assert vetor.shape == (N_TOTAL,) and not ok
