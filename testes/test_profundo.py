"""Testes do caminho de transferência de aprendizado, com um ONNX de brinquedo (3 saídas)."""
from pathlib import Path

import numpy as np
import pytest

from src.machine_learning import baixar_modelo, treino
from src.machine_learning.conjuntos import CLASSICO, COMBINADO, PROFUNDO, montar
from src.pipeline import EcoIA
from src.pln import treino_pln
from src.visao_computacional.profundo import ExtratorProfundo, quadrado_com_margem
from testes.conftest import gerar_imagem

RAIZ = Path(__file__).resolve().parents[1]


def test_quadrado_com_margem_fica_quadrado():
    img = np.full((100, 40, 3), 200, np.uint8)
    q = quadrado_com_margem(img)
    assert q.shape == (100, 100, 3) and (q[:, 0] == 200).all()


def test_extrator_gera_vetor_finito(onnx_minimo):
    ext = ExtratorProfundo(onnx_minimo)
    v = ext.extrair(gerar_imagem("plastic", np.random.default_rng(1)))
    assert v.shape == (3,) and np.isfinite(v).all()


def test_extrator_sem_arquivo_da_erro_claro(tmp_path):
    with pytest.raises(FileNotFoundError, match="baixar_modelo"):
        ExtratorProfundo(str(tmp_path / "nao_existe.onnx"))


def test_montar_conjuntos():
    c, p = np.ones((4, 5)), np.zeros((4, 3))
    assert montar(CLASSICO, c, None).shape == (4, 5)
    assert montar(PROFUNDO, c, p).shape == (4, 3)
    assert montar(COMBINADO, c, p).shape == (4, 8)
    with pytest.raises(ValueError):
        montar(PROFUNDO, c, None)


def test_downloader_rejeita_arquivo_pequeno(tmp_path):
    falso = tmp_path / "lfs_pointer.onnx"
    falso.write_text("version https://git-lfs.github.com/spec/v1\n")
    assert baixar_modelo._valido(falso) is False


def test_treino_e_pipeline_com_rede(dataset_sintetico, onnx_minimo, tmp_path):
    mv, mp = str(tmp_path / "visao.joblib"), str(tmp_path / "pln.joblib")
    m = treino.treinar(str(dataset_sintetico), str(tmp_path / "res"), mv, onnx=onnx_minimo)
    assert set(m["por_conjunto_teste"]) == {CLASSICO, PROFUNDO, COMBINADO}
    assert m["conjunto_caracteristicas"] in {CLASSICO, PROFUNDO, COMBINADO}
    assert (tmp_path / "res" / "graficos" / "comparacao_caracteristicas.png").exists()
    treino_pln.treinar(str(RAIZ / "dados" / "pln" / "perguntas.csv"), str(tmp_path / "res"), mp)

    sistema = EcoIA(mv, mp)
    r = sistema.processar(gerar_imagem("glass", np.random.default_rng(3)), "Onde jogo isso?")
    assert r["visual"].classe in {"plastico", "papel", "vidro", "metal", "outros"}


def test_pipeline_com_modelo_antigo_sem_chave_conjunto(dataset_sintetico, tmp_path):
    """Modelos treinados antes da transferência de aprendizado continuam carregando."""
    mv = str(tmp_path / "visao.joblib")
    treino.treinar(str(dataset_sintetico), str(tmp_path / "res"), mv, onnx=None)
    import joblib
    d = joblib.load(mv)
    d.pop("conjunto"), d.pop("onnx")
    joblib.dump(d, mv)
    sistema = EcoIA(mv, str(tmp_path / "nao_existe_pln.joblib"))
    assert sistema.visual.conjunto == CLASSICO
