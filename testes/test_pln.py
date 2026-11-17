import pandas as pd
from pathlib import Path

from src.pln.entidades import extrair_entidades
from src.pln.preprocessamento import normalizar, remover_acentos, tokens_basicos
from src.pln import treino_pln

RAIZ = Path(__file__).resolve().parents[1]


def test_remover_acentos_e_tokens():
    assert remover_acentos("plástico") == "plastico"
    assert tokens_basicos("Onde jogo o PAPELÃO?") == ["onde", "jogo", "o", "papelao"]


def test_stopwords_removidas():
    assert "o" not in normalizar("o papel") and "de" not in normalizar("lata de atum")


def test_entidade_material_explicito():
    assert extrair_entidades("Onde jogo uma garrafa de vidro?")["material"] == "vidro"


def test_entidade_objeto_com_material():
    e = extrair_entidades("Posso amassar a lata?")
    assert e["material"] == "metal" and e["objeto"] == "lata"


def test_objeto_ambiguo_nao_define_material():
    e = extrair_entidades("Onde devo jogar uma garrafa de refrigerante?")
    assert e["material"] is None and e["objeto"] == "garrafa"


def test_varios_materiais_viram_candidatos():
    e = extrair_entidades("Isso é plástico ou vidro?")
    assert e["material"] is None and set(e["candidatos"]) == {"plastico", "vidro"}


def test_singularizacao_plural():
    assert extrair_entidades("jornais e revistas")["material"] == "papel"


def test_dataset_pln_balanceado_e_sem_duplicatas():
    df = pd.read_csv(RAIZ / "dados" / "pln" / "perguntas.csv")
    assert df.texto.duplicated().sum() == 0
    assert df.intencao.nunique() == 7 and df.intencao.value_counts().min() >= 25


def test_treino_pln_supera_baseline_majoritaria(tmp_path):
    m = treino_pln.treinar(str(RAIZ / "dados" / "pln" / "perguntas.csv"), str(tmp_path), str(tmp_path / "m.joblib"))
    assert m["f1_macro_teste"] > m["baseline_majoritaria"]["f1_macro"] + 0.3
    assert (tmp_path / "metricas_pln.json").exists()


def test_entidade_cor_da_lixeira():
    assert extrair_entidades("Posso jogar papel na lixeira azul?")["lixeira"] == "azul"
    assert extrair_entidades("Lixeira vermelho serve?")["lixeira"] == "vermelha"
    assert extrair_entidades("Onde jogo isso?")["lixeira"] is None
