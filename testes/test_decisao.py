from src.decisao.agente import AgenteDescarte
from src.tipos import AnaliseTexto, PredicaoVisual


def visual(classe, conf, segunda="outros", conf2=None):
    conf2 = conf2 if conf2 is not None else (1 - conf) / 2
    return PredicaoVisual(classe, conf, {classe: conf, segunda: conf2}, conf - conf2)


def texto(intencao="onde_descartar", material=None, conf=0.9, lixeira=None):
    return AnaliseTexto("x", intencao, conf, material=material, lixeira=lixeira)


ag = AgenteDescarte()


def test_fora_do_escopo():
    assert ag.decidir(None, texto("fora_escopo")).acao == "fora_escopo"


def test_sem_nada_pede_esclarecimento():
    assert ag.decidir(None, texto()).acao == "pedir_esclarecimento"


def test_somente_texto_usa_material_citado():
    d = ag.decidir(None, texto(material="vidro"))
    assert d.material == "vidro" and d.cor_lixeira == "verde" and d.fonte == "texto"


def test_imagem_pouco_confiavel_pede_nova_imagem():
    assert ag.decidir(visual("plastico", 0.40), None).acao == "pedir_nova_imagem"


def test_imagem_ambigua_responde_com_aviso():
    d = ag.decidir(visual("vidro", 0.52, "plastico", 0.45), None)
    assert d.acao == "responder_com_aviso" and d.avisos


def test_imagem_confiavel_sem_texto():
    d = ag.decidir(visual("metal", 0.9), None)
    assert d.acao == "responder" and d.cor_lixeira == "amarela" and d.fonte == "imagem"


def test_concordancia_imagem_texto():
    d = ag.decidir(visual("papel", 0.9), texto(material="papel"))
    assert d.fonte == "imagem+texto" and d.cor_lixeira == "azul"


def test_conflito_com_imagem_confiavel_segue_imagem():
    d = ag.decidir(visual("plastico", 0.92), texto(material="papel"))
    assert d.material == "plastico" and d.avisos and any("R7" in r for r in d.regras)


def test_conflito_com_imagem_moderada_segue_texto():
    d = ag.decidir(visual("plastico", 0.60), texto(material="papel"))
    assert d.material == "papel" and any("R8" in r for r in d.regras)


def test_imagem_incerta_com_texto_segue_texto():
    d = ag.decidir(visual("plastico", 0.35), texto(material="vidro"))
    assert d.material == "vidro" and any("R9" in r for r in d.regras)


def test_intencao_incerta_vira_orientacao_completa():
    d = ag.decidir(visual("vidro", 0.9), texto("cor_lixeira", conf=0.2))
    assert d.intencao == "onde_descartar" and any("R0" in r for r in d.regras)


def test_mensagem_de_cor():
    d = ag.decidir(visual("plastico", 0.9), texto("cor_lixeira"))
    assert "vermelha" in d.mensagem


def test_confirmacao_correta_diz_sim():
    d = ag.decidir(None, texto("confirmar_descarte", material="papel", lixeira="azul"))
    assert d.mensagem.startswith("Sim") and d.cor_lixeira == "azul"


def test_confirmacao_errada_diz_nao_e_indica_a_certa():
    d = ag.decidir(None, texto("confirmar_descarte", material="papel", lixeira="vermelha"))
    assert d.mensagem.startswith("Não") and "azul" in d.mensagem


def test_confirmacao_com_conflito_segue_a_imagem_e_corrige():
    """Caso da demonstração: pergunta cita papel + lixeira azul, imagem mostra plástico."""
    d = ag.decidir(visual("plastico", 0.75), texto("confirmar_descarte", material="papel", lixeira="azul"))
    assert d.mensagem.startswith("Não") and "vermelha" in d.mensagem and d.avisos
    assert any("R7" in r for r in d.regras)


def test_confirmacao_sem_cor_citada_da_orientacao_completa():
    d = ag.decidir(visual("vidro", 0.9), texto("confirmar_descarte"))
    assert "verde" in d.mensagem and not d.mensagem.startswith(("Sim", "Não"))
