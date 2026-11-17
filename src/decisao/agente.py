"""Agente de decisão baseado em regras (agente reativo com utilidade por confiança).

Percepções: PredicaoVisual (classe, confiança, margem) e AnaliseTexto (intenção, material citado).
Ações: responder, responder_com_aviso, pedir_nova_imagem, pedir_esclarecimento, fora_escopo.

Função real no sistema: é o ÚNICO componente que combina a saída da Visão/ML com a do PLN.
Ele decide em quem confiar quando as fontes divergem, quando a imagem é incerta demais e que
resposta montar. Cada regra acionada é registrada em `Decisao.regras`, para auditoria.

Regras (ordem de avaliação):
  R0  intenção incerta (confiança < limiar_intencao)         -> trata como "onde_descartar"
  R1  intenção fora do escopo                                -> recusa educada
  R2  sem imagem e sem material no texto                     -> pede esclarecimento
  R3  só texto                                               -> usa o material do texto
  R4  só imagem, confiança < limiar_minimo                   -> pede nova imagem
  R5  só imagem, margem top1-top2 < margem_ambigua           -> responde com aviso (duas hipóteses)
  R6  imagem + texto concordam                               -> responde (fonte imagem+texto)
  R7  imagem + texto divergem, confiança >= limiar_alto      -> segue a imagem e avisa o conflito
  R8  imagem + texto divergem, confiança < limiar_alto       -> segue o texto e avisa a incerteza da imagem
  R9  imagem incerta (< limiar_minimo) + material no texto   -> segue o texto
"""
from __future__ import annotations

from typing import Optional

from src.constantes import LIXEIRAS, NOMES_EXIBICAO
from src.decisao.conhecimento import MENSAGEM_FORA_ESCOPO, montar_mensagem
from src.tipos import AnaliseTexto, Decisao, PredicaoVisual


class AgenteDescarte:
    def __init__(self, limiar_minimo: float = 0.50, limiar_alto: float = 0.75,
                 margem_ambigua: float = 0.15, limiar_intencao: float = 0.40):
        self.limiar_minimo = limiar_minimo
        self.limiar_alto = limiar_alto
        self.margem_ambigua = margem_ambigua
        self.limiar_intencao = limiar_intencao

    def decidir(self, visual: Optional[PredicaoVisual], texto: Optional[AnaliseTexto]) -> Decisao:
        regras, avisos = [], []
        intencao = texto.intencao if texto else "onde_descartar"

        if texto and intencao != "fora_escopo" and texto.confianca_intencao < self.limiar_intencao:
            regras.append("R0: intenção incerta -> orientação completa de descarte")
            intencao = "onde_descartar"

        if intencao == "fora_escopo":
            regras.append("R1: pergunta fora do escopo")
            return Decisao("fora_escopo", None, None, "nenhuma", intencao, MENSAGEM_FORA_ESCOPO, regras=regras)

        mat_texto = texto.material if texto else None
        lixeira = texto.lixeira if texto else None

        if visual is None and mat_texto is None:
            regras.append("R2: sem imagem e sem material identificável no texto")
            msg = ("Não consegui identificar o resíduo. Envie uma foto do item ou cite o material "
                   "(plástico, papel, vidro ou metal) na pergunta.")
            return Decisao("pedir_esclarecimento", None, None, "nenhuma", intencao, msg, regras=regras)

        if visual is None:
            regras.append("R3: apenas texto -> material citado na pergunta")
            return self._responder("responder", mat_texto, None, "texto", intencao, avisos, regras, lixeira)

        # A partir daqui há imagem.
        if mat_texto is None:
            if visual.confianca < self.limiar_minimo:
                regras.append(f"R4: confiança visual {visual.confianca:.2f} < {self.limiar_minimo:.2f}")
                msg = ("Não tenho confiança suficiente para classificar esta imagem. Tente outra foto, com o "
                       "objeto centralizado, bem iluminado e fundo liso, ou cite o material na pergunta.")
                return Decisao("pedir_nova_imagem", None, visual.confianca, "imagem", intencao, msg, regras=regras)
            if visual.margem < self.margem_ambigua:
                regras.append(f"R5: margem {visual.margem:.2f} < {self.margem_ambigua:.2f} (duas hipóteses próximas)")
                top2 = sorted(visual.probabilidades.items(), key=lambda kv: kv[1], reverse=True)[:2]
                avisos.append("A imagem é ambígua entre "
                              f"{NOMES_EXIBICAO[top2[0][0]]} ({top2[0][1]:.0%}) e "
                              f"{NOMES_EXIBICAO[top2[1][0]]} ({top2[1][1]:.0%}). A orientação segue a primeira "
                              "hipótese; confirme o material.")
                return self._responder("responder_com_aviso", visual.classe, visual.confianca, "imagem",
                                       intencao, avisos, regras, lixeira)
            regras.append("Imagem confiável sem material citado -> segue a imagem")
            return self._responder("responder", visual.classe, visual.confianca, "imagem", intencao, avisos, regras, lixeira)

        # Imagem + material no texto.
        if visual.confianca < self.limiar_minimo:
            regras.append(f"R9: imagem incerta ({visual.confianca:.2f}) -> segue o texto")
            avisos.append("A imagem não ficou clara; a resposta usa o material informado na pergunta.")
            return self._responder("responder_com_aviso", mat_texto, visual.confianca, "texto", intencao, avisos, regras, lixeira)
        if visual.classe == mat_texto:
            regras.append("R6: imagem e texto concordam")
            return self._responder("responder", mat_texto, visual.confianca, "imagem+texto", intencao, avisos, regras, lixeira)
        if visual.confianca >= self.limiar_alto:
            regras.append(f"R7: divergência com imagem confiável ({visual.confianca:.2f}) -> segue a imagem")
            avisos.append(f"Você citou {NOMES_EXIBICAO[mat_texto]}, mas a imagem indica "
                          f"{NOMES_EXIBICAO[visual.classe]} com {visual.confianca:.0%} de confiança. "
                          "A orientação segue a imagem; confira o item.")
            return self._responder("responder_com_aviso", visual.classe, visual.confianca, "imagem", intencao, avisos, regras, lixeira)
        regras.append(f"R8: divergência com imagem pouco confiável ({visual.confianca:.2f}) -> segue o texto")
        avisos.append(f"A imagem sugere {NOMES_EXIBICAO[visual.classe]} ({visual.confianca:.0%}), mas a "
                      f"confiança é moderada; a orientação segue o material citado ({NOMES_EXIBICAO[mat_texto]}).")
        return self._responder("responder_com_aviso", mat_texto, visual.confianca, "texto", intencao, avisos, regras, lixeira)

    @staticmethod
    def _responder(acao, material, confianca, fonte, intencao, avisos, regras, lixeira) -> Decisao:
        msg = montar_mensagem(material, intencao, lixeira)
        return Decisao(acao, material, confianca, fonte, intencao, msg, cor_lixeira=LIXEIRAS[material],
                       avisos=list(avisos), regras=list(regras))
