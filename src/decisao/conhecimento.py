"""Base de conhecimento de descarte e geração das mensagens de resposta.

Cores: Resolução CONAMA nº 275/2001 (confira o texto oficial antes de citar no artigo).
As orientações de preparo são práticas gerais de coleta seletiva e podem variar por município.
"""
from __future__ import annotations

from src.constantes import LIXEIRAS, NOMES_EXIBICAO

BASE = {
    "plastico": {
        "reciclavel": True,
        "descarte": "Encaminhe para a coleta seletiva, na lixeira vermelha.",
        "preparo": "Esvazie e enxágue rapidamente para tirar restos de comida ou líquido. "
                   "Amassar a embalagem reduz o volume; a tampa pode ir junto.",
        "obs": "Sacolas e filmes plásticos limpos e secos costumam ser aceitos, mas confirme com a coleta local.",
    },
    "papel": {
        "reciclavel": True,
        "descarte": "Encaminhe para a coleta seletiva, na lixeira azul.",
        "preparo": "Mantenha o papel limpo e seco. Desmonte caixas de papelão para economizar espaço.",
        "obs": "Papel engordurado (como caixa de pizza suja), papel higiênico, guardanapo usado e papel "
               "plastificado ou metalizado normalmente não são recicláveis.",
    },
    "vidro": {
        "reciclavel": True,
        "descarte": "Encaminhe para a coleta seletiva, na lixeira verde.",
        "preparo": "Esvazie e enxágue. Retire tampas e rótulos soltos. Não é necessário quebrar o vidro.",
        "obs": "Vidro quebrado deve ser embalado com cuidado (jornal ou caixa) e sinalizado. Espelhos, "
               "cerâmica, lâmpadas e vidros de janela têm composição diferente e normalmente não entram "
               "na reciclagem de embalagens de vidro.",
    },
    "metal": {
        "reciclavel": True,
        "descarte": "Encaminhe para a coleta seletiva, na lixeira amarela.",
        "preparo": "Esvazie e enxágue. Amassar latas reduz o volume; tome cuidado com bordas cortantes.",
        "obs": "Embalagens de aerossol devem estar completamente vazias. Pilhas e baterias NÃO são metal "
               "reciclável comum: use pontos de coleta específicos.",
    },
    "outros": {
        "reciclavel": False,
        "descarte": "Este item não foi reconhecido como plástico, papel, vidro ou metal. Em geral, resíduo "
                    "não reciclável vai para a lixeira cinza; orgânicos (marrom) e itens como pilhas, "
                    "baterias e lâmpadas têm destinação própria. Consulte o serviço de limpeza urbana local.",
        "preparo": "Não há preparo específico. Evite misturar com materiais recicláveis limpos.",
        "obs": "Se achar que o item é reciclável, envie outra foto ou informe o material na pergunta.",
    },
}

MENSAGEM_FORA_ESCOPO = ("Posso ajudar com dúvidas sobre descarte de resíduos: onde jogar, cor da lixeira, "
                        "como preparar o item e se é reciclável. Envie uma foto ou faça uma pergunta sobre isso.")


def montar_mensagem(material: str, intencao: str, lixeira_citada: str | None = None) -> str:
    info = BASE[material]
    nome = NOMES_EXIBICAO[material]
    cor = LIXEIRAS[material]
    if intencao == "confirmar_descarte" and lixeira_citada:
        if lixeira_citada == cor:
            return f"Sim. {nome} vai na lixeira {cor}. {info['preparo']}"
        return (f"Não. {nome} vai na lixeira {cor}, e não na {lixeira_citada}. {info['preparo']}")
    if intencao == "cor_lixeira":
        return f"Material: {nome}. A lixeira recomendada é a {cor}."
    if intencao == "preparo":
        return f"Material: {nome}. {info['preparo']}"
    if intencao == "reciclavel":
        sim = "Sim, é reciclável." if info["reciclavel"] else "Em geral, não é reciclável."
        return f"Material: {nome}. {sim} {info['obs']}"
    if intencao == "identificar_material":
        return f"O material identificado é {nome} (lixeira {cor})."
    # onde_descartar (padrão): orientação completa
    return f"Material: {nome}. Lixeira {cor}. {info['descarte']} {info['preparo']}"
