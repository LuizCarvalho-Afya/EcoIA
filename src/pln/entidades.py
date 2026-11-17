"""PLN: extração de entidades (material e objeto) por léxico.

Regras:
  * Material explícito (plástico, vidro, alumínio, ...) tem prioridade.
  * Se não houver material explícito, o objeto citado (lata, jornal, ...) pode indicar o
    material. Objetos ambíguos (garrafa, pote, copo, ...) não definem material.
  * Mais de um material explícito diferente => `material=None` e todos em `candidatos`.
  * A cor de lixeira citada (azul, vermelha, ...) é extraída em `lixeira`.
"""
from __future__ import annotations

from src.pln.preprocessamento import tokens_basicos

MATERIAIS = {
    "plastico": "plastico", "pet": "plastico", "pvc": "plastico", "pead": "plastico",
    "papel": "papel", "papelao": "papel",
    "vidro": "vidro",
    "metal": "metal", "aluminio": "metal", "ferro": "metal", "aco": "metal", "inox": "metal",
    "isopor": "outros", "organico": "outros", "pilha": "outros", "bateria": "outros", "fralda": "outros",
    "lampada": "outros", "casca": "outros", "comida": "outros",
}

OBJETOS_COM_MATERIAL = {
    "lata": "metal", "latinha": "metal",
    "jornal": "papel", "revista": "papel", "caderno": "papel", "caixa": "papel", "sulfite": "papel", "folha": "papel",
    "sacola": "plastico", "canudo": "plastico",
    "espelho": "vidro", "cacos": "vidro",
}

CORES_LIXEIRA = {"azul": "azul", "vermelha": "vermelha", "vermelho": "vermelha", "verde": "verde",
                 "amarela": "amarela", "amarelo": "amarela", "cinza": "cinza"}

OBJETOS_AMBIGUOS = {"garrafa", "pote", "copo", "frasco", "tampa", "embalagem", "bandeja", "recipiente"}


def singular(token: str) -> str:
    """Singularização simples para o léxico (garrafas->garrafa, jornais->jornal, papeis->papel)."""
    if token.endswith("ais") and len(token) > 4:
        return token[:-3] + "al"
    if token.endswith("eis") and len(token) > 4:
        return token[:-3] + "el"
    if token.endswith("oes") and len(token) > 4:
        return token[:-3] + "ao"
    if token.endswith("s") and len(token) > 3:
        return token[:-1]
    return token


def extrair_entidades(texto: str) -> dict:
    toks = [singular(t) for t in tokens_basicos(texto)]
    explicitos = []
    for t in toks:
        if t in MATERIAIS and MATERIAIS[t] not in explicitos:
            explicitos.append(MATERIAIS[t])
    objeto = next((t for t in toks if t in OBJETOS_COM_MATERIAL or t in OBJETOS_AMBIGUOS), None)

    material, candidatos = None, []
    if len(explicitos) == 1:
        material = explicitos[0]
    elif len(explicitos) > 1:
        candidatos = explicitos
    elif objeto in OBJETOS_COM_MATERIAL:
        material = OBJETOS_COM_MATERIAL[objeto]
    lixeira = next((CORES_LIXEIRA[t] for t in toks if t in CORES_LIXEIRA), None)
    return {"material": material, "objeto": objeto, "candidatos": candidatos, "lixeira": lixeira}
