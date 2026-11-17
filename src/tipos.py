"""Estruturas de dados trocadas entre visão, PLN e decisão."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class PredicaoVisual:
    """Saída do módulo de Visão Computacional + Machine Learning."""

    classe: str
    confianca: float
    probabilidades: dict
    margem: float  # diferença entre a 1ª e a 2ª maior probabilidade
    recorte_ok: bool = True  # False = segmentação falhou e a imagem inteira foi usada


@dataclass
class AnaliseTexto:
    """Saída do módulo de PLN."""

    texto: str
    intencao: str
    confianca_intencao: float
    material: Optional[str] = None  # material citado explicitamente ou inferido do objeto
    objeto: Optional[str] = None
    candidatos: list = field(default_factory=list)  # vários materiais citados
    lixeira: Optional[str] = None  # cor de lixeira citada (azul, vermelha, verde, amarela, cinza)
    tokens: list = field(default_factory=list)


@dataclass
class Decisao:
    """Saída do agente de decisão."""

    acao: str  # responder | responder_com_aviso | pedir_nova_imagem | pedir_esclarecimento | fora_escopo
    material: Optional[str]
    confianca: Optional[float]
    fonte: str  # imagem | texto | imagem+texto | nenhuma
    intencao: str
    mensagem: str
    cor_lixeira: Optional[str] = None
    avisos: list = field(default_factory=list)
    regras: list = field(default_factory=list)  # regras acionadas (rastreabilidade)

    def para_dict(self) -> dict:
        return asdict(self)
