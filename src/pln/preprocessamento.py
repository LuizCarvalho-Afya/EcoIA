"""PLN: normalização de texto em português (minúsculas, sem acentos, stopwords, radicais).

O radicalizador usa o RSLP do NLTK quando disponível (requer `nltk.download('rslp')`).
Caso contrário, usa um removedor de sufixos simples e determinístico. A variável de
ambiente ECOIA_STEMMER=simples|rslp|auto (padrão: auto) permite fixar a escolha; o
backend usado é registrado nas métricas para garantir reprodutibilidade.
"""
from __future__ import annotations

import os
import re
import unicodedata

STOPWORDS = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das", "em", "no", "na", "nos",
    "nas", "para", "pra", "por", "com", "e", "ou", "que", "se", "ao", "aos", "eh", "foi", "ser", "meu", "minha",
    "esse", "essa", "esses", "essas", "isso", "este", "esta", "isto", "me", "mim", "eu", "voce", "la", "ai",
}

_SUFIXOS = ("coes", "cao", "mente", "ando", "endo", "indo", "ados", "idos", "ado", "ido", "ada", "ida",
            "ar", "er", "ir", "es", "s")


def remover_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def tokens_basicos(texto: str) -> list:
    """Minúsculas, sem acentos, apenas letras/dígitos. Sem stopwords nem radicalização."""
    texto = remover_acentos(texto.lower())
    return re.findall(r"[a-z0-9]+", texto)


def _stem_simples(palavra: str) -> str:
    for suf in _SUFIXOS:
        if palavra.endswith(suf) and len(palavra) - len(suf) >= 4:
            return palavra[: -len(suf)]
    return palavra


def _carregar_stemmer():
    modo = os.environ.get("ECOIA_STEMMER", "auto").lower()
    if modo in ("auto", "rslp"):
        try:
            from nltk.stem import RSLPStemmer

            s = RSLPStemmer()
            s.stem("teste")  # falha se o recurso 'rslp' não estiver baixado
            return s.stem, "nltk-rslp"
        except Exception:
            if modo == "rslp":
                raise
    return _stem_simples, "stem-simples"


_STEM, BACKEND = _carregar_stemmer()


def normalizar(texto: str) -> list:
    """Pipeline completo: tokens sem acento, sem stopwords, radicalizados."""
    return [_STEM(t) for t in tokens_basicos(texto) if t not in STOPWORDS]
