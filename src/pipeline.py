"""Fluxo integrado do EcoIA: imagem -> visão/ML -> pergunta -> PLN -> decisão -> resposta."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np

from src.decisao.agente import AgenteDescarte
from src.machine_learning.conjuntos import CLASSICO, montar
from src.pln.intencoes import ClassificadorIntencoes
from src.tipos import AnaliseTexto, Decisao, PredicaoVisual
from src.visao_computacional.extracao import processar_imagem
from src.visao_computacional.profundo import ExtratorProfundo


class AnalisadorVisual:
    def __init__(self, caminho_modelo: str):
        dados = joblib.load(caminho_modelo)
        self.modelo = dados["modelo"]
        self.classes = list(self.modelo.classes_)
        # Modelos salvos antes da transferência de aprendizado não trazem a chave "conjunto".
        self.conjunto = dados.get("conjunto", CLASSICO)
        self.extrator = None
        if self.conjunto != CLASSICO:
            self.extrator = ExtratorProfundo(dados["onnx"])  # erro claro se o arquivo sumiu

    def analisar(self, img_bgr: np.ndarray):
        """Retorna (PredicaoVisual, recorte do objeto)."""
        vetor_classico, recorte, ok = processar_imagem(img_bgr)
        vetor_profundo = self.extrator.extrair(recorte) if self.extrator is not None else None
        vetor = montar(self.conjunto, vetor_classico, vetor_profundo)
        proba = self.modelo.predict_proba(vetor.reshape(1, -1))[0]
        ordem = np.argsort(proba)[::-1]
        margem = float(proba[ordem[0]] - proba[ordem[1]]) if len(proba) > 1 else 1.0
        pred = PredicaoVisual(classe=str(self.classes[ordem[0]]), confianca=float(proba[ordem[0]]),
                              probabilidades={str(c): float(p) for c, p in zip(self.classes, proba)},
                              margem=margem, recorte_ok=ok)
        return pred, recorte


class EcoIA:
    """Orquestra os módulos. Qualquer componente de aprendizado pode faltar (modo degradado)."""

    def __init__(self, modelo_visao: str = "modelos/modelo_visao.joblib",
                 modelo_pln: str = "modelos/modelo_pln.joblib", agente: Optional[AgenteDescarte] = None):
        self.visual = AnalisadorVisual(modelo_visao) if Path(modelo_visao).exists() else None
        self.pln = ClassificadorIntencoes.carregar(modelo_pln) if Path(modelo_pln).exists() else None
        self.agente = agente or AgenteDescarte()

    @property
    def pronto(self) -> dict:
        return {"visao": self.visual is not None, "pln": self.pln is not None}

    def processar(self, img_bgr: Optional[np.ndarray] = None, pergunta: Optional[str] = None):
        """Retorna dict com predição visual, análise do texto, decisão e recorte (se houver)."""
        pred, recorte, analise = None, None, None
        if img_bgr is not None:
            if self.visual is None:
                raise RuntimeError("Modelo de visão não encontrado. Rode o treino (ver README).")
            pred, recorte = self.visual.analisar(img_bgr)
        if pergunta and pergunta.strip():
            if self.pln is None:
                raise RuntimeError("Modelo de PLN não encontrado. Rode o treino (ver README).")
            analise = self.pln.analisar(pergunta.strip())
        decisao = self.agente.decidir(pred, analise)
        return {"visual": pred, "texto": analise, "decisao": decisao, "recorte": recorte}
