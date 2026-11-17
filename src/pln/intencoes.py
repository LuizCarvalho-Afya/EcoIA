"""PLN: classificador de intenções (TF-IDF + Regressão Logística) e análise de perguntas."""
from __future__ import annotations

import joblib

from src.pln.entidades import extrair_entidades
from src.pln.preprocessamento import normalizar
from src.tipos import AnaliseTexto


class ClassificadorIntencoes:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    @classmethod
    def carregar(cls, caminho: str) -> "ClassificadorIntencoes":
        return cls(joblib.load(caminho)["pipeline"])

    def prever(self, texto: str):
        proba = self.pipeline.predict_proba([texto])[0]
        i = int(proba.argmax())
        return self.pipeline.classes_[i], float(proba[i])

    def analisar(self, texto: str) -> AnaliseTexto:
        intencao, conf = self.prever(texto)
        ent = extrair_entidades(texto)
        return AnaliseTexto(texto=texto, intencao=str(intencao), confianca_intencao=conf,
                            material=ent["material"], objeto=ent["objeto"],
                            candidatos=ent["candidatos"], lixeira=ent["lixeira"], tokens=normalizar(texto))
