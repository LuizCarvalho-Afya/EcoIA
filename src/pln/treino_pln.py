"""Treino e avaliação do classificador de intenções.

Uso:
    python -m src.pln.treino_pln --csv dados/pln/perguntas.csv --saida resultados

Protocolo: divisão estratificada 80/20 (semente 42); comparação de três modelos com
validação cruzada 5-fold no treino; avaliação única do modelo escolhido no teste; baseline
de regras por palavras-chave e baseline da classe majoritária para referência.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.constantes import INTENCOES, SEMENTE
from src.machine_learning import avaliacao
from src.pln.preprocessamento import BACKEND, normalizar, tokens_basicos


def _vetorizador():
    return TfidfVectorizer(tokenizer=normalizar, lowercase=False, token_pattern=None,
                           ngram_range=(1, 2), sublinear_tf=True)


def candidatos() -> dict:
    return {
        "TF-IDF + Regressão Logística": Pipeline([("tfidf", _vetorizador()),
                                                  ("clf", LogisticRegression(C=10, max_iter=2000, random_state=SEMENTE))]),
        "TF-IDF + LinearSVC": Pipeline([("tfidf", _vetorizador()), ("clf", LinearSVC(random_state=SEMENTE))]),
        "TF-IDF + Naive Bayes": Pipeline([("tfidf", _vetorizador()), ("clf", MultinomialNB(alpha=0.5))]),
    }


# Baseline de regras: palavras-chave simples, sem aprendizado.
_REGRAS = [
    ("cor_lixeira", {"cor", "lixeira", "coletor", "azul", "verde", "amarela", "vermelha", "vermelho", "amarelo"}),
    ("preparo", {"lavar", "limpar", "enxaguar", "amassar", "esvaziar", "tampa", "rotulo", "etiqueta", "preparar",
                 "preparo", "seco", "suja", "sujo", "desmontar", "quebrar", "separo"}),
    ("reciclavel", {"reciclavel", "reciclar", "reciclado", "recicla", "reciclagem"}),
    ("identificar_material", {"material", "identifique", "identifica", "classifique", "categoria", "classe",
                              "reconheca", "detectou", "identificou", "classificou", "foto", "imagem"}),
    ("onde_descartar", {"onde", "descartar", "descarto", "descarte", "jogar", "jogo", "destino", "aonde"}),
]


_CORES = {"azul", "verde", "amarela", "amarelo", "vermelha", "vermelho", "cinza"}
_VERBOS_CONFIRMACAO = {"posso", "pode", "certo", "correto", "errado", "devo", "mesmo", "confirma"}


def prever_por_regras(texto: str) -> str:
    toks = set(tokens_basicos(texto))
    if toks & _CORES and toks & _VERBOS_CONFIRMACAO:
        return "confirmar_descarte"
    for intencao, palavras in _REGRAS:
        if toks & palavras:
            return intencao
    return "fora_escopo"


def treinar(csv: str, saida: str, modelo_saida: str = "modelos/modelo_pln.joblib") -> dict:
    saida = Path(saida)
    (saida / "graficos").mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv)
    # Arrays NumPy comuns: com pandas 3 + pyarrow, .values devolve um array do Arrow que o
    # train_test_split do scikit-learn não consegue indexar.
    X = df["texto"].astype(str).to_numpy(dtype=object)
    y = df["intencao"].to_numpy(dtype=object)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMENTE)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    linhas = []
    for nome, modelo in candidatos().items():
        cvs = cross_val_score(clone(modelo), X_tr, y_tr, cv=cv, scoring="f1_macro")
        linhas.append({"modelo": nome, "f1_macro_cv_media": cvs.mean(), "f1_macro_cv_desvio": cvs.std()})
    comp = pd.DataFrame(linhas).sort_values("f1_macro_cv_media", ascending=False)
    comp.to_csv(saida / "comparacao_modelos_pln.csv", index=False)
    melhor = comp.iloc[0]["modelo"]

    # O classificador usado na aplicação precisa de probabilidades (confiança da intenção);
    # LinearSVC e Naive Bayes seguem apenas como comparação.
    nome_final = "TF-IDF + Regressão Logística"
    final = clone(candidatos()[nome_final]).fit(X_tr, y_tr)
    pred = final.predict(X_te)

    # Baselines
    pred_regras = np.array([prever_por_regras(t) for t in X_te])
    dummy = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
    pred_dummy = dummy.predict(X_te)

    metricas = {
        "backend_radicalizacao": BACKEND,
        "n_exemplos": int(len(df)),
        "n_teste": int(len(X_te)),
        "distribuicao_intencoes": df["intencao"].value_counts().to_dict(),
        "melhor_modelo_cv": melhor,
        "modelo_usado_na_aplicacao": nome_final,
        "acuracia_teste": float(accuracy_score(y_te, pred)),
        "f1_macro_teste": float(f1_score(y_te, pred, average="macro")),
        "baseline_regras": {"acuracia": float(accuracy_score(y_te, pred_regras)),
                            "f1_macro": float(f1_score(y_te, pred_regras, average="macro"))},
        "baseline_majoritaria": {"acuracia": float(accuracy_score(y_te, pred_dummy)),
                                 "f1_macro": float(f1_score(y_te, pred_dummy, average="macro"))},
        "relatorio_por_classe": classification_report(y_te, pred, labels=INTENCOES, output_dict=True, zero_division=0),
        "semente": SEMENTE,
    }
    pd.DataFrame({"texto": X_te, "real": y_te, "previsto": pred})[pred != y_te] \
        .to_csv(saida / "erros_pln.csv", index=False)
    avaliacao.matriz_confusao(y_te, pred, INTENCOES, saida / "graficos" / "matriz_confusao_pln.png")
    avaliacao.barras(["majoritária", "regras", "TF-IDF+LogReg"],
                     [metricas["baseline_majoritaria"]["f1_macro"], metricas["baseline_regras"]["f1_macro"],
                      metricas["f1_macro_teste"]],
                     "PLN: F1 macro no teste", "F1 macro", saida / "graficos" / "comparacao_pln.png")
    (saida / "metricas_pln.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False))

    # Para uso na aplicação, o modelo final é retreinado com todos os exemplos.
    completo = clone(candidatos()[nome_final]).fit(X, y)
    Path(modelo_saida).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": completo, "backend": BACKEND}, modelo_saida)
    print(f"[{BACKEND}] acurácia no teste = {metricas['acuracia_teste']:.3f} | F1 macro = {metricas['f1_macro_teste']:.3f}"
          f" | regras: {metricas['baseline_regras']['f1_macro']:.3f}")
    return metricas


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Treina o classificador de intenções do EcoIA.")
    ap.add_argument("--csv", default="dados/pln/perguntas.csv")
    ap.add_argument("--saida", default="resultados")
    ap.add_argument("--modelo", default="modelos/modelo_pln.joblib")
    a = ap.parse_args()
    treinar(a.csv, a.saida, a.modelo)
