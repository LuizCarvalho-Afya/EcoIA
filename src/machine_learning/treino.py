"""Treino e avaliação do classificador de resíduos (aprendizado supervisionado).

Uso:
    python -m src.machine_learning.baixar_modelo          # uma vez: rede pré-treinada (opcional)
    python -m src.machine_learning.treino --dados dados/trashnet --saida resultados

Protocolo experimental (todas as divisões usam SEMENTE=42):
  1. Divisão estratificada 70% treino / 15% validação / 15% teste.
  2. Três conjuntos de características são comparados: clássico (cor + forma/textura),
     profundo (rede MobileNetV2 pré-treinada, via OpenCV DNN) e combinado. Se o arquivo da
     rede não existir, apenas o conjunto clássico é avaliado.
  3. Para cada conjunto, quatro modelos são treinados no treino e comparados na validação
     (F1 macro) e por validação cruzada 5-fold no treino.
  4. O melhor par (conjunto, modelo) pela validação é retreinado em treino+validação e
     avaliado UMA vez no teste. O melhor modelo de cada conjunto também é avaliado no teste,
     apenas para relatório (a escolha nunca usa o teste).
  5. Ablação das características clássicas e teste de robustez (brilho, ruído, desfoque).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.constantes import CLASSES, SEMENTE
from src.machine_learning import avaliacao
from src.machine_learning.conjuntos import CLASSICO, COMBINADO, PROFUNDO, montar
from src.machine_learning.dados import extrair_matrizes, listar_imagens
from src.visao_computacional.extracao import GRUPOS
from src.visao_computacional.profundo import CAMINHO_PADRAO, ExtratorProfundo


def candidatos() -> dict:
    return {
        # Probabilidades via calibração sigmoide (probability=True foi descontinuado no scikit-learn 1.9).
        "SVM (RBF)": Pipeline([("esc", StandardScaler()),
                               ("clf", CalibratedClassifierCV(
                                   SVC(C=10, gamma="scale", class_weight="balanced", random_state=SEMENTE),
                                   method="sigmoid", cv=3, ensemble=False))]),
        "Random Forest": Pipeline([("esc", StandardScaler()),
                                   ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                                                  random_state=SEMENTE, n_jobs=-1))]),
        "k-NN (k=5)": Pipeline([("esc", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=5))]),
        "Regressão Logística": Pipeline([("esc", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=3000, class_weight="balanced",
                                                                    random_state=SEMENTE))]),
    }


def dividir(y):
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEMENTE)
    val, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=SEMENTE)
    return tr, val, te


def _metricas(y_true, y_pred) -> dict:
    return {"acuracia": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro"))}


def _carregar_extrator(onnx: str | None):
    if not onnx or not ExtratorProfundo.disponivel(onnx):
        return None
    try:
        return ExtratorProfundo(onnx)
    except Exception as e:  # noqa: BLE001 - rede inválida ou OpenCV sem suporte: segue só com o clássico
        print(f"AVISO: não foi possível carregar a rede pré-treinada ({e}). Usando só características clássicas.")
        return None


def treinar(dados: str, saida: str, modelo_saida: str = "modelos/modelo_visao.joblib",
            onnx: str | None = CAMINHO_PADRAO) -> dict:
    saida = Path(saida)
    (saida / "graficos").mkdir(parents=True, exist_ok=True)
    (saida / "exemplos").mkdir(parents=True, exist_ok=True)

    caminhos, y = listar_imagens(dados)
    print(f"{len(caminhos)} imagens encontradas.")
    extrator = _carregar_extrator(onnx)
    print("Rede pré-treinada: " + ("carregada" if extrator else "não disponível (só características clássicas)"))
    Xc, Xp, y, caminhos = extrair_matrizes(caminhos, y, extrator=extrator)
    caminhos = np.array(caminhos)
    conjuntos = {CLASSICO: Xc}
    if Xp is not None:
        conjuntos[PROFUNDO] = Xp
        conjuntos[COMBINADO] = montar(COMBINADO, Xc, Xp)
    print({nome: X.shape for nome, X in conjuntos.items()})

    tr, val, te = dividir(y)
    pd.DataFrame({"arquivo": caminhos,
                  "conjunto": np.select([np.isin(np.arange(len(y)), tr), np.isin(np.arange(len(y)), val)],
                                        ["treino", "validacao"], "teste"),
                  "classe": y}).to_csv(saida / "divisao_dados.csv", index=False)

    # --- Comparação de conjuntos de características e modelos -------------------
    linhas = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMENTE)
    for nome_conj, X in conjuntos.items():
        for nome, modelo in candidatos().items():
            m = clone(modelo).fit(X[tr], y[tr])
            pred = m.predict(X[val])
            cvs = cross_val_score(clone(modelo), X[tr], y[tr], cv=cv, scoring="f1_macro", n_jobs=-1)
            linhas.append({"caracteristicas": nome_conj, "modelo": nome,
                           "acuracia_val": accuracy_score(y[val], pred),
                           "f1_macro_val": f1_score(y[val], pred, average="macro"),
                           "f1_macro_cv_media": cvs.mean(), "f1_macro_cv_desvio": cvs.std()})
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in linhas[-1].items()})
    comp = pd.DataFrame(linhas).sort_values("f1_macro_val", ascending=False).reset_index(drop=True)
    comp.to_csv(saida / "comparacao_modelos.csv", index=False)
    topo = comp.head(8)
    avaliacao.barras([f"{c}\n{m}" for c, m in zip(topo["caracteristicas"], topo["modelo"])], topo["f1_macro_val"],
                     "F1 macro na validação (8 melhores)", "F1 macro", saida / "graficos" / "comparacao_modelos.png",
                     figsize=(9, 4.2))
    melhor = comp.iloc[0]
    conjunto_final, nome_final = melhor["caracteristicas"], melhor["modelo"]

    # --- Melhor modelo de cada conjunto, avaliado no teste (só relatório) -------
    tr_val = np.concatenate([tr, val])
    por_conjunto, finais = {}, {}
    for nome_conj, X in conjuntos.items():
        linha = comp[comp["caracteristicas"] == nome_conj].iloc[0]
        m = clone(candidatos()[linha["modelo"]]).fit(X[tr_val], y[tr_val])
        finais[nome_conj] = m
        por_conjunto[nome_conj] = {"modelo": linha["modelo"], "f1_macro_val": float(linha["f1_macro_val"]),
                                   **_metricas(y[te], m.predict(X[te]))}

    # --- Modelo final avaliado no teste ---------------------------------------
    final = finais[conjunto_final]
    X_final = conjuntos[conjunto_final]
    proba = final.predict_proba(X_final[te])
    pred = final.classes_[proba.argmax(axis=1)]
    conf = proba.max(axis=1)
    metricas = {
        "conjunto_caracteristicas": conjunto_final,
        "modelo": nome_final,
        "n_imagens": int(len(y)),
        "n_teste": int(len(te)),
        "distribuicao_classes": {c: int((y == c).sum()) for c in CLASSES},
        **_metricas(y[te], pred),
        "por_conjunto_teste": por_conjunto,
        "relatorio_por_classe": classification_report(y[te], pred, labels=CLASSES, output_dict=True, zero_division=0),
        "versao_sklearn": sklearn.__version__,
        "semente": SEMENTE,
    }
    metricas["acuracia_teste"] = metricas.pop("acuracia")
    metricas["f1_macro_teste"] = metricas.pop("f1_macro")
    avaliacao.matriz_confusao(y[te], pred, CLASSES, saida / "graficos" / "matriz_confusao.png")
    avaliacao.exemplos_predicoes(caminhos[te], y[te], pred, conf, saida / "exemplos" / "predicoes_teste.png")
    avaliacao.exemplos_predicoes(caminhos[te], y[te], pred, conf, saida / "exemplos" / "erros_teste.png",
                                 apenas_erros=True)
    pd.DataFrame({"arquivo": caminhos[te], "real": y[te], "previsto": pred, "confianca": conf}) \
        .to_csv(saida / "predicoes_teste.csv", index=False)
    avaliacao.barras(list(por_conjunto), [v["f1_macro"] for v in por_conjunto.values()],
                     "F1 macro no teste por conjunto de características", "F1 macro",
                     saida / "graficos" / "comparacao_caracteristicas.png")

    # --- Ablação das características clássicas --------------------------------
    nome_classico = por_conjunto[CLASSICO]["modelo"]
    abl = {}
    for grupo, fatia in GRUPOS.items():
        m = clone(candidatos()[nome_classico]).fit(Xc[tr_val][:, fatia], y[tr_val])
        abl[grupo] = _metricas(y[te], m.predict(Xc[te][:, fatia]))
    metricas["ablacao_caracteristicas_classicas"] = {"modelo": nome_classico, **abl}
    avaliacao.barras(list(abl), [v["f1_macro"] for v in abl.values()],
                     "Ablação das características clássicas (teste)", "F1 macro",
                     saida / "graficos" / "ablacao_caracteristicas.png")

    # --- Robustez a perturbações (modelo final) --------------------------------
    rob = {}
    for nome, f in avaliacao.PERTURBACOES.items():
        Xc_p, Xp_p, yp, _ = extrair_matrizes(caminhos[te], y[te], extrator=extrator if Xp is not None else None,
                                             transformacao=f)
        rob[nome] = _metricas(yp, final.predict(montar(conjunto_final, Xc_p, Xp_p)))
    metricas["robustez"] = rob
    avaliacao.barras(list(rob), [v["acuracia"] for v in rob.values()], "Robustez a perturbações (teste)",
                     "Acurácia", saida / "graficos" / "robustez.png")

    (saida / "metricas_visao.json").write_text(json.dumps(metricas, indent=2, ensure_ascii=False))
    Path(modelo_saida).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"modelo": final, "classes": list(final.classes_), "conjunto": conjunto_final,
                 "onnx": onnx if conjunto_final != CLASSICO else None,
                 "metricas": {"acuracia_teste": metricas["acuracia_teste"],
                              "f1_macro_teste": metricas["f1_macro_teste"]}}, modelo_saida)
    print("\nResumo no teste (melhor modelo de cada conjunto):")
    for nome_conj, v in por_conjunto.items():
        print(f"  {nome_conj:10s} {v['modelo']:20s} acurácia = {v['acuracia']:.3f} | F1 macro = {v['f1_macro']:.3f}")
    print(f"Modelo final: {conjunto_final} + {nome_final} | acurácia no teste = {metricas['acuracia_teste']:.3f} "
          f"| F1 macro = {metricas['f1_macro_teste']:.3f}")
    return metricas


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Treina o classificador de resíduos do EcoIA.")
    ap.add_argument("--dados", default="dados/trashnet", help="pasta com subpastas por classe")
    ap.add_argument("--saida", default="resultados", help="pasta de gráficos e métricas")
    ap.add_argument("--modelo", default="modelos/modelo_visao.joblib")
    ap.add_argument("--onnx", default=CAMINHO_PADRAO, help="rede pré-treinada (ONNX); ignorada se não existir")
    ap.add_argument("--sem-profundo", action="store_true", help="usa só as características clássicas")
    a = ap.parse_args()
    treinar(a.dados, a.saida, a.modelo, None if a.sem_profundo else a.onnx)
