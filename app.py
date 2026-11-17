"""Interface Streamlit do EcoIA.  Execução:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from src.constantes import NOMES_EXIBICAO
from src.pipeline import EcoIA
from src.visao_computacional.extracao import decodificar_bytes

st.set_page_config(page_title="EcoIA", page_icon="♻️", layout="centered")


@st.cache_resource
def carregar_sistema() -> EcoIA:
    return EcoIA()


sistema = carregar_sistema()

st.title("♻️ EcoIA")
st.caption("Separador inteligente de resíduos: envie uma foto e/ou faça uma pergunta sobre o descarte.")

if not sistema.pronto["visao"] or not sistema.pronto["pln"]:
    st.warning("Modelos não encontrados. Rode o treino antes (veja o README): "
               f"visão={'ok' if sistema.pronto['visao'] else 'faltando'}, "
               f"PLN={'ok' if sistema.pronto['pln'] else 'faltando'}.")

with st.sidebar:
    st.header("Parâmetros do agente")
    sistema.agente.limiar_minimo = st.slider("Confiança mínima da imagem", 0.2, 0.9, 0.50, 0.05)
    sistema.agente.limiar_alto = st.slider("Confiança alta (imagem prevalece)", 0.5, 0.99, 0.75, 0.01)
    sistema.agente.margem_ambigua = st.slider("Margem de ambiguidade", 0.0, 0.4, 0.15, 0.01)
    st.caption("Protótipo acadêmico. Confirme a orientação com a coleta seletiva do seu município.")

modo = st.radio("Entrada da imagem", ["Enviar arquivo", "Usar câmera"], horizontal=True)
arquivo = st.file_uploader("Foto do resíduo", type=["jpg", "jpeg", "png"]) if modo == "Enviar arquivo" \
    else st.camera_input("Tire uma foto do resíduo")

exemplos = ["Onde devo jogar isso?", "Qual a cor da lixeira?", "Preciso lavar antes de descartar?",
            "Isso é reciclável?", "Que material é esse?", "Posso jogar isso na lixeira azul?"]
st.write("Perguntas de exemplo:")
colunas = st.columns(len(exemplos))
for col, ex in zip(colunas, exemplos):
    if col.button(ex, use_container_width=True):
        st.session_state["pergunta"] = ex
pergunta = st.text_input("Sua pergunta (opcional)", key="pergunta",
                         placeholder="Ex.: Posso jogar essa garrafa na lixeira azul?")

if st.button("Analisar", type="primary"):
    imagem = decodificar_bytes(arquivo.getvalue()) if arquivo is not None else None
    if imagem is None and not pergunta.strip():
        st.info("Envie uma imagem ou escreva uma pergunta.")
    else:
        try:
            r = sistema.processar(imagem, pergunta)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()
        d, v, t = r["decisao"], r["visual"], r["texto"]

        if d.acao in ("pedir_nova_imagem", "pedir_esclarecimento", "fora_escopo"):
            st.info(d.mensagem)
        else:
            st.success(d.mensagem)
            for aviso in d.avisos:
                st.warning(aviso)

        c1, c2 = st.columns(2)
        if imagem is not None:
            c1.image(imagem, caption="Imagem enviada", channels="BGR")
            if r["recorte"] is not None:
                c2.image(r["recorte"], caption="Região analisada" + ("" if v.recorte_ok else " (imagem inteira)"),
                         channels="BGR")
        if v is not None:
            st.subheader("Visão Computacional + Machine Learning")
            st.write(f"Classe prevista: **{NOMES_EXIBICAO[v.classe]}** (confiança {v.confianca:.0%})")
            st.bar_chart(pd.Series({NOMES_EXIBICAO[k]: p for k, p in v.probabilidades.items()}))
        if t is not None:
            st.subheader("Processamento de Linguagem Natural")
            st.write(f"Intenção: **{t.intencao}** (confiança {t.confianca_intencao:.0%}) | "
                     f"material citado: **{t.material or '—'}** | objeto: **{t.objeto or '—'}**")
            st.caption(f"Tokens normalizados: {', '.join(t.tokens)}")
        with st.expander("Como o agente decidiu"):
            st.write(f"Ação: `{d.acao}` | fonte: `{d.fonte}`")
            for regra in d.regras:
                st.write("• " + regra)
