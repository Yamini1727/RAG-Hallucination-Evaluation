import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st

R = Path(__file__).resolve().parent.parent / "results"
st.set_page_config(page_title="RAG Hallucination Evaluation", page_icon="🔎", layout="wide")

@st.cache_data
def csv(name): return pd.read_csv(R / name)
@st.cache_data
def js(name): return json.load(open(R / name))

summary, pairwise, factors = csv("config_summary.csv"), csv("pairwise_tests.csv"), csv("factor_tests.csv")
detectors, grid, choice = csv("detector_comparison.csv"), csv("grid_results.csv"), js("detector_choice.json")
retr = js("retrieval_vs_hallucination.json")
FACTOR_COLORS = {"baseline": "#495057", "chunk_size": "#4C6EF5", "top_k": "#12B886", "embedding": "#E8590C",
                 "retrieval": "#AE3EC9", "generator": "#F59F00"}

st.title("🔎 How RAG design choices affect hallucination")
st.caption(f"482 questions over 300 Wikipedia articles · {summary.shape[0]} configurations · "
           f"detector: **{choice['chosen']}** (threshold {choice['threshold']:.3f})")

tab_over, tab_cfg, tab_stats, tab_det, tab_ex, tab_about = st.tabs(
    ["Overview", "Configurations", "Statistics", "Detectors", "Answer explorer", "Method"])

with tab_over:
    b = summary.set_index("config").loc["baseline"]
    qwen = summary[summary.generator != "gemini"]
    best, worst = qwen.loc[qwen.halluc_rate.idxmin()], qwen.loc[qwen.halluc_rate.idxmax()]
    c = st.columns(4)
    c[0].metric("Baseline hallucination", f"{b.halluc_rate:.1%}", help="512-token chunks, top-3, bge-small, dense, Qwen2.5-1.5B")
    c[1].metric("Lowest", f"{best.halluc_rate:.1%}", best.config)
    c[2].metric("Highest", f"{worst.halluc_rate:.1%}", worst.config)
    c[3].metric("Significant differences", int(pairwise[(pairwise.outcome == "hallucination") & pairwise.significant].shape[0]),
                help="McNemar vs baseline, Holm-corrected, p < 0.05")
    fig = px.scatter(summary, x="halluc_rate", y="config", color="factor", color_discrete_map=FACTOR_COLORS,
                     error_x=summary.halluc_hi - summary.halluc_rate, error_x_minus=summary.halluc_rate - summary.halluc_lo,
                     labels={"halluc_rate": "Hallucination rate (95% Wilson CI)", "config": ""})
    fig.update_layout(xaxis_tickformat=".0%", height=430, legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    st.info(f"When the gold article was **retrieved**, the hallucination rate was "
            f"{retr['halluc_rate_when_gold_retrieved']:.1%}; when it was **missed**, {retr['halluc_rate_when_gold_missed']:.1%} "
            f"(χ² p = {retr['p']:.3g}).")

with tab_cfg:
    f = st.selectbox("Factor", ["chunk_size", "top_k", "embedding", "retrieval", "generator"])
    d = summary[summary.factor.isin(["baseline", f])]
    metric = st.radio("Metric", ["halluc_rate", "abstain_rate", "gold_retrieved_rate", "answer_recall"], horizontal=True)
    fig = px.bar(d, x="config", y=metric, color="factor", color_discrete_map=FACTOR_COLORS, text_auto=".1%")
    fig.update_layout(yaxis_tickformat=".0%", showlegend=False, height=380)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(d.round(3), width="stretch", hide_index=True)

with tab_stats:
    st.subheader("Each configuration vs baseline (paired McNemar, Holm-corrected)")
    outc = st.radio("Outcome", ["hallucination", "abstention"], horizontal=True)
    st.dataframe(pairwise[pairwise.outcome == outc].round(4), width="stretch", hide_index=True)
    st.subheader("Does the factor matter overall? (Cochran's Q, χ², Cramér's V)")
    st.dataframe(factors.round(4), width="stretch", hide_index=True)

with tab_det:
    st.write(f"Chosen detector: **{choice['chosen']}** — {choice['rule']}")
    st.dataframe(detectors.round(3), width="stretch", hide_index=True)
    roc = R / "figures" / "detector_roc.png"
    if roc.exists(): st.image(str(roc))

with tab_ex:
    c1, c2 = st.columns(2)
    cfg_a = c1.selectbox("Configuration A", summary.config, index=0)
    cfg_b = c2.selectbox("Configuration B", summary.config, index=min(1, len(summary) - 1))
    only = st.checkbox("Only questions where A and B disagree on hallucination")
    a = grid[grid.config == cfg_a].set_index("qid"); bb = grid[grid.config == cfg_b].set_index("qid")
    qids = a.index[(a.hallucinated != bb.hallucinated)] if only else a.index
    if len(qids) == 0:
        st.write("No disagreements.")
    else:
        q = st.selectbox("Question", qids, format_func=lambda i: a.loc[i, "question"][:110])
        st.markdown(f"**Question:** {a.loc[q, 'question']}  \n**Gold answer:** {a.loc[q, 'gold_answer']}")
        for col, cfg, d in [(c1, cfg_a, a), (c2, cfg_b, bb)]:
            r = d.loc[q]
            tag = "🟡 abstained" if r.abstained else ("🔴 hallucinated" if r.hallucinated else "🟢 faithful")
            col.markdown(f"**{cfg}** — {tag}  (signal {r.signal:.2f})" if r.signal == r.signal else f"**{cfg}** — {tag}")
            col.write(r.answer)
            col.caption(f"Retrieved: {r.retrieved_titles}")
            with col.expander("Top retrieved chunk"): st.write(r.top_chunk)

with tab_about:
    st.markdown("""
**Pipeline.** Wikipedia (300 articles) → SentenceSplitter chunks → embeddings → top-k retrieval → generator → answer.
**Outcome.** An answer is *abstained* if it refuses ("context doesn't contain the answer"), otherwise it is scored
against each retrieved chunk by the detector; the best-supporting chunk decides (per-chunk max).
**Design.** One factor changed at a time; every configuration answers the same 482 questions, so tests are paired
(McNemar, Cochran's Q) with Holm correction, Wilson CIs and paired bootstrap CIs.
**Detector.** HHEM vs DeBERTa-v3-small+LoRA (trained on RAGTruth+HaluEval), validated on a held-out benchmark,
a controlled injected-hallucination set and 75 human labels.
""")
