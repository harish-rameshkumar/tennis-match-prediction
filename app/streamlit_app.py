"""Tennis match predictor demo (owner: Jayant).

Run from the repo root:   streamlit run app/streamlit_app.py
Needs a trained model:    python -m tennis.predict --train
"""
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)                      # predict.py looks for models/ relative to the repo root

import pandas as pd
import streamlit as st

from tennis import predict as P

st.set_page_config(page_title="Tennis match predictor", page_icon="🎾", layout="centered")
st.title("🎾 Tennis match predictor")
st.caption("Pre-match win probability for an ATP match, from logistic regression on "
           "Elo, form, head-to-head, surface and serve statistics. Test accuracy is "
           "about 66% (rank baseline about 61-63%), so treat it as an estimate, "
           "not a certainty.")

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]


@st.cache_data(show_spinner="Loading player list...")
def player_names():
    """Sorted player names from the match history, or None if the history has
    no recognisable name columns (then we fall back to free-text boxes)."""
    matches, _ = P._history()
    cols = [c for c in matches.columns if c.lower().endswith("name")
            and any(k in c.lower() for k in ("winner", "loser", "player"))]
    if not cols:
        return None
    names = pd.concat([matches[c] for c in cols]).dropna().astype(str).unique()
    return sorted(names)


def trained_models():
    return [m for m in ("logreg", "linear_svc", "hgb", "rf") if P._path(m).exists()]


models = trained_models()
if not models:
    st.error("No trained model found. Run `python -m tennis.predict --train` "
             "from the repo root, then refresh this page.")
    st.stop()

names = player_names()


def pick(label, default):
    if names:
        idx = names.index(default) if default in names else 0
        return st.selectbox(label, names, index=idx)
    return st.text_input(label, default)


c1, c2 = st.columns(2)
with c1:
    player_a = pick("Player A", "Novak Djokovic")
with c2:
    player_b = pick("Player B", "Carlos Alcaraz")

c3, c4, c5 = st.columns(3)
surface = c3.selectbox("Surface", SURFACES)
match_date = c4.date_input("Match date", value=date.today())
tournament = c5.text_input("Tournament (optional)", "")

with st.expander("Advanced: override rank, ranking points, age, height, hand"):
    st.caption("Leave blank/0 to use each player's latest known values before the date. "
               "A rank is required for a player with no earlier match.")
    o1, o2 = st.columns(2)
    attrs = {}
    for col, side, label in ((o1, "a", player_a), (o2, "b", player_b)):
        with col:
            r = st.number_input(f"Rank of {label}", min_value=0, value=0, step=1, key=f"r{side}")
            pts = st.number_input(f"Ranking points of {label}", min_value=0, value=0,
                                  step=10, key=f"p{side}")
            if r:
                attrs[f"rank_{side}"] = int(r)
            if pts:
                attrs[f"rank_points_{side}"] = int(pts)

model_name = st.selectbox("Model", models,
                          help="Feature explanations are available for logistic regression only.")
top_k = st.slider("Features to show", 3, 15, 8)

if st.button("Predict", type="primary"):
    if player_a == player_b:
        st.warning("Pick two different players.")
        st.stop()
    try:
        with st.spinner("Building features (the first prediction loads the match history)..."):
            res = P.predict_match(player_a, player_b, surface, match_date,
                                  tournament=tournament or None,
                                  attributes=attrs or None, model=model_name, top_k=top_k)
    except Exception as e:                       # name problems come from inference.py
        kind = type(e).__name__
        if kind == "UnknownPlayerError":
            sug = getattr(e, "suggestions", None)
            st.error(f"Unknown player. {e}" + (f" Did you mean: {', '.join(map(str, sug))}?" if sug else ""))
        elif kind == "AmbiguousPlayerError":
            cand = getattr(e, "candidates", None)
            st.error(f"That name matches several players: {', '.join(map(str, cand))}"
                     if cand else f"Ambiguous name. {e}")
        else:
            st.error(f"{kind}: {e}")
        st.stop()

    na, nb = res["player_a"], res["player_b"]
    m1, m2 = st.columns(2)
    m1.metric(na, f"{res['p_a']:.1%}")
    m2.metric(nb, f"{res['p_b']:.1%}")
    st.progress(min(max(res["p_a"], 0.0), 1.0))
    winner = na if res["p_a"] >= 0.5 else nb
    st.markdown(f"**Predicted winner: {winner}** ({max(res['p_a'], res['p_b']):.1%})")

    if res["top_features"]:
        df = pd.DataFrame(res["top_features"])
        df["favours"] = df["favours"].map({"A": na, "B": nb})
        st.subheader("What drove the prediction")
        st.caption(f"Contribution = coefficient x standardized value, in log-odds. "
                   f"Positive favours {na}; negative favours {nb}.")
        st.bar_chart(df.set_index("feature")["contribution"])
        st.dataframe(df.rename(columns={"contribution": "contribution (log-odds)",
                                        "value": "raw value (A minus B for diff_*)"}),
                     hide_index=True)
    elif model_name != "logreg":
        st.info("Feature explanations are only available for logistic regression.")

    with st.expander("How the probability is computed"):
        st.write(
            f"Every match is scored twice, once from each player's side: "
            f"p({na} beats {nb}) = {res['p_ab']:.3f} and p({nb} beats {na}) = {res['p_ba']:.3f}. "
            f"These are combined as ½·(p_ab + 1 − p_ba) = {res['p_a']:.3f}, which makes "
            f"the two players' probabilities add up to exactly 1.")
