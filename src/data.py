"""The DAG used for the project (the team's final DAG: figures/dag_final.png) and the data loader."""
import os
import numpy as np
import pandas as pd
from .bbn import BayesNet

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NODES = ["ethnicity", "parent_edu", "income", "wic_child", "snap_past_year", "food_security", "anaemia"]
STATES = {
    "ethnicity": ["MexAm", "OthHisp", "NHWhite", "NHBlack", "Asian_Other"],
    "parent_edu": ["low", "mid", "high"],
    "income": ["below_poverty", "to_185pct", "above_185pct"],
    "wic_child": ["no", "yes"],
    "snap_past_year": ["no", "yes"],
    "food_security": ["secure", "insecure"],
    "anaemia": ["no", "yes"],
}
PARENTS = {
    "ethnicity": [], "parent_edu": [],
    "income": ["ethnicity", "parent_edu"],
    "wic_child": ["income"],
    "snap_past_year": ["income"],
    "food_security": ["income", "wic_child", "snap_past_year"],
    "anaemia": ["ethnicity", "food_security", "wic_child"],
}
COVARIATES = [n for n in NODES if n != "anaemia"]
ETH = {1: "MexAm", 2: "OthHisp", 3: "NHWhite", 4: "NHBlack", 6: "Asian_Other", 7: "Asian_Other"}


def make_net(nodes=NODES):
    nodes = list(nodes)
    return BayesNet(nodes, {n: STATES[n] for n in nodes},
                    {n: [p for p in PARENTS[n] if p in nodes] for n in nodes})


def load(path=None):
    """children_model.csv -> DataFrame with one column of state labels per node (NaN = missing)."""
    path = path or os.path.join(ROOT, "data", "children_model.csv")
    m = pd.read_csv(path)
    d = pd.DataFrame({"id": m["id"]})
    d["ethnicity"] = m["ethnicity"].map(ETH)
    for n in NODES[1:]:
        d[n] = m[n]
    d["age_years"] = m["age_years"]
    return d
