"""Tests for the belief network. Run from the repo root:  python -m unittest discover -s tests -v
(pytest also works if installed)."""
import itertools
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import NODES, STATES, PARENTS, load, make_net  # noqa: E402

SEED_STRENGTH = 2.0


def fitted_net():
    d = load()
    net = make_net()
    net.fit_em(net.encode(d), strength=SEED_STRENGTH)
    return net


NET = fitted_net()


def random_net(seed=0):
    rng = np.random.default_rng(seed)
    net = make_net()
    for n in NODES:
        c = rng.gamma(0.7, size=net.cpt[n].shape) + 0.05
        net.cpt[n] = c / c.sum(axis=1, keepdims=True)
    return net


def sample(net, n, rng):
    X = np.zeros((n, len(NODES)), dtype=np.int16)
    for j, node in enumerate(NODES):
        cfg = np.zeros(n, dtype=np.int64)
        for p in net.parents[node]:
            cfg = cfg * len(net.states[p]) + X[:, net.idx[p]]
        pr = net.cpt[node][cfg]
        X[:, j] = (pr.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
    return X


class TestParameters(unittest.TestCase):
    def test_cpt_rows_sum_to_one(self):
        for n in NODES:
            self.assertTrue(np.allclose(NET.cpt[n].sum(axis=1), 1.0))
            self.assertTrue((NET.cpt[n] > 0).all())

    def test_em_objective_never_decreases(self):
        tr = np.array(NET.loglik_trace)
        self.assertGreater(len(tr), 2)
        self.assertTrue((np.diff(tr) > -1e-9).all())

    def test_em_recovers_true_parameters_better_than_deletion(self):
        rng = np.random.default_rng(1)
        truth = random_net(3)
        X = sample(truth, 1500, rng)
        Xm = X.copy()
        Xm[rng.random(X.shape) < 0.3] = -1
        em = make_net().fit_em(Xm, strength=0.5)
        complete = Xm[(Xm >= 0).all(axis=1)]
        cc = make_net().fit_counts(complete, strength=0.5)
        err = lambda m: np.mean([np.abs(m.cpt[n] - truth.cpt[n]).mean() for n in NODES])
        self.assertLess(err(em), err(cc))


class TestInference(unittest.TestCase):
    def test_variable_elimination_matches_enumeration(self):
        net = random_net(5)
        evs = [{}, {"food_security": "insecure"}, {"income": "below_poverty", "wic_child": "yes"},
               {"ethnicity": "NHBlack", "snap_past_year": "no", "anaemia": "yes"}]
        for ev in evs:
            enum = net.enum_marginals(ev)
            for t in NODES:
                if t in ev:
                    continue
                self.assertTrue(np.allclose(net.ve_query(t, ev), enum[t], atol=1e-10), (t, ev))

    def test_ve_matches_enumeration_under_do(self):
        net = random_net(6)
        for do in [{"food_security": "secure"}, {"wic_child": "yes"}, {"wic_child": "yes", "snap_past_year": "yes"}]:
            ev = {"ethnicity": "OthHisp", "income": "below_poverty"}
            enum = net.enum_marginals(ev, do)
            for t in NODES:
                if t in do or t in ev:
                    continue
                self.assertTrue(np.allclose(net.ve_query(t, ev, do), enum[t], atol=1e-10), (t, do))

    def test_probabilities_are_distributions(self):
        for t in NODES:
            p = NET.ve_query(t, {"income": "to_185pct"})
            self.assertAlmostEqual(p.sum(), 1.0)
            self.assertTrue((p >= 0).all())

    def test_evidence_on_children_revises_beliefs(self):
        base = NET.ve_query("food_security", {"income": "below_poverty"})[1]
        withwic = NET.ve_query("food_security", {"income": "below_poverty", "wic_child": "yes"})[1]
        self.assertNotAlmostEqual(base, withwic, places=3)


class TestStructure(unittest.TestCase):
    def test_d_separation(self):
        self.assertTrue(NET.d_separated("ethnicity", "parent_edu"))                      # no path
        self.assertFalse(NET.d_separated("ethnicity", "parent_edu", ["income"]))         # collider opened
        self.assertTrue(NET.d_separated("anaemia", "income", ["ethnicity", "food_security", "wic_child"]))
        self.assertFalse(NET.d_separated("anaemia", "income", ["food_security"]))        # via wic and ethnicity
        self.assertFalse(NET.d_separated("snap_past_year", "wic_child"))                 # common cause income
        self.assertTrue(NET.d_separated("snap_past_year", "wic_child", ["income"]))

    def test_markov_blanket_of_outcome(self):
        self.assertEqual(NET.markov_blanket("anaemia"), {"ethnicity", "food_security", "wic_child"})

    def test_outcome_independent_of_non_blanket_given_blanket_numerically(self):
        net = random_net(8)
        mb = {"ethnicity": "NHBlack", "food_security": "insecure", "wic_child": "yes"}
        a = net.ve_query("anaemia", mb)
        b = net.ve_query("anaemia", {**mb, "income": "below_poverty", "parent_edu": "low", "snap_past_year": "no"})
        self.assertTrue(np.allclose(a, b))


class TestCausal(unittest.TestCase):
    def test_surgery_equals_backdoor_adjustment(self):
        for net in (NET, random_net(11)):
            cases = [("food_security", ["income", "wic_child"]),
                     ("food_security", ["income", "wic_child", "snap_past_year"]),
                     ("wic_child", ["income"]), ("snap_past_year", ["income"])]
            for treat, adj in cases:
                for s in STATES[treat]:
                    surgery = net.ve_query("anaemia", None, {treat: s})[1]
                    backdoor = net.backdoor_effect(treat, s, "anaemia", adj, "yes")
                    self.assertAlmostEqual(surgery, backdoor, places=10, msg=(treat, s, adj))

    def test_do_on_a_root_equals_conditioning(self):
        for s in STATES["ethnicity"]:
            self.assertTrue(np.allclose(NET.ve_query("anaemia", {"ethnicity": s}),
                                        NET.ve_query("anaemia", None, {"ethnicity": s})))

    def test_conditioning_differs_from_intervening_when_confounded(self):
        cond = NET.ve_query("anaemia", {"food_security": "secure"})[1]
        do = NET.ve_query("anaemia", None, {"food_security": "secure"})[1]
        self.assertGreater(abs(cond - do), 1e-4)

    def test_intervening_on_the_outcome_leaves_causes_unchanged(self):
        a = NET.ve_query("food_security", {"income": "below_poverty"})
        b = NET.ve_query("food_security", {"income": "below_poverty"}, {"anaemia": "yes"})
        self.assertTrue(np.allclose(a, b))

    def test_evidence_on_descendants_is_dropped_under_do(self):
        a = NET.ve_query("anaemia", {"income": "below_poverty"}, {"food_security": "secure"})
        b = NET.ve_query("anaemia", {"income": "below_poverty", "anaemia": "yes"}, {"food_security": "secure"})
        self.assertTrue(np.allclose(a, b))


class TestSampleCases(unittest.TestCase):
    """Directional checks on the model fitted to the NHANES data."""
    def risk(self, ev, do=None):
        return NET.ve_query("anaemia", ev, do)[1]

    def test_disadvantaged_child_has_higher_risk_than_advantaged_child(self):
        high = self.risk({"ethnicity": "NHBlack", "income": "below_poverty", "food_security": "insecure"})
        low = self.risk({"ethnicity": "NHWhite", "income": "above_185pct", "food_security": "secure"})
        self.assertGreater(high, 1.5 * low)

    def test_food_insecurity_raises_risk(self):
        self.assertGreater(self.risk({"food_security": "insecure"}), self.risk({"food_security": "secure"}))

    def test_securing_food_lowers_risk(self):
        ev = {"ethnicity": "OthHisp", "income": "below_poverty"}
        self.assertLess(self.risk(ev, {"food_security": "secure"}), self.risk(ev, {"food_security": "insecure"}))

    def test_baseline_risk_is_close_to_observed_prevalence(self):
        prev = 24 / 340
        self.assertLess(abs(NET.ve_query("anaemia")[1] - prev), 0.03)


if __name__ == "__main__":
    unittest.main()
