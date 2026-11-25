"""A small discrete Bayesian belief network, written from scratch.

Covers: CPT estimation with a Dirichlet prior, EM for missing data, exact inference
(by variable elimination and, as a cross-check, by joint enumeration), do()
interventions by graph surgery, backdoor adjustment and d-separation.

Only numpy is needed. The joint space of the child-nutrition DAG has 720 states, so
enumeration is exact and fast; variable elimination is the main inference routine.
"""
import itertools
import numpy as np


class BayesNet:
    def __init__(self, nodes, states, parents):
        self.nodes = list(nodes)
        self.states = {n: list(states[n]) for n in self.nodes}
        self.parents = {n: list(parents.get(n, [])) for n in self.nodes}
        self.idx = {n: i for i, n in enumerate(self.nodes)}
        for n in self.nodes:                      # nodes must be listed in topological order
            for p in self.parents[n]:
                assert self.idx[p] < self.idx[n], f"{p} must come before {n}"
        self.card = [len(self.states[n]) for n in self.nodes]
        self.q = {n: int(np.prod([len(self.states[p]) for p in self.parents[n]])) if self.parents[n] else 1
                  for n in self.nodes}
        self.grid = np.array(list(itertools.product(*[range(c) for c in self.card])), dtype=np.int16)
        self._cfg = {}                            # parent-configuration index of every joint state
        for n in self.nodes:
            cfg = np.zeros(len(self.grid), dtype=np.int64)
            for p in self.parents[n]:
                cfg = cfg * len(self.states[p]) + self.grid[:, self.idx[p]]
            self._cfg[n] = cfg
        self.cpt = {n: np.full((self.q[n], len(self.states[n])), 1.0 / len(self.states[n])) for n in self.nodes}
        self.loglik_trace = []

    # ------------------------------------------------------------------ data
    def encode(self, df):
        """DataFrame of state labels (NaN = missing) -> int array, -1 for missing."""
        X = np.full((len(df), len(self.nodes)), -1, dtype=np.int16)
        for j, n in enumerate(self.nodes):
            m = {s: i for i, s in enumerate(self.states[n])}
            X[:, j] = [m.get(v, -1) if v == v and v is not None else -1 for v in df[n].tolist()]
        return X

    def copy(self):
        b = BayesNet(self.nodes, self.states, self.parents)
        b.cpt = {n: c.copy() for n, c in self.cpt.items()}
        return b

    # ------------------------------------------------------------------ fitting
    def _alpha(self, X, strength):
        """Dirichlet pseudo-counts: `strength` per parent configuration, spread like the node's marginal."""
        a = {}
        for j, n in enumerate(self.nodes):
            r = len(self.states[n])
            c = np.bincount(X[X[:, j] >= 0, j], minlength=r).astype(float) + 0.5
            a[n] = np.tile(strength * c / c.sum(), (self.q[n], 1))
        return a

    def fit_counts(self, X, strength=2.0):
        """Available-case counting: each CPT uses the rows where the node and its parents are observed."""
        alpha = self._alpha(X, strength)
        for j, n in enumerate(self.nodes):
            cols = [self.idx[p] for p in self.parents[n]] + [j]
            ok = (X[:, cols] >= 0).all(axis=1)
            cfg = np.zeros(ok.sum(), dtype=np.int64)
            for p in self.parents[n]:
                cfg = cfg * len(self.states[p]) + X[ok, self.idx[p]]
            r = len(self.states[n])
            cnt = np.bincount(cfg * r + X[ok, j], minlength=self.q[n] * r).reshape(self.q[n], r).astype(float)
            c = cnt + alpha[n]
            self.cpt[n] = c / c.sum(axis=1, keepdims=True)
        return self

    def _log_joint(self, do=None):
        lp = np.zeros(len(self.grid))
        for j, n in enumerate(self.nodes):
            if do and n in do:
                s = self.states[n].index(do[n]) if isinstance(do[n], str) else do[n]
                lp += np.where(self.grid[:, j] == s, 0.0, -np.inf)
            else:
                lp += np.log(self.cpt[n][self._cfg[n], self.grid[:, j]])
        return lp

    def _compatible(self, X):
        ok = np.ones((len(X), len(self.grid)), dtype=bool)
        for j in range(len(self.nodes)):
            obs = X[:, j] >= 0
            ok[obs] &= self.grid[:, j][None, :] == X[obs, j][:, None]
        return ok

    def fit_em(self, X, strength=2.0, max_iter=500, tol=1e-8, init=None):
        """EM for the CPTs when some cells of X are missing. Returns self; loglik_trace holds the
        penalised log-likelihood (log-likelihood + log Dirichlet prior) of every iteration."""
        alpha = self._alpha(X, strength)
        if init is not None:
            self.cpt = {n: c.copy() for n, c in init.items()}
        else:
            self.fit_counts(X, strength)
        ok = self._compatible(X)
        self.loglik_trace = []
        prev = -np.inf
        for _ in range(max_iter):
            lp = self._log_joint()
            p = np.exp(lp)[None, :] * ok
            z = p.sum(axis=1)
            ll = np.log(z).sum()
            pen = ll + sum((alpha[n] * np.log(self.cpt[n])).sum() for n in self.nodes)
            self.loglik_trace.append(pen)
            w = (p / z[:, None]).sum(axis=0)                       # expected counts of every joint state
            for n in self.nodes:
                r = len(self.states[n])
                cnt = np.bincount(self._cfg[n] * r + self.grid[:, self.idx[n]], weights=w,
                                  minlength=self.q[n] * r).reshape(self.q[n], r)
                c = cnt + alpha[n]
                self.cpt[n] = c / c.sum(axis=1, keepdims=True)
            if abs(pen - prev) < tol:
                break
            prev = pen
        return self

    def loglik(self, X):
        p = np.exp(self._log_joint())[None, :] * self._compatible(X)
        return float(np.log(p.sum(axis=1)).sum())

    # ------------------------------------------------------------------ inference by enumeration
    def enum_marginals(self, evidence=None, do=None):
        """P(node | evidence) for every node, by summing the full joint. Reference implementation."""
        ev = self._clean_evidence(evidence, do)
        p = np.exp(self._log_joint(do))
        for n, s in ev.items():
            p = p * (self.grid[:, self.idx[n]] == s)
        z = p.sum()
        if z <= 0:
            raise ValueError("evidence has zero probability")
        p = p / z
        return {n: np.bincount(self.grid[:, j], weights=p, minlength=self.card[j]) for j, n in enumerate(self.nodes)}

    def _state_index(self, n, s):
        return self.states[n].index(s) if isinstance(s, str) else int(s)

    def descendants(self, n):
        out, frontier = set(), [n]
        while frontier:
            c = frontier.pop()
            for m in self.nodes:
                if c in self.parents[m] and m not in out:
                    out.add(m)
                    frontier.append(m)
        return out

    def _clean_evidence(self, evidence, do):
        """Evidence is dropped for intervened nodes and their descendants: it was observed before the
        intervention, and the intervention changes those variables."""
        ev = {n: self._state_index(n, s) for n, s in (evidence or {}).items() if s is not None}
        if do:
            gone = set(do)
            for n in do:
                gone |= self.descendants(n)
            ev = {n: s for n, s in ev.items() if n not in gone}
        return ev

    # ------------------------------------------------------------------ variable elimination
    def _factors(self, do=None):
        fs = []
        for n in self.nodes:
            if do and n in do:
                t = np.zeros(len(self.states[n]))
                t[self._state_index(n, do[n])] = 1.0
                fs.append(((n,), t))
            else:
                vars_ = tuple(self.parents[n]) + (n,)
                shape = [len(self.states[v]) for v in vars_]
                fs.append((vars_, self.cpt[n].reshape(shape)))
        return fs

    @staticmethod
    def _multiply(f, g):
        (vf, tf), (vg, tg) = f, g
        allv = list(dict.fromkeys(vf + vg))
        letters = {v: chr(97 + i) for i, v in enumerate(allv)}
        sub = lambda vs: "".join(letters[v] for v in vs)
        out = np.einsum(f"{sub(vf)},{sub(vg)}->{sub(allv)}", tf, tg)
        return tuple(allv), out

    @staticmethod
    def _sum_out(f, v):
        vs, t = f
        ax = vs.index(v)
        return tuple(x for x in vs if x != v), t.sum(axis=ax)

    def ve_query(self, target, evidence=None, do=None):
        """P(target | evidence) by variable elimination (greedy min-size elimination order)."""
        ev = self._clean_evidence(evidence, do)
        if target in ev:                              # already observed
            out = np.zeros(len(self.states[target]))
            out[ev[target]] = 1.0
            return out
        factors = []
        for vars_, t in self._factors(do):
            for v in list(vars_):                    # restrict every factor to the evidence
                if v in ev:
                    ax = vars_.index(v)
                    t = np.take(t, ev[v], axis=ax)
                    vars_ = tuple(x for x in vars_ if x != v)
            factors.append((vars_, np.asarray(t, dtype=float)))
        hidden = [n for n in self.nodes if n != target and n not in ev]
        while hidden:
            def size(v):
                scope = set()
                for f in factors:
                    if v in f[0]:
                        scope |= set(f[0])
                return int(np.prod([len(self.states[x]) for x in scope]))
            v = min(hidden, key=size)
            hidden.remove(v)
            touching = [f for f in factors if v in f[0]]
            factors = [f for f in factors if v not in f[0]]
            prod = touching[0]
            for f in touching[1:]:
                prod = self._multiply(prod, f)
            factors.append(self._sum_out(prod, v))
        prod = factors[0]
        for f in factors[1:]:
            prod = self._multiply(prod, f)
        vs, t = prod
        assert vs == (target,), vs
        z = t.sum()
        if z <= 0:
            raise ValueError("evidence has zero probability")
        return t / z

    # ------------------------------------------------------------------ causal helpers
    def backdoor_effect(self, treat, treat_state, outcome, adjust, outcome_state):
        """P(outcome=state | do(treat=treat_state)) = sum_z P(outcome | treat, z) P(z), z over the adjustment set."""
        total = 0.0
        joint = np.exp(self._log_joint())
        j_t = self.idx[treat]
        t_idx = self._state_index(treat, treat_state)
        o_idx = self._state_index(outcome, outcome_state)
        cols = [self.idx[a] for a in adjust]
        zkeys = {}
        for k, row in enumerate(self.grid[:, cols]):
            zkeys.setdefault(tuple(row), []).append(k)
        for key, ks in zkeys.items():
            ks = np.array(ks)
            p_z = joint[ks].sum()
            sel = ks[self.grid[ks, j_t] == t_idx]
            p_tz = joint[sel].sum()
            if p_tz <= 0:
                continue
            p_o = joint[sel][self.grid[sel, self.idx[outcome]] == o_idx].sum()
            total += (p_o / p_tz) * p_z
        return total

    def markov_blanket(self, n):
        mb = set(self.parents[n])
        for m in self.nodes:
            if n in self.parents[m]:
                mb.add(m)
                mb |= set(self.parents[m])
        mb.discard(n)
        return mb

    def d_separated(self, x, y, given=()):
        """d-separation by the moralised ancestral graph criterion."""
        keep = {x, y, *given}
        frontier = list(keep)
        while frontier:
            c = frontier.pop()
            for p in self.parents[c]:
                if p not in keep:
                    keep.add(p)
                    frontier.append(p)
        adj = {n: set() for n in keep}
        for n in keep:
            ps = [p for p in self.parents[n] if p in keep]
            for p in ps:
                adj[n].add(p); adj[p].add(n)
            for a, b in itertools.combinations(ps, 2):
                adj[a].add(b); adj[b].add(a)
        blocked = set(given)
        seen, stack = set(), [x]
        while stack:
            c = stack.pop()
            if c == y:
                return False
            if c in seen:
                continue
            seen.add(c)
            stack.extend(a for a in adj[c] if a not in blocked)
        return True

    def implied_independencies(self):
        """For each node: the non-descendants it is independent of given its parents (local Markov property),
        plus everything outside the Markov blanket."""
        out = []
        for n in self.nodes:
            mb = self.markov_blanket(n)
            for m in self.nodes:
                if m != n and m not in mb and self.d_separated(n, m, mb):
                    out.append((n, m, tuple(sorted(mb))))
        seen, uniq = set(), []
        for a, b, z in out:
            k = (frozenset((a, b)), z)
            if k not in seen:
                seen.add(k)
                uniq.append((a, b, z))
        return uniq

    # ------------------------------------------------------------------ export
    def to_dict(self):
        return {"nodes": self.nodes, "states": self.states, "parents": self.parents,
                "cpt": {n: np.round(self.cpt[n], 6).tolist() for n in self.nodes}}
