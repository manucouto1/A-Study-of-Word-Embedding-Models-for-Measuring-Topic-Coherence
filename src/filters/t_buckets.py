"""
TBuckets - Measuring Topic Coherence through Optimal Word Buckets

Based on:
    Ramrakhiyani et al. (2017) - EACL 2017
    "Measuring Topic Coherence through Optimal Word Buckets"

This filter receives pre-computed topic embeddings (shape: n_topics × n_words × emb_dim)
from a prior embedding filter in the pipeline (e.g. GensimEmbedder) and computes the
TBuckets coherence score for each topic.

Algorithm summary (n = number of words per topic, d = embedding dim):
    1. Build A (n×d) from the topic's word embeddings.
    2. SVD: A = U S V^T  →  take first n rows of V^T as eigenvectors (themes).
    3. Build matrices E (n×n), W (n×n), L ((n-1)×n).
    4. Solve the ILP with objective + constraints C1–C5 from the paper.
    5. Score = size of the principal bucket (bucket 0).
"""

import numpy as np
from scipy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpBinary, PULP_CBC_CMD

from labchain import Container, BaseFilter, XYData


@Container.bind()
class TBuckets(BaseFilter):
    """
    TBuckets coherence metric.

    Expects x.value to be a 3-D tensor/array of shape (n_topics, n_words, emb_dim),
    exactly as produced by GensimEmbedder or TransformersEmbedder.

    The algorithm is parameter-free: n (number of buckets) equals n_words,
    exactly as specified in the paper.
    """

    def __init__(self, fixer:str=""):
        super().__init__()
        self.fixer=fixer

    # ------------------------------------------------------------------
    # Core algorithm helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _svd_eigenvectors(A: np.ndarray) -> np.ndarray:
        """
        SVD on A (n×d).  Returns the first n eigenvectors of A^T A,
        i.e. the first n rows of V^T  →  shape (n, d).

        scipy.linalg.svd with full_matrices=False returns V^T with shape
        (min(n, d), d).  Since n << d (typically 10 vs 300) we get exactly
        n rows — precisely what the paper needs.

        The rows are ordered by descending singular value, so row 0 is
        the principal eigenvector (central theme).
        """
        n = A.shape[0]
        _, _, Vt = svd(A, full_matrices=False)
        return Vt[:n, :]  # (n, d)

    @staticmethod
    def _build_E(embeddings: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
        """
        E[i][j] = cosine similarity between eigenvector i and word j.
        Shape: (n, n).

        Paper definition:
            "E: Matrix of dimensions n × n, where E_ij represents
             similarity of the j-th word with the i-th eigenvector"

        Sign correction — row 0 only:
            SVD eigenvectors have arbitrary sign.  The paper's objective
            penalises words NOT in the principal bucket by their similarity
            to the principal eigenvector (E[0][j]).  This penalty only works
            correctly when E[0][j] > 0 for words that semantically align
            with the principal theme.  If the principal eigenvector points
            away from the cluster, all E[0][j] are negative and the penalty
            flips into a reward, breaking the optimisation.

            We therefore flip row 0 when its mean is negative.  Other rows
            are left untouched: their signs are meaningful *relative to each
            other* within the SVD and flipping them independently would
            corrupt the L matrix and the C4/C5 constraint logic.
        """
        E = cosine_similarity(eigenvectors, embeddings)  # (n, n)
        if E[0].mean() < 0:
            E[0] = -E[0]
        return E

    @staticmethod
    def _build_W(embeddings: np.ndarray) -> np.ndarray:
        """
        W[i][j] = cosine similarity between word i and word j.
        Shape: (n, n).

        Used in constraint C3 to ensure a word assigned to a non-principal
        bucket is more similar to that bucket's eigenvector than to any
        word currently in the principal bucket.
        """
        return cosine_similarity(embeddings, embeddings)  # (n, n)

    @staticmethod
    def _build_L(E: np.ndarray) -> np.ndarray:
        """
        L is (n-1) × n.
        L[i][j] = 1  if  E[i+1][j] > E[0][j]   (eigenvector i+1 is more
                                                   similar to word j than the
                                                   principal eigenvector)
                   0  otherwise.

        Paper definition (1-indexed):
            "L: Matrix of dimensions (n−1)×n, where L_ij = 1 if E_(i+1)j > E_1j else 0"

        Used in C4 and C5 to count how many eigenvectors beat the principal
        for each word, controlling which words may join the principal bucket.
        """
        n = E.shape[1]
        L = np.zeros((n - 1, n), dtype=float)
        for i in range(n - 1):
            for j in range(n):
                if E[i + 1, j] > E[0, j]:
                    L[i, j] = 1.0
        return L

    # ------------------------------------------------------------------
    # ILP (Table 1 of the paper, verbatim)
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_ilp(E: np.ndarray, W: np.ndarray, L: np.ndarray) -> np.ndarray:
        """
        Solves the ILP from Table 1 of Ramrakhiyani et al. (2017).

        Returns X as a (n, n) numpy array of 0/1 assignments:
            X[i][j] = 1  →  word j is assigned to bucket i (eigenvector i).

        Note on edge cases:
            If the solver reports Infeasible (can happen with embeddings that
            have negative pairwise cosine similarity — rare with GloVe/W2V but
            possible with transformer embeddings), PuLP still returns the best
            solution found.  We extract whatever assignments are available;
            words with no assignment default to the principal bucket.
        """
        n = E.shape[0]  # number of words = number of eigenvectors/buckets

        prob = LpProblem("TBuckets", LpMaximize)

        # Decision variables  X[i][j] ∈ {0, 1}
        X = {
            (i, j): LpVariable(f"X_{i}_{j}", cat=LpBinary)
            for i in range(n)
            for j in range(n)
        }

        # ----------------------------------------------------------
        # Objective  (Section 3.1)
        #
        #   max  Σ_i Σ_j  E[i][j] · X[i][j]          ← assignment reward
        #       − Σ_{i=1..n-1} Σ_j  E[0][j] · X[i][j] ← principal penalty
        #
        # First term:  maximise similarity of each word with its assigned
        #              eigenvector (equivalent to SVD-only when optimised alone).
        # Second term: for every word NOT in the principal bucket, subtract
        #              its similarity to the principal eigenvector.  This
        #              favours pulling "borderline" words into bucket 0.
        # ----------------------------------------------------------
        first_term = lpSum(E[i, j] * X[i, j] for i in range(n) for j in range(n))
        penalty_term = lpSum(
            E[0, j] * X[i, j] for i in range(1, n) for j in range(n)
        )
        prob += first_term - penalty_term

        # ----------------------------------------------------------
        # C1: each word assigned to exactly one bucket
        #     ∀j:  Σ_i X[i][j] = 1
        # ----------------------------------------------------------
        for j in range(n):
            prob += lpSum(X[i, j] for i in range(n)) == 1

        # ----------------------------------------------------------
        # C2: at least one word in the principal bucket
        #     Σ_j X[0][j] ≥ 1
        # ----------------------------------------------------------
        prob += lpSum(X[0, j] for j in range(n)) >= 1

        # ----------------------------------------------------------
        # C3: a word j in non-principal bucket i must be more similar
        #     to eigenvector i than to any word k in the principal bucket.
        #
        #     ∀ i∈[1,n-1], ∀ j,k∈[0,n-1], j≠k:
        #       E[i][j] · X[i][j]  ≥  W[j][k] · (X[0][k] − X[0][j] − Σ_{m≠0,m≠i} X[m][j])
        #
        # When j IS in bucket i and k IS in principal, this reduces to:
        #       E[i][j]  ≥  W[j][k]
        # i.e. the word-eigenvector similarity must beat word-word similarity
        # with any principal-bucket member.  All other variable combinations
        # make the RHS ≤ 0, so the constraint is trivially satisfied.
        # ----------------------------------------------------------
        for i in range(1, n):
            for j in range(n):
                for k in range(n):
                    if j == k:
                        continue
                    other_buckets = lpSum(X[m, j] for m in range(1, n) if m != i)
                    rhs = W[j, k] * (X[0, k] - X[0, j] - other_buckets)
                    prob += E[i, j] * X[i, j] >= rhs

        # ----------------------------------------------------------
        # C4: any word in the principal bucket must have the principal
        #     eigenvector as its 1st or 2nd most similar eigenvector.
        #
        #     ∀j:  X[0][j] · (Σ_{i=0}^{n-2} L[i][j])  ≤  1
        #
        #     Σ L[i][j] counts how many non-principal eigenvectors are more
        #     similar to word j than the principal.  Capping at 1 means the
        #     principal eigenvector is at worst the 2nd most similar.
        #     Words with L_sum > 1 are automatically excluded from bucket 0.
        # ----------------------------------------------------------
        for j in range(n):
            L_sum_j = float(np.sum(L[:, j]))
            prob += X[0, j] * L_sum_j <= 1

        # ----------------------------------------------------------
        # C5: in the principal bucket, "type-i" words (principal eigvec is
        #     THE most similar) must outnumber "type-ii" words (principal
        #     is only 2nd most similar).
        #
        #     2 · Σ_j ( X[0][j] · Σ_{i=0}^{n-2} L[i][j] )  ≤  Σ_j X[0][j]
        #
        #     LHS = 2 × (number of type-ii words in principal bucket).
        #     RHS = total words in principal bucket.
        #     → type-ii ≤ type-i  (type-i words in majority).
        # ----------------------------------------------------------
        L_sums = [float(np.sum(L[:, j])) for j in range(n)]
        prob += 2 * lpSum(X[0, j] * L_sums[j] for j in range(n)) <= lpSum(
            X[0, j] for j in range(n)
        )

        # Solve silently
        prob.solve(PULP_CBC_CMD(msg=False))

        # Extract solution; default unset variables to principal bucket
        X_sol = np.zeros((n, n), dtype=int)
        assigned = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                val = X[i, j].varValue
                if val is not None and val > 0.5:
                    X_sol[i, j] = 1
                    assigned[j] = True

        # Fallback: any unassigned word goes to principal bucket
        for j in range(n):
            if not assigned[j]:
                X_sol[0, j] = 1

        return X_sol

    # ------------------------------------------------------------------
    # Per-topic coherence
    # ------------------------------------------------------------------

    @staticmethod
    def _topic_coherence(topic_embeddings: np.ndarray) -> float:
        """
        Full TBuckets pipeline for a single topic.

        Args:
            topic_embeddings: (n_words, emb_dim) array of pre-computed embeddings.

        Returns:
            Coherence score = size of the principal bucket (integer, as float).
            Range: [1, n_words].  Higher = more coherent.
        """
        A = topic_embeddings.astype(np.float64)

        # 1. SVD → eigenvectors (themes)
        eigenvectors = TBuckets._svd_eigenvectors(A)  # (n, d)

        # 2. Build matrices
        E = TBuckets._build_E(A, eigenvectors)  # (n, n)
        W = TBuckets._build_W(A)                # (n, n)
        L = TBuckets._build_L(E)                # (n-1, n)

        # 3. Solve ILP → optimal bucket assignment
        X_sol = TBuckets._solve_ilp(E, W, L)    # (n, n)

        # 4. Score = number of words in principal bucket (row 0)
        score = float(np.sum(X_sol[0, :]))

        return score

    # ------------------------------------------------------------------
    # Pipeline interface
    # ------------------------------------------------------------------

    def predict(self, x: XYData) -> XYData:
        """
        Compute TBuckets coherence for each topic in the batch.

        Args:
            x.value: tensor/array of shape (n_topics, n_words, emb_dim)
                     — the output of GensimEmbedder or TransformersEmbedder.

        Returns:
            XYData with value = 1-D array of shape (n_topics,) containing
            the coherence score (principal bucket size) for each topic.
        """
        import torch

        # Accept both torch tensors and numpy arrays
        if isinstance(x.value, torch.Tensor):
            topic_matrix = x.value.detach().cpu().numpy()
        else:
            topic_matrix = np.array(x.value)

        scores = []
        for t in range(topic_matrix.shape[0]):
            scores.append(self._topic_coherence(topic_matrix[t]))

        return XYData.mock(np.array(scores))