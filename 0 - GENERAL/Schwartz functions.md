Let $\mathcal{S}$ be a set of functions defined on $\mathbb{R}^n$. A function $f$ belongs to the Schwartz space $\mathcal{S}$ if it satisfies the following properties:

1. Smoothness: $f$ is infinitely differentiable, i.e., $f \in C^\infty(\mathbb{R}^n)$.
2. Rapid Decrease at Infinity: For all non-negative integers $m, n$, there exist constants $C_{m,n}$ such that for all $x \in \mathbb{R}^n$,
   $|x|^m \left| \frac{\partial^k}{\partial x_i^k} f(x) \right| \leq C_{m,n}$, where $k = 0, 1, 2, \ldots$ and $i = 1, 2, \ldots, n$.

The Schwartz space $\mathcal{S}$ is a vector space under pointwise addition and scalar multiplication. It is also a topological vector space when equipped with the topology induced by the family of seminorms given by the rapid decrease condition.

Schwartz spaces are important in the theory of distributions and Fourier analysis, providing a framework for defining and manipulating generalized functions.
