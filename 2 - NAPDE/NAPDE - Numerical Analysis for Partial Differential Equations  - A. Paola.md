# Numerical Analysis for Partial Differential Equations
$$
$$
$$
$$
$$
$$
$$
$$
$$
$$
$$
$$
![[Pasted image 20240530173124.png]]

<figcaption>Rayleigh–Bénard convection</figcaption>

$$
$$
$$
$$
$$
$$
$$
$$
$$
$$
$$
$$
###### Held by Prof. Antonietti Paola at Politecnico di Milano 2023/2024
~~Notes by Rayan Emara~~


<div style="page-break-after: always; visibility: hidden">
\pagebreak
</div>

## Table of contents

$$
$$
$$
$$
$$
$$
$$
$$
$$
$$
$$
$$




<div style="page-break-after: always; visibility: hidden">
\pagebreak
</div>

## Disclaimers and preface

These notes were taken during AY 2023/2024 using older material, your mileage may vary. They're meant to accompany the lectures and in no way aim to substitute lectures.

These notes are in part based on material by **Ravizza**
 
 For any questions/mistakes you can reach me [here](mailto:notes@rayanemara.com?subject=NAPDE2324%20Notes%20-%20Problem%20problem%20).

All rights go to their respective owners.


<div style="page-break-after: always; visibility: hidden">
\pagebreak
</div>

## Boundary-value problems

In general, these types of problems are written as: ”Some operator applied to some function u set to some value”.
$$
 \begin{cases}
        \mathcal{L} u=f   & \text{in }\Omega          \\
        \text{BC} & \text{in } \partial\Omega \\
    \end{cases}
$$
Some specific examples include the diffusion problem, where:
$$
\mathcal{L} u = -\text{div}(\mu (x) \nabla u)
$$
Where $\mu (x)$ is the diffusion coefficient which is positive almost everywhere (from here on out a.e.).
There's also the ADR problem where:
$$
\mathcal{L} u = -\text{div}(\mu (x) \nabla u) + \underline{b} \cdot \nabla u + \sigma u
$$
Where $\mu (x)$ is as above, $\underline{b} \in  \mathbb{R}^d$ and $\sigma = \sigma (x) \geq 0$.
In PDE courses you're usually taught to think of these as at least $L^2(\Omega)$ functions but for the purposes of this course
we'll relax the constraints and assume them to be $L^{\infty}(\Omega)$ to simplify the analysis. 
Here, have some notation, this is what we're going to assume:
$$
\begin{cases}
        \mu (x) \in L^\infty(\Omega)            \\
        \sigma (x) \in L^\infty(\Omega)           \\
        \underline{b} \in [ L^\infty(\Omega) ]^d \\
        f \in L^2(\Omega)
    \end{cases}
$$
### Weak formulation

Consider 
$$\begin{cases}\mathcal{L}u=f&\text{in}\Omega\\+\text{B.C.}&\text{on}\partial\Omega\end{cases}$$

where $\Omega$ is an open bounded domain in $\mathbb{R}^d$ where $d$ is the number of dimensions and $\mathcal{L}$ is a $2^{nd}$ order differential operator.
>[!example] Examples of $2^{nd}$ order operators and a general boundary value problem
>The following are respectively a non-conservative and a conservative form
>$$\mathcal{L}u=-\operatorname{div}(\mu\nabla u)+\mathbf{b}\cdot\nabla u+\sigma u$$
>$$\mathcal{L}u=-\operatorname{div}(\mu\nabla u)+\operatorname{div}(\mathbf{b}u)+\sigma u$$
>The following is an example of an applied BVP
>$$
>\begin{cases}\mathcal{L}u=-\operatorname{div}(\mu\nabla u)+\mathbf{b}\cdot\nabla u+\sigma u=f&\text{in }\Omega\\u=0&\text{on }\Gamma_{\mathrm{D}}\\\mu\nabla u\cdot\mathbf{n}=g&\text{on }\Gamma_{\mathrm{N}}\end{cases}$$
>$$g\in L^{2}(\Gamma_{\mathrm{N}}),\quad\partial\Omega=\Gamma_{\mathrm{D}}\cup\Gamma_{\mathrm{N}},\quad\tilde{\Gamma}_{\mathrm{D}}\cap\tilde{\Gamma}_{\mathrm{N}}=\emptyset 
>$$
>![[image-removebg-preview (3).png]]

This is a very general form which isn’t very useful for numerical analysis, it’s better to use some form of weak formulation, this will help us find an integral representation of the problem and derive numerical models from that.
Let's start quick and dirty, ignore the legality of steps, the regularity of $v$ and just try to come up with an integral form, we'll worry about conditions later !
$$
\color {teal}   \int_{\Omega} [\color{black} -\text{div}(\mu (x) \nabla u) + \underline{b} \cdot \nabla u + \sigma u \color{teal} \cdot v ] = \color {teal}   \int_{\Omega} [\color{black} f \color{teal} \cdot v]
$$
Now, integrating by parts we get:
$$
\int_{\Omega} \mu (x) \nabla u \cdot n v - \color{teal}\underbrace{\int_{\partial\Omega}\mu \nabla u \cdot n v}_{\text{notice the dominion}} \color{black} + \int_{\Omega} \underbrace{b}\nabla u v + \int_{\Omega} \sigma u v = \int_{\Omega} f v \qquad \forall v
$$

$$\begin{aligned}\underbrace{\int_\Omega\mu\nabla u\cdot\nabla v+\int_\Omega\mathbf{b}\cdot\nabla uv+\int_\Omega\sigma uv}_{=:z(u,v)}\\&=\int_\Omega fv+\underbrace{\int_{\Gamma_D}\mu\nabla u\cdot\mathbf{n}v}_{=0\text{ if }v|_{\Gamma_D}=0}+\int_{\Gamma_N}\underbrace{\mu\nabla u\cdot\mathbf{n}}_{=g}v\end{aligned}$$

whence

>[!definition] Abstract weak formulation
> Find $u \in V$ such that
> $$
> a(u,v)=F(v)\quad\forall v\in V
> $$
> where $a:V\times V \to \mathbb{R}$ is a bilinear form and $F: V \to \mathbb{R}$ is a linear form $\langle F,v\rangle\equiv F(v)=\int_\Omega f v+\int_{\Gamma_N}g v$

>[!theorem] Lax-Milgram Lemma
>The following theorem provides sufficient conditions for the existence and uniqueness of a solution to a weakly formulated problem.
>Let:
>- $V$ be a Hilbert space with norm $||\cdot||_V$ and inner product $\left( \cdot , \cdot \right)$
>- $F\in V^{\prime}\colon|F(v)|\leq\|F\|_{V^{\prime}}\|v\|_{V}\forall v\in V$
>- The bilinear form $a$ is **continuous** meaning:
>  $$
>  \exists M>0\colon|a(u,v)|\leq M\|u\|_V\|v\|_V\forall u,v\in V
>  $$
>- The bilinear form $a$ is **coercive** meaning:
>  $$
>  \exists\alpha>0\colon a(v,v)\geq\alpha\|v\|_{V}^{2}\quad\forall v\in V
>  $$
>
>Then there exists a unique solution $u$ for the abstract weak formulation.
>
>More over
>$$
>\alpha\|u\|_V^2\leq a(u,u)=F(u)\leq\|F\|_{V'}\|u\|_V
>$$
>therefore
>$$\|u\|_V\leq\frac{\|F\|_{V'}}{\alpha}$$
which is a stability result that implies the continuous dependence of the solution from the data.

Note that I'm skipping the examples given by prof. Antonietti as they're already well described in the slides.

>[!theorem] Poincarè inequality
>Let $\Gamma_D$ be a set of positive measure (in 1D it is sufficient that it contains a single point) then:
>$$\exists C_{\mathbb{P}}>0\colon\|v\|_{L^2(\Omega)}\leq C_{\mathbb{P}}\|\nabla v\|_{L^2(\Omega)}\quad\forall v\in V=H_{\Gamma_{\mathbb{D}}}^1(\Omega)$$
>
>which can be rewritten in terms of the $L^2(\Omega)$ norm starting from
>$$
>\|v\|_V^2=\|v\|_{L^2(\Omega)}^2+\|\nabla v\|_{L^2(\Omega)}^2\leq\left(1+C_P^2\right)\|\nabla v\|_{L^2(\Omega)}^2
>$$
>therefore
>$$\|\nabla v\|_{L^2(\Omega)}^2\geq\left(1+C_{\mathrm{P}}^2\right)^{-1}\|v\|_V^2$$

### Galerkin approximation

We define $V_h$ to be any finite dimensional subspace of $V$ where $h>0$. 
We've seen how we can apply the Lax-Milgram lemma in order to find solutions for weakly formulated abstract problems. Given that $V_h \subset V$ we can conclude that if any problem of type
$$
\text{Find }u\in V:a(u,v)=F(v)\quad\forall v\in V
$$
has a solution then a problem of the following type will also have a solution
$$
\text{Find }u_h\in V_h\colon a(u_h,v_h)=F(v_h)\quad\forall v_h\in V_h
$$
where the $\dim(V_h) = N_h < +\infty$.
We'll denote the second problem as the $(G)$ or Galerkin formulation problem.
>[!theorem] Linear system equivalent for the galerkin problem
>Problem $(G)$ is equivalent to the following linear system of equations:
>$$
>\mathrm{Find~}\boldsymbol{u}\in\mathbb{R}^{N_h}\mathrm{~s.t.}\quad A\boldsymbol{u}=\boldsymbol{F}
>$$
>where $A\in\mathbb{R}^{N_{h}\times N_{h}}$ and $F\in\mathbb{R}^{N_{h}}$
>$$
>A =\begin{pmatrix}\dots&\ldots&\dots\\a_{i1}&\ldots&a_{iN_h}\\\ldots&\ldots&\ldots\end{pmatrix}
>$$

Now. keep in mind that each element of $A$ is an [[NAPDE - Numerical Analysis for Partial Differential Equations  - A. Paola#Weak formulation|integral]] and as such, before we feed this linear system into our laptop, we need to approximate those integrals numerically.

Another important consideration is that $u_h$ is a projection of $u\in V$ in $V_h$, the assumption that $u_h$ converges to $u$, as $V_h$ gets larger and larger, is called **space saturation**. We'll be making use of space saturation for our three step solution finding method. ^4cf093
- Try to write your problem as an abstract problem in order to apply *Lax-Milgram*. 
- Use the Galerkin paradigm.
- Choose a subspace that satisfies *space saturation* to automatically get convergence for our method.

Finally we'll write elements of $V_h$ as linear combinations of basis functions for $V_h$
$$
\big\{\phi_i(\mathbf{z})\big\}_{j=1}^{N_h}
$$
there any $v_h\in V_h$ can be expanded as a linear combination of the basis in the following fashion
$$
v_h(\mathbf{x})=\sum_{j=1}^{N_h}v_j \phi_j(\mathbf{x})
$$
where $v_j$ are the coefficients that identify $v_h$. This means that finding $u_h$ in the galerkin paradigm actually means finding the **coefficients** that identify $u_h$ *given* the basis functions (we'll find out later why the choice of basis functions is actually very important for numerical reasons).

##### Analysis of the Galerkin method
- **Existence and uniqueness** is a consequence of the Lax-Milgram lemma given that $V_h \subset V$
- **Stability** is also a consequence of the Lax-Milgram lemma from which we a uniform bound with respect to $h$
  $$\|u_h\|_V\leq\frac{\|F\|_{V^{\prime}}}{\alpha}$$
- **Consistency (or Galerkin orthogonality)** is basically measuring how well we're projecting the actual solution in $V_h$, for which we have the following result
  $$a(u-u_h\ , \ v_h) = 0 \quad \forall v_h\in V_h$$
  which can be proven by computing difference between $a(\cdot,\cdot)$ both in *weak-formulation* and *galerkin form* and using $v=v_h$
- **Convergence (Cèa Lemma)**:
  $$\begin{aligned}
\alpha\|u-u_{h}\|_{V}^{2}& \leq a(u-u_h,u-u_h) \\
&=a(u-u_h,u-v_h)+\underbrace{a(u-u_h,v_h-u_h)}_{=0\text{(Galerkin orthogonality)}} \\
&\leq M\|u-u_h\|_V\|u-v_h\|_V\quad\forall v_h\in V_h
\end{aligned}$$
$$\|u-u_h\|_V\leq\frac M\alpha\|u-v_h\|_V\quad\forall v_h\in V_h$$
$$
  \|u-u_h\|_V\leq\frac{M}{\alpha}\inf_{v_h\in V_h}\|u-v_h\|_V$$
which is the best approximation we could hope for.

As previously mentioned, one of the assumptions we'll make on $V$ is that it satisfies **space saturation**, in other words that
$$
V_h \longrightarrow V \quad \text{when } h \longrightarrow 0
$$
which implies 
$$
\forall v\in V\quad\lim_{h\to0}\inf_{v_h\in V_h}\|v-v_h\|_V=0
$$

### The Finite Element Method 

As in other numerical schemes, we'll try to find a **tessellation** to break up our domain space. Then we'll construct a finite dimensional space made by piece-wise polynomials in $H^1_0$.

>[!definition]
>We'll call $\mathcal{T}_h=\bigcup K$ a triangulation of our discrete space $\Omega_h$,  for any $r\geq1$
>$$V_h=\{v_h\in\mathcal{C}^0(\overline{\Omega})\colon v_h|_K\in\mathbb{P}^r(K)\forall K\in\mathcal{T}_h,v_h|_{\Gamma_0}=0\}$$

So essentially each element $K$ is comprised of a continuous (up to and including the boundary) polynomial of degree $r$ or less, that go to zero at some boundary $\Gamma_0$.
We can prove that for a suitable choice of an interpolant 
$$
\inf_{v_h\in V_h}\|u-v_h\|_V\leq\|u-\overline{u}_h\|_V
$$
we'll later see that if we take $\overline{u}_h=\Pi_h^ru$ then we get space saturation
$$
\|u-\overline{u}_h\|\leq Ch^r|u|_{H^{r+1}(\Omega)}
$$

##### Properties of basis/shape functions

We'll use some specific notation
- We'll use $v = (v_1 , \dots, v_{N_h})^T$ to denote a real vector containing all the basis coefficients (also called **degrees of freedom**)
- A basis is called **Lagrangian** if it satisifes the following property
  $$\phi_i(\mathbf{x}_j)=\delta_{ij}$$
  for a *suitable* collection of points called **nodes**
- When the basis is Lagrangian the following property holds
  $$v_h(\mathbf{x}_i)=v_i\quad\forall 1\leq j\leq N_h$$

Suppose I have a generic function $v \in V$ of which I want to compute the finite element interpolant $\prod_h^1v \in V_h$, we'll restrict ourselves to $r=1$ for now. 
By definition this object has to be a piece-wise continuous linear polynomial over my mesh so basically
- $\phi_i(x)$ is a piece-wise continuous polynomial such that
  $$
  \phi_i(\underline{x}_i)= 1 \mathrm{~and~} \phi_i(\underline{x}_{j\neq i})= 1
  $$

We can then compute our **interpolant** (essentially the approximated version of our function) as the dot product between the coefficients and our basis functions
$$
\prod_h^1 v(x) = \sum_{i=1}^{N_h} v(x_i)\phi_i(x)
$$

>[!col]
>Note that we set the basis function nodes to be zero on the boundary in order to comply with the boundary conditions, therefore we'd have something like $v_h(a) = v_h(b) =0$ in the 1D case.
>
>![[Pasted image 20240603160023.png]]



![[Pasted image 20240603144049.png]]

<figcaption>Note how overlapping between the basis functions is not allowed</figcaption> 

A small (not in importance) note has to be made about the space $V$ in which we're operating
$$
V_h \subset V = H^1_0
$$
This is intuitively because $V$ is comprised of functions that are (along with their gradients) $L^2$ while $V_h$ is comprised of piece-wise polynomials that are trivially integrable on bounded intervals, the gradients of $V_h$ are also integrable if we assume to $glue$ them without jumps.


##### Error estimate and analysis

>[!theorem] Bounds for the interpolation error
> Let $r\geq 1$, $v \in H^{r+1}(\Omega)$, and $\prod_h^r$ to be the finite element interpolant of $v$ at the finite element nodes, meaning
> -  $\Pi_h^rv\in X_h^r$
> - $\Pi_h^r(\mathbf{x}_i)=v(\mathbf{x}_i)\quad\forall\text{ node }\mathbf{x}_i\text{ of }T_h$
> ![[Pasted_image_20240603171049-removebg-preview.png|350]]
> 
> Then, for $m=0,1$, $\exists C=C(r,m,\hat{k})$ such that:
> $$
> |v-\Pi_h^rv|_{H^m(\Omega)}\leq C\left(\sum_{K\in\mathcal{T}_h}h_K^{2(r+1-m)}|v|_{H^{r+1}(K)}^2\right)^{1/2} \tag{8}
> $$
> 
> The constant then depends on $r$, the norm and the shape of the triangle. $h_K = \mathrm{diam}(K)$ and since $h_K \leq h$ then $\forall K$ the following holds
> $$
> |v-\Pi_h^rv|_{H^m(\Omega)}\leq C h^{r+1-m}|v|_{H^{r+1}(K)}\quad\forall v\in H^{r+1}(\Omega),m=0,1 \tag{9}
> $$ 
> 
> In essence $(8)$ gives a localized element-wise interpolation error while equation $(9)$ global bound

So know that the interpolation has an inherent error that thankfully goes to zero as $h$ goes to zero. We can then try to get an estimate for $\| u - u_h \|_V$
$$
\begin{aligned}
\|u-u_{h}\|_{V}& =\|u-u_{h}\|_{H^{1}(\Omega)} \\
&\leq\frac M\alpha\inf_{v_h\in V_h}\|u-v_h\|_{H^1(\Omega)} \\
&\leq\frac M\alpha\|u-\Pi_h^ru\|_{H^1(\Omega)}
\end{aligned}
$$

where we can then use $(8)$ and then $(9)$ to get
$$
\|u-u_h\|_V\leq C\frac M\alpha\left(\sum_{K\in\mathcal{T}_h}h_K^{2r}|u|_{H^{r+1}(\Omega)}^2\right)^{1/2} \tag{using 8}
$$
$$
\|u-u_h\|_V\leq C\frac{M}{\alpha}h^r|u|_{H^{r+1}(\Omega)} \tag{using 9}
$$

Remember that we're assuming quite a bit of regularity, let $s$ be the regularity for $u \in H$ then we have the following error estimates for $u_H$, remember that we're working on the *ADR* problem, we're therefore using $V=H^1$

| $_r \backslash ^s$   | $s <2$ | $s=2$| $s=3$| $s=4$ | 
| --- | --- | ---| ---| ---| 
|$r=1$ |  $N/A$   |$h^1$ |$h^2$ |$h^3$ | 
|$r=2$ |  $N/A$   |$h^1$ |$h^2$ |$h^3$ | 
|$r=3$ |  $N/A$   |$h^1$ |$h^2$ |$h^3$ |


<figcaption>Convergence rates</figcaption> 

##### Error estimates in the $L^2$ norm

We start by defining the **adjoint form** starting from a bilinear form
>[!definition] Adjoint form
>Consider a bilinear form $a:V\times V \to \mathbb{R}$, the adjoint form $a^*$ is defined as 
>$$
>\begin{aligned}&a^{*}\colon V\times V\to\mathbb{R}\\&a^{*}(v,w)=a(w,v)\quad\forall v,w\in V\end{aligned}
>$$
>
>It immediately follows that if $a$ is symmetric then
>$$ a^* = a$$

An adjoint problem can be given in the following form
$$
\begin{cases}&\text{Find} \  \phi=\phi(g)\in V\\&a^*(\phi,v)=(g,v)=\int_\Omega g v\quad\forall v\in V\end{cases} \tag{12}
$$

>[!example] Example
>Consider $\mathcal{L} = -\Delta$, then the solution of the Poisson problem
>$$
>\begin{cases}-\Delta\phi=g&\text{in }\Omega\\u=0&\text{on }\partial\Omega\end{cases}
>$$
>satisfies $\phi \in H^2 (\Omega)$, moreover
>$$
>\exists C_1>0:\quad\|\phi(g)\|_{H^2(\Omega)}\leq C_1\|g\|_{L^2(\Omega)} \tag{13}
>$$
>
>This is a direct result of the fact that this $a$ is symmetric in this case, therefore we can apply Lax-Milgram and conclude that there exists a unique $\phi \in V$ that continuously depends on the data.

We will now take $g$ to be our error and do some magic$^\mathrm{~TM~}$ to get an error estimate in the $L^2$ norm.
Taking $g=e_h=u-u_h$ in $(12)$
$$
\begin{aligned}
\|e_{h}\|_{L^{2}(\Omega)}^{2}& =(e_h,e_h)=a^*(\phi,e_h)\underbrace{=}_{\mathrm{~by ~symmetry~}}a(e_h,\phi) \\
&=a(e_h,\phi-\phi_h)\quad\text{(Galerkin orthogonality, for }\phi_h\in V_h) \\
&\leq M\|e_{h}\|_{H^{1}(\Omega)}\|\phi-\phi_{h}\|_{H^{1}(\Omega)}
\end{aligned}
$$

The second row is a consequence of the fact that the error $u-u_h$ is orthogonal to anything in the discrete space [[NAPDE - Numerical Analysis for Partial Differential Equations  - A. Paola#^4cf093|since it is a projection]], we then use the continuity of the bilinear form and the fact that $V=H^1$ to get the inequality.
We assume $\phi \in H^2 (\Omega) \cap V$, this is also referred to as **elliptic regularity**, we also take $\phi_h = \prod_h^1 \phi$, then
$$
\begin{aligned}
\|e_{h}\|_{L^{2}(\Omega)}^{2}& \leq M\|e_{h}\|_{H^{1}(\Omega)}\|\phi-\Pi_{h}^{1}\phi\|_{H^{1}(\Omega)} \\
&\leq M\|e_h\|_{H^1(\Omega)}C_2h|\phi|_{H^2(\Omega)}\quad\mathrm{(for~(9)~with~}m=r=1) \\
&\leq M\|e_h\|_{H^1(\Omega)}C_2hC_1\|e_h\|_{L^2(\Omega)}\quad\mathrm{(for~(13))}
\end{aligned}
$$

then
$$
\begin{aligned}\|e_{h}\|_{L^{2}(\Omega)}&\leq M C_{1} C_{2} h\|e_{h}\|_{H^{1}(\Omega)}\\&\leq M C_{1} C_{2} h C_{3}h^{r} | u|_{H^{r+1}(\Omega)}&\text{(for (11))}\end{aligned}
$$

in conclusion
$$
\|e_h\|_{L^2(\Omega)}\leq\overline{C}h^{r+1}|u|_{H^{r+1}(\Omega)}
$$

In essence if the solution is regular enough (in $r=1$) we get linear convergence in the energy norm but quadratic convergence in the $L^2$ norm. If the domain is convex the solution is in $H^2$ and the $H^2$ is bounded by the $L^2$ norm.
Finally it can be proven that if elliptic regularity isn't satisfied then the $L^2$ norm doesn't gain an order of convergence.

##### Stiffness matrix

Let's take the Poisson problem
$$
\begin{cases}-\Delta\phi=f&\text{in }\Omega\\u=0&\text{on }\Gamma_D
\end{cases}
$$

take $V= H^1_0(\Omega)$ the weak formulation then becomes

$$\int_\Omega \nabla u\cdot\nabla v = \int_{\Omega} fv \quad \forall \ v \in V$$

let
- Space be two dimensional $d =2$
- Linear polynomials $r=1$
- Basis functions be $\phi_i$,  also referred to as hat functions (associated to a vertex $i$)

recall that 
- $\underline{u} \in \mathbb{R}^{N_h}$ is defined as 
  $$
  \underline{u}= [u_1, \dots, u_{N_h}]^T
  $$
- $A$ is a real valued $N_h\times N_h$ matrix $$A_{i,j} = a(\phi_j, \phi_i) \quad \forall \ i,j=1,\dots,N_h$$
- $\underline{F}$ is a real valued $N_h$ long vector defined as 
  $$
  F_i = F(\phi_i) \quad \forall \ i = 1 , \dots , N_h  $$
>[!col]
>We're going to define a special element called a **reference element** with a variable substitution on which we construct our shape functions.
>The reference element is nothing but a  equilateral triangle with coordinates $(0,0) , (0,1) , (1,0)$
>
>```tikz
>\begin{document}
>
>\begin{tikzpicture}[>=stealth, line cap=round, line join=round]
>
>    % Draw the triangle with the scaled coordinates
>    \draw[->, thick] (0,0) coordinate[label=below:\( \hat{V}_1 \)] (v1) -- 
>                     (2.5,0) coordinate[label={[above] \( \hat{V}_2 \)}] (v2);
>    \draw[->, thick] (0,0) -- (0,2.5) coordinate[label=right:\( \hat{V}_3 \)] (v3);
>    \draw[thick] (v2) -- (v3);
>    
>    % Draw the points
>    \fill[black] (v1) circle (2pt);
>    \fill[black] (v2) circle (2pt);
>    \fill[black] (v3) circle (2pt);
>    
>    % Add axis names
>    \node[below] at (2.5,-0.1) {\( \xi \)};
>    \node[left] at (-0.1,2.5) {\( \eta \)};
>
>\end{tikzpicture}
>
>\end{document}
>```tikz

The idea is to compute the integrals for the stiffness matrix once on the reference element and map them to each triangle afterwards, saving a lot of compute.
We'll call the reference triangle $\hat{K}$.
Let's now define the space of piece-wise linear polynomials (hat/basis/shape functions)$\mathcal{P}^1(K)$, we can write any function in $\mathcal{P}^1(K)$ in the form of 
$$
C_1 + C_2\xi + C_3 \eta
$$

in order to force the hat functions $\hat{\phi_i}$ to be equal to 1 only on their respective node $\hat{V_i}$ we'll impose
$$
\begin{cases} 
 \hat{\phi}_1(\xi,\eta) = 1 - \xi . \eta \\    
 \hat{\phi}_2(\xi,\eta) = \xi \\
 \hat{\phi}_3(\xi,\eta) = \eta \\
\end{cases}
$$

<figcaption>vertices are enumerated in anti clock-wise order for stability reasons</figcaption> 

At this point we can map these hat functions into our mesh (triangulation) for any triangle $K$, in order to do this we can employ a linear transformation that can be represented by a matrix 
$$
\begin{aligned}F_{k}:\hat{K}&\to K\\\begin{bmatrix}\xi\\\eta\end{bmatrix}&\to\begin{bmatrix}x\\y\end{bmatrix}=B_K\begin{bmatrix}\xi\\\eta\end{bmatrix}+b_K\end{aligned}
$$

where the $B_K$ and $b_K$ all depend on the physical vertices, in this specific example we have
- 
  $$
B_k \in \mathbb{R}^{2\times2} \qquad B_k \begin{bmatrix} {x_2 - x_1}&{x_3-x_1} \\ {y_2 - y_1}&{y_3-y_1} \end{bmatrix}
$$
- 
  $$
 b_k \in \mathbb{R}^2 \qquad b_k = \begin{bmatrix}{x_1}\\{y_1} \end{bmatrix} 
 $$

We'll have to invert this map later on, this is the reason why this construction won't work easily for quadrilaterals as they'd need a bilinear map in place of a linear one and those are harder to invert. We then compute the derivatives of the hat functions as follows.

$$
\begin{aligned}&\widehat{\phi}_{1}\left(\xi,\eta\right)=1-\xi-\eta&&\widehat{\nabla}\widehat{\phi}_{1}=\left(-1,1\right)\\&\widehat{\phi}_{2}\left(\xi,\eta\right)=\xi&&\widehat{\nabla}\widehat{\phi}_{2}=\left(1,0\right)\\&\widehat{\phi}_{3}\left(\xi,\eta\right)=\eta&&\widehat{\nabla}\widehat{\phi}_{3}=\left(0,1\right)\end{aligned}
$$

it can be shown that you can write a map from the reference gradients to the physical gradients as follows

$$
\begin{aligned}&\nabla\phi_{i}-B_{K}^{-T}\widehat{\nabla\phi}_{i}\\&\widehat{\nabla\phi}_{i}=B_{K}^{T}\nabla\phi_{i}\end{aligned}
$$

>[!theorem] 
>If $A$ is **spd** then 
>$$
> K_2(A)= \frac{\lambda_{\mathrm{max}(A)}}{\lambda_{\mathrm{min}(A)}}
>$$
>
>conversely if the bilinear form $a(\cdot, \cdot)$ is symmetric and coercive then $A$ is spd.
>>[!theorem] Proof
>>To be written down !

>[!definition] $A$-norm
>Let $A$  be spd, we define the $A$-norm of $\mathbf{v}$ as
>$$
>\begin{aligned}
\|\mathbf{v}\|_{A}& :=(A\mathbf{v},\mathbf{v})^{1/2} \\
&=\left(\sum_{i,j}a_{ij} v_i v_j\right)^{1/2}
\end{aligned}
>$$
>

We can prove that $\exists C_1 , C_2 >0 \colon \ \forall \lambda_h$ eigenvalue of $A$:
$$
\alpha C_1 h^d\leq\lambda_h\leq M C_2 h^{d-2}\quad d=1,2,3
$$
which we can fold to get
$$
\frac{\lambda_{\max}(A)}{\lambda_{\min}(A)}\leq\frac{MC_2}{\alpha C_1}h^{-2}
$$
in other words the estimate for the conditioning becomes
$$
K_{2}(A)=\mathcal{O}(h^{-2})
$$
which means that if we use the conjugate gradient method to solve $A \mathbf{u} = \mathbf{f}$ then
$$
\|\mathbf{u}^{(k)}-\mathbf{u}\|_A\leq2\left(\frac{\sqrt{K_2(A)}-1}{\sqrt{K_2(A)}+1}\right)^k\|\mathbf{u}^{(0)}-\mathbf{u}\|_A
$$

### Implementation of the Finite Element Method 

Consider the following homogeneous **Poisson** problem in $d=2$ and $r=1$ (linear two-dimensional case).
$$
\begin{cases} 
 -\Delta u &= f && \mathrm{~in~} \Omega \\
 u &=0 && \mathrm{~in~} \Gamma_D   \equiv \partial \Omega
  \end{cases}
$$

The weak formulation for this problem becomes finding $u \in V = H^1_0$ such that
$$
\int_\Omega\nabla u\cdot\nabla\nu=\int_\Omega f\nu\quad\forall\nu\in V
$$

alternatively noted as 
$$
a(u,v) = F(v) \quad \forall v \in V
$$

on domain $\Omega$ we define the infrastructure we need to implement **fem** which is:


>[!col]
>- Triangulation of elements $K$ with vertices $v_j$
>- A $N_h$ dimensional space $V_h$ of continuous linear functions within each $K$
>- On each vertex a shape/basis/hat function 
>  $$
>  \phi_i \in V_h , \quad \phi_i(v_j) = \delta_{i,j} \quad \forall \ 1 \leq i, \ j \leq N_h
>  $$
>
>![[Pasted image 20240605143504.png]]

We can then write our solution $u_h$ as a linear combination of the basis $V_h$
$$
u_h(x)=\sum_{j=1}^{N_h}u_j\phi_j(x)
$$

we can now write the problem as 
$$
\sum_{j=1}^{N_h}u_j\int_\Omega\nabla\phi_j\cdot\nabla\phi_i=\int_\Omega f\phi_i\quad\forall i=1,\ldots,N_h
$$
which is also known as the **finite element formulation**.
The above can also be rewritten in an **algebraic form** recalling what we said about the stiffness matrix
$$
A \mathbf{u}= \mathbf{F}
$$
where
$$
A_{i,j}=a(\phi_{j},\phi_{i})=\int_{\Omega}\nabla\phi_{j}\cdot\nabla\phi_{i}\quad\forall i,j=1,\ldots,N_{h}
$$
$$
F_{i}=F(\phi_{i})=\int_{\Omega}f\phi_{i}\quad\forall i=1,\ldots,N_{h}
$$
$$
u = \begin{bmatrix} 
   {u_1}\\{u_2}\\ {\vdots} \\ {u_{N_h}}
  \end{bmatrix}
$$

But remember, we're not *actually* going to compute this system on our physical system of coordinates. The only thing we really want to compute is the mapping from the reference system into our physical one.

![[Pasted image 20240605151229.png|400]]
as previously mentioned the affine mapping from $(\xi,\eta)$ onto $(x,y)$ is 

$$
\begin{gathered}\begin{pmatrix}x\\y\end{pmatrix}=B_k\begin{pmatrix}\xi\\\eta\end{pmatrix}+\boldsymbol{b}_K\\B_K=\begin{pmatrix}x_2-x_1&x_3-x_1\\y_2-y_1&y_3-y_1\end{pmatrix}\quad b_K=\begin{pmatrix}x_1\\y_1\end{pmatrix}\end{gathered}
$$

## Discontinuous Galerkin Methods for dffusion problems

