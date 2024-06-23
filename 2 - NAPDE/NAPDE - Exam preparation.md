# Exam preparation

## Discontinuous Galerkin F.E.M.


We focus on the approximation of Poisson problem, stability and convergence properties with respect to  the **DG** norm (no proofs).
We start from the model problem 
$$\begin{cases}-\Delta u=f&\text{in }\Omega\\\quad u=0&\text{on }\partial\Omega\end{cases}$$
we consider the approximation as done on triangles
$K$ like 

![[Pasted image 20240608165518.png|250]]


<figcaption>Triangulation of the domain</figcaption> 

The weak formulation is obtained by multiplying both sides of the partial differential equation by a smooth test function $v$ on each triangle $K$
$$
\int_K-\Delta uv=\int_Kfv
$$
we integrate by parts and sum over all $K$
$$
\sum_{K\in T_h}\int_K\nabla u\cdot\nabla v-\boxed{\sum_{K\in T_h}\int_{\partial K}\nabla u\cdot\mathbf{n}_Kv}=\int_\Omega fv
$$
to deal with the directional derivative some notation is introduced, after some steps we end up with the following **DG** formulation.
Find $u_h \in V^r_h$ such that 
$$\mathcal{A}(u_h,v_h) = \int_\Omega fv_h \quad \forall v_h \in V_h^r$$
where
$$
\begin{aligned}
\mathcal{A}(w,v)
&=
\underbrace{\sum_{K\in T_{h}}\int_{K}\nabla w\cdot\nabla v}_{\mathrm{Volume~integral}}
-
\underbrace{\sum_{F\in\mathcal{F}_{h}}\int_{F}\{\nabla_{h}w\}\cdot[v]}_{\mathrm{Flux~term~ on~faces}}
\\
&-
\underbrace{\sum_{F\in\mathcal{F}_{h}}\int_{F}[w]\cdot\{\{\nabla_{h}v\}\}}_{\mathrm{Symmetric~term}}
+
\underbrace{\sum_{F\in\mathcal{F}_{h}}\int_{F}\gamma[w]\cdot[v]}_{\mathrm{Stabilization~term}}
\end{aligned}
$$
and where
- $\mathcal{T}_h$ is the triangulation of the domain $\Omega$ into finite elements $K$
- The average operator $\{\!\!\{\cdot\}\!\!\}$ computes the vector average between the two sides for all the faces
- The jump operator $[\![\cdot]\!]$ computes the discontinuity across faces.
- $\nabla_h$ is the discrete gradient operator 
- $V_h^r$ is the finite element space consisting of piecewise polynomial functions of degree $r$, which are allowed to be discontinuous across element boundaries.

finally we have the interior penalty forms

$$
\begin{aligned}\mathcal{A}(w,v)&=\sum_{K\in T_{h}}\int_{K}\nabla w\cdot\nabla v-\sum_{F\in\mathcal{F}_{h}}\int_{F}\{\{\nabla_{h}w\}\}\cdot[v]\\&-\theta\sum_{F\in\mathcal{F}_{h}}\int_{F}[w]\cdot\{\{\nabla_{h}v\}\}+\sum_{F\in\mathcal{F}_{h}}\int_{F}\gamma[w]\cdot[v]\end{aligned}
$$
where
- $\theta=1$ : **Symmetric** interior penalty
- $\theta=-1$ : **Non-Symmetric** interior penalty 
- $\theta=0$ : **Incomplete** interior penalty

##### Non standard boundary conditions

In the case of **Non-homogeneous Dirichlet** boundary conditions we modify the right hand side in order to apply the boundary condition
$$
u = g_D \quad \mathrm{~on~} \partial \Omega
$$
the interior penalty formulation of the bilinear form becomes
$$
\mathcal{A}(w,v) = \int_\Omega fv-\theta\sum_{F\in\mathcal{F}_h^B}\int_Fg_D\nabla_hv\cdot\mathbf{n}+\sum_{F\in\mathcal{F}_h^B}\int_F\gamma g_Dv
$$

In the case of **Neumann** boundary conditions of the form
$$
\nabla u \cdot \mathbf{n} = g_N \quad \mathrm{~on~} \partial \Omega
$$

the bilinear form has to be modified as 
$$
\begin{aligned}\mathcal{A}(w,v)&=\sum_{K\in T_{h}}\int_{K}\nabla w\cdot\nabla v-\sum_{F\in\mathcal{F}_{h}^{\prime}}\int_{F}\{\nabla_{h}w\}\cdot[v]\\&-\theta\sum_{F\in\mathcal{F}_{h}^{\prime}}\int_{F}[w]\cdot\{\{\nabla_{h}v\}\}+\sum_{F\in\mathcal{F}_{h}^{\prime}}\int_{F}\gamma[w]\cdot[v]\end{aligned}
$$

and the r.h.s has to be modifies as 
$$
\int_\Omega fv+\sum_{F\in\mathcal{F}_h^B}\int_Fg_Nv
$$

## Spectral element methods

Given $f$ find $u \colon \Omega \subset \mathbb{R}^d$ such that
$$
\left\{\begin{matrix}-\Delta u=f&\text{in }\Omega\\u=0&\text{on }\partial\Omega\end{matrix}\right.
$$
in weak form this becomes, given $f \in L^2(\Omega)$ find $u \in V = H^1_0(\Omega)$ such that
$$
a(u,v)=F(v)\quad\forall v\in V
$$
$$
a(u,v)=\int_{\Omega}\nabla u\cdot\nabla v,\quad F(v)=\int_{\Omega}fv
$$

### Galerkin s.e.m.
Let $\hat{K}$ be the reference element $\hat{K} = (-1,1)^d$, for any element $K \in \mathcal{T}_h$ (the mesh) , there exists a (bijective and differentiable) mapping
$$
F_k \ \colon \ \hat{K}\to K
$$

The problem becomes 
$$
\text{Find }u_{h}\in V_{h}^{p}:a\left(u_{h},\nu_{h}\right)=F\left(\nu_{h}\right)\quad\forall\nu_{h}\in V_{h}^{p}
$$
where
- $p \geq 1$ integer and $\mathbb{Q}^p$ is the space of polynomials of degree $\leq p$ w.r.t each variable
- $X_{h}^{p}=\left\{\nu_{h}\in C^{0}(\overline{\Omega}) :\nu_{h}|_{K}=\widehat{\nu}\circ F_{K}^{-1}\mathrm{~with~} \hat{\nu}\in\mathbb{Q}^{p}( \widehat{K} ) \forall K\in\mathcal{T}_{h}\right\}$
- $V_h^p = X_h^p \cap V$ (To account for the homogeneous boundary conditions)

### Interpolation estimates

Let $v \in H^{s+1}(\Omega)$, $s\geq 0$ then there exists an interpolant $\prod_h^p v$ such that
$$
\begin{aligned}&\|\nu-\Pi_{h}^{p}\|_{H^{1}(\Omega)}\leq C_{s}\frac{h^{\min(s,p)}}{p^{s}}\|\nu\|_{H^{s+1}(\Omega)}\\&\|\nu-\Pi_{h}^{p}\|_{L^{2}(\Omega)}\leq C_{s}\frac{h^{\min(s,p)+1}}{p^{s+1}}\|\nu\|_{H^{s+1}(\Omega)}\end{aligned}
$$

### Error estimates
Let $u$ be the solution of the weak formulation and let $u_h$ be the approximate solution with the SEM. Assume that $u \in H^{s+1}(\Omega)$, then
$$
\|u-u_{h}\|_{V}\leq C_{s}\frac{h^{\min(s,p)}}{p^{s}}\|u\|_{H^{s+1}(\Omega)}
$$
Moreover, if $u$ is in analytic form
$$
\|u-u_{h}\|_{V}\lesssim\exp(-\gamma p)
$$
where $\gamma$ depends on $u$

### SEM-NI
The idea is to use GLL quadrature to switch out the integrals for numerical integration, the mass matrix then becomes **diagonal** which simplifies the implementation and reduces computation cost.
Let $u$ be the solution of the weak formulation and let $u_h$ be the approximate solution with the $SEM-NI$ if we assume $u \in H^{s+1}(\Omega)$ then
$$
\|u-u_h\|_V\leq C_s\frac{h^{\min(s,p)}}{p^s}(\|f\|_{H^s(\Omega)}+\|u\|_{H^{s+1}(\Omega)})
$$


## Heat equation
Consider the following problem
$$
\begin{cases} 
   \begin{aligned}\frac{\partial u}{\partial t}-\Delta u&=f,\quad&\mathbf{x}\in\Omega,&t>0,\end{aligned} \\
   u(\mathbf{x},0)=u_0(\mathbf{x}),\quad\mathbf{x}\in\Omega \\
   \mathrm{Boundary~condition}
\end{cases}
$$
where the boundary condition can take one of two forms
- **Dirichlet**:
$$
  u(\mathbf{x},t)=g_D(\mathbf{x},t),\quad\mathbf{x}\in\Gamma_D\mathrm{~and~}t>0,
$$
- **Neumann**:
$$
  \frac{\partial u(\mathbf{x},t)}{\partial n}=g_N(\mathbf{x},t),\quad\mathbf{x}\in\Gamma_N\mathrm{~and~}t>0,
$$

##### Stability 
Suppose that the data are regular enough. Then, the following a priori estimates hold for the exact solution
$$
\|u(t)\|_{L^2(\Omega)}^2+\alpha\int_0^t\|\nabla u(s)\|_{L^2(\Omega)}^2ds\leq\|u_0\|_{L^2(\Omega)}^2+\frac{C_\Omega^2}\alpha\int_0^t\|f(s)\|_{L^2(\Omega)}^2ds
$$
where $C_{\Omega}$ is the Poincare inequality constant and $\alpha$ is the coercivity constant of $a(\cdot,\cdot)$
>[!theorem] Proof
>Let us consider problem (4); since the corresponding equations must
hold for each $v\in V$, it will be legitimate to set $v=u(t)$ $(t$ being
given), solution of the problem itself, yielding
$$\int_\Omega\frac{\partial u(t)}{\partial t}u(t)\mathrm{~}d\Omega+a(u(t),u(t))=\int_\Omega f(t)u(t)\mathrm{~}d\Omega\mathrm{~}\quad\forall t>0.$$
Considering the individual terms, we have
$$\int_\Omega\frac{\partial u(t)}{\partial t}u(t)\mathrm{~}d\Omega=\frac12\frac\partial{\partial t}\int_\Omega|u(t)|^2d\Omega=\frac12\frac\partial{\partial t}\|u(t)\|_{L^2(\Omega)}^2.$$
The bilinear form is coercive, then we obtain
$$a(u(t),u(t))\geq\alpha\|u(t)\|_V^2.$$
Thanks to the Cauchy-Schwarz inequality, we find
$$(f(t),u(t))\leq\|f(t)\|_{\mathcal{L}^2(\Omega)}\|u(t)\|_{\mathcal{L}^2(\Omega)}.$$

### Semi-discrete form
Let $V = H^1_{\Gamma_D} (\Omega)$ the weak formulation for the **heat** equation becomes 
$$
\int_\Omega\frac{\partial u(t)}{\partial t}v\mathrm{~}d\Omega+a(u(t),v)=\int_\Omega f(t)v\mathrm{~}d\Omega\mathrm{~}\quad\forall v\in V
$$
where $a(\cdot,\cdot) = \int_\Omega\nabla u \nabla v$, the initial condition is preserved as $u(0) = u_0$ and the boundary conditions are assumed to be homogeneous.
The semi-discrete form can then be obtained by 
$$
\int_\Omega\frac{\partial u_h(t)}{\partial t}v_h\mathrm{~}d\Omega+a(u_h(t),v_h)=\int_\Omega f(t)v_h\mathrm{~}d\Omega\quad\forall v_h\in V_h
$$
where $V_h \subset V$.



### Time discretization and the $\theta$-method
Proper care must be taken in order to discretize the time component, one way of handling it is to introduce the $\theta$-method by approximating the temporal derivative with a simple difference quotient and replacing **all** the other terms with a linear combination of the values at time $t^k$ and $t^{k1}$ in the following manner
$$
\int_\Omega
\frac{u_h^{k+1} - u_h^{k}}{\Delta t} d\Omega
+
\int_\Omega \left(\theta \nabla u_h^{k+1}  + (1-\theta) \nabla u_h^{k} \right)\nabla v_h
=
\int_\Omega
\left( \theta f^{k+1} + (1-\theta)f^{k} \right) v_h\mathrm{~}d\Omega\quad\forall v_h\in V_h
$$
this yields
- **Forward Euler**: when $\theta = 0$ accurate to order one with respect to $\Delta t$
- **Backwards Euler**: when $\theta = 1$ accurate to order one with respect to $\Delta t$
- **Crank-Nicolson**: when $\theta = \frac{1}{2}$ accurate to order two with respect to $\Delta t$

Moreover, order two is achieved only $\theta  = \frac{1}{2}$.

##### Stability
In the case where $\theta = 0$ we have the following condition for stability
$$
\exists c>0\quad:\quad\Delta t\leq ch^2\quad\forall h>0
$$
