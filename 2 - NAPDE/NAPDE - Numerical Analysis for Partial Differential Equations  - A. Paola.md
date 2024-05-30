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

# Table of contents
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

- [[#Disclaimers and preface|Disclaimers and preface]]


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

# Part 1
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