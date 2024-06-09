

# 31$^{\mathrm{st}}$ August, 2020

## Exercise 1 
Consider the 1D heat equation in the domain $(0,d)$

$$
\begin{cases}\frac{\partial u(t,x)}{\partial t}-\frac{\partial^2u(t,x)}{\partial x^2}=f(t,x),&0<x<d,0<t\leq T,\\u(t,0)=0,&0<t\leq T,\\\frac{\partial u(t,d)}{\partial x}=g(t),&0<t\leq T,\\u(0,x)=u_0(x),&0<x<d,\end{cases} \tag{1}
$$

for suitable functions $f$, $g$, and $u_0$
### 
##### Question
Introduce the **semi-discrete Galerkin** Finite Element formulation of $(1)$, suitably defining the introduced function space(s).

##### Solution
We start by multiplying for each $t>0$ the differential equation by a test function $v=v(\mathbf{x})$ and integrating  on $(0,d)$. We set the set to which the test function belongs as $V= H^1_{\Gamma_D}(\Omega)$ and for each $t>0$ we aim to find $u(t) \in V$ such that 

$$
\int_{\Omega}\frac{\partial u(t)}{\partial t}v \ d\Omega+\int_{\Omega}\nabla u\cdot\nabla v=\int_{\Omega}f(t)v d\Omega\quad\forall v\in V
$$

where $\Omega = (0,d)$.
At this point we discretize the equation into (note how the rightmost term appears as a result of integration by parts)
$$
\int_0^d \frac{\partial u_h(t)}{\partial t} v_h \ \mathrm{~dx~} + \int_0^d \frac{\partial u_h(t)}{\partial x} \frac{\partial v_h(t)}{\partial x} \ \mathrm{~dx~}  = \int_0^d f\ v_h \mathrm{~dx~} +
$$
$$
+ g \ v_h \quad \forall v_h \in V_h
\tag{2}
$$
where we've set $u_h(t=0) = u_{0_h} \in V_h$ and $V_h$ is the space of continuous on $(0,d)$ piece-wise polynomials of degree $\leq r$, in short we seek a solution $u_h(t) \in V_h$ where
$$
V_h = \left\{ v_h \in C^0(0,d) \colon \ v_h(0) = 0\ , \ {v_{h}}_{\bigr{|}I_k} \in \mathbb{P}^r \quad \forall k=0,\dots,N_h-1 \right\}
$$
where each $I_k$ is a piece of the interval
![[Pasted image 20240607193145.png]]

<figcaption>Discretization of the interval</figcaption>

### 
##### Question 
Write the $\theta$-method for the full discretization of $(1)$

##### Solution
The $\theta$-method generalizes numerical approximation schemes for time discretization of space-discretized schemes where $0 \leq \theta \leq 1$ taking $(2)$ we discretize the time derivative as a simple difference normalize by the timestep, in essence
$$
\frac{1}{\Delta t} \int_0^d (u_h^{n+1}-u_h^{n})v_h \mathrm{~dx~}
+
\int_0^d (\theta u_{h_x}^{n+1} + ( 1 - \theta)u_{h_x}^n)\ v_h \mathrm{~dx~}
=
\int_0^d (\theta f^{n+1} + (1-\theta)f^n)v_h \mathrm{~dx~} 
+
$$
$$
+ (\theta g^{n+1} + (1-\theta)g^n)v_h \qquad \forall v_h \in V_h \ \land \ n=0,1,\dots
$$

where each timestep is $n \Delta t$ long.
For $\theta=0$ and $\theta=1$ we obtain respectively **forward** and **backward Euler** methods which are accurate to order one with respect $\Delta t$, while for $\theta = \frac{1}{2}$ we obtain the **Crank-Nicolson** which is of second order in $\Delta t$, more precisely $\theta = \frac{1}{2}$ is the only value for which we obtain a second-order method,

###
##### Question
Discuss the numerical stability of the scheme, depending upon the value of $\theta$ (no proof needed).

##### Solution
Given that $f \neq 0$.
- If $\theta < \frac{1}{2}$ it can be shown that the $\theta$-method is absolutely stable only for 
$$
\Delta t\leq\frac2{(1-2\theta)\lambda_h^{N_h}}
$$
where $\lambda_h^{N_h}$ is the largest eigenvalue of the generalized eigenvalue problem $A \mathbf{x} = \lambda M \mathbf{x}$. It can be shown that $\lambda_{h}^{N_{h}}\lesssim h^{-2}$ therefore
$$
\Delta t \approxeq \frac{2 h^{2}}{(1-2\sigma)}
$$
- If $\theta \geq \frac{1}{2}$ then it is unconditionally absolutely stable.


### 
##### Question 
Prove the stability properties of the backward Euler discretization (assume for simplicity $g = 0$).

##### Solution
Backward Euler is equivalent to applying $\theta = 1$, therefore we take $(2)$, apply $g=0$ and ,finally, impose the test function $v_h = u_h^{n+1}$, then we get

$$\underbrace{\frac{1}{\Delta t}\int_{0}^{d}\left(u_{h}^{n+1}-u_{h}^{n}\right)u_{h}^{n+1}dx}
_{(1)}

+
\underbrace{\int_{0}^{d}\left(u_{h_x}^{n+1}\right)^{2}dx}
_{(2)}
=
\underbrace{\int_{0}^{d}f^{n+1}u_{h}^{n+1}}_{(3)}

$$

where $\Omega = (0,d)$.
1. We apply $(a-b,a)\geq\frac12(\|a\|^2-\|b\|^2)\quad\forall a,b$ to get 
   $$
   (1)\geq\frac{1}{2 \Delta t}\left(\|u_{h}^{k+1}\|_{L^{2}}^{2}-\|u_{h}^{k}\|_{L^{2}}^{2}\right)
   $$
2. We use the coercivity of $a(\cdot,\cdot)$ to get
   $$
   (2)\geq\alpha\|u_h^{k+1}\|_{L^2}^2
   $$
3. We apply Cauchy-Schwartz $\rightarrow$ Poincare $\rightarrow$ Young 
   $$
  \begin{aligned} 
   \left|\int_{0}^{d}f^{n+1} u_{h}^{n+1}\right|
   \underbrace{\leq}_{\mathrm{C-S}}
   &\left\| f^{n+1} \right\|_{L^2}
   \left\| u^{n+1}_h \right\|_{L^2} 
   \\
   \underbrace{\leq}_{\mathrm{Poincare}}
   & \left\| f^{n+1} \right\|_{L^2} C_p 
   \left\| u_{h_x}^{n+1} \right\|_{L^2} 
   \\
   \underbrace{\leq}_{\mathrm{Young}}
   & \frac{1}{2}
   \left( 
   \left\|
   f^{n+1}
   \right\|
   C_p
   \right)^2
   +
   \frac{1}{2} \left\| u_{h_x}^{n+1} \right\|_{L^2}^2
  \end{aligned}
  $$


# 16$^{\mathrm{th}}$ June, 2023

## Problem 1
This problem isn't in the course material for AA 2023/2024, won't be in future exams.

## Problem 2
Consider the following problem for $\Omega= (0,1)^2$
$$
\begin{cases}-\operatorname{div}\left(\varepsilon\nabla u\right)+\frac{\partial u}{\partial y}=f&\text{in} \ \Omega,\\[2ex]u=g&\text{on}\ \partial\Omega,\end{cases} \tag{2}
$$
where $\epsilon >0$ , and $f \in L^2(\Omega)$, $g \in L^2(\partial\Omega)$ are two *suitable* functions.
###
##### Question 
Introduce the variational (weak) formulation of $(2)$ and explain why a standard Galerkin Finite Element approximation may not be suitable for $(2)$ when $\epsilon \ll 1$

##### Solution
Problem $(2)$ is the non-conservative form of the **Advection-Diffusion-Reaction** equation where $\mathbf{b} = 1$ and $\sigma = 0$, we multiply both sides of the equation by a test function and integrate over the domain. 
The problem then becomes finding $u \in V = H^1_0(\Omega)$ such that
$$
a(u,v) = F(v) \quad \forall v \in V
$$
where the bilinear form and the linear forms are defined as 
$$
a(u,v)=\int_\Omega\varepsilon\nabla u\cdot\nabla v d\Omega+\int_\Omega\frac{\partial u}{\partial y}v d\Omega,
$$
$$
F(v)=\int_\Omega f v
$$
In cases where $\epsilon \ll 1$ the Peclet number becomes too big and therefore **GLS** or **SUPG** methods are needed to stabilize the method.

###
##### Question
Introduce a Finite Element **Galerkin-Least-Squares (GLS)** approximation of $(2)$ for $g =0$
##### Solution
