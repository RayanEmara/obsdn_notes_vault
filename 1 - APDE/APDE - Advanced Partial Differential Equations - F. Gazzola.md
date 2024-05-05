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
# Table of contents 
 ```table-of-contents
title:
style: nestedList # TOC style (nestedList|inlineFirstLevel)
minLevel: 2 # Include headings from the specified level
maxLevel: 4 # Include headings up to the specified level
includeLinks: false # Make headings clickable
debugInConsole: false # Print debug info in Obsidian console
```


<div style="page-break-after: always; visibility: hidden">
\pagebreak
</div>

## Disclaimers and preface

These notes were taken during AY 2023/2024 using older material, your mileage may vary. They're meant to accompany the lectures and in no way aim to substitute a professor yapping away at an iPad 30m away.

Any reference of form $\left( x.x.x \right)$ where $x \in \mathbb{N}$ is a reference to a a Theorem/Definition/etc... in the professors own book:
 **[Elements of Advanced Mathematical Analysis for Physics and Engineering](https://www.amazon.com/Elements-Advanced-Mathematical-Analysis-Engineering/dp/8874886454)
 I'm using the September 2013 version.
 
 These notes are based on the work done by students Ravizza and Mescolini, you can access their notes by logging on the [AIM website](https://aim-mate.it/) and checking out **Portale appunti**. This would NOT have been possible without their notes..
 
 For any questions/mistakes you can reach me [here](mailto:notes@rayanemara.com?subject=APDE%20Notes%20-%20Problem%20problem%20).

All rights go to their respective owners.


<div style="page-break-after: always; visibility: hidden">
\pagebreak
</div>



## Sobolev spaces and initial derivation for discrete domains
$C^*$ are banach spaces but not *Hilbert* spaces (i.e. we cannot use Lax-Milgram).

> [!definition] Example
> Second ordinary differential equation:
> 
> $$
> \begin{cases}
> -u'' + u = f \\ 
> u(a) = u(b) = 0
> \end{cases}
> $$
> 
> What happens if $f$ is not continuous ? Let's try a weak solution.
> 
> $$
> -u''\varphi + u\varphi = f\varphi \qquad \forall \varphi \in \mathcal{D}(a,b)
> $$
> 
> $$
> \mathcal{D}(a,b) = C^\infty_c(a,b)
> $$
> 
> We then integrate 
> 
> $$
> -\int_a^b u''\varphi + \int_a^b u \varphi = \int_a^b f \varphi
> $$
> $$
> \int_a^b (-u''\varphi + u \varphi) = \int_a^b f \varphi
> $$
> 
> This integral obviously has sense if and only if the integrands are $L^1$:
> - To have $u\varphi \in L^1$ we need both of those functions to be $L^2$
> - To have $u'\varphi' \in L^1$ we need both of those derivatives to be in $L^1$ 
> - We also need $f \in L^2$

In order to be talking sense (i.e. having a legal formulation) this is enough. We will end up with a weak formulation satisfying these properties.
- $(u,u',f)\in L^2$

Keep in mind that $L^2$ contains discontinuous functions.

>[!definition] Weak derivative 
>We say that $v=u'$ in the weak sense (distributional sense) if:
>$$\int_I u\varphi' = -\int_I u\varphi \qquad \forall \varphi \in \mathcal{D}(I)$$


> [!definition] The simplest Sobolev space
> We define the $H^1$ space as the following
> $H^1(I) = \{u\in L^2(I) ,\ u' \in L^2(I)$ in a weak sense$\}$ 

$H^1$ is the simplest one dimensional [[Sobolev space | sobolev]] space. We are going to change the dimensionality of the space, the order of the derivative and the summability of the space. 
Can we strengthen the assumptions on the hypotheses on our solution if we strengthen those on $f\in L^2$ ? Answer is yes but we'll get there.

>[!example] Examples of weak derivatives
> **First example**:
> $C^1(\overline{I}) \subset H^1(I)$ : The weak derivative coincides with the classic one.
> 
> **Second example**:
> Let $f(x) = |x|$ and $I = (-1,1)$
>  Conjecture: We want to prove that the derivative $f'(x)$, defined as below, belongs to $L^2$.
>  $$
>  f'(x) = \begin{cases}
> -1 \qquad \text{if } x<0\\ 
> +1 \qquad \text{if } x>0
> \end{cases}
>  $$
>  Let's try it using the definition of weak derivative:
>  ![[Pasted image 20240427162531.png]]
>  We used the definition of weak derivative, written our initial integral in it's form, therefore proven that it is the weak derivative.
>  
>  **Third example**:
>  ![[Pasted image 20240427162829.png]]
>  
>  **Fourth example**: 
>  ![[Pasted image 20240427162932.png]]


>[!theorem] Separability of $H^1$ (6.2.3) 
>$H^1(I)$ is a [[Separable set | separable]] Hilbert space when endowed with he following scalar product.
>$$
>(u,v)_{H^1} = \int_I (u'v' + u \ v)
>$$
>>[!theorem] Proof
>>We start by proving that it is indeed a scalar product, i.e. the following properties:
>>- Symmetry (obvious)
>>- It is a norm:
>> $$||u||^2_{H^1} = \int_I[(u')^2 + u^2 ]$$
>> We then need to show that it is a Banach space with the following scheme:
>> ![[Pasted image 20240427163554.png]]
>>  All we need now is proving that the Hilbert space we constructed is indeed **separable**, we can use $H^1$ own isomorphism to $L^2$ which is separable in its own right, then construct a linear map between the two:
>>  $$
>>  Lu = (u,u')
>>  $$

>[!theorem] Map between Sobolev and Continuous spaces (6.2.5)
> Every function in $H^1$ can be represented by a continuous function, formally:
> $$
> H^1(I)\subset C^0(\overline{I})
> $$
> 
> This is interesting because $H^1$ also contains discontinuous functions, then how is it a subset of $C^0$ ?? 
> This is of course not what the theorem is saying, more specifically
> >[!theorem] Proof
> >![[Pasted image 20240427174140.png]]

>[!definition] Definition (6.2.7)
> We define the closure of $\mathcal{D}(I)$ with respect to the $H^1(I)$ norm as the following: 
> $$
> H^1_0(I) = \overline{\mathcal{D}(I)}^{H^1(I)}
> $$
> Recall that $\mathcal{D}(\Omega)$ is the space of *smooth* functions with compact support over whatever $\Omega$ is. 
> 
> Essentially we're defining the space of $H^1(I)$ functions that vanish at the boundary of $I$.

Note that the closure with respect to the $L^2$ norm has special properties, specifically:
$$
\overline{\mathcal{D}(I)}^{L^2(I)} = L^2(I)
$$
This is because $\mathcal{D}(I)$ is dense in $L^2(I)$, , meaning that the space of *smooth enough* functions with a compact support within the boundaries of $I$ is dense in $L^2$. If you think about it this makes sense since these functions would be $L^\infty$ (i think...).

>[!definition] Remark (6.2.7)
>If and only if $I \neq \mathbb{R}$
>$$
> H^1_0(I) \subsetneq H^1(I)
>$$

>[!definition]
>A weak solution satisfies 
>$$
>u \in H^1_0(a,b) 
>$$
>$$
>\int_a^b(u'\varphi' + u \ \varphi) = \int_a^bf\varphi \qquad \forall \varphi \in H^1_0(a,b)
>$$
>We can rewrite this as 
>$$
>(u,\varphi)_{H^1} = (f,\varphi)_{L^2}
>$$
>You can then apply **Lax-Milgram** (1.7.4), prove the hypotheses on the bilinear form and conclude that $\exists!u$.

>[!theorem] Poincarè inequality (6.2.9)
>Let $I=(a,b)$ bounded then:
>$$
>||u||_{L^2(I)} \leq (b-a) \ ||u'||_{L^2(a,b)} \qquad \forall u \in H^1_0(I)
>$$
>As a consequence the map $u \to ||u'||_{L^2}$ defines a norm in $H^1_0(I)$ which is equivalent to the norm of $H^1(I)$
>>[!theorem] Proof 
>> It's better if you use the books proof.
>>![[Pasted image 20240427182436.png]]

Let $H_1$ and $H_2$ be Hilbert spaces such that:
$$
H_1 \subset H_2 \qquad (H^1 \subset L^2 , \ H^1_0 \subset L^2)
$$
Any Hilbert space is the dual of itself:
$$
H_1' \approx H_1
$$
$$
H_2' \approx H_2
$$
Therefore if $H_1\subset H_2$ and $H_2'\subset H_1'$ (since a smaller Hilbert space will have a larger number of linear and continuous functionals, this is taken straight from notes I don't condone this behaviour):
$$
H_2\subset H_1
$$
But this is absurd, look at the hypothesis !
There's clearly a mistake somewhere and that mistake lies in the assumption that we can use both isomorphisms at the same time, which isn't true due to the **Riesz representation theorem** (1.5.5).

>[!definition] Hilbert (Gelfand) triple (Bottom of 6.2.18)
>The following is true:
>$$
>H_1 \subset H_2 \underbrace{\approx}_{(1)} H_2' \subset H_1'
>$$
>$$
>H_0^1(I)\subset L^2(I) \underbrace{\subset}_{(2)}(H_0^1(I))' = H^{-1}(I)
>$$
>Where:
>1. Is the Riesz representation theorem (1.5.5)
>2. The pivot space is the space in which we make use of the dual (in this case $L^2$)
>   
>   Moreover if we define:
>   $$
>  Lu(v) = \int_Iu\ v \qquad \forall u \in L^2(I) 
>  $$
>  Then the map $v \to Lu(v)$ is linear and continuous $\forall v \in H^1_0(I)$

>[!definition] Proposition 6.2.10
>Let $F\in H^{-1}(I)$ then:
>$$
>\exists f_0,f_1\in L^2(I)
>$$
>Such that:
>$$
><F,v> = \int_I f_0v + \int_I f_1 v \qquad \forall v \in H^1_0
>$$
>and
>$$
>||F||_{H^{-1}(I)} = \max\{||f_0||_{L^2}(I) \ , \ ||f_1||_{L^2}(I) \}
>$$
>In the case where $I=\mathbb{R}$ then substitute $H^1_0(I)$ with $H^1(I)$.
>
>In the case where $I$ is unbounded we can take $f_0 = 0$. ^6-2-10

This a sort of representation for linear and continuous functionals over $H^1_0(I)$.

Now suppose that $f_0 = 0$ :
$$
<F,v> =\int_I f_1v' \underbrace{=}_{IBP} -\int_I f_1'v
$$
but this basically means $F=-f_1'$. 

...

Not so fast bucko, this is an illegal move since $f_1 \in L^2$ so we don't know its derivative necessarily.
We, on the other hand, can say (by defining it as such) that :
$$
F = -f_1'
$$
Implies that $F$ has $-1$ derivative in $L^2$.
>[!example] Dirac delta
>![[Pasted image 20240427193930.png]]

## The $\mathbb{R}^n$ case
We'll take a domain $\Omega$ in $\mathbb{R}^n$ to be an open set, for example $\partial\Omega$.
>[!example] 
>$$
>\Omega = \{ (x,y) \in [0,1]^2 \ , \ x,y \in \mathbb{Q} \}
>$$
>$$
>\partial\Omega = [0,1]^2
>$$

Our main assumptions will be that $\Omega$ has to be an open set and $\partial\Omega$ smooth. Let's say we want $\partial\Omega \in C^1$, it can be constructed as the union of locally $C^1$ functions. (3.1.2)

Now we want to define $H^1(\Omega)$ (we'll worry about the boundary later on).

>[!definition] Weak derivative in $\mathbb{R}^n$
>The $i$-th weak derivative $w =\frac{\partial u}{\partial x_i}$ of $u$ is such that:
>$$
>\int_{\Omega} u \ \frac{\partial v}{\partial x_i} = \int_{\Omega} w \ v \qquad \forall v \in \mathbb{D}(I)
>$$

>[!definition] Sobolev space (6.,2.11)
>Let $\Omega \in \mathbb{R}^n$ be an open set, the Sobolev space $H^1$ is defined as:
>$$
> H^1(\Omega) = \left\{ u \in L^2(\Omega) \ ; \ \frac{\partial u}{\partial x_i} \in L^2(\Omega) \qquad \forall i = 1,..,n \right\}
>$$

(6.2.12)
Given $u \in H^2(\Omega)$:
 We take $$H^1(\Omega) \approx L^2(\Omega)^{n+1}$$ 
 Meaning we take these spaces to be  isomorphic.
 
 We define the gradient as :
  $$\nabla u = \begin{bmatrix} \frac{\partial u}{\partial x_i} \\ .. \\ .. \end{bmatrix} \in L^2(\Omega)^n$$
  
  We define the norm to be:
  $$| \nabla u |^2 = \sum_{i=1}^{n}  \left( \frac{\partial u}{\partial x_i} \right)$$
  
Taking derivative in the classical sense:
$$
\left(u  \in C^1(\Omega) \bigcap L^2(\Omega)\right) \land \left(u \frac{\partial u}{\partial x_i} \in L^2(\Omega) \right) \Longrightarrow u \in H^1(\Omega)
$$
Moreover the classical partial derivatives coincides with the weak ones, additionally if $\Omega$ is bounded:
$$
\Longrightarrow C^1(\overline{\Omega}) \subset H^1(\Omega)
  $$
We finally define the bilinear form on $H^1$ to be:
$$
(u,v)_{H^1} = \int_\Omega \left( u \cdot v + \nabla u \times\nabla v \right)
$$
Where the first is a product between two scalar functions and the second is a scalar product (sum of $n+1$ terms). 
The former will be sometimes omitted.

We can define a norm (6.2.13) therefore
>[!theorem] $H^1$ separability in $\mathbb{R}^n$ (6.2.14)
> $H^1$ is [[Separable set | separable]] since we can:
> - Define a scalar product such as:
>   $$
>   (u,v)_{H^1}
>  $$
> - We can define a norm
> - The space is complete
>
>Refer to the book for an actual proof.

>[!definition] Closure to $H^1$ (6.2.15)
>$$
> H^1_0(\Omega) = \overline{\mathbb{D}(\Omega)}^{H^1(\Omega)} \subset H^1(\Omega)
>$$
>The book goes in depth on how actually it is defined..

>[!example]
>![[Pasted image 20240427233111.png]]

>[!example] Example (6.2.16)
>Basically just show that the function and its gradient are $L^2(\Omega)$ and you're good to go. This is also interesting because it's a counter example to the notion that (in $n\geq 2$) a function only need be limited to be in the respective Sobolev space.
>
>![[Pasted image 20240427234308.png]]

>[!theorem] Poincarè inequality (6.2.18)
>Assume that $\Omega$ is bouned. Then there exists a constant $C(\Omega)$ such that:
>$$
> \| u \|_{L^2} \leq C \ \| \nabla u \|_{L^2} \qquad \forall u \in H^1_0(\Omega)
>$$
>Moreover the map $u \to \| u \|_{L^2}$ us a norm in $H^1_0(\Omega)$ which is equivalent to the norm $\| u \|_{H^1}$.
>
>![[Pasted image 20240427234705.png]]
>
>Professor then goes on to define the relevant Hilibert triple
>$$
>H^1_0(\Omega) \subset L^2(\Omega) \subset H^{-1}(\Omega)
>$$
>Where the second continuous embedding is to be meant as follows:
>Each function $u \in L^2(\Omega)$ is identified with the linear functional $I_u \in H^{-1}(\Omega)$ defined as :
>$$
> <I_u, v> \ := \int_{\Omega}u \ v \qquad \forall v \in H^1_0(\Omega)
>$$
>Essentially we can map each $L^2$ function to a functional from $H^1$ to $\mathbb{R}$.

Note that we denote by $H^{-1}(\Omega)$ the dual space of $H^1_0(\Omega)$.

>[!definition] Proposition (6.2.19)
>Let $\Omega \subset \mathbb{R}^n$ be open and let $F \in H^{-1}(\Omega)$, then $\exists \{ f_i\}_{i=0}^n \in L^"(\Omega)$ such that:
>$$
><F,v> \ = \int_{\Omega}f_0 \ v+ \sum_{i=1}^n f_i \frac{\partial u}{\partial x_i} \qquad \forall v \in H^1_0(\Omega)
>$$
>And:
>$$
>\|F \|_{H^{-1}} = \max_{i=0,...,n}\|f_i\|_L^2
>$$
>Moreover, if $\Omega$ is bounded, we can take $f_0 = 0$
>>[!definition] Remark
>>Note that in this case $v \in L^2( \Omega )$ because $v \in H^1_0(\Omega)$
>
>>[!definition] Proof
>>There's a proof in the book but here's what the prof wrote:
>>![[Pasted image 20240428120422.png]]

>[!theorem] Sobolev embedding theorem (6.2.20 - 6.2.21)
>Let $\Omega \subset \mathbb{R}^n$ be an open domain with $\partial \Omega \in Lip$ and $n \geq 2$ . Then:
>$$
>H^1(\Omega) \subset L^p(\Omega) \qquad \begin{cases}
>\forall 2 \leq p <\infty \qquad if \ n=2 \\
> \forall 2 \leq p < \frac{2n}{n-2} \qquad if \ n = 3
\end{cases}
>$$
>In addition to the above, if $\Omega$ is bounded, then the embeddings become compact:
>![[Pasted image 20240428121318.png]]
>>[!definition] Remark
>>The following is called the **Critical Sobolev exponent**:
>> $$
>> 2^* = \frac{2n}{n-2}
>> $$
>> A sequence that converges in the $H^1(\Omega)$ sense implies convergence in a certain $L^p$ sense, formally:
>> $$
>> 
>>   
u_n\mathop{\longrightarrow}^{H^1(\Omega)} u 
\ \Longrightarrow \
u_n\mathop{\longrightarrow}^{L^p} u
>> $$
>
>>[!definition] Remark 
>>The functions inside $H^1(\Omega)$ are defined up to a negligible set.

## The $H^k$ spaces
We basically take 6.2.11 and apply it a bunch of times to define higher order weak derivatives.

Note that in the image below the prof *formally* sets $H^0 = L^2$ (6.3.4)
![[Pasted image 20240428122540.png]]

>[!definition] Multi-index notation
>We call a multi-index a vector:
>$$
>\alpha=\left( \alpha_1 , ..., \alpha_n \right) \in \mathbb{N}^n
>$$
>With norm:
>$$
>|\alpha| = \sum_{i=1}^m \alpha_i
>$$
>We'll use this to define partial derivatives kinda like this:
>$$
>D^\alpha v = \frac{\partial^{|\alpha|}v}{\partial x_1^{\alpha_1} , ..., \partial x_n^{\alpha_n}}
>$$

>[!definition] Definition 6.3.1
>Let $u\in L^1_{\text{loc}}(\Omega)$ and $\Omega \subseteq \mathbb{R}^n$ be an open set. Given a multi-index $\alpha$ we'll say that $u$ admits a weak derivative $D^\alpha u$  if there exists $g \in L^1_{\text{loc}}(\Omega)$ such that:
>$$
> \int_{\Omega} u D^{\alpha} \varphi = (-1)^{| \alpha|} \int_{\Omega}g \varphi \qquad \forall \varphi \in \mathcal{D}(\Omega)
>$$

>[!definition] Definition and Separability Theorem (6.3.2 - 6.3.5)
>Let $\Omega$ be as above and $k \in \mathbb{N}$. The $H^k(\Omega)$ is defined by:
>$$
>H^k(\Omega) = \left\{
>
u \in L^2(\Omega), \qquad D^{\alpha}u \in L^2(\Omega) \qquad \forall | \alpha | \leq k \
>\right\} \qquad
>\forall k \in \mathbb{N}
>$$
>We take $H^k(\Omega)$ to be a [[Separable set | separable]] Hilbert space with scalar product:
>$$
>(u,v)_{H^k(\Omega)} = \int_{\Omega}{\left(u v + \sum_{1 \leq |\alpha| \leq k} D^{\alpha}u\  D^{\alpha}v\right)}
>$$
>With induced norm:
>$$
>\| u\|_{H^k(\Omega)} = \left( \int_{\Omega \land 0 \leq | \alpha| \leq k} \sum |D^{\alpha} u |^2 \right) ^{\frac{1}{2}}
>$$
>>[!theorem] Proof 
>>Check the book for a more detailed proof, here's what the prof wrote,
>>![[Pasted image 20240428141905.png]]

>[!definition] Separable Banach spaces with no scalar product. (6.3.8)
>![[Pasted image 20240428142515.png]]

>[!definition] Remark (6.3.4)
> We define $H^k(\Omega)$ inductively. Check the book.

>[!theorem] Fourier definition for higher order Sobolev spaces(6.3.5 - 6.3.6)
>We can define $H^k(\mathbb{R}^n, \mathbb{C})$ in a more straightforward way using the Fourier transform. 
>$$H^k(\mathbb{R}^n, \mathbb{C}) = \{ u \in L^2(\mathbb{R}^n, \mathbb{C}); \quad (1 + |\xi|^2)^{\frac{k}{2}} \ \hat{u}(\xi) \in L^2(\mathbb{R}^n, \mathbb{C}) \}$$
>In this construction $H^0 = L^2$ because $(1 + |\xi|^2)^{\frac{k}{2}} = 1$ if $k=0$ , we need, however, to define a scalar product to go further.
>
>![[Pasted image 20240428143529.png]]
>
>Note that $k$ doesn't *have* to be an integer in this construction, it just has to be $k = s \geq 1$. We then end up with what are called **Non Local PDE's** 
>>[!definition] Remark
>>This only works on $\mathbb{R}^n$, not on bounded domains (Duh!)

## Non-integer $H^k$

As previously mentioned, we'll take $s\geq 0$.
$$
H^s(\mathbb{R}^n ,\mathbb{C}) = 
\left\{ u \in L^2(\mathbb{R}^n ,\mathbb{C}) \ , \quad (1 + |\xi|^2)^{\frac{k}{2}} \hat{u}(\xi) \in L^2(\mathbb{R}^n ,\mathbb{C}) \right\}
$$
Endowed with the scalar product

![[Pasted image 20240428145134.png]](6.4.1)

We can extend this construction to a general domain $\Omega$.
>[!theorem] Theorem (6.4.5)
>![[Pasted image 20240428145442.png]]
>![[Pasted image 20240428150406.png]]

## $H^s_0$ spaces and Trace operators
>[!definition] Definition (6.5.1)
>For every $s \geq 0$:
>$$
>H^s_0(\Omega) = \overline{\mathcal{D}(\Omega)}^{H^s(\Omega)}
>$$
>A special case is the following: 
>$$\Omega = \mathbb{R}^n \Longrightarrow H^s_0(\Omega) = H^s(\Omega)$$ 
>Otherwise 
>
>![[Pasted image 20240428152304.png]]
>
>What if  $s \in \left[ 0,1 \right]$ ?
>Let's put ourselves in the case where $\Omega \neq \mathbb{R}^n$
>
>![[Pasted image 20240428152727.png]]
>
>Read the book for a better understanding of what's going on in the background. 
>Essentially $\mathcal{D}(\Omega)$ is dense in $H^s(\Omega)$ if and only if $s \leq \frac{1}{2}$. Meaning the "closure" and the space are the same. (What's written in the pic). 
>The $\mathbb{R}^n$ case is also explained.

A problem now arises, how do we define the value of a function $u \in H^s(\Omega)$ on the boundary ? Remember these functions are defined up to a negligible set.

![[Pasted image 20240428153839.png]]

We'll solve it this way
>[!definition] Trace operator (6.5.2 - 6.5.4)
>Assume $\partial \Omega \in C^\infty$, now note that $C^\infty(\overline{\Omega})$ is [[Dense sets | dense]] in $H^s(\Omega) \ \forall s \geq 0$ so the idea is to define a sequence that approximates $u$ and evaluate it at the boundary in this fashion:
>
>Take $u \in H^s(\Omega)$, $s> \frac{1}{2}$, then construct a sequence:
>$$
>\{ u_m \}\subset C^{\infty}(\overline{\Omega})
>$$
>Such that:
>$$
>u_m \mathop{\longrightarrow}^{H^s(\Omega)} u
>$$
>Now define the **Trace operator** as:
>$$
>\gamma_0 u = \lim_{m \rightarrow 0} u_m\big\rvert_{\partial \Omega}
>$$
>This $\gamma_0 u$ is an approximation and in general this definition fails to actually define what happens at the boundary, or even define a way to evaluate *a type* of boundary condition. The book goes *much* more in depth.
>>[!definition] Redefine the boundary points
>> We can see any point $x_0 \in \partial \Omega$ as:
>> $$x_0 = \begin{bmatrix} x_1^0\\ x_2^0 \\ .. \\ x_n^0\end{bmatrix}$$
>> At this point there exists a veeery tiny $\delta > 0$ and a function $\varphi \in C^\infty(B_\delta(x_0))$ (locally smooth) such that:
>> $$
>> x \in \partial \Omega \iff x_n = \varphi(x') \quad \forall x' \in B_\delta(x_0')
>> $$
>> Where $x'$ are as below:
>> 
>> ![[Pasted image 20240428170128.png]]
>> 
>> The relevant Sobolev space is 
>> $$
>> H^s(B_\delta (x_0'))
>> $$
>> Note that it's of dimension $n-1$
>
>We say that $\gamma_0 u \in H^s(\partial\Omega)$ if 
>$$
>u_0(x') = u(x' , \varphi(x')) \in H^s(B_\delta(x_0')) \quad \forall x_0 \in \partial \Omega
>$$
>The following is called the *restriction* of $u$ to the boundary:
>$$
>u(x', \varphi(x')) 
>$$

>[!theorem] Theorem (6.5.3)
>Let $s > \frac{1}{2}$, the trace operator is such that:
>$$
>\gamma_0 \colon H^s(\Omega) \to H^{s -\frac{1}{2}}(\partial\Omega)
>$$
>We essentially lose half a degree of derivative. 
>>[!definition] Remark
>>$\gamma_0 u$ is only surjective, not injective. We *can* get injectivity if we add more constraints but we can't conclude that with our current formulation (we get it in PDEs, using the lifting operator)

Some examples follow.

![[Pasted image 20240428205615.png]]

>[!definition] Trace operator in the case of the Neumann boundary condition (6.5.7)
>![[Pasted image 20240428210318.png]]

>[!definition] $H^s_0(\Omega)$
>$$
>H^s_0(\Omega) = \overline{\mathcal{D}(\Omega)}^{H^s(\Omega)}
>$$


>[!theorem] Theorem (6.5.8)
>![[Pasted image 20240504183110.png]]

## Embeddings theorems
In this section we will make use of [[Continuous embedding  | embeddings]] in order to evaluate the regularity of $H^s(\Omega)$ elements.

>[!theorem] Sobolev Theorem (6.6.1)
> Let $\Omega = \mathbb{R}^n$ or an open set wiht *Lipschitz* boundary. Let $s \geq 0$, then the following [[Continuous embedding | continuous embeddings]] hold:
> $$
> H^s(\Omega) \subset \begin{cases} 
> L^p(\Omega) \quad \forall \ 2 \leq p \leq \frac{2n}{n-2s} \quad \text{if } n >2s \\
> L^p(\Omega) \quad \forall \ 2 \leq p \leq \infty \qquad  \ \text{if } n = 2s \\
> C^0(\overline{\Omega}) \ \ \ \ \ \ \  \ \ \ \ \ \ \ \ \ \ \ \qquad \qquad \text{if } n < 2s 
> 
> \end{cases}
> $$
> So if we increase $s$ we increase the regularity but we also have to take into account $n$ (the number of dimensions). 
> >Theorem 6.6.1 ensures the continuity of the elements of $H^1(\Omega)$ when $n = 1$, while if $n = 2$, the fact that a function belongs to $H^1(\Omega)$ is not any more sufficient to guarantee its continuity. In the case $n = 2$, in order to have continuity of a function $u$ we have to assume that $u \in H^s(\Omega)$ with $s > 1$. We deduce that, as the space dimension increases, the regularity of the functions of $H^s(\Omega)$ decreases.
> 
> If $\Omega \subseteq \mathbb{R}^n$ is bounded then all of the above embeddings are [[Compact embedding | compact ]] except for:
> $$
> H^s(\Omega) \subset L^{\frac{2n}{n-2s}} \qquad \text{when } n > 2s
> $$

Here's a demonstration (example 6.6.5) of this theorem in action , assume the following:
- $n \geq 3$
- $\Omega$ unbounded
- $\big\{ u_m\big\} \subset H^1(\Omega)$ such that $\big\| u_m \big\|_{H^1(\Omega)} \leq c < \infty$

Thanks to theorem 1.4.13, we know that:
$$
\big\{ u_{m_k} \big\} \quad s.t. \quad u_{m_k}  \mathop{\longrightarrow}^{H^1(\Omega)}_{\text{weakly}} u \in H^1(\Omega)
$$
We can therefore conclude that the above sequence converges to $L^p \ \  \forall \ 2 \leq p \leq \frac{2n}{n-2s}$, formally:
$$
u_k , u \in L^p{\Omega} \ \ \text{and} \ \ u_k\mathop{\longrightarrow}^{L^p(\Omega)} u 
$$

### $H^s(\Omega)$ where $s < 0$
These spaces are defined extending definition [[APDE - Advanced Partial Differential Equations - F. Gazzola#Non-integer $H k$ | 6.4.1]].
>[!definition] Negative non-integer Sobolev spaces (6.7)
>Let $s < 0$
>$$H^s(\mathbb{R}^n) = \big\{ u \in \mathcal{S'}(\mathbb{R}^n) \ ; \ (1 + |\xi|^2)^{\frac{s}{2}} \hat{u}(\xi) \in L^2(\mathbb{R}^n) \big\}$$
>Where $\mathcal{S'}$ is the complex space of [[Tempered distributions | tempered distributions]] (the dual for the space of fast decreasing functions).

If we let $k$ be an integer then we can define the following duality:
$$
H^{-k}(\Omega) = \big[ H^k_0(\Omega) \big]'
$$
We can also define the following Hilbert triple:
$$
H^k_0(\Omega) \subset L^2(\Omega) \subset H^{-k}(\Omega)
$$
At this point there's a lot of boilerplate that can honestly be studied far better in the book anyway, I won't bother writing it down since it's really just stuff that is gonna be referenced later. I'll include references to the book whenever needed.

## Weak solutions of partial differential equations
### Homogeneous Dirichlet problem
Let $\Omega$ be an open, bounded subset of $\mathbb{R}^n$  and let its boundary $\partial \Omega \in C^1$. Take $\alpha \in \mathbb{R}$ , $f \in L^2(\Omega)$. 
We define the Homogeneous Dirichlet Problem (7.1) as:
$$
\begin{cases}
-\Delta u +  \alpha u &= f \qquad &&in  \ \Omega \\
u &=0 &&on\ \partial \Omega
\end{cases}
$$
If $f \in C^0 (\Omega)$ and $u \in C^2 (\Omega) \ \cap \ C^0 (\overline{\Omega})$ the equation is satisfied in the classical sense, meaning $u$ is a classical solution.

A **weak solution** on the other hand satisfies (7.1.1) $u \in H^1_0(\Omega)$ and:
$$
\int_{\Omega} \nabla u \nabla v + \alpha \int_{\Omega} uv = \int_{\Omega} fv \quad \forall v \in H^1_0(\Omega)
$$
If $u$ weak solution is $C^2 (\overline{\Omega})$ then you can easily conclude that it is also a classical solution.

>[!example] Important counter example
>If $n > 1$ then:
>$$
>\Delta u \in C^0 (\Omega) \ \centernot\Longrightarrow \  u \in C^2 (\Omega)
>$$
>In higher dimensional spaces it's wise to avoid formulating your problems using such regular spaces, prefer instead Sobolev spaces

>[!example] Remark (7.1.2)
>If $u$ is a classical solution of (7.1) then necessarily $f \in C^0 (\overline{\Omega})$. Therefore if $f$ is $L^2(\Omega)$ we cannot expect a weak solution to also be classical.

>[!theorem] Theorem (7.1.3)
>Let $\alpha \geq 0$ and $f \in L^2(\Omega)$ then there exists a *unique* solution $u \in H^1_0(\Omega)$ of 7.1 (the Homogeneous Dirichlet problem). 
>Moreover the unique solution is also the unique minimizer of $J(v)$:
>$$
>J(v) = \frac{1}{2} \int_{\Omega}\big(|\nabla v |^2 + \alpha v^2 \big) - \int_{\Omega} fv \quad v \in H^1_0(\Omega)
>$$
>The intuition for the "minimum" being unique is that this is basically a parabola.
>>[!theorem] Proof
>>We endow the Hilbert space $H^1_0(\Omega)$ with the following scalar product (6.2.18):
>>$$
>>(u,v)=\int_\Omega\nabla u\nabla v\quad\forall u,v\in H_0^1(\Omega).
>>$$
>>Given the assumption $a \geq 0$, the bilinera form:
>>$$
>>a(u,v)=\int_\Omega(\nabla u\nabla v+\alpha uv)\quad\forall u,v\in H_0^1(\Omega)
>>$$
>>is coercive over $H^1_0(\Omega)$.
>>Moreover the linear form $v \to \int_{\Omega}fv$ is continuous over $H^1_0(\Omega)$.
>>Then the assertion follows from the Lax-Milgram theorem.

It's interesting to note that what we wrote above doesn't necessarily break for $\alpha <0$, in fact, it *can* work. By the Poincarè inequality there exists a constant $\lambda_1 > 0$ such that:
$$
\lambda_1 \|u \|^2_{L^2(\Omega)} \leq \|\nabla u\|^2_{L^2(\Omega)} \quad \forall u \in H^1_0(\Omega)
$$
Now....
$$
\begin{align}
\int_{\Omega}(|\nabla u|^{2}+\alpha u^{2}) &=\int_{\Omega}|\nabla u|^{2}+\alpha\int_{\Omega}u^{2} \\
\text{(using Poincarè)}&\geq \int_{\Omega}|\nabla u|^{2}-\frac{|\alpha|}{\lambda_1}\int_{\Omega}|\nabla u|^{2} \\
&=\frac{\lambda_{1}-|\alpha|}{\lambda_{1}}\int_{\Omega}|\nabla u|^{2}

\end{align}
$$

In fact if $\alpha < 0$:
$$
\alpha\lambda_{1}\|u\|_{L^{2}(\Omega)}^{2}\geq\alpha\|\nabla u\|_{L^{2}(\Omega)}^{2}=>\alpha\|u\|_{L^{2}(\Omega)}^{2}\geq-\frac{|\alpha|}{\lambda_{1}}\|\nabla u\|_{L^{2}(\Omega)}^{2}
$$
Meaning $\alpha$ just needs to be bigger than $-\lambda_1$ to retain coercivity.


>[!example] Exercise - Alternative characterization of poincarè constant (7.1.5) 
> Let $\Omega \subset \mathbb{R}^n$ be an open, bounded set with $\partial \Omega \in C^1$. Find $\lambda \in \mathbb{R}$ such that $u = 0$ is not the only solution to the HDP:
> $$
> \begin{cases}-\Delta u=\lambda u&\mathrm{~in~}\Omega\\u=0&\mathrm{~on~}\partial\Omega\end{cases}
> $$
> Generally speaking $u= 0$ is always a solution to Eigenvalue problems, let's assume $u \neq 0$ as a sort of 3rd constraint to our problem:
> $$
> \int_{\Omega}|\nabla u|^{2}=\lambda\int_{\Omega}u^{2} \Longrightarrow \lambda>0
> $$
> Note that if $\lambda = 0$ then $u=0$ is the only solution.
> The spectrum of $-\Delta$, which contains the values of $\lambda$ for which the problem has a non trivial solution:
> $$
> \sigma (-\Delta) \subset (0 , +\infty)
> $$
> In fact thanks to Theorem 1.8.10 we can order the eigenvalues of $-\Delta$ as a sequence of strictly increasing values.
> $$
> \sigma(-\Delta)=\{0<\lambda_{1}<\lambda_{2}\leq\lambda_{3}\cdots\leq\lambda_{n}\leq\cdots\}
> $$
> Therefore we conclude that 
> $$
> \lambda_{1}=\inf_{u\in H_{0}^{\prime}(\Omega)\setminus\{0\}}\frac{\|\nabla u\|_{2}^{2}}{\|u\|_{2}^{2}}
> $$

There's a small tangent on Frèchet derivatives, there's no mention of it in the book so I'll avoid it, you can find references to it at page 17 of Ravizza's notes.
### Non Homogeneous Dirichlet problem
Let $\Omega \subset \mathbb{R}^n$ be a bounded open set of class $C^1$, we look for a function $u \colon \overline{\Omega} \to \mathbb{R}$ satisfying:
$$
\begin{cases}
-\Delta u +  \alpha u &= f \qquad &&in  \ \Omega \\
u &=g &&on\ \partial \Omega
\end{cases}
$$
with $f$ defined over $H^{-1} (\Omega)$, $g$ on $\ H^{1/2}(\partial\Omega)$ and $\alpha \in \mathbb{R}$. If $g \in H^{1/2}(\partial\Omega)$ we can use Theorem 6.5.3 to infer that:
$$
\exists \ u_0 \in H^1(\Omega) \ s.t. \ \gamma_0 u_0 = g
$$
We then define the following set which will aid in later constructions:
$$
K=\{v\in H^{1}(\Omega); v-u_{0}\in H_{0}^{1}(\Omega)\}
$$
>[!definition] Classical solution for NHDP (7.1.7)
>A classical solution for the NHDP is a function $u \in C^2 (\overline{\Omega})$ satisfying the NHDP.

>[!definition] Weak solution for the NHDP (7.1.7)
>A weak solution for the NHDP is a function $u \in K$ such that:
>$$
>\int_{\Omega}(\nabla u\nabla v+\alpha uv)=<f,v>\quad\forall v\in H_{0}^{1}(\Omega)
>$$
>Where $<f,v>$ is equal to $\int_{\Omega}fv$
>The same reasoning as HDP applies, therefore any classical solution is automatically also a weak solution, conversely any weak solution $u \in C^2 (\overline{\Omega})$ is also a classical solution.

We'll now analyse the existence of weak solutions
>[!theorem] Dirichlet Principle (7.1.8)
>Let $\alpha > -\lambda_1$ (where the eigenvalue is to be meant as the smallest eigenvalue under homogeneous Dirichlet conditions), $f \in H^{-1}(\Omega)$, $g \in H^{\frac{1}{2}}(\partial\Omega)$.
>Then the NHDP admits a unique *weak* solution $\overline{u} \in K$.
>Moreover $\overline{u}$ is a weak solution if and only if it minimizes the Dirichlet *energy* over $K$
>$$
>J(u)=\frac{1}{2}\int_{\Omega}(|\nabla u|^{2}+\alpha u^{2}) - <f,u>
>$$
>>[!theorem] Proof
>>The proof was given by the prof but it is way too long, it's far better to study it from the book, if need be you can also make use of Ravizza p.20 

### Non Homogeneous Neumann problem
Let $\Omega \subset \mathbb{R}^n$ be a bounded open set of class $C^1$, we look for a function $u $ satisfying:
$$
\begin{cases}-\Delta u+\alpha u&=f&&\Omega\\\frac{\partial u}{\partial\nu}&=g&&\partial\Omega\end{cases}
$$

with $f \in L^2(\Omega)$, $g \in H^{\frac{1}{2}}(\partial\Omega)$ and $(\alpha >0) \in \mathbb{R}$.

>[!definition] Classical solution for the NHNP (7.1.11)
>We define a classical solution one where the boundary conditions are to be meant $f \in C^0, g \in C^0$
>$$
>u\in C^{2}(\Omega)\cap C^{1}(\bar{\Omega})
>$$

>[!definition] Weak solution for the NHNP (7.1.11)
>Recalling that the trace operator "takes" half an order in terms of Sobolev regularity, take $g \in H^{\frac{1}{2}} (\partial\Omega)$. A weak solution is $u \in H^1(\Omega)$ such that:
>$$
>\int_{\Omega}(\nabla u\cdot\nabla v+\alpha uv)=\int_{\Omega}fv+\langle g,\gamma_{0}v\rangle \quad \forall v \in H^1(\Omega)
>$$
Where $\langle.,.\rangle$ denotes the duality between $H^{-\frac{1}{2}} (\partial\Omega)$ and $H^{\frac{1}{2}} (\partial\Omega)$ by way of theorem 6.5.3 (it's really important and mentioned a lot, make sure you understand it)

Let's now work on the regularity conditions, when is a regular weak solution also a classical one ? In order to do this let's assume $u \in C^2 (\overline{\Omega})$ is a weak solution for the NHNP, using the Gauss-Green formula (really we're just integrating by parts) we infer that 
$$
\int_{\Omega}(-\Delta u+\alpha u)v+\int_{\partial\Omega}\frac{\partial u}{\partial\nu}v=\langle g,v\rangle+\int_{\Omega}fv\quad\forall v\in C^1(\overline{\Omega}).
$$
If we take $v \in \mathcal{D}(\Omega)$, meaning it has compact support
$$
\int_{\Omega}(-\Delta u+\alpha u)v=\int_{\Omega}fv\quad\forall v\in\mathcal{D}(\Omega)
$$
Which, as a consequence of 2.4.22, implies that
$$
-\Delta u+\alpha u=f \quad \mathrm{a.e. in}\  \Omega 
$$
Inserting this equation into the one we got from the Gauss-Green formula we get 
$$
\int_{\partial\Omega}\frac{\partial u}{\partial\nu}v=\langle g,v\rangle\quad\forall v\in C^1(\overline{\Omega}).
$$
The previous identity holds true for every $\nu\in H^1(\Omega)$, since $C^1(\overline{\Omega})$ is dense in $H^{1}(\Omega)$ and $\gamma_0:H^{1}(\Omega)\to H^{1/2}(\partial\Omega)$ is a continuous surjective function. Hence we mav conclude that $\frac{\partial u}{\partial\nu}=g$ on $\partial\Omega$

We now want to prove the existence and uniqueness of a weak solution

>[!theorem] Existence and uniqueness of a NHNP weak solution
>Let $f \in L^{2}(\Omega),g\in H^{-1/2}(\partial\Omega)$ and $\alpha>0.$Then there exists a unique weak solution $u \in H^l( \Omega )$ Moreover, $u$ can be obtained as the solution of the following minimum problem
>$$
>\min_{v\in H^{1}(\Omega)}\left\{\frac{1}{2}\int_{\Omega}(|\nabla v|^{2}+\alpha v^{2})-\int_{\Omega}fv-\langle g,\gamma_{0}v\rangle\right\}.
>$$
>>[!theorem] Proof
>>You can prove this by applying the Lax-Milgram theorem 1.7.4

Just like for homogeneous we want to explore what happens when we relax the $a > 0$ constraint 