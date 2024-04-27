
### Held by prof. Gazzola at Politecnico di Milano



> [!NOTE] Disclaimer
> - Any reference of type $X.X.X$ where $X \in \mathbb{N}$ is a reference to a a Theorem/Definition/etc... in the professors own book:
> **[Elements of Advanced Mathematical Analysis for Physics and Engineering](https://www.amazon.com/Elements-Advanced-Mathematical-Analysis-Engineering/dp/8874886454) 
> By: A. Ferrero , F. Gazzola , M. Zanotti 
> ISBN: 978-88-7488-645-6**. 
>  I'm using the September 2013 version.
> 
> - These notes are based on the work done by students Ravizza and Mescolini, you can access their notes by logging on the [AIM website](https://aim-mate.it/) and checking out "Portale appunti".
> - For any questions/mistakes you can contact me via smoke signals and/or CFU donation.




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

$H^1$ is the simplest one dimensional Sobolev space. We are going to change the dimensionality of the space, the order of the derivative and the summability of the space. 
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
>$H^1(I)$ is a separable Hilbert space when endowed with he following scalar product.
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

>[!definition] Proposition 6.2.19
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
>||F||_{H^{-1}(I)} = max\{||f_0||_{L^2}(I) \ , \ ||f_1||_{L^2}(I) \}
>$$
>In the case where $I=\mathbb{R}$ then substitute $H^1_0(I)$ with $H^1(I)$.
>
>In the case where $I$ is unbounded we can take $f_0 = 0$.

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
