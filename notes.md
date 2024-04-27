# Notes for the Advanced Partial Differential Equations course
### Held by prof. Gazzola at Politecnico di Milano

## Sobolev spaces 

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

>[!theorem] Separability of $H^1$ 
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

>[!theorem]
> Every function in $H^1$ can be represented by a continuous function, formally:
> $$
> H^1(C)\subset C^0(\overline{I})
> $$

