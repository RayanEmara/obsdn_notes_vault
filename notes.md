# Notes for the Advanced Partial Differential Equations course
### Held by prof. Gazzola at Politecnico di Milano

## Sobolev spaces 
$C^*$ are banach spaces but not *Hilbert* spaces (i.e. we cannot use Lax-Milgram).
### Example
Second ordinary differential equation:

$$
\begin{cases}
-u'' + u = f \\ 
u(a) = u(b) = 0
\end{cases}
$$

What happens if $f$ is not continuous ? Let's try a weak solution.

$$
-u''\varphi + u\varphi = f\varphi \qquad \forall \varphi \in \mathcal{D}(a,b)
$$

$$
\mathcal{D}(a,b) = C^\infty_c(a,b)
$$

We then integrate 

$$
-\int_a^b u''\varphi + \int_a^b u \varphi = \int_a^b f \varphi
$$
$$
\int_a^b (-u''\varphi + u \varphi) = \int_a^b f \varphi
$$

This integral obviously has sense if and only if the integrands are $L^1$:
- To have $u\varphi \in L^1$ we need both of those functions to be $L^2$
- To have $u'\varphi' \in L^1$ we need both of those derivatives to be in $L^1$ 
- We also need $f \in L^2$

In order to be talking sense (i.e. having a legal formulation) this is enough. We will end up with a weak formulation satisfying these properties.
- $(u,u',f)\in L^2$

Keep in mind that $L^2$ contains discontinuous functions.

>[!definition] Weak derivative 
>We say that $v=u'$ in the weak sense (distributional sense) if:
>$$\int_I u\varphi' = -\int_I u\varphi \qquad \forall \varphi \in \mathcal{D}(I)$$


> [!definition] The simplest Sobolev space
> We define the $H^1$ space as the following
> $H^1(I) = \{u\in L^2(I) ,\ u' \in L^2(I)$ in a weak sense$\}$ 

$H^1$ is the simplest one dimensional Sobolev space. We are going to change the dimensionality of the space, the order of the derivative and the summability of the space. If $f$ becomes continuous and we have a weak formulation we use 

