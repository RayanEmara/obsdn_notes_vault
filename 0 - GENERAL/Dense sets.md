A set $A$ is said to be dense in another set $B$ if the closure of $A$ contains $B$. 
Here's a more intuitive way to understand this:
1. Every point of $B$ is in $A$.
2. Even if a point $b$ is not in $A$, you can construct a sequence in $A$ that gets arbitrarily close.

Formally let $X$ be a normed space and let $A \subseteq B$. 
$A$ is said to be **dense** in $B$ if 
$$
\forall x \in B, \ \forall \epsilon > 0 \exists y \in A \quad s.t. \|x-y\| < \epsilon
$$

>[!example]
>Even though there are real numbers that aren't rational (like $\sqrt{2}$ and $\pi$) you can always "build up to them" using only real numbers and get arbitrarily close.
>For $\pi$ you'd do
>1. 3
>2. 3,1
>3. 3,14
>4. 3,141
>5. 3,1415
>6. ..
>   
>   Always getting closer to $\pi$.