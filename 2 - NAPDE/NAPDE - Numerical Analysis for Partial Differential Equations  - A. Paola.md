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
maxLevel: 2 # Include headings up to the specified level
includeLinks: true # Make headings clickable
debugInConsole: false # Print debug info in Obsidian console
```


<div style="page-break-after: always; visibility: hidden">
\pagebreak
</div>

## Disclaimer and preface
These notes were taken during AY 2023-2024, the lecturer is prof. Antonietti Paola.
For any questions or errors you can reach out to me [here](mailto:notes@rayanemara.com?subject=NAPDE%20Notes%20-%20Problem%20problem%20).

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
This is a very general form which isn’t very useful for numerical analysis, it’s better to use some form of weak formulation, this will help us find an integral representation of the problem and derive numerical models from that.
Let's start quick and dirty, ignore the legality of steps, the regularity of $v$ and just try to come up with an integral form, we'll worry about conditions later !
$$
\color {purple}   \int_{\Omega} [\color{black} -\text{div}(\mu (x) \nabla u) + \underline{b} \cdot \nabla u + \sigma u \color{purple} \cdot v ] = \color {purple}   \int_{\Omega} [\color{black} f \color{purple} \cdot v]
$$
Now, integrating by parts we get:
$$
\int_{\Omega} \mu (x) \nabla u \cdot n v - \color{teal}\underbrace{\int_{\partial\Omega}\mu \nabla u \cdot n v}_{\text{notice the dominion}} \color{black} + \int_{\Omega} \underbrace{b}\nabla u v + \int_{\Omega} \sigma u v = \int_{\Omega} f v \qquad \forall v
$$