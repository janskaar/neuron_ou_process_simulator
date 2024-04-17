 ________________________________________
/ Simulator for experimenting with first \
\ passage time densities                 /
 ----------------------------------------
        \   ^__^
         \  (oo)\_______
            (__)\       )\/\
                ||----w |
                ||     ||

A repository for experimenting with various aspects of time evolution and first-passage-time densities of an exponential LIF neuron driven by noise. I also implement two different previous approaches to approximations that have been made, first by Chizov and Graham in 2007/2008 and a second by Schwalger in 2021. They take two very different approaches, and both end up with very good results. I give a brief summary below.


We consider a neuron model with dynamics 

$$
\begin{align}
\tau_x\frac{dx}{dt} &= -x + Ry(t) \\
\tau_y\frac{dy}{dt} &= -y + \sqrt{2\tau_y} \sigma \xi (t) \text{,}
\end{align}
$$

where $\xi (t)$ is a white noise process, and a moving threshold is denoted by $b(t)$ with derivative $\dot{b}(t)$.


Define the vector

$$
\mathbf{w} = 
\begin{bmatrix}
    y \\
    x
\end{bmatrix} \text{,}
$$

and the expectation is given by

$$
\begin{align}
    \mathbb{E}[\mathbf{w}] = \mathrm{exp} (-At)\mathbb{E}[\mathbf{w}_0]
\end{align}
$$

where

$$
\begin{align}
A = 
\begin{bmatrix}
    \frac{1}{\tau_y} & 0 \\
    -\frac{1}{C}     & \frac{1}{\tau_x}
\end{bmatrix}\mathrm{,} \quad \quad
\mathrm{exp} (-At) =
\begin{bmatrix}
    e^-{\frac{t}{\tau_y}} & 0 \\
    \frac{1}{C(\tau_y^{-1} - \tau_x^{-1})}\Big(e^{-\frac{t}{\tau_x}} - e^{-\frac{t}{\tau_y}}\Big) & e^-{\frac{t}{\tau_x}}
\end{bmatrix}
\end{align}
$$

will continue later
