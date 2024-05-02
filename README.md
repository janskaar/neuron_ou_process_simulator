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
\end{bmatrix} \text{.}
$$

The expectation is given by

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

and the covariance is given by  

$$
\begin{align}
    \langle \mathbf{w}(t), \mathbf{w}^T(t) \rangle = \mathrm{exp} (-At) \langle \mathbf{w}(t_0), \mathbf{w}^T(t_0) \rangle \mathrm{exp} (-At) + \int_0^t \mathrm{exp} (-A(t-t')) BB^T \mathrm{exp} (-A(t-t')) dt' \mathrm{,}
\end{align}
$$

where

$$
B = 
\begin{bmatrix}
     \sigma & 0\\
     0      & 0\\
\end{bmatrix}
$$

The integral can be evaluated as the following matrix

$$
\begin{align}
S = 
    \begin{bmatrix}
        \sigma_0 (t) & \sigma_1 (t) \\
        \sigma_1 (t) & \sigma_2 (t)
    \end{bmatrix}
\end{align}
$$

where

$$
\begin{align}
    \sigma_0 (t) &= \frac{\sigma^2 \tau_y }{2} \big[ 1 - \mathrm{exp} \big(-\frac{2t}{\tau_y}) \big] \\
    \sigma_1 (t) &=    
    \frac{
        \sigma^2 \tau_x \tau_y^2}
        {2C(\tau_x^2 - \tau_y^2 )
    }
    \Big[ 2 \tau_x \big[ 1 - \mathrm{exp}\big(-t \big[\frac{1}{\tau_x} + \frac{1}{\tau_y} \big]\big) \big] 
    - \big[1 - \mathrm{exp}\big( -\frac{2t}{\tau_y}\big)\big] (\tau_x + \tau_y)  \Big] \\
    \sigma_2 (t) &= 
    \frac
    {\sigma^2 \tau_x^2 \tau_y^2}
    {2C^2(\tau_x^2 - \tau_y^2)(\tau_x - \tau_y)}
    \Big[
        \big(\tau_x^2  + \tau_x \tau_y) \big[1 - \mathrm{exp}\big(-\frac{2t}{\tau_x}\big) \big]
       +\big(\tau_y^2  + \tau_x \tau_y) \big[1 - \mathrm{exp}\big(-\frac{2t}{\tau_y}\big)\big]
       -4\tau_x \tau_y \big[1 - \mathrm{exp}\big(-t \Big[\frac{1}{\tau_y} + \frac{1}{\tau_x}\Big] \big)\big]
    \Big]
\end{align}
$$

The corresponding distribution over $x$ and it's time derivative $\dot{x}$, has the expectation and covariance

$$
\begin{align}
    \mathbb{E}[\dot{x}] &= -\frac{1}{\tau_x}\mathbb{E}[x] + \frac{1}{C_m}\mathbb{E}[y] = m_1\\
    &= 
    \bigg(
        e^{-\frac{t}{\tau_y}}
        -
        \frac{e^{-\frac{t}{\tau_x}} - e^{-\frac{t}{\tau_y}}}{\tau_x / \tau_y - 1}
    \bigg)
    \dot{x}_0
    -
    \Big(
        e^{-\frac{t}{\tau_x}} - e^{-\frac{t}{\tau_y}}
    \Big)
    \Big(
        \frac{1}{\tau_x} + \frac{1}{\tau_x (\tau_x / \tau_y - 1)}
    \Big)
    x_0
    \\
    \mathbb{E}[x] &= m_2 
    \\
    &= 
    \frac{e^{-\frac{t}{\tau_x}} - e^{-\frac{t}{\tau_y}}}{\tau_y^{-1} - \tau_x^{-1}} \dot{x}_0 
    +
    \bigg(
        e^{-\frac{t}{\tau_x}}
        + 
        \frac{e^{-\frac{t}{\tau_x}} - e^{-\frac{t}{\tau_y}}}{\tau_x / \tau_y - 1} 
    \bigg)x_0 
    \\
    \mathrm{Var}(\dot{x}) &= \frac{1}{\tau_x^2}\mathrm{Var}(x) + \frac{1}{C_m}\mathrm{Var}(y) + \frac{1}{\tau_x C_m}\mathrm{Cov}(x, y) \\ 
    &= \frac{1}{C_m}\sigma_0 + \frac{1}{\tau_x C_m}\sigma_1 + \frac{1}{\tau_x^2}\sigma_2 = \rho_0 \\
    \mathrm{Cov} (x, \dot{x}) &= -\frac{1}{\tau_x}\mathrm{Var}(x) + \frac{1}{C_m}\mathrm{Cov}(x, y) \\
    &= \frac{1}{C_m}\sigma_1 -\frac{1}{\tau_x}\sigma_2 = \rho_1 \\
    \mathrm{Cov} (x, x) &= \mathrm{Cov} (x, x) = \rho_2
\end{align}
$$

#### Probability of upcrossings
The probability density in time of the x-variable crossing a threshold $b$ with derivative $\dot{b}$ at any given time can be found as the integral

$$
\int^{\infty}_{\dot{b}} dt (\dot{x} - \dot{b}) p(x=b, \dot{x}) d\dot{x} \mathrm{,}
$$

which can be seen as the integral of the joint distribution over $(\dot{x}, x)$ over the area shown in the figure below: 

![upcrossing_integral](https://github.com/janskaar/neuron_ou_process_simulator/assets/29370469/1eb7b116-5901-4b0f-a00f-84d8bb79b9e6)

will be continued...
