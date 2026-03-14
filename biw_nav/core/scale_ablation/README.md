# MS_TCANN: Basic Implementation of a Continuous-attractor Neural Network

The equation used in the python code was not exactly the same as that in those papers. This is a rescaled version:

$$ \alpha \cdot \tau \frac{du(x,t)}{dt} = -u(x,t) + \int J\left(x,x^\prime\right) r\left(x^\prime,t\right) dx^\prime+A\exp\left[-\frac{\left|x-z(t)\right|^2}{4a^2}\right]$$

$$ r(x,t) = \frac{\left[u(x,t)\right]_+{}^2}{1+\frac{k}{8\sqrt{2\pi}a}\int dx^\prime \left[u(x^\prime,t)\right]_+{}^2} $$

$$ J\left(x,x^\prime\right) = \frac{1}{\sqrt{2\pi}a}\exp\left(-\frac{\left|x-x^\prime\right|^2}{2a^2}\right) $$

Extern input

$$ I_{ext}(x,t) = A\exp\left[-\frac{\left|x-z(t)\right|^2}{4a^2}\right] $$

Except the dynamic variable and parameters are rescaled versions, the magnitude of external input is now decoupled from the magnitude of $u$ in stationary states.

In the simulation, how the change of network states responding to change of external input is demonstrated. In which, the external input is initially at $z=0$ in the preferred stimuli space. At $t=0$, the external input was changed from 0 to $z_0$. Snapshots taken per $10 \tau$ are presented to show the transient of the process.

## Parameters:                                                    

-$k$ [float] : Degree of the rescaled inhibition                           
-a [float] : Half-width of the range of excitatory connections           
-N [int] : Number of neurons / units                                   
-A [float] : Magnitude of the rescaled external input                   
-$z_0$ [float] : New position of the external input after the sudden change 

