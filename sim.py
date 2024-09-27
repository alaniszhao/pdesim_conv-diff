import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
np.set_printoptions(suppress=True)

def solve(arguments,boundaries,initial, conditions,pde):
    args = arguments
    posX_boundary = boundaries[0][1]
    negX_boundary = boundaries[0][0]
    posY_boundary = boundaries[1][1]
    negY_boundary = boundaries[1][0]
    size,dx,dy,T,dt = conditions
    nt = int(T / dt) #total time points
    var0 = initial

    def laplacian(Z,):
        Z_new = np.zeros_like(Z)

        Ztop = np.roll(Z, 1, axis=0)
        Zleft = np.roll(Z, 1, axis=1)
        Zbottom = np.roll(Z, -1, axis=0)
        Zright = np.roll(Z, -1, axis=1)
        
        Z_new = (Ztop + Zleft + Zbottom + Zright - 4 * Z) / dx**2

        return Z_new

    def gradient_x(c):
        grad_x = np.zeros_like(c)

        grad_x = (np.roll(c, -1, axis=1) - np.roll(c, 1, axis=1)) / (2 * dx)

        return grad_x

    def gradient_y(c):
        grad_y = np.zeros_like(c)
    
        grad_y = (np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)) / (2 * dy)
        
        return grad_y

    def pd_eq(t, y, size, rho,mu,f,p):
        vx = y[:size**2].reshape((size, size))
        vy = y[size**2:].reshape((size, size))

        lap_vx = laplacian(vx)
        lap_vy = laplacian(vy)

        grad_p_x = gradient_x(p)
        grad_p_y = gradient_y(p)

        dvx_dt = (-grad_p_x + mu * lap_vx + f[0]) / rho
        dvy_dt = (-grad_p_y + mu * lap_vy + f[1]) / rho

        return np.concatenate([dvx_dt.ravel(), dvy_dt.ravel()])

    y0 = var0.ravel()

    t_span = (0, T)
    t_eval = np.linspace(0, T, nt)
    sol = solve_ivp(pd_eq, t_span, y0, args=(size, *args), t_eval=t_eval, method='RK45')

    vx_sol = sol.y[:size*size, :].reshape((size, size, len(t_eval)))
    vy_sol = sol.y[size*size:2*size*size, :].reshape((size, size, len(t_eval)))

    time_steps = []
    num = 10
    curr = 0

    for i in range(num):
        time_steps.append(curr)
        curr+= int(nt/num)
    time_steps[num-1]=time_steps[num-1]-1


    extent = [negX_boundary,posX_boundary,negY_boundary,posY_boundary] 

    fig, axes = plt.subplots(1, len(time_steps), figsize=(15, 7))

    for i, t_idx in enumerate(time_steps):
        ax_var = axes[i]
        velocity_magnitude = np.sqrt(vx_sol**2 + vy_sol**2)
        ax_var.imshow(velocity_magnitude[:,:,t_idx], cmap='jet', extent=extent,
                    interpolation='bilinear',)


    plt.show()

negX_boundary =-3
posX_boundary= 3
negY_boundary=-3
posY_boundary=3
X_length = posX_boundary-negX_boundary
Y_length = posY_boundary-negY_boundary
radius = 1

size = 500
dx = float(X_length) / size
dy = float(Y_length) / size
T = 0.3 #time    
dt = 0.001 #change in time

x = np.linspace(negX_boundary,posX_boundary, size)
y = np.linspace(negY_boundary,posY_boundary, size)
X, Y = np.meshgrid(x, y)
vx0 = np.zeros((size,size))
vy0 = np.zeros((size,size))
p = np.zeros((size,size))
var0 = np.concatenate([vx0.ravel(), vy0.ravel()])

rho = 1.0
mu = 0.1
f = [0, -9.81]
pde = 'D * delta_var - (v[0]*gradient_x(var)+v[1]*gradient_y(var)) + R*var'

solve([rho,mu,f,p],
      [[negX_boundary,posX_boundary],
       [negY_boundary,posY_boundary]],
       var0,
       [size, dx, dy, T, dt],
       pde)
