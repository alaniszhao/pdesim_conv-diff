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

    def laplacian(Z,D):
        Z_new = np.zeros_like(Z)
        D_x = gradient_x(D)
        D_y = gradient_y(D)

        Ztop = np.roll(Z, 1, axis=0)
        Zleft = np.roll(Z, 1, axis=1)
        Zbottom = np.roll(Z, -1, axis=0)
        Zright = np.roll(Z, -1, axis=1)
        
        lap = (Ztop + Zleft + Zbottom + Zright - 4 * Z) / dx**2

        laplacian_with_D = D * lap + D_x * gradient_x(Z) + D_y * gradient_y(Z)

        return laplacian_with_D

    def gradient_x(c):
        grad_x = np.zeros_like(c)

        grad_x = (np.roll(c, -1, axis=1) - np.roll(c, 1, axis=1)) / (2 * dx)

        return grad_x

    def gradient_y(c):
        grad_y = np.zeros_like(c)
    
        grad_y = (np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)) / (2 * dy)
        
        return grad_y

    def pd_eq(t, y, size, D, v_x,v_y, R):
        var = y.reshape((size, size))

        delta_var = laplacian(var, D)

        advection_term = np.multiply(v_x, gradient_x(var)) + np.multiply(v_y, gradient_y(var))

        dvar_dt = delta_var - advection_term + R * var

        return dvar_dt.ravel()

    y0 = var0.ravel()

    t_span = (0, T)
    t_eval = np.linspace(0, T, nt)
    sol = solve_ivp(pd_eq, t_span, y0, args=(size, *args), t_eval=t_eval, method='RK45')

    var_sol = sol.y.reshape((size, size, len(t_eval)))

    time_steps = []
    num = 10
    curr = 0
    for i in range(num):
        time_steps.append(curr)
        curr+= int(nt/num)
    time_steps[num-1]=time_steps[num-1]-1


    extent = [negX_boundary,posX_boundary,negY_boundary,posY_boundary] 

    fig, axes = plt.subplots(1, len(time_steps), figsize=(12, 4))

    for i, t_idx in enumerate(time_steps):
        ax_var = axes[i]
        im_var = ax_var.imshow(var_sol[:, :, t_idx], cmap=plt.cm.cool,
                        interpolation='bilinear', extent=extent)

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
var0 = np.zeros((size, size))
var0[np.sqrt((X+2)**2 + (Y+2)**2) < radius] = 1.0

def gaussian_filter(sigma=1, muu=0):
    x, y = np.meshgrid(np.linspace(negX_boundary, posX_boundary, size),
                       np.linspace(negY_boundary, posY_boundary, size))
    dst = np.sqrt(x**2+y**2)
 
    normal = 1/(2.0 * np.pi * sigma**2)
 
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal
    gauss /= np.max(gauss)
    return gauss

D = gaussian_filter(1,0)
R = 0.0
v_x = np.full((size,size),5)
v_y = np.full((size,size),0)
pde = 'D * delta_var - (v[0]*gradient_x(var)+v[1]*gradient_y(var)) + R*var'

solve([D, v_x,v_y, R],
      [[negX_boundary,posX_boundary],
       [negY_boundary,posY_boundary]],
       var0,
       [size, dx, dy, T, dt],
       pde)
