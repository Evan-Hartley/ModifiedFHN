import numpy as np
from timebudget import timebudget
from multiprocessing import Pool

def laplacian(Z, params):
    Z_top = np.roll(Z, 1, axis=0)
    Z_bottom = np.roll(Z, -1, axis=0)
    Z_left = np.roll(Z, 1, axis=1)
    Z_right = np.roll(Z, -1, axis=1)
    return (Z_top + Z_bottom + Z_left + Z_right - 4 * Z) / (params.dx ** 2)

def fitzhugh_nagumo(u, v, params):
    lap_u = laplacian(u, params)
    u_new = u + params.dt * (params.D_u * lap_u + params.mu * u * (1 - u) * (u - params.alpha) - v * u)
    v_new = v + params.dt * (params.epsilon * ((params.beta - u) * (u - params.gamma) - params.delta*v - params.theta))

    # Apply no-flux boundary conditions
    u_new[0, :] = u_new[1, :]
    u_new[-1, :] = u_new[-2, :]
    u_new[:, 0] = u_new[:, 1]
    u_new[:, -1] = u_new[:, -2]

    v_new[0, :] = v_new[1, :]
    v_new[-1, :] = v_new[-2, :]
    v_new[:, 0] = v_new[:, 1]
    v_new[:, -1] = v_new[:, -2]

    return u_new, v_new

def run_fitzhugh_nagumo(operation, args, pool):
    results = pool.starmap(fitzhugh_nagumo, args)
    u = np.vstack([r[0] for r in results])
    v = np.vstack([r[1] for r in results])
    return u, v

def split_array(arr, num_chunks):
    return np.array_split(arr, num_chunks, axis=0)

@timebudget
def update(u, v, params):
    results = []
    processes_count = 8

    with Pool(processes_count) as pool:
        for t in np.arange(0, params.last_step, params.dt):
            u_chunks = split_array(u, processes_count)
            v_chunks = split_array(v, processes_count)
            args = [(u_chunks[i], v_chunks[i], params) for i in range(processes_count)]
            u, v = run_fitzhugh_nagumo(fitzhugh_nagumo, args, pool)

    return u, v
