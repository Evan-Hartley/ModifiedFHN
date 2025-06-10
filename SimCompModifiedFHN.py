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

def run_fitzhugh_nagumo(func, args, pool):
    results = pool.starmap(func, args)

    trimmed_u = []
    trimmed_v = []

    for u_chunk, v_chunk, start, end in results:
        if start == 0:
            u_trimmed = u_chunk[:end - start]
            v_trimmed = v_chunk[:end - start]
        elif end == args[-1][4]:  # args[i][4] is the `end` index
            u_trimmed = u_chunk[1:]
            v_trimmed = v_chunk[1:]
        else:
            u_trimmed = u_chunk[1:-1]
            v_trimmed = v_chunk[1:-1]

        trimmed_u.append(u_trimmed)
        trimmed_v.append(v_trimmed)

    u = np.vstack(trimmed_u)
    v = np.vstack(trimmed_v)

    return u, v


def split_array(arr, num_chunks):
    rows = arr.shape[0]
    chunk_size = rows // num_chunks
    remainder = rows % num_chunks
    chunks = []
    start = 0

    for i in range(num_chunks):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra

        # Add ghost rows
        top = max(start - 1, 0)
        bottom = min(end + 1, rows)
        chunk = arr[top:bottom].copy()
        chunks.append((chunk, start, end))  # Keep track of original indices
        start = end

    return chunks

@timebudget
def update(u, v, params, perturb):
    results = []
    processes_count = 8

    with Pool(processes_count) as pool:
        for t in np.arange(0, params.last_step, params.dt):
            u_chunks = split_array(u, processes_count)
            v_chunks = split_array(v, processes_count)
            args = [(u_chunks[i][0], v_chunks[i][0], params, u_chunks[i][1], u_chunks[i][2]) for i in range(len(u_chunks))]
            u, v = run_fitzhugh_nagumo(fitzhugh_nagumo, args, pool)

            if np.any(np.isclose(t, params.graph_times, atol=1e-8)):
                plot_heatmap(u, v, t)

    return u, v
