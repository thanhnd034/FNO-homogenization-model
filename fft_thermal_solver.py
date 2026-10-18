import jax
import jax.numpy as jnp


@jax.jit
def heat_flux(lambda_field, gradT):
    # (Nx, Ny, 1) * (Nx, Ny, 2) -> (Nx, Ny, 2).
    return -lambda_field[..., None] * gradT


@jax.jit
def perturbation_field(lambda0, lambda_field, gradT):
    Q = heat_flux(lambda_field, gradT)
    return lambda0 * gradT + Q


@jax.jit
def fft_gradient_update(P_hat, kvec, k2, lambda0):
    # P_hat, kvec: (Nx, Ny, 2); k2 and T_hat: (Nx, Ny).
    k_dot_P_hat = jnp.einsum('...i,...i->...', kvec, P_hat)
    active = k2 > 0
    safe_k2 = jnp.where(active, k2, 1.0)

    numerator = -1j * k_dot_P_hat
    T_hat = jnp.where(active, numerator / (lambda0 * safe_k2), 0.0 + 0.0j)
    T_hat = T_hat.at[0, 0].set(0.0 + 0.0j)

    gradT_hat = 1j * kvec * T_hat[..., None]
    return jnp.fft.ifft2(gradT_hat, axes=(0, 1)).real


@jax.jit
def thermal_step(lambda_field, gradT, kvec, k2, lambda0, G_macro):
    
    P = perturbation_field(lambda0, lambda_field, gradT)
    P_hat = jnp.fft.fft2(P, axes=(0, 1))
    gradT_fluct_new = fft_gradient_update(P_hat, kvec, k2, lambda0)
    gradT_new = G_macro[None, None, :] + gradT_fluct_new
    diff = jnp.linalg.norm(gradT_new - gradT) / jnp.maximum(jnp.linalg.norm(gradT), 1e-12)
    return gradT_new, diff


def solve_thermal_fft(lambda_field, wavevectors, G_macro, lambda0, *,
                      tol=1e-6, max_iter=500):
    
    if tol <= 0 or max_iter < 0:
        raise ValueError('Use tol > 0 and max_iter >= 0.')

    lambda_field = jnp.asarray(lambda_field)
    G_macro = jnp.asarray(G_macro, dtype=lambda_field.dtype)
    kx, ky, _ = wavevectors
    kvec = jnp.stack([kx, ky], axis=-1)
    
    k2 = kx**2 + ky**2
    gradT = jnp.broadcast_to(G_macro, (*lambda_field.shape, 2))

    for iteration in range(max_iter + 1):
        gradT_new, diff = thermal_step(lambda_field, gradT, kvec, k2, lambda0, G_macro)
        residual = float(diff)
        if residual < tol:
            return jnp.moveaxis(gradT, -1, 0), iteration, residual
        if iteration < max_iter:
            gradT = gradT_new       

    raise RuntimeError(f'FFT reference did not converge: residual={residual:.3e}')
