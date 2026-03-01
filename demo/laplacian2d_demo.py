from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from vectoradam import VectorAdam, create_circle, laplacian_uniform_2d, plot_mesh2d


def _laplacian_step(
    vertices: torch.Tensor,
    edges: torch.Tensor,
    optimizer: VectorAdam,
) -> tuple[np.ndarray, np.ndarray]:
    optimizer.zero_grad()
    vertices_before = vertices.detach().clone()

    laplacian = laplacian_uniform_2d(vertices, edges)
    loss = (vertices * (laplacian @ vertices)).mean()
    loss.backward()
    optimizer.step()

    update = (vertices - vertices_before).detach().cpu().numpy()
    grad_tensor = vertices.grad
    if grad_tensor is None:
        raise RuntimeError("Gradient is None after backward().")
    grad = grad_tensor.detach().cpu().numpy()
    return update, grad


def _plot_update(
    original_mesh: tuple[np.ndarray, np.ndarray],
    gradient: np.ndarray,
    update: np.ndarray,
    update_color: str,
    x_lim: Sequence[float],
    y_lim: Sequence[float],
    show: bool,
    save_path: Path | None,
) -> None:
    plot_result = plot_mesh2d(
        original_mesh[0],
        original_mesh[1],
        x_lim=x_lim,
        y_lim=y_lim,
        return_ax=True,
    )
    if plot_result is None:
        raise RuntimeError("plot_mesh2d(..., return_ax=True) returned None.")
    fig, ax = plot_result
    ax.quiver(
        original_mesh[0][:, 0],
        original_mesh[0][:, 1],
        -gradient[:, 0],
        -gradient[:, 1],
        scale=1.0,
        color="#EC8A19",
        width=0.015,
    )
    ax.quiver(
        original_mesh[0][:, 0],
        original_mesh[0][:, 1],
        update[:, 0],
        update[:, 1],
        scale=2.5,
        color=update_color,
        width=0.015,
    )
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _plot_mesh(
    mesh: tuple[np.ndarray, np.ndarray],
    x_lim: Sequence[float],
    y_lim: Sequence[float],
    show: bool,
    save_path: Path | None,
) -> None:
    if save_path is not None:
        plot_mesh2d(mesh[0], mesh[1], x_lim=x_lim, y_lim=y_lim, filename=str(save_path))
    else:
        plot_mesh2d(mesh[0], mesh[1], x_lim=x_lim, y_lim=y_lim, showfig=show)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Laplacian smoothing demo: Adam-style update vs VectorAdam update."
    )
    parser.add_argument("--n-points", type=int, default=12)
    parser.add_argument("--noise-level", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=5, help="Additional steps after step 1.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--show", action="store_true", help="Show plots interactively.")
    parser.add_argument("--save-dir", type=Path, default=None, help="Optional directory to save plots.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    v_np, l_np = create_circle(n_points=args.n_points, noise_level=args.noise_level)
    x_lim = [float(np.min(v_np[:, 0]) - 1.0), float(np.max(v_np[:, 0]) + 1.0)]
    y_lim = [float(np.min(v_np[:, 1]) - 1.0), float(np.max(v_np[:, 1]) + 1.0)]

    original_mesh = (v_np.copy(), l_np.copy())

    v1 = torch.from_numpy(v_np).to(torch.float32).to(device).requires_grad_(True)
    l1 = torch.from_numpy(l_np).to(torch.long).to(device)
    v2 = torch.from_numpy(v_np).to(torch.float32).to(device).requires_grad_(True)
    l2 = torch.from_numpy(l_np).to(torch.long).to(device)

    betas = (0.9, 0.999)
    eps = 1e-8
    regadam = VectorAdam([{"params": v1, "axis": None}], lr=args.lr, betas=betas, eps=eps)
    vadam = VectorAdam([{"params": v2, "axis": -1}], lr=args.lr, betas=betas, eps=eps)

    _plot_mesh(
        original_mesh,
        x_lim,
        y_lim,
        show=args.show,
        save_path=args.save_dir / "mesh_original.png" if args.save_dir else None,
    )

    adam_update, original_grad = _laplacian_step(v1, l1, regadam)
    radam_mesh_step1 = (v1.detach().cpu().numpy(), l1.detach().cpu().numpy())

    _plot_update(
        original_mesh,
        original_grad,
        adam_update,
        update_color="#226843",
        x_lim=x_lim,
        y_lim=y_lim,
        show=args.show,
        save_path=args.save_dir / "adam_step1_update.png" if args.save_dir else None,
    )
    _plot_mesh(
        radam_mesh_step1,
        x_lim,
        y_lim,
        show=args.show,
        save_path=args.save_dir / "adam_step1_mesh.png" if args.save_dir else None,
    )

    vadam_update, _ = _laplacian_step(v2, l2, vadam)
    vadam_mesh_step1 = (v2.detach().cpu().numpy(), l2.detach().cpu().numpy())

    _plot_update(
        original_mesh,
        original_grad,
        vadam_update,
        update_color="#2A63AD",
        x_lim=x_lim,
        y_lim=y_lim,
        show=args.show,
        save_path=args.save_dir / "vectoradam_step1_update.png" if args.save_dir else None,
    )
    _plot_mesh(
        vadam_mesh_step1,
        x_lim,
        y_lim,
        show=args.show,
        save_path=args.save_dir / "vectoradam_step1_mesh.png" if args.save_dir else None,
    )

    for _ in tqdm(range(args.steps), desc="Adam extra steps"):
        _laplacian_step(v1, l1, regadam)
    radam_mesh_final = (v1.detach().cpu().numpy(), l1.detach().cpu().numpy())

    for _ in tqdm(range(args.steps), desc="VectorAdam extra steps"):
        _laplacian_step(v2, l2, vadam)
    vadam_mesh_final = (v2.detach().cpu().numpy(), l2.detach().cpu().numpy())

    _plot_mesh(
        radam_mesh_final,
        x_lim,
        y_lim,
        show=args.show,
        save_path=args.save_dir / f"adam_step{args.steps}_mesh.png" if args.save_dir else None,
    )
    _plot_mesh(
        vadam_mesh_final,
        x_lim,
        y_lim,
        show=args.show,
        save_path=args.save_dir / f"vectoradam_step{args.steps}_mesh.png" if args.save_dir else None,
    )


if __name__ == "__main__":
    main()
