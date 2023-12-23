import numpy as np
from matplotlib import pyplot as plt
from numpy.lib import poly1d
import seaborn as sns


def newton_iterations(max_iter: int, X: np.ndarray, p: np.poly1d, eps: float):
    res = X.shape[0]
    S = np.zeros(shape=(res, res))
    X0 = X.copy()
    dp = p.deriv()
    for _ in range(max_iter):
        X = X - p(X) / dp(X)
        S[X - X0 > eps] += 1
        X0 = X.copy()
    return X, S


def generate_fractal(
    p: poly1d = poly1d([1, 0, 0, 1]),
    limits: list = [-1, 1, -1, 1],
    resolution: int = 2048,
    eps: float = 1e-8,
):
    # IMPLEMENTATION
    X, Y = np.meshgrid(
        np.linspace(limits[0], limits[1], resolution),
        np.linspace(limits[2], limits[3], resolution),
    )
    Z = X + Y * 1j

    Z, S = newton_iterations(max_iter=30, X=Z, p=p, eps=eps)

    img = np.zeros(shape=(resolution, resolution, 3))

    roots = p.roots
    colors = sns.color_palette("hls", n_colors=len(roots))
    for i, root in enumerate(roots):
        img[abs(Z - root) < eps] = colors[i]

    for i, line in enumerate(img):
        for j, color in enumerate(line):
            img[i][j] = (
                max(color[0] - 10 * S[i][j] / 256, 0),
                max(color[1] - 10 * S[i][j] / 256, 0),
                max(color[2] - 10 * S[i][j] / 256, 0),
            )
    return img


def main():
    limits = [-1, 1, -1, 1]
    img = generate_fractal(limits=limits)
    plt.imsave("generated_fractal_shaders.png", img)
    _, axs = plt.subplots()
    axs.imshow(img, extent=limits)
    plt.show()


if __name__ == "__main__":
    main()
