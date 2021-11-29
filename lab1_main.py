import numpy as np
import numpy.linalg as la
import math as m
import plotly.figure_factory as ff
import plotly.graph_objects as go


def func(x):
    res1 = x[0] ** 2 + x[1] ** 2 + x[0] - x[1] - 4
    res2 = 2 * x[0] - x[1]
    return np.maximum(res1, res2)


def sub_grad(x):
    if x[0] ** 2 + x[1] ** 2 <= x[0] + 4:
        g_0 = 2
        g_1 = -1
    else:
        g_0 = 2 * x[0] + 1
        g_1 = 2 * x[1] - 1
    return np.array([g_0, g_1])


# constant step implementation
def optimize(x_0=np.array([0, 0]), step=0.01, acc=0.001, max_iter=10000):
    def find_best_iter(vals):
        n = len(vals)
        k_best = 0
        for k in range(n):
            if vals[k] < vals[k_best]:
                k_best = k
        return k_best

    args = [x_0]
    vals = [func(x_0)]

    iter_count = 0
    for k in range(max_iter):
        iter_count += 1
        g = sub_grad(args[-1])
        args.append(args[-1] - step * g)
        vals.append(func(args[-1]))

        stop_condition = la.norm(args[-1] - args[-2]) < acc and la.norm(vals[-1] - vals[-2]) < acc
        if stop_condition:
            break
    print(f'optimize() reached the end, iterations count: {iter_count}')

    best_iter = find_best_iter(vals)
    print(f'Best iteration index: {best_iter}')

    return args[best_iter], vals[best_iter]


def sub_grad_plot(optimum):
    size = 5
    dpi = 21
    x1 = np.linspace(-size, size, dpi)
    x2 = np.linspace(-size, size, dpi)
    X1, X2 = np.meshgrid(x1, x2)
    g = np.array([sub_grad((x1i, x2i)) for (x1i, x2i) in zip(X1.ravel(), X2.ravel())]).reshape((*X1.shape, 2))
    u = np.array([np.array([dx for (dx, dy) in gi]) for gi in g])
    v = np.array([np.array([dy for (dx, dy) in gi]) for gi in g])
    w = 0.5 * np.sqrt(u ** 2 + v ** 2)
    u /= w
    v /= w
    fig = ff.create_quiver(x=X1, y=X2, u=-u, v=-v, name='gradient')
    fig.add_trace(
        go.Scatter(
            x=[optimum[0]],
            y=[optimum[1]],
            name='optimum',
            line=dict(color='red', width=2)
        )
    )
    fig.update_layout(width=800, height=800)
    return fig


def func_plot(optimum, _width, _height, r=5, dpi=21):
    X1, X2 = np.meshgrid(
        np.linspace(-r, r, dpi),
        np.linspace(-r, r, dpi)
    )
    Z = func((X1, X2))
    fig = go.Figure(data=[
        go.Surface(z=Z, x=X1, y=X2, name='function'),
        go.Scatter3d(
            x=[optimum[0]],
            y=[optimum[1]],
            z=[func(optimum)],
            name='optimum',
            line=dict(color='green', width=5))
    ])
    fig.update_layout(width=_width, height=_height)
    return fig


def main():
    x_opt, f_opt = optimize(x_0=np.array([-1, 1]), step=0.01, acc=0.001, max_iter=50000)
    x_opt_real, f_opt_real = [1 / 2 - m.sqrt(17 / 5), m.sqrt(17 / 5) / 2], 1 - m.sqrt(85) / 2
    print(f'accurate answer: \tx_opt=({(x_opt_real[0]):.4f}, {(x_opt_real[1]):.4f}), f_opt = {f_opt :.4f}')
    print(f'computed answer: \tx_opt=({(x_opt[0]):.4f}, {(x_opt[1]):.4f}), f_opt = {f_opt :.4f}')
    print('drawing plots ...')
    sb_plot = sub_grad_plot(x_opt)
    fn_plot = func_plot(x_opt, 600, 800)
    sb_plot.show()
    fn_plot.show()


if __name__ == '__main__':
    main()
