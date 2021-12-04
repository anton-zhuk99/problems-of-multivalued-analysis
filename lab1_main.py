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
    g = [x[0], x[1]]
    if x[0] ** 2 + x[1] ** 2 < x[0] + 4:
        g[0] = 2
        g[1] = -1
    elif x[0] ** 2 + x[1] ** 2 == x[0] + 4:
        g[0] = np.random.uniform(-2, 2, g[0].shape)
        g[1] = np.random.uniform(-2, 2, g[1].shape)
    else:
        g[0] = 2 * x[0] + 1
        g[1] = 2 * x[1] - 1
    return np.array(g)


# constant step implementation
def optimize(x_0=np.array([0, 0]), step=0.01, max_iter=10000):
    def find_best_iter(vals):
        n = len(vals)
        k_best = 0
        for k in range(n):
            if vals[k] < vals[k_best]:
                k_best = k
        return k_best

    print(f'optimization parameters: x0=({x_0[0]}, {x_0[1]}), step={step}, iterations={max_iter}')

    args = [x_0]
    vals = [func(x_0)]

    iter_count = 0
    for k in range(max_iter):
        iter_count += 1
        g = sub_grad(args[-1])
        args.append(args[-1] - step * g)
        vals.append(func(args[-1]))
    print('optimize() reached the end')

    best_iter = find_best_iter(vals)
    print(f'best iteration index: {best_iter}')

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
    # draw gradient
    fig = ff.create_quiver(x=X1, y=X2, u=-u, v=-v, name='gradient')
    # draw optimum
    fig.add_trace(
        go.Scatter(
            x=[optimum[0]],
            y=[optimum[1]],
            name='optimum',
            line=dict(color='red', width=2)
        )
    )
    # draw circle
    fig.add_trace(
        go.Contour(
            z=X1 ** 2 + X2 ** 2 - X1 - 4,
            x=x1,
            y=x2,
            name='$x_1^2+x_2^2-x_1-4=0$',
            line=dict(
                width=2,
                color='orange'
            ),
            contours=dict(
                coloring='lines',
                start=0,
                end=0,
                size=2,
            ),
            showlegend=True
        )
    )
    fig.update_layout(width=800, height=800)
    return fig


def func_plot(optimum, _width, _height, r=5, dpi=21):
    # calculate surface points
    X1, X2 = np.meshgrid(
        np.linspace(-r, r, dpi),
        np.linspace(-r, r, dpi)
    )
    Z = func((X1, X2))
    # draw circle
    x1 = np.linspace(-r, r, dpi)
    x21 = np.sqrt(-x1 ** 2 + x1 + 4)
    x22 = - np.sqrt(-x1 ** 2 + x1 + 4)
    z1 = func((x1, x21))
    z2 = func((x1, x22))
    x1_data = np.concatenate([x1, np.flip(x1)])
    x2_data = np.concatenate([x21, np.flip(x22)])
    z_data = np.concatenate([z1, np.flip(z2)])
    # filter data
    x1_data_clean = []
    x2_data_clean = []
    z_data_clean = []
    num = x1_data.shape[0]
    for i in range(num):
        if not np.isnan(x2_data[i]):
            x1_data_clean.append(x1_data[i])
            x2_data_clean.append(x2_data[i])
            z_data_clean.append(z_data[i])
    fig = go.Figure(data=[
        # draw optimum
        go.Surface(
            z=Z,
            x=X1,
            y=X2,
            name='function',
            showlegend=True
        ),
        # draw optimum
        go.Scatter3d(
            x=[optimum[0]],
            y=[optimum[1]],
            z=[func(optimum)],
            name='optimum',
            line=dict(color='green', width=2)),
        # draw circle
        go.Scatter3d(
            z=np.array(z_data_clean),
            x=np.array(x1_data_clean),
            y=np.array(x2_data_clean),
            name='$x_1^2+x_2^2-x_1-4=0$',
            line=dict(
                color='orange',
                width=1
            ),
            connectgaps=True
        ),
        # fill the gap
        go.Scatter3d(
            z=np.array([z_data_clean[0], z_data_clean[-1]]),
            x=np.array([x1_data_clean[0], x1_data_clean[-1]]),
            y=np.array([x2_data_clean[0], x2_data_clean[-1]]),
            connectgaps=True,
            line=dict(
                color='orange',
                width=1
            ),
            showlegend=False
        )
    ])
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        width=_width,
        height=_height
    )
    return fig


def main():
    x_opt, f_opt = optimize(x_0=np.array([-100, 100]), step=0.0001, max_iter=70000)
    x_opt_real, f_opt_real = [1 / 2 - m.sqrt(17 / 5), m.sqrt(17 / 5) / 2], 1 - m.sqrt(85) / 2
    print(f'accurate answer: \tx_opt=({(x_opt_real[0]):.6f}, {(x_opt_real[1]):.6f}), f_opt = {f_opt_real :.6f}')
    print(f'computed answer: \tx_opt=({(x_opt[0]):.6f}, {(x_opt[1]):.6f}), f_opt = {f_opt :.6f}')
    print('drawing plots ...')
    sb_plot = sub_grad_plot(x_opt)
    fn_plot = func_plot(x_opt, 1000, 800)
    sb_plot.show()
    fn_plot.show()


if __name__ == '__main__':
    main()
