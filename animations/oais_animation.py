from nsimpkg.gaussian_OAIS import SG_OAIS, ADAM_OAIS
from nsimpkg.random_variables import NormalRV
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from nsimpkg.mcsim import rho
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt



def fill_z(p, npoints=100):
    X = np.linspace(-20, 20, npoints)
    Y = np.linspace(-20, 20, npoints)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((npoints, npoints))
    for i in range(npoints):
        for j in range(npoints):
            Z[i,j] = p.pdf(np.array([X[i,j], Y[i,j]]))
    return X, Y, Z

q = NormalRV(np.array([10, -10]), np.array([[40, 0], [0, 40]]))
pi = NormalRV(np.array([1, -1]), np.array([[2, -0.5], [-0.5, 2]]))

def phi(x):
    return (np.sum(np.abs(x) < 1, axis=0) == 2).astype(int)
GT = 0.195595
N = 100
Niter = int(1e4)
alpha = 1e-1

results, distributions = ADAM_OAIS(phi, pi, q, N, Niter, alpha=alpha)

npoints_plot = 100
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(xlim = (-20, 20), ylim = (-20, 20), xlabel = "$x_1$", ylabel = "$x_2$")
X, Y, Z_pi = fill_z(pi, npoints=npoints_plot)
ax.contourf(X, Y, Z_pi, cmap="Blues", zorder=0, alpha=0.5)
pi_patch = mpatches.Patch(color='blue', alpha=0.5, label='$\pi$')

X, Y, Z_q = fill_z(q, npoints=npoints_plot)
cont = ax.contourf(X, Y, Z_q, zorder=0, alpha=0.5, cmap="Greys")
q_patch = mpatches.Patch(color='grey', alpha=0.5, label='$q_{\\theta}$')
ax.fill_between([-1, 1], [-1, -1], [1, 1], color='red', zorder=5, alpha=0.7, label="Area $D$")
q.construct_ellipse(1, ax, "blue", label="68% confidence interval")
q.construct_ellipse(2, ax, "green", label="95% confidence interval")
q.construct_ellipse(3, ax, "red", label="99.7% confidence interval")
handles, labels = ax.get_legend_handles_labels()
handles.append(pi_patch)
handles.append(q_patch)

plt.legend(handles=handles, loc='upper right')
ax.set_title("ADAM OAIS with normal target and proposal, iteration 0", fontsize=20)
updated_Zs = [fill_z(d, npoints=npoints_plot)[-1] for d in distributions]
print("computed this")

def update(i):
    global cont
    q_i = distributions[i]
    Z_updated = updated_Zs[i]
    for c in cont.collections:
        c.remove()

    cont = plt.contour(X, Y, Z_updated, zorder=0, alpha=0.5, cmap="Greys")
    # remove old confidence ellipses, which are plotted as patches instead of Ellipses
    for c in ax.patches:
        if c.get_label() in ["68% confidence interval", "95% confidence interval", "99.7% confidence interval"]:
            c.remove()

    # add new confidence ellipses
    q_i.construct_ellipse(1, ax, "blue", label="68% confidence interval")
    q_i.construct_ellipse(2, ax, "green", label="95% confidence interval")
    q_i.construct_ellipse(3, ax, "red", label="99.7% confidence interval")
    ax.set_title("ADAM OAIS with normal target and proposal, iteration {}".format(i), fontsize=20)
    return cont


fig.tight_layout()
ani = animation.FuncAnimation(fig=fig, func=update, frames=len(distributions), interval=1)
Writer = animation.writers['ffmpeg']
writer = Writer(fps=30, metadata=dict(artist='Me'), bitrate=1800)
ani.save('AdamOAIS.mp4', writer=writer)
print("done saving")
plt.show()
