import matplotlib.pyplot as plt
import numpy as np
from nsimpkg.random_variables import average_normal_dist, BetaRV
from nsimpkg.mcsim import rho

def plotter(distributions, pi, title, kind="mean", average=False, mix=False, alpha=0.2):

    Niter = len(distributions[0])
    if average:

        distributions_t = np.array(distributions).T
        average_distributions = [average_normal_dist(d) for d in distributions_t]
        if kind=="mean":
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            ax1.plot([d.mu[0] for d in average_distributions], label="Average distribution", color="black")
            if mix:
                ax1.hlines(pi.avg_dist.mu[0], 0, Niter, label="True value", color="red", linestyle="--")
                ax2.hlines(pi.avg_dist.mu[1], 0, Niter, label="True value", color="red", linestyle="--")
            else:
                ax1.hlines(pi.mu[0], 0, Niter, label="True value", color="red", linestyle="--")
                ax2.hlines(pi.mu[1], 0, Niter, label="True value", color="red", linestyle="--")
            ax1.legend(fontsize=12)
            ax1.set_xlabel("Iteration number (log scale)")
            ax1.set_ylabel("Numerical value")
            ax1.set_title("$(\mu_k)_1$")
            ax1.set_xscale("log")

            ax2.plot([d.mu[1] for d in average_distributions], label="Average distribution", color="black")
            ax2.set_xscale("log")
            ax2.set_xlabel("Iteration number (log scale)")
            ax2.set_ylabel("Numerical value")
            ax2.set_title("$(\mu_k)_2$")
            ax2.legend(fontsize=12)
            fig.suptitle(title)
            fig.tight_layout()

        elif kind=="cov":
            fig, axs = plt.subplots(2, 2, figsize=(10,10))
            if mix:
                true00 = pi.avg_dist.Sigma[0][0]
                true10 = pi.avg_dist.Sigma[1][0]
                true01 = pi.avg_dist.Sigma[0][1]
                true11 = pi.avg_dist.Sigma[1][1]
            else:
                true00 = pi.Sigma[0][0]
                true10 = pi.Sigma[1][0]
                true01 = pi.Sigma[0][1]
                true11 = pi.Sigma[1][1]
            sigma00 = [d.Sigma[0][0] for d in average_distributions]
            sigma10 = [d.Sigma[1][0] for d in average_distributions]
            sigma01 = [d.Sigma[0][1] for d in average_distributions]
            sigma11 = [d.Sigma[1][1] for d in average_distributions]
            axs[0, 0].plot(sigma00, color="black", label="Average distribution")
            axs[1, 0].plot(sigma10, color="black", label="Average distribution")
            axs[0, 1].plot(sigma01, color="black", label="Average distribution")
            axs[1, 1].plot(sigma11, color="black", label="Average distribution")

            axs[0, 0].hlines(y=true00, color='r', linestyle='--', xmin=0, xmax=Niter, label="True value")
            axs[0, 0].set_title('$(\Sigma_k)_{1,1}$')
            axs[0, 1].hlines(y=true01, color='r', linestyle='--', xmin=0, xmax=Niter, label="True value")
            axs[0, 1].set_title('$(\Sigma_k)_{1,2}$')
            axs[1, 0].hlines(y=true10, color='r', linestyle='--', xmin=0, xmax=Niter, label="True value")
            axs[1, 0].set_title('$(\Sigma_k)_{2,1}$')
            axs[1, 1].hlines(y=true11, color='r', linestyle='--', xmin=0, xmax=Niter, label="True value")
            axs[1, 1].set_title('$(\Sigma_k)_{2,2}$')
            for ax in axs.flat:
                ax.set_xscale('log')
                ax.set_xlabel("Iteration number (log scale)")
                ax.set_ylabel("Numerical value")
                ax.legend(fontsize=12)
            fig.suptitle(title)
            fig.tight_layout()

    elif not average:
            
        if kind=="mean":

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            for i, distribution_list in enumerate(distributions):
                if i == 0:
                    ax1.plot([d.mu[0] for d in distribution_list], color="black", alpha=alpha, label="Arbitrary experiment", linewidth=1)
                    ax2.plot([d.mu[1] for d in distribution_list], color="black", alpha=alpha, label="Arbitrary experiment", linewidth=1)
                else:
                    mu_1 = [d.mu[0] for d in distribution_list]
                    mu_2 = [d.mu[1] for d in distribution_list]
                    ax1.plot(mu_1, color="black", alpha=alpha)
                    ax2.plot(mu_2, color="black", alpha=alpha)
            if mix:
                ax1.hlines(pi.avg_dist.mu[0], 0, Niter, label="True value", color="red", linestyle="--")
                ax2.hlines(pi.avg_dist.mu[1], 0, Niter, label="True value", color="red", linestyle="--")
            else:
                ax1.hlines(pi.mu[0], 0, Niter, label="True value", color="red", linestyle="--")
                ax2.hlines(pi.mu[1], 0, Niter, label="True value", color="red", linestyle="--")
            ax1.legend(fontsize=12)
            ax1.set_xlabel("Iteration number (log scale)")
            ax1.set_ylabel("Numerical value")
            ax1.set_title("$(\mu_k)_1$")
            ax1.set_xscale("log")

            ax2.set_xscale("log")
            ax2.set_xlabel("Iteration number (log scale)")
            ax2.set_ylabel("Numerical value")
            ax2.set_title("$(\mu_k)_2$")
            ax2.legend(fontsize=12)
            fig.suptitle(title)
            fig.tight_layout()

        elif kind=="cov":
            if mix:
                true = pi.avg_dist.Sigma
            else:
                true = pi.Sigma
            true00 = true[0][0]
            true11 = true[1][1]
            true01 = true[0][1]
            true10 = true[1][0]
            # make 4 subplots
            fig, axs = plt.subplots(2, 2, figsize=(10,10))
            for i, distribution_list in enumerate(distributions):
                if i == 0:
                    sigma00 = [d.Sigma[0][0] for d in distribution_list]
                    sigma10 = [d.Sigma[1][0] for d in distribution_list]
                    sigma01 = [d.Sigma[0][1] for d in distribution_list]
                    sigma11 = [d.Sigma[1][1] for d in distribution_list]
                    axs[0, 0].plot(sigma00, color="black", alpha=alpha, label="Arbitrary experiment", linewidth=1)
                    axs[1, 0].plot(sigma10, color="black", alpha=alpha, label="Arbitrary experiment", linewidth=1)
                    axs[0, 1].plot(sigma01, color="black", alpha=alpha, label="Arbitrary experiment", linewidth=1)
                    axs[1, 1].plot(sigma11, color="black", alpha=alpha, label="Arbitrary experiment", linewidth=1)
                sigma00 = [d.Sigma[0][0] for d in distribution_list]
                sigma10 = [d.Sigma[1][0] for d in distribution_list]
                sigma01 = [d.Sigma[0][1] for d in distribution_list]
                sigma11 = [d.Sigma[1][1] for d in distribution_list]
                axs[0, 0].plot(sigma00, color="black", alpha=alpha)
                axs[1, 0].plot(sigma10, color="black", alpha=alpha)
                axs[0, 1].plot(sigma01, color="black", alpha=alpha)
                axs[1, 1].plot(sigma11, color="black", alpha=alpha)

            axs[0, 0].hlines(y=true00, color='r', linestyle='--', xmin=0, xmax=Niter, label="True value")
            axs[0, 0].set_title('$(\Sigma_k)_{1,1}$')
            axs[0, 1].hlines(y=true01, color='r', linestyle='--', xmin=0, xmax=Niter, label="True value")
            axs[0, 1].set_title('$(\Sigma_k)_{1,2}$')
            axs[1, 0].hlines(y=true10, color='r', linestyle='--', xmin=0, xmax=Niter, label="True value")
            axs[1, 0].set_title('$(\Sigma_k)_{2,1}$')
            axs[1, 1].hlines(y=true11, color='r', linestyle='--', xmin=0, xmax=Niter, label="True value")
            axs[1, 1].set_title('$(\Sigma_k)_{2,2}$')
            for ax in axs.flat:
                ax.set_xscale('log')
                ax.set_xlabel("Iteration number (log scale)")
                ax.set_ylabel("Numerical value")
                ax.legend(fontsize=12)
            fig.suptitle(title)
            fig.tight_layout()

def plot_rho(distributions, pi, title, Nsamples=1000, average=False, ylog=False, alpha=0.2):
    if average:
        distributions_t = np.array(distributions).T
        average_distributions = [average_normal_dist(d) for d in distributions_t]
        rhos = [rho(pi, d, Nsamples=Nsamples) for d in average_distributions]
        plt.plot(rhos, color="black")
        plt.xlabel("Iteration number (log scale)")
        if ylog:
            plt.yscale("log")
            plt.ylabel("Log numerical value")
        else:
            plt.ylabel("Numerical value")
        plt.title(title)
    else:
        for experiment in distributions:
            rhos = [rho(pi, d, Nsamples=Nsamples) for d in experiment]
            plt.plot(rhos, color="black", alpha=alpha)
        plt.xlabel("Iteration number (log scale)")
        if ylog:
            plt.yscale("log")
            plt.ylabel("Log numerical value")
        else:
            plt.ylabel("Numerical value")
        plt.title(title)

def fill_z(p, npoints=100, xlim=[-20, 20], ylim=[-20, 20]):
    xmin, xmax = xlim
    ymin, ymax = ylim
    X = np.linspace(xmin, xmax, npoints)
    Y = np.linspace(ymin, ymax, npoints)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((npoints, npoints))
    for i in range(npoints):
        for j in range(npoints):
            Z[i,j] = p.pdf(np.array([X[i,j], Y[i,j]]))
    return X, Y, Z

def plot_contours(distributions, pi, title, mix=False):
    Niter = len(distributions[0])
    distributions_t = np.array(distributions).T
    average_distributions = [average_normal_dist(d) for d in distributions_t]

    fig, axs = plt.subplots(2, 3, figsize=(20,10))
    #axs[0,0].scatter(out_sq[0,:], out_sq[1,:], s=0.1, c='black', label="Samples from the target distribution", zorder=-5)
    #axs[0,0].scatter(in_sq[0,:], in_sq[1,:], s=0.1, c='red', label="Samples in the unit square", zorder=-5)
    X, Y, Z_pi = fill_z(pi)
    axs[0,0].contourf(X, Y, Z_pi, levels=10, cmap="Greys", zorder=-10)
    axs[0,0].fill_between([-1, 1], [-1, -1], [1, 1], color='red', zorder=5, alpha=0.7, label="Area $D$")
    if not mix:
        pi.construct_ellipse(1, axs[0,0], "blue", label="68% confidence interval")
        pi.construct_ellipse(2, axs[0,0], "green", label="95% confidence interval")
        pi.construct_ellipse(3, axs[0,0], "red", label="99.7% confidence interval")
    axs[0,0].legend(fontsize=6)
    axs[0,0].set_xlim(-20, 20)
    axs[0,0].set_ylim(-20, 20)
    axs[0,0].set_title("Target distribution $\pi$")

    dist_to_get = dist_to_get = [0, Niter//4, 2*Niter//4, 3*Niter//4, Niter-1]


    remaining_axs = [axs[0,1],axs[0,2],axs[1,0],axs[1,1], axs[1,2]]

    for i, ax in zip(dist_to_get, remaining_axs):
        #ax.scatter(out_sq[0,:], out_sq[1,:], s=0.1, c='black', label="Samples from the target distribution", zorder=-5)
        ax.fill_between([-1, 1], [-1, -1], [1, 1], color='red', zorder=5, alpha=0.7, label="Area $D$")
        X, Y, Z_d = fill_z(average_distributions[i])
        ax.contourf(X, Y, Z_d, levels=10, cmap="Greys", zorder=-10)

        average_distributions[i].construct_ellipse(3, ax, "red", label="99.7% confidence interval")
        average_distributions[i].construct_ellipse(2, ax, "green" , label="95% confidence interval")
        average_distributions[i].construct_ellipse(1, ax, "blue", label="68% confidence interval")
        ax.set_ylim(-20, 20)
        ax.set_xlim(-20, 20)
        ax.legend(fontsize=6)
        str_mid = "$q_{\\theta_k}$"
        ax_title = "Proposal{}".format(i) if i != Niter-1 else f"Final proposal distribution at iteration {Niter-1}"
        ax.set_title(ax_title)

    fig.suptitle(title, fontsize=16)
    fig.tight_layout()

# define function to plot parameters over time
def plot_params_beta(experiment_distributions, pi, title, mode="params", average=False, Nsamples = 10000, alpha=0.2, xlog=False, ylog=False):
    Niter = len(experiment_distributions[0])
    alphas =  np.array([np.array([dist.alpha for dist in experiment]) for experiment in experiment_distributions])
    betas =  np.array([np.array([dist.beta for dist in experiment]) for experiment in experiment_distributions])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    if ylog:
        ax2.set_yscale("log")
    if xlog:
        ax2.set_xscale("log")
    
    if mode=="gaussian":
        pi_mean = pi.mean(Nsamples)
        pi_std = pi.std(Nsamples)
    if average:
        average_alphas = np.mean(alphas, axis=0)
        average_betas = np.mean(betas, axis=0)
        if mode=="params":
            # make 2 subplots
            ax1.plot(average_alphas, color="black")
            ax2.plot(average_betas, color="black")
            ax1.set_title("$\\alpha_k$")
            ax2.set_title("$\\beta_k$")
            ax1.set_xlabel("Iteration number")
            ax2.set_xlabel("Iteration number" if not xlog else "Iteration number (log scale)")
            ax1.set_ylabel("Value")
            ax2.set_ylabel("Value")
            fig.suptitle(title)
            plt.tight_layout()
        elif mode=="gaussian":
            means = average_alphas / (average_alphas + average_betas)
            stds = np.sqrt(average_alphas * average_betas / ((average_alphas + average_betas)**2 * (average_alphas + average_betas + 1)))
            ax1.plot(means, color="black")
            ax2.plot(stds, color="black")
            ax1.set_title("Mean")
            ax2.set_title("Standard deviation")
            ax1.set_xlabel("Iteration number")
            ax2.set_xlabel("Iteration number" if not xlog else "Iteration number (log scale)")
            ax1.set_ylabel("Value")
            ax2.set_ylabel("Value")
            ax1.hlines(pi_mean, 0, Niter, color="red", linestyle="--", label="True mean")
            ax2.hlines(pi_std, 0, Niter, color="red", linestyle="--", label="True std")
            fig.suptitle(title)
            ax1.legend()
            ax2.legend()
            plt.tight_layout()
    
    else:
        if mode=="params":
            for alpha_experiment, beta_experiment in zip(alphas, betas):
                ax1.plot(alpha_experiment, color="black", alpha=alpha)
                ax2.plot(beta_experiment, color="black", alpha=alpha)
            ax1.set_title("$\\alpha_k$")
            ax2.set_title("$\\beta_k$")
            ax1.set_xlabel("Iteration number")
            ax2.set_xlabel("Iteration number" if not xlog else "Iteration number (log scale)")
            ax1.set_ylabel("Value")
            ax2.set_ylabel("Value")
            fig.suptitle(title)
            plt.tight_layout()
        elif mode=="gaussian":
            for i in range(len(alphas)):
                alpha_experiment = alphas[i]
                beta_experiment = betas[i]
                means = alpha_experiment / (alpha_experiment + beta_experiment)
                stds = np.sqrt(alpha_experiment * beta_experiment / ((alpha_experiment + beta_experiment)**2 * (alpha_experiment + beta_experiment + 1)))
                if i == 0:
                    ax1.plot(means, color="black", alpha=alpha, label="Arbitrary experiment", linewidth=1)
                    ax2.plot(stds, color="black", alpha=alpha, label="Arbitrary experiment", linewidth=1)
                ax1.plot(means, color="black", alpha=alpha)
                ax2.plot(stds, color="black", alpha=alpha)
            ax1.set_title("Mean")
            ax2.set_title("Standard deviation")
            ax1.set_xlabel("Iteration number")
            ax2.set_xlabel("Iteration number" if not xlog else "Iteration number (log scale)")
            ax1.set_ylabel("Value")
            ax2.set_ylabel("Value")
            ax1.hlines(pi_mean, 0, Niter, color="red", linestyle="--", label="True mean")
            ax2.hlines(pi_std, 0, Niter, color="red", linestyle="--", label="True std")
            ax1.legend()
            ax2.legend()
            fig.suptitle(title)
            plt.tight_layout()

def plot_iters_beta(experiment_distributions, pi, title):
    Niter = len(experiment_distributions[0])
    grid = np.linspace(0.001, 0.999, Niter)
    alphas =  np.array([np.array([dist.alpha for dist in experiment]) for experiment in experiment_distributions])
    betas =  np.array([np.array([dist.beta for dist in experiment]) for experiment in experiment_distributions])
    average_alphas = np.mean(alphas, axis=0)
    average_betas = np.mean(betas, axis=0)
    fig, axs = plt.subplots(2, 3, figsize=(20,10))
    proposal_to_get = np.logspace(0, np.log10(Niter), 6, dtype=int)
    proposal_to_get[proposal_to_get==1] = 0
    for i, proposal in enumerate(proposal_to_get):
        ax = axs[i//3, i%3]
        alpha_plot = average_alphas[proposal]
        beta_plot = average_betas[proposal]
        proposal_plot = BetaRV(alpha_plot, beta_plot)
        proposal_plot_pdf = proposal_plot.pdf(grid)
        ax.plot(grid, proposal_plot_pdf, color="red", label="Proposal distribution")
        ax.plot(grid, pi.pdf(grid), color="black", label="True distribution")
        ax.set_title("Iteration {}".format(proposal))
        ax.set_xlabel("$x$")
        ax.set_ylabel("$p(x)$")
        ax.scatter([0.25, 0.75], [0,0], marker="|", color="blue", zorder=10000, linewidth=2)
        ax.plot([0.25, 0.75], [0,0], color="blue", zorder=1000, linewidth=2, label="Interval $D$")
        ax.set_ylim(0, 1.7)
        ax.legend()
    fig.suptitle(title, fontsize=20)
    plt.tight_layout()

def plot_mse(results_list, GT, title, xlog = False, ylog=False):
    results = np.array(results_list)
    mses = np.mean((results-GT)**2, axis=0)
    plt.plot(mses, color="black", linewidth=1)
    plt.title(title, fontsize=20)
    if xlog:
        plt.xscale("log")
        plt.xlabel("Iteration number (log scale)")
    else:
        plt.xlabel("Iteration number")
    if ylog:
        plt.yscale("log")
        plt.ylabel("MSE (log scale)")
    else:
        plt.ylabel("MSE")
    plt.tight_layout()