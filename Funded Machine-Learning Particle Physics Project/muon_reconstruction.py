import copy
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import MultipleLocator

import pandas as pd
from sklearn.model_selection import train_test_split

from torch import optim
from nflows.transforms.base import CompositeTransform
from nflows.transforms.permutations import RandomPermutation
from nflows.transforms.autoregressive import (
    MaskedAffineAutoregressiveTransform,
)
from nflows.distributions.normal import StandardNormal
from nflows.flows.base import Flow
import uproot

# =============================================================================
# Configuration
# =============================================================================

USE_TOY_DATA = True

NEVENTS = 10_000
TEST_SIZE = 0.2

MUON_MASS_GEV = 0.105658
M_Z = 91.1876

BATCH_SIZE = 1024

N_STEPS = 10_000
LR = 1e-4

N_LAYERS = 6
HIDDEN_FEATURES = 128


plt.rcParams.update({

    "figure.figsize": (7.2, 6.4),
    "figure.dpi": 120,

    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11,

    # Axes
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "axes.linewidth": 1.0,

    # Ticks
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,

    # Legend
    "legend.fontsize": 10,
    "legend.frameon": False,

    # Output
    "savefig.dpi": 300,
    "savefig.bbox": "tight",

    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "data": "#222222",
    "mc": "#377eb8",
    "transformed": "#e41a1c",
    "reference": "#666666",
}

MASS_LOSS_WEIGHT = 1.0
MOMENTUM_LOSS_WEIGHT = 0.0
REGULARISATION_WEIGHT = 0.001

# Invariant-mass plotting/training window

XMIN = 30.0
XMAX = 120.0
N_BINS = 50

KDE_WIDTH = 0.5

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

DATA_FILE_PATH = "Zmumu.csv"
MC_file = uproot.open("13TeV_2016_Down_Z_Sim09h_42112001.root")

df = pd.read_csv(DATA_FILE_PATH)


# =============================================================================
# Reproducibility
# =============================================================================

def get_data(filetype, muon):
    return np.stack([filetype['Z/DecayTree']['mu'+muon+'_PT'].array(library='np'),
                     filetype['Z/DecayTree']['mu'+muon+'_ETA'].array(library='np'),
                     filetype['Z/DecayTree']['mu'+muon+'_PHI'].array(library='np')])

mup_MC = get_data(MC_file, 'p')
mum_MC = get_data(MC_file, 'm')


def set_seed(seed=42):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Toy data generation
# =============================================================================

# def generate_toy_data(mean, std_dev, nevents):

#     ptm = torch.Tensor(np.random.normal(mean, std_dev, nevents).astype(np.float32))
#     etam = torch.Tensor(np.random.uniform(2.0, 4.5, nevents).astype(np.float32))
#     phim = torch.Tensor(np.random.uniform(-np.pi, np.pi, nevents).astype(np.float32))

#     ptp = torch.Tensor(np.random.normal(mean, std_dev, nevents).astype(np.float32))
#     etap = torch.Tensor(np.random.uniform(2.0, 4.5, nevents).astype(np.float32))
#     phip = torch.Tensor(np.random.uniform(-np.pi, np.pi, nevents).astype(np.float32))

#     return torch.stack([ptm, etam, phim, ptp, etap, phip], dim=1)


# =============================================================================
# Real data loading
# =============================================================================

def get_data(
    momentum_unit="MeV",
    nevents=NEVENTS,
):

    cols = [
        "pt1",
        "eta1",
        "phi1",
        "pt2",
        "eta2",
        "phi2",
    ]

    arrays = [
        df[col].to_numpy()
        for col in cols
    ]

    data = torch.tensor(
        np.stack(
            arrays,
            axis=1,
        ),
        dtype=torch.float32,
    )

    if momentum_unit == "MeV":

        data[:, [0, 3]] /= 1000.0

    elif momentum_unit != "GeV":

        raise ValueError(
            "momentum_unit must be either 'MeV' or 'GeV'"
        )

    return data[:nevents]


# =============================================================================
# Kinematics
# =============================================================================

def get_four_momenta(muon_data):

    """
    Convert:

        (pt, eta, phi) x 2

    into:

        (px, py, pz, E) x 2
    """

    pt1 = muon_data[:, 0]
    eta1 = muon_data[:, 1]
    phi1 = muon_data[:, 2]

    pt2 = muon_data[:, 3]
    eta2 = muon_data[:, 4]
    phi2 = muon_data[:, 5]

    px1 = pt1 * torch.cos(phi1)
    py1 = pt1 * torch.sin(phi1)
    pz1 = pt1 * torch.sinh(eta1)

    px2 = pt2 * torch.cos(phi2)
    py2 = pt2 * torch.sin(phi2)
    pz2 = pt2 * torch.sinh(eta2)

    E1 = torch.sqrt(
        px1**2
        + py1**2
        + pz1**2
        + MUON_MASS_GEV**2
    )

    E2 = torch.sqrt(
        px2**2
        + py2**2
        + pz2**2
        + MUON_MASS_GEV**2
    )

    return torch.stack(
        [
            px1,
            py1,
            pz1,
            E1,
            px2,
            py2,
            pz2,
            E2,
        ],
        dim=1,
    )


def calculate_invariant_mass(muon_data):

    four_momenta = get_four_momenta(
        muon_data
    )

    E_total = (
        four_momenta[:, 3]
        + four_momenta[:, 7]
    )

    px_total = (
        four_momenta[:, 0]
        + four_momenta[:, 4]
    )

    py_total = (
        four_momenta[:, 1]
        + four_momenta[:, 5]
    )

    pz_total = (
        four_momenta[:, 2]
        + four_momenta[:, 6]
    )

    mass_squared = (
        E_total**2
        - px_total**2
        - py_total**2
        - pz_total**2
    )

    return torch.sqrt(
        torch.clamp(
            mass_squared,
            min=1e-8,
        )
    )


# =============================================================================
# Standardisation
# =============================================================================

def fit_standardisation(data):

    mean = data.mean(
        dim=0,
        keepdim=True,
    )

    std = data.std(
        dim=0,
        keepdim=True,
    ).clamp_min(1e-6)

    return mean, std


def standardise(
    data,
    mean,
    std,
):

    return (
        data - mean
    ) / std


def destandardise(
    data,
    mean,
    std,
):

    return (
        data * std
    ) + mean


# =============================================================================
# Normalising flow
# =============================================================================

def make_flow(
    features=6,
    num_layers=N_LAYERS,
    hidden_features=HIDDEN_FEATURES,
):

    transforms = []

    for _ in range(num_layers):

        transforms.append(
            RandomPermutation(
                features=features
            )
        )

        transforms.append(
            MaskedAffineAutoregressiveTransform(
                features=features,
                hidden_features=hidden_features,
            )
        )

    return CompositeTransform(
        transforms
    )


# =============================================================================
# Differentiable soft histogram
# =============================================================================
def soft_histogram(
    values,
    bin_centres,
    bandwidth,
):

    distances = (
        values[:, None]
        - bin_centres[None, :]
    )

    weights = torch.exp(
        -0.5
        * (
            distances
            / bandwidth
        ) ** 2
    )

    weights = weights / (
        weights.sum(
            dim=1,
            keepdim=True,
        )
        + 1e-8
    )

    histogram = weights.mean(
        dim=0
    )

    return histogram / (
        histogram.sum()
        + 1e-8
    )
def mass_distribution_loss(
    mc_mass,
    data_mass,
    bin_centres,
):

    mc_pdf = soft_histogram(
        mc_mass,
        bin_centres,
        KDE_WIDTH,
    )

    data_pdf = soft_histogram(
        data_mass,
        bin_centres,
        KDE_WIDTH,
    )

    return torch.mean(
        (
            mc_pdf
            - data_pdf
        ) ** 2
    )


def regularisation_loss(
    transformed_output,
    transformed_input,
):

    return torch.mean(
        (
            transformed_output
            - transformed_input
        ) ** 2
    )


# =============================================================================
# Quantitative metrics
# =============================================================================

def wasserstein_distance_1d(
    x,
    y,
):

    """
    Empirical 1D Wasserstein distance.

    Both distributions are sorted and interpolated onto the same
    quantile grid.
    """

    x = np.sort(x)
    y = np.sort(y)

    n = min(
        len(x),
        len(y),
    )

    quantiles = np.linspace(
        0.0,
        1.0,
        n,
    )

    x_quantiles = np.quantile(
        x,
        quantiles,
    )

    y_quantiles = np.quantile(
        y,
        quantiles,
    )

    return np.mean(
        np.abs(
            x_quantiles
            - y_quantiles
        )
    )


def chi_squared_distance(
    data,
    mc,
    bins,
):

    data_counts, _ = np.histogram(
        data,
        bins=bins,
    )

    mc_counts, _ = np.histogram(
        mc,
        bins=bins,
    )

    # Normalise MC to the same total event count
    mc_counts = (
        mc_counts
        * data_counts.sum()
        / max(
            mc_counts.sum(),
            1,
        )
    )

    denominator = np.maximum(
        data_counts,
        1,
    )

    return np.sum(
        (
            mc_counts
            - data_counts
        ) ** 2
        / denominator
    )


# =============================================================================
# Plotting utilities
# =============================================================================
def moving_average(values, window=200):

    values = np.asarray(values)

    if len(values) < window:

        return values

    kernel = np.ones(window) / window

    return np.convolve(
        values,
        kernel,
        mode="valid",
    )
def plot_training_history(
    steps,
    total_losses,
    mass_losses,
    regularisation_losses,
    filename,
):

    steps = np.asarray(list(steps))
    mass_losses = np.asarray(mass_losses)
    regularisation_losses = np.asarray(regularisation_losses)

    weighted_mass = MASS_LOSS_WEIGHT * mass_losses
    weighted_reg = REGULARISATION_WEIGHT * regularisation_losses

    window = 200

    total_smooth = moving_average(weighted_mass + weighted_reg, window)
    mass_smooth = moving_average(weighted_mass, window)
    reg_smooth = moving_average(weighted_reg, window)
    smooth_steps = steps[window - 1:] if len(steps) >= window else steps

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    ax.plot(smooth_steps, total_smooth, color="black",
            linewidth=1.8, label="Total loss")
    ax.plot(smooth_steps, mass_smooth, color=COLORS["transformed"],
            linewidth=1.8, label=r"Mass loss $\times$ weight")
    ax.plot(smooth_steps, reg_smooth, color=COLORS["mc"],
            linewidth=1.8, label=r"Regularisation $\times$ weight")

    ax.set_xlabel("Training step")
    ax.set_ylabel("Weighted loss contribution")
    ax.set_yscale("log")
    ax.set_xlim(0, steps[-1])

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(
                direction="in",
                which="both",
                top=True,
                right=True,
            )

    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.legend(loc="upper right", frameon=False)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()
    fig.savefig(filename + ".png", dpi=400, bbox_inches="tight")
    fig.savefig(filename + ".pdf", bbox_inches="tight")

    plt.show()
    plt.close(fig)




def plot_mass_distribution(
    data_mass,
    mc_mass,
    transformed_mass,
    filename,
    sample_label="Validation sample",
    x_min=60.0,
    x_max=110.0,
    n_bins=25,
):

    # ------------------------------------------------------------------
    # Convert to clean NumPy arrays
    # ------------------------------------------------------------------

    data_mass = np.asarray(data_mass, dtype=float)
    mc_mass = np.asarray(mc_mass, dtype=float)
    transformed_mass = np.asarray(
        transformed_mass,
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Remove invalid values and restrict to physics window
    # ------------------------------------------------------------------

    def clean_mass(sample):

        sample = sample[
            np.isfinite(sample)
        ]

        sample = sample[
            (sample >= x_min)
            & (sample <= x_max)
        ]

        return sample

    data_mass = clean_mass(data_mass)
    mc_mass = clean_mass(mc_mass)
    transformed_mass = clean_mass(
        transformed_mass
    )

    # ------------------------------------------------------------------
    # Common binning
    # ------------------------------------------------------------------

    bins = np.linspace(
        x_min,
        x_max,
        n_bins + 1,
    )

    bin_centres = 0.5 * (
        bins[:-1]
        + bins[1:]
    )

    bin_widths = np.diff(bins)

    # ------------------------------------------------------------------
    # Normalised distributions
    # ------------------------------------------------------------------

    data_counts, _ = np.histogram(
        data_mass,
        bins=bins,
    )

    mc_counts, _ = np.histogram(
        mc_mass,
        bins=bins,
    )

    transformed_counts, _ = np.histogram(
        transformed_mass,
        bins=bins,
    )

    # Normalise to probability density
    data_density = (
        data_counts
        / max(
            data_counts.sum(),
            1,
        )
        / bin_widths
    )

    mc_density = (
        mc_counts
        / max(
            mc_counts.sum(),
            1,
        )
        / bin_widths
    )

    transformed_density = (
        transformed_counts
        / max(
            transformed_counts.sum(),
            1,
        )
        / bin_widths
    )

    # ------------------------------------------------------------------
    # Statistical uncertainty on DATA
    # ------------------------------------------------------------------

    data_uncertainty = np.zeros_like(
        data_density
    )

    valid_data = (
        data_counts > 0
    )

    data_uncertainty[
        valid_data
    ] = (
        np.sqrt(
            data_counts[
                valid_data
            ]
        )
        / data_counts.sum()
        / bin_widths[
            valid_data
        ]
    )

    # ------------------------------------------------------------------
    # Ratios
    # ------------------------------------------------------------------

    valid_ratio = (
        data_density > 0
    )

    mc_ratio = np.full(
        len(data_density),
        np.nan,
    )

    transformed_ratio = np.full(
        len(data_density),
        np.nan,
    )

    mc_ratio[
        valid_ratio
    ] = (
        mc_density[
            valid_ratio
        ]
        / data_density[
            valid_ratio
        ]
    )

    transformed_ratio[
        valid_ratio
    ] = (
        transformed_density[
            valid_ratio
        ]
        / data_density[
            valid_ratio
        ]
    )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    chi2_mc = np.sum(
        (
            mc_counts[
                valid_data
            ]
            - data_counts[
                valid_data
            ]
        ) ** 2
        / np.maximum(
            data_counts[
                valid_data
            ],
            1,
        )
    )

    chi2_transformed = np.sum(
        (
            transformed_counts[
                valid_data
            ]
            - data_counts[
                valid_data
            ]
        ) ** 2
        / np.maximum(
            data_counts[
                valid_data
            ],
            1,
        )
    )

    ndf = np.sum(
        valid_data
    ) - 1

    chi2_ndf_mc = (
        chi2_mc
        / max(
            ndf,
            1,
        )
    )

    chi2_ndf_transformed = (
        chi2_transformed
        / max(
            ndf,
            1,
        )
    )

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------

    fig = plt.figure(
        figsize=(7.2, 6.4)
    )

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[
            3.0,
            1.5,
        ],
        hspace=0.03,
    )

    ax = fig.add_subplot(
        gs[0]
    )

    ax_ratio = fig.add_subplot(
        gs[1],
        sharex=ax,
    )

    # ------------------------------------------------------------------
    # Upper panel
    # ------------------------------------------------------------------

    ax.errorbar(
    bin_centres,
    data_density,
    yerr=data_uncertainty,
    fmt="o",
    markersize=4,
    color=COLORS["data"],
    ecolor=COLORS["data"],
    elinewidth=1.2,
    capsize=2,
    label="Data",
)

    ax.stairs(
        mc_density,
        bins,
        color=COLORS["mc"],
        linewidth=2.0,
        linestyle="--",
        label="Original MC",
    )

    ax.stairs(
        transformed_density,
        bins,
        color=COLORS["transformed"],
        linewidth=2.2,
        label="Transformed MC",
    )

    # ------------------------------------------------------------------
    # Determine sensible y-axis range automatically
    # ------------------------------------------------------------------

    all_y_values = np.concatenate(
        [
            data_density[
                data_density > 0
            ],
            mc_density[
                mc_density > 0
            ],
            transformed_density[
                transformed_density > 0
            ],
        ]
    )

    y_max = np.max(
        all_y_values
    )

    ax.set_ylim(
        0.0,
        1.10 * y_max,
    )

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_ylabel(
        "Normalised density",
    )

    ax.tick_params(
        labelbottom=False,
    )

    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # ------------------------------------------------------------------
    # Physics annotation
    # ------------------------------------------------------------------

    ax.text(
        0.025,
        0.965,
        r"$Z \rightarrow \mu^+\mu^-$",
        transform=ax.transAxes,
        fontsize=17,
        ha="left",
        va="top",
    )

    ax.text(
        0.025,
        0.885,
        sample_label,
        transform=ax.transAxes,
        fontsize=13,
        ha="left",
        va="top",
    )

    # ------------------------------------------------------------------
    # Statistics annotation
    # ------------------------------------------------------------------

    # statistics_text = (
    #     r"$\chi^2/\mathrm{ndf}$ (MC)"
    #     f" = {chi2_ndf_mc:.2f}\n"
    #     r"$\chi^2/\mathrm{ndf}$ (transformed)"
    #     f" = {chi2_ndf_transformed:.2f}"
    # )

    print(f"$\chi^2/\mathrm{ndf}$ (MC) = {chi2_ndf_mc:.2f}\n"
              f"$\chi^2/\mathrm{ndf}$ (transformed) = {chi2_ndf_transformed:.2f}")

    # ax.text(
    #     0.97,
    #     0.95,
    #     statistics_text,
    #     transform=ax.transAxes,
    #     fontsize=10.5,
    #     ha="right",
    #     va="top",
    #     bbox=dict(
    #         facecolor="white",
    #         edgecolor="none",
    #         alpha=0.85,
    #         pad=3.0,
    #     ),
    # )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(
            0.99,
            0.97,
        ),
        frameon=False,
        handlelength=2.8,
    )

    # ------------------------------------------------------------------
    # Lower ratio panel
    # ------------------------------------------------------------------

    ax_ratio.axhline(
        1.0,
        color="black",
        linewidth=1.0,
    )

    ax_ratio.axhline(
        0.8,
        color="0.65",
        linestyle="--",
        linewidth=0.8,
    )

    ax_ratio.axhline(
        1.2,
        color="0.65",
        linestyle="--",
        linewidth=0.8,
    )

    ax_ratio.stairs(
    mc_ratio,
    bins,
    color=COLORS["mc"],
    linewidth=1.8,
    linestyle="--",
    label="MC / Data",
)

    ax_ratio.stairs(
        transformed_ratio,
        bins,
        color=COLORS["transformed"],
        linewidth=2.0,
        label="Transformed MC / Data",
    )

    # Data uncertainty in ratio
    ratio_uncertainty = np.full(
        len(data_density),
        np.nan,
    )

    ratio_uncertainty[
        valid_ratio
    ] = (
        data_uncertainty[
            valid_ratio
        ]
        / data_density[
            valid_ratio
        ]
    )

    ax_ratio.fill_between(
        bin_centres,
        1.0 - ratio_uncertainty,
        1.0 + ratio_uncertainty,
        step="mid",
        color=COLORS["data"],
        alpha=0.10,
        linewidth=0,
        label="Data stat. unc.",
    )

    yrange = ax_ratio.get_ylim()[1] - ax_ratio.get_ylim()[0]

    if yrange < 0.5:
        spacing = 0.1
    elif yrange < 1.0:
        spacing = 0.2
    else:
        spacing = 0.5

    ax_ratio.yaxis.set_major_locator(MultipleLocator(spacing))
    ax_ratio.yaxis.set_minor_locator(MultipleLocator(spacing/2))

    ax_ratio.set_ylabel(
        "MC / Data",
    )

    ax_ratio.set_xlabel(
        r"$m_{\mu\mu}$ [GeV]",
    )

    ax_ratio.xaxis.set_minor_locator(AutoMinorLocator())

    # ax_ratio.legend(
    #     loc="upper right",
    #     frameon=False,
    #     ncol=1,
    #     fontsize=8.5,
    # )

    # ------------------------------------------------------------------
    # Axes styling
    # ------------------------------------------------------------------

    for axis in [
        ax,
        ax_ratio,
    ]:

        axis.tick_params(
            direction="in",
            which="both",
        )

        axis.grid(
            axis="y",
            linestyle=":",
            linewidth=0.6,
            alpha=0.45,
        )

        axis.spines["top"].set_visible(True)
        axis.spines["right"].set_visible(True)
        axis.spines["left"].set_visible(True)
        axis.spines["bottom"].set_visible(True)

        for spine in axis.spines.values():
            spine.set_linewidth(1.0)

    # ------------------------------------------------------------------
    # Final layout and saving
    # ------------------------------------------------------------------

    fig.subplots_adjust(
        left=0.14,
        right=0.97,
        bottom=0.12,
        top=0.97,
        hspace=0.04,
    )

    fig.savefig(
        filename + ".png",
        dpi=400,
        bbox_inches="tight",
    )

    fig.savefig(
        filename + ".pdf",
        bbox_inches="tight",
    )

    plt.show()

    plt.close(fig)

    return {
        "chi2_ndf_mc": chi2_ndf_mc,
        "chi2_ndf_transformed": chi2_ndf_transformed,
    }

def plot_pt_distribution(
    data,
    mc,
    transformed_mc,
    filename,
    sample_label="Test sample",
):
    """
    HEP-style transverse-momentum comparison plot.

    Upper panel:
        Data
        Original MC
        Transformed MC

    Lower panel:
        MC / Data
        Transformed MC / Data

    Plots both muons together.
    """

    # ------------------------------------------------------------------
    # Combine the two muons
    # ------------------------------------------------------------------

    data_pt = np.concatenate([
        np.asarray(data[:, 0], dtype=float),
        np.asarray(data[:, 3], dtype=float),
    ])

    mc_pt = np.concatenate([
        np.asarray(mc[:, 0], dtype=float),
        np.asarray(mc[:, 3], dtype=float),
    ])

    transformed_pt = np.concatenate([
        np.asarray(transformed_mc[:, 0], dtype=float),
        np.asarray(transformed_mc[:, 3], dtype=float),
    ])

    # ------------------------------------------------------------------
    # Remove invalid values
    # ------------------------------------------------------------------

    data_pt = data_pt[np.isfinite(data_pt)]
    mc_pt = mc_pt[np.isfinite(mc_pt)]
    transformed_pt = transformed_pt[np.isfinite(transformed_pt)]

    # ------------------------------------------------------------------
    # Common pT range and binning
    # ------------------------------------------------------------------

    all_pt = np.concatenate([
        data_pt,
        mc_pt,
        transformed_pt,
    ])

    x_min = max(0.0, np.percentile(all_pt, 0.5))
    x_max = np.percentile(all_pt, 99.5)

    bins = np.linspace(
        x_min,
        x_max,
        31,
    )

    bin_centres = 0.5 * (
        bins[:-1] + bins[1:]
    )

    bin_widths = np.diff(bins)

    # ------------------------------------------------------------------
    # Histograms
    # ------------------------------------------------------------------

    data_counts, _ = np.histogram(
        data_pt,
        bins=bins,
    )

    mc_counts, _ = np.histogram(
        mc_pt,
        bins=bins,
    )

    transformed_counts, _ = np.histogram(
        transformed_pt,
        bins=bins,
    )

    # ------------------------------------------------------------------
    # Normalised densities
    # ------------------------------------------------------------------

    data_density = (
        data_counts
        / max(data_counts.sum(), 1)
        / bin_widths
    )

    mc_density = (
        mc_counts
        / max(mc_counts.sum(), 1)
        / bin_widths
    )

    transformed_density = (
        transformed_counts
        / max(transformed_counts.sum(), 1)
        / bin_widths
    )

    # ------------------------------------------------------------------
    # Statistical uncertainty on DATA
    # ------------------------------------------------------------------

    data_uncertainty = np.zeros_like(
        data_density
    )

    valid_data = data_counts > 0

    data_uncertainty[valid_data] = (
        np.sqrt(data_counts[valid_data])
        / data_counts.sum()
        / bin_widths[valid_data]
    )

    # ------------------------------------------------------------------
    # Ratios
    # ------------------------------------------------------------------

    valid_ratio = data_density > 0

    mc_ratio = np.full(
        len(data_density),
        np.nan,
    )

    transformed_ratio = np.full(
        len(data_density),
        np.nan,
    )

    mc_ratio[valid_ratio] = (
        mc_density[valid_ratio]
        / data_density[valid_ratio]
    )

    transformed_ratio[valid_ratio] = (
        transformed_density[valid_ratio]
        / data_density[valid_ratio]
    )

    # ------------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------------

    fig = plt.figure(
        figsize=(7.2, 6.4)
    )

    gs = fig.add_gridspec(
        2,
        1,
        height_ratios=[3.0, 1.5],
        hspace=0.03,
    )

    ax = fig.add_subplot(gs[0])

    ax_ratio = fig.add_subplot(
        gs[1],
        sharex=ax,
    )

    # ------------------------------------------------------------------
    # Upper panel
    # ------------------------------------------------------------------

    ax.errorbar(
        bin_centres,
        data_density,
        yerr=data_uncertainty,
        fmt="o",
        markersize=4,
        color=COLORS["data"],
        ecolor=COLORS["data"],
        elinewidth=1.2,
        capsize=2,
        label="Data",
    )

    ax.stairs(
        mc_density,
        bins,
        color=COLORS["mc"],
        linewidth=2.0,
        linestyle="--",
        label="Original MC",
    )

    ax.stairs(
        transformed_density,
        bins,
        color=COLORS["transformed"],
        linewidth=2.2,
        label="Transformed MC",
    )

    all_y_values = np.concatenate([
        data_density[data_density > 0],
        mc_density[mc_density > 0],
        transformed_density[
            transformed_density > 0
        ],
    ])

    y_max = np.max(all_y_values)

    ax.set_ylim(
        0.0,
        1.10 * y_max,
    )

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_ylabel(
        "Normalised density",
    )

    ax.tick_params(
        labelbottom=False,
    )

    ax.xaxis.set_minor_locator(
        AutoMinorLocator()
    )

    ax.yaxis.set_minor_locator(
        AutoMinorLocator()
    )

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------

    ax.text(
        0.025,
        0.965,
        r"$Z \rightarrow \mu^+\mu^-$",
        transform=ax.transAxes,
        fontsize=17,
        ha="left",
        va="top",
    )

    ax.text(
        0.025,
        0.885,
        sample_label,
        transform=ax.transAxes,
        fontsize=13,
        ha="left",
        va="top",
    )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.99, 0.97),
        frameon=False,
        handlelength=2.8,
    )

    # ------------------------------------------------------------------
    # Ratio panel
    # ------------------------------------------------------------------

    ax_ratio.axhline(
        1.0,
        color="black",
        linewidth=1.0,
    )

    ax_ratio.axhline(
        0.8,
        color="0.65",
        linestyle="--",
        linewidth=0.8,
    )

    ax_ratio.axhline(
        1.2,
        color="0.65",
        linestyle="--",
        linewidth=0.8,
    )

    ax_ratio.stairs(
        mc_ratio,
        bins,
        color=COLORS["mc"],
        linewidth=1.8,
        linestyle="--",
    )

    ax_ratio.stairs(
        transformed_ratio,
        bins,
        color=COLORS["transformed"],
        linewidth=2.0,
    )

    # ------------------------------------------------------------------
    # Data uncertainty band
    # ------------------------------------------------------------------

    ratio_uncertainty = np.full(
        len(data_density),
        np.nan,
    )

    ratio_uncertainty[valid_ratio] = (
        data_uncertainty[valid_ratio]
        / data_density[valid_ratio]
    )

    ax_ratio.fill_between(
        bin_centres,
        1.0 - ratio_uncertainty,
        1.0 + ratio_uncertainty,
        step="mid",
        color=COLORS["data"],
        alpha=0.10,
        linewidth=0,
    )

    # ------------------------------------------------------------------
    # Ratio axis
    # ------------------------------------------------------------------

    ax_ratio.set_ylabel(
        "MC / Data",
    )

    ax_ratio.set_xlabel(
        r"$p_T$ [GeV]",
    )

    ax_ratio.xaxis.set_minor_locator(
        AutoMinorLocator()
    )

    ax_ratio.yaxis.set_minor_locator(
        AutoMinorLocator()
    )

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------

    for axis in [
        ax,
        ax_ratio,
    ]:

        axis.tick_params(
            direction="in",
            which="both",
            top=True,
            right=True,
        )

        axis.grid(
            axis="y",
            linestyle=":",
            linewidth=0.6,
            alpha=0.45,
        )

        axis.spines["top"].set_visible(True)
        axis.spines["right"].set_visible(True)
        axis.spines["left"].set_visible(True)
        axis.spines["bottom"].set_visible(True)

        for spine in axis.spines.values():
            spine.set_linewidth(1.0)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    fig.subplots_adjust(
        left=0.14,
        right=0.97,
        bottom=0.12,
        top=0.97,
        hspace=0.04,
    )

    fig.savefig(
        filename + ".png",
        dpi=400,
        bbox_inches="tight",
    )

    fig.savefig(
        filename + ".pdf",
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


def save_kinematic_diagnostics(
    mc,
    transformed_mc,
    data,
):

    names = [
        r"$p_{T,1}$ [GeV]",
        r"$\eta_1$",
        r"$\phi_1$",
        r"$p_{T,2}$ [GeV]",
        r"$\eta_2$",
        r"$\phi_2$",
    ]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 7.6),
    )

    for i, ax in enumerate(axes.flat):

        ax.hist(
            data[:, i], bins=40, density=True, histtype="step",
            linewidth=1.8, color=COLORS["data"], label="Data",
        )

        ax.hist(
            mc[:, i], bins=40, density=True, histtype="step",
            linewidth=1.8, color=COLORS["mc"], linestyle="--",
            label="Original MC",
        )

        ax.hist(
            transformed_mc[:, i], bins=40, density=True, histtype="step",
            linewidth=1.8, color=COLORS["transformed"], label="Transformed MC",
        )

        ax.set_xlabel(names[i])
        ax.set_ylabel("Density")

        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(
            direction="in",
            which="both",
            top=True,
            right=True,
        )        
        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.35)
        

        if i == 0:
            ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Kinematic transformation diagnostics", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig("03_kinematic_diagnostics_realisticMC.png", dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def save_correlation_plot(
    mc,
    transformed_mc,
    data,
):
    """
    Two-muon pT correlation for Data / Original MC / Transformed MC.

    All three panels share the same bin edges and the same colour scale
    (vmin/vmax fixed from the combined counts), so densities are directly
    comparable across panels rather than each subplot auto-scaling to its
    own data range independently.
    """

    datasets = [
        (data, "Data"),
        (mc, "Original MC"),
        (transformed_mc, "Transformed MC"),
    ]

    # Shared binning, derived from the data sample (the reference distribution)
    pt1_all = np.concatenate([np.asarray(s[:, 0]) for s, _ in datasets])
    pt2_all = np.concatenate([np.asarray(s[:, 3]) for s, _ in datasets])

    lo1, hi1 = np.percentile(pt1_all, [0.5, 99.5])
    lo2, hi2 = np.percentile(pt2_all, [0.5, 99.5])

    x_edges = np.linspace(lo1, hi1, 41)
    y_edges = np.linspace(lo2, hi2, 41)

    # First pass to find a shared, sensible colour scale
    counts_list = [
        np.histogram2d(np.asarray(s[:, 0]), np.asarray(s[:, 3]),
                        bins=[x_edges, y_edges])[0]
        for s, _ in datasets
    ]
    vmax = max(c.max() for c in counts_list)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5), sharex=True, sharey=True)

    im = None
    for ax, (sample, title), counts in zip(axes, datasets, counts_list):

        im = ax.pcolormesh(
            x_edges, y_edges, counts.T,
            cmap="viridis", vmin=0, vmax=vmax,
        )

        ax.set_xlabel(r"$p_{T,1}$ [GeV]")
        ax.set_title(title, fontsize=13)

        ax.tick_params(
            direction="in",
            which="both",
            top=True,
            right=True,
        )
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())

    axes[0].set_ylabel(r"$p_{T,2}$ [GeV]")

    fig.suptitle(r"Two-muon $p_T$ correlation", fontsize=15)

    fig.subplots_adjust(left=0.06, right=0.90, bottom=0.13, top=0.88, wspace=0.08)

    cax = fig.add_axes([0.92, 0.13, 0.015, 0.75])
    fig.colorbar(im, cax=cax, label="Events / bin")

    fig.savefig("04_muon_pt_correlation_realisticMC.png", dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)


def save_performance_plot(
    data_mass,
    mc_mass,
    transformed_mass,
):

    original_wasserstein = (
        wasserstein_distance_1d(
            data_mass,
            mc_mass,
        )
    )

    transformed_wasserstein = (
        wasserstein_distance_1d(
            data_mass,
            transformed_mass,
        )
    )

    original_chi2 = (
        chi_squared_distance(
            data_mass,
            mc_mass,
            bins=np.linspace(
                XMIN,
                XMAX,
                N_BINS + 1,
            ),
        )
    )

    transformed_chi2 = (
        chi_squared_distance(
            data_mass,
            transformed_mass,
            bins=np.linspace(
                XMIN,
                XMAX,
                N_BINS + 1,
            ),
        )
    )

    print("\nQuantitative performance:")
    print("-----------------------------------")

    print(
        f"Wasserstein distance:"
    )

    print(
        f"Original MC:     "
        f"{original_wasserstein:.4f}"
    )

    print(
        f"Transformed MC:  "
        f"{transformed_wasserstein:.4f}"
    )

    print(
        f"\nChi-squared distance:"
    )

    print(
        f"Original MC:     "
        f"{original_chi2:.4f}"
    )

    print(
        f"Transformed MC:  "
        f"{transformed_chi2:.4f}"
    )

    labels = [
        "Original MC",
        "Transformed MC",
    ]

    wasserstein_values = [
        original_wasserstein,
        transformed_wasserstein,
    ]

    bar_colors = [COLORS["mc"], COLORS["transformed"]]

    fig, ax = plt.subplots(figsize=(6.4, 5.4))

    bars = ax.bar(
        labels,
        wasserstein_values,
        color=bar_colors,
        width=0.55,
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, value in zip(bars, wasserstein_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02 * max(wasserstein_values),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_ylabel("Wasserstein distance")
    ax.set_ylim(0, 1.20 * max(wasserstein_values))
    ax.set_title("Agreement with data invariant-mass distribution")

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(
                direction="in",
                which="both",
                top=True,
                right=True,
            )
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()

    fig.savefig(
        "05_performance_comparison_realisticMC.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    return {
        "original_wasserstein":
            original_wasserstein,

        "transformed_wasserstein":
            transformed_wasserstein,

        "original_chi2":
            original_chi2,

        "transformed_chi2":
            transformed_chi2,
    }


# =============================================================================
# Main
# =============================================================================

def main():

    set_seed(SEED)

    # =========================================================================
    # Load the full dataset
    # =========================================================================

    data = get_data(
        momentum_unit="GeV",
        nevents=NEVENTS,
    )

    mc = torch.tensor(
    np.stack(
        [
            mum_MC[0] / 1000.0,   # pt1 (GeV)
            mum_MC[1],            # eta1
            mum_MC[2],            # phi1
            mup_MC[0] / 1000.0,   # pt2 (GeV)
            mup_MC[1],            # eta2
            mup_MC[2],            # phi2
        ],
        axis=1,
    ),
    dtype=torch.float32,
)[:NEVENTS]

    print(
        f"Total data events: {len(data)}"
    )

    print(
        f"Total MC events: {len(mc)}"
    )

    # =========================================================================
    # Remove invalid events
    # =========================================================================

    data_mask = torch.isfinite(
        data
    ).all(
        dim=1
    )

    mc_mask = torch.isfinite(
        mc
    ).all(
        dim=1
    )

    data = data[
        data_mask
    ]

    mc = mc[
        mc_mask
    ]

    # =========================================================================
    # 80/20 TRAIN-TEST SPLIT
    # =========================================================================

    data_train, data_test = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )

    mc_train, mc_test = train_test_split(
        mc,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
    )

    print(
        "\nDataset split:"
    )

    print(
        f"Data training events: {len(data_train)}"
    )

    print(
        f"Data test events:     {len(data_test)}"
    )

    print(
        f"MC training events:   {len(mc_train)}"
    )

    print(
        f"MC test events:       {len(mc_test)}"
    )

    # =========================================================================
    # Move to device
    # =========================================================================

    data_train = data_train.to(
        DEVICE
    )

    data_test = data_test.to(
        DEVICE
    )

    mc_train = mc_train.to(
        DEVICE
    )

    mc_test = mc_test.to(
        DEVICE
    )

    # =========================================================================
    # Standardisation
    # =========================================================================


    mc_mean, mc_std = fit_standardisation(
        mc_train
    )

    mc_train_standardised = standardise(
        mc_train,
        mc_mean,
        mc_std,
    )

    mc_test_standardised = standardise(
        mc_test,
        mc_mean,
        mc_std,
    )

    # =========================================================================
    # Training target: DATA TRAIN invariant mass
    # =========================================================================

    with torch.no_grad():

        data_train_mass = calculate_invariant_mass(
            data_train
        )

    data_train_mass_valid = (
        torch.isfinite(
            data_train_mass
        )
        & (
            data_train_mass >= XMIN
        )
        & (
            data_train_mass <= XMAX
        )
    )

    data_train_mass = data_train_mass[
        data_train_mass_valid
    ]

    bin_centres = torch.linspace(
            XMIN,
            XMAX,
            N_BINS,
            device=DEVICE,
        )

    # =========================================================================
    # Build flow
    # =========================================================================

    transform = make_flow(
        features=6,
    )

    flow = Flow(
        transform=transform,
        distribution=StandardNormal(
            [
                6
            ]
        ),
    ).to(
        DEVICE
    )

    optimizer = optim.Adam(
        flow.parameters(),
        lr=LR,
    )

    # =========================================================================
    # Training history
    # =========================================================================

    history = {
        "total": [],
        "mass": [],
        "regularisation": [],
    }

    best_loss = float(
        "inf"
    )

    best_state = None

    n_mc_train = len(
        mc_train_standardised
    )

    # =========================================================================
    # TRAINING
    # =========================================================================

    print(
        "\nStarting training..."
    )

    for step in range(
        N_STEPS
    ):

        # -------------------------------------------------------------
        # Random batch from TRAINING MC ONLY
        # -------------------------------------------------------------

        mc_indices = torch.randint(
            0,
            n_mc_train,
            (
                BATCH_SIZE,
            ),
            device=DEVICE,
        )

        x_mc = mc_train_standardised[
            mc_indices
        ]

        # -------------------------------------------------------------
        # Transform MC through flow
        # -------------------------------------------------------------

        transformed_mc, _ = transform(
            x_mc
        )

        transformed_mc_physical = destandardise(
            transformed_mc,
            mc_mean,
            mc_std,
        )

        # -------------------------------------------------------------
        # Calculate transformed invariant mass
        # -------------------------------------------------------------

        transformed_mass = calculate_invariant_mass(
            transformed_mc_physical
        )

        valid = (
            torch.isfinite(
                transformed_mass
            )
            & (
                transformed_mass >= XMIN
            )
            & (
                transformed_mass <= XMAX
            )
        )

        if valid.sum() < 10:

            continue

        transformed_mass_valid = transformed_mass[
            valid
        ]

        # -------------------------------------------------------------
        # Invariant-mass loss
        # -------------------------------------------------------------

        mass_loss = mass_distribution_loss(
            transformed_mass_valid,
            data_train_mass,
            bin_centres,
        )

        # -------------------------------------------------------------
        # Weak regularisation
        # -------------------------------------------------------------

        regularisation = regularisation_loss(
            transformed_mc,
            x_mc,
        )

        # -------------------------------------------------------------
        # Total loss
        # -------------------------------------------------------------

        loss = (
            MASS_LOSS_WEIGHT
            * mass_loss
            + REGULARISATION_WEIGHT
            * regularisation
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            flow.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        # -------------------------------------------------------------
        # Record training history
        # -------------------------------------------------------------

        history["total"].append(
            loss.item()
        )

        history["mass"].append(
            mass_loss.item()
        )

        history["regularisation"].append(
            regularisation.item()
        )

        # -------------------------------------------------------------
        # Save best model
        # -------------------------------------------------------------

        if loss.item() < best_loss:

            best_loss = loss.item()

            best_state = copy.deepcopy(
                flow.state_dict()
            )

        if (
            step == 0
            or (
                step + 1
            ) % 50 == 0
        ):

            print(
                f"Step {step + 1:5d}/{N_STEPS} | "
                f"Total: {loss.item():.6f} | "
                f"Mass: {mass_loss.item():.6f} | "
                f"Reg: {regularisation.item():.6f}"
            )

    # =========================================================================
    # Restore best model
    # =========================================================================

    if best_state is not None:

        flow.load_state_dict(
            best_state
        )

    print(
        "\nTraining completed."
    )

    # =========================================================================
    # APPLY TRAINED MODEL TO TRAINING AND TEST MC
    # =========================================================================

    with torch.no_grad():

        # -------------------------------------------------------------
        # Training MC transformation
        # -------------------------------------------------------------

        transformed_mc_train, _ = transform(
            mc_train_standardised
        )

        transformed_mc_train = destandardise(
            transformed_mc_train,
            mc_mean,
            mc_std,
        )

        # -------------------------------------------------------------
        # Test MC transformation
        # -------------------------------------------------------------

        transformed_mc_test, _ = transform(
            mc_test_standardised
        )

        transformed_mc_test = destandardise(
            transformed_mc_test,
            mc_mean,
            mc_std,
        )

    # =========================================================================
    # Calculate invariant masses
    # =========================================================================

    with torch.no_grad():

        # Training distributions

        data_train_mass_final = calculate_invariant_mass(
            data_train
        )

        mc_train_mass_final = calculate_invariant_mass(
            mc_train
        )

        transformed_train_mass_final = calculate_invariant_mass(
            transformed_mc_train
        )

        # Test distributions

        data_test_mass_final = calculate_invariant_mass(
            data_test
        )

        mc_test_mass_final = calculate_invariant_mass(
            mc_test
        )

        transformed_test_mass_final = calculate_invariant_mass(
            transformed_mc_test
        )

    # =========================================================================
    # Convert to NumPy
    # =========================================================================

    data_train_mass_np = (
        data_train_mass_final
        .detach()
        .cpu()
        .numpy()
    )

    mc_train_mass_np = (
        mc_train_mass_final
        .detach()
        .cpu()
        .numpy()
    )

    transformed_train_mass_np = (
        transformed_train_mass_final
        .detach()
        .cpu()
        .numpy()
    )

    data_test_mass_np = (
        data_test_mass_final
        .detach()
        .cpu()
        .numpy()
    )

    mc_test_mass_np = (
        mc_test_mass_final
        .detach()
        .cpu()
        .numpy()
    )

    transformed_test_mass_np = (
        transformed_test_mass_final
        .detach()
        .cpu()
        .numpy()
    )

    # =========================================================================
    # Select valid masses
    # =========================================================================

    def select_valid_mass(
        mass
    ):

        mask = (
            np.isfinite(
                mass
            )
            & (
                mass >= XMIN
            )
            & (
                mass <= XMAX
            )
        )

        return mass[
            mask
        ]

    data_train_mass_np = select_valid_mass(
        data_train_mass_np
    )

    mc_train_mass_np = select_valid_mass(
        mc_train_mass_np
    )

    transformed_train_mass_np = select_valid_mass(
        transformed_train_mass_np
    )

    data_test_mass_np = select_valid_mass(
        data_test_mass_np
    )

    mc_test_mass_np = select_valid_mass(
        mc_test_mass_np
    )

    transformed_test_mass_np = select_valid_mass(
        transformed_test_mass_np
    )

    # =========================================================================
    # TRAINING RESULT
    # =========================================================================

    print(
        "\n======================================"
    )

    print(
        "TRAINING SET PERFORMANCE"
    )

    print(
        "======================================"
    )

    print(
        f"Data training mean: "
        f"{np.mean(data_train_mass_np):.3f} GeV"
    )

    print(
        f"Original MC training mean: "
        f"{np.mean(mc_train_mass_np):.3f} GeV"
    )

    print(
        f"Transformed MC training mean: "
        f"{np.mean(transformed_train_mass_np):.3f} GeV"
    )

    train_wasserstein_before = (
        wasserstein_distance_1d(
            data_train_mass_np,
            mc_train_mass_np,
        )
    )

    train_wasserstein_after = (
        wasserstein_distance_1d(
            data_train_mass_np,
            transformed_train_mass_np,
        )
    )

    print(
        f"\nWasserstein distance BEFORE: "
        f"{train_wasserstein_before:.4f}"
    )

    print(
        f"Wasserstein distance AFTER:  "
        f"{train_wasserstein_after:.4f}"
    )

    # =========================================================================
    # TEST RESULT
    # =========================================================================

    print(
        "\n======================================"
    )

    print(
        "TEST SET PERFORMANCE"
    )

    print(
        "======================================"
    )

    print(
        f"Data test mean: "
        f"{np.mean(data_test_mass_np):.3f} GeV"
    )

    print(
        f"Original MC test mean: "
        f"{np.mean(mc_test_mass_np):.3f} GeV"
    )

    print(
        f"Transformed MC test mean: "
        f"{np.mean(transformed_test_mass_np):.3f} GeV"
    )

    test_wasserstein_before = (
        wasserstein_distance_1d(
            data_test_mass_np,
            mc_test_mass_np,
        )
    )

    test_wasserstein_after = (
        wasserstein_distance_1d(
            data_test_mass_np,
            transformed_test_mass_np,
        )
    )

    print(
        f"\nWasserstein distance BEFORE: "
        f"{test_wasserstein_before:.4f}"
    )

    print(
        f"Wasserstein distance AFTER:  "
        f"{test_wasserstein_after:.4f}"
    )

    improvement = (
        100.0
        * (
            test_wasserstein_before
            - test_wasserstein_after
        )
        / max(
            test_wasserstein_before,
            1e-12,
        )
    )

    print(
        f"\nTest-set improvement: "
        f"{improvement:.2f}%"
    )

    # =========================================================================
    # PLOTS
    # =========================================================================

    print(
        "\nGenerating plots..."
    )

    # -------------------------------------------------------------------------
    # 1. Training loss
    # -------------------------------------------------------------------------

    plot_training_history(
        steps=range(len(history["total"])),
        total_losses=history["total"],
        mass_losses=history["mass"],
        regularisation_losses=history["regularisation"],
        filename="01_training_loss_realisticMC"
    )

    # -------------------------------------------------------------------------
    # 2. Training set result
    # -------------------------------------------------------------------------

    plot_mass_distribution(
    data_train_mass_np,
    mc_train_mass_np,
    transformed_train_mass_np,
    filename="mass_distribution_training_realisticMC",
    sample_label="Training sample",
)

    # -------------------------------------------------------------------------
    # 3. TEST SET RESULT
    # -------------------------------------------------------------------------

    plot_mass_distribution(
    data_test_mass_np,
    mc_test_mass_np,
    transformed_test_mass_np,
    filename="mass_distribution_test_realisticMC",
    sample_label="Test sample",
)

    # -------------------------------------------------------------------------
    # 4. TEST SET pT COMPARISON
    # -------------------------------------------------------------------------

    plot_pt_distribution(
        data_test.detach().cpu().numpy(),
        mc_test.detach().cpu().numpy(),
        transformed_mc_test.detach().cpu().numpy(),
        filename="pt_distribution_test_realisticMC",
        sample_label="Test sample",
    )

    # -------------------------------------------------------------------------
    # 4. Compare training and test performance
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    labels = [
        "Training\nOriginal",
        "Training\nTransformed",
        "Test\nOriginal",
        "Test\nTransformed",
    ]

    values = [
        train_wasserstein_before,
        train_wasserstein_after,
        test_wasserstein_before,
        test_wasserstein_after,
    ]

    bar_colors = [
        COLORS["mc"], COLORS["transformed"],
        COLORS["mc"], COLORS["transformed"],
    ]

    bars = ax.bar(
        labels,
        values,
        color=bar_colors,
        width=0.6,
        edgecolor="black",
        linewidth=0.8,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02 * max(values),
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10.5,
        )

    # visually separate the training and test groups
    ax.axvline(1.5, color="0.75", linestyle="--", linewidth=1.0, zorder=0)

    ax.set_ylabel("Wasserstein distance")
    ax.set_ylim(0, 1.20 * max(values))
    ax.set_title("Invariant-mass agreement before and after transformation")

    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(
                direction="in",
                which="both",
                top=True,
                right=True,
            )
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.4)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    fig.tight_layout()

    fig.savefig(
        "06_train_test_performance_realisticMC.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


if __name__ == "__main__":

    main()