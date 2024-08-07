"""
An analysis script for 2.5D Cartesian convection. The script can create gifs showing
a heatmap of convection, as well as time-tracks of the Kinetic Energy and 1/<T>, the
time-averaged flux balance with depth, and both the instantaneous and time-averaged
Nusselt Number.

Usage:
    analysis.py FILE [options]
    
Options:
    -t --time-tracks                # Plot the time-track data (KE and 1/<T>)
    -d --depth-profile              # Plot the depth profile of the temperature
    -f --flux-balance               # Plot the flux-balance with depth
    -p --profile-dissipation        # Plot the depth profile of the dissipation
    -i --info                       # Information required
    -n --nusselt                    # Print different Nusselt Numbers
    -g --gif                        # Create a gif of the convection
    -h --help                       # Display this help message
    -v --version                    # Display the version
    --cadence [CADENCE]             # Cadence of the gifs [default: 1]
    --heat-func [HEAT]              # Heat function to use if not in snapshot [default: exp]
    --ASI [TIME]                    # Sim-time to begin average [default: 0.65]

"""

from docopt import docopt
import h5py as h5
import numpy as np
from scipy.integrate import cumtrapz
import re
import time as timer
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import warnings
import imageio.v2 as imageio
from os import makedirs
from os.path import normpath
from shutil import rmtree
from glob import glob


def rolling_average(quantity, time, window=0.05):
    assert len(time) == len(quantity)
    run_ave = []
    for i, t0 in enumerate(time):
        mean = np.nanmean(
            quantity[(time > t0 - window / 2) & (time <= t0 + window / 2)]
        )
        run_ave.append(mean)
    return np.array(run_ave)


def get_index(time, start_time):
    return np.abs(time - start_time).argmin()


def calculate_crossing(curve):
    sign_change = np.roll(np.sign(curve - 1.0), 1) - np.sign(curve - 1.0)
    sign_change[0] = 0
    crossing_idxs = np.argwhere(((sign_change) != 0).astype(int))
    crossings = []
    threshold = 6
    for i, x in enumerate(crossing_idxs):
        if i == 0:
            crossings.append(x)
        else:
            if np.abs(x - crossing_idxs[i - 1]) >= threshold:
                crossings.append(x)
    for idx in crossings:
        if np.gradient(curve)[idx] < 0:
            return idx

    return -10


def get_heat_func(heat):
    try:
        with h5.File(direc + "snapshots/snapshots_s1.h5", "r") as file:
            heat_func = np.array(file["tasks"]["heat"])[0, 0, 0, :]
        print("Reading heat function from snapshots")
        return heat_func
    except:
        print(f"Heat Func not found.\nWriting {heat} heat function")
        if heat == "exp":
            l = 0.1
            beta = 1
            a = 1 / (0.1 * (1 - np.exp(-1 / l)))
            heating = lambda z: a * np.exp(-z / l) - beta
            heat_func = heating(z)
        elif heat == "cos":
            Lz = 4
            F = 1
            heating_width = 0.2
            H = Lz / (1 + 2 * heating_width)
            # Width of heating and cooling layers
            Delta = heating_width * H
            heat_func = lambda z: (F / Delta) * (
                1 + np.cos((2 * np.pi * (z - (Delta / 2))) / Delta)
            )
            cool_func = lambda z: (F / Delta) * (
                -1 - np.cos((2 * np.pi * (z - Lz + (Delta / 2))) / Delta)
            )

            heat_func = np.piecewise(
                z, [z <= Delta, z >= Lz - Delta], [heat_func, cool_func, 0]
            )
        else:
            raise ValueError(f"Invalid heat function {heat}")

        return heat_func


args = docopt(__doc__, version="2.0")
print(args)

direc = normpath(args["FILE"]) + "/"

if not glob(direc):
    raise FileNotFoundError("Directory does not exist")

outpath = direc + "images/"
makedirs(outpath, exist_ok=True)

start_time = timer.time()
print("====== Data Read-In ======")
if args["--nusselt"]:
    scalar_files = glob(direc + "scalars/scalars_s*.h5")
    scalar_files.sort(key=lambda f: int(re.sub("\D", "", f)))
    snap_files = glob(direc + "snapshots/snapshots_s*.h5")
    snap_files.sort(key=lambda f: int(re.sub("\D", "", f)))
    prof_files = glob(direc + "horiz_aves/horiz_aves_s*.h5")
    prof_files.sort(key=lambda f: int(re.sub("\D", "", f)))

    for i, sc_file in enumerate(scalar_files):
        if i == 0:
            with h5.File(sc_file, "r") as file:
                sc_time = np.array(file["scales"]["sim_time"])
                T_0 = np.array(file["tasks"]["<T(0)>"])[:, 0, 0, 0]
        else:
            with h5.File(sc_file, "r") as file:
                sc_time = np.concatenate(
                    (sc_time, np.array(file["scales"]["sim_time"])), axis=0
                )
                T_0 = np.concatenate(
                    (T_0, np.array(file["tasks"]["<T(0)>"])[:, 0, 0, 0]), axis=0
                )

    for i, snap_file in enumerate(snap_files):
        if i == 0:
            with h5.File(snap_file, "r") as file:
                snap_time = np.array(file["scales"]["sim_time"])
                max_T = np.max(np.array(file["tasks"]["Temp"])[:, 0, :, :], (1, 2))
                min_T = np.min(np.array(file["tasks"]["Temp"])[:, 0, :, :], (1, 2))
        else:
            with h5.File(snap_file, "r") as file:
                snap_time = np.concatenate(
                    (snap_time, np.array(file["scales"]["sim_time"])), axis=0
                )
                max_T = np.concatenate(
                    (
                        max_T,
                        np.max(np.array(file["tasks"]["Temp"])[:, 0, :, :], (1, 2)),
                    ),
                    axis=0,
                )
                min_T = np.concatenate(
                    (
                        min_T,
                        np.min(np.array(file["tasks"]["Temp"])[:, 0, :, :], (1, 2)),
                    ),
                    axis=0,
                )

    for i, prof_file in enumerate(prof_files):
        if i == 0:
            with h5.File(prof_file, "r") as file:
                prof_time = np.array(file["scales"]["sim_time"])
                z = np.array(file["tasks"]["<T>"].dims[3]["z"])
                F_cond = np.array(file["tasks"]["<F_cond>"])[:, 0, 0, :]
                F_conv = np.array(file["tasks"]["<F_conv>"])[:, 0, 0, :]
        else:
            with h5.File(prof_file, "r") as file:
                prof_time = np.concatenate(
                    (prof_time, np.array(file["scales"]["sim_time"])), axis=0
                )
                F_cond = np.concatenate(
                    (F_cond, np.array(file["tasks"]["<F_cond>"])[:, 0, 0, :]), axis=0
                )
                F_conv = np.concatenate(
                    (F_conv, np.array(file["tasks"]["<F_conv>"])[:, 0, 0, :]), axis=0
                )

    scalar_ASI = get_index(sc_time, float(args["--ASI"]))
    snap_ASI = get_index(snap_time, float(args["--ASI"]))
    prof_ASI = get_index(prof_time, float(args["--ASI"]))

    inv_t0 = 1 / T_0
    inv_t0_ave = np.nanmean(inv_t0[scalar_ASI:], axis=0)
    print(f"\t1/<T(z=0)> =\t\t{inv_t0_ave:.5f}")

    # F_cond_ave = np.nanmean(F_cond, axis=1)
    # F_conv_ave = np.nanmean(F_conv, axis=1)

    F_cond_bar = np.nanmean(F_cond[prof_ASI:, :], axis=0)
    F_conv_bar = np.nanmean(F_conv[prof_ASI:, :], axis=0)
    F_cond_ave = np.trapz(F_cond_bar, z, axis=0)
    F_conv_ave = np.trapz(F_conv_bar, z, axis=0)
    flux_nu_ave = 1 + (F_conv_ave / F_cond_ave)

    print(f"\t1 + F_conv/F_cond =\t{flux_nu_ave:.5f}")

    inv_delta_T = 1 / (max_T - min_T)
    inv_delta_T_ave = np.nanmean(inv_delta_T[snap_ASI:], axis=0)
    print(f"\t1/dT = \t\t\t{inv_delta_T_ave:.5f}")
    print(
        f"\t\t T_max = {np.nanmean(max_T[snap_ASI:]):.5f}, T_min = {np.nanmean(min_T[snap_ASI:]):.5f}"
    )

    with open(direc + "run_params/runparams.json", "r") as file:
        run_params = json.load(file)
        Ra = run_params["Ra"]

    with open(direc + "Nu_compar.json", "w") as file:
        json.dump(
            {
                "Ra": Ra,
                "inv_T0": inv_t0_ave,
                "flux_ratio": flux_nu_ave,
                "ind_dT": inv_delta_T_ave,
            },
            file,
            indent=4,
        )

if args["--info"] or args["--time-tracks"]:
    scalar_files = glob(direc + "scalars/scalars_s*.h5")
    scalar_files.sort(key=lambda f: int(re.sub("\D", "", f)))

    for i, sc_file in enumerate(scalar_files):
        if i == 0:
            with h5.File(sc_file, "r") as file:
                sc_time = np.array(file["scales"]["sim_time"])
                deltaT = np.array(file["tasks"]["<T(0)>"])[:, 0, 0, 0]
                Re = np.array(file["tasks"]["Re"])[:, 0, 0, 0]
                if args["--time-tracks"]:
                    KE = np.array(file["tasks"]["KE"])[:, 0, 0, 0]
        else:
            with h5.File(sc_file, "r") as file:
                sc_time = np.concatenate(
                    (sc_time, np.array(file["scales"]["sim_time"])), axis=0
                )
                deltaT = np.concatenate(
                    (deltaT, np.array(file["tasks"]["<T(0)>"])[:, 0, 0, 0]), axis=0
                )
                Re = np.concatenate(
                    (Re, np.array(file["tasks"]["Re"])[:, 0, 0, 0]), axis=0
                )
                if args["--time-tracks"]:
                    KE = np.concatenate(
                        (KE, np.array(file["tasks"]["KE"])[:, 0, 0, 0]), axis=0
                    )
    # print(sc_time)
if args["--flux-balance"] or args["--depth-profile"] or args["--info"]:
    horiz_files = glob(direc + "horiz_aves/horiz_aves_s*.h5")
    horiz_files.sort(
        key=lambda f: int(re.sub("\D", "", f))
    )  # sort by number, avoiding s1, s10, s11, etc.
    for i, h_file in enumerate(horiz_files):
        if i == 0:
            with h5.File(h_file, "r") as file:
                horiz_time = np.array(file["scales"]["sim_time"])
                z = np.array(file["tasks"]["<T>"].dims[3]["z"])
                if args["--flux-balance"] or args["--info"]:
                    F_cond = np.array(file["tasks"]["<F_cond>"])[:, 0, 0, :] / 4
                    F_conv = np.array(file["tasks"]["<F_conv>"])[:, 0, 0, :] / 4
                if args["--depth-profile"]:
                    temp_hor = np.array(file["tasks"]["<T>"])[:, 0, 0, :] / 4
        else:
            with h5.File(h_file, "r") as file:
                horiz_time = np.concatenate(
                    (horiz_time, np.array(file["scales"]["sim_time"])), axis=0
                )
                if args["--flux-balance"] or args["--info"]:
                    F_cond = np.concatenate(
                        (F_cond, np.array(file["tasks"]["<F_cond>"])[:, 0, 0, :] / 4),
                        axis=0,
                    )
                    F_conv = np.concatenate(
                        (F_conv, np.array(file["tasks"]["<F_conv>"])[:, 0, 0, :] / 4),
                        axis=0,
                    )
                if args["--depth-profile"]:
                    temp_hor = np.concatenate(
                        (temp_hor, np.array(file["tasks"]["<T>"])[:, 0, 0, :] / 4),
                        axis=0,
                    )

if args["--gif"] or args["--profile-dissipation"]:
    snap_files = glob(direc + "snapshots/snapshots_s*.h5")
    snap_files.sort(key=lambda f: int(re.sub("\D", "", f)))
    for i, s_file in enumerate(snap_files):
        if i == 0:
            with h5.File(s_file, "r") as file:
                snap_time = np.array(file["scales"]["sim_time"])
                temp = np.array(file["tasks"]["Temp"])[:, 0, :, :]
                vel = np.array(file["tasks"]["u"])[:, :, 0, :, :]
                y = np.array(file["tasks"]["Temp"].dims[2]["y"])
                z = np.array(file["tasks"]["Temp"].dims[3]["z"])
        else:
            with h5.File(s_file, "r") as file:
                snap_time = np.concatenate(
                    (snap_time, np.array(file["scales"]["sim_time"])), axis=0
                )
                temp = np.concatenate(
                    (temp, np.array(file["tasks"]["Temp"])[:, 0, :, :]), axis=0
                )
                vel = np.concatenate(
                    (vel, np.array(file["tasks"]["u"])[:, :, 0, :, :]), axis=0
                )


read_finish = timer.time() - start_time
print(f"Done ({read_finish:.2f} seconds)")
# ! ============== Nusselt Number ============== ! #
if args["--info"]:
    print("====== Nusselt Number ======")
    Nu_start = timer.time()
    scalar_ASI = get_index(sc_time, float(args["--ASI"]))
    scalar_AEI = get_index(sc_time, float(2.0))
    scalar_AEI = None

    prof_ASI = get_index(horiz_time, float(args["--ASI"]))
    prof_AEI = get_index(horiz_time, float(2.0))
    prof_AEI = None

    # dT = np.nanmean(deltaT[scalar_ASI:scalar_AEI], axis=0)
    Re_ave = np.nanmean(Re[scalar_ASI:scalar_AEI], axis=0)

    F_cond_bar = np.nanmean(F_cond[prof_ASI:prof_AEI, :], axis=0)
    F_conv_bar = np.nanmean(F_conv[prof_ASI:prof_AEI, :], axis=0)
    F_cond_ave = np.trapz(F_cond_bar, z, axis=0)
    F_conv_ave = np.trapz(F_conv_bar, z, axis=0)
    nu = 1 + (F_conv_ave / F_cond_ave)

    print(f"\t Nu = {nu:.3f}")
    print(f"\t Re = {Re_ave:.3f}")

    with open(direc + "run_params/runparams.json", "r") as file:
        run_params = json.load(file)
        Ra = run_params["Ra"]
        Pr = run_params["Pr"]
        Ta = run_params["Ta"]
    with open(direc + "Nu.json", "w") as file:
        json.dump({"Ra": Ra, "Pr": Pr, "Ta": Ta, "Nu": nu}, file, indent=4)
    print(f"Done ({timer.time() - Nu_start:.2f}s).")

# # ! ============== Time-Tracks ============== ! #
if args["--time-tracks"]:
    print("====== Time-Tracks ======")
    time_start = timer.time()
    if len(deltaT) > 50000:
        skip_cadence = 100
    elif len(deltaT) > 5000:
        skip_cadence = 10
    else:
        skip_cadence = 1
    AEI = get_index(sc_time, float(1.9))
    AEI = None
    # skip_cadence = 1

    KE_run_ave = rolling_average(KE[::skip_cadence], sc_time[::skip_cadence])
    # Nu = 1 / deltaT
    # Nu_run_ave = rolling_average(Nu[::skip_cadence], sc_time[::skip_cadence])
    Re_run_ave = rolling_average(Re[::skip_cadence], sc_time[::skip_cadence])
    # fig, [KE_ax, Nu_ax] = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    fig, KE_ax = plt.subplots(1, 1, figsize=(6, 3))
    KE_ax.plot(
        sc_time[:AEI:skip_cadence], KE_run_ave[:AEI], label="Kinetic Energy", c="r"
    )

    RE_ax = KE_ax.twinx()
    RE_ax.scatter(
        sc_time[:AEI:skip_cadence],
        Re[:AEI:skip_cadence],
        marker="+",
        c="cyan",
        alpha=0.5,
    )
    RE_ax.plot(
        sc_time[:AEI:skip_cadence], Re_run_ave[:AEI], label="Reynolds Number", c="b"
    )
    # RE_ax.set_ylabel("Re")
    RE_ax.set_ylabel("Re", color="blue")
    RE_ax.tick_params(axis="y", labelcolor="blue")

    ylims = KE_ax.get_ylim()
    KE_ax.scatter(sc_time[:AEI:skip_cadence], KE[:AEI:skip_cadence], marker="+", c="k")
    # KE_ax.set_ylim(ylims)
    KE_ax.set_xlabel(r"Time, $\tau$")
    KE_ax.set_ylabel("KE")

    # Nu_ax.plot(
    #     sc_time[:AEI:skip_cadence],
    #     Nu_run_ave[:AEI:skip_cadence],
    #     label="Nusselt Number",
    #     c="r",
    # )
    # ylims = Nu_ax.get_ylim()
    # Nu_ax.scatter(sc_time[:AEI:skip_cadence], Nu[:AEI:skip_cadence], marker="+", c="k")
    # Nu_ax.set_ylim(ylims)
    # Nu_ax.set_xlabel(r"Time, $\tau$")
    # Nu_ax.set_ylabel("Nu")
    plt.tight_layout()
    plt.savefig(outpath + "time_tracks.pdf")
    print(f"Done ({timer.time() - time_start:.2f}s).")

# # ! ============== Dissipation  ============== ! #
if args["--profile-dissipation"]:
    print("====== Dissipation Profile ======")
    diss_start = timer.time()

    ASI = get_index(snap_time, float(args["--ASI"]))

    u = vel[:, 0, :, :]
    v = vel[:, 1, :, :]
    w = vel[:, 2, :, :]

    dyu = np.gradient(u, y, axis=1)
    dzu = np.gradient(u, z, axis=2)
    dyv = np.gradient(v, y, axis=1)
    dzv = np.gradient(v, z, axis=2)
    dyw = np.gradient(w, y, axis=1)
    dzw = np.gradient(w, z, axis=2)
    dyT = np.gradient(temp, y, axis=1)
    dzT = np.gradient(temp, z, axis=2)

    viscous_diss = (
        dyu**2 + 2 * dyv**2 + dyw**2 + dzu**2 + dzv**2 + 2 * dzw**2 + 2 * dyw * dzv
    )
    thermal_diss = dyT**2 + dzT**2

    viscous_prof_inst = np.trapz(viscous_diss, x=y, axis=1)
    thermal_prof_inst = np.trapz(thermal_diss, x=y, axis=1)

    viscous_prof_ave = np.trapz(viscous_prof_inst[ASI:], x=snap_time[ASI:], axis=0)
    thermal_prof_ave = np.trapz(thermal_prof_inst[ASI:], x=snap_time[ASI:], axis=0)

    viscous_diss_all = np.trapz(viscous_prof_ave, x=z)
    thermal_diss_all = np.trapz(thermal_prof_ave, x=z)

    viscous_lambda_idx = calculate_crossing(viscous_prof_ave / viscous_diss_all)
    thermal_lambda_idx = calculate_crossing(thermal_prof_ave / thermal_diss_all)
    if viscous_lambda_idx == -10 or thermal_lambda_idx == -10:
        print(f"diss doesn't cross 1.0")
    viscous_lambda = z[viscous_lambda_idx]
    thermal_lambda = z[thermal_lambda_idx]

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    # ax2 = ax.twinx()
    ax.plot(
        z,
        viscous_prof_ave / viscous_diss_all,
        label=r"$\epsilon_U$",
        c="r",
    )
    ax.plot(
        z,
        thermal_prof_ave / thermal_diss_all,
        label=r"$\epsilon_T$",
        c="b",
    )
    ax.axhline(1.0, c="k", ls=":")
    ax.axvline(viscous_lambda, c="r", ls=":")
    ax.axvline(thermal_lambda, c="b", ls=":")

    # ax.spines['left'].set_color('r')
    # ax.tick_params(axis='y', colors='r')
    ax.set_ylabel(r"$\langle \epsilon \rangle_H (z)$  / $\langle \epsilon \rangle_V$")

    # ax2.axhline(thermal_diss_all / thermal_diss_all, c='b', ls=':')
    # ax2.spines['left'].set_color('r')
    # ax2.spines['right'].set_color('b')
    # ax2.tick_params(axis='y', colors='b')
    # ax2.set_ylabel(r"$\langle \epsilon_T \rangle_H (z)$ / $\langle \epsilon_T \rangle_V$", color='b')
    ax.legend()
    ax.set_title("Time-Averaged Dissipation")
    ax.set_xlabel("z")
    plt.tight_layout()
    plt.savefig(outpath + "dissipation_profile.pdf")

    print(f"\n<ϵ_U> = {viscous_diss_all:.3e},\n<ϵ_T> = {thermal_diss_all:.3f}")
    # print(f"\nλ_U = {viscous_lambda[0]:.3f},\nλ_T = {thermal_lambda[0]:.3f}")

    np.savez(
        direc + "/dissipations",
        viscous=viscous_prof_ave / viscous_diss_all,
        thermal=thermal_prof_ave / thermal_diss_all,
        z=z,
    )

    print(f"Done ({timer.time() - diss_start:.2f}s).")

# # ! ============== Flux Balance ============== ! #
if args["--flux-balance"]:
    print("====== Flux Balance ======")
    flux_start = timer.time()
    #     with open(direc+'run_params/runparams.json', 'r') as file:
    #         params = json.load(file)
    #         driving_flux = params['F']
    ASI = get_index(horiz_time, float(args["--ASI"]))
    # AEI = get_index(horiz_time, float(2.0))
    with open(direc + "run_params/runparams.json", "r") as file:
        params = json.load(file)
        Ly = params["Ly"]

    AEI = None
    f_tot = F_cond + F_conv
    F_cond_bar = np.nanmean(F_cond[ASI:AEI], axis=0)
    F_conv_bar = np.nanmean(F_conv[ASI:AEI], axis=0)
    F_tot_bar = np.nanmean(f_tot[ASI:AEI], axis=0)

    heat_func = get_heat_func(args["--heat-func"])
    F_imp = cumtrapz(heat_func, z, initial=0)
    discrepency = np.mean(np.abs(F_imp - F_tot_bar))
    print(f"F_imp - F_tot discrepency = {discrepency:.3f}")
    # F_imp *= scaling

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(F_cond_bar, z, label=r"$F_{cond}$", c="b")
    ax.plot(F_conv_bar, z, label=r"$F_{conv}$", c="r")
    ax.plot(F_imp, z, label=r"$F_{imp}$", c="g")
    ax.plot(F_tot_bar, z, label=r"$F_{tot}$", c="k", ls="--")
    ax.set_xlabel("Flux")
    ax.set_ylabel("z")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath + "flux_balance.pdf")

    print(f"Done ({timer.time() - flux_start:.2f}s).")

# # ! ============== Temperature Profiles ============== ! #
if args["--depth-profile"]:
    print("====== Depth Profile ======")
    depth_start = timer.time()
    ASI = get_index(horiz_time, float(args["--ASI"]))
    temp_hor_bar = np.nanmean(temp_hor[ASI:], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(temp_hor_bar, z, c="k")
    ax.set_xlabel(r"$\langle \overline{T} \rangle$")
    ax.set_ylabel("z")
    ax.text(0.05, 0.05, f"ΔT = {temp_hor_bar[0]:.2f}", transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(outpath + "t_ave_depth_profile.pdf")
    print(f"Done ({timer.time() - depth_start:.2f}s).")
    depth_start = timer.time()
    print("Creating depth profile animation...")
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    makedirs(f"{direc}/plots", exist_ok=True)
    fnames = []
    cadence = int(args["--cadence"])
    for i, t in enumerate(horiz_time):
        if i % cadence == 0:
            print(f"\t Plotting frame {i+1}/{len(horiz_time)}", end="\r")
            ax.plot(temp_hor[i, :], z, c="k")
            ax.set_xlabel(r"$\langle \overline{T} \rangle$")
            ax.set_ylabel("z")
            ax.set_title(f"t = {t:.2f}")
            ax.text(0.05, 0.05, f"ΔT = {temp_hor[i, 0]:.2f}", transform=ax.transAxes)
            fnames.append(f"{direc}/plots/t_{i:04d}.png")
            plt.savefig(fnames[-1])
            ax.clear()
    print("\nDone. Creating GIF...")

    with imageio.get_writer(f"{outpath}/depth_profile.gif", mode="I") as writer:
        for i, filename in enumerate(fnames):
            print(f"\t frame {i+1}/{len(fnames)}", end="\r")
            image = imageio.imread(filename)
            writer.append_data(image)

    rmtree(f"{direc}/plots")
    print(f"\nDone ({timer.time() - depth_start:.2f}s).")

if args["--gif"]:
    print("====== Heatmap GIF ======")
    heatmap_start = timer.time()
    zz, yy = np.meshgrid(z, y)
    fnames = []
    vmax = 0.5
    # vmax = np.max(temp[len(temp) // 2:])
    # vmin = 1e-3
    # vmin = np.min(temp[len(temp) // 3:])
    vmin = np.min(temp)
    # print(vmin, vmax)
    linthresh = np.abs(vmin)
    # cNorm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)

    # cNorm = mpl.colors.SymLogNorm(linthresh=0.05, linscale=1, vmin=vmin, vmax=vmax, clip=False)
    levels = np.logspace(-2, 0, 1000, endpoint=True)
    levels = np.linspace(0, 1, 1000)
    # cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    # levels = np.linspace(vmin, vmax, 1000, endpoint=True)
    # all values of temp below 0, set to 0
    # temp[temp < 0] = vmin
    print("Plotting Frames...")
    makedirs(f"{direc}/plots", exist_ok=True)
    yspace = len(y) // 15
    zspace = len(z) // 5
    cadence = int(args["--cadence"])
    mpl.ticker.Locator.MAXTICKS = 1100
    for i, t in enumerate(snap_time):
        if i % cadence == 0:
            print(f"\t{(i+1) / len(snap_time) * 100:3.0f}% complete", end="\r")
            fig, ax = plt.subplots()
            # cax = fig.add_axes([0.90, 0.1, 0.02, 0.8])
            # cb1 = mpl.colorbar.ColorbarBase(cax, cmap='inferno', norm=cNorm, extend='min')
            # fig.subplots_adjust(left=0.1, right=0.85)
            # print(f"{i+1}/{len(snap_time)}", end='\r')
            # cax = ax.contourf(yy, zz, temp[i, :, :], cmap='inferno', extend='min', levels=levels, norm=cNorm)
            cax = ax.contourf(
                yy, zz, temp[i, :, :], cmap="inferno", extend="min", levels=999
            )
            warnings.filterwarnings("ignore")
            ax.quiver(
                yy[::yspace, ::zspace],
                zz[::yspace, ::zspace],
                vel[i, 1, ::yspace, ::zspace],
                vel[i, 2, ::yspace, ::zspace],
                color="w",
                pivot="mid",
            )
            ax.set_xlabel("y")
            ax.set_ylabel("z")
            ax.set_title(rf"{t:.2f} $\tau$")
            fig.colorbar(cax, label=r"$T$")
            fnames.append(f"{direc}plots/{i:04d}.png")
            plt.savefig(fnames[-1])
            fig.clf()
    print("\nCreating GIF...")
    with imageio.get_writer(f"{outpath}/heatmap.gif", mode="I") as writer:
        for i, filename in enumerate(fnames):
            print(f"\t frame {i+1}/{len(fnames)}", end="\r")
            image = imageio.imread(filename)
            writer.append_data(image)
    rmtree(f"{direc}/plots")
    print(f"\nDone ({timer.time() - heatmap_start:.2f}s).")
