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
    -l --last-frames                # Plot the last 30 frames of the simulation
    -s --shraiman-siggia            # Calculate the Shraiman-Siggia constant
    -u --vel_aves                   # Plot the velocity averages
    -a --anisotropy                 # Calculate the anisotropy of the flow
    -g --gif                        # Create a gif of the convection
    -h --help                       # Display this help message
    -v --version                    # Display the version
    --cadence [CADENCE]             # Cadence of the gifs [default: 1]
    --window [WINDOW]               # Window for rolling average [default: 0.05]
    --heat-func [HEAT]              # Heat function to use if not in snapshot [default: exp]
    --ASI [TIME]                    # Sim-time to begin average [default: -1]
    --AD [DURATION]                 # Time average duration [default: 2.0]
    --no-prefactor                  # Don't use the prefactor for the flux balance [default: True]

"""

from docopt import docopt
import h5py as h5
import numpy as np
from scipy.integrate import cumtrapz, trapezoid
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


def get_ave_indices(time):
    if float(args["--ASI"]) < 0:
        AEI = -1
        ASI = get_index(time, time[AEI] - float(args["--AD"]))
    else:
        ASI = get_index(time, float(args["--ASI"]))
        AEI = get_index(time, float(args["--ASI"]) + float(args["--AD"]))
    return ASI, AEI


def rolling_average(quantity, time, window: float):
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
        snaps = glob(direc + "snapshots/snapshots_s*.h5")
        snaps.sort(key=lambda f: int(re.sub("\D", "", f)))
        with h5.File(snaps[-1], "r") as file:
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


def dx(field):
    return np.zeros_like(field)


def dy(field):
    return np.array(np.gradient(field, y, axis=1))


def dz(field):
    return np.array(np.gradient(field, z, axis=2))


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

    scalar_ASI, scalar_AEI = get_ave_indices(sc_time)
    snap_ASI, snap_AEI = get_ave_indices(snap_time)
    prof_ASI, prof_AEI = get_ave_indices(prof_time)

    inv_t0 = 1 / T_0
    inv_t0_ave = np.nanmean(inv_t0[scalar_ASI:scalar_AEI], axis=0)
    print(f"\t1/<T(z=0)> =\t\t{inv_t0_ave:.5f}")

    # F_cond_ave = np.nanmean(F_cond, axis=1)
    # F_conv_ave = np.nanmean(F_conv, axis=1)

    F_cond_bar = np.nanmean(F_cond[prof_ASI:prof_AEI, :], axis=0)
    F_conv_bar = np.nanmean(F_conv[prof_ASI:prof_AEI, :], axis=0)
    F_cond_ave = np.trapz(F_cond_bar, z, axis=0)
    F_conv_ave = np.trapz(F_conv_bar, z, axis=0)
    flux_nu_ave = 1 + (F_conv_ave / F_cond_ave)

    print(f"\t1 + F_conv/F_cond =\t{flux_nu_ave:.5f}")

    inv_delta_T = 1 / (max_T - min_T)
    inv_delta_T_ave = np.nanmean(inv_delta_T[snap_ASI:snap_AEI], axis=0)
    print(f"\t1/dT = \t\t\t{inv_delta_T_ave:.5f}")
    print(
        f"\t\t T_max = {np.nanmean(max_T[snap_ASI:snap_AEI]):.5f}, T_min = {np.nanmean(min_T[snap_ASI:snap_AEI]):.5f}"
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

# ! ============== Scalar Load ============== ! #
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

# ! ============== Profile Load ============== ! #
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
                if args["--depth-profile"] or args["--info"]:
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
                if args["--depth-profile"] or args["--info"]:
                    temp_hor = np.concatenate(
                        (temp_hor, np.array(file["tasks"]["<T>"])[:, 0, 0, :] / 4),
                        axis=0,
                    )

# ! ============== Snapshot Load ============== ! #
if (
    args["--gif"]
    or args["--profile-dissipation"]
    or args["--vel_aves"]
    or args["--anisotropy"]
):
    snap_files = glob(direc + "snapshots/snapshots_s*.h5")
    snap_files.sort(key=lambda f: int(re.sub("\D", "", f)))
    for i, s_file in enumerate(snap_files):
        if i == 0:
            with h5.File(s_file, "r") as file:
                snap_time = np.array(file["scales"]["sim_time"])
                temp = np.array(file["tasks"]["Temp"])[:, 0, :, :]
                if args["--anisotropy"]:
                    vel = np.array(file["tasks"]["u"])[:, :, :, :, :]
                else:
                    vel = np.array(file["tasks"]["u"])[:, :, 0, :, :]
                x = np.array(file["tasks"]["Temp"].dims[1]["x"])
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
                if args["--anisotropy"]:
                    vel = np.concatenate(
                        (vel, np.array(file["tasks"]["u"])[:, :, :, :, :]), axis=0
                    )
                else:
                    vel = np.concatenate(
                        (vel, np.array(file["tasks"]["u"])[:, :, 0, :, :]), axis=0
                    )


read_finish = timer.time() - start_time
print(f"Done ({read_finish:.2f} seconds)")
# ! ============== Nusselt Number ============== ! #
if args["--info"]:
    print("====== Nusselt Number ======")
    Nu_start = timer.time()
    scalar_ASI, scalar_AEI = get_ave_indices(sc_time)
    print(f"ASI: {scalar_ASI}, AEI: {scalar_AEI}")
    prof_ASI, prof_AEI = get_ave_indices(horiz_time)

    l = 0.1
    beta = 1
    a = 1 / (l * (1 - np.exp(-1 / l)))
    T_steady = a * l * l * (np.exp(-1 / l) - 1) - (beta / 2) + a * l
    print(T_steady)
    temp_hor_bar = np.nanmean(temp_hor[prof_ASI:prof_AEI], axis=0)
    print(temp_hor_bar[0])
    KNu = T_steady / temp_hor_bar[0]
    print(f"\t Kaz Nu = {T_steady/temp_hor_bar[0]:.3f}")
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
        json.dump(
            {"Ra": Ra, "Pr": Pr, "Ta": Ta, "Nu": nu, "KNu": KNu, "Re": Re_ave},
            file,
            indent=4,
        )
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
    ASI, AEI = get_ave_indices(sc_time)
    # skip_cadence = 1

    KE_run_ave = rolling_average(
        KE[::skip_cadence], sc_time[::skip_cadence], window=float(args["--window"])
    )
    Re_run_ave = rolling_average(
        Re[::skip_cadence], sc_time[::skip_cadence], window=float(args["--window"])
    )

    fig, KE_ax = plt.subplots(1, 1, figsize=(6, 3))

    KE_ax.plot(
        sc_time[::skip_cadence],
        KE_run_ave[::],
        label="Kinetic Energy",
        c="r",
    )

    RE_ax = KE_ax.twinx()
    RE_ax.scatter(
        sc_time[::skip_cadence],
        Re[::skip_cadence],
        marker="+",
        c="cyan",
        alpha=0.5,
    )
    RE_ax.plot(sc_time[::skip_cadence], Re_run_ave[:], label="Reynolds Number", c="b")
    # RE_ax.set_ylabel("Re")
    RE_ax.set_ylabel("Re", color="blue")
    RE_ax.tick_params(axis="y", labelcolor="blue")

    ylims = KE_ax.get_ylim()
    KE_ax.scatter(sc_time[::skip_cadence], KE[::skip_cadence], marker="+", c="k")
    # KE_ax.set_ylim(ylims)
    KE_ax.set_xlabel(r"Time, $\tau$")
    KE_ax.set_ylabel("KE")
    plt.tight_layout()
    plt.savefig(outpath + "time_tracks.pdf")
    print(f"Done ({timer.time() - time_start:.2f}s).")

# # ! ============== Dissipation  ============== ! #
if args["--profile-dissipation"]:
    print("====== Dissipation Profile ======")
    diss_start = timer.time()

    ASI, AEI = get_ave_indices(snap_time)

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

    viscous_prof_ave = np.trapz(
        viscous_prof_inst[ASI:AEI], x=snap_time[ASI:AEI], axis=0
    )
    thermal_prof_ave = np.trapz(
        thermal_prof_inst[ASI:AEI], x=snap_time[ASI:AEI], axis=0
    )

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
    ASI, AEI = get_ave_indices(horiz_time)
    with open(direc + "run_params/runparams.json", "r") as file:
        params = json.load(file)
        Ly = params["Ly"]

    not_prefactor = args["--no-prefactor"]

    f_tot = F_cond + F_conv
    F_cond_bar = np.nanmean(F_cond[ASI:AEI], axis=0)
    F_conv_bar = np.nanmean(F_conv[ASI:AEI], axis=0)
    F_tot_bar = np.nanmean(f_tot[ASI:AEI], axis=0)
    a = 1 if not_prefactor else 1 / Ly
    print(a)
    heat_func = get_heat_func(args["--heat-func"])
    F_imp = a * cumtrapz(heat_func, z, initial=0)
    discrepency = np.trapz(np.abs(F_imp - F_tot_bar) / F_tot_bar, x=z) * 100
    print(f"F_imp - F_tot discrepency = {discrepency:.1f}%")
    print(f"Averaging between {horiz_time[ASI]:.2f} and {horiz_time[AEI]:.2f}")
    dicti = {}
    dicti["flux_match"] = discrepency
    # F_imp *= scaling
    with open(direc + "steady_state.json", "w") as file:
        json.dump(dicti, file, indent=4)

    np.savez(
        direc + "fluxes",
        F_cond=F_cond_bar,
        F_conv=F_conv_bar,
        F_tot=F_tot_bar,
        F_imp=F_imp,
        z=z,
    )

    indexes = np.asarray((np.diff(np.sign(F_cond_bar - F_conv_bar)) != 0) * 1).nonzero()
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(F_cond_bar, z, label=r"$F_{cond}$", c="b")
    ax.plot(F_conv_bar, z, label=r"$F_{conv}$", c="r")
    ax.plot(F_imp, z, label=r"$F_{imp}$", c="purple")
    ax.plot(F_tot_bar, z, label=r"$F_{tot}$", c="k", ls="--")
    if len(indexes[0]) < 2:
        print("Not enough crossings found.")
    else:
        therm_bot = z[indexes[0][0]]
        therm_top = z[indexes[0][-1]]
        mean_therm_bl = np.mean([therm_bot, 1 - therm_top])
        ax.axhspan(z[0], therm_bot, color="gray", alpha=0.5)
        ax.axhspan(therm_top, z[-1], color="gray", alpha=0.5)

        with open(direc + "therm_layer.json", "w") as file:
            json.dump(
                {"bot": therm_bot, "top": therm_top, "layer": mean_therm_bl},
                file,
                indent=4,
            )

    ax.set_xlabel("Flux")
    ax.set_ylabel("z")
    plt.legend(loc="best")
    plt.tight_layout()
    print(f"Saving to {outpath + 'new_flux_balance.pdf'}")
    plt.savefig(outpath + "new_flux_balance.pdf")

    print(f"Done ({timer.time() - flux_start:.2f}s).")

# # ! ============== Temperature Profiles ============== ! #
if args["--depth-profile"]:
    print("====== Depth Profile ======")
    depth_start = timer.time()
    ASI, AEI = get_ave_indices(horiz_time)
    temp_hor_bar = np.nanmean(temp_hor[ASI:AEI], axis=0)
    temp_bottom = temp_hor_bar[0]
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(temp_hor_bar, z, c="k")
    ax.set_xlabel(r"$\langle \overline{T} \rangle / T_b$")
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

# # ! ============== Heatmap Gif ============== ! #
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
            print(
                f"\t{((i/cadence)+1):.0f} / {len(snap_time)/cadence:.0f} frames",
                end="\r",
            )
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

# # ! ============== Shraiman-Siggia ============== ! #
if args["--shraiman-siggia"]:
    dir_path = normpath(args["FILE"]) + "/"
    print("======= Shraiman-Siggia =======")
    try:
        sc_files = sorted(
            glob(dir_path + "scalars/scalars_s*.h5"),
            key=lambda f: int(re.sub("\D", "", f)),
        )
        for i, sc_file in enumerate(sc_files[31:]):
            print(sc_file)
            if i == 0:
                with h5.File(sc_file, "r") as f:
                    gradT_sq = np.array(f["tasks"]["<(grad T)^2>"]).flatten()
                    gradu_sq = np.array(f["tasks"]["<(grad u)^2>"]).flatten()
                    qT = np.array(f["tasks"]["<QT>"]).flatten()
                    RawT = np.array(f["tasks"]["Ra*<wT>"]).flatten()
                    Re = np.array(f["tasks"]["Re"]).flatten()
                    t = np.array(f["scales"]["sim_time"]).flatten()
            else:
                with h5.File(sc_file, "r") as f:
                    gradT_sq = np.concatenate(
                        (gradT_sq, np.array(f["tasks"]["<(grad T)^2>"]).flatten()),
                        axis=0,
                    )
                    gradu_sq = np.concatenate(
                        (gradu_sq, np.array(f["tasks"]["<(grad u)^2>"]).flatten()),
                        axis=0,
                    )
                    qT = np.concatenate(
                        (qT, np.array(f["tasks"]["<QT>"]).flatten()), axis=0
                    )
                    RawT = np.concatenate(
                        (RawT, np.array(f["tasks"]["Ra*<wT>"]).flatten()), axis=0
                    )
                    Re = np.concatenate(
                        (Re, np.array(f["tasks"]["Re"]).flatten()), axis=0
                    )
                    t = np.concatenate((t, np.array(f["scales"]["sim_time"])), axis=0)

        ASI, AEI = get_ave_indices(t)

        ave_gradT_sq = np.nanmean(gradT_sq[ASI:AEI], axis=0)
        ave_RawT = np.nanmean(RawT[ASI:AEI], axis=0)
        ave_qT = np.nanmean(qT[ASI:AEI], axis=0)
        ave_gradu_sq = np.nanmean(gradu_sq[ASI:AEI], axis=0)
        ave_Re = np.nanmean(Re[ASI:AEI], axis=0)

        print(
            f"<(grad T)²> = {ave_gradT_sq:.5f}, <Ra*<wT>> = {ave_RawT:.5f}, <QT> = {ave_qT:.5f}, <(grad u)²> = {ave_gradu_sq:.5f}"
        )
        print(f"<wT> = {(ave_RawT / 8.3e8):.5f}")
        print(f"Re = {ave_Re:.5f}")
        with open(dir_path + "/run_params/runparams.json", "r") as file:
            run_params = json.load(file)
        with open(dir_path + "shraiman_siggia.json", "w") as file:
            json.dump(
                {
                    "Rf": run_params["Ra"],
                    "Ta": run_params["Ta"],
                    "gradT_sq": ave_gradT_sq,
                    "TQ": ave_qT,
                    "gradu_sq": ave_gradu_sq,
                    "RfwT": ave_RawT,
                },
                file,
                indent=4,
            )
        exit()
    except:
        print("****No scalar files found.****")
        exit()
        print("****Calculating from snapshots.****")

    snap_files = sorted(glob(dir_path + "snapshots/snapshots_s*.h5"))
    for i, snap_file in enumerate(snap_files):
        if i == 0:
            with h5.File(snap_file, "r") as f:
                Temp = np.array(f["tasks"]["Temp"])[:, :, :, :]
                vel_field = np.array(f["tasks"]["u"])[:, :, :, :, :]
                t = np.array(f["tasks"]["u"].dims[0]["sim_time"])
                x = np.array(f["tasks"]["u"].dims[2]["x"])
                y = np.array(f["tasks"]["u"].dims[3]["y"])
                z = np.array(f["tasks"]["u"].dims[4]["z"])
        else:
            with h5.File(snap_file, "r") as f:
                vel_field = np.concatenate(
                    (vel_field, np.array(f["tasks"]["u"])[:, :, :, :, :]), axis=0
                )
                Temp = np.concatenate(
                    (Temp, np.array(f["tasks"]["Temp"])[:, :, :, :]), axis=0
                )
                t = np.concatenate(
                    (t, np.array(f["tasks"]["Temp"].dims[0]["sim_time"])), axis=0
                )

    with open(dir_path + "/run_params/runparams.json", "r") as file:
        run_params = json.load(file)

    Lz = run_params["Lz"]
    Ly = run_params["Ly"]
    Lx = int(x[-1])
    prefactor = 1 / (Lx * Ly * Lz)
    Rf = run_params["Ra"]
    Ta = run_params["Ta"]
    ASI, AEI = get_ave_indices(t)

    print(Temp.shape)
    T_dt, T_dx, T_dy, T_dz = np.gradient(Temp, t, x, y, z)
    # T_dx = np.gradient(Temp, x, axis=1)
    # T_dy = np.gradient(Temp, y, axis=2)
    # T_dz = np.gradient(Temp, z, axis=3)

    gradT_sq = T_dx**2 + T_dy**2 + T_dz**2
    # vol_gradT_sq = (1 / 8) * np.trapz(
    #    np.trapz(np.trapz(gradT_sq, x, axis=1), y, axis=1), z, axis=1
    # )
    dx = np.gradient(x)
    dy = np.gradient(y)
    dz = np.gradient(z)
    weight = dx[None, :, None, None] * dy[None, None, :, None] * dz[None, None, None, :]
    weighted_gradT_sq = gradT_sq * weight
    vol_gradT_sq = np.mean(
        np.mean(np.mean(weighted_gradT_sq, axis=1), axis=1), axis=1
    ) / np.mean(weight)
    ave_gradT_sq = np.nanmean(vol_gradT_sq[ASI:AEI], axis=0)
    print(ave_gradT_sq)


# # ! ============== Velocity Averages ============== ! #
if args["--vel_aves"]:
    u = vel[:, 0, :, :]
    v = vel[:, 1, :, :]
    w = vel[:, 2, :, :]

    u_hor = np.trapz(
        u,
        y,
        axis=1,
    )
    v_hor = np.trapz(
        v,
        y,
        axis=1,
    )
    w_hor = np.trapz(
        w,
        y,
        axis=1,
    )

    ASI, AEI = get_ave_indices(snap_time)

    u_ave = np.nanmean(u_hor[ASI:AEI], axis=0)
    v_ave = np.nanmean(v_hor[ASI:AEI], axis=0)
    w_ave = np.nanmean(w_hor[ASI:AEI], axis=0)

    zt, tz = np.meshgrid(z, snap_time[ASI:AEI])

    print(u_hor.shape)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    c = ax.contourf(tz, zt, u_hor[ASI:AEI, :], cmap="RdBu_r", levels=100)
    ax2 = ax.twiny()
    ax2.plot(u_ave, z, c="k")
    ax.set_xlabel(r"$\tau_\nu$")
    ax.set_ylabel("z")
    ax.set_title(r"$\langle u \rangle_H$")
    fig.colorbar(c, ax=ax)
    fig.savefig(outpath + "u_aves.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    c = ax.contourf(tz, zt, v_hor[ASI:AEI, :], cmap="RdBu_r", levels=100)
    ax2 = ax.twiny()
    ax2.plot(v_ave, z, c="k")
    ax.set_xlabel(r"$\tau_\nu$")
    ax.set_ylabel("z")
    ax.set_title(r"$\langle v \rangle_H$")
    fig.colorbar(c, ax=ax)
    fig.savefig(outpath + "v_aves.pdf")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    c = ax.contourf(tz, zt, w_hor[ASI:AEI, :], cmap="RdBu_r", levels=100)
    ax2 = ax.twiny()
    ax2.plot(w_ave, z, c="k")
    ax.set_xlabel(r"$\tau_\nu$")
    ax.set_ylabel("z")
    ax.set_title(r"$\langle w \rangle_H$")
    fig.colorbar(c, ax=ax)
    fig.savefig(outpath + "w_aves.pdf")

# # ! ============== Anisotropy ============== ! #
if args["--anisotropy"]:
    dir_path = normpath(args["FILE"]) + "/"
    u = vel[:, 0, :, :, :]
    v = vel[:, 1, :, :, :]
    w = vel[:, 2, :, :, :]
    ASI, AEI = get_ave_indices(snap_time)

    u_rms = np.nanmean(
        np.sqrt(
            trapezoid(
                trapezoid(trapezoid(u * u, x=x, axis=1) ** 2, x=y, axis=1), x=z, axis=1
            )
        )[ASI:],
        axis=0,
    )
    u_xrms = np.nanmean(
        np.sqrt(
            trapezoid(
                trapezoid(trapezoid(u, x=x, axis=1) ** 2, x=y, axis=1), x=z, axis=1
            )
        )[ASI:],
        axis=0,
    )
    u_yrms = np.nanmean(
        np.sqrt(
            trapezoid(
                trapezoid(trapezoid(u, x=y, axis=2) ** 2, x=x, axis=1), x=z, axis=1
            )
        )[ASI:],
        axis=0,
    )
    u_zrms = np.nanmean(
        np.sqrt(
            trapezoid(
                trapezoid(trapezoid(u, x=z, axis=3) ** 2, x=x, axis=1), x=y, axis=1
            )
        )[ASI:],
        axis=0,
    )

    v_rms = np.nanmean(
        np.sqrt(
            trapezoid(
                trapezoid(trapezoid(v * v, x=x, axis=1) ** 2, x=y, axis=1), x=z, axis=1
            )
        )[ASI:],
        axis=0,
    )
    v_xrms = np.nanmean(
        np.sqrt(
            trapezoid(
                trapezoid(trapezoid(v, x=x, axis=1) ** 2, x=y, axis=1), x=z, axis=1
            )
        )[ASI:],
        axis=0,
    )
    v_yrms = np.nanmean(
        np.sqrt(
            trapezoid(
                trapezoid(trapezoid(v, x=y, axis=2) ** 2, x=x, axis=1), x=z, axis=1
            )
        )[ASI:],
        axis=0,
    )
    v_zrms = np.nanmean(
        np.sqrt(
            trapezoid(
                trapezoid(trapezoid(v, x=z, axis=3) ** 2, x=x, axis=1), x=y, axis=1
            )
        )[ASI:],
        axis=0,
    )

    w_rms = np.nanmean(
        np.sqrt(
            trapezoid(
                trapezoid(trapezoid(w * w, x=x, axis=1) ** 2, x=y, axis=1), x=z, axis=1
            )
        )[ASI:],
        axis=0,
    )
    w_xrms = np.nanmean(
        np.sqrt(
            trapezoid(
                trapezoid(trapezoid(w, x=x, axis=1) ** 2, x=y, axis=1), x=z, axis=1
            )
        )[ASI:],
        axis=0,
    )
    w_yrms = np.nanmean(
        np.sqrt(
            trapezoid(
                trapezoid(trapezoid(w, x=y, axis=2) ** 2, x=x, axis=1), x=z, axis=1
            )
        )[ASI:],
        axis=0,
    )
    w_zrms = np.nanmean(
        np.sqrt(
            trapezoid(
                trapezoid(trapezoid(w, x=z, axis=3) ** 2, x=x, axis=1), x=y, axis=1
            )
        )[ASI:],
        axis=0,
    )

    anisotropy = (u_rms - v_rms) / (u_rms + v_rms)
    print(anisotropy)

    with open(dir_path + "anisotropy.json", "w") as file:
        json.dump(
            {
                "u_rms": u_rms,
                "v_rms": v_rms,
                "w_rms": w_rms,
                "u_xrms": u_xrms,
                "v_xrms": v_xrms,
                "w_xrms": w_xrms,
                "u_yrms": u_yrms,
                "v_yrms": v_yrms,
                "w_yrms": w_yrms,
                "u_zrms": u_zrms,
                "v_zrms": v_zrms,
                "w_zrms": w_zrms,
                "anisotropy": anisotropy,
            },
            file,
            indent=4,
        )

# # ! ============== Last Frames ============== ! #
if args["--last-frames"]:
    print("==== Last Frames ====")
    dir_path = normpath(args["FILE"]) + "/"
    snap_files = sorted(glob(dir_path + "snapshots/snapshots_s*.h5"))
    print("Reading...")
    with h5.File(snap_files[-1], "r") as f:
        Temp = np.array(f["tasks"]["Temp"])[-30:, 0, :, :]
        vel_field = np.array(f["tasks"]["u"])[-30:, :, 0, :, :]
        t = np.array(f["tasks"]["u"].dims[0]["sim_time"])[-30:]
        y = np.array(f["tasks"]["u"].dims[3]["y"])
        z = np.array(f["tasks"]["u"].dims[4]["z"])

    zz, yy = np.meshgrid(z, y)
    yspace = len(y) // 15
    zspace = len(z) // 5
    fnames = []
    makedirs(f"{dir_path}plots", exist_ok=True)
    makedirs(f"{dir_path}images", exist_ok=True)
    print("Plotting Frames...")
    for i, f in enumerate(t):
        fig, ax = plt.subplots(figsize=(8, 4))
        cax = ax.contourf(
            yy,
            zz,
            Temp[i, :, :],
            cmap="inferno",
            levels=100,
            extend="min",
        )
        # ax.quiver(
        #     yy[::yspace, ::zspace],
        #     zz[::yspace, ::zspace],
        #     vel_field[i, 1, ::yspace, ::zspace],
        #     vel_field[i, 2, ::yspace, ::zspace],
        #     color="w",
        #     pivot="mid",
        # )
        ax.set_xlabel("y")
        ax.set_ylabel("z")
        ax.set_title(f"t = {f:.2f}")
        fig.colorbar(cax, label=r"$T$")
        fnames.append(f"{dir_path}plots/{i:04d}.png")
        plt.savefig(fnames[-1])
        plt.close()
    print("Creating GIF...")
    with imageio.get_writer(f"{dir_path}images/last_frames.gif", mode="I") as writer:
        for i, filename in enumerate(fnames):
            print(f"\t frame {i+1}/{len(fnames)}", end="\r")
            image = imageio.imread(filename)
            writer.append_data(image)
    rmtree(f"{dir_path}plots")
