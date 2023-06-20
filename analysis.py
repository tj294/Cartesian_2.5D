"""
An analysis script for 2.5D Cartesian convection. The script can create gifs showing
a heatmap of convection, as well as time-tracks of the Kinetic Energy and 1/<T>, the
time-averaged flux balance with depth, and both the instantaneous and time-averaged
Nusselt Number.

Usage:
    analysis.py FILE [options]
    
Options:
    -t --time-tracks                # Plot the time-track data (KE and 1/<T>)
    --index [index]                 # Index(es) of quicks to plot [default: -1]
    -d --depth-profile              # Plot the depth profile of the temperature
    -f --flux-balance               # Plot the flux-balance with depth
    -i --info                       # Information required 
    -g --gif                        # Create a gif of the convection
    -h --help                       # Display this help message
    -v --version                    # Display the version
    --ASI [TIME]                    # Sim-time to begin average [default: 0.65]

"""
from docopt import docopt
import h5py as h5
import numpy as np
import time as timer
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import imageio.v2 as imageio
from os import makedirs
from os.path import normpath
from shutil import rmtree
from glob import glob

def rolling_average(quantity, time, window=0.1):
    assert len(time) == len(quantity)
    run_ave = []
    for i, t0 in enumerate(time):
        mean = np.nanmean(quantity[(time > t0 - window/2) & (time <= t0 + window/2)])
        run_ave.append(mean)
    return np.array(run_ave)

def get_index(time, start_time):
    return np.abs(time - start_time).argmin()


args = docopt(__doc__, version='1.0')
print(args)

direc = normpath(args['FILE']) + '/'

start_time = timer.time()
print("====== Data Read-In ======")
if args['--info'] or args['--time-tracks']:
    scalar_files = glob(direc + 'scalars/scalars_s*.h5')
    for i, sc_file in enumerate(sorted(scalar_files)):
        if i==0:
            with h5.File(sc_file, 'r') as file:
                sc_time = np.array(file['scales']['sim_time'])
                deltaT = np.array(file['tasks']['<T(0)>'].dims[2])
                if args['--time-tracks']:
                    KE = np.array(file['tasks']['KE'])
        else:
            with h5.File(sc_file, 'r') as file:
                sc_time = np.concatenate((sc_time, np.array(file['scales']['sim_time'])), axis=0)
                deltaT = np.concatenate((deltaT, np.array(file['tasks']['<T(0)>'])), axis=0)
                if args['--time-tracks']:
                    KE = np.concatenate((KE, np.array(file['tasks']['KE'])), axis=0)
    
print(sc_time.shape)
print(deltaT)
print(KE.shape)
# for i, a_file in enumerate(sorted(analysis_files)):
#     if i==0:
#         with h5.File(a_file, 'r') as file:
#             Temp_vol = np.array(file['tasks']['<T>'])[:, Xidx, 0, 0]
#             Temp_hor = np.array(file['tasks']['<T>y'])[:, Xidx, 0, :]
#             F_cond = np.array(file['tasks']['F_cond'])[:, Xidx, 0, :]
#             F_conv = np.array(file['tasks']['F_conv'])[:, Xidx, 0, :]
#             KE = np.array(file['tasks']['KE'])[:, Xidx, :, :]
#             Nu_inst = np.array(file['tasks']['Nu_inst'])[:, Xidx, 0, 0]
#             x = np.array(file['tasks']['KE'].dims[1]['x'])
#             y = np.array(file['tasks']['KE'].dims[2]['y'])
#             z = np.array(file['tasks']['KE'].dims[3]['z'])
#             time = np.array(file['scales']['sim_time'])
#     else:
#         with h5.File(a_file, 'r') as file:
#             Temp_vol = np.concatenate(Temp_vol, np.array(file['tasks']['<T>'])[:, Xidx, 0, 0], axis=0)
#             Temp_hor = np.concatenate(Temp_hor, np.array(file['tasks']['<T>y'])[:, Xidx, 0, :], axis=0)
#             F_cond = np.concatenate(F_cond, np.array(file['tasks']['F_cond'])[:, Xidx, 0, :], axis=0)
#             F_conv = np.concatenate(F_conv, np.array(file['tasks']['F_conv'])[:, Xidx, 0, :], axis=0)
#             KE = np.concatenate(KE, np.array(file['tasks']['KE'])[:, Xidx, :, :], axis=0)
#             Nu_inst = np.concatenate(Nu_inst, np.array(file['tasks']['Nu_inst'])[:, Xidx, 0, 0], axis=0)
#             time = np.concatenate(time, np.array(file['scales']['sim_time']), axis=0)

# ASI = get_index(time, float(args['--ASI']))
exit(1)

# read_finish = timer.time() - start_time
# print(f"Done ({read_finish:.2f} seconds)")
# ! ============== Nusselt Number ============== ! #
if args["--info"]:
    print("====== Nusselt Number ======")
    Nu_start = timer.time()
    

    
    print(f"\t ΔT = {deltaT:.3f}\n"+
          f"\t 1/ΔT = {deltaT:.3f}")

    with open(direc + 'run_params/runparams.json', 'r') as file:
        run_params = json.load(file)
        Ra = run_params['Ra']
        Pr = run_params['Pr']
        Ta = run_params['Ta']
    with open(direc+'Nu.json', 'w') as file:
        json.dump({'Ra': Ra, 'Pr': Pr, 'Ta': Ta, 'F_cond/F_conv': Nu_char, '1/<T>': inv_t_ave, '1/<T[z=0]>': inv_t_0_ave}, file, indent=4)
    print(f"Done ({timer.time() - Nu_start:.2f}s).")

# ! ============== Time-Tracks ============== ! #
if args["--time-tracks"]:
    print("====== Time-Tracks ======")
    time_start = timer.time()
    KE_vol = np.trapz((np.trapz(KE, x=y, axis=1) / (y[-1] - y[0])), x=z, axis=1) / (z[-1] - z[0])
    KE_run_ave = rolling_average(KE_vol, time)
    Nu_run_ave = rolling_average(Nu_inst, time)
    PLOT_I = get_index(time, 0.1)

    fig, [KE_ax, Nu_ax] = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    KE_ax.plot(time, KE_run_ave, label='Kinetic Energy', c='r')
    KE_ax.scatter(time, KE_vol, marker='+', c='k')
    KE_ax.set_xlabel(r"Time, $\tau_\nu$")
    KE_ax.set_ylabel("KE")
    KE_ax.set_ylim([0, 1.2 * np.nanmax(KE_vol[PLOT_I:])])

    Nu_ax.plot(time, Nu_run_ave, label='Nusselt Number', c='r')
    Nu_ax.scatter(time, Nu_inst, marker='+', c='k')
    Nu_ax.set_xlabel(r"Time, $\tau_\nu$")
    Nu_ax.set_ylabel("Nu")
    Nu_ax.set_ylim([0, 1.2 * np.nanmax(Nu_inst[PLOT_I:])])
    
    plt.tight_layout()
    plt.savefig(direc + 'time_tracks.pdf')

    print(f'Done ({timer.time() - time_start:.2f}s).')

# ! ============== Flux Balance ============== ! #
if args["--flux-balance"]:
    print("====== Flux Balance ======")
    flux_start = timer.time()
    with open(direc+'run_params/runparams.json', 'r') as file:
        params = json.load(file)
        driving_flux = params['F']
    
    f_tot = F_cond + F_conv
    F_cond_bar = np.nanmean(F_cond[ASI:], axis=0)
    F_conv_bar = np.nanmean(F_conv[ASI:], axis=0)
    F_tot_bar = np.nanmean(f_tot[ASI:], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(F_cond_bar, z, label=r'$F_{cond}$', c='b')
    ax.plot(F_conv_bar, z, label=r'$F_{conv}$', c='r')
    ax.plot(F_tot_bar, z, label=r'$F_{tot}$', c='k')
    ax.set_xlabel('Flux')
    ax.set_ylabel('z')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(direc + 'flux_balance.pdf')

    print(f'Done ({timer.time() - flux_start:.2f}s).')

# ! ============== Temperature Profiles ============== ! #
if args['--depth-profile']:
    print('====== Depth Profile ======')
    depth_start = timer.time()
    temp_hor_bar = np.nanmean(Temp_hor[ASI:], axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(temp_hor_bar/temp_hor_bar[0], z, c='k')
    ax.set_xlabel(r'$\langle \overline{T} \rangle / \langle \overline{T[z=0]} \rangle$')
    ax.set_ylabel('z')
    plt.tight_layout()
    plt.savefig(direc + 't_ave_depth_profile.pdf')
    print(f'Done ({timer.time() - depth_start:.2f}s).')
    depth_start = timer.time()
    print('Creating depth profile animation...')
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    makedirs(f"{direc}/plots", exist_ok=True)
    fnames = []
    for i, t in enumerate(time):
        print(f"\t Plotting frame {i+1}/{len(time)}", end='\r')
        ax.plot(Temp_hor[i, :]/Temp_hor[i, 0], z, c='k')
        ax.set_xlabel(r'$\langle \overline{T} \rangle / \langle \overline{T[z=0]} \rangle$')
        ax.set_ylabel('z')
        ax.set_title(f't = {t:.2f}')
        fnames.append(f"{direc}/plots/t_{i:04d}.png")
        plt.savefig(fnames[-1])
        ax.clear()
    print('\nDone. Creating GIF...')
    
    with imageio.get_writer(f"{direc}/depth_profile.gif", mode="I") as writer:
        for i, filename in enumerate(fnames):
            print(f"\t frame {i+1}/{len(fnames)}", end='\r')
            image = imageio.imread(filename)
            writer.append_data(image)

    rmtree(f"{direc}/plots")
    print(f'\nDone ({timer.time() - depth_start:.2f}s).')

if args['--gif']:
    print('====== Heatmap GIF ======')
    heatmap_start = timer.time()
    for i, s_file in enumerate(sorted(snapshot_files)):
        if i==0:
            with h5.File(s_file, 'r') as file:
                Temp = np.array(file['tasks']['Temp'])[:, 0, :, :]
        else:
            with h5.File(s_file, 'r') as file:
                Temp = np.concatenate(np.array(file['tasks']['Temp']), axis=0)
    print(Temp.shape)
    zz, yy = np.meshgrid(z, y)
    fnames = []
    vmin = np.min(Temp[len(Temp) // 3:])
    vmax = np.max(Temp[len(Temp) // 3:])
    cNorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    levels = np.linspace(vmin, vmax, 100, endpoint=True)
    fig, ax = plt.subplots()
    cax = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    cb1 = mpl.colorbar.ColorbarBase(cax, cmap='inferno', norm=cNorm)
    fig.subplots_adjust(left=0.05, right=0.85)
    print("Plotting Frames...")
    makedirs(f'{direc}/figure', exist_ok=True)
    for i, t in enumerate(time):
        print(f"\t{(i+1) / len(time) * 100:3.0f}% complete", end='\r')
        cax = ax.contourf(yy, zz, Temp[i, :, :], levels=levels, cmap='inferno', extend='both')
        ax.set_title(rf"{t:.2f} $\tau_\nu$")
        fnames.append(f"{direc}/figure/{i:04d}.png")
        plt.savefig(fnames[-1])
        ax.cla()
    print("\nCreating GIF...")
    with imageio.get_writer(f"{direc}/heatmap.gif", mode="I") as writer:
        for i, filename in enumerate(fnames):
            print(f"\t frame {i+1}/{len(fnames)}", end='\r')
            image = imageio.imread(filename)
            writer.append_data(image)
    rmtree(f"{direc}/figure")
    print(f'\nDone ({timer.time() - heatmap_start:.2f}s).')
