import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rc("axes.spines", top=False, right=False)


def load_decoding(bids_root, subs, task='rcor'):
    import os, json
    from collections import defaultdict

    out = defaultdict(list)
    for sub in subs:
        base = os.path.join(bids_root, 'derivatives', 'decoding', sub, 'eeg',
                            f'{sub}_task-{task}_decoding')

        dat = np.load(base + '.npz')
        with open(base + '.json') as f:
            meta = json.load(f)

        out.update({k: out[k] + [dat[k]] for k in dat})
        out.update({k: out[k] + [meta[k]] for k in meta})

    return {k: np.array(l) for k,l in out.items()}


def plot_decoding(data):

    def boot(arr, metric, n_iterations=1000, *args, **kwargs):
        n = arr.shape[0]
        out = []
        for _ in range(n_iterations):
            idx = np.random.randint(0, n, n)
            out.append(metric(arr[idx], *args, **kwargs))
        return np.asarray(out)

    conditions = ['challenge', 'control']
    order = [[1,1],[0,0],[1,0],[0,1]]
    palette = ["#177782","#a42918","#67c1c1","#df7712","#ffd149"]
    t = data['time'][0]

    da = np.nanmean(np.diagonal(data['TG'], axis1=-2, axis2=-1), axis=(3,4)) - .5
    avg = boot(da, np.mean, axis=0).mean(axis=0)
    sem = boot(da, np.mean, axis=0).std(axis=0)

    dif = da[:,1,1] - da[:,0,0]
    dif_avg = boot(dif, np.mean, axis=0).mean(axis=0)
    dif_sem = boot(dif, np.mean, axis=0).std(axis=0)

    plt.figure()
    plt.axvline(0,c='tab:gray',linestyle='-', alpha =.4)
    plt.axhline(0,c='tab:gray',linestyle='-', alpha =.4)
    plt.plot(t, dif_avg, label=f'difference within-set', color=palette[4])
    plt.fill_between(t, dif_avg-dif_sem, dif_avg+dif_sem, alpha=.3, color=palette[4])
    for i,(c1,c2) in enumerate(order):
            plt.plot(t, avg[c1,c2], label=f'{conditions[c1]} - {conditions[c2]}', color=palette[i])
            plt.fill_between(t, avg[c1,c2]-sem[c1,c2], avg[c1,c2]+sem[c1,c2], alpha=.3, color=palette[i])
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Decoding accuracy - chance (%)')
    plt.show()


def plot_tgm(data):
    import matplotlib.colors as mcolors

    conditions = ['challenge', 'control']
    order = [[1,1],[0,0],[1,0],[0,1]]
    tgm = np.nanmean(data['TG'], axis=(0,3,4)) - .5
    t = data['time'][0]

    t_dif = np.diff(t)[0] / 2
    plt_extent = tuple([t[0] - t_dif, t[-1] + t_dif] * 2)

    vmax = np.round(tgm.max(), 2)
    vmin = -vmax
    cnorm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cticks = [vmin, 0, vmax]

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True)
    axf = axs.flatten()
    for i,(c1,c2)  in enumerate(order):
        ax = axf[i]
        dat = tgm[c1,c2]
        im = ax.imshow(dat, origin='lower', extent=plt_extent,
                interpolation='none', cmap='RdBu_r', norm=cnorm)
        ax.axvline(0,c='tab:gray',linestyle='-', alpha=.4)
        ax.axhline(0,c='tab:gray',linestyle='-', alpha=.4)
        ax.plot((-.05,t[-1]),(-.05,t[-1]),c='tab:gray',linestyle='-', alpha=.4)
        ax.set_title(f'{conditions[c1]} - {conditions[c2]}')
        ax.set_ylabel('Training time (s)')
        ax.set_xlabel('Testing time (s)')
    fig.colorbar(im, ax=axf, shrink=.5, orientation='horizontal', 
                    ticks=cticks, label='Decoding accuracy - chance (%)')
    plt.show()


def plot_asymmetry(data):

    def boot(arr, metric, n_iterations=1000, *args, **kwargs):
        n = arr.shape[0]
        out = []
        for _ in range(n_iterations):
            idx = np.random.randint(0, n, n)
            out.append(metric(arr[idx], *args, **kwargs))
        return np.asarray(out)

    def antidiag_asymmetry(x):
        x = np.nanmean(x, axis=0) # average over subjects (conditions, conditions, time, time)
        n_conds, _, n_time, _ = x.shape
        dlist = np.arange(-n_time, n_time)
        dlist = dlist[1::2]
        asym = np.full((n_conds, n_conds, len(dlist)), np.nan)
        for k, d in enumerate(dlist):
            diag = np.flip(x, axis=2).diagonal(d, axis1=2, axis2=3)  # anti-diagonal in original matrix
            diag = np.delete(diag, diag.shape[2] // 2, axis=2)  # remove center
            a, b = np.split(diag, 2, axis=2)
            asym[:,:,k] = np.nansum(a, axis=2) - np.nansum(b, axis=2)
        return asym


    conditions = ['challenge', 'control']
    order = [[1,1],[0,0],[1,0],[0,1]]
    palette = ["#177782","#a42918","#67c1c1","#df7712","#ffd149"]
    t = data['time'][0]

    tgm = np.nanmean(data['TG'], axis=(3,4)) - .5
    asym_boot = boot(tgm, antidiag_asymmetry)
    avg = asym_boot.mean(axis=0)
    sem = asym_boot.std(axis=0)

    plt.figure()
    plt.axvline(0,c='tab:gray',linestyle='-', alpha =.4)
    plt.axhline(0,c='tab:gray',linestyle='-', alpha =.4)
    for i,(c1,c2) in enumerate(order):
            plt.plot(t, avg[c1,c2], label=f'{conditions[c1]} - {conditions[c2]}', color=palette[i])
            plt.fill_between(t, avg[c1,c2]-sem[c1,c2], avg[c1,c2]+sem[c1,c2], alpha=.3, color=palette[i])
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Decoding accuracy - chance (%)')
    plt.show()



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('bids_root', help='Path to BIDS dataset root')
    args = parser.parse_args()

    bids_root = args.bids_root

    # infer subjects from derivatives/decoding
    dec_dir = os.path.join(bids_root, 'derivatives', 'decoding')
    subs = sorted(d for d in os.listdir(dec_dir) if d.startswith('sub-'))

    data = load_decoding(bids_root, subs)

    plot_decoding(data)
    plot_tgm(data)
    plot_asymmetry(data)
