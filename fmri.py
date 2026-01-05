import numpy as np
import matplotlib.pyplot as plt
plt.rc('axes.spines', top=False, right=False)


def load_rois(bids_root, subs, task='rcor', result='decoding-rois'):
    import os, json
    from collections import defaultdict

    out = defaultdict(list)
    for sub in subs:
        base = os.path.join(bids_root, 'derivatives', result, sub,
                            f'{sub}_task-{task}_{result}')
        dat = np.load(base + '.npz')
        out.update({k: out[k] + [dat[k]] for k in dat})

    return {k: np.array(l) for k,l in out.items()}


def plot_decoding_roi(data):
    import pandas as pd
    import seaborn as sns
    import matplotlib as mpl

    order = [[1,1],[0,0],[1,0],[0,1]]
    palette = ["#177782","#a42918","#67c1c1","#df7712"]
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=palette)
    
    conditions = data['cat'][0]
    hue_order = [f'{conditions[i]} - {conditions[j]}' for i,j in order]
    roi_labels = data['rois'][0]

    da = np.nanmean(data['metric'], axis=(4,5)) - .5
    n_sub, n_train, n_test, n_roi = da.shape

    da_f = da.transpose(1, 2, 3, 0)  # train, test, roi, subject
    df = pd.DataFrame({
        'da': da_f.reshape(-1),
        'train': np.repeat(conditions, n_test * n_roi * n_sub),
        'test':  np.tile(np.repeat(conditions, n_roi * n_sub), n_train),
        'roi':   np.tile(np.repeat(roi_labels, n_sub), n_train * n_test),
        'subject': np.tile(np.arange(n_sub), n_train * n_test * n_roi),
    })
    df['condition'] = df.train.str.cat(others=[df.test], sep=' - ')

    plt.figure()
    plt.axhline(0,c='tab:gray',linestyle='-', alpha =.4)
    plt.xlabel('Region of interest')
    plt.ylabel('Decoding accuracy - chance (%)')
    sns.stripplot(
        data=df, x='roi', y='da', hue='condition',hue_order=hue_order,
        dodge=True, alpha=.15, zorder=1, legend=False
    )
    sns.pointplot(
        data=df, x='roi', y='da', hue='condition',hue_order=hue_order,
        join=False, dodge=.8 - .8 / 4,
        markers='o', scale=.75, errorbar='ci'
    )


def load_searchlight(bids_root, subs, task='rcor'):
    import os
    from nilearn import image

    conditions = ['challenge', 'control']
    n_cond = len(conditions)
    out = []
    ref_img = None
    for sub in subs:
        in_dir = os.path.join(bids_root, 'derivatives', 'decoding-searchlight', sub)
        vols = [[None] * n_cond for _ in range(n_cond)]
        for i, c1 in enumerate(conditions):
            for j, c2 in enumerate(conditions):
                in_file = os.path.join(in_dir, f'{sub}_task-{task}_decoding-searchlight_{c1}-{c2}.nii.gz')
                img = image.load_img(in_file)
                if ref_img is None:
                    ref_img = img 
                vols[i][j] = img.get_fdata().astype(float)  #(x,y,z)
        out.append(vols)

    return np.array(out), ref_img, conditions



def plot_decoding_sl(data, ref_img, conditions):
    import matplotlib.colors as mcolors
    from nilearn import image
    from nilearn.datasets import load_mni152_template

    def truncate_colormap(cmap, minval=0, maxval=1, n=100):
        return mcolors.LinearSegmentedColormap.from_list(
            f'trunc({cmap.name},{minval:.2f},{maxval:.2f})',
            cmap(np.linspace(minval, maxval, n))
        )

    mean_map = np.nanmean(data, axis=0)  # (c1,c2,x,y,z)
    mean_map[mean_map < .1] = np.nan    # threshold

    comparisons = [(1,1), (0,0), (1,0), (0,1)]
    n_comps = len(comparisons)
    z_cuts = [22, 29, 33, 36, 40, 44, 48, 51, 55, 60, 66]
    n_cuts = len(z_cuts)

    # background
    template = load_mni152_template()
    template = image.resample_to_img(template, ref_img, interpolation='continuous').get_fdata().astype(float)
    template[template < .1] = np.nan

    # colormaps
    brain_cmap = truncate_colormap(plt.get_cmap('binary'), .05, .9)
    backnorm = mcolors.TwoSlopeNorm(vmin=.2, vcenter=.85, vmax=.9)
    da_cmap = plt.get_cmap('plasma')

    fig, axs = plt.subplots(n_comps, n_cuts, constrained_layout=True)
    for i, (c1, c2) in enumerate(comparisons):
        dat = mean_map[c1, c2]
        label = f'{conditions[c1]} - {conditions[c2]}'
        axs[i, n_cuts // 2].set_title(label, pad=10)

        for j,z in enumerate(z_cuts):
            ax = axs[i,j]
            ax.imshow(template[:, :, z].T, cmap=brain_cmap, norm=backnorm, alpha=0.7, origin='lower')
            ax.imshow(dat[:, :, z].T, cmap=da_cmap, origin='lower')
            ax.set_xlim(2, 76)
            ax.set_ylim(2, 93)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ["left", "right", "bottom", "top"]:
                ax.spines[s].set_visible(False)


def plot_rsa(rdms):
    from scipy.stats import spearmanr
    from scipy.linalg import orthogonal_procrustes
    import matplotlib.colors as mcolors

    def plot_matrix(ax, d, tri=None,threshold=.1):
        for x in range(n_roi):
            for y in range(n_roi):
                if tri == 'lower' and y >= x:
                    continue
                if tri == 'upper' and y <= x:
                    continue

                i, j = (y, x) if tri == 'lower' else (x, y)
                val = np.round(d[i, j],2)
                lab = f'{val:.2f}'.replace('0.', '.')
                ax.scatter(x, y, c=[val], cmap=palette, norm=cnorm, marker='s', s=150)
                if np.abs(val) > threshold:
                    ax.annotate(lab, (x, y), ha='center', va='center')
                else:
                    ax.annotate(lab, (x, y), ha='center', va='center', fontsize='small', color='gray')

    def style_matrix(ax, title, xaxis='top'):
        ax.set_xticks(np.arange(n_roi))
        ax.set_yticks(np.arange(n_roi))
        ax.set_xticklabels(rois, rotation=90)
        ax.set_yticklabels(rois)
        ax.set_title(title)
        if xaxis == 'top':
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()
        for spine in ax.spines.values():
            spine.set_visible(False)

    def compute_mds_coords(sim):
        from sklearn.manifold import MDS
        sim = np.nanmean([sim, sim.T], axis=0)
        np.fill_diagonal(sim, 1)
        dist = 1 - sim
        coords = MDS(n_components=2, dissimilarity='precomputed', random_state=0).fit_transform(dist)
        return coords

    def add_mds_inset(ax, coords, rois):
        roi_groups = ['visual','visual','visual','visual','visual','mTC','frontal','frontal','frontal','frontal','frontal']
        group_colors={'visual': '#377eb8', 'mTC': '#E6AB02', 'frontal': '#4DAF4A'}
        inset_ax = ax.inset_axes([0.5, 0.05, 0.45, 0.45])
        for i in range(len(rois)):
            group = roi_groups[i]
            color = group_colors.get(group, 'gray')
            inset_ax.scatter(coords[i, 0], coords[i, 1], color=color, s=20)
            inset_ax.text(coords[i, 0], coords[i, 1] -.05, rois[i], fontsize=10,
                          ha='center', va='top')
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_frame_on(True)


    sim = rdms['metric']    # (sub,c1,c2,roi,obj,obj)
    n_sub, n_cond, _, n_roi, n_obj, _ = sim.shape
    conditions = rdms['cat'][0]
    rois = rdms['rois'][0]
    

    mask_i, mask_j = np.triu_indices(n_obj, 1)
    sim = sim[:,:,:,:,mask_i,mask_j]

    within = np.array([[spearmanr(sim[s,c,c], axis=1).statistic 
                        for c in range(n_cond)] for s in range(n_sub)])

    between = np.array([spearmanr(sim[s,1,1], sim[s,0,0], axis=1).statistic[:n_roi, n_roi:]
                        for s in range(n_sub)])

    avg_w = np.nanmean(within, axis=0)  # (cond,roi,roi)
    avg_b = np.nanmean(between, axis=0)  # (roi,roi)

    # MDS
    coords_ctrl = compute_mds_coords(avg_w[1])
    coords_chal = compute_mds_coords(avg_w[0])
    # Align challenge to control
    R, _ = orthogonal_procrustes(coords_chal, coords_ctrl)
    coords_chal_aligned = coords_chal @ R

    cnorm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    palette = 'RdBu_r'

    fig, axs = plt.subplots(3, 2, figsize=(8,12), constrained_layout=True)
    # top left: control
    plot_matrix(axs[0,0], avg_w[1], tri='upper')
    style_matrix(axs[0,0], 'control')
    add_mds_inset(axs[0,0], coords_ctrl, rois)

    # top right: challenge
    plot_matrix(axs[0,1], avg_w[0].T, tri='upper')
    style_matrix(axs[0,1], 'challenge')
    add_mds_inset(axs[0,1], coords_chal_aligned, rois)

    # center left: cross-set similarity
    plot_matrix(axs[1,0], avg_b)
    style_matrix(axs[1,0], 'cross-set similarity', xaxis='bottom')

    # center right: difference
    plot_matrix(axs[1,1], avg_w[1] - avg_w[0], tri='upper')
    style_matrix(axs[1,1], 'control minus challenge')

    # bottom left: within control - between
    plot_matrix(axs[2,0], avg_w[1] - avg_b, tri='upper')
    style_matrix(axs[2,0], 'control minus between')

    # bottom right: within challenge - between
    plot_matrix(axs[2,1], avg_w[0] - avg_b, tri='upper')
    style_matrix(axs[2,1], 'challenge minus between')


if __name__=='__main__':

    bids_root = '/path/to/dataset'
    subs = [f'sub-{i:02d}' for i in range(1,31)]

    dec_roi = load_rois(bids_root, subs, result='decoding-rois')
    plot_decoding_roi(dec_roi)

    dec_sl = load_searchlight(bids_root, subs)
    plot_decoding_sl(*dec_sl)

    rdms = load_rois(bids_root, subs, result='rdms')
    plot_rsa(rdms)


