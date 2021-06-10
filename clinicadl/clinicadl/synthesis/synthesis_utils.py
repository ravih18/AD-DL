import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import os

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))

def save_io_diff(tensor_a, tensor_b, path):
    fig, axes = plt.subplots(1, 3)
    (ax1, ax2, ax3) = axes
    fig.suptitle("Input-Output comparison")

    X = tensor_a.permute(1, 2, 0)[:, :, 0]
    ax1.set_title('Input image')
    ax1.imshow(X, cmap='gray')

    Y = tensor_b.permute(1, 2, 0)[:, :, 0]
    ax2.set_title('Output image')
    ax2.imshow(Y, cmap='gray')

    ax3.set_title('Difference')
    diff = Y - X 
    vmin = float(diff.min())
    vmax = float(diff.max())
    norm = MidpointNormalize(vmin=vmin, vmax=vmax, midpoint=0)
    mappable = ax3.imshow(diff, cmap="bwr", norm=norm)

    cax,kw = mpl.colorbar.make_axes([ax for ax in axes.flat], shrink=0.3, aspect=10)
    plt.colorbar(mappable, cax=cax, **kw)

    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    
    plt.close()


def save_pair(tensor_a, tensor_b, path):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Input image')
    ax1.imshow(tensor_a.permute(1, 2, 0)[:, :, 0])
    ax2.set_title('Output image')
    ax2.imshow(tensor_b.permute(1, 2, 0)[:, :, 0])
    plt.savefig(path)
    plt.close()


def save_latent_space(Z_r, labels, path):
    plt.figure()
    colors = ['navy', 'turquoise']
    lw = 2

    for color, i in zip(colors, [0, 1]):
        plt.scatter(Z_r[labels == i, 0], Z_r[labels == i, 1], color=color, alpha=.8, lw=lw)
    plt.legend(loc='best', shadow=False, scatterpoints=1)
    plt.title('PCA of latent space')
    plt.savefig(path)


def save_mean_score(eval_dict, path):

    eval_dict['mean_mse_score'] = sum(eval_dict['mse']) / len(eval_dict['mse'])
    eval_dict['mean_psnr_score'] = sum(eval_dict['psnr']) / len(eval_dict['psnr'])
    eval_dict['mean_ssim_score'] = sum(eval_dict['ssim']) / len(eval_dict['ssim'])

    file_path = os.path.join(path, "evaluation.txt")
    with open(file_path, "w") as f:
        for score in ['mean_mse_score', 'mean_psnr_score', 'mean_ssim_score']:
            f.write("{} :\t{}\n".format(score, eval_dict[score]))


def save_eval(eval_dict, sub, ses, label, path):
    df = pd.DataFrame({
        "participant_id": sub,
        "sessions_id": ses,
        "label": label,
        "mse_score": eval_dict['mse'],
        "psnr_score": eval_dict['psnr'],
        "ssim_score": eval_dict['ssim'],
    })
    df.to_csv(os.path.join(path, "test_result.tsv"), sep="\t")