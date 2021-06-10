import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
import os

def save_io_diff(tensor_a, tensor_b, path):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle("Input-Output comparison")

    X = tensor_a.permute(1, 2, 0)[:, :, 0]
    ax1.set_title('Input image')
    ax1.imshow(X, cmap='gray')

    Y = tensor_b.permute(1, 2, 0)[:, :, 0]
    ax2.set_title('Output image')
    ax2.imshow(Y, cmap='gray')

    ax3.set_title('Difference')
    mappable = ax3.imshow(Y - X, cmap="bwr")

    plt.colorbar(mappable, ax=ax3)

    plt.savefig(path)
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