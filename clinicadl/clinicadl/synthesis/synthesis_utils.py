import matplotlib.pyplot as plt

def save_pair(tensor_a, tensor_b, path):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('Input image')
    ax1.imshow(tensor_a.permute(1, 2, 0)[:, :, 0])
    ax2.set_title('Output image')
    ax2.imshow(tensor_b.permute(1, 2, 0)[:, :, 0])
    plt.savefig(path)