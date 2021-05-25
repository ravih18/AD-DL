import numpy as np
import os
import logging
import warnings
import torch
from torch import nn
from torch.nn import functional as F
import pandas as pd
from time import time

from clinicadl.tools.deep_learning.iotools import check_and_clean
from clinicadl.tools.deep_learning import EarlyStopping, save_checkpoint


def train(model, train_loader, valid_loader, criterion, optimizer, resume,
          log_dir, model_dir, options, logger=None):
    """
    Function used to train an VAE.
    The best VAE will be found in the 'best_model_dir' of options.output_dir.

    Args:
        decoder: (VAE) VAE constructed from a CNN with the VAE class.
        train_loader: (DataLoader) wrapper of the training dataset.
        valid_loader: (DataLoader) wrapper of the validation dataset.
        criterion: (loss) function to calculate the loss.
        optimizer: (torch.optim) optimizer linked to model parameters.
        resume: (bool) if True, a begun job is resumed.
        log_dir: (str) path to the folder containing the logs.
        model_dir: (str) path to the folder containing the models weights and biases.
        options: (Namespace) ensemble of other options given to the main script.
        logger: (logging object) writer to stdout and stderr
    """
    from tensorboardX import SummaryWriter

    columns = ['epoch', 'iteration', 'time', 'loss_train', 'loss_valid']
    filename = os.path.join(os.path.dirname(log_dir), 'training.tsv')

    if logger is None:
        logger = logging

    columns = ['epoch', 'iteration', 'time', 'loss_train', 'loss_valid']
    filename = os.path.join(os.path.dirname(log_dir), 'training.tsv')

    if not resume:
        check_and_clean(model_dir)
        check_and_clean(log_dir)

        results_df = pd.DataFrame(columns=columns)
        with open(filename, 'w') as f:
            results_df.to_csv(f, index=False, sep='\t')
        options.beginning_epoch = 0

    else:
        if not os.path.exists(filename):
            raise ValueError(
                'The training.tsv file of the resumed experiment does not exist.')
        truncated_tsv = pd.read_csv(filename, sep='\t')
        truncated_tsv.set_index(['epoch', 'iteration'], inplace=True)
        truncated_tsv.drop(options.beginning_epoch, level=0, inplace=True)
        truncated_tsv.to_csv(filename, index=True, sep='\t')

    # Create writers
    writer_train = SummaryWriter(os.path.join(log_dir, 'train'))
    writer_valid = SummaryWriter(os.path.join(log_dir, 'validation'))

    model.train()
    train_loader.dataset.train()
    logger.debug(model)

    if options.gpu:
        model.cuda()

    # Initialize variables
    best_loss_valid = np.inf
    epoch = options.beginning_epoch

    early_stopping = EarlyStopping(
        'min', min_delta=options.tolerance, patience=options.patience)
    loss_valid = None
    t_beginning = time()

    logger.debug("Beginning training")

    while epoch < options.epochs and not early_stopping.step(loss_valid):
        logger.info("Beginning epoch %i." % epoch)

        model.zero_grad()
        evaluation_flag = True
        step_flag = True

        for i, data in enumerate(train_loader):
            if options.gpu:
                imgs = data['image'].cuda()
            else:
                imgs = data['image']
            
            optimizer.zero_grad()
            recon_imgs, mu, log_var = model(imgs)
            
            loss = criterion(recon_imgs, imgs, mu, log_var)
            loss.backward()

            del imgs, recon_imgs, mu, log_var

            if (i + 1) % options.accumulation_steps == 0:
                step_flag = False
                optimizer.step()
                optimizer.zero_grad()

                # Evaluate the decoder only when no gradients are accumulated
                if options.evaluation_steps != 0 and (i + 1) % options.evaluation_steps == 0:
                    evaluation_flag = False
                    loss_train = test_vae(
                        model, train_loader, options.gpu, criterion)
                    mean_loss_train = loss_train / \
                        (len(train_loader) * train_loader.batch_size)

                    loss_valid = test_vae(
                        model, valid_loader, options.gpu, criterion)
                    mean_loss_valid = loss_valid / \
                        (len(valid_loader) * valid_loader.batch_size)
                    model.train()
                    train_loader.dataset.train()

                    writer_train.add_scalar(
                        'loss', mean_loss_train, i + epoch * len(train_loader))
                    writer_valid.add_scalar(
                        'loss', mean_loss_valid, i + epoch * len(train_loader))
                    logger.info("%s level training loss is %f at the end of iteration %d"
                                % (options.mode, mean_loss_train, i))
                    logger.info("%s level validation loss is %f at the end of iteration %d"
                                % (options.mode, mean_loss_valid, i))

                    t_current = time() - t_beginning
                    row = [epoch, i, t_current,
                           mean_loss_train, mean_loss_valid]
                    row_df = pd.DataFrame([row], columns=columns)
                    with open(filename, 'a') as f:
                        row_df.to_csv(f, header=False, index=False, sep='\t')
        
         # If no step has been performed, raise Exception
        if step_flag:
            raise Exception(
                'The model has not been updated once in the epoch. The accumulation step may be too large.')

        # If no evaluation has been performed, warn the user
        if evaluation_flag and options.evaluation_steps != 0:
            logger.warning('Your evaluation steps are too big compared to the size of the dataset.'
                           'The model is evaluated only once at the end of the epoch')

        # Always test the results and save them once at the end of the epoch
        logger.debug('Last checkpoint at the end of the epoch %d' % epoch)

        loss_train = test_vae(model, train_loader, options.gpu, criterion)
        mean_loss_train = loss_train / \
            (len(train_loader) * train_loader.batch_size)

        loss_valid = test_vae(model, valid_loader, options.gpu, criterion)
        mean_loss_valid = loss_valid / \
            (len(valid_loader) * valid_loader.batch_size)
        model.train()
        train_loader.dataset.train()

        writer_train.add_scalar('loss', mean_loss_train,
                                i + epoch * len(train_loader))
        writer_valid.add_scalar('loss', mean_loss_valid,
                                i + epoch * len(train_loader))
        logger.info("%s level training loss is %f at the end of iteration %d"
                    % (options.mode, mean_loss_train, i))
        logger.info("%s level validation loss is %f at the end of iteration %d"
                    % (options.mode, mean_loss_valid, i))

        t_current = time() - t_beginning
        row = [epoch, i, t_current, mean_loss_train, mean_loss_valid]
        row_df = pd.DataFrame([row], columns=columns)
        with open(filename, 'a') as f:
            row_df.to_csv(f, header=False, index=False, sep='\t')

        is_best = loss_valid < best_loss_valid
        best_loss_valid = min(best_loss_valid, loss_valid)
        # Always save the model at the end of the epoch and update best model
        save_checkpoint({'model': model.state_dict(),
                         'epoch': epoch,
                         'valid_loss': loss_valid},
                        False, is_best,
                        model_dir)
        # Save optimizer state_dict to be able to reload
        save_checkpoint({'optimizer': optimizer.state_dict(),
                         'epoch': epoch,
                         'name': options.optimizer,
                         },
                        False, False,
                        model_dir,
                        filename='optimizer.pth.tar')

        epoch += 1

    os.remove(os.path.join(model_dir, "optimizer.pth.tar"))
    os.remove(os.path.join(model_dir, "checkpoint.pth.tar"))


def test_vae(model, dataloader, use_cuda, criterion):
    """
    Computes the total loss of a given VAE and dataset wrapped by DataLoader.

    Args:
        model: (VAE) VAE constructed from a CNN with the VanillaVAE class.
        dataloader: (DataLoader) wrapper of the dataset.
        use_cuda: (bool) if True a gpu is used.
        criterion: (loss) function to calculate the loss.

    Returns:
        (float) total loss of the model
    """
    model.eval()
    dataloader.dataset.eval()

    total_loss = 0
    for i, data in enumerate(dataloader, 0):
        if use_cuda:
            inputs = data['image'].cuda()
        else:
            inputs = data['image']

        recon_imgs, mu, log_var = model(inputs)
        loss = criterion(recon_imgs, inputs, mu, log_var)
        total_loss += loss.item()

        del inputs, recon_imgs, mu, log_var, loss

    return total_loss

def visualize_image(model, dataloader, visualization_path, nb_images=1):
    """
    Writes the nifti files of images and their reconstructions by an VAE.

    Args:
        model: (VAE) VAE constructed from a CNN with the VanillaVAE class.
        dataloader: (DataLoader) wrapper of the dataset.
        visualization_path: (str) directory in which the inputs and reconstructions will be stored.
        nb_images: (int) number of images to reconstruct.
    """
    import nibabel as nib
    import numpy as np
    from .iotools import check_and_clean

    check_and_clean(visualization_path)

    dataset = dataloader.dataset
    model.eval()
    dataset.eval()

    for image_index in range(nb_images):
        data = dataset[image_index]
        image = data["image"].unsqueeze(0)
        output = model(image)

        output_np = output.squeeze(0).squeeze(0).cpu().detach().numpy()
        input_np = image.squeeze(0).squeeze(0).cpu().detach().numpy()
        output_nii = nib.Nifti1Image(output_np, np.eye(4))
        input_nii = nib.Nifti1Image(input_np, np.eye(4))
        nib.save(output_nii, os.path.join(
            visualization_path, 'output-%i.nii.gz' % image_index))
        nib.save(input_nii, os.path.join(
            visualization_path, 'input-%i.nii.gz' % image_index))



###########
# Loss
###########
# return reconstruction error + KL divergence losses
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 32768), x.view(-1, 32768), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def get_criterion():
    """Returns the appropriate loss for VAE"""
    return loss_function