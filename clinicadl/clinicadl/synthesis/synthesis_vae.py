import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader
from skimage.measure import compare_psnr, compare_mse, compare_ssim
from sklearn.decomposition import PCA

from clinicadl.synthesis.synthesis_utils import save_eval, save_mean_score, save_pair
from clinicadl.tools.deep_learning.models import create_vae, load_model
from clinicadl.tools.deep_learning.data import (load_data_test,
                                                get_transforms,
                                                return_dataset,
                                                generate_sampler)
from clinicadl.tools.deep_learning.iotools import return_logger, check_and_clean
from clinicadl.tools.deep_learning.iotools import commandline_to_json, write_requirements_version, translate_parameters


def evaluate_vae(params):
    """
    Synthesize output of vae model on the training set and compute evaluation metrics
    """

    main_logger = return_logger(params.verbose, "main process")
    AMap_logger = return_logger(params.verbose, "attention map")
    # check_and_clean(params.output_dir)

    commandline_to_json(params, logger=main_logger)
    # write_requirements_version(params.output_dir)
    params = translate_parameters(params)

    _, all_transforms = get_transforms(
        params.mode,
        minmaxnormalization=params.minmaxnormalization,
    )

    # Load dataset
    test_df = load_data_test(
        params.tsv_path,
        params.diagnoses,
        baseline=params.baseline
    )

    data_test = return_dataset(params.mode, params.input_dir, test_df, params.preprocessing,
                               all_transformations=all_transforms, prepare_dl=params.prepare_dl,
                               multi_cohort=params.multi_cohort, params=params)

    test_loader = DataLoader(
        data_test,
        batch_size=1,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True
    )

    # Load model
    model_dir = os.path.join(params.output_dir, 'fold-%i' % 0, 'models')
    vae = create_vae(params, initial_shape=data_test.size, latent_dim=2, train=False)
    model, _ = load_model(vae, os.path.join(model_dir, "best_loss"),
                          params.gpu, filename='model_best.pth.tar')

    # create output dir
    im_path = os.path.join(params.output_dir, 'output_images')
    if not os.path.exists(im_path):
        os.mkdir(im_path)
    test_path = os.path.join(params.output_dir, 'model_eval')
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    
    # Create evaluation metrics data
    sub_list, ses_list, label_list = [], [], []
    eval_dict = {'mse': [], 'psnr': [], 'ssim': []}

    # loop on data set
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            model.eval()
            if params.gpu:
                imgs = data['image'].cuda()
            else:
                imgs = data['image']

            synthesized_imgs, _, _ = model(imgs)

            x, y = imgs[0].cpu(), synthesized_imgs[0].cpu() 
            sub, ses, label = data['participant_id'][0], data['session_id'][0], data['label'][0]

            path_imgs = os.path.join(
                im_path,
                "{}_{}_{}-io.png".format(sub, ses, label))
            save_pair(x, y, path_imgs)

            x, y = x.numpy(), y.numpy()
            eval_dict['mse'].append(compare_mse(x, y))
            eval_dict['psnr'].append(compare_psnr(x, y))
            eval_dict['ssim'].append(compare_ssim(x, y))
            sub_list.append(sub)
            ses_list.append(ses)
            label_list.append(label)
    
    save_mean_score(eval_dict, test_path)
    save_eval(eval_dict, sub_list, ses_list, label_list, test_path)


def plot_latent_space(params):
    """
    Synthesize output of vae model on the training set and compute evaluation metrics
    """

    main_logger = return_logger(params.verbose, "main process")
    AMap_logger = return_logger(params.verbose, "attention map")
    # check_and_clean(params.output_dir)

    commandline_to_json(params, logger=main_logger)
    # write_requirements_version(params.output_dir)
    params = translate_parameters(params)

    _, all_transforms = get_transforms(
        params.mode,
        minmaxnormalization=params.minmaxnormalization,
    )

    # Load dataset
    data_df = load_data_test(
        params.tsv_path,
        params.diagnoses,
        baseline=params.baseline
    )

    dataset = return_dataset(params.mode, params.input_dir, data_df, params.preprocessing,
                               all_transformations=all_transforms, prepare_dl=params.prepare_dl,
                               multi_cohort=params.multi_cohort, params=params)

    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True
    )

    # Load model
    model_dir = os.path.join(params.output_dir, 'fold-%i' % 0, 'models')
    vae = create_vae(params, initial_shape=dataset.size, latent_dim=2, train=False)
    model, _ = load_model(vae, os.path.join(model_dir, "best_loss"),
                          params.gpu, filename='model_best.pth.tar')

    # create output dir
    test_path = os.path.join(params.output_dir, 'model_eval')
    if not os.path.exists(test_path):
        os.mkdir(test_path)
    
    latent_representations, labels = [], []
    # loop on data set
    with torch.no_grad():
        for _, data in enumerate(data_loader):
            model.eval()
            if params.gpu:
                x = data['image'].cuda()
            else:
                x = data['image']
    
            mu, logvar = model.encode(x)
            z = model.reparameterize_eval(mu, logvar)[0]

            latent_representations.append(z.cpu().detach().numpy())
            labels.append(data['label'][0])

            


