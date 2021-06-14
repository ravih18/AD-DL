import numpy as np
import pandas as pd
import torch
import os
from torch.utils.data import DataLoader
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity
from sklearn.decomposition import PCA

from clinicadl.synthesis.synthesis_utils import (save_eval,
                                                 save_mean_score,
                                                 save_pair,
                                                 save_latent_space,
                                                 save_io_diff)
from clinicadl.tools.deep_learning.models import create_vae, load_model
from clinicadl.tools.deep_learning.data import (load_data_test,
                                                get_transforms,
                                                return_dataset,
                                                generate_sampler)
from clinicadl.tools.deep_learning.iotools import return_logger, check_and_clean
from clinicadl.tools.deep_learning.iotools import commandline_to_json, write_requirements_version, translate_parameters
from clinicadl.utils.model_utils import select_device


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

    # Select the working device
    device = select_device(params.gpu)

    # Load model
    model_dir = os.path.join(params.output_dir, 'fold-%i' % 0, 'models')
    vae = create_vae(params, initial_shape=data_test.size, latent_dim=1, train=False)
    model, _ = load_model(vae, os.path.join(model_dir, "best_loss"),
                          device, filename='model_best.pth.tar')

    # create output dir
    im_path = os.path.join(params.output_dir, 'output_images')
    if not os.path.exists(im_path):
        os.mkdir(im_path)
    diff_path = os.path.join(params.output_dir, 'difference_maps')
    if not os.path.exists(diff_path):
        os.mkdir(diff_path)
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
            imgs = data['image'].to(device)

            synthesized_imgs, _, _ = model(imgs)

            x, y = imgs[0].cpu(), synthesized_imgs[0].cpu() 
            sub, ses, label = data['participant_id'][0], data['session_id'][0], data['label'][0]

            path_imgs = os.path.join(
                im_path,
                "{}_{}_{}-io.png".format(sub, ses, label))
            save_pair(x, y, path_imgs)
            path_diff = os.path.join(
                diff_path,
                "{}_{}_{}-difference_map.png".format(sub, ses, label))
            save_io_diff(x, y, path_diff)
            

            x, y = x[0].numpy(), y[0].numpy()
            eval_dict['mse'].append(mean_squared_error(x, y))
            eval_dict['psnr'].append(peak_signal_noise_ratio(x, y))
            eval_dict['ssim'].append(structural_similarity(x, y))
            sub_list.append(sub)
            ses_list.append(ses)
            label_list.append(int(label))
    
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

    # Select the working device
    device = select_device(params.gpu)

    # Load model
    model_dir = os.path.join(params.output_dir, 'fold-%i' % 0, 'models')
    vae = create_vae(params, initial_shape=dataset.size, latent_dim=1, train=False)
    model, _ = load_model(vae, os.path.join(model_dir, "best_loss"),
                          device, filename='model_best.pth.tar')

    # create output dir
    test_path = os.path.join(params.output_dir, 'model_eval')
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    # create PCA
    pca =PCA(n_components=2)

    latent_representations, sub_list, ses_list, label_list = [], [], [], []
    # loop on data set
    with torch.no_grad():
        for _, data in enumerate(data_loader):
            model.eval()
            x = data['image'].to(device)

            mu, logvar = model.encode(x)
            z = model.reparameterize_eval(mu, logvar)[0]

            latent_representations.append(z.cpu().detach().numpy()[0].flatten())

            sub, ses, label = data['participant_id'][0], data['session_id'][0], data['label'][0]
            label_list.append(int(label))
            sub_list.append(sub)
            ses_list.append(ses)

    feat_cols = ['feature'+str(i) for i in range(len(latent_representations[0]))]
    df_latent = pd.DataFrame(latent_representations,columns=feat_cols)
    df_latent['label'] = label_list
    df_latent['subject'] = sub_list
    df_latent['session'] = ses_list
    print('Size of the dataframe: {}'.format(df_latent.shape))
    latent_tsv_path = os.path.join(test_path, "latent_representation.tsv")
    df_latent.to_csv(latent_tsv_path, sep='\t', index=False)

    #pca.fit(latent_representations)
    # principal_components = pca.fit_transform(latent_representations)
    # print(principal_components.shape)

    # np.save("latent_representation")

    # img_path = os.path.join(test_path, "latent_space_pca.png")
    # save_latent_space(principal_components,labels, img_path)
