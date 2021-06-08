# coding: utf8

import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from PIL import Image

from clinicadl.visualize.gradcam.gradcam_utils import GradCAM, save_cam
from clinicadl.tools.deep_learning.models import init_model, load_model
from clinicadl.tools.deep_learning.data import (load_data_test,
                                                get_transforms,
                                                return_dataset,
                                                generate_sampler)
from clinicadl.tools.deep_learning.iotools import return_logger, check_and_clean
from clinicadl.tools.deep_learning.iotools import commandline_to_json, write_requirements_version, translate_parameters
from clinicadl.utils.model_utils import select_device


def attention_map(params):
    """
    Create attention map for a model (VAE) on a testing set to visualize
    anomalies
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
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True
    )

    # device selection
    device = select_device(params.gpu)

    # Load model
    model_dir = os.path.join(params.output_dir, 'fold-%i' % 0, 'models')
    vae = init_model(params, initial_shape=data_test.size, architecture="vae")
    model, _ = load_model(vae, os.path.join(model_dir, "best_loss"),
                          device, filename='model_best.pth.tar')
    # create GradCAM
    mu_avg, logvar_avg = 0, 1
    gcam = GradCAM(model, target_layer='encoder.sequential.2.layer.0')

    # create output dir
    im_path = os.path.join(params.output_dir, 'attention_maps')
    if not os.path.exists(im_path):
        os.mkdir(im_path)

    # loop on data set
    for _, data in enumerate(test_loader):
        model.eval()
        imgs = data['image'].to(device)
        _, mu, logvar = gcam.forward(imgs)

        model.zero_grad()
        gcam.backward(mu, logvar, mu_avg, logvar_avg)
        gcam_map = gcam.generate()

        ## Visualize and save attention maps  ##
        # make a 3 channel image for colors
        imgs = imgs.repeat(1, 3, 1, 1)
        for i in range(imgs.size(0)):
            # multiply by 255 to change from grey scale to colors
            raw_image = imgs[i] * 255.0
            ndarr = raw_image.permute(1, 2, 0).cpu().byte().numpy()
            im = Image.fromarray(ndarr.astype(np.uint8))
            # save original image
            im.save(os.path.join(
                im_path,
                "{}_{}_{}-origin.png".format(
                    data['participant_id'][0],
                    data['session_id'][0],
                    data['label'][0])
                )
            )
            file_path = os.path.join(
                im_path,
                "{}_{}—{}-attmap.png".format(
                    data['participant_id'][0],
                    data['session_id'][0],
                    data['label'][0])
                )
            print(file_path)
            r_im = np.asarray(im)
            # save grad cam
            save_cam(r_im, file_path, gcam_map[i].squeeze().cpu().data.numpy())
