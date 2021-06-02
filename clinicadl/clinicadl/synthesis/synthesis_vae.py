import numpy as np
import torch
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# from clinicadl.synthesis.synthesis_utils import 
from clinicadl.tools.deep_learning.models import init_model, load_model
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
        batch_size=params.batch_size,
        shuffle=False,
        num_workers=params.num_workers,
        pin_memory=True
    )

    # Load model
    model_dir = os.path.join(params.output_dir, 'fold-%i' % 0, 'models')
    vae = init_model(params, initial_shape=data_test.size, architecture="vae")
    model, _ = load_model(vae, os.path.join(model_dir, "best_loss"),
                          params.gpu, filename='model_best.pth.tar')

    # create output dir
    im_path = os.path.join(params.output_dir, 'output_images')
    if not os.path.exists(im_path):
        os.mkdir(im_path)

    # loop on data set
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            model.eval()
            
            imgs = data['image']

            synthesized_imgs, _, _ = model(imgs)

            for i in range(imgs.size(0)):
                plt.imshow(synthesized_imgs[i].permute(1, 2, 0)[:, :, 0])
                plt.savefig(os.path.join(
                    im_path,
                    "{}_{}_{}-synthesized.png".format(
                        data['participant_id'][0],
                        data['session_id'][0],
                        data['label'][0])
                    )
                )