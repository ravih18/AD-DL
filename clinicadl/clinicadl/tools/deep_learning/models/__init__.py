from .autoencoder import AutoEncoder, initialize_other_autoencoder, transfer_learning
from .iotools import load_model, load_optimizer, save_checkpoint
from .image_level import Conv5_FC3, Conv5_FC3_mni, Conv6_FC3, VConv5_FC3
from .patch_level import Conv4_FC3
from .slice_level import resnet18, ConvNet
from .random import RandomArchitecture
from clinicadl.utils.model_utils import select_device


def create_model(options, initial_shape):
    """
    Creates model object from the model_name.

    :param options: (Namespace) arguments needed to create the model.
    :param initial_shape: (array-like) shape of the input data.
    :return: (Module) the model object
    """

    if not hasattr(options, "model"):
        model = RandomArchitecture(options.convolutions, options.n_fcblocks, initial_shape,
                                   options.dropout, options.network_normalization, n_classes=2)
    else:
        try:
            model = eval(options.model)(dropout=options.dropout)
        except NameError:
            raise NotImplementedError(
                'The model wanted %s has not been implemented.' % options.model)

    if options.gpu:
        model.cuda()
    else:
        model.cpu()

    return model


def create_autoencoder(options, initial_shape, difference=0):
    """
    Creates an autoencoder object from the model_name.

    :param options: (Namespace) arguments needed to create the model.
    :param initial_shape: (array-like) shape of the input data.    :param difference: (int) difference of depth between the pretrained encoder and the new one.
    :return: (Module) the model object
    """
    from .autoencoder import AutoEncoder, initialize_other_autoencoder
    from os import path

    model = create_model(options, initial_shape)
    decoder = AutoEncoder(model)

    if options.transfer_learning_path is not None:
        if path.splitext(options.transfer_learning_path) != ".pth.tar":
            raise ValueError("The full path to the model must be given (filename included).")
        decoder = initialize_other_autoencoder(decoder, options.transfer_learning_path, difference)

    return decoder

def create_vae(options, initial_shape, latent_dim, train=False):
    """
    Creates a variational autoencoder object from the model_name.

    :param options: (Namespace) arguments needed to create the model.
    :param initial_shape: (array-like) shape of the input data.    :param difference: (int) difference of depth between the pretrained encoder and the new one.
    :return: (Module) the model object
    """
    from .vae import VanillaVAE
    from os import path

    if latent_dim==1:
        feature_size=1024
        latent_size=64
    elif latent_dim==2:
        feature_size=16
        latent_size=1

    vae = VanillaVAE(
        input_shape=initial_shape,
        latent_dim=latent_dim,
        feature_size=feature_size,
        latent_size=latent_size,
        n_conv=4,
        io_layer_channel=32,
        train = train
    )
    model.to(select_device(options.gpu))
    return vae

def init_model(options, initial_shape, architecture="cnn"):

    possible_architecture = {"cnn", "autoencoder", "vae"}
    if architecture not in possible_architecture:
        raise ValueError(f"architecture must be one of {possible_architecture}")
    
    if architecture=="cnn":
        model = create_model(options, initial_shape)
    elif architecture=="autoencoder":
        model = AutoEncoder(model)
    elif architecture=="vae":
        model = create_vae(options, initial_shape, latent_dim=2, train=False)

    return model
