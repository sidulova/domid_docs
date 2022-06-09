import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F

from libdg.utils.utils_class import store_args
from libdg.compos.vae.compos.decoder_concat_vec_reshape_conv_gated_conv \
    import DecoderConcatLatentFCReshapeConvGatedConv
from libdg.compos.vae.compos.encoder import LSEncoderDense
from libdg.models.a_model_classif import AModelClassif
from libdg.utils.utils_classif import logit2preds_vpic, get_label_na

from domid.compos.nn_net import Net_MNIST


class ModelVaDE(torch.nn.Module):
    """
    TODO: implement the actual VaDE model...
    """

    @store_args
    def __init__(self, device):
        """
        :param zd_dim: dimension of latent variable $z_d$ dimension
        """
        super(ModelVaDE, self).__init__()

    def forward(self, tensor_x):
        q_zd = dist.Normal(torch.randn(5,), torch.rand(5,))
        zd_q = q_zd.rsample()  # Reparameterization trick
        return q_zd, zd_q


def test_fun():
    model = ModelVaDE(device=torch.device("cuda"))
    device = torch.device("cuda")
    x = torch.rand(2, 3, 28, 28)
    model(x)
