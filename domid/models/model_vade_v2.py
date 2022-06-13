import torch
import torch.nn as nn
import torch.distributions as dist
import torch.nn.functional as F
import os
import itertools
import numpy as np
from sklearn.mixture import GaussianMixture
import tqdm
# from libdg.utils.utils_class import store_args
# from libdg.compos.vae.compos.decoder_concat_vec_reshape_conv_gated_conv \
#     import DecoderConcatLatentFCReshapeConvGatedConv
# from libdg.compos.vae.compos.encoder import LSEncoderDense
# from libdg.models.a_model_classif import AModelClassif
from libdg.utils.utils_classif import logit2preds_vpic, get_label_na
from torch.optim import Adam
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter
from sklearn.manifold import TSNE
import torch.nn as nn
from libdg.utils.utils_classif import logit2preds_vpic, get_label_na
from domid.compos.nn_net import Net_MNIST
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,TensorDataset

def cluster_acc(Y_pred, Y):
    #from sklearn.utils.linear_assignment_ import linear_assignment

    from scipy.optimize import linear_sum_assignment as linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)

    #print('here', Y_pred.size, w)
    return sum([w[ind[0],ind[1]] for counter in ind])*1.0/Y_pred.size#, w[0]

def block(in_c,out_c):
    layers=[
        nn.Linear(in_c,out_c),
        nn.ReLU(True)
    ]
    return layers

class Encoder(nn.Module):
    def __init__(self,z_dim, dim_input=784,filter1=500, filter2 =500, filter3=2000):
        super(Encoder,self).__init__()

        self.encoder=nn.Sequential(
            *block(dim_input,filter1),
            *block(filter1,filter2),
            *block(filter2,filter3),
        )

        self.mu_l=nn.Linear(filter3,z_dim)
        self.log_sigma2_l=nn.Linear(filter3,z_dim)

    def forward(self, x):
        #print('I was in the encoder')
        e=self.encoder(x) #output shape: [batch_size, z_dim] [800x2000]
        mu=self.mu_l(e) #output shape: [batch_size, num_clusters]
        log_sigma2=self.log_sigma2_l(e) #same as mu shape
        #print('shapes in the encder', e.shape, mu.shape, log_sigma2.shape)
        #FIXME:
        #self.encoder.log_sigma2_l.load_state_dict(self.encoder.mu_l.state_dict())
        return mu,log_sigma2

                #epoch_bar.write('L2={:.4f}'.format(L/len(dataloader)))
            #print('DONE pretraining')


class Decoder(nn.Module):
    def __init__(self,z_dim, dim_input=784,filter1 =500, filter2 = 500, filter3 = 2000):
        super(Decoder,self).__init__()

        self.decoder=nn.Sequential(
            *block(z_dim,filter3),
            *block(filter3,filter2),
            *block(filter2,filter1),
            nn.Linear(filter1,dim_input),
            nn.Sigmoid()
        )



    def forward(self, z):
        #print('I was in the decoder')
        """
        Decoder input shape is [batch_size, 10]
        """
        #print('decoder input shape', z.shape)
        x_pro=self.decoder(z)

        return x_pro


class ModelVaDE(nn.Module):

    def __init__(self, y_dim, zd_dim, device, i_c, i_h, i_w, gamma_y, list_str_y):
        # y_dim=y_dim, zd_dim=zd_dim, device=device,  i_c = task.isize.c,
        #                           i_h = task.isize.h, i_w = task.isize.w, gamma_y = args.gamma_y,list_str_y = task.list_str_y
        super(ModelVaDE,self).__init__()
        #self.args = args
        self.z_dim = zd_dim
        self.dim_y = y_dim #???? dim_y or y_dim
        self.i_h = i_h
        self.i_w = i_w
        self.i_c = i_c
        self.gamma_y = gamma_y
        self.list_str_y = list_str_y

        self.infer_y_from_x = Net_MNIST(y_dim, self.i_h) #convolutional neural network
        self.feat_x2concat_y = Net_MNIST(self.dim_feat_x, self.i_h)
        self.infer_domain = Encoder(z_dim=zd_dim, dim_input=self.dim_feat_x+self.y_dim).to(device)
        self.decoder=Decoder(z_dim=zd_dim, dim_input=self.dim_feat_x+self.y_dim).to(device)
        self.pi_=nn.Parameter(torch.FloatTensor(y_dim,).fill_(1)/y_dim,requires_grad=True)
        self.mu_c=nn.Parameter(torch.FloatTensor(y_dim,zd_dim).fill_(0),requires_grad=True)
        self.log_sigma2_c=nn.Parameter(torch.FloatTensor(y_dim,zd_dim).fill_(0),requires_grad=True)

    def cal_logit_y(self, tensor_x):
        """
        calculate the logit for softmax classification

        """
        return self.infer_y_from_x(tensor_x) #gets label


    def pre_train(self,device, dataloader, y_dim, pre_epoch=10):
        """
        AutoEncoder is used to pretrain the encoder-decoder and initialize GMM.
        And then, GMM is applied to encoded inputs to initialize the parameters of {pi_, mu_c , log_sigma_c}

        Input:
        dataloader - dataset with dimensions [   ]
        pre_epoch - epochs to run for pre_training
        y_dim - number of clusters in the latent space to initialize

        Outputs:
        self.encoder
        self.pi
        self.mu_c
        self.log_sigma_c

        """

        if  not os.path.exists('./pretrain_model.pk'):

            Loss=nn.MSELoss()
            opti=Adam(itertools.chain(self.encoder.parameters(),self.decoder.parameters()))
            cuda = True
            print('Pretraining......')
            pre_epoch = np.arange(0, pre_epoch)
            for ii in pre_epoch:
                L=0
                for x,y in dataloader:
                    if cuda:
                        x=x.cuda()
                        y = y.to(device)
                    z,_=self.encoder(x) #input: x, output: mu,log_sigma2
                    x_=self.decoder(z) #input: z, output: x_pro
                    loss=Loss(x,x_)
                    L+=loss.detach().cpu().numpy()

                    opti.zero_grad()
                    loss.backward()
                    opti.step()
            self.encoder.log_sigma2_l.load_state_dict(self.encoder.mu_l.state_dict())
            Z = [] #latents space representation
            Y = []
            with torch.no_grad():
                for x, y in dataloader:
                    if torch.cuda.is_available():
                        x = x.cuda()

                    z1, z2 = self.encoder(x)
                    assert F.mse_loss(z1, z2) == 0
                    Z.append(z1)
                    Y.append(y)

            Z = torch.cat(Z, 0).detach().cpu().numpy()
            Y = torch.cat(Y, 0).detach().numpy()

            gmm = GaussianMixture(n_components= y_dim, covariance_type='diag')

            pre = gmm.fit_predict(Z) #clusterization in latent space
            print('Pretrained Cluster Acc={:.4f}%'.format(cluster_acc(pre, Y)[0] * 100))

            self.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
            self.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
            self.log_sigma2_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())


    def predict(self, x): #nClusters = ydim
        """
        Inputs: one sample shape []
        Outputs:

        """

        print('i was here')
        z_mu, z_sigma2_log = self.encoder(x)
        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        pi = self.pi_
        log_sigma2_c = self.log_sigma2_c
        mu_c = self.mu_c
        yita_c = torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(nClusters,z,mu_c,log_sigma2_c)) #shape [batch_size, 10]
        #print('yitac shape', yita_c.shape)
        yita=yita_c.detach().cpu().numpy() #shape: [batch_size, 10]
        #print('predict function, yita', yita_c.shape, yita.shape)
        #print('predict function output', np.argmax(yita,axis=1).shape)
        return np.argmax(yita,axis=1) #number 0-9, shape [batch_size,]


    def ELBO_Loss(self,zd_dim, x,L=1):
        """
        Calculates loss between x and x_pro (input to the encoder and output from the decoder)
        Inputs:
        x - dataloader dataset with the input shape of [batch_size, 28*28]
        L - what is L and why are we iterating through it?????? (Monte Carlon sampling number, optimize)
        Outputs:
        Loss - calculated loss between
        """
        #print('ELBO LOSS input SHAPE', x.shape)
        det=1e-10
        L_rec=0
        z_mu, z_sigma2_log = self.encoder(x)
        #print('L', L)
        for l in range(L): #not quite sure what the loop is for
            z=torch.randn_like(z_mu)*torch.exp(z_sigma2_log/2)+z_mu
            x_pro=self.decoder(z)
            L_rec+=F.binary_cross_entropy(x_pro,x) #why binary cross entropy?

        L_rec/=L
        Loss=L_rec*x.size(1)

        pi=self.pi_
        log_sigma2_c=self.log_sigma2_c
        mu_c=self.mu_c

        z = torch.randn_like(z_mu) * torch.exp(z_sigma2_log / 2) + z_mu
        yita_c=torch.exp(torch.log(pi.unsqueeze(0))+self.gaussian_pdfs_log(zd_dim, z,mu_c,log_sigma2_c))+det
        yita_c=yita_c/(yita_c.sum(1).view(-1,1))#batch_size*Clusters
        Loss+=0.5*torch.mean(torch.sum(yita_c*torch.sum(log_sigma2_c.unsqueeze(0)+
                                                torch.exp(z_sigma2_log.unsqueeze(1)-log_sigma2_c.unsqueeze(0))+
                                                (z_mu.unsqueeze(1)-mu_c.unsqueeze(0)).pow(2)/torch.exp(log_sigma2_c.unsqueeze(0)),2),1))

        Loss-=torch.mean(torch.sum(yita_c*torch.log(pi.unsqueeze(0)/(yita_c)),1))+0.5*torch.mean(torch.sum(1+z_sigma2_log,1))


        return Loss


    def gaussian_pdfs_log(self,y_dim, x,mus,log_sigma2s):
        """
        helper function that used to calculate ELBO loss
        """
        G=[]

        for c in range(y_dim):
            G.append(self.gaussian_pdf_log(x,mus[c:c+1,:],log_sigma2s[c:c+1,:]).view(-1,1))
            #print(len(G))
        return torch.cat(G,1)
    @staticmethod
    def gaussian_pdf_log(x,mu,log_sigma2):
        """
        subhelper function just one gausian pdf log calculation, used as a basis for gaussia_pdfs_logs function
        """
        return -0.5*(torch.sum(np.log(np.pi*2)+log_sigma2+(x-mu).pow(2)/torch.exp(log_sigma2),1))

    def infer_y_vpicn(self, tensor_x):

        with torch.no_grad():
            logit_y = self.infer_y_from_x(tensor_x)
        vec_one_hot, prob, ind, confidence = logit2preds_vpic(logit_y)  #
        # na_class = get_label_na(ind, self.list_str_y)
        vec_one_hot2 = predict(logit_y)

        return vec_one_hot2, prob, ind, confidence, na_class


    def infer_d_v(self, tensor_x):
        with torch.no_grad():
            y_hat_logit = self.infer_y_from_x(tensor_x)
            feat_x = self.feat_x2concat_y(tensor_x)
            feat_y_x = torch.cat((y_hat_logit, feat_x), dim=1)
            q_zd, zd_q = self.infer_domain(feat_y_x)
        vec_one_hot2, *_ = predict(q_zd.mean)
        return vec_one_hot2


    def forward(self, tensor_x, vec_y, vec_d=None):
        # print('foward', tensor_x.shape)
        y_hat_logit = self.predict(tensor_x)

        feat_x = self.feat_x2concat_y(tensor_x)
        feat_y_x = torch.cat((y_hat_logit, feat_x), dim=1)
        q_zd, zd_q = self.infer_domain(feat_y_x)

        return q_zd, zd_q, y_hat_logit


    def cal_loss(self, x, y, L, d=None):
        loss = vade.ELBO_Loss(y_dim, x)
        L += loss.detach().cpu().numpy()
        loss = L
        return loss.mean()



def test_fun(y_dim, zd_dim, device):

    # batch_size = 800
    # data_dir = './data/mnist'
    #
    # train=MNIST(root=data_dir,train=True,download=True)
    # test=MNIST(root=data_dir,train=False,download=True)
    #
    # X=torch.cat([train.data.float().view(-1,784)/255.,test.data.float().view(-1,784)/255.],0)
    # Y=torch.cat([train.targets,test.targets],0)
    #
    # dataset=dict()
    # dataset['X']=X
    # dataset['Y']=Y
    #
    # DL=DataLoader(TensorDataset(X,Y),batch_size=batch_size,shuffle=True,num_workers=1)

    vade = ModelVaDE(y_dim=y_dim, zd_dim=zd_dim, device=device)
    print('vade model', vade)

    # if torch.cuda.is_available():
    #     vade = vade.to(device)
    #device, dataloader, nClusters, pre_epoch=10




    # FIXME: add the summary of the outputs from pretrained model
    # vade.module.pre_train(DL,pre_epoch=5)
    print('VADE PAREMETERS', vade.parameters())
    opti = Adam(vade.parameters(), lr=2e-3)
    lr_s = StepLR(opti, step_size=10, gamma=0.95)

    epoch_bar = np.arange(0,10)
    for epoch in epoch_bar:

        lr_s.step()
        #FIXME: lr_s is outdated? getting warning from this line
        L = 0
        for x, _ in DL:
            if torch.cuda.is_available():
                x = x.cuda()

            loss = vade.ELBO_Loss(y_dim, x) #nClusters or ydim?

            opti.zero_grad()
            loss.backward()
            opti.step()

            L += loss.detach().cpu().numpy()
        #print('done elbo training')

        pre = []
        tru = []

        with torch.no_grad():
            for x, y in DL:
                if torch.cuda.is_available():
                    x = x.cuda()

                tru.append(y.numpy())
                pre.append(vade.predict(y_dim, x))

        tru = np.concatenate(tru, 0)
        pre = np.concatenate(pre, 0)
        print('ELBO loss',L/len(DL),epoch)
        #print(pre, tru)
        print('ELBO acc',cluster_acc(pre,tru)[0]*100,epoch)
        print('lr',lr_s.get_lr()[0],epoch)
