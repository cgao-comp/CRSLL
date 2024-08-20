import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tqdm import tqdm
import math
from GCN_EN import self_loop_attention_GCN
import torch
import math
class feature_Model(nn.Module):

    def __init__(self, feature_hidden_dim=64, profiles_dim=7, pro_dyn_feature_dim=7, head=2, self_head=4, multimodal_dim=512):
        super(feature_Model, self).__init__()
        self.feature_hidden_dim = feature_hidden_dim

        self.MLP = nn.Linear(pro_dyn_feature_dim, feature_hidden_dim)
        self.norm1 = nn.LayerNorm(feature_hidden_dim)

        self.high_focus1 = nn.Parameter(torch.Tensor(pro_dyn_feature_dim, feature_hidden_dim // 2))
        self.high_bias1 = nn.Parameter(torch.Tensor(feature_hidden_dim // 2))
        self.high_focus2 = nn.Parameter(torch.Tensor(feature_hidden_dim // 2, feature_hidden_dim))
        self.high_bias2 = nn.Parameter(torch.Tensor(feature_hidden_dim))
        self.high_norm1 = nn.LayerNorm(feature_hidden_dim // 2)
        self.high_norm2 = nn.LayerNorm(feature_hidden_dim)

        self.low_focus1 = nn.Parameter(torch.Tensor(pro_dyn_feature_dim, feature_hidden_dim // 2))
        self.low_bias1 = nn.Parameter(torch.Tensor(feature_hidden_dim // 2))
        self.low_focus2 = nn.Parameter(torch.Tensor(feature_hidden_dim // 2, feature_hidden_dim))
        self.low_bias2 = nn.Parameter(torch.Tensor(feature_hidden_dim))
        self.low_norm1 = nn.LayerNorm(feature_hidden_dim // 2)
        self.low_norm2 = nn.LayerNorm(feature_hidden_dim)

        self.crossAtt = nn.MultiheadAttention(feature_hidden_dim, num_heads=head)
        self.cross_norm = nn.LayerNorm(feature_hidden_dim)

        self.profiles_decoder = nn.Linear(profiles_dim, multimodal_dim)
        self.norm_profiles = nn.LayerNorm(multimodal_dim)
        self.loop_att_gcn = self_loop_attention_GCN(multimodal_dim, self_head)
        self.de_Conv = nn.Parameter(torch.Tensor(2 * feature_hidden_dim, multimodal_dim))
        self.de_bias = nn.Parameter(torch.Tensor(multimodal_dim))
        self.de_norm1 = nn.LayerNorm(multimodal_dim)
        self.de_MLP = nn.Linear(2 * multimodal_dim, multimodal_dim)

        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.high_focus1)
        nn.init.xavier_uniform_(self.low_focus1)
        nn.init.xavier_uniform_(self.high_focus2)
        nn.init.xavier_uniform_(self.low_focus2)
        nn.init.constant_(self.high_bias1, 0)
        nn.init.constant_(self.low_bias1, 0)
        nn.init.constant_(self.high_bias2, 0)
        nn.init.constant_(self.low_bias2, 0)
        nn.init.xavier_uniform_(self.de_Conv)
        nn.init.constant_(self.de_bias, 0)
    def forward(self, graph_topo, pro_dyn_feature, profiles):
        assert pro_dyn_feature.shape[-1] == 7, 'error!'
        assert profiles.shape[-1] == 7, 'error!'

        graph_topo = graph_topo.squeeze(0)
        laplacian = torch.diag(torch.sum(graph_topo, dim=1)) - graph_topo
        high_freq = torch.matmul(laplacian, pro_dyn_feature)
        high_freq = torch.matmul(high_freq, self.high_focus1) + self.high_bias1
        high_freq = self.high_norm1(high_freq)
        high_freq = F.relu(high_freq)
        high_freq = torch.matmul(high_freq, self.high_focus2) + self.high_bias2
        high_freq = self.high_norm2(high_freq)
        high_freq = F.relu(high_freq)  

        D = torch.diag(torch.pow(graph_topo.sum(dim=1), -0.5))
        D[torch.isinf(D)] = 0
        A_hat = graph_topo + torch.eye(graph_topo.size(0))
        DAD = torch.mm(torch.mm(D, A_hat), D)
        low_freq = torch.matmul(DAD, pro_dyn_feature)
        low_freq = torch.matmul(low_freq, self.low_focus1) + self.low_bias1
        low_freq = self.low_norm1(low_freq)
        low_freq = F.relu(low_freq)
        low_freq = torch.matmul(low_freq, self.low_focus2) + self.low_bias2
        low_freq = self.low_norm2(low_freq)
        low_freq = F.relu(low_freq)

        high_freq = high_freq.unsqueeze(0)
        low_freq = low_freq.unsqueeze(0)
        cross_att_output1, _ = self.crossAtt(high_freq, low_freq, low_freq)
        cross_att_output1 = cross_att_output1.squeeze(0)
        cross_att_output1 = self.cross_norm(cross_att_output1) 
        cross_att_output2, _ = self.crossAtt(low_freq, high_freq, high_freq)
        cross_att_output2 = cross_att_output2.squeeze(0)
        cross_att_output2 = self.cross_norm(cross_att_output2) 
        encoder_input = torch.cat((cross_att_output1, cross_att_output2), dim=-1) 
        encoder_DAD = torch.matmul(DAD, encoder_input) 
        encoder_output = torch.matmul(encoder_DAD, self.de_Conv) + self.de_bias  
        encoder_output = self.de_norm1(encoder_output)  

        encoder_output = encoder_output + pro_dyn_feature[:, -2].unsqueeze(1)   

        profiles_emb = self.norm_profiles(self.profiles_decoder(profiles))
        enhanced_profiles = self.loop_att_gcn(graph_topo, profiles_emb) 
        decoder_input = torch.cat((encoder_output, enhanced_profiles), dim=-1)  
        decoder_output = self.de_MLP(decoder_input) + pro_dyn_feature[:, -2].unsqueeze(1)  
        return decoder_output
class FeatureMaskingModulev2(nn.Module):
    def __init__(self, num_features, temperature=0.5, reg_weights=[0, 0]):
        super(FeatureMaskingModulev2, self).__init__()
        self.temperature = temperature
        self.reg_weights = reg_weights  

        self.decision_network = nn.Linear(num_features, 2 * num_features)
    def forward(self, x):
        num_nodes, num_features = x.shape
        logits = self.decision_network(x)
        logits = logits.view(num_nodes, num_features, 2) 

        decisions = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)

        masked_x = x * decisions[..., 0] * 0.01 

        keep_x = x * decisions[..., 1]  

        reg_loss = (decisions * torch.tensor(self.reg_weights).to(decisions.device)).mean()

        final_x = masked_x + keep_x

        return final_x, reg_loss
class VAE(nn.Module):
    def __init__(self, input_channels, latent_dim, out_channels):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_channels, int(input_channels/2)),

            nn.ReLU(),
            nn.Linear( int(input_channels/2), latent_dim * 2)  
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, int(out_channels/2)),

            nn.ReLU(),
            nn.Linear(int(out_channels/2), out_channels)
        )
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, -1)
        z = self.reparameterize(mu, logvar)
        return z, self.decoder(z), mu, logvar
    def vae_loss_function(self, recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x, reduction='mean')

        KLD = -1 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE + KLD
    def vae_kl_loss_only(self, mu, logvar):
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return KLD