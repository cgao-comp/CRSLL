import sys
import args
import torch.nn as nn
import numpy as np
import setproctitle
import torch
import random
import networkx as nx
import heapq
import pickle
import time
from tqdm import tqdm
import math
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler
from sklearn.metrics import f1_score
from a2026_dataReader import DataReader_snapshot
from a2026_dataLoader import GraphData
from torch.utils.data import DataLoader
from a2026_model import feature_Model
from a2026_model import VAE
from a2026_model import FeatureMaskingModulev2
torch.cuda.set_device(0)
def collate_batch(batch): 

    Chanels = batch[0][-3].shape[1] 
    N_nodes_max = batch[0][-3].shape[0]
    x = batch[0][-3]
    A = batch[0][0]
    P = batch[0][2]
    labels = batch[0][1]
    N_nodes = torch.from_numpy(np.array(N_nodes_max)).long()
    influence = batch[0][4]
    embedding = batch[0][-1]
    return [A, labels, P, x, N_nodes, influence, embedding] 
def train(train_loader, args):
    args.device = 'cuda'
    start = time.time()
    train_loss, n_samples = 0, 0
    exe_time = 0
    F_score = 0
    for batch_idx, data in enumerate(train_loader):
        opt.zero_grad()
        feature_pro = feature_aggregation(data[0], data[3][:, 7:], data[3][:, 0:7])  
        frequent_indices_neg = random.sample(range(int(100*0.9)-1), 5)
        range_without_batch_idx = list(range(int(200*0.9)-1)) 
        if batch_idx in range_without_batch_idx: 
            range_without_batch_idx.remove(batch_idx) 
        frequent_indices_pos = random.sample(range_without_batch_idx, 5)  
        sampler_neg = SubsetRandomSampler(frequent_indices_neg)
        sampler_pos = SubsetRandomSampler(frequent_indices_pos)
        subset_neg = DataLoader(gdata_POS1_NEG2[1], batch_size=1, sampler=sampler_neg, collate_fn=collate_batch)
        subset_pos = DataLoader(gdata_POS1_NEG2[0], batch_size=1, sampler=sampler_pos, collate_fn=collate_batch)
        similarity_NEG_POS = []
        VAE_loss_total = 0
        z1_central, decoder_z1, mu, logvar = VAE_POS(data[-1])  
        vae_loss = VAE_POS.vae_loss_function(recon_x=decoder_z1, x=data[-1], mu=mu, logvar=logvar)
        central_z1_avg = torch.mean(z1_central, dim=0)   
        VAE_loss_total += vae_loss
        for neg_index, neg in enumerate(subset_neg):
            z1, decoder_z1, mu, logvar = VAE_NEG(neg[-1])
            vae_loss = VAE_NEG.vae_loss_function(recon_x=decoder_z1, x=neg[-1], mu=mu, logvar=logvar)
            z1_neg_avg = torch.mean(z1, dim=0)  
            VAE_loss_total += vae_loss
            sim = torch.matmul(z1_neg_avg, central_z1_avg.T) / args.scale
            similarity_NEG_POS.append(sim)
        for pos_index, pos in enumerate(subset_pos):
            z1, decoder_z1, mu, logvar = VAE_POS(pos[-1])
            vae_loss = VAE_POS.vae_loss_function(recon_x=decoder_z1, x=pos[-1], mu=mu, logvar=logvar)
            z1_pos_avg = torch.mean(z1, dim=0)  
            VAE_loss_total += vae_loss
            sim = torch.matmul(z1_pos_avg, central_z1_avg.T) / args.scale
            similarity_NEG_POS.append(sim)
        VAE_loss_total = VAE_loss_total/11
        assert len(similarity_NEG_POS[5:]) == 5 and len(similarity_NEG_POS[0:5]) == 5, 'error!'
        max_sim = max(similarity_NEG_POS)  
        exp_sum_neg = sum(torch.exp(sim - max_sim) for sim in similarity_NEG_POS[0:5])
        exp_sum_pos = sum(torch.exp(sim - max_sim) for sim in similarity_NEG_POS[5:])
        total_exp_sum = exp_sum_neg + exp_sum_pos
        N = data[0].shape[0]
        GCL_loss = -torch.log(exp_sum_pos / total_exp_sum)
        feature_pro_m, reg_loss1 = F_MASK1(feature_pro)
        feature_pro_z, reg_loss2 = F_MASK2(z1_central)
        cross_att_output1, _ = crossAtt(feature_pro_m, feature_pro_z, feature_pro_z)
        cross_att_output1 = cross_att_output1.squeeze(0)
        cross_att_output2, _ = crossAtt(feature_pro_z, feature_pro_m, feature_pro_m)
        cross_att_output2 = cross_att_output2.squeeze(0)
        cross_out = torch.cat((cross_att_output1, cross_att_output2), dim=-1) 
        pred = decoder_(cross_out)
        pred = pred.clone()
        pred[:, 0] = pred[:, 0] + 1 * (1 - data[2])
        pred[:, 1] = pred[:, 1] + 1 * (data[2])
        pred = F.softmax(pred, dim=1)  
        ground_truth = data[1]  
        V = ground_truth.shape[0]
        weight_0 = 1 / V  
        weight_1 = (V - 1) / V  
        weights = torch.tensor([weight_0, weight_1])
        criterion = nn.CrossEntropyLoss(weight=weights)
        weighted_loss = criterion(pred, ground_truth.long())  
        loss_total = weighted_loss + 0.1 * VAE_loss_total + 0.1 * GCL_loss
        loss_total.backward()
        opt.step()
        index_data1 = (data[1] == 1).nonzero(as_tuple=True)[0]
        sorted_indices = pred[:, 1].sort(descending=True).indices
        ranks = [(sorted_indices == idx).nonzero(as_tuple=True)[0].item() for idx in index_data1][0] + 1
        f1 = 2 * 1 * (1/ranks)/(1 + 1/ranks)
        time_iter = time.time() - start
        train_loss += weighted_loss.item() * 1
        n_samples += 1
        F_score += f1
        exe_time += 1
        if batch_idx % 20 == 0 or batch_idx == len(train_loader) - 1:  
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} (avg: {:.6f})\tF-score:{:.4f} \tsecond/iteration: {:.4f}'.format(
                epoch + 1, n_samples, len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader), loss_total.item(), loss_total / n_samples, F_score / exe_time,
                time_iter / (batch_idx + 1)))
def test(train_loader, args):
    args.device = 'cuda'
    start = time.time()
    train_loss, n_samples = 0, 0
    exe_time = 0
    F_score = 0
    feature_aggregation.eval()
    VAE_NEG.eval()
    F_MASK1.eval()
    F_MASK2.eval()
    crossAtt.eval()
    decoder_.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):  
            opt.zero_grad()
            feature_pro = feature_aggregation(data[0], data[3][:, 7:], data[3][:, 0:7])  
            frequent_indices_neg = random.sample(range(int(100*0.9)-1), 5)
            range_without_batch_idx = list(range(int(200*0.9)-1))  
            if batch_idx in range_without_batch_idx: 
                range_without_batch_idx.remove(batch_idx) 
            frequent_indices_pos = random.sample(range_without_batch_idx, 5) 
            sampler_neg = SubsetRandomSampler(frequent_indices_neg)
            sampler_pos = SubsetRandomSampler(frequent_indices_pos)
            subset_neg = DataLoader(gdata_POS1_NEG2[1], batch_size=1, sampler=sampler_neg, collate_fn=collate_batch)
            subset_pos = DataLoader(gdata_POS1_NEG2[0], batch_size=1, sampler=sampler_pos, collate_fn=collate_batch)
            similarity_NEG_POS = []
            VAE_loss_total = 0
            z1_central, decoder_z1, mu, logvar = VAE_POS(data[-1]) 
            vae_loss = VAE_POS.vae_loss_function(recon_x=decoder_z1, x=data[-1], mu=mu, logvar=logvar)
            central_z1_avg = torch.mean(z1_central, dim=0)  
            VAE_loss_total += vae_loss
            for neg_index, neg in enumerate(subset_neg):
                z1, decoder_z1, mu, logvar = VAE_NEG(neg[-1])
                vae_loss = VAE_NEG.vae_loss_function(recon_x=decoder_z1, x=neg[-1], mu=mu, logvar=logvar)
                z1_neg_avg = torch.mean(z1, dim=0)  
                VAE_loss_total += vae_loss
                sim = torch.matmul(z1_neg_avg, central_z1_avg.T) / args.scale
                similarity_NEG_POS.append(sim)
            for pos_index, pos in enumerate(subset_pos):
                z1, decoder_z1, mu, logvar = VAE_POS(pos[-1])
                vae_loss = VAE_POS.vae_loss_function(recon_x=decoder_z1, x=pos[-1], mu=mu, logvar=logvar)
                z1_pos_avg = torch.mean(z1, dim=0)  
                VAE_loss_total += vae_loss
                sim = torch.matmul(z1_pos_avg, central_z1_avg.T) / args.scale
                similarity_NEG_POS.append(sim)
            VAE_loss_total = VAE_loss_total/11
            assert len(similarity_NEG_POS[5:]) == 5 and len(similarity_NEG_POS[0:5]) == 5, 'error!'
            max_sim = max(similarity_NEG_POS)  
            exp_sum_neg = sum(torch.exp(sim - max_sim) for sim in similarity_NEG_POS[0:5])
            exp_sum_pos = sum(torch.exp(sim - max_sim) for sim in similarity_NEG_POS[5:])
            total_exp_sum = exp_sum_neg + exp_sum_pos
            N = data[0].shape[0]
            GCL_loss = -torch.log(exp_sum_pos / total_exp_sum)
            feature_pro_m, reg_loss1 = F_MASK1(feature_pro)
            feature_pro_z, reg_loss2 = F_MASK2(z1_central)
            cross_att_output1, _ = crossAtt(feature_pro_m, feature_pro_z, feature_pro_z)
            cross_att_output1 = cross_att_output1.squeeze(0)
            cross_att_output2, _ = crossAtt(feature_pro_z, feature_pro_m, feature_pro_m)
            cross_att_output2 = cross_att_output2.squeeze(0)
            cross_out = torch.cat((cross_att_output1, cross_att_output2), dim=-1) 
            pred = decoder_(cross_out)
            pred = pred.clone()
            pred[:, 0] = pred[:, 0] + 1 * (1 - data[2])
            pred[:, 1] = pred[:, 1] + 1 * (data[2])
            pred = F.softmax(pred, dim=1) 
            ground_truth = data[1] 
            V = ground_truth.shape[0]
            weight_0 = 1 / V 
            weight_1 = (V - 1) / V  
            weights = torch.tensor([weight_0, weight_1])
            criterion = nn.CrossEntropyLoss(weight=weights)
            weighted_loss = criterion(pred, ground_truth.long())  
            loss_total = weighted_loss + 0.1 * VAE_loss_total + 0.1 * GCL_loss
            index_data1 = (data[1] == 1).nonzero(as_tuple=True)[0]
            sorted_indices = pred[:, 1].sort(descending=True).indices
            ranks = [(sorted_indices == idx).nonzero(as_tuple=True)[0].item() for idx in index_data1][0] + 1
            f1 = 2 * 1 * (1/ranks)/(1 + 1/ranks)
            time_iter = time.time() - start
            train_loss += weighted_loss.item() * 1
            n_samples += 1
            F_score += f1
            exe_time += 1
            if batch_idx % 5 == 0 or batch_idx == len(train_loader) - 1: 
                print('**Test Epoch: {} [{}/{} ({:.0f}%)]\tF-score:{:.4f} \tsecond/iteration: {:.4f}'.format(
                    epoch + 1, n_samples, len(train_loader.dataset),
                    100. * (batch_idx + 1) / len(train_loader), F_score / exe_time,
                    time_iter / (batch_idx + 1)))
with open('data_True.pkl', 'rb') as f:
    Twitter_True = pickle.load(f)
with open('data_False.pkl', 'rb') as f:
    Twitter_False = pickle.load(f)
with open('data_graph.pkl', 'rb') as f:
    Twitter_union_graph_inf = pickle.load(f)
rnd_state = np.random.RandomState(1111)
datareader_Twitter = DataReader_snapshot(Twitter_False,
                                rnd_state=rnd_state,
                                folds=10,
                                union_graph_inf = Twitter_union_graph_inf)
datareader_Cons = DataReader_snapshot(Twitter_True,
                                rnd_state=rnd_state,
                                folds=10,
                                union_graph_inf = Twitter_union_graph_inf)
print('Datareader构建完成')
n_folds = 10
for fold_id in range(n_folds):
    loaders_Twitter = []
    gdata_POS1_NEG2 = []
    for split in ['train', 'test']:
        gdata_Twitter = GraphData(fold_id=fold_id,
                                  datareader=datareader_Twitter,
                                  split=split)
        if split == 'train':
            gdata_Cons = GraphData(fold_id=fold_id,
                                      datareader=datareader_Cons,  
                                      split=split)
            gdata_POS1_NEG2.append(gdata_Twitter)
            gdata_POS1_NEG2.append(gdata_Cons)
        loader_Twitter = DataLoader(gdata_Twitter, 
                                    batch_size=1, 
                                    shuffle=False, 
                                    num_workers=4,
                                    collate_fn=collate_batch) 
        loaders_Twitter.append(
            loader_Twitter)  
    print('\nDatasets: FOLD {}/{}, train {}, test {}'.format(fold_id + 1, n_folds, len(loaders_Twitter[0].dataset),
                                                            len(loaders_Twitter[1].dataset)))
    feature_aggregation = feature_Model(feature_hidden_dim=64, multimodal_dim=256)
    VAE_NEG = VAE(input_channels=768, latent_dim=256, out_channels=768)
    VAE_POS = VAE(input_channels=768, latent_dim=256, out_channels=768)
    F_MASK1 = FeatureMaskingModulev2(num_features=256, temperature=1, reg_weights=[1, -1]) 
    F_MASK2 = FeatureMaskingModulev2(num_features=256, temperature=1, reg_weights=[1, -1])
    crossAtt = nn.MultiheadAttention(256, num_heads=4)
    decoder_ = nn.Linear(512, 2)
    opt = torch.optim.Adam([
        {'params': feature_aggregation.parameters(), 'lr': 0.0001},
        {'params': VAE_NEG.parameters(), 'lr': 0.0001},
        {'params': F_MASK1.parameters(), 'lr': 0.0001},
        {'params': F_MASK2.parameters(), 'lr': 0.0001},
        {'params': crossAtt.parameters(), 'lr': 0.0001},
        {'params': decoder_.parameters(), 'lr': 0.0001},
    ])
    for epoch in range(args.epochs):
        train(loaders_Twitter[0], args)
        test(loaders_Twitter[1], args)