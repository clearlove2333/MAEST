'''Loading package'''
import os
import scanpy as sc
from sklearn import metrics
from .preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, construct_interaction_KNN, get_feature
import pandas as pd
import numpy as np
import torch

from tqdm import tqdm

import matplotlib.pyplot as plt
import ot

class Function():
    def __init__(
            self,
            sample: str = '151673',
            datatype: str = '10x',
            device: int = 0,
            n_clusters: int = 0,
         ):
        super(Function, self).__init__()
        self._sample = sample
        self._datatype = datatype
        self._device = device
        self.n_clusters = n_clusters
        if(self._datatype == "DLPFC"):
            self.file_fold = '/zhupengfei/STExperiment/Data/GraphST/1.DLPFC/' + str(self._sample) #please replace 'file_fold' with the download path
        if(self._datatype == 'HBC'):
            self.file_fold = '/zhupengfei/STModel/Data/GraphST/3.Human_Breast_Cancer'
        if(self._datatype == "MBA"):
            self.file_fold = '/zhupengfei/STModel/Data/GraphST/2.Mouse_Brain_Anterior'
        if(self._datatype == "MBM_s1"):
            self.file_fold = '/zhupengfei/STExperiment/Data/GraphST/9.Mouse_Brain_Merge_Anterior_Posterior_Section_1/'
        if(self._datatype == "MBM_s2"):
            self.file_fold = '/zhupengfei/STExperiment/Data/GraphST/10.Mouse_Brain_Merge_Anterior_Posterior_Section_2/'
        if(self._datatype == "MBC_s1"):
            self.file_fold = '/zhupengfei/STExperiment/Data/GraphST/7.Mouse_Breast_Cancer_Sample_1/'
        if(self._datatype == "MBC_s2"):
            self.file_fold = '/zhupengfei/STExperiment/Data/GraphST/8.Mouse_Breast_Cancer_Sample_2/'
        if(self._datatype == "Mouse_Olf"):
            self.file_fold = '/zhupengfei/STExperiment/Data/GraphST/5.Mouse_Olfactory/'
        if(self._datatype == "Mouse_HT"):
            self.file_fold = '/zhupengfei/STExperiment/Data/GraphST/6.Mouse_Hippocampus_Tissue/'

    def loadData(self):
        '''Reading ST data'''
        # if(self._datatype == "10x" or self._datatype == "MBA"):
        #     adata = sc.read_visium(self.file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        if(self._datatype in ["MBM_s1", "MBM_s2", "MBC_s1", "MBC_s2", "Mouse_Olf"]):
            adata = sc.read_h5ad(self.file_fold + 'filtered_feature_bc_matrix.h5ad')
        elif(self._datatype == "Mouse_HT"):
            adata = sc.read_h5ad(self.file_fold + 'filtered_feature_bc_matrix_200115_08.h5ad')
        else:
            adata = sc.read_visium(self.file_fold, count_file='filtered_feature_bc_matrix.h5', load_images=True)
        # if(self._datatype == "MBS2"):
        #     adata = sc.read_h5ad(self.file_fold + 'Mouse_Brain_Serial_Section_2_Sagittal-' + self.sample + '_10xvisium_processed.h5ad')

        if(self.n_clusters == 0):    
            n_clusters, adata = self.get_label(adata)
            self.n_clusters = n_clusters

        adata.var_names_make_unique()

        return adata

    def process(self,adata):
        device = self._device
        adata = adata.copy()
        if 'highly_variable' not in adata.var.keys():
            preprocess(adata)

        if 'adj' not in adata.obsm.keys():
            if self._datatype in ['Stereo', 'Slide']:
                u,v = construct_interaction_KNN(adata)
            else:
                u,v = construct_interaction(adata)

        if 'feat' not in adata.obsm.keys():
            get_feature(adata)


        features = torch.FloatTensor(adata.obsm['feat'].copy())
        adj = adata.obsm['adj']
        graph_neigh = torch.FloatTensor(adata.obsm['graph_neigh'].copy() + np.eye(adj.shape[0]))

        if self._datatype in ['Stereo', 'Slide']:
            # using sparse
            print('Building sparse matrix ...')
            adj = preprocess_adj_sparse(adj).to(device)
        else:
            adj = preprocess_adj(adj)
            adj = torch.FloatTensor(adj).to(device)

        return adata, u, v

    def clusting(self, adata, model, graph, x, power, device, refinement=False):
        model.eval()

        with torch.no_grad():
            if(power > 0):
                x = model.embed_power(graph.to(device), x.to(device), power).detach().cpu().numpy()
            else:
                x = model.embed(graph.to(device), x.to(device)).detach().cpu().numpy()

        adata.obsm['emb'] = x
        # print(x)

        '''Spatial clustering and refinement'''
        # set radius to specify the number of neighbors considered during refinement
        radius = 50

        tool = 'mclust'  # mclust, leiden, and louvain

        # clustering
        from .utils import clustering

        if tool == 'mclust':
            clustering(adata, self.n_clusters, radius=radius, method=tool,
                       refinement=refinement)  # For DLPFC dataset, we use optional refinement step.
        elif tool in ['leiden', 'louvain']:
            clustering(adata, self.n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01,
                       refinement=False)

        # filter out NA nodes
        adata = adata[~pd.isnull(adata.obs['ground_truth'])]

        # calculate metric ARI
        ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
        adata.uns['ARI'] = ARI

        print('Sample:', self._sample)
        print('ARI:', ARI)

        return ARI
    
    def clustingAndPlot(self, adata, model, graph, x, power, device, sample, path, refinement=False):
        model.eval()

        with torch.no_grad():
            if(power > 0):
                x = model.embed_power(graph.to(device), x.to(device), power).detach().cpu().numpy()
            else:
                x = model.embed(graph.to(device), x.to(device)).detach().cpu().numpy()

        adata.obsm['emb'] = x
        # print(x)

        '''Spatial clustering and refinement'''
        # set radius to specify the number of neighbors considered during refinement
        radius = 50

        tool = 'mclust'  # mclust, leiden, and louvain

        # clustering
        from .utils import clustering

        if tool == 'mclust':
            clustering(adata, self.n_clusters, radius=radius, method=tool,
                       refinement=refinement)  # For DLPFC dataset, we use optional refinement step.
        elif tool in ['leiden', 'louvain']:
            clustering(adata, self.n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01,
                       refinement=False)

        # plotting predicted labels by UMAP
        plt.rcParams["figure.figsize"] = (4, 4)
        sc.pp.neighbors(adata, use_rep='emb', n_neighbors=10)
        sc.tl.umap(adata)
        sc.pl.umap(adata, color='domain', title=[''], legend_loc='', show=False)
        # plt.savefig(all_path.joinpath(final_name), bbox_inches='tight', pad_inches=0)
        # plt.axis('off')
        plt.savefig(path + sample + '/umap.pdf')

        # filter out NA nodes
        adata = adata[~pd.isnull(adata.obs['ground_truth'])]

        used_adata = adata[adata.obs['ground_truth'] != 'nan',]
        sc.tl.paga(used_adata, groups='ground_truth')
        plt.rcParams["figure.figsize"] = (4, 3)
        sc.pl.paga_compare(used_adata, legend_fontsize=10, frameon=True, size=20,
                       title=sample, legend_fontoutline=2, show=False)
        plt.savefig(path + sample + '/PAGA.pdf')

        # calculate metric ARI
        ARI = metrics.adjusted_rand_score(adata.obs['domain'], adata.obs['ground_truth'])
        adata.uns['ARI'] = ARI

        print('Sample:', self._sample)
        print('ARI:', ARI)

    def clustingAndSave(self, adata, model, graph, x, power, device, sample, path, refinement=False):
        model.eval()

        with torch.no_grad():
            # x为隐藏层
            # if(power > 0):
            #     x = model.embed_power(graph.to(device), x.to(device), power).detach().cpu().numpy()
            # else:
            #     x = model.embed(graph.to(device), x.to(device)).detach().cpu().numpy()
            # x为重构的表达谱
            x = model.recon(graph.to(device), x.to(device)).detach().cpu().numpy()

        adata.obsm['emb'] = x
        # print(x)

        '''Spatial clustering and refinement'''
        # set radius to specify the number of neighbors considered during refinement
        radius = 50

        tool = 'mclust'  # mclust, leiden, and louvain

        # clustering
        from .utils import clustering

        if tool == 'mclust':
            clustering(adata, self.n_clusters, radius=radius, method=tool,
                       refinement=refinement)  # For DLPFC dataset, we use optional refinement step.
        elif tool in ['leiden', 'louvain']:
            clustering(adata, self.n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01,
                       refinement=False)

        adata.write(path + sample + '.h5ad',compression="gzip")

    
    def clusting_no_label(self, adata, model, graph, x, power, device, refinement=False):
        model.eval()

        with torch.no_grad():
            # x为隐藏层
            if(power > 0):
                x = model.embed_power(graph.to(device), x.to(device), power).detach().cpu().numpy()
            else:
                x = model.embed(graph.to(device), x.to(device)).detach().cpu().numpy()

        adata.obsm['emb'] = x

        '''Spatial clustering and refinement'''
        # set radius to specify the number of neighbors considered during refinement
        radius = 50

        tool = 'mclust'  # mclust, leiden, and louvain

        # clustering
        from .utils import clustering

        if tool == 'mclust':
            clustering(adata, self.n_clusters, radius=radius, method=tool,
                       refinement=refinement)  # For DLPFC dataset, we use optional refinement step.
        elif tool in ['leiden', 'louvain']:
            clustering(adata, self.n_clusters, radius=radius, method=tool, start=0.1, end=2.0, increment=0.01,
                       refinement=refinement)
            
    
    def get_label(self, adata):
        #加载ground_truth
        df_meta = pd.read_csv(self.file_fold + '/metadata.tsv', sep='\t')
        if(self._datatype == 'DLPFC'):
            df_meta_layer = df_meta['layer_guess']
        else:
            df_meta_layer = df_meta['ground_truth']

        adata.obs['ground_truth'] = df_meta_layer.values

        label = df_meta_layer.values
        not_null_list = ~pd.isnull(label)
        unique_label, label_temp = np.unique(label[not_null_list], return_inverse=True)
        label[not_null_list] = label_temp

        #计算聚类数目
        n_clusters = len(unique_label)

        return n_clusters, adata
    
    def refine_label(self, adata, radius=50, key='label'):
        n_neigh = radius
        new_type = []
        old_type = adata.obs[key].values
        
        #calculate distance
        position = adata.obsm['spatial']
        distance = ot.dist(position, position, metric='euclidean')
            
        n_cell = distance.shape[0]
        
        for i in range(n_cell):
            vec  = distance[i, :]
            index = vec.argsort()
            neigh_type = []
            for j in range(1, n_neigh+1):
                neigh_type.append(old_type[index[j]])
            max_type = max(neigh_type, key=neigh_type.count)
            new_type.append(max_type)
            
        new_type = [str(i) for i in list(new_type)]    
        
        return new_type

    def plot_label(self, adata, ari):
        # 用于计算ACC
        # y_hat = np.array(new_type).astype(int)

        # filter out NA nodes
        adata = adata[~pd.isnull(adata.obs['ground_truth'])]

        # plotting spatial clustering result
        adata.obs['domain'] = adata.obs['domain'].astype(str)

        fig, ax = plt.subplots(figsize=(10, 4))
        sc.pl.spatial(adata,
                        ax=ax,
                        img_key="hires",
                        color=["domain"],
                        title=["ARI=%.4f" % ari],
                        show=False)
        
        plt.savefig("./results/" + self._datatype + "/clusting.png")
