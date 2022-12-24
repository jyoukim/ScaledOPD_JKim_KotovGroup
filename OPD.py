# written by Minjeong Cha and Ji-Young Kim


import numpy as np
import os
from biopandas.pdb import PandasPdb
import pandas as pd
import itertools


## prepare 3D atomic coordinates based on file format and usage. 

def coord_pdb(pdb_name):
    coord = PandasPdb()
    coord.read_pdb(pdb_name+'.pdb')
    c1 = coord.df['HETATM']
    
    c1_xyz = pd.DataFrame(np.zeros((len(c1), 3)), columns=['x', 'y', 'z'])
    c1_xyz['x'] = c1['x_coord']
    c1_xyz['y'] = c1['y_coord']
    c1_xyz['z'] = c1['z_coord']
       
    c1_all = c1_xyz.to_numpy()


    return c1_all


def coord_txt(txt_name):
    
    coord = pd.read_csv(txt_name+'.txt', sep='\t', header=None)
    xyz = []
    for ii in range(int(coord.shape[1]/3)):
        x = coord.iloc[:,ii*3]
        y = coord.iloc[:,ii*3+1]
        z = coord.iloc[:,ii*3+2]
        xyz_ = np.hstack([x, y, z])
        xyz.append(xyz_)
        
    xyz_txt = np.asarray(xyz)   
    
    return xyz_txt 
    

def grain_coord_pdb(pdb_name, num_cl):
    
    coord = PandasPdb()
    coord.read_pdb(pdb_name+'.pdb')
    c1 = coord.df['HETATM']
    
    c1_xyz = pd.DataFrame(np.zeros((len(c1), 3)), columns=['x', 'y', 'z'])
    c1_xyz['x'] = c1['x_coord']
    c1_xyz['y'] = c1['y_coord']
    c1_xyz['z'] = c1['z_coord']
    
    
    c1_all = c1_xyz.to_numpy()
    
    from sklearn.cluster import KMeans
    X = c1_xyz.iloc[:,:3].to_numpy()
    kmeans = KMeans(n_clusters=num_cl, random_state=0).fit(X)
    
    c_coord = []
    for ii in range(num_cl):
        c_coord.append(c1_xyz.iloc[np.where(kmeans.labels_==ii)[0]])
    
    
    c_coord_w = []
    for ii in range(len(c_coord)):
        xyz = c_coord[ii].iloc[:,:3].to_numpy()
        com = np.mean(xyz, axis=0)
        c_coord_w.append(com)
        
    c_cl = np.asarray(c_coord_w)
    
    return c_cl

    
## compute Osipov-Pickup-Dunmur Index

def osipov_gen(xyz):
    n = 2
    m = 1
    
    coord = xyz[:, :3]

    
    N = len(coord)
    P = itertools.permutations(np.arange(N),4)

    for kk in P:

    
        r_ij = coord[kk[0]]-coord[kk[1]]
        r_kl = coord[kk[2]]-coord[kk[3]]
        r_il = coord[kk[0]]-coord[kk[3]]
        r_jk = coord[kk[1]]-coord[kk[2]]
        r_ij_mag = np.linalg.norm(r_ij)
        r_kl_mag = np.linalg.norm(r_kl)
        r_il_mag = np.linalg.norm(r_il)
        r_jk_mag = np.linalg.norm(r_jk)



        G_p_up = np.dot(np.cross(r_ij, r_kl),r_il)*(np.dot(r_ij, r_jk))*(np.dot(r_jk, r_kl))
        G_p_down = ((r_ij_mag*r_jk_mag*r_kl_mag)**n)*((r_il_mag)**m)
        G_p = G_p_up/G_p_down
  
    
        yield G_p


def opd(coord):
    
    N = len(coord)
    OPD = (4*3*2*1)/((N)**4)*(1/3)*sum(osipov_gen(coord))
    
    return OPD



