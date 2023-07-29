import numpy as np
import torch
import pandas as pd
import random
import math


def datasets():
    df_d = pd.read_csv('../data/interaction_NPInterv4.txt', sep="\t")

    # str_d = list(g_df_d.SMILES.values) + list(g_df_d_v.SMILES.values) + list(g_df_d_t.SMILES.values)
    # tar_d = list(g_df_d.Protein.values) + list(g_df_d_v.Protein.values) + list(g_df_d_t.Protein.values)
    # set_str_d = list(set(str_d))
    # set_tar_d = list(set(tar_d))
    df_d2 = pd.DataFrame()

    # df_d1['ts'] = list(df_d['Y'])
    # df_d1 = (df_d[df_d['Y'] == 1]).copy()

    # df_d1 = pd.concat([df_d1, df_d1])
    str_d = df_d2['ncName'] = list(df_d['ncName'][:300000])
    tar_d = df_d2['tarName'] = list(df_d['tarName'][:300000])
    set_all = str_d + tar_d

    set_str_d = list(set(set_all))

    set_str_d.sort(key=set_all.index)

    smile_map = {}
    d = 1
    for i in set_str_d:
        smile_map[i] = d
        d += 1

    df_d3 = pd.DataFrame()

    str_d1 = df_d3['u'] = list(df_d2['ncName'].map(smile_map).values)
    tar_d1 = df_d3['i'] = list(df_d2['tarName'].map(smile_map).values)
    d=100

    list1 = list(range(5, d+5,5))
    list1 = [x / 100 for x in list1]

    list2 = [val for val in list1 for i in range(math.ceil(len(df_d2) / 20))]

    q = (len(df_d2))
    b1 = list(list2[:int(q)])

    random.shuffle(b1)

    df_d3['ts'] = b1
    df_d3['lable'] = 1
    df_d3['idx'] = list(range(len(df_d2)))



    return df_d3



def reindex(df):
    upper_src = df.u.max() + 1
    print('upper_src', upper_src)

    new_df = df.copy()
    print(new_df.u.max())
    print(new_df.i.max())

    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

    print(new_df.u.min())
    print(new_df.u.max())
    print(new_df.i.min())
    print(new_df.i.max())
    print(new_df.idx.min())
    print(new_df.idx.max())

    return new_df


def run(data_name):
    OUT_DF = '../data/{0}/ml_{0}.csv'.format(data_name)
    OUT_FEAT = '../data/{0}/ml_{0}.npy'.format(data_name)
    OUT_NODE_FEAT = '../data/{0}/ml_{0}_node.npy'.format(data_name)


    df = datasets()
    df.to_csv(OUT_DF)
    new_df = reindex(df)
    new_df['idx'] = list(range(1, len(new_df) + 1))

    feat = torch.empty((new_df.idx.max() + 1, 172))
    feat = torch.nn.init.xavier_uniform_(feat, gain=1.414)

    max_idx = max(new_df.u.max(), new_df.i.max())
    # rand_feat = np.zeros((max_idx + 1, feat.shape[1]))
    rand_feat = torch.empty((max_idx + 1, feat.shape[1]))
    rand_feat = torch.nn.init.xavier_uniform_(rand_feat)

    print(feat.shape)
    np.save(OUT_FEAT, feat)
    np.save(OUT_NODE_FEAT, rand_feat)
    print('exit')

    return


run('npi')


