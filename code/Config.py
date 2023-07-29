

config_npi = {
    'dataset': 'npi',

    'use_cuda': True,

    # model setting
    'embedding_dim': 64,
    'time_embedding_dim': 64,
    'k_shots': 3,
    'interval': 2,

    'phi_update': 1,
    'omega_update': 1,

    'lr': 1e-3,
    'base_lr': 25e-3,
    'encoder_lr': 25e-3,
    'local_lr': 25e-3,

    'num_neighbors': 16,
    'batch_size': 48,  # for each batch, the number of tasks
    'num_epoch': 24,
    'attn_mode': 'simple',
    'use_time': 'time',
    'agg_method': 'attn',
    'num_layers': 2,
    'n_head': 2,
    'drop_out': 0.5,
}


