from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data/xxx/IMEC25/data/got10k_lmdb'
    settings.got10k_path = '/data/xxx/IMEC25/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data/xxx/IMEC25/data/itb'
    settings.lasot_extension_subset_path = '/data/xxx/IMEC25/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data/xxx/IMEC25/data/lasot_lmdb'
    settings.lasot_path = '/data/xxx/IMEC25/data/lasot'
    settings.network_path = '/data/xxx/IMEC25/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data/xxx/IMEC25/data/nfs'
    settings.otb_path = ''
    settings.imec25_path = '/data/xxx/IMEC25/test'
    settings.prj_dir = '/data/xxx/IMEC25'
    settings.result_plot_path = '/data/xxx/IMEC25/output/test/result_plots'
    settings.results_path = '/data/xxx/IMEC25/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/xxx/IMEC25/output'
    settings.segmentation_path = '/data/xxx/IMEC25/output/test/segmentation_results'
    settings.tc128_path = '/data/xxx/IMEC25/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data/xxx/IMEC25/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data/xxx/IMEC25/data/trackingnet'
    settings.uav_path = '/data/xxx/IMEC25/data/uav'
    settings.vot18_path = '/data/xxx/IMEC25/data/vot2018'
    settings.vot22_path = '/data/xxx/IMEC25/data/vot2022'
    settings.vot_path = '/data/xxx/IMEC25/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

