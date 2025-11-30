from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/data/got10k_lmdb'
    settings.got10k_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/data/itb'
    settings.lasot_extension_subset_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/data/lasot_lmdb'
    settings.lasot_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/data/lasot'
    settings.network_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/data/nfs'
    settings.otb_path = ''
    settings.hot2020_path = '/data/lizf/HOT/dataset/test/test_HSI'
    settings.imec25_path = '/data/lizf/HOT/IMEC25Dataset/test/test_HSI'
    settings.prj_dir = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020'
    settings.result_plot_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/output/test/result_plots'
    settings.results_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/output'
    settings.segmentation_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/output/test/segmentation_results'
    settings.tc128_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/data/trackingnet'
    settings.uav_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/data/uav'
    settings.vot18_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/data/vot2018'
    settings.vot22_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/data/vot2022'
    settings.vot_path = '/data/lizf/work_2025/new_band_selection/0422/github/CSSTrack-HOT2020/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

