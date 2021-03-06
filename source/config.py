FFT_SIZE = 8192
Q_NORM = 2

DATA_DICT = {'cent_avg': 0, 'cent_std': 0, 'cent_skw': 0, 'cent_krt': 0, 'cent_min': 0, 'cent_max': 0,
             'spread_avg': 0, 'spread_std': 0, 'spread_skw': 0, 'spread_krt': 0, 'spread_min': 0, 'spread_max': 0,
             'skew_avg': 0, 'skew_std': 0, 'skew_skw': 0, 'skew_krt': 0, 'skew_min': 0, 'skew_max': 0,
             'kurt_avg': 0, 'kurt_std': 0, 'kurt_skw': 0, 'kurt_krt': 0, 'kurt_min': 0, 'kurt_max': 0,
             'n_cent_avg': 0, 'n_cent_std': 0, 'n_cent_skw': 0, 'n_cent_krt': 0, 'n_cent_min': 0, 'n_cent_max': 0,
             'n_spread_avg': 0, 'n_spread_std': 0, 'n_spread_skw': 0, 'n_spread_krt': 0, 'n_spread_min': 0, 'n_spread_max': 0,
             'n_skew_avg': 0, 'n_skew_std': 0, 'n_skew_skw': 0, 'n_skew_krt': 0, 'n_skew_min': 0, 'n_skew_max': 0,
             'n_kurt_avg': 0, 'n_kurt_std': 0, 'n_kurt_skw': 0, 'n_kurt_krt': 0, 'n_kurt_min': 0, 'n_kurt_max': 0,
             'flux_avg': 0, 'flux_std': 0, 'flux_skw': 0, 'flux_krt': 0, 'flux_min': 0, 'flux_max': 0,
             'rolloff_avg': 0, 'rolloff_std': 0, 'rolloff_skw': 0, 'rolloff_krt': 0, 'rolloff_min': 0, 'rolloff_max': 0,
             'slope_avg': 0, 'slope_std': 0, 'slope_skw': 0, 'slope_krt': 0, 'slope_min': 0, 'slope_max': 0,
             'flat_avg': 0, 'flat_std': 0, 'flat_skw': 0, 'flat_krt': 0, 'flat_min': 0, 'flat_max': 0,
             'cent_hp_avg': 0, 'cent_hp_std': 0, 'cent_hp_skw': 0, 'cent_hp_krt': 0, 'cent_hp_min': 0, 'cent_hp_max': 0,
             'spread_hp_avg': 0, 'spread_hp_std': 0, 'spread_hp_skw': 0, 'spread_hp_krt': 0, 'spread_hp_min': 0, 'spread_hp_max': 0,
             'skew_hp_avg': 0, 'skew_hp_std': 0, 'skew_hp_skw': 0, 'skew_hp_krt': 0, 'skew_hp_min': 0, 'skew_hp_max': 0,
             'kurt_hp_avg': 0, 'kurt_hp_std': 0, 'kurt_hp_skw': 0, 'kurt_hp_krt': 0, 'kurt_hp_min': 0, 'kurt_hp_max': 0,
             'n_cent_hp_avg': 0, 'n_cent_hp_std': 0, 'n_cent_hp_skw': 0, 'n_cent_hp_krt': 0, 'n_cent_hp_min': 0, 'n_cent_hp_max': 0,
             'n_spread_hp_avg': 0, 'n_spread_hp_std': 0, 'n_spread_hp_skw': 0, 'n_spread_hp_krt': 0, 'n_spread_hp_min': 0,
             'n_spread_hp_max': 0,
             'n_skew_hp_avg': 0, 'n_skew_hp_std': 0, 'n_skew_hp_skw': 0, 'n_skew_hp_krt': 0, 'n_skew_hp_min': 0, 'n_skew_hp_max': 0,
             'n_kurt_hp_avg': 0, 'n_kurt_hp_std': 0, 'n_kurt_hp_skw': 0, 'n_kurt_hp_krt': 0, 'n_kurt_hp_min': 0, 'n_kurt_hp_max': 0,
             'flux_hp_avg': 0, 'flux_hp_std': 0, 'flux_hp_skw': 0, 'flux_hp_krt': 0, 'flux_hp_min': 0, 'flux_hp_max': 0,
             'rolloff_hp_avg': 0, 'rolloff_hp_std': 0, 'rolloff_hp_skw': 0, 'rolloff_hp_krt': 0, 'rolloff_hp_min': 0, 'rolloff_hp_max': 0,
             'slope_hp_avg': 0, 'slope_hp_std': 0, 'slope_hp_skw': 0, 'slope_hp_krt': 0, 'slope_hp_min': 0, 'slope_hp_max': 0,
             'flat_hp_avg': 0, 'flat_hp_std': 0, 'flat_hp_skw': 0, 'flat_hp_krt': 0, 'flat_hp_min': 0, 'flat_hp_max': 0,
             }

