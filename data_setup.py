import breast_scripts.data_setup as breast_data_setup
import skin_scripts.data_setup as skin_data_setup

datasets=['ISIC2019-Clean', 'PH2', 'Derm7pt', 'DDSM+CBIS+MIAS_CLAHE-Binary', 'DDSM+CBIS+MIAS_CLAHE', 'INbreast']

def Build_Dataset(data_path, input_size, args):
    
    if args.dataset in datasets:
        if args.dataset_type == 'Skin':
            return skin_data_setup.Build_Dataset(True, data_path, args), skin_data_setup.Build_Dataset(False, data_path, args)
        elif args.dataset_type == 'Breast':
            return breast_data_setup.Build_Datasets(data_path, input_size, args)
    else:
        ValueError('Invalid dataset. Please choose from the following datasets: ISIC2019-Clean, PH2, Derm7pt, DDSM+CBIS+MIAS_CLAHE-Binary, DDSM+CBIS+MIAS_CLAHE, INbreast')