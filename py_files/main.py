import runner
import pandas as pd


data_folder = 'C:/Users/carbo/Documents/MIMII/RawData/+6dB/'

writer_path = 'C:/Users/carbo/Documents/MIMII/Data/' + '+6dB' +'/PSD/'


def main():

    AUCs = []
    Normal_MSE = []
    Abnormal_MSE = []
    num_latent_dim = range(13,20)
    for latent_dim in num_latent_dim:

        hyper_param = {"machine_name" : "valve", 
                "method_name" : "spectro",
                "IDchosen" : 1,
                "max_freq" : 3000,
                'learner' : 'autoencoder',
                'latent_dim' : latent_dim
                }

        AUC, avg_abnormal_MSE, avg_normal_MSE = runner.foo(data_folder = data_folder , 
            hyper_param = hyper_param,
            )

        AUCs.append(AUC)
        Normal_MSE.append(avg_abnormal_MSE)
        Abnormal_MSE.append(avg_normal_MSE)

    df = pd.DataFrame([AUCs, Normal_MSE, Abnormal_MSE], index =['AUCs', 'Normal_MSE', 'Abnormal_MSE'],
    columns =['LatentDim' + str(i) for i in num_latent_dim])

    df.to_csv('latent_space_param_search.csv')

if __name__ == '__main__':
    main()