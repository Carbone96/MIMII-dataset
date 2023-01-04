import runner
import pandas as pd


def main():
    
    data_folder = "C:/Users/carbo/Documents/MIMII/RawData/+6dB/"
    latent_dim = 8
    hyper_param = {"machine_name" : "test", 
                "method_name" : "spectro",
                "IDchosen" : 3,
                "max_freq" : 5000,
                'learner' : 'autoencoder',
                'latent_dim' : latent_dim
            }

    
    ID_list = [1,2,3,4]
    Machine_dic = {1 : 'slider',
                   2 : 'fan',
                   3 : 'pump',
                   4 :  'valve'}
    AUCs = []
    for i in range(4):
        machine_chosen = Machine_dic[i+1]
        AUC_temp = []
        for ID in range(4):
            hyper_param['machine_name'] = machine_chosen
            hyper_param["IDchosen"] = ID
    
            AUC = runner.foo(data_folder = data_folder,
                                       hyper_param = hyper_param
                                       )
            AUC_temp.append(AUC)
        AUCs.append(AUC_temp)
    return AUCs


if __name__ == "__main__":              
    AUCs = main()
    df = pd.DataFrame(AUCs,index = ['slider', 'fan', 'pump', 'valve'], columns = ['id_00','id_02', 'id_04', 'id_06']) 
    df.to_csv('AE_MSEError_mahalaReconstructEval_PSD_+6dB.csv')