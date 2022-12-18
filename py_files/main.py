import runner

data_folder = 'C:/Users/carbo/Documents/MIMII/RawData/+6dB/'

writer_path = 'C:/Users/carbo/Documents/MIMII/Data/' + '+6dB' +'/PSD/'


def main():
    hyper_param = {"machine_name" : "fan", 
                "method_name" : "spectro",
                "IDchosen" : 2,
                "max_freq" : 3000,
                'learner' : 'autoencoder'
                }


    runner.foo(data_folder = data_folder , 
            hyper_param = hyper_param
            )

if __name__ == '__main__':
    main()