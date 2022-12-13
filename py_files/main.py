import runner

data_folder = '/MIMII/RawData/' + '+6dB' + '/' # fan/id_00/abnormal'
user_path = 'C:/Users/carbo/Documents/'
writer_path = 'C:/Users/carbo/Documents/MIMII/Data/' + '+6dB' +'/PSD/'


def main():
    hyper_param = {"machine_name" : "test", 
                "method_name" : "scalo",
                "IDchosen" : 3,
                "max_freq" : 3000
                }


    runner.foo(user_path = user_path , 
            data_folder = data_folder,
            hyper_param = hyper_param
            )

if __name__ == '__main__':
    main()