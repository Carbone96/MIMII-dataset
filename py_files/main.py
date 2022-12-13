import runner
import os 
import tensorflow as tf
# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')

base_folder = '/MIMII/RawData/' + '+6dB' + '/' # fan/id_00/abnormal'
user_path = 'C:/Users/carbo/Documents/'
writer_path = 'C:/Users/carbo/Documents/MIMII/Data/' + '+6dB' +'/PSD/'



def main():



    hyper_param = {"machine_name" : "test", 
                "method_name" : "psd",
                "IDchosen" : 3,
                "max_freq" : 3000
                }


    runner.foo(user_path = user_path , 
            base_folder = base_folder,
            hyper_param = hyper_param
            )

if __name__ == '__main__':
    main()