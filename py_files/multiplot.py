#### Multiple plots
import matplotlib.pyplot as plt

def multiplot(f_vec,df_PSD) -> None:
    """ Create multiple PSD plots from different audio files of a dataframe"""
    subplot_list = [411, 412, 413, 414]
    file_num = [50,100,150,200]
    
    plt.figure(1)
    
    for i in range(len(subplot_list)):
        plt.subplot(subplot_list[i])
        plt.semilogy(f_vec, df_maker.get_raw(df_PSD,file_num[i]))
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.ylim([0.5e-9, 1e-6])
        plt.title('PSD ' + str(file_num[i]))

    plt.show()