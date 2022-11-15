import scipy


(f, S)= scipy.signal.welch(signal, fs, nperseg=1024)

plt.semilogy(f, S)
plt.xlim([0, 100])
plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()
