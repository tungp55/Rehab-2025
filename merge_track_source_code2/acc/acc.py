import scipy.io

f = scipy.io.loadmat('D:/Work/pCNTT/2021/handpose/Ngoc Bach/file wavelet/video2/left_acc_dwt.mat')
data= f['left_acc_dwt']
print(data)