clear all;

filename1 = '2016-10-12_12-50-38 /train.log';
filename2 = '2016-10-12_12-50-38/train_l2_loss.log';
filename3 = '2016-10-12_12-50-38/test.log';
filename4 = '2016-10-12_12-50-38/test_l2_loss.log';

delimiterIn = '\n';
headerlinesIn = 1;
Train_A = importdata(filename1,delimiterIn,headerlinesIn);
Train_L = importdata(filename2,delimiterIn,headerlinesIn);
Test_A = importdata(filename3,delimiterIn,headerlinesIn);
Test_L = importdata(filename4,delimiterIn,headerlinesIn);

%plot(A.data)
%title('testing angle error variation')/
%xlabel('epoch sequence') % x-axis label
%ylabel('angle error') % y-axis label

figure
subplot(2,2,1)
plot(Train_A.data)
title('Training Angle Error Variation')
xlabel('epoch sequence') % x-axis label
ylabel('angle error') % y-axis label

subplot(2,2,2)
plot(Test_A.data)
title('Testing Angle Error Variation')
xlabel('epoch sequence') % x-axis label
ylabel('angle error') % y-axis label

subplot(2,2,3)
plot(Train_L.data)
title('Training L2 Loss Variation')
xlabel('epoch sequence') % x-axis label
ylabel('L2 loss') % y-axis label

subplot(2,2,4)
plot(Test_L.data)
title('Testing L2 Loss Variation')
xlabel('epoch sequence') % x-axis label
ylabel('L2 loss') % y-axis label