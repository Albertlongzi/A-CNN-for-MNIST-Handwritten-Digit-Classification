clear all;close all;
load('MNISTData.mat');
%W1=zeros(9,9,20);%老师用的W1=randn(9,9,20)正态
% W2=ones(2000,1);
rng(1);
W1 = randn([5 5 5]);
W3 = randn([3 3 5 10]);
% W5=(2*rand(100,250) - 1)/20;%老师在这里做了个/20的操作
% W6=(2*rand(10,100) - 1)/10;%老师说什么为了能量稳定
W5 = (2*rand(50, 250) - 1)/(2000/100);
W6 = (2*rand(50,50) - 1);
W7 = (2*rand(10, 50) - 1)/(50/10);
XX_TrainTest = X_Train(:,:,1:100:end);
D_TrainTest=D_Train(:,1:100:end);
N_TT = size(XX_TrainTest,3);
d_comp_tt = zeros(1,N_TT);
[~, d_true_tt] = max(D_TrainTest);

t1 = cputime;
for epoch = 1:10
  epoch
  [W1, W3, W5, W6, W7] = MnistConvDeeper(W1, W3, W5, W6, W7, X_Train, D_Train);
  
%     % Test
%     for k = 1:N_TT
%       X = XX_TrainTest(:, :, k);                   % Input,           28x28
%       V1 = Conv(X, W1);                 % Convolution,  24x24x5
%       Y1 = ReLU(V1);                    %
%       Y2 = Pool(Y1);                    % Pool,         12x12x5
%       
%       V3 = Conv(Y2, W3);                 % Convolution,  10x10x10
%       Y3 = ReLU(V3);                    %
%       Y4 = Pool(Y3);                    % Pool,         5x5x10
%       
%       y4 = reshape(Y4, [], 1);          %                   250  
%       v5 = W5*y4;                       %                        100
%       y5 = ReLU(v5);                    %ReLU,     
%       v6  = W6*y5;                       %            100
%       y6 = ReLU(v6);                    %ReLU,
%       v  = W7*y6;                          %     10
%       y  = Softmax(v);                  %Softmax,            10
% 
%       [~, i] = max(y);
%       d_comp_tt(k) = i;
%     end
%     fprintf('Train accuracy is %f\n', sum(d_comp_tt == d_true_tt)/N_TT);
    
    
% Test
%
N = length(D_Test);
d_comp = zeros(1,N);
for k = 1:N
      X = X_Test(:, :, k);                   % Input,           28x28
      V1 = Conv(X, W1);                 % Convolution,  24x24x5
      Y1 = ReLU(V1);                    %
      Y2 = Pool(Y1);                    % Pool,         12x12x5
      
      V3 = Conv(Y2, W3);                 % Convolution,  10x10x10
      Y3 = ReLU(V3);                    %
      Y4 = Pool(Y3);                    % Pool,         5x5x10
      
      y4 = reshape(Y4, [], 1);          %                   250  
      v5 = W5*y4;                       %                        100
      y5 = ReLU(v5);                    %ReLU,     
      
      v6  = W6*y5;                       %            100
      y6 = ReLU(v6);                    %ReLU,
      
      v  = W7*y6;                          %     10
      y  = Softmax(v);                  %Softmax,            10

  [~, i] = max(y);
  d_comp(k) = i;
end
[~, d_true] = max(D_Test);
acc = sum(d_comp == d_true)/N;
fprintf('Accuracy is %f\n', acc);

end
t2 = cputime;
fprintf('CPU time is %f (min)\n', (t2-t1)/60);
% X=X_Train;%这里是加载了训练数据X
% D=D_Train;%这里是加载了训练数据D
% for epoch = 1:3
%   epoch
% [W1, W3, W5, W6,W7] = ConvDeeper(W1, W3, W5, W6, W7,X, D);
% N=length(D_Test);
% d_comp=zeros(1,N);
% V1=zeros(24,24,10);
% for k=1:N
% %     x = X(:, :, k);               % Input, 28x28
% %     V1 = Conv(x, W1);      % Convolution without rotation, 24x24x5
% %     Y1 = ReLU(V1);             % ReLU
% %     Y2 = Pool(Y1);               % Pooling,      12x12x5
% %     
% %     V3 = Conv(Y2, W3);                 % Convolution,  10x10x10
% %     Y3 = ReLU(V3);                    %
% %     Y4 = Pool(Y3);                    % Pool,         5x5x10
% %     
% %     y4 = reshape(Y4, [], 1);          %                   250  
% %     v5 = W5*y4;                       %                        100
% %     y5 = ReLU(v5);                    %ReLU,     
% %     v  = W6*y5;                          %     10
% %     y  = Softmax(v);                  %Softmax, 
%  X = X_Test(:, :, k);                   % Input,           28x28
%       V1 = Conv(X, W1);                 % Convolution,  24x24x5
%       Y1 = ReLU(V1);                    %
%       Y2 = Pool(Y1);                    % Pool,         12x12x5
%       
%       V3 = Conv(Y2, W3);                 % Convolution,  10x10x10
%       Y3 = ReLU(V3);                    %
%       Y4 = Pool(Y3);                    % Pool,         5x5x10
%       
%       y4 = reshape(Y4, [], 1);          %                   250  
%       v5 = W5*y4;                       %                        100
%       y5 = ReLU(v5);                    %ReLU,     
%       
%       v6  = W6*y5;                       %            100
%       y6 = ReLU(v6);                    %ReLU,
%       
%       v  = W7*y6;                          %     10
%       y  = Softmax(v);                  %Softmax,            10
%      [~,i]=max(y);
%      d_comp(k)=i;
% end
% [~,d_true]=max(D_Test);
% acc=sum(d_comp==d_true);
% fprintf('Accuracy is %f\n',acc/N);
% end
