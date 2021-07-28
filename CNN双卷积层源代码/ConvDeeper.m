function  [W1,W3,W5,W6,W7]=ConvDeeper(W1,W3,W5,W6,W7,Xx,D)
alpha=0.01;%步进为0.01
[Xxc,Xxk,Xxn]=size(Xx);%Xx的长，宽，个数
[w1c, w1k, w1n] = size(W1);%W1的长，宽，个数
E1=zeros(Xxc-w1c+1,Xxk-w1k+1,w1n);%第一次线性卷积后的长度变化
update_x=zeros(w1c, w1k, w1n);%更新与W1的大小一致
[w3c,w3k,~,w3n]=size(W3);%W3的长，宽和个数，提高代码的可改性
E3=zeros((Xxc-w1c+1)/2-w3c+1,(Xxk-w1k+1)/2-w3k+1,w3n);%第二次线性卷积后的长度变化
update_y2=zeros(w3c,w3k,w1n,w3n);%更新与W3的大小一致
    for k=1:Xxn%循环Xxn的个数次
    X=Xx(:,:,k);
    %卷积1和池化1
    V1 = Conv(X, W1);      % Convolution without rotation, 24x24x5
    Y1 = ReLU(V1);             % ReLU
    Y2 = Pool(Y1);               % Pooling,      12x12x5
    %卷积2和池化2
    V3 = Conv(Y2, W3);                 % Convolution,  10x10x10
    Y3 = ReLU(V3);                    %
    Y4 = Pool(Y3);                    % Pool,         5x5x10
    %全连接
    y4 = reshape(Y4, [], 1);          %                   250  
    v5 = W5*y4;                       %                        100
    y5 = ReLU(v5);                    %ReLU,     
    %第二隐层
    v6  = W6*y5;                       %            100
    y6 = ReLU(v6);                    %ReLU,
    %输出层
    v  = W7*y6;                          %     10
    y  = Softmax(v);    
    
    %反向传播
    delta = D(:,k)-y; % Output layer这里delta和e相等
    e6 = W7'*delta;             % Hidden(ReLU) layer
    delta6 = (v6>0).* e6;
    e5 = W6'*delta6; 
    delta5 = (v5>0).* e5;
    e4 = W5'*delta5; 
    %第二个池化层
    E4 = reshape(e4, size(Y4))/4; % Pooling layer
    E3(1:2:end,1:2:end,:) = E4;
    E3(1:2:end,2:2:end,:) = E4;
    E3(2:2:end,1:2:end,:) = E4;
    E3(2:2:end,2:2:end,:) = E4;
    
    delta3 = (V3>0).* E3;      % ReLU layer 10x10x10
    %第二个卷积层
    for m=1:w3n
        for n=1:w1n
            update_y2(:,:,n,m)=filter2(delta3(:,:,m),Y2(:,:,n),'valid');
            %update_y2(:,:,n,m)=conv2(Y2(:,:,n), rot90(delta3(:, :, m),2), 'valid');
        end
    end
    E2Ex=zeros(size(Y2,1),size(Y2,2),w1n,w3n);
    for p=1:w3n
        for o=1:w1n
            E2Ex(:,:,o,p)=filter2(delta3(:,:,p),W3(:,:,o,p),'full');
            %E2Ex(:,:,o,p)=conv2(delta3(:,:,p), W3(:,:,o,p));
        end
    end
    %第一个池化层
    E2 = sum(E2Ex,4); %12x12x5
    E2 = E2/4; % Pooling layer
    E1(1:2:end,1:2:end,:) = E2;
    E1(1:2:end,2:2:end,:) = E2;
    E1(2:2:end,1:2:end,:) = E2;
    E1(2:2:end,2:2:end,:) = E2;
    delta1 = (V1>0).* E1;      % ReLU layer
    %第一个卷积层
    for j=1:w1n
        update_x(:,:,j)=filter2(delta1(:,:,j),X,'valid');
        %update_x(:,:,j)=conv2(X, rot90(delta1(:, :, j),2), 'valid');
    end
    
    W1 = W1 + alpha*update_x; 
    W3 = W3 + alpha*update_y2; 
    W5 = W5 + alpha*delta5*y4';    
    W6 = W6 + alpha*delta6*y5';    
    W7 = W7 + alpha*delta *y6';
    end
end
