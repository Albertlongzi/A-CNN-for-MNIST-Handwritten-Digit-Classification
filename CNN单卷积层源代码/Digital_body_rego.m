function [W1, W3, W4] = Digital_body_rego(W1, W3, W4, X, D)
alpha=0.01;%步进
N=60000;%60000个训练数据
V1=zeros(20,20,20);%预先分配内存
dW1=zeros(9,9,20);
% Y2=zeros(10,10,20);%预先分配内存
for k=1:N
    x=reshape(X(:, :, k), 28, 28); 
    for m=1:20%20个卷积核
        V1(:,:,m)=filter2(W1(:,:,m),x,'valid');
    end
    y1=max(0,V1);
    Y2=(y1(1:2:end,1:2:end,:)+y1(2:2:end,1:2:end,:)+y1(1:2:end,2:2:end,:)+y1(2:2:end,2:2:end,:))/4;
    y2=reshape(Y2,2000,1);
    y3=ReLU(W3*y2);%转化为2000维度
    y=Softmax(W4*y3);
    %反向传播
    d = D(:,k);
    e = d - y;
    delta = e;%输出层
    e3 = W4'*delta;%反向到第二隐层
    delta3 = (y3 > 0).*e3;%反向到最开始的reshape层
    e2=W3'*delta3;
    E2=reshape(e2,size(Y2));
    E2_4=E2/4;
    E1=zeros(size(y1));
    E1(1:2:end,1:2:end,:)=E2_4;
    E1(1:2:end,2:2:end,:)=E2_4;
    E1(2:2:end,1:2:end,:)=E2_4;
    E1(2:2:end,2:2:end,:)=E2_4;
    delta1=(V1>0).*E1;
    %开始更新
    dW4 = alpha*delta*y3';
    W4  = W4 + dW4;
    
    dW3 = alpha*delta3*y2';
    W3  = W3 + dW3;
    for n=1:20
    dW1(:,:,n)=alpha*filter2(delta1(:,:,n),x,'valid');
    end
    W1=W1+dW1;
end
end