load('MNISTData.mat');
%W1=zeros(9,9,20);%老师用的W1=randn(9,9,20)正态
% W2=ones(2000,1);
W1=randn(9,9,20);
W3=(2*rand(100,2000) - 1)/20;%老师在这里做了个/20的操作
W4=(2*rand(10,100) - 1)/10;%老师说什么为了能量稳定
X=X_Train;%这里是加载了训练数据X
D=D_Train;%这里是加载了训练数据D
% for N=1:20
% W1(:,:,N) = 2*rand(9, 9) - 1;%随机初始化定义20个卷积核
% end
       % train
[W1, W3, W4] = Digital_body_rego(W1, W3, W4, X, D);
N=length(D_Test);
V1=zeros(20,20,20);%预先分配内存
d_comp=zeros(1,N);
for k=1:N
    X=X_Test(:,:,k);
   for m=1:20%20个卷积核
        V1(:,:,m)=filter2(W1(:,:,m),X,'valid');
        y1=max(0,V1);
        Y2=(y1(1:2:end,1:2:end,:)+y1(2:2:end,1:2:end,:)+y1(1:2:end,2:2:end,:)+y1(2:2:end,2:2:end,:))/4;
    end
    y2=reshape(Y2,2000,1);
    v3=W3*y2;y3=ReLU(v3);
    v=W4*y3;y=Softmax(v);
    [~,i]=max(y);
    d_comp(k)=i;
end
[~,d_true]=max(D_Test);
acc=sum(d_comp==d_true);
fprintf('Accuracy is %f\n',acc/N);
