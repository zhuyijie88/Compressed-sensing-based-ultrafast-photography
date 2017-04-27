% ---------------------- section 5.1 全变分模型的图像去噪 ----------------------
org=imread('lena.bmp');
%mask=rgb2gray(imread('ma.bmp'))>160;
[m n]=size(org);
SIGMA=0.01;
img=imnoise(org,'gaussian',SIGMA);
% kernel =[1, 0, 0, 0, 0  
%          0, 1, 0, 0, 0  
%          0, 0, 1, 0, 0  
%          0, 0, 0, 1, 0  
%          0, 0, 0, 0, 1] / 5;
% img =zeros(size(org));  
% img(:,:) = conv2(double(org(:, :)), double(kernel), 'same'); 
% mask=rand(m,n)>0.4;
% mask=uint8(mask);
% img=org.*mask;
% img
% for i=1:m
%     for j=1:n
%         if mask(i,j)==0
%            img(i,j)=0; 
%         end
%     end
% end

figure,imshow(org);
figure,imshow(uint8(img));     %合成的需要修复的图像

lambda=0.1;
a=0.1;
org=double(org);
img=double(img);
img_noise=img;
%模糊信噪比BSNR
BSNR=10*log10(sum(sum((org-img).^2))/(m*n)/SIGMA)

imgn=img;
for l=1:300         %迭代次数
    for i=2:m-1
        for j=2:n-1
                Un=sqrt((img(i,j)-img(i-1,j))^2+((img(i-1,j-1)-img(i-1,j+1))/2)^2);
                Ue=sqrt((img(i,j)-img(i,j+1))^2+((img(i-1,j+1)-img(i+1,j+1))/2)^2);
                Uw=sqrt((img(i,j)-img(i,j-1))^2+((img(i-1,j-1)-img(i+1,j-1))/2)^2);
                Us=sqrt((img(i,j)-img(i+1,j))^2+((img(i+1,j-1)-img(i+1,j+1))/2)^2);

                Wn=1/sqrt(Un^2+a^2);
                We=1/sqrt(Ue^2+a^2);
                Ww=1/sqrt(Uw^2+a^2);
                Ws=1/sqrt(Us^2+a^2);

                Hon=Wn/((Wn+We+Ww+Ws)+lambda);
                Hoe=We/((Wn+We+Ww+Ws)+lambda);
                How=Ww/((Wn+We+Ww+Ws)+lambda);
                Hos=Ws/((Wn+We+Ww+Ws)+lambda);

                Hoo=lambda/((Wn+We+Ww+Ws)+lambda);

                imgn(i,j)=Hon*img(i-1,j)+Hoe*img(i,j+1)+How*img(i,j-1)+Hos*img(i+1,j)+Hoo*img_noise(i,j);
        end
    end
    img=imgn;    
end
figure,imshow(img,[])
%峰值信噪比PSNR
B=8;                %编码一个像素用多少二进制位
MAX=2^B-1;          %图像有多少灰度级
MES=sum(sum((org-img).^2))/(m*n);     %均方差
PSNR=20*log10(MAX/sqrt(MES))           %峰值信噪比
ISNR=10*log10(sum((img_noise(:)-org(:)).^2)/sum((img(:)-org(:)).^2))


% --------------------------- section 5.2 L1范数正则器图像重构 -------------------------
M = 64;%观测值个数          
N = 256;%信号x的长度          
S = 10;%信号x的稀疏度          
Index_S = randperm(N);          
x = zeros(N,1);          
x(Index_S(1:S)) = 5*randn(S,1);%x为K稀疏的，且位置是随机的                 
K = randn(M,N);%测量矩阵为高斯矩阵      
K = orth(K')';              
sigma = 0.005;       
e = sigma*randn(M,1);      
y = K * x + e;%得到观测向量y           
lamda = 0.1*max(abs(K'*y));  
fprintf('\nlamda = %f\n',lamda);        
%% 恢复重构信号x     
%(1)TwIST 
fprintf('\nTwIST begin...');    
tic  
[x_r1] = twist(y,K,K',x,lamda);   
toc    
%Debias  
[xsorted inds] = sort(abs(x_r1), 'descend');   
AI = K(:,inds(xsorted(:)>1e-3));  
xI = pinv(AI'*AI)*AI'*y;  
x_bias1 = zeros(length(x),1);  
x_bias1(inds(xsorted(:)>1e-3)) = xI;  
%(2)IST_Basic  
fprintf('\nIST_Basic begin...\n');    
tic  
[x_r2] = ist(y,K,K',x,lamda);
toc    
%Debias  
[xsorted inds] = sort(abs(x_r2), 'descend');   
AI = K(:,inds(xsorted(:)>1e-3));  
xI = pinv(AI'*AI)*AI'*y;  
x_bias2 = zeros(length(x),1);  
x_bias2(inds(xsorted(:)>1e-3)) = xI;   
%% 绘图     
figure;
plot(y,'k.-');%绘出加噪信号y          
hold on;          
plot(x,'r');%绘出原信号x          
hold off; 
figure;          
plot(x_bias1,'k.-');%绘出x的恢复信号          
hold on;          
plot(x,'r');%绘出原信号x          
hold off;          
legend('twist','Original')    
resid=norm(x_bias1-x);
st=sprintf('Restore residuals(twist)：%f\n',resid);     
title(st);   

%Debias  
figure;          
plot(x_bias2,'k.-');%绘出x的恢复信号          
hold on;          
plot(x,'r');%绘出原信号x          
hold off;          
legend('IST','Original')   
resid=norm(x_bias2-x);
st=sprintf('Restore residuals(IST)：%f\n',resid);     
title(st);  


% --------------------------- section 5.3 TV范数正则器图像重构 ------------------------
% TV based image restoration using TwIST
% cameraman, blur uniform 9*9, BSNR = 40 dB
%
% This test computes  
%
%     x_e = arg min 0.5 ||Ax-y||^2 + tau TV(x)
%             x
% 
% with the TwIST/IST algorithm.            

%% 
x = double(imread('cameraman.tif'));
N=length(x);
% define the blur operator
middle=N/2+1;
% blurr matrix
B=zeros(N);
% Uniform blur
lx=4; %blur x-size
ly=4; %blurr y-size
B((middle-ly):(middle+ly),(middle-lx):(middle+lx))=1;
%circularly center
B=fftshift(B);
%normalize
B=B/sum(sum(B));
% convolve
y = real(ifft2(fft2(B).*fft2(x)));
% set BSNR
BSNR = 40;
Py = var(y(:));
sigma= sqrt((Py/10^(BSNR/10)));
% add noise
y=y+ sigma*randn(N);

% plot figures
figure(1); colormap gray; 
imagesc(x); axis off;
title('Original image')
figure(2); colormap gray; 
imagesc(y); axis off;
title('Noisy and blurred image')
drawnow;

tau=0.01;
%%
K=fft2(B);
KT=conj(K);
% handle functions for TwIST, that is convolution operators
A = @(x) real(ifft2(K.*fft2(x)));
AT = @(x) real(ifft2(KT.*fft2(x)));
x0=medfilt2(y);
%x0=zeros(N);
Phi=2;
eps = 1e-4;

%TwIST
tic
[x_twist,obj_twist,times_twist,mses_twist]=twist(y,A,AT,x,tau,Phi,eps,x0);
toc
figure(3); colormap gray; 
imagesc(x_twist); axis off;
title('TwIST restored image');
drawnow;
     
%IST(beta=1)
tic
[x_ist,obj_ist,times_ist,mses_ist]=ist(y,A,AT,x,tau,Phi,eps,x0);
toc
figure(4); colormap gray; 
imagesc(x_ist); axis off;
title('IST restored image');
drawnow;
%      
figure(5)
subplot(2,1,1)
semilogy(times_twist,obj_twist,'r',times_ist,obj_ist,'b','LineWidth',2)
legend('TwIST','IST')
st=sprintf('tau = %2.2e, sigma = %2.2e x0=medfilt2(y)',tau,sigma);
title(st)
ylabel('Obj function')
xlabel('CPU time (sec)')

grid
subplot(2,1,2)
plot(times_twist,mses_twist,'r',times_ist,mses_ist,'b','LineWidth',2)
legend('TwIST','IST')
ylabel('MSE')
xlabel('CPU time (sec)')


% -------------------------------------- twist ------------------------------------
function [x,objective,times,mses]=twist(y,K,KT,true_x,lambda,Phi,eps,x0,loopmax,beta,alpha)
% This function solves the linear inverse problem with a regularizer, such as
% arg min_x = 0.5*|| y - K*x ||_2^2 + lambda*Phi( x )
%
% Author: Yijie Zhu, March, 2017.

%default eigenvalues and parameters
ebsh1=1e-4;
ebshm=1;
rho=(1-sqrt(ebsh1))/(1+sqrt(ebsh1));
% %default eigenvalues and parameters
% alpha=rho^2+1; %alpha=~2
% beta=2*alpha/(ebshm+ebsh1); %beta=~2
if nargin < 11
    alpha=rho^2+1; %alpha=~2
end
if nargin < 10        
    beta=2*alpha/(ebshm+ebsh1); %beta=~2
end
if nargin < 9
    loopmax = 1000;      
if nargin<8
    no_x0=1;
else
    no_x0=0;
end      
if nargin < 7
    eps=1e-3;
end
if nargin < 6 
    Phi=1; %L1 norm
end       
if nargin < 5        
    lambda = 0.1*max(abs(KT(y)));      
end    

[m,n]=size(y);
if m<n
    y=y'; % y should be a column vector
end
dim=[0 0];
if ~isa(K, 'function_handle')
    dim = size(K);
    KT = @(x) K'*x;
    K = @(x) K*x;      
end
% when p=1, 'Phi' operator has the form Phi(x)=sum(|x|), for vector/image x.
% the denoising operator 'Psi' turns to soft thresholding function
if Phi==1
    Psi = @(x,T) sign(x).*max(abs(x) - T,0); % soft thresholding function
    Phi_operator = @(x) norm(x,1);
end
if Phi==2
%when p=2, 'Phi' operator has the form Phi(x)=TVnorm(x), for vector/image x.
% the denoising operator 'Psi' turns to 'tvdenoise' function
    tv_iters = 5;
    Psi = @(x,th)  tvdenoise_1(x,2/th,tv_iters);
    Phi_operator = @(x) TVnorm_1(x);
end
if no_x0==1
if dim(2)>0
    x_pre2 = zeros(dim(2),n); % Initialize x0=0  and  supposing K is a matrix 
else 
    x_pre2=zeros(m,n);
end
else
    x_pre2=x0;
end

x_pre1 = Psi(x_pre2+KT(y-K(x_pre2)),lambda);  % Initialize x1=soft(x0+Phi'*(y-Phi*x0),lambda)  
resid=y-K(x_pre1);
f_pre1 = 0.5*(resid(:)'*resid(:))+lambda*Phi_operator(x_pre1); % current value of the objective function
loop = 0; % Initialize iterative number
objective(1) = f_pre1;
mses(1) = sum(sum((x_pre1-true_x).^2));
% start the clock
t0 = cputime;
times(1) = cputime - t0;

while (loop<loopmax)
    %% main idea of the algorithm
    temp=Psi(x_pre1+KT(y-K(x_pre1)),lambda);
    mask = (temp ~= 0);  
    x_pre2 = x_pre2.* mask;  
    x_pre1 = x_pre1.* mask;
    x=(1-alpha)*x_pre2+(alpha-beta)*x_pre1+beta*temp;
    %%
    resid=y-K(x);
    f=0.5*(resid(:)'*resid(:))+lambda*Phi_operator(x);
    loop=loop+1;
    if(f<=f_pre1)
         x_pre2=x_pre1; % update x_pre1 and x_pre2
         x_pre1=x;
         objective(loop+1) = f;
    else
         x_pre2=x_pre1; % update x_pre1 and x_pre2
         %x_pre1=temp;
         x_pre1=x;
         resid=y-K(x_pre1);
         f=0.5*(resid(:)'*resid(:))+lambda*Phi_operator(x_pre1);
         objective(loop+1) = f;
    end
    times(loop+1) = cputime-t0;
    mses(loop+1) = sum(sum((x_pre1-true_x).^2));
    
    % judge the quality of the current signal/image at the certain iteration
    if (abs(f-f_pre1))/f_pre1<eps  && loop>=100
        fprintf('abs(f-f_pre1)/f_pre1 < %f\n',eps);
        break;
    end
    if norm(x_pre1-x_pre2)<eps && loop>=100
        fprintf('norm(x-x_pre) < %f\n',eps); 
        break;  
    end
    f_pre1=f;
end
%% record the loop time
times(end)-times(1)
if loop>=loopmax
    fprintf('the loop times are beyond the limit %d\n',loopmax);
else 
    fprintf('the loop times are %d\n',loop);
end
mses=mses/length(true_x(:));
end


% -------------------------------------- ist ------------------------------------
%IST algorithm is similar to TWIST, and I will not give it anymore.


% --------------------------- tvdneoise_1 半隐式梯度下降法-------------------------
function u = tvdenoise_1(f,lambda,iters)
%   u = tvdenoise_1(f,lambda) denoises the input image f. 
%   The output u approximately minimizes the Rudin-Osher-Fatemi (ROF)
%   denoising model
%
%       Min  TV(u) + lambda/2 || f - u ||^2_2,
%        u
%
%   where TV(u) is the total variation of u. 
%   The minimization is solved using Chambolle's method,
%   A. Chambolle, "An Algorithm for Total Variation Minimization and
%   Applications" 
%
if nargin < 3
    Tol = 1e-2;
end
dt = 0.25;
N = size(f);
id = [2:N(1),N(1)];
ir = [2:N(2),N(2)];
il = [1,1:N(2)-1];
iu = [1,1:N(1)-1];
k1=ones(N);
k1(1,:)=2;
k1(N(1),:)=0;
k2=ones(N);
k2(:,1)=2;
k2(:,N(2))=0;
p1 = zeros(N);
p2 = zeros(N);
divp = zeros(N);
lastdivp = ones(N);

for i=1:iters
    lastdivp = divp;
    z = divp - f*lambda;
    z1 = z(id,:) - z;
    z2 = z(:,ir) - z;
    denom = 1 + dt*sqrt(z1.^2 + z2.^2);
    p1 = (p1 + dt*z1)./denom;
    p2 = (p2 + dt*z2)./denom;
    divp = p1.*k1 - p1(iu,:)+p2.*k2 - p2(:,il);
end
u = f - divp/lambda;
end
