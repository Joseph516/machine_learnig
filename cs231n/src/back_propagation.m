clc; clear;close all;

x = [1,2,3,4,5,6]; % 1x6
w1 = ones(length(x),1).*0.01; % 6x1
w2 = ones(length(x),1).*0.01; % 6x1
L_true = 10;


epoch = 1;
epochs = 50;
learning_rate = 0.5;
gradient = zeros(1, epochs);
while(epoch < epochs)
% forward propagation
z1 = x*w1; % 1x1
h1 = max(0,z1); % 1x1
y = w2*h1; % 6x1
L = norm(y);% 1x1

% loss
dL_L = L - L_true;
gradient(epoch) = dL_L;
% back propagation
dL_y = dL_L * 2*y;
dL_w2 = dL_y * h1';
dL_h1 = w2' * dL_y;
dL_z1 = max(0, dL_h1);
dL_w1 = x'*dL_z1;

% gradient descent
w1 = w1 - dL_w1.*learning_rate;
w2 = w2 - dL_w2.*learning_rate;

epoch = epoch + 1;
end

figure
plot(gradient,'r', 'lineWidth', 2)


