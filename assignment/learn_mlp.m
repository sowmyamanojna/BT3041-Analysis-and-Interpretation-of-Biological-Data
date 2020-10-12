% The following example is based on the tutorial available in the
% following webpage:
% https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
% This has been used as a trail to ensure that the MLP equations
% are right

x1 = [0.05; 0.10];
y = [0.01; 0.99];

% In this example we are considering only one input and training
% it for the same :)
m = 1;
mt = 1;

n1 = 2;
n2 = 2;
n3 = 2;
id_mat = eye(n3);

W1 = [0.15, 0.20; 0.25, 0.30]; %zeros(n2, n1);
W2 = [0.4 0.45; 0.5 0.55]; %zeros(n3,n2);

b1 = 0.35;
b2 = 0.60;

save_every = 10;
fact = 1000;
epochs = 10000;
eta = 0.5;

train_accuracy = zeros(epochs/save_every,1);

for e = 1:epochs
	z1 = W1*x1 + b1;
	a1 = sigmf(z1, [1, 0]);
	
    z2 = W2*a1 + b2;
    a2 = sigmf(z2, [1, 0]);
    
    d3 = a2 - y;
    d2 = d3.*a2.*(1-a2);
    d1 = (W2'*d2).*a1.*(1-a1);

    dW2 = (d2*a1')/m;
    dW1 = (d1*x1')/m;
    
    db2 = sum(d2, 2)/m;
    db1 = sum(d1, 2)/m;

    W2 = W2 - eta*dW2;
    W1 = W1 - eta*dW1;
    b2 = b2 - eta*db2;
    b1 = b1 - eta*db1;

    if rem(e,save_every)==0
    	fprintf('Epoch: %d; eta: %f \n', e, eta);
    	train_accuracy(e/save_every) = norm(a2);
    end
end

figure;
plot(1:save_every:epochs, train_accuracy)
title("Training Accuracy vs Epochs");
xlabel("Epochs")
ylabel("Accuracy")