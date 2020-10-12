% Accessing the images, labels and normalizing the images
x1 = double(trainImages')/255;
y = double(trainLabels');
t1 = double(testImages')/255;
yt = double(testLabels');

% Initializing the number of train and test samples
m = 800;
mt = 200;

% Defining all essential constants
% - eta_max and eta_min for the exponential eta decay function
% - epochs for training
% - fact (factor) for determinig the rate of eta decay
% - c1 rate of logistic function 
% - save_every to save the accracy values after a fixed number of epochs
eta_max = 0.3;
eta_min = 0.01;
epochs = 6000;
fact = 5000;
c1 = 15;
c2 = 0;
save_every = 10;

% Defining the size of the network
n1 = size(x1,1);
n2 = 512;
n3 = 512;
n4 = 10;

% Initializing the weights, biases and additional variables
% to store the accuracy and 
W1 = zeros(n2, n1);
W2 = zeros(n3, n2);
W3 = zeros(n4, n3);

b1 = rand(n2, 1);
b2 = rand(n3, 1);
b3 = rand(n4, 1);

id_mat = eye(n4);
train_accuracy = zeros(ceil(epochs/save_every),1);
test_accuracy = zeros(ceil(epochs/save_every),1);

% Start training
for i = 1:epochs
    % Forward pass
    z1 = W1*x1 - b1;
    a1 = sigmf(z1, [c1, 0]);

    z2 = W2*a1 - b2;
    a2 = sigmf(z2, [c1, 0]);

    z3 = W3*a2 - b3;
    a3 = sigmf(z3, [1, 0]);

    % Train and test accuracy calculation 
    if rem(i, save_every) == 0
        fprintf('Epoch: %d; eta: %f \n', i, eta);  
        [~, idx] = max(a3);
        train_accuracy(i/save_every) = sum(idx' == y+1)/m;

        % Testing
        t2 = sigmf(W1*t1 - b1, [c1,0]);
        t3 = sigmf(W2*t2 - b2, [c1,0]);
        t4 = sigmf(W3*t3 - b3, [1,0]);
        [~, idx] = max(t4);  
        test_accuracy(i/save_every) = sum(idx' == yt + 1)/mt;
    end
    
    % Backpropagation
    d4 = a3 - id_mat(:, y+1);
    d3 = d4.*c1.*a3.*(1-a3);
    d2 = (W3'*d3).*c1.*a2.*(1-a2);
    d1 = (W2'*d2).*c1.*a1.*(1-a1);

    dW3 = (d3*a2')/m;
    dW2 = (d2*a1')/m;
    dW1 = (d1*x1')/m;
    db3 = -sum(d3, 2)/m;
    db2 = -sum(d2, 2)/m;
    db1 = -sum(d1, 2)/m;

    % get eta based on epoch
    eta = get_exp_eta(i, eta_max, eta_min, fact);
    
    % Weight and bias update    
    W3 = W3 - eta*dW3;
    W2 = W2 - eta*dW2;
    W1 = W1 - eta*dW1;
    b3 = b3 - eta*db3;
    b2 = b2 - eta*db2;
    b1 = b1 - eta*db1;
end
