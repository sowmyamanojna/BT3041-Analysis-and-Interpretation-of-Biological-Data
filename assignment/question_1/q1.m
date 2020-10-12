clc;
clear;
load mnist_data.mat

x1 = double(trainImages')/255;
y = double(trainLabels');
t1 = double(testImages')/255;
yt = double(testLabels');

m = 800;
mt = 200;

eta_max = 0.3;
eta_min = 0.01;
epochs = 6000;
fact = 5000;
test_size = 200;
train_size = 800;
c1 = 10;
c2 = 0;
save_every = 10;


% Initialize important parameters
n1 = size(x1,1);
n2 = 512;
n3 = 512;
n4 = 10;

rng('default');

W1 = zeros(n2, n1);
W2 = zeros(n3, n2);
W3 = zeros(n4, n3);

b1 = rand(n2, 1);
b2 = rand(n3, 1);
b3 = rand(n4, 1);

id_mat = eye(n4);
loss_hist = zeros(ceil(epochs/save_every),1);
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
        
        loss = sum(sum((a3 - id_mat(:,y+1)).^2))/m;
        loss_hist(i/save_every) = loss;
    end

    % Backward pass
    delta4 = a3 - id_mat(:, y+1);
    delta3 = delta4.*a3.*(1-a3);
    delta2 = (W3'*delta3).*c1.*a2.*(1-a2);
    delta1 = (W2'*delta2).*c1.*a1.*(1-a1);

    dW3 = (delta3*a2')/m;
    dW2 = (delta2*a1')/m;
    dW1 = (delta1*x1')/m;
    db3 = -sum(delta3, 2)/m;
    db2 = -sum(delta2, 2)/m;
    db1 = -sum(delta1, 2)/m;

    % get eta based on epoch
    eta = get_exp_eta(i, eta_max, eta_min, fact);
    
    W3 = W3 - eta*dW3;
    W2 = W2 - eta*dW2;
    W1 = W1 - eta*dW1;
    b3 = b3 - eta*db3;
    b2 = b2 - eta*db2;
    b1 = b1 - eta*db1;
end

figure;
plot(1:save_every:epochs, 100*train_accuracy);
hold on;
plot(1:save_every:epochs, 100*test_accuracy)
plot(1:save_every:epochs, 100*(1-train_accuracy))
plot(1:save_every:epochs, 100*(1-test_accuracy))
hold off;
legend('Train accuracy', 'Test accuracy', 'Train mis-classification', 'Test mis-classification', 'Location', 'best')
xlabel('Epoch Number')
ylabel('Accuracy')
title("Accuracy across Epochs")

figure;
plot(1:save_every:epochs, loss_hist);
xlabel('Epoch Number')
ylabel('Loss');
title('Loss across Epochs');

accuracy_train = train_accuracy(end)*100;
fprintf('Train Accuracy : %f\n', accuracy_train);

accuracy_test = test_accuracy(end)*100;
fprintf('Test Accuracy : %f\n', accuracy_test);

%% Get Exponential decay eta function
function eta = get_exp_eta(epochs, maximum, minimum, fact)
    if epochs < fact
        eta = minimum + (maximum - minimum)*exp(-epochs/(fact/2));
    else
        eta = minimum;
    end
end

%% Function to plot images
% x1 is the image matrix of size (n*n, m)
% n is assumed to be equal to 28 here
% y is the lable vector

function plot_images(x1, y)
    for i = 1:50:800
        image = reshape(x1(:,i), [28,28])';
        figure;
        imshow(image);
        str = string(y(i)) + ' idx:' + string(i);
        title(str);
    end
end