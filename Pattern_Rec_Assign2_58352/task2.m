clear;
clc;
close all;

% Set random seed for reproducibility
%rng(7);

mu1 = [2; 2];
Sigma1 = [2 -0.5; -0.5 1];
mu2 = [-8; 2];
Sigma2 = [1 0.5; 0.5 1];

% Generate 150 samples for each class
num_samples = 150;
data_class1 = mvnrnd(mu1, Sigma1, num_samples);
data_class2 = mvnrnd(mu2, Sigma2, num_samples);

data = [data_class1; data_class2];
labels = [ones(num_samples, 1); -ones(num_samples, 1)];

% Plot the generated samples
scatterplot(data_class1, data_class2, 0, 0, [0 0], '', 'Sample points', 'ignore')

x_vals = linspace(min(data(:,1)), max(data(:,1)), 100);

% Train Batch Perceptron
w_perceptron = batch_perceptron(data, labels, 0.1, 1000);
% solving for g(x)=0 and solving for one of the features
y_vals = -(w_perceptron(1) * x_vals + w_perceptron(3)) / w_perceptron(2); 
scatterplot(data_class1, data_class2, x_vals, y_vals, [0 0], 'g', 'Batch Perceptron', 'ignore')

% Train Ho-Kashyap
w_hk = ho_kashyap(data, labels, 100000, 2, 0.1);
y_vals_hk = -(w_hk(1) * x_vals + w_hk(3)) / w_hk(2);
scatterplot(data_class1, data_class2, x_vals, y_vals_hk, [0 0], 'g', 'Ho-Kashyap', 'ignore')

% Train SVM
SVMModel = fitcsvm(data, labels, 'KernelFunction', 'linear', 'Standardize', false);
% Plot decision boundary for SVM
sv = SVMModel.SupportVectors;
y_vals_svm = -(SVMModel.Beta(1) * x_vals + SVMModel.Bias) / SVMModel.Beta(2);
scatterplot(data_class1, data_class2, x_vals, y_vals_svm, sv, 'g', 'SVM', 'Support Vectors')



% Batch Perceptron algorithm
function w = batch_perceptron(data, labels, eta, max_iter)
    [num_samples, num_features] = size(data);
    w = zeros(num_features + 1, 1); % Initialize weights (including bias w0)
    data = [data, ones(num_samples, 1)]; % Add bias term - augmented feature vector - y

    for iter = 1:max_iter
        misclassified = false;
        for i = 1:num_samples
            if labels(i) * (data(i, :) * w) <= 0 % if sample is missclassified
                w = w + eta * (labels(i) * data(i, :)'); % x for class 1, -x for class 2
                misclassified = true;
            end
        end
        if ~misclassified % if accuracy is 100% terminate loop
            disp('Batch  Perceptron terminated in iteration: ' + string(iter))
            break;
        end
    end
end

% Ho-Kashyap algorithm
function w = ho_kashyap(data, labels, max_iter, b0, eta)
    [num_samples, num_features] = size(data);
    data = [data, ones(num_samples, 1)]; % Add bias term
    Y = labels .* data; % Combine labels with data
    b = b0 * ones(num_samples, 1); % Initialize margin (sttrictly positive)
    w = inv(Y' * Y) * Y' * b;
    
    for iter = 1:max_iter
        e = Y * w - b;
        eplus = 1/2 * (e + abs(e));
        b = b + eta * (e + abs(e)); % Update margin
        w = inv(Y' * Y) * Y' * b; %

        % Terminate if convergence criterion is met
        if max(abs(e)) < 1e-0
            disp('Ho-Kashyap terminated in iteration: ' + string(iter))
            break;
        end
    end
end

function scatterplot(data_class1, data_class2, x_vals, y_vals, sv, color, header, otherheader)
    figure;
    hold on;
    scatter(data_class1(:,1), data_class1(:,2), 'r', 'filled');
    scatter(data_class2(:,1), data_class2(:,2), 'b', 'filled');
    xlabel('x1'); ylabel('x2');
    legend('Class ω1', 'Class ω2');
    plot(x_vals, y_vals, color, 'DisplayName', header);
    plot(sv(:,1), sv(:,2), 'ko', 'MarkerSize', 10, 'DisplayName', otherheader);
    title(header)
    grid on;
    hold off;
end


