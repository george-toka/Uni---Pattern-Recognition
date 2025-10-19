clear;
clc;
close all;

%--------- a ---------
% Dimensions - Chosen Characteristics
d = 2;
x1 = linspace(-5, 5, 100);
x2 = linspace(-5, 5, 100);

% Create the grid of all possible combinations
[X1, X2] = meshgrid(x1, x2);
X = [X1(:), X2(:)];

% Normal distribution mean value and covariance matrix
mu = [0 0];           % Mean for class ω1
Sigma = [1 0.3; 0.3 0.5]; % Covariance for class ω1
P = 0.5;

% Classes' PDFs
Npdf = mvnpdf(X, mu, Sigma);
Nplot = reshape(Npdf,length(x2),length(x1));

% Plot the PDF of a class
figure;
surf(x1,x2,Nplot);
hold on;
xlabel('x1');
ylabel('x2');
zlabel('Probability Density');
hold off;

% Discriminant function
g = discriminant_function(X, mu, Sigma, P);

% Euclidean distance between arbitrary points
a = [x1(35) x2(40)];
b = [x1(10) x2(20)];
euclid_dist = euclidean(a, b);

disp("The euclidean distance between a & b is: " + string(euclid_dist))

% Mahalanobis distance from point a to distribution
mahalanobis_dist = mahalanobis(b, mu, Sigma);
disp("The Mahalanobis distance from point b to the distribution is: " + string(mahalanobis_dist));

%---------b---------

% Import data
data = csvread('data.csv',1,0);

% Separate features and class labels
X = data(:, 1:3); % Features in the first 3 columns
Y = data(:, 4);   % Class labels in the 4th column

% Identify unique classes
classes = unique(Y);
num_classes = length(classes);

% Separate features based on class
X_class1 = X(Y==classes(1),:);
X_class2 = X(Y==classes(2),:);
X_class3 = X(Y==classes(3),:);

Y_class1 = Y(Y==classes(1));
Y_class2 = Y(Y==classes(2));
Y_class3 = Y(Y==classes(3));

% equal amount of data for all classes so it stays constant
N = size(X_class1, 1); 

% Find means and covariances using 1,2 and all features
all_means = cell(1,3); % cell to store arrays of varying dimensions
all_covs = cell(1,3); % essentially a cell of cells
for m = 1:3
    mask = [1:m];
    num_feats = length(mask);
    % First 70 samples serve as training data
    x = [X_class1(1:70,mask), X_class2(1:70,mask), X_class3(1:70,mask)];
    means = meansML(x, num_classes, num_feats, N);
    covs = covsML(x, means, num_classes, num_feats, N);
    all_means{m} = means;
    all_covs{m} = covs;
end

%--------- c & d ---------
P1 = 1/2; 
P2 = 1/2;
P = [P1 P2];
num_classes = length(P);
errors = [];
test_labels = [Y_class1(71:100); Y_class2(71:100)];
M = length(test_labels);

% Loop through each class for testing
for m = 1:3  
    error_count = 0;
    mask = [1:m];
    test_data = [X_class1(71:100, mask); X_class2(71:100, mask)];
    % Loop through each test sample
    for i = 1:M
        g_vals = [];
        for c = 1:num_classes    
            g = discriminant_function(test_data(i,:), all_means{m}(c), all_covs{m}{c}, P(c));
            g_vals = [g_vals, g];
        end
        % index of g_values that will return is identical to the class
        % indices
        [~, prediction] = max(g_vals);
        if prediction ~= test_labels(i)
            error_count = error_count + 1;
        end      
    end
    errors = [errors; error_count];
    disp("Classification Error for the first two classes using (" + string(m) + ") features is : "...
        + string(error_count/M));
end

%--------- e ---------

m = 3;
P1 = 0.8; 
P2 = 0.1;
P3 = 0.1;
P = [P1 P2 P3];
num_classes = length(P);
error_count = 0;
mask = [1:m];
test_data = [X_class1(71:100, mask); X_class2(71:100, mask); X_class3(71:100, mask)];
test_labels = [Y_class1(71:100); Y_class2(71:100); Y_class3(71:100)];
M = length(test_labels);

% Loop through each test sample
for i = 1:M
    g_vals = [];
    for c = 1:num_classes    
        g = discriminant_function(test_data(i,:), all_means{m}(c), all_covs{m}{c}, P(c));
        g_vals = [g_vals, g];
    end
    % index of g_values that will return is identical to the class
    % indices
    [~, prediction] = max(g_vals);
    if prediction ~= test_labels(i)
        error_count = error_count + 1;
    end      
end
errors = [errors; error_count];
disp("Classification Error for all classes using (" + string(m) + ") features is : "...
    + string(error_count/M));


%-------- Functions ----------
function  g = discriminant_function(x, mu, Sigma, P)
    d = length(mu);
    inv_Sigma = inv(Sigma); 
    ln_det_Sigma = log(det(Sigma)); 
        
    g = -0.5 * ((x - mu) * inv_Sigma * (x - mu)') ...
        - (d / 2) * log(2 * pi) - 0.5 * ln_det_Sigma + log(P);
end

function dr = euclidean(x1, x2) 
    dr = sqrt(sum((x1-x2).^2));
end

function dr = mahalanobis(x, mu, Sigma)
    inv_Sigma = inv(Sigma);
    dr = sqrt((x - mu) * inv_Sigma * (x - mu)');
end

function means = meansML(x, num_classes, num_feats, N) 
    means = zeros(num_classes, num_feats);
    for i = 1:num_classes
        for j = 1:num_feats
            idx = (i-1)*num_feats + j;
            means(i,j) = sum(x(:,idx))/N;
        end
    end
end

function covs = covsML(x, means, num_classes, num_feats, N)
    % Initialize covariance cell array for each class and feature subset
    covs = cell(1, num_classes);
    for i = 1:num_classes
        % Select data for the current class and calculate deviations
        x_class = x(:, (i-1)*num_feats+1:i*num_feats);
        mean_vec = means(i, :);
        
        % Calculate covariance matrix for the class
        cov_matrix = (x_class - mean_vec)' * (x_class - mean_vec) / N;
        covs{i} = cov_matrix;
    end
end