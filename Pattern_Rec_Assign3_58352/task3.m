clear;
clc;
close all;

data = importdata("seeds_dataset.txt");
X = data(:, 1:end-1); 
Y = data(:, end);     

% Standardize the data
mu = mean(X);
sigma = std(X);
X = (X - mu) ./ sigma;

% Perform PCA
[coeff, score, latent, ~, explained] = pca(X);

% Calculate cumulative variance
cumulative_variance = cumsum(explained);
found80=0; , found995 = 0;
% Find the number of components for 80% and 99.5% variance
for i = 1:size(explained)
    if cumulative_variance(i) >= 80 && ~found80
        num_components_80 = i;
        found80 = 1;
    end
    if cumulative_variance(i) >= 99.5 && ~found995
        num_components_995 = i;
        found995 = 1;
    end
    if found80 && found995
        break
    end
end

fprintf('Number of components for 80%% variance: %d\n', num_components_80);
fprintf('Number of components for 99.5%% variance: %d\n', num_components_995);

% Reconstruction Error
reconstruction_error = zeros(1, 7);
for k = 1:7
    X_reconstructed = score(:, 1:k) * coeff(:, 1:k)';
    reconstruction_error(k) = mean(mean((X - X_reconstructed).^2));
end

% Plot reconstruction error
figure;
plot(1:7, reconstruction_error, '-o', 'LineWidth', 1.5);
xlabel('Number of Principal Components');
ylabel('Reconstruction Error');
title('Reconstruction Error vs. Number of Principal Components');
grid on;

% Compute Scatter Matrices to determine most significant discriminants
classes = unique(Y);
num_classes = length(classes);
num_features = size(X, 2);
overall_mean = mean(X);

S_W = zeros(num_features); % Within-class scatter matrix
S_B = zeros(num_features); % Between-class scatter matrix

for c = 1:num_classes
    class_data = X(Y == classes(c), :);
    class_mean = mean(class_data);
    % Within-class scatter
    S_W = S_W + (class_data - class_mean)' * (class_data - class_mean);
    % Between-class scatter
    n_c = size(class_data, 1);
    S_B = S_B + n_c * (class_mean - overall_mean)' * (class_mean - overall_mean);
end

% Solve generalized eigenvalue problem: S_B * w = lambda * S_W * w
[eigvecs, eigvals] = eig(S_B, S_W);
[~, idx] = sort(diag(eigvals), 'descend'); % Sort eigenvalues in descending order

% Select the two most discriminative linear discriminants
LD1 = eigvecs(:, idx(1));
LD2 = eigvecs(:, idx(2));

% Project the data onto the selected discriminants
X_lda = X * [LD1, LD2];
X_pca = score(:, 1:2);

% Plot LDA vs PCA
figure;
subplot(1, 2, 1);
gscatter(X_pca(:, 1), X_pca(:, 2), Y);
title('PCA Projection (2D)');
xlabel('PC1');
ylabel('PC2');
grid on;

subplot(1, 2, 2);
gscatter(X_lda(:, 1), X_lda(:, 2), Y);
title('LDA Projection (2D)');
xlabel('LD1');
ylabel('LD2');
grid on;

% Feature Contribution Analysis
feature_contributions = sum(abs([LD1, LD2]), 2);  % Sum along the columns (features) to get overall feature contribution

% Find the two most and least contributing features
[~, sorted_idx] = sort(feature_contributions, 'descend');
most_contributing_features = sorted_idx(1:2);
least_contributing_features = sorted_idx(end-1:end);

fprintf('Most contributing features: %d, %d\n', most_contributing_features);
fprintf('Least contributing features: %d, %d\n', least_contributing_features);

% 2D visualization using most contributing features
figure;
subplot(1, 2, 1);
gscatter(X(:, most_contributing_features(1)), X(:, most_contributing_features(2)), Y);
xlabel(sprintf('Feature %d', most_contributing_features(1)));
ylabel(sprintf('Feature %d', most_contributing_features(2)));
title('Most Contributing Features');
grid on;

% 2D visualization using least contributing features
subplot(1, 2, 2);
gscatter(X(:, least_contributing_features(1)), X(:, least_contributing_features(2)), Y);
xlabel(sprintf('Feature %d', least_contributing_features(1)));
ylabel(sprintf('Feature %d', least_contributing_features(2)));
title('Least Contributing Features');
grid on;
