clear;
clc;
close all;

data = importdata("seeds_dataset.txt");
X = data(:, 1:end-1); 
Y = data(:, end);    

[N, D] = size(data);

% Pairwise distances (NxN symmetric matrix with 0 diagonal)
euclid = zeros(N);
cosine = zeros(N);
for i = 1:N
    for j = i+1:N
        % Euclidean Distance
        res = sqrt(sum((X(i, :) - X(j, :)).^2));
        euclid(i, j) = res;
        euclid(j, i) = res;
        
        % Cosine Distance
        dot_product = dot(X(i, :), X(j, :));      
        norm_i = norm(X(i, :));                   
        norm_j = norm(X(j, :));                  
        cosine_dist = 1 - (dot_product / (norm_i * norm_j)); %the second term is cosine similarity
        cosine(i, j) = cosine_dist;
        cosine(j, i) = cosine_dist;               
    end
end

% Visualise Distance Matrices
figure;
heatmap(euclid);
title('Euclidean Distance');
xlabel('Seeds');
ylabel('Seeds');

figure;
heatmap(cosine);
title('Cosine Distance');
xlabel('Seeds');
ylabel('Seeds');

% Silhouette Analysis
k_values = 2:10;
silhouette_analysis(X, k_values, 'sqeuclidean');
silhouette_analysis(normalize_data(X), k_values, 'cosine');

% Rand Index for k-means with squared Euclidean metric
num_clusters = 3;
[mean_ri, var_ri] = compute_rand_index(X, Y, num_clusters, 'sqeuclidean');
fprintf('Mean Rand Index (Squared Euclidean): %.4f\n', mean_ri);
fprintf('Variance of Rand Index (Squared Euclidean): %.4f\n', var_ri);

% Rand Index for k-means with cosine metric
[mean_ri_cos, var_ri_cos] = compute_rand_index(X, Y, num_clusters, 'cosine');
fprintf('Mean Rand Index (Cosine): %.4f\n', mean_ri_cos);
fprintf('Variance of Rand Index (Cosine): %.4f\n', var_ri_cos);


function silhouette_analysis(X, k_values, distance_metric)
    % Silhouette analysis for given distance metric
    silhouette_means = zeros(length(k_values), 1);

    for i = 1:length(k_values)
        k = k_values(i);
        [labels, ~] = kmeans(X, k, 'Distance', distance_metric);
        silhouette_values = silhouette(X, labels, distance_metric);
        silhouette_means(i) = mean(silhouette_values);
    end
    
    % Plot the silhouette results
    figure;
    plot(k_values, silhouette_means, '-o', 'LineWidth', 2);
    xlabel('Number of Clusters (k)');
    ylabel('Mean Silhouette Coefficient');
    title(['Silhouette Coefficient (', distance_metric, ' Distance)']);
    grid on;

    % Display optimal k
    [~, optimal_idx] = max(silhouette_means);
    fprintf('Optimal k for %s metric: %d\n', distance_metric, k_values(optimal_idx));
end

function [mean_ri, var_ri] = compute_rand_index(X, Y, num_clusters, distance_metric)
    % Compute Rand Index over multiple runs
    num_runs = 5;
    rand_indices = zeros(num_runs, 1);

    for run = 1:num_runs
        predicted_labels = kmeans(X, num_clusters, 'Distance', distance_metric);
        rand_indices(run) = calculate_rand_index(Y, predicted_labels);
    end

    mean_ri = mean(rand_indices);
    var_ri = var(rand_indices);
end

function ri = calculate_rand_index(true_labels, predicted_labels)
    % Calculate the Rand Index between true and predicted labels
    n = length(true_labels);
    a = 0; b = 0;
    for i = 1:n-1
        for j = i+1:n
            same_true = (true_labels(i) == true_labels(j));
            same_pred = (predicted_labels(i) == predicted_labels(j));

            if same_true && same_pred
                a = a + 1; % True Positive
            elseif ~same_true && ~same_pred
                b = b + 1; % True Negative
            end
        end
    end

    ri = (a + b) / (n * (n-1) / 2);
end

function X_norm = normalize_data(X)
    % Normalize data to zero mean and unit variance
    mu = mean(X);
    sigma = std(X);
    X_norm = (X - mu) ./ sigma;
end

