clear;
clc;
close all;

data = importdata("seeds_dataset.txt");
X = data(:, 1:end-1); 
Y = data(:, end);     

% Hierarchical Agglomerative Clustering
% Compute the linkage matrix
linkage_method = 'average';
Z = linkage(X, linkage_method);

% Plot the dendrogram
figure;
dendrogram(Z, size(X,1));
title('Dendrogram of Agglomerative Hierarchical Clustering');
xlabel('Sample Index');
ylabel('Distance');

% 4. Form flat clusters from the dendrogram
numClusters = numel(unique(Y)); % Number of true clusters
clusterLabels_hier = cluster(Z, 'maxclust', numClusters, 'Criterion','distance');

% Plot the dendrogram with the 3 clusters colored
figure;
cutoff = median([Z(end-2,3) Z(end-1,3)]);
dendrogram(Z, size(X,1),'ColorThreshold',cutoff)
title('Dendrogram of Agglomerative Hierarchical Clustering');
xlabel('Sample Index');
ylabel('Distance');

% Compute Rand Index for the Agglomerative Clustering
RI_hierarchical = calculate_rand_index(Y, clusterLabels_hier);

% K-means Clustering
[idxKMeans, ~] = kmeans(X, numClusters);

% Compute Rand Index for K-means Clustering
RI_kmeans = calculate_rand_index(Y, idxKMeans);

% Display the Rand Index results
fprintf('Rand Index (Hierarchical Clustering): %.4f\n', RI_hierarchical);
fprintf('Rand Index (K-means Clustering): %.4f\n', RI_kmeans);

% Rand Index Function
function ri = calculate_rand_index(true_labels, predicted_labels)
    % Calculate Rand Index between true and predicted labels
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

    ri = (a + b) / ((n * (n - 1)) / 2);
end
