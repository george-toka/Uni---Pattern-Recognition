clear;
clc;
close all;

% -------------- Parzen Estimator --------------
data = readmatrix('Data_exercise1.csv');
X = data(:, 1:end-1);
Y = data(:, end);

% For last question regarding 25% of the initial dataset
% X = [data(1:25, 1:end-1); data(101:125, 1:end-1); data(201:225, 1:end-1)];
% Y = [data(1:25, end); data(101:125, end); data(201:225, end)];

Xraw = X; % Keep a copy of the raw data

% Feature-wise normalization to avoid bias
for i = 1:size(X,2)
        X(:,i) = ( X(:, i) - min(X(:, i)) )...
            / ( max(X(:, i)) - min(X(:, i)) );
end

X_norm = X; % Keep a copy of the normalised data

X_w1 = X(Y==1,:); X_w2 = X(Y==2, :); X_w3 = X(Y==3, :);
d = size(X, 2);
num_classes = length(unique(Y));

phi = @(x, hn) (1 / (hn*sqrt(2*pi)) * exp(-x.^2/(2*hn^2)));
h = [0.3, 0.7, 0.1, 1.5];

x1 = linspace(min(X(:, 1))-1, max(X(:, 1))+1, 100);
x2 = linspace(min(X(:, 1))-1, max(X(:, 2))+1, 100);
[X1, X2] = meshgrid(x1,x2);
grid_points = [X1(:), X2(:)];

X = [X_w1, X_w2, X_w3];  % Horizontal stack of the features of each class
parzen_pdfs = cell(length(h),num_classes); % Store all PDFs for all hn values
n = size(X_w1,1); % Number of samples - equal # of them for every class

for N = 1:length(h)
    hn = h(N);          
    Vn = hn^d;         
  
    for c = 1:num_classes
        mask = [d*(c-1)+1 : d*c];
        class_samples = X(:, mask); % Samples for the current classs
        
        % Estimate density at each grid point
        density = zeros(size(grid_points, 1), 1);
        
        % Calculate Parzen window density for each grid point
        for i = 1:size(grid_points, 1)
            distances = vecnorm(grid_points(i, :) - class_samples, 2, 2);
            kernel_values = phi(distances, hn); 
            density(i) = (1 / (n * Vn)) * sum(kernel_values);
        end
        
        % Reshape density to match the grid shape for plotting
        density_grid = reshape(density, size(X1));
        parzen_pdfs{N, c} = density_grid; 
    end
    
    % Classify each grid point based on highest estimated PDF
    decision_map = zeros(size(X1));
    for i = 1:numel(X1)
        [~, decision_map(i)] = max(cellfun(@(pdf) pdf(i), parzen_pdfs(N, :)));
    end
    
    % Plot Parzen Decision Regions
    figure;
    gscatter(X1(:), X2(:), decision_map(:), 'rgb', '.', 1);
    hold on;
    scatter(X_norm(Y==1,1), X_norm(Y==1,2), 50, 'r', 'filled');
    scatter(X_norm(Y==2,1), X_norm(Y==2,2), 50, 'g', 'filled');
    scatter(X_norm(Y==3,1), X_norm(Y==3,2), 50, 'b', 'filled');
    title(['Parzen Decision Regions with h = ', num2str(h(N))]);
    xlabel('x1');
    ylabel('x2');
    legend('Class 1', 'Class 2', 'Class 3', 'Location', 'Best');
    hold off;
end

colors = ['r', 'g', 'b']; % Colors for each class

% Plotting PDF estimates
for N = 1:length(h)
    figure;
    hold on;
    title(['Parzen Density Estimation with h = ', num2str(h(N))]);
    xlabel('x1');
    ylabel('x2');
    zlabel('Density');
    
    % Plot each class density in 3D
    for c = 1:num_classes    
        surf(X1, X2, parzen_pdfs{N, c},'FaceColor', colors(c), 'FaceAlpha', 0.5, 'EdgeColor', 'none', ...
             'DisplayName', ['Class ', num2str(c)]);
    end
    
    legend show;
    view(3); % Set view to 3D
    hold off;
end

%-------------- KNN Estimator --------------
X = Xraw;

% Feature-wise normalization to avoid bias
for i = 1:size(X,2)
        X(:,i) = ( X(:, i) - min(X(:, i)) )...
            / ( max(X(:, i)) - min(X(:, i)) );
end

k_values = [10, 3, 30, 8];
volume_constant = pi^(d/2) / gamma(d/2 + 1);
knn_pdfs = cell(length(k_values), num_classes);

for idx = 1:length(k_values)
    k_i = k_values(idx); 
    pdf_estimates = zeros(size(grid_points, 1), num_classes);

    for c = 1:num_classes
        class_samples = X(Y == c, :);
        n_i = size(class_samples, 1);

        for i = 1:size(grid_points, 1)
            distances = vecnorm(grid_points(i, :) - class_samples, 2, 2);
            sorted_distances = sort(distances);
            r_ki = sorted_distances(k_i);
            V_ki = volume_constant * r_ki^d;
            pdf_estimates(i, c) = k_i / (n_i * V_ki);
        end
    end
    
    knn_pdfs{idx} = pdf_estimates;
    
    % Classify each grid point based on highest estimated PDF
    [~, decision_map] = max(pdf_estimates, [], 2);
    
    % Plot kNN Decision Regions
    figure;
    gscatter(X1(:), X2(:), decision_map, 'rgb', '.', 1);
    hold on;
    scatter(X(Y==1,1), X(Y==1,2), 50, 'r', 'filled');
    scatter(X(Y==2,1), X(Y==2,2), 50, 'g', 'filled');
    scatter(X(Y==3,1), X(Y==3,2), 50, 'b', 'filled');
    title(['k-NN Decision Regions with k = ', num2str(k_i)]);
    xlabel('x1');
    ylabel('x2');
    legend('Class 1', 'Class 2', 'Class 3', 'Location', 'Best');
    hold off;
end

% Plotting PDF Estimates
colors = {'r', 'g', 'b'};
for idx = 1:length(k_values)
    k_i = k_values(idx);
    figure;
    for c = 1:num_classes
        % Reshape the density estimates for class c into grid shape
        density_grid = reshape(knn_pdfs{idx}(:, c), size(X1));
        
        % Plot the density estimate for class c    
        hold on;
        surf(X1, X2, density_grid, 'FaceColor', colors{c}, 'FaceAlpha', 0.5, ...
             'EdgeColor', 'none', 'DisplayName', ['Class ', num2str(c)]);     
        title(['Knn Density Estimation with k = ', num2str(k_i)]);
        xlabel('x1');
        ylabel('x2');
        zlabel('Density');
        legend show;
        view(3); % Set view to 3D
        hold off;
    end
end
