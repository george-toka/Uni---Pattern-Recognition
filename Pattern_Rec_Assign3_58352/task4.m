clear;
clc;
close all;

% Load the distance matrix and city names
distance_matrix = importdata('Distance_Matrix_US.txt'); 
city_names = importdata('City_names_US.txt');            

% Number of cities
num_cities = size(distance_matrix, 1);

% A. Classical MDS for 2 and 3 dimensions
[coords_2d, ~] = cmdscale(distance_matrix, 2); % 2D representation
[coords_3d, ~] = cmdscale(distance_matrix, 3); % 3D representation

% Plot the 2D representation
figure;
scatter(coords_2d(:, 1), coords_2d(:, 2), 50, 'filled');
text(coords_2d(:, 1), coords_2d(:, 2), city_names);
title('Classical MDS (2D)');
xlabel('Dimension 1');
ylabel('Dimension 2');
grid on;

% Plot the 3D representation
figure;
scatter3(coords_3d(:, 1), coords_3d(:, 2), coords_3d(:, 3), 50, 'filled');
text(coords_3d(:, 1), coords_3d(:, 2), coords_3d(:, 3), city_names);
title('Classical MDS (3D)');
xlabel('Dimension 1');
ylabel('Dimension 2');
zlabel('Dimension 3');
grid on;

% Full dimensional representation and eigenvalues
[full_coords, eigvals] = cmdscale(distance_matrix);

% Plot eigenvalues in descending order
figure;
plot(eigvals, '-o', 'LineWidth', 1.5);
title('Eigenvalues of Y Y^T');
xlabel('Index');
ylabel('Eigenvalue');
grid on;


