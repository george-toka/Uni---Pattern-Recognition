clear;
clc;
close all;

% First attribute of the file is the label
data = importdata('wine/wine.data');

X = data(:, 2:6); % Use only the first 5 features as specified
Y = data(:, 1);   % Class labels (1, 2, or 3)

% Select samples from classes 2 and 3 only
X = X(Y ~= 1, :);
Y = Y(Y ~= 1);
Y(Y == 2) = -1; % Relabel class 2 as -1
Y(Y == 3) = 1;  % Relabel class 3 as +1

% Split data into training, validation, and test sets (50%-25%-25%)
num_samples = length(Y);
idx = randperm(num_samples);

train_idx = idx(1:round(0.5 * num_samples));
val_idx = idx(round(0.5 * num_samples) + 1:round(0.75 * num_samples));
test_idx = idx(round(0.75 * num_samples) + 1:end);

X_train = X(train_idx, :);
Y_train = Y(train_idx);

X_val = X(val_idx, :);
Y_val = Y(val_idx);

X_test = X(test_idx, :);
Y_test = Y(test_idx);

% Initialize possible values of the box constraint parameter C
C_values = logspace(-3, 3, 10);
best_C = C_values(1);
best_accuracy = 0;

% Train and validate SVM with different values of C
for C = C_values
    SVMModel = fitcsvm(X_train, Y_train, 'KernelFunction', 'linear', 'BoxConstraint', C);
    
    % Validate the model
    predictions = predict(SVMModel, X_val);
    accuracy = sum(predictions == Y_val) / length(Y_val);
    
    % Update best C if current accuracy is higher
    if accuracy > best_accuracy
        best_accuracy = accuracy;
        best_C = C;
    end
end

% Train final model on training set using the best C and evaluate on test set
SVMModel_final = fitcsvm(X_train, Y_train, 'KernelFunction', 'linear', 'BoxConstraint', best_C);
predictions_test = predict(SVMModel_final, X_test);
test_accuracy = sum(predictions_test == Y_test) / length(Y_test);

fprintf('Best C: %f\n', best_C);
fprintf('Test Accuracy: %.2f%%\n', test_accuracy * 100);

% Test SVM with RBF kernel
SVMModel_rbf = fitcsvm(X_train, Y_train, 'KernelFunction', 'rbf', 'BoxConstraint', best_C);
predictions_rbf = predict(SVMModel_rbf, X_test);
accuracy_rbf = sum(predictions_rbf == Y_test) / length(Y_test);

% Test SVM with Polynomial kernel
SVMModel_poly = fitcsvm(X_train, Y_train, 'KernelFunction', 'polynomial', 'BoxConstraint', best_C);
predictions_poly = predict(SVMModel_poly, X_test);
accuracy_poly = sum(predictions_poly == Y_test) / length(Y_test);

fprintf('Test Accuracy (RBF Kernel): %.2f%%\n', accuracy_rbf * 100);
fprintf('Test Accuracy (Polynomial Kernel): %.2f%%\n', accuracy_poly * 100);

% Multi-class classification using one-vs-one approach
SVMModel_multi_one_vs_one = fitcecoc(X, Y, 'Coding', 'onevsone');
predictions_multi_one_vs_one = predict(SVMModel_multi_one_vs_one, X_test);

% Confusion matrix for one-vs-one approach
confusion_matrix_one_vs_one = confusionmat(Y_test, predictions_multi_one_vs_one);
disp('Confusion Matrix (One-vs-One):');
disp(confusion_matrix_one_vs_one);

% Multi-class classification using one-vs-all approach
SVMModel_multi_one_vs_all = fitcecoc(X, Y, 'Coding', 'onevsall');
predictions_multi_one_vs_all = predict(SVMModel_multi_one_vs_all, X_test);

% Confusion matrix for one-vs-all approach
confusion_matrix_one_vs_all = confusionmat(Y_test, predictions_multi_one_vs_all);
disp('Confusion Matrix (One-vs-All):');
disp(confusion_matrix_one_vs_all);

% Calculate and display accuracy for each approach
accuracy_one_vs_one = sum(predictions_multi_one_vs_one == Y_test) / length(Y_test);
accuracy_one_vs_all = sum(predictions_multi_one_vs_all == Y_test) / length(Y_test);

fprintf('Test Accuracy (One-vs-One): %.2f%%\n', accuracy_one_vs_one * 100);
fprintf('Test Accuracy (One-vs-All): %.2f%%\n', accuracy_one_vs_all * 100);

