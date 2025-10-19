clear;
clc;
close all;

data = importdata('wine/wine.data');

% -------------TASK A - D / Generalized Split for N Classes--------------
dataA = data(data(:,1)~=1, 1:6);
X = dataA(:, 2:end);
Y = dataA(:, 1);

best_Cs = zeros(1,6);
test_accuracies = zeros(1,6);
for m=1:6
    rng(m);
    [X_train, Y_train, X_val, Y_val, X_test, Y_test] = dataSplitter(X,Y);

    % Initialize possible values of the box constraint parameter C
    C_values = linspace(0.001, 0.2, 500);
    best_Cs(m) = C_values(1);
    best_accuracy = 0;

    % Train and validate SVM with different values of C
    for C = C_values
        SVMModel = fitcsvm(X_train, Y_train, 'KernelFunction', 'rbf', 'BoxConstraint', C);

        % Validate the model
        predictions = predict(SVMModel, X_val);
        accuracy = sum(predictions == Y_val) / length(Y_val);

        % Update best C if current accuracy is higher
        if accuracy > best_accuracy
            best_accuracy = accuracy;
            best_Cs(m) = C;
        end
    end

    % Train final model on training set using the best C and evaluate on test set
    SVMModel_final = fitcsvm(X_train, Y_train, 'KernelFunction', 'rbf', 'BoxConstraint', best_Cs(m));
    predictions_test = predict(SVMModel_final, X_test);
    test_accuracy = sum(predictions_test == Y_test) / length(Y_test);
    
    fprintf('Best C: %f\n', best_Cs(m));
    fprintf('Test %d Accuracy: %.2f%%\n', m, test_accuracy * 100);
    
    test_accuracies(m) = test_accuracy;
end

errors = 1-test_accuracies;
mean_error = sum(errors) / length(errors);
std_error = sqrt(sum((errors - mean_error).^2) / length(errors));

fprintf('Mean error is: %f\n', mean_error);
fprintf('Standard Deviation of error is: %f\n', std_error);


% -------------TASK E---------------

% Separate features and labels
X_all = data(:, 2:end);
Y_all = data(:, 1);

feature_sets = {X_all(:, 1:5), X_all}; % First 5 features and all features

% Set the box constraint for the SVM
C = 1;

mean_errors = zeros(1, 2);
confusion_matrices = cell(1, 2);

for f = 1:2
    X = feature_sets{f};
    
    % Perform 5-fold cross-validation
    kfold = 5;
    indices = crossvalind('Kfold', Y_all, kfold); % mark the samples of the kth run with index k
    fold_errors = zeros(kfold, 1);
    conf_matrix_total = zeros(3, 3); % Initialize total confusion matrix for each feature set

    for k = 1:kfold
        % Split the data
        valIdx = (indices == k);
        trainIdx = ~valIdx;
        
        X_train = X(trainIdx, :);
        Y_train = Y_all(trainIdx);
        X_val = X(valIdx, :);
        Y_val = Y_all(valIdx);

        % One-vs-One SVM training with majority voting
        classes = unique(Y_train);
        num_classes = length(classes);
        classifiers = cell(num_classes);

        % Train one SVM per pair of classes
        for i = 1:num_classes
            for j = i+1:num_classes
                % Select data for the two classes
                class1 = classes(i);
                class2 = classes(j);
                idx = (Y_train == class1) | (Y_train == class2); % choose samples only from the two classes
                X_pair = X_train(idx, :);
                Y_pair = Y_train(idx);
                
                % Train SVM
                classifiers{i, j} = fitcsvm(X_pair, Y_pair, 'KernelFunction', 'linear', 'BoxConstraint', C);
            end
        end

        % Majority voting on the test set
        votes = zeros(length(Y_val), num_classes);
        for i = 1:num_classes
            for j = i+1:num_classes
                % Make predictions with the classifier for the class pair (i, j)
                class1 = classes(i);
                class2 = classes(j);
                model = classifiers{i, j};
                
                % Predict for the test set
                pair_predictions = predict(model, X_val); % for the classifier ij
                
                % Convert predictions to voting format
                votes(pair_predictions == class1, i) = votes(pair_predictions == class1, i) + 1;
                votes(pair_predictions == class2, j) = votes(pair_predictions == class2, j) + 1;
            end
        end

        % Final predictions based on majority vote
        [~, final_predictions] = max(votes, [], 2);
        final_predictions = classes(final_predictions);

        % Calculate confusion matrix and accuracy
        conf_matrix = confusionmat(Y_val, final_predictions);
        conf_matrix_total = conf_matrix_total + conf_matrix;
        fold_errors(k) = 1 - sum(diag(conf_matrix)) / sum(conf_matrix(:));
    end

    % Calculate mean error and store confusion matrix
    mean_errors(f) = mean(fold_errors);
    confusion_matrices{f} = conf_matrix_total;
    
    % Display results
    fprintf('Feature set %d:\n', f);
    fprintf('Mean error: %.4f\n', mean_errors(f));
    fprintf('Confusion matrix:\n');
    disp(conf_matrix_total);
end



function [X_train, Y_train, X_val, Y_val, X_test, Y_test] = dataSplitter(X,Y)
    % Get unique classes
    classes = unique(Y);
    num_classes = length(classes);

    % Initialize structures to store the train, validation, and test sets
    data_split = struct();

    % Loop through each class and split data accordingly
    for i = 1:num_classes
        class_label = classes(i);

        % Extract samples for the current class
        X_class = X(Y == class_label, :);
        Y_class = Y(Y == class_label);
        num_samples_class = length(Y_class);

        % Randomize indices
        idx = randperm(num_samples_class);

        % Split indices for train, validation, and test sets (50%-25%-25%)
        train_idx = idx(1:round(0.5 * num_samples_class));
        val_idx = idx(round(0.5 * num_samples_class) + 1:round(0.75 * num_samples_class));
        test_idx = idx(round(0.75 * num_samples_class) + 1:end);

        % Store data in the structure
        data_split(i).class = class_label;
        data_split(i).X_train = X_class(train_idx, :);
        data_split(i).Y_train = Y_class(train_idx);
        data_split(i).X_val = X_class(val_idx, :);
        data_split(i).Y_val = Y_class(val_idx);
        data_split(i).X_test = X_class(test_idx, :);
        data_split(i).Y_test = Y_class(test_idx);
    end

    % Concatenate training, validation, and test sets from each class
    X_train = vertcat(data_split(:).X_train);
    Y_train = vertcat(data_split(:).Y_train);
    X_val = vertcat(data_split(:).X_val);
    Y_val = vertcat(data_split(:).Y_val);
    X_test = vertcat(data_split(:).X_test);
    Y_test = vertcat(data_split(:).Y_test);
end
