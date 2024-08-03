function [eigenvectors, eigenvalues, projections, variances, indexes] = My_CDA(data,labels)
    % data: Matrix of observations (rows) by features (cols)
    % labels: Vector type "categorical" with the classes

    % GENERAL DATA DISPERSION  
    global_mean = mean(data)';  % Global mean
    [rows,cols] = size(data);   % Data size

    txt = strcat("There are n = ", string(rows), " observations in dimension p = ", string(cols));
    disp(txt)

    % DATA DISPERSION BY CLASS
    classes = unique(labels);
    n_classes = length(classes);
    means_matrix = zeros(n_classes,cols);          % Matrix of means
    W = zeros(cols,cols);                          % Within groups matrix 
    for i = 1:n_classes
        indexes = find(labels == classes(i));
        Matrix_Subclass_i = data(indexes,:);
        [classSize,~] = size(Matrix_Subclass_i);
        means_matrix(i,:) = mean(Matrix_Subclass_i);
        classesCovariance = cov(Matrix_Subclass_i);
        group_matrix_weighted = classSize * classesCovariance;
        W = W + group_matrix_weighted;
    end
    txt = strcat("There are g = ", string(n_classes), " classes.");
    disp(txt)
    %----------------------------------------------------------------------
    % BETWEEN-GROUPS DISPERSION
    X_bar = means_matrix - global_mean';
    A = X_bar' * X_bar;  % Between groups matrix

    % WITHIN-GROUPS DISPERSION
    S = (1/(rows - n_classes)) * W ; % Pooled within matrix

    %----------------------------------------------------------------------
    % CANONICAL AXES CONSTRUCTION
    [V,L] = eig(A,S);
    eigenvalues = diag(L)';
    [eigenvalues, indexes] = sort(eigenvalues, 'descend');
    eigenvectors = V(:,indexes);

    % CANONICAL AXES PROJECTIONS
    projections = data*eigenvectors;

    std_projections = std(projections);
    txt = "The standard desviation for canonical axis is presented in the following vector: ";
    disp(txt)
    disp(std_projections)
    m = min(cols,(n_classes-1));
    txt = strcat("The actual number of canonical axes is m = ", string(m), ".");
    disp(txt)
    percent =  sum(eigenvalues(1:m))/sum(eigenvalues) * 100;
    txt = strcat("The percentage of geometric variability explained by the first m = ", string(m), " canonical coordinates is: ", string(percent),"%.");
    disp(txt)
    %----------------------------------------------------------------------
    % Graphic if m is equal to 1, 2 or 3.

    variances = var(projections)/sum(var(projections))*100;

end