data = readtable('');
% Loop over columns
for i = 9:(size(data,2)-1)
    
    % Get column name
    col_name = data.Properties.VariableNames{i};
    
    % Get data for pain and non-pain groups
    x = data{data.pain==0,i};
    y = data{data.pain==1,i};
    
    % Test for normality using Shapiro-Wilk test
    [~, p_x] = kstest(x);
    [~, p_y] = kstest(y);
    
    % Check if both distributions are normal
    if p_x > 0.05 && p_y > 0.05 % Both normal
        [~, p] = ttest2(x, y);
        d = cohen_d_parametric(x, y);
        test_type = 't-test';
    else % At least one non-normal
        p = ranksum(x, y);
        d = cohen_d_nonparametric(x, y);
        test_type = 'Mann-Whitney U test and Cliff effect size';
    end
    
    % Print results
    fprintf('Column %s:\n', col_name);
    fprintf('  Test type: %s\n', test_type);
    fprintf('  p-value: %.4f\n', p);
    fprintf('  Effect size (''s d): %.4f\n', d.Effect);
    
end

% Define function for computing Cohen's d for parametric distributions
function d = cohen_d_parametric(x, y)
    d = meanEffectSize(x, y, Effect='Cohen');

    %cohen effect size small (d = 0.2), medium (d = 0.5), and large (d ≥ 0.8)
end

% Define function for computing Cohen's d for non-parametric distributions
function d = cohen_d_nonparametric(x, y)
    d = meanEffectSize(x, y, Effect= 'Cliff');

    %significance  0.147 (small), 0.33 (medium), and 0.474 (large) 
end
