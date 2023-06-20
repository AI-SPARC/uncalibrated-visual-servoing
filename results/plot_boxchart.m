clear
clc

boxTable = table();

%% Getting stats for KF
data = readtable("data/29_05_2023_12_04_46/results.csv");

% Getting list of rhos
alphas = unique(data.rho);

for i = 1:size(alphas)
    % Getting rho subset
    subdata = data(data.rho == alphas(i), :);

    % Remove fails
    subdata = subdata(string(subdata.status) == 'ExperimentStatus.SUCCESS', :);

    % Getting list of experiments for that rho
    experiments = unique(subdata.experiment_id);

    for j = 1:size(experiments)
        % Getting experiment subset
        experiment = subdata(subdata.experiment_id == experiments(j), :);

        % Errors
        error1 = experiment.desired_f_1 - experiment.f_1;
        error2 = experiment.desired_f_2 - experiment.f_2;
        error3 = experiment.desired_f_3 - experiment.f_3;
        error4 = experiment.desired_f_4 - experiment.f_4;
        error5 = experiment.desired_f_5 - experiment.f_5;
        error6 = experiment.desired_f_6 - experiment.f_6;
        error7 = experiment.desired_f_7 - experiment.f_7;
        error8 = experiment.desired_f_8 - experiment.f_8;

        % Stats
        % ISE
        ise = zeros(8, 1);
        ise(1) = error1' * error1;
        ise(2) = error2' * error2;
        ise(3) = error3' * error3;
        ise(4) = error4' * error4;
        ise(5) = error5' * error5;
        ise(6) = error6' * error6;
        ise(7) = error7' * error7;
        ise(8) = error8' * error8;
        
        % IAE
        iae = zeros(8, 1);
        iae(1) = sum(abs(error1));
        iae(2) = sum(abs(error2));
        iae(3) = sum(abs(error3));
        iae(4) = sum(abs(error4));
        iae(5) = sum(abs(error5));
        iae(6) = sum(abs(error6));
        iae(7) = sum(abs(error7));
        iae(8) = sum(abs(error8));
        
        % ITAE
        itae = zeros(8, 1);
        itae(1) = experiment.t' * abs(error1);
        itae(2) = experiment.t' * abs(error2);
        itae(3) = experiment.t' * abs(error3);
        itae(4) = experiment.t' * abs(error4);
        itae(5) = experiment.t' * abs(error5);
        itae(6) = experiment.t' * abs(error6);
        itae(7) = experiment.t' * abs(error7);
        itae(8) = experiment.t' * abs(error8);

        newRow = {experiment.experiment_id(1), alphas(i), norm(ise), norm(iae), norm(itae), "KF"};
        boxTable = [boxTable;newRow];
    end
end

%% Getting stats for GMCKF
data = readtable("data/changing_q_start/26_05_2023_22_42_02/results.csv");

% Getting list of rhos
alphas = unique(data.rho);

for i = 1:size(alphas)
    % Getting rho subset
    subdata = data(data.rho == alphas(i), :);

    % Remove fails
    subdata = subdata(string(subdata.status) == 'ExperimentStatus.SUCCESS', :);

    % Getting list of experiments for that rho
    experiments = unique(subdata.experiment_id);

    for j = 1:size(experiments)
        % Getting experiment subset
        experiment = subdata(subdata.experiment_id == experiments(j), :);

        % Errors
        error1 = experiment.desired_f_1 - experiment.f_1;
        error2 = experiment.desired_f_2 - experiment.f_2;
        error3 = experiment.desired_f_3 - experiment.f_3;
        error4 = experiment.desired_f_4 - experiment.f_4;
        error5 = experiment.desired_f_5 - experiment.f_5;
        error6 = experiment.desired_f_6 - experiment.f_6;
        error7 = experiment.desired_f_7 - experiment.f_7;
        error8 = experiment.desired_f_8 - experiment.f_8;

        % Stats
        % ISE
        ise = zeros(8, 1);
        ise(1) = error1' * error1;
        ise(2) = error2' * error2;
        ise(3) = error3' * error3;
        ise(4) = error4' * error4;
        ise(5) = error5' * error5;
        ise(6) = error6' * error6;
        ise(7) = error7' * error7;
        ise(8) = error8' * error8;
        
        % IAE
        iae = zeros(8, 1);
        iae(1) = sum(abs(error1));
        iae(2) = sum(abs(error2));
        iae(3) = sum(abs(error3));
        iae(4) = sum(abs(error4));
        iae(5) = sum(abs(error5));
        iae(6) = sum(abs(error6));
        iae(7) = sum(abs(error7));
        iae(8) = sum(abs(error8));
        
        % ITAE
        itae = zeros(8, 1);
        itae(1) = experiment.t' * abs(error1);
        itae(2) = experiment.t' * abs(error2);
        itae(3) = experiment.t' * abs(error3);
        itae(4) = experiment.t' * abs(error4);
        itae(5) = experiment.t' * abs(error5);
        itae(6) = experiment.t' * abs(error6);
        itae(7) = experiment.t' * abs(error7);
        itae(8) = experiment.t' * abs(error8);

        newRow = {experiment.experiment_id(1), alphas(i), norm(ise), norm(iae), norm(itae), "GMCKF"};
        boxTable = [boxTable;newRow];
    end
end

%% Plot
ax = axes('FontSize', 20);
boxTable.Properties.VariableNames = ["experiment_id", "alpha", "ise", "iae", "itae", "method"];
boxchart(ax, boxTable.alpha*10, boxTable.iae, "BoxWidth", 0.5, "GroupByColor", boxTable.method);
xlim(10.*[0.9, 2.1]);
xlabel("\alpha (10^1)");
ylabel("IAE");
legend();