clear
clc

data = readtable("data/seed_imcckf_annealing/results.csv");

%% Getting stats
% Getting list of rhos
rhos = unique(data.rho);

% Preallocation
ise_mean = zeros(size(rhos));
ise_std = zeros(size(rhos));
iae_mean = zeros(size(rhos));
iae_median = zeros(size(rhos));
iae_std = zeros(size(rhos));
itae_mean = zeros(size(rhos));
itae_median = zeros(size(rhos));
itae_std = zeros(size(rhos));

for i = 1:size(rhos)
    % Getting rho subset
    subdata = data(data.rho == rhos(i), :);

    % Remove fails
    subdata = subdata(string(subdata.status) == 'ExperimentStatus.SUCCESS', :);

    % Getting list of experiments for that rho
    experiments = unique(subdata.experiment_id);

    % Preallocation
    ise_experiments = zeros(size(experiments));
    iae_experiments = zeros(size(experiments));
    itae_experiments = zeros(size(experiments));

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
        ise_experiments(j) = norm(ise);
        
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
        iae_experiments(j) = norm(iae);
        
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
        itae_experiments(j) = norm(itae);
    end

    ise_mean(i) = mean(ise_experiments);
    ise_std(i) = std(ise_experiments);
    iae_mean(i) = mean(iae_experiments);
    iae_median(i) = median(iae_experiments);
    iae_std(i) = std(iae_experiments);
    itae_mean(i) = mean(itae_experiments);
    itae_median(i) = median(itae_experiments);
    itae_std(i) = std(itae_experiments);
end
%% Ploting
%plot(rhos, iae_mean)
errorbar(rhos, itae_mean, itae_std, 'LineWidth', 2)