clear
clc

experiment = readtable("experiment/4_feat_mckf_tr.csv");
%experiment = readtable("experiment/4_feat_kf.csv");

SCENARIO = 4;

%% Feature motion plot
figure(1)
%subplot(2,2,1)
plot(experiment.f_1, experiment.f_2, 'r', 'LineWidth', 2);
hold on
plot(experiment.f_1(1), experiment.f_2(1), 'o','MarkerFaceColor','red','MarkerEdgeColor', 'red', 'MarkerSize', 10);
plot(experiment.desired_f_1(1), experiment.desired_f_2(1), '*','MarkerFaceColor','red','MarkerEdgeColor', 'red', 'MarkerSize', 10);
xlabel('u (pixels)')
ylabel('v (pixels)')
set(gca, 'YDir','reverse')
xlim([0 256])
ylim([0 256])

if (SCENARIO >= 3)
    plot(experiment.f_3, experiment.f_4, 'g', 'LineWidth', 2);
    plot(experiment.f_3(1), experiment.f_4(1), 'o','MarkerFaceColor','green','MarkerEdgeColor', 'green', 'MarkerSize', 10);
    plot(experiment.desired_f_3(1), experiment.desired_f_4(1), '*','MarkerFaceColor','green','MarkerEdgeColor', 'green', 'MarkerSize', 10);

    plot(experiment.f_5, experiment.f_6, 'b', 'LineWidth', 2);
    plot(experiment.f_5(1), experiment.f_6(1), 'o','MarkerFaceColor','blue','MarkerEdgeColor', 'blue', 'MarkerSize', 10);
    plot(experiment.desired_f_5(1), experiment.desired_f_6(1), '*','MarkerFaceColor','blue','MarkerEdgeColor', 'blue', 'MarkerSize', 10);
end
if (SCENARIO >= 4)
    plot(experiment.f_7, experiment.f_8, 'm', 'LineWidth', 2);
    plot(experiment.f_7(1), experiment.f_8(1), 'o','MarkerFaceColor','magenta','MarkerEdgeColor', 'magenta', 'MarkerSize', 10);
    plot(experiment.desired_f_7(1), experiment.desired_f_8(1), '*','MarkerFaceColor','magenta','MarkerEdgeColor', 'magenta', 'MarkerSize', 10);
end
hold off

%% Error plot
figure(2)
%subplot(2,2,2)
error1 = experiment.desired_f_1 - experiment.f_1;
error2 = experiment.desired_f_2 - experiment.f_2;
plot(experiment.t, error1, 'LineWidth', 2);
hold on
plot(experiment.t, error2, 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Error (pixels)');
if (SCENARIO >= 3)
    error3 = experiment.desired_f_3 - experiment.f_3;
    error4 = experiment.desired_f_4 - experiment.f_4;

    plot(experiment.t, error3, 'LineWidth', 2);
    plot(experiment.t, error4, 'LineWidth', 2);

    error5 = experiment.desired_f_5 - experiment.f_5;
    error6 = experiment.desired_f_6 - experiment.f_6;

    plot(experiment.t, error5, 'LineWidth', 2);
    plot(experiment.t, error6, 'LineWidth', 2);
end
if (SCENARIO >= 4)
    error7 = experiment.desired_f_7 - experiment.f_7;
    error8 = experiment.desired_f_8 - experiment.f_8;

    plot(experiment.t, error6, 'LineWidth', 2);
    plot(experiment.t, error7, 'LineWidth', 2);
end
if (SCENARIO == 1)
    legend('error 1', 'error 2')
elseif (SCENARIO == 3)
    legend('error 1', 'error 2', 'error 3', 'error 4', 'error 5', 'error 6')
elseif (SCENARIO == 4)
    legend('error 1', 'error 2', 'error 3', 'error 4', 'error 5', 'error 6', 'error 7', 'error 8')
end
hold off

%% Camera pose plot
figure(3)
%subplot(2,2,3)
camera = @(k) SE3(rpy2r(experiment.camera_roll(k), experiment.camera_pitch(k), experiment.camera_yaw(k)), [experiment.camera_x(k), experiment.camera_y(k), experiment.camera_z(k)]);
tranimate(@(k) camera(k), 1:100:height(experiment), 'length', 0.05, 'retain', 'rgb', 'notext')
hold on
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
plot3(experiment.camera_x(1), experiment.camera_y(1), experiment.camera_z(1), 'o','MarkerFaceColor','black','MarkerEdgeColor', 'black', 'MarkerSize', 10);
plot3(experiment.camera_x(end), experiment.camera_y(end), experiment.camera_z(end), '*','MarkerFaceColor','black','MarkerEdgeColor', 'black', 'MarkerSize', 10);
plot3(experiment.camera_x, experiment.camera_y, experiment.camera_z, 'black', 'LineWidth', 2);
grid on
camproj perspective
axis equal
hold off

%% Joints plot
figure(4)
%subplot(2,2,4)
plot(experiment.t, experiment.q_1, 'LineWidth', 2);
hold on
plot(experiment.t, experiment.q_2, 'LineWidth', 2);
plot(experiment.t, experiment.q_3, 'LineWidth', 2);
plot(experiment.t, experiment.q_4, 'LineWidth', 2);
plot(experiment.t, experiment.q_5, 'LineWidth', 2);
plot(experiment.t, experiment.q_6, 'LineWidth', 2);
legend('q_1', 'q_2', 'q_3', 'q_4', 'q_5', 'q_6');
xlabel('Time (s)')
ylabel('Angle (rad)')

%% Noise plot
figure(5)
subplot(4,2,1)
histogram(experiment.noise_1, 30)
subplot(4,2,2)
histogram(experiment.noise_2, 30)
subplot(4,2,3)
histogram(experiment.noise_3, 30)
subplot(4,2,4)
histogram(experiment.noise_4, 30)
subplot(4,2,5)
histogram(experiment.noise_5, 30)
subplot(4,2,6)
histogram(experiment.noise_6, 30)
subplot(4,2,7)
histogram(experiment.noise_7, 30)
subplot(4,2,8)
histogram(experiment.noise_8, 30)

%% Stats
% ISE
ise(1) = error1' * error1;
ise(2) = error2' * error2;
ise(3) = error3' * error3;
ise(4) = error4' * error4;
ise(5) = error5' * error5;
ise(6) = error6' * error6;
ise(7) = error7' * error7;
ise(8) = error8' * error8;
disp(['ISE: ', num2str(norm(ise))])

% IAE
iae(1) = sum(abs(error1));
iae(2) = sum(abs(error2));
iae(3) = sum(abs(error3));
iae(4) = sum(abs(error4));
iae(5) = sum(abs(error5));
iae(6) = sum(abs(error6));
iae(7) = sum(abs(error7));
iae(8) = sum(abs(error8));
disp(['IAE: ', num2str(norm(iae))])

% ITAE
itae(1) = experiment.t' * abs(error1);
itae(2) = experiment.t' * abs(error2);
itae(3) = experiment.t' * abs(error3);
itae(4) = experiment.t' * abs(error4);
itae(5) = experiment.t' * abs(error5);
itae(6) = experiment.t' * abs(error6);
itae(7) = experiment.t' * abs(error7);
itae(8) = experiment.t' * abs(error8);
disp(['ITAE: ', num2str(norm(itae))])
