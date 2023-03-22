%data = readmatrix("data/4_feat_mckf.csv");

SCENARIO = 4;

%% Feature motion plot
figure(1)
plot(data.f_1, data.f_2, 'r', 'LineWidth', 2);
hold on
plot(data.f_1(1), data.f_2(1), 'o','MarkerFaceColor','red','MarkerEdgeColor', 'red', 'MarkerSize', 10);
plot(data.desired_f_1(1), data.desired_f_2(1), '*','MarkerFaceColor','red','MarkerEdgeColor', 'red', 'MarkerSize', 10);
xlabel('u (pixels)')
ylabel('v (pixels)')
set(gca, 'YDir','reverse')
xlim([0 256])
ylim([0 256])

if (SCENARIO >= 3)
    plot(data.f_3, data.f_4, 'g', 'LineWidth', 2);
    plot(data.f_3(1), data.f_4(1), 'o','MarkerFaceColor','green','MarkerEdgeColor', 'green', 'MarkerSize', 10);
    plot(data.desired_f_3(1), data.desired_f_4(1), '*','MarkerFaceColor','green','MarkerEdgeColor', 'green', 'MarkerSize', 10);

    plot(data.f_5, data.f_6, 'b', 'LineWidth', 2);
    plot(data.f_5(1), data.f_6(1), 'o','MarkerFaceColor','blue','MarkerEdgeColor', 'blue', 'MarkerSize', 10);
    plot(data.desired_f_5(1), data.desired_f_6(1), '*','MarkerFaceColor','blue','MarkerEdgeColor', 'blue', 'MarkerSize', 10);
end
if (SCENARIO >= 4)
    plot(data.f_7, data.f_8, 'm', 'LineWidth', 2);
    plot(data.f_7(1), data.f_8(1), 'o','MarkerFaceColor','magenta','MarkerEdgeColor', 'magenta', 'MarkerSize', 10);
    plot(data.desired_f_7(1), data.desired_f_8(1), '*','MarkerFaceColor','magenta','MarkerEdgeColor', 'magenta', 'MarkerSize', 10);
end
hold off

%% Error plot
figure(2)
error1 = data.desired_f_1 - data.f_1;
error2 = data.desired_f_2 - data.f_2;
plot(data.t, error1, 'LineWidth', 2);
hold on
plot(data.t, error2, 'LineWidth', 2);
xlabel('Time (s)');
ylabel('Error (pixels)');
if (SCENARIO >= 3)
    error3 = data.desired_f_3 - data.f_3;
    error4 = data.desired_f_4 - data.f_4;

    plot(data.t, error3, 'LineWidth', 2);
    plot(data.t, error4, 'LineWidth', 2);

    error5 = data.desired_f_5 - data.f_5;
    error6 = data.desired_f_6 - data.f_6;

    plot(data.t, error5, 'LineWidth', 2);
    plot(data.t, error6, 'LineWidth', 2);
end
if (SCENARIO >= 4)
    error7 = data.desired_f_7 - data.f_7;
    error8 = data.desired_f_8 - data.f_8;

    plot(data.t, error6, 'LineWidth', 2);
    plot(data.t, error7, 'LineWidth', 2);
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
camera = @(k) SE3(rpy2r(data.camera_roll(k), data.camera_pitch(k), data.camera_yaw(k)), [data.camera_x(k), data.camera_y(k), data.camera_z(k)]);
tranimate(@(k) camera(k), 1:100:height(data), 'length', 0.05, 'retain', 'rgb', 'notext')
hold on
xlabel('X (m)')
ylabel('Y (m)')
zlabel('Z (m)')
plot3(data.camera_x(1), data.camera_y(1), data.camera_z(1), 'o','MarkerFaceColor','black','MarkerEdgeColor', 'black', 'MarkerSize', 10);
plot3(data.camera_x(end), data.camera_y(end), data.camera_z(end), '*','MarkerFaceColor','black','MarkerEdgeColor', 'black', 'MarkerSize', 10);
plot3(data.camera_x, data.camera_y, data.camera_z, 'black', 'LineWidth', 2);
grid on
camproj perspective
axis equal
hold off

%% Joints plot
figure(4)
plot(data.t, data.q_1, 'LineWidth', 2);
hold on
plot(data.t, data.q_2, 'LineWidth', 2);
plot(data.t, data.q_3, 'LineWidth', 2);
plot(data.t, data.q_4, 'LineWidth', 2);
plot(data.t, data.q_5, 'LineWidth', 2);
plot(data.t, data.q_6, 'LineWidth', 2);
legend('q_1', 'q_2', 'q_3', 'q_4', 'q_5', 'q_6');
xlabel('Time (s)')
ylabel('Angle (rad)')