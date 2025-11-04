% Unicycle Trajectory Optimization with fmincon
% ----------------------------------------------------------
% Drive a unicycle from A to B while avoiding circular obstacles:
%   x = [px, py, theta],  u = [v, omega]
% Discrete dynamics (Euler, step h):
%   px_{k+1}    = px_k + h * v_k * cos(theta_k)
%   py_{k+1}    = py_k + h * v_k * sin(theta_k)
%   theta_{k+1} = theta_k + h * omega_k
%
% Minimize (terminal goal error) + (control effort) + (control smoothness)
% Subject to:
%   - initial state equality
%   - dynamics equalities
%   - obstacle inequalities: (px - cx)^2 + (py - cy)^2 >= (r + margin)^2
%     (implemented as: (r+margin)^2 - dist^2 <= 0)
%
% Requires: Optimization Toolbox (fmincon)

clear; clc; close all;

%% Problem setup
params.N  = 60;                 % number of steps
params.h  = 0.10;               % step size [s]

% Start / Goal states: [px, py, theta]
params.A = [0; 0; 0];
params.B = [6; 5; 0];

% Obstacles: rows [cx, cy, r]
params.obstacles = [
    2.0, 1.5, 0.8;
    3.5, 3.5, 0.9;
    5.2, 3.8, 0.7
];
% params.obstacles = [
%     2.0, 1.5, 0.8;
%     3.5, 3.5, 0.9
% ];
params.safety_margin = 0.25;    % extra buffer around each obstacle

% Control bounds
params.v_min = -1.5; params.v_max =  1.5;
params.w_min = -2.0; params.w_max =  2.0;

% Weights
params.w_goal_pos   = 400.0;
params.w_goal_theta = 20.0;
params.w_u          = 0.05;
params.w_du         = 0.20;

% Dimensions
params.n_x = 3; params.n_u = 2;
params.Z_DIM = (params.n_x + params.n_u) * params.N + params.n_x;

%% Initial guess: straight line in position, constant heading and speed
[X0, U0] = initial_guess(params);
z0 = pack(X0, U0, params);

%% Variable bounds
lb = -inf(params.Z_DIM,1);  ub =  inf(params.Z_DIM,1);
for k = 0:params.N-1
    iu = idx_u(k, params);
    lb(iu(1)) = params.v_min;  ub(iu(1)) = params.v_max;
    lb(iu(2)) = params.w_min;  ub(iu(2)) = params.w_max;
end

%% Objective and constraints (pass params via anonymous handles)
obj  = @(z) objective(z, params);
nlin = @(z) nonlin_constraints(z, params);

opts = optimoptions('fmincon', ...
    'Algorithm','sqp', ...
    'MaxIterations', 1000, ...
    'OptimalityTolerance', 1e-6, ...
    'ConstraintTolerance', 1e-6, ...
    'StepTolerance', 1e-10, ...
    'Display','iter', ...
    'SpecifyObjectiveGradient',false, ...
    'SpecifyConstraintGradient',false);

%% Solve
[z_star, fval, exitflag, output] = fmincon(obj, z0, [], [], [], [], lb, ub, nlin, opts);

fprintf('\n=== fmincon status ===\n');
fprintf('exitflag: %d\n', exitflag);
disp(output.message);
fprintf('final objective: %.6f\n', fval);

[X_star, U_star] = unpack(z_star, params);
fprintf('Terminal state [px py theta]^T = [%.3f %.3f %.3f]\n', X_star(end,:));

%% Plot: environment and trajectories
blue = [0 0.4470 0.7410];  % MATLAB default blue
fig = figure('Color','w','Position',[100 100 800 650]); hold on; box on; grid on;
% Obstacles + margins
ang = linspace(0,2*pi,200);
for j = 1:size(params.obstacles,1)
    cx = params.obstacles(j,1); cy = params.obstacles(j,2);
    r = params.obstacles(j,3) + params.safety_margin;
    plot(cx + r*cos(ang), cy + r*sin(ang), 'r-', 'LineWidth',1.2);
    fill(cx + r*cos(ang), cy + r*sin(ang), [1 0 0], 'FaceAlpha',0.12, 'EdgeColor','none');
end
% Start / goal
plot(params.A(1), params.A(2),'go','MarkerFaceColor','g','DisplayName','start');
plot(params.B(1), params.B(2),'b*','MarkerSize',10,'DisplayName','goal');

% Trajectories
[X_init,~] = unpack(z0, params);
plot(X_init(:,1), X_init(:,2), 'k--', 'LineWidth',1.2, 'DisplayName','initial guess');
plot(X_star(:,1), X_star(:,2), 'Color',blue, 'LineWidth',2.0, 'DisplayName','optimized');

% Heading arrows (sparse)
skip = max(1, floor(params.N/20));
for k = 1:skip:(params.N+1)
    px = X_star(k,1); py = X_star(k,2); th = X_star(k,3);
    quiver(px, py, 0.4*cos(th), 0.4*sin(th), 0, 'Color',blue, 'LineWidth',1);
end

axis equal;
xlabel('x'); ylabel('y');
title('Unicycle TO with Obstacle Avoidance (fmincon / SQP)');
% legend('Location','best'); hold off;

%% Plot controls
t = (0:params.N-1)' * params.h;
figure('Color','w','Position',[100 100 850 450]);
subplot(2,1,1);
plot(t, U0(:,1),'k--','DisplayName','v init'); hold on; grid on;
plot(t, U_star(:,1),'Color',blue,'DisplayName','v*');
yline(params.v_min,':k'); yline(params.v_max,':k');
ylabel('v [m/s]'); legend;

subplot(2,1,2);
plot(t, U0(:,2),'k--','DisplayName','\omega init'); hold on; grid on;
plot(t, U_star(:,2),'Color',blue,'DisplayName','\omega*');
yline(params.w_min,':k'); yline(params.w_max,':k');
xlabel('time [s]'); ylabel('\omega [rad/s]'); legend;

%% ===== Helper functions (stateless; all data via params) =====
function I = idx_x(k, params)
    I = (params.n_x + params.n_u)*k + (1:params.n_x);
end

function I = idx_u(k, params)
    I = (params.n_x + params.n_u)*k + params.n_x + (1:params.n_u);
end

function z = pack(X, U, params)
    % Pack states and controls into a single decision vector z.
    % Uses 0..N-1 stage index for z-blocks, but 1-based indexing for X,U.
    z = zeros(params.Z_DIM,1);
    for kk = 0:params.N-1
        z(idx_x(kk,params)) = X(kk+1,:).';   % OK: X is (N+1)x3, use kk+1
        z(idx_u(kk,params)) = U(kk+1,:).';   % FIX: U is Nx2, use kk+1 (NOT kk)
    end
    z(idx_x(params.N,params)) = X(params.N+1,:).';
end


function [X,U] = unpack(z, params)
    X = zeros(params.N+1, params.n_x);
    U = zeros(params.N,   params.n_u);
    for kk = 0:params.N-1
        X(kk+1,:) = z(idx_x(kk,params)).';
        U(kk+1,:) = z(idx_u(kk,params)).';
    end
    X(params.N+1,:) = z(idx_x(params.N,params)).';
end

function xnext = f_disc(xk, uk, hstep)
    px = xk(1); py = xk(2); th = xk(3);
    v = uk(1);  w  = uk(2);
    xnext = [ ...
        px + hstep * v * cos(th); ...
        py + hstep * v * sin(th); ...
        th + hstep * w ];
end

function [X,U] = initial_guess(params)
    N = params.N; h = params.h; A = params.A; B = params.B;
    X = zeros(N+1, params.n_x);
    U = zeros(N,   params.n_u);
    dx = B(1) - A(1);
    dy = B(2) - A(2);
    path_ang = atan2(dy, dx);
    for kk = 0:N
        alpha = kk / N;
        X(kk+1,1) = A(1) + alpha * dx;
        X(kk+1,2) = A(2) + alpha * dy;
        X(kk+1,3) = path_ang;
    end
    L = hypot(dx, dy);
    v_guess = L / (N * h);
    U(:,1) = max(params.v_min, min(params.v_max, v_guess));
    U(:,2) = 0.0;
end

function J = objective(z, params)
    [X,U] = unpack(z, params);
    % Terminal goal tracking
    pos_err   = X(end,1:2)' - params.B(1:2);
    theta_err = X(end,3)    - params.B(3);
    J_goal = params.w_goal_pos * (pos_err.'*pos_err) + params.w_goal_theta * (theta_err^2);

    % Control effort
    J_u = 0;
    for kk = 1:params.N
        J_u = J_u + params.w_u * (U(kk,:)*U(kk,:)');
    end

    % Control smoothness
    J_du = 0;
    for kk = 1:params.N-1
        du = U(kk+1,:) - U(kk,:);
        J_du = J_du + params.w_du * (du*du');
    end

    J = 0.5 * (J_goal + J_u + J_du);
end

function [c, ceq] = nonlin_constraints(z, params)
    [X,U] = unpack(z, params);

    % Equality: initial condition + dynamics
    ceq = zeros(3 + 3*params.N, 1);
    ceq(1:3) = X(1,:)' - params.A;

    row = 4;
    for kk = 1:params.N
        xk    = X(kk,:)';
        uk    = U(kk,:)';
        xnext = f_disc(xk, uk, params.h);
        ceq(row:row+2) = X(kk+1,:)' - xnext;
        row = row + 3;
    end

    % Inequalities: (r+margin)^2 - ((px-cx)^2 + (py-cy)^2) <= 0  for all k, all obstacles
    nObs = size(params.obstacles,1);
    c = zeros((params.N+1)*nObs, 1);
    rr = 1;
    for kk = 1:(params.N+1)
        px = X(kk,1); py = X(kk,2);
        for jj = 1:nObs
            cx = params.obstacles(jj,1); cy = params.obstacles(jj,2);
            r0 = params.obstacles(jj,3) + params.safety_margin;
            c(rr) = (r0^2) - ((px - cx)^2 + (py - cy)^2);
            rr = rr + 1;
        end
    end
end
