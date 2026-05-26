function netbp_soccer_random
% NETBP_SOCCER_RANDOM
%   Generates a RANDOM soccer shot dataset each run.
%   Trains a neural network to classify Goal vs Miss using backprop.

%% --------- RANDOM DATASET -------------------
rng('shuffle');        % different data every run
Ndata = 15;            % number of shots; we can change this to anything

% Random shot locations on the field
x1 = rand(1, Ndata);   % horizontal position
x2 = rand(1, Ndata);   % vertical distance toward goal

% Probability of scoring increases as shot gets closer to the goal line
goal_prob = 1 ./ (1 + exp(-12*(x2 - 0.65)));  % scoring probability

% Generate outcomes (1=goal, 0=miss)
outcome = rand(1, Ndata) < goal_prob;

% Convert outcomes into 2D network outputs
y = [ outcome ; 1 - outcome ];   % [1;0]=Goal, [0;1]=Miss

%% ---------------------- PLOT DATASET ---------------------
figure(1); clf
goal_idx = find(outcome == 1);
miss_idx = find(outcome == 0);

plot(x1(goal_idx), x2(goal_idx), 'go', 'MarkerSize',10, 'LineWidth',2); hold on
plot(x1(miss_idx), x2(miss_idx), 'rx', 'MarkerSize',10, 'LineWidth',2);

xlabel('x (horizontal field position)');
ylabel('y (distance toward goal)');
title('Random soccer shot dataset');
legend('Goal','Miss','Location','best');

%% ---------------------- INITIALIZE NETWORK ------------------------
rng(7); 

% Network: 2 -> 4 -> 4 -> 2
W2 = 0.5*randn(4,2);
W3 = 0.5*randn(4,4);
W4 = 0.5*randn(2,4);

b2 = 0.5*randn(4,1);
b3 = 0.5*randn(4,1);
b4 = 0.5*randn(2,1);

%% ---------------------- TRAINING (Stochastic Gradient Decsent) ----------------------------
eta = 0.05;    
Niter = 2e5;
savecost = zeros(Niter,1);

for counter = 1:Niter
    
    % Pick one random shot
    k = randi(Ndata);
    x = [x1(k); x2(k)];
    
    % Forward pass
    a2 = activate(x, W2, b2);
    a3 = activate(a2, W3, b3);
    a4 = activate(a3, W4, b4);
    
    % Back Propagation
    delta4 = a4 .* (1 - a4) .* (a4 - y(:,k));
    delta3 = a3 .* (1 - a3) .* (W4' * delta4);
    delta2 = a2 .* (1 - a2) .* (W3' * delta3);
    
    % Update parameters
    W2 = W2 - eta * delta2 * x';
    W3 = W3 - eta * delta3 * a2';
    W4 = W4 - eta * delta4 * a3';
    
    b2 = b2 - eta * delta2;
    b3 = b3 - eta * delta3;
    b4 = b4 - eta * delta4;
    
    % Track cost
    savecost(counter) = cost(W2,W3,W4,b2,b3,b4,x1,x2,y);
end

%% -------------------- PLOT COST --------------------------
figure(2); clf
semilogy(1:1e3:Niter, savecost(1:1e3:Niter), 'LineWidth', 2);
xlabel('Iteration number'); ylabel('Cost');
title('Training progress (random soccer net)');
grid on

%% ------------------ DECISION REGION -----------------------
N = 200;
xvals = linspace(0,1,N);
yvals = linspace(0,1,N);

for i = 1:N
    for j = 1:N
        xy = [xvals(j); yvals(i)];
        a2 = activate(xy, W2, b2);
        a3 = activate(a2, W3, b3);
        a4 = activate(a3, W4, b4);
        Aval(i,j) = a4(1);
        Bval(i,j) = a4(2);
    end
end

Mval = Aval > Bval;

figure(3); clf
contourf(xvals, yvals, Mval, [0.5 0.5]); hold on
colormap([1 1 1; 0.8 0.8 0.8]);
plot(x1(goal_idx), x2(goal_idx), 'go','MarkerSize',10,'LineWidth',2);
plot(x1(miss_idx), x2(miss_idx), 'rx','MarkerSize',10,'LineWidth',2);
xlabel('x (horizontal field position)');
ylabel('y (distance toward goal)');
title('Decision regions (goal or miss)');
legend('Goal','Miss','Location','best');

end

%% --------------- COST FUNCTION ---------------------
function costval = cost(W2,W3,W4,b2,b3,b4,x1,x2,y)
Ndata = length(x1);
costvec = zeros(Ndata,1);
for i = 1:Ndata
    x = [x1(i); x2(i)];
    a2 = activate(x, W2, b2);
    a3 = activate(a2, W3, b3);
    a4 = activate(a3, W4, b4);
    costvec(i) = norm(y(:,i) - a4,2);
end
costval = norm(costvec,2)^2;
end

%% -------------- SIGMOID LAYER -------------------
function y = activate(x,W,b)
y = 1 ./ (1 + exp(-(W*x + b)));
end

