% using equation 5 from the foucault sheet to model a harmonograph that includes coriolis forces

function main_pendulum()
    g = 9.81;

    % New 'harmonograph' parameters
    Lx = 2.45;
    Ly = 9.81;
    goLx = g / Lx;
    goLy = g / Ly;
    gammax = 0;
    gammay = 0;
    V0 = [0; 2; 0; 1];

    % Parameters 
    omega = (2*pi/86400);
    sigma = deg2rad(41.2);
    dt = 0.001;
    fig_dt = 0.01;
    t0 = 0;
    t_end = 600;  

    F = @(t, V) [
        V(2);                       
        -goLx * V(1) + 2*omega*cos(sigma)* V(4)-gammax*V(2);
        V(4);                       
        -goLy * V(3) - 2*omega*cos(sigma) * V(2) - gammay*V(4)
    ];

    % Run numerical integration (RK4)
    [t_list, V_list] = runge_kutta_2nd_order_system(F, V0, t0, t_end, dt, fig_dt);

    % unpack data for convenience
    x_list = V_list(1, :);
    y_list = V_list(3, :);

    % phase plot: y vs x
    
    figure;
    plot(x_list, y_list, "LineWidth", 1.5);
    xlabel("x", "FontSize", 20);
    ylabel("y", "FontSize", 20);
    grid on;

end

%   Runge-Kutta 4th order solver for a system of 2nd-order ODEs

function [times, y_store] = runge_kutta_2nd_order_system(f, y0, t0, t_end, dt, fig_dt)
    % f(t, y) -> acceleration vector (same size as y)
    
    % y0 : initial vector
    % t0 : initial time
    % t_end : final time
    % dt : time step
    % fig_dt : time interval between stored results

    y = y0(:);      % ensure column vector
    t = t0;

    % pre-allocate estimated size 
    nSteps = ceil((t_end - t0) / fig_dt) + 5;
    times = zeros(nSteps, 1);
    y_store = zeros(length(y0), nSteps);

    store_idx = 0;
    next_store_t = 0;

    % main loop
    while t <= t_end
        if t >= next_store_t
            store_idx = store_idx + 1;
            times(store_idx) = t;
            y_store(:, store_idx) = y;
            next_store_t = next_store_t + fig_dt;
        end

        % RK4 coefficients
        k1 = f(t,          y);
        k2 = f(t + dt/2,   y + dt/2 * k1);
        k3 = f(t + dt/2,   y + dt/2 * k2);
        k4 = f(t + dt,     y + dt * k3);

        % Update solution
        y = y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);

        t = t + dt;
    end

    % remove unused preallocated tail
    times = times(1:store_idx);
    y_store = y_store(:,1:store_idx);
end
