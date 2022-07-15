
classdef SPCAOrthoNewton
properties
    n
    r
    A
    mu
    manifold
    verbose
    Delta
    X
    Y
    W
    U
    V
    gX
    gY
    gU
    gV
    loss
    sparse
    sigma
    iter_num
    residual
    cg_regularizer
    record
    tau
    maxcg_iter
    maxinner_iter
    maxnewton_iter
    last_newtonstatus
    gradnorm
    gradnorm_decay
    gradnorm_min
    sigma_factor
    elapsed_time
end

methods
    function ret = mul_AtA(obj, X)
        ret = X * obj.A' * obj.A;
    end

    function ret = AtA_mul(obj, X)
        ret = obj.A' * (obj.A * X);
    end

    function ret = l1_prox(obj, X, lam)
        ret = sign(X) .* max(0, abs(X) - lam);
    end

    function ret = l1_moreau(obj, X, lam)
        p = (abs(X) < lam);
        ret = p .* (0.5 * X .* X) + (1 - p) .* (lam * abs(X) - 0.5 * lam * lam);
    end

    function ret = objective_smooth(obj, X)
        ret = -sum(sum(X .* obj.AtA_mul(X)));
    end

    function ret = objective(obj, X)
        ret = obj.objective_smooth(X) + obj.mu * sum(sum(abs(X)));
    end

    function G = egrad(obj, X)
        T = X + obj.U / obj.sigma;
        G = -2 * obj.AtA_mul(X) + obj.sigma * (T - obj.l1_prox(T, obj.mu / obj.sigma));
    end

    function [G, DzG] = off_diag_hess(obj, X, W, D)
        U = obj.AtA_mul(X);
        Z = X' * U;
        K = -W + obj.V / obj.sigma;

        S = 2 * Z + K(:, 1:obj.r) - K(:, obj.r+1:obj.r*2);
        S = S - diag(diag(S));

        R = 2 * D' * U;
        R = R + R';
        R = R - diag(diag(R));

        G = obj.sigma * U * (S + S');
        DzG = obj.sigma * (obj.AtA_mul(D) * (S + S') + U * (R + R'));
    end

    function DzG = ehess(obj, X, Z)
        T = X + obj.U / obj.sigma;
        E = (abs(T) <= obj.mu / obj.sigma);
        DzG = -2 * obj.AtA_mul(Z) + obj.sigma * Z .* E;
    end

    function H = rhess(obj, X, W, Z)
        [diag_G, diag_DzG] = obj.off_diag_hess(X, W, Z);
        H = obj.manifold.ehess2rhess(X, obj.egrad(X) + diag_G, ...
                obj.ehess(X, Z) + diag_DzG, Z);
    end

    function [Z, norm_res, it, flag, eig_min] = solve_newton_system_cg(obj, X, W, G, max_iter, tol, regularizer)
        metric = @(U, V) sum(sum(U .* V));
        Z = zeros(size(G));
        R = G;
        P = -R;
        eig_min = 1e10;
        flag = 1;

        for it = 1:max_iter
            norm_rr = metric(R, R);
            if sqrt(norm_rr) < tol
                break;
            end

            Hp = obj.rhess(X, W, P) + regularizer * P;
            norm_pp = metric(P, P);
            norm_pHp = metric(P, Hp);
            eig_min = min(eig_min, norm_pHp / norm_pp);

            % find small, or negative direction
            if norm_pHp < 1.0e-6 * norm_pp
                flag = 1;
                if norm_pHp < -1.0e-6 * norm_pp 
                    if obj.verbose
                        fprintf(' CG, Find Non-PSD\n');
                    end
                    flag = 0;
                else
                    if obj.verbose
                        fprintf(' CG, Term\n');
                    end
                end
                break;
            end

            alpha = norm_rr / norm_pHp;
            Z = Z + alpha * P;
            R = alpha * Hp + R;
            beta = metric(R, R) / norm_rr;
            P = beta * P - R;
        end

        norm_res = sqrt(norm_rr);
    end

    function ret = H2(obj, X)
        Z = X' * obj.AtA_mul(X);
        ret = [ obj.Delta + Z  obj.Delta - Z ];
    end

    function ret = off_diag(obj, D)
        D(:,1:obj.r) = D(:,1:obj.r) - diag(diag(D(:,1:obj.r)));
        D(:,obj.r+1:2*obj.r) = D(:,obj.r+1:2*obj.r) - diag(diag(D(:,obj.r+1:2*obj.r)));
        ret = D;
    end

    function ret = get_off_diag(obj, X, W, V, sigma)
        D = obj.H2(X) - W + V / sigma;
        ret = obj.off_diag(D);
    end

    function ret = off_diag_grad(obj, X, W, V, sigma)
        % ---
        U = obj.AtA_mul(X);
        Z = X' * U;
        K = -W + V / sigma;
        S = 2 * Z + K(:, 1:obj.r) - K(:, obj.r+1:obj.r*2);
        S = S - diag(diag(S));
        ret = sigma * U * (S + S');
    end

    function ret = alm_cost(obj, X, W, sigma, u, v)
        G = sigma * obj.l1_moreau(X + u / sigma, obj.mu / sigma);
        f = obj.objective_smooth(X);
        Ws = min(0, obj.H2(X) + v / sigma);
        Ws = obj.off_diag(Ws);
        ret = f + 0.5 * sigma * sum(sum(Ws .^ 2)) + sum(sum(G));
    end

    function ret = alm_costgrad(obj, X, W, sigma, u, v)
        T = X + u / sigma;


        % ---
        Ws = (0 > obj.H2(X) + v / sigma);
        U = obj.AtA_mul(X);
        Z = X' * U;
        K = Ws .* (obj.H2(X) + v / sigma);
        S = K(:, 1:obj.r) - K(:, obj.r+1:obj.r*2);
        S = S - diag(diag(S));
        DG = sigma * U * (S + S');

        ret = -2 * obj.AtA_mul(X) + DG + sigma * (T - obj.l1_prox(T, obj.mu / sigma));
    end

    function obj = subopt_newton(obj, W, sigma, u, v)
        cost = @(X) obj.alm_cost(X, W, sigma, u, v);
        g_cost = @(X) obj.alm_costgrad(X, W, sigma, u, v);
        cost_and_grad = @(X) obj.alm_cost_costgrad(X, W, sigma, u, v);

        if obj.iter_num == 1
            X = zeros([obj.n, obj.r]);
            r = obj.r;
            X(1:r, 1:r) = eye(r);
        else
            X = obj.X;
        end

        % fprintf('=== Find initial point ===\n');
        feps = 5.0e-4;

        gX = obj.manifold.proj(X, g_cost(X));

        if sqrt(sum(sum(gX .^ 2))) > feps
            options_sd.maxiter = obj.maxinner_iter;
            options_sd.tolgradnorm = max(obj.gradnorm, feps);
            options_sd.verbosity = 0;
            x_init = [];
            if obj.iter_num > 1
                x_init = obj.X;
            else
                x_init = obj.manifold.rand();
            end

            options_sd.record = 0;
            options_sd.mxitr = options_sd.maxiter;
            if obj.iter_num == 1
                options_sd.mxitr = ceil(options_sd.mxitr * 2);
            elseif obj.iter_num < 6
                options_sd.mxitr = max(500, ceil(options_sd.mxitr / 2));
            end
            options_sd.gtol = options_sd.tolgradnorm;
            options_sd.xtol = 1.0e-20;
            options_sd.ftol = 1.0e-20;
            [X, info] = OptStiefelGBB(x_init, cost_and_grad, options_sd);
            if obj.verbose
                fprintf('\nOptM: obj: %7.6e, itr: %d, nfe: %d, norm(XT*X-I): %3.2e \n', ...
                            info.fval, info.itr, info.nfe, norm(X'*X - eye(obj.r), 'fro') );
            end
        end

        gX = obj.manifold.proj(X, g_cost(X));
        if obj.verbose
            fprintf('    Initial: |gX|_2 = %g\n', sqrt(sum(sum(gX .^ 2))));
        end

        if sqrt(sum(sum(gX .^ 2))) > feps && obj.last_newtonstatus == 0
            obj.X = X;
            return
        end

        if obj.verbose
            fprintf('=== Second-order algorithm ===\n');
            fprintf('    #%d, |gX|_2 = %g, cost = %g, norm_rr = N/A, alpha = N/A \n', 0, sqrt(sum(sum(gX .^ 2))), cost(X));
        end

        for it = 1:obj.maxnewton_iter
            gX_norm = sqrt(sum(sum(gX .^ 2)));
            if gX_norm < obj.gradnorm
                break;
            end

            [Z, norm_rr, it_newton, cg_flag, cg_eigmin] = obj.solve_newton_system_cg(X, W, gX, obj.maxcg_iter, min(1.0e-5, obj.gradnorm), obj.cg_regularizer);
            lower_eig = -1.5 * min(cg_eigmin - obj.cg_regularizer, 0);

            prev_regularizer = obj.cg_regularizer;
            if ~cg_flag
                obj.cg_regularizer = obj.cg_regularizer * 2;
                continue;
            elseif cg_eigmin > 1.0e-3
                obj.cg_regularizer = obj.cg_regularizer * 0.8;
%            elseif cg_eigmin < 0.4
%                obj.cg_regularizer = obj.cg_regularizer / 0.8;
            end
            obj.cg_regularizer = max(lower_eig, obj.cg_regularizer);
%            obj.cg_regularizer = 1.0e-5;

            prev_X = X;
            alpha = 1.0;

            alpha_thres = 1.0e-4;
            tX = obj.manifold.retr(X, Z);
            while sum(sum(obj.manifold.proj(tX, g_cost(tX)).^2)) > (1 - 0.1 * alpha) * sum(sum(gX.^2)) && alpha > alpha_thres
                alpha = alpha * 0.5;
                tX = obj.manifold.retr(X, alpha * Z);
            end

            X = tX;
            gX = obj.manifold.proj(X, g_cost(X));

            if obj.verbose
                fprintf('    #%d, |gX|_2 = %g, cost = %g, norm_rr = %g, cg_iter = %d, regularizer = %g, eigmin = %g, alpha = %g \n',...
                    it + 1, sqrt(sum(sum(gX.^2))), cost(X), norm_rr, it_newton, prev_regularizer, cg_eigmin, alpha);
            end
            if alpha < alpha_thres
                break;
            end
        end

        if obj.verbose
            fprintf('    Newton iter #%d, |gX|_2 = %g\n', it, sqrt(sum(sum(gX.^2))));
        end
        if it < obj.maxnewton_iter
            obj.last_newtonstatus = 1;
        else
            if obj.verbose
                fprintf('    Failed Newton!\n')
            end
            obj.last_newtonstatus = 0;
        end
        obj.X = X;
    end

    function [F, G] = alm_cost_costgrad(obj, X, W, sigma, u, v)
        F = obj.alm_cost(X, W, sigma, u, v);
        G = obj.alm_costgrad(X, W, sigma, u, v);
    end

    function obj = update(obj)
        obj.gradnorm = max(obj.gradnorm_min, 3 * obj.gradnorm_decay.^obj.iter_num);
        if obj.iter_num > 2
            gap_U = sqrt(sum(sum(obj.gU .^ 2)));
            gap_V = sqrt(sum(sum(obj.gV .^ 2)));
            gap_X = sqrt(sum(sum(obj.gX .^ 2)));
            obj.gradnorm = max(obj.gradnorm_min, min(obj.gradnorm, max(gap_U, gap_V) * 20));
        end

%        obj = obj.subopt_firstorder(obj.gradnorm);
        obj = obj.subopt_newton(obj.W, obj.sigma, obj.U, obj.V);

        obj.Y = obj.l1_prox(obj.X + obj.U / obj.sigma, obj.mu / obj.sigma);
        D = obj.H2(obj.X) + obj.V / obj.sigma;
        obj.W = max(D, 0);

        obj.U = obj.U + obj.sigma * (obj.X - obj.Y);
        obj.V = obj.V + obj.sigma * (obj.H2(obj.X) - obj.W);
        obj.V = obj.off_diag(obj.V);
        obj = obj.KKT();
        residual = max(max(max(abs(obj.gU))), max(max(abs(obj.gV))));
        L = -0.5 * log10(sum(sum(obj.gX.^2))) + 0.5 * log10(sum(sum(obj.gU.^2))) + 0.5 * log10(sum(sum(obj.gV.^2)));
        if residual >= obj.tau * obj.residual || L > 0.4
            obj.sigma = obj.sigma * obj.sigma_factor;
            obj.sigma = max(max(obj.sigma, sum(sum(obj.U.^2))^0.51), sum(sum(obj.V.^2))^0.51);
        end
        obj.residual = residual;
        obj.iter_num = obj.iter_num + 1;
    end

    function obj = KKT(obj)
        AtAX = obj.AtA_mul(obj.X);
        T = obj.V(:, 1:obj.r) - obj.V(:, obj.r+1:obj.r*2);
        T = T - diag(diag(T));
        obj.gX = -2 * AtAX + obj.U + AtAX * (T + T');
        obj.gX = obj.manifold.proj(obj.X, obj.gX) / (sqrt(sum(sum(obj.X.^2))) + 1);
        zero_Y = (abs(obj.Y) < 1.0e-8);
        obj.gY = (1 - zero_Y) .* (obj.mu * sign(obj.Y) - obj.U) - zero_Y .* obj.l1_prox(obj.U, obj.mu);
        obj.gY = obj.gY / (sqrt(sum(sum(obj.Y.^2))) + 1);
        obj.gU = (obj.X - obj.Y); %/ (sqrt(sum(sum(obj.X.^2))) + 1);
        obj.gV = obj.H2(obj.X) - obj.W;
        obj.gV = obj.off_diag(obj.gV);
        obj.gV = max(abs(obj.gV) - obj.Delta, 0); % the code of ALSPCA use the absolute error
        obj.sparse = sum(sum(abs(obj.X) < 1.0e-6)) / (obj.n * obj.r);
        obj.loss = obj.objective(obj.X);
        obj.record.gX = [ obj.record.gX max(max(abs(obj.gX))) ];
        obj.record.gY = [ obj.record.gY max(max(abs(obj.gY))) ];
        obj.record.gU = [ obj.record.gU max(max(abs(obj.gU))) ];
        obj.record.gV = [ obj.record.gV max(max(abs(obj.gV))) ];
        obj.record.gradnorm = [ obj.record.gradnorm obj.gradnorm ];
        obj.record.sigma = [ obj.record.sigma obj.sigma ];
        obj.record.sparse = [ obj.record.sparse obj.sparse ];
        obj.record.loss   = [ obj.record.loss obj.loss ];
        obj.record.time   = [ obj.record.time toc ];

        if obj.verbose
            fprintf('gX = %g, gY = %g, gU = %g, gV = %g, gradnorm = %g, loss = %g\n', ...
                max(max(abs(obj.gX))), max(max(abs(obj.gY))), max(max(abs(obj.gU))), max(max(abs(obj.gV))), obj.gradnorm, obj.loss);
        end
    end

    function [cpav, time, sparity, ortho, eigortho, obj] = run(obj, tol)
        tic;
        for i = 1:300
            obj = obj.update();
            Ct = obj.X' * obj.AtA_mul(obj.X);
            cpav = (sum(diag(Ct)) - norm(Ct - diag(diag(Ct)), 'fro')) / sum(sum(obj.A .^ 2));
            if obj.verbose
                fprintf("Iter = %d, sigma = %g, gap = %g, cpav = %2.2f, sparse = %f\n", i, obj.sigma, max(max(abs(obj.X - obj.Y))), cpav * 100, obj.sparse);
            end
            if max(max(abs(obj.gU))) < tol && max(max(abs(obj.gV))) < tol
                break
            end
        end
        time = toc;
        obj.elapsed_time = time;
        Et = obj.X' * obj.AtA_mul(obj.X);
        Et = max(abs(Et - diag(diag(Et))) - obj.Delta, 0);
        Ct = obj.X' * obj.AtA_mul(obj.X);
        cpav = (sum(diag(Ct)) - norm(Ct - diag(diag(Ct)), 'fro')) / sum(sum(obj.A .^ 2));
        cpav = cpav * 100;
        sparity = obj.record.sparse(obj.iter_num-1);
        ortho = max(max(abs(obj.X' * obj.X - eye(obj.r))));
        eigortho = max(max(abs(Et)));

        fprintf('ALMSSN:Iter ***  Fval *** CPU  **** sparsity *** ortho      ***     eigortho *** opt_norm  ***  CPAV  \n');
        print_format = ' %i     %1.5e    %1.2f     %1.4f           %1.3e      %1.3e     %1.3e      %2.2f \n';
        fprintf(1,print_format, obj.iter_num, obj.record.loss(obj.iter_num-1), time, obj.record.sparse(obj.iter_num-1), ortho, eigortho, obj.record.gX(obj.iter_num -1), cpav);
    end

    function obj = init(obj, A, options)
        obj.A = A;
        obj.Delta = options.Delta;
        obj.verbose = options.verbose;
        obj.mu = options.mu;
        obj.n = options.n;
        obj.r = options.r;
        obj.tau = options.tau;
        obj.sigma = 1; %(svds(full(A), 1) ^ 2) * 0.1; % * 0.75
        obj.maxinner_iter = options.maxinner_iter;
        obj.maxcg_iter = options.maxcg_iter;
        obj.maxnewton_iter = options.maxnewton_iter;
        obj.gradnorm_decay = options.gradnorm_decay;
        obj.gradnorm_min = options.gradnorm_min;
        obj.sigma_factor = options.sigma_factor;
        obj.manifold = stiefelfactory(obj.n, obj.r);

        obj.X = zeros([obj.n obj.r]);
        obj.Y = zeros([obj.n obj.r]);
        obj.U = zeros([obj.n obj.r]);
        obj.W = zeros([obj.r 2 * obj.r]);
        obj.V = zeros([obj.r 2 * obj.r]);
        obj.iter_num = 1;
        obj.last_newtonstatus = 0;
        obj.residual = 1.0e10;
        obj.cg_regularizer = 4;

        obj.record.gX = [];
        obj.record.gY = [];
        obj.record.gU = [];
        obj.record.gV = [];
        obj.record.gradnorm = [];
        obj.record.sigma = [];
        obj.record.sparse = [];
        obj.record.loss = [];
        obj.record.time = [];
        obj.record.is_newton = [];
    end

end
end
