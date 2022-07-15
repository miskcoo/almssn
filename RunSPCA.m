
addpath ./OptM;
addpath ./almssn;

mkdir('log');
dfile=['log/spca_', datestr(now,'yyyy-mm-dd-HHMMSS'),'.log'];
diary(dfile);
diary on;

RunRep(500,  50, 20, 1.00, 1, 20);
RunRep(1000, 50, 20, 1.00, 1, 20);
RunRep(1500, 50, 20, 1.00, 1, 20);
RunRep(2000, 50, 20, 1.00, 1, 20);
RunRep(2500, 50, 20, 1.00, 1, 20);
RunRep(3000, 50, 20, 1.00, 1, 20);
%%
RunRep(2000, 50, 5,  1.00, 1, 20);
RunRep(2000, 50, 10, 1.00, 1, 20);
RunRep(2000, 50, 15, 1.00, 1, 20);
RunRep(2000, 50, 25, 1.00, 1, 20);
%
RunRep(2000, 50, 20, 0.25, 1, 20);
RunRep(2000, 50, 20, 0.50, 1, 20);
RunRep(2000, 50, 20, 0.75, 1, 20);
RunRep(2000, 50, 20, 1.25, 1, 20);

diary off;

function RunRep(n, m, r, lambda, start, reps)

    T0 = []; T1 = []; 
    S0 = []; S1 = []; 
    F0 = []; F1 = []; 


    for num = start:reps

        fprintf(1, '\n   --------- #%d/%d, n = %d, r = %d, mu = %.2f ---------\n', num, reps, n, r, lambda);

        A = randn(m, n);
        A = A - repmat(mean(A, 1), m, 1);
        A = A ./ repmat(sqrt(sum(A .* A)), m, 1);
        
        [U, S, V] = svd(A, 'econ');
        D = diag(S(1:r, 1:r));
        D = sort(abs(randn([m 1])) .^ 4) + 1.0e-5;

        A = U * diag(D) * V';
        A = A - repmat(mean(A, 1), m, 1);
        A = A ./ repmat(sqrt(sum(A .* A)), m, 1);

        [phi_init,~] = svd(randn(n,r),0);  % random intialization

        Xinitial.main = phi_init;

        tol = 1e-8;

        % =================== LSq-I ===================
        option_almssn.mu = lambda;
        option_almssn.n = n; option_almssn.r = r;
        option_almssn.tau = 0.99;
        option_almssn.sigma_factor = 1.25;
        option_almssn.maxinner_iter = 300;
        option_almssn.maxnewton_iter = 10;
        option_almssn.maxcg_iter = 300;
        option_almssn.retraction = 'retr';
        option_almssn.algname = 'LSq-I';
        option_almssn.gradnorm_decay = 0.95; 
        option_almssn.gradnorm_min = 1.0e-13;
        option_almssn.verbose = 0;
        option_almssn.LS = 1;
        option_almssn.x_init = Xinitial.main;
        almssn_solver = SPCANewtonNew();
        almssn_solver = almssn_solver.init(A, option_almssn);

        tic;
        almssn_solver = almssn_solver.run(tol);
        time0 = toc;

        record0 = almssn_solver.record;
        xopt0 = almssn_solver.X; fv0 = record0.loss(end); sparsity0 = record0.sparse(end); 

        % =================== LSq-II ===================
        option_almssn.LS = 2;
        option_almssn.x_init = Xinitial.main;
        option_almssn.algname = 'LSq-II';
        almssn_solver = SPCANewtonNew();
        almssn_solver = almssn_solver.init(A, option_almssn);

        tic;
        almssn_solver = almssn_solver.run(tol);
        time1 = toc;

        record1 = almssn_solver.record;
        xopt1 = almssn_solver.X; fv1 = record1.loss(end); sparsity1 = record1.sparse(end); 


        T0 = [T0; time0];
        F0 = [F0; fv0];
        S0 = [S0; sparsity0 * 100];

        T1 = [T1; time1];
        F1 = [F1; fv1];
        S1 = [S1; sparsity1 * 100];

    end

    fprintf(1, '=========== Summary: n = %d, r = %d, mu = %.3f ==========\n', n, r, lambda);
    fprintf(1, 'LS-I:    time = %.3fs,  sparsity = %.2f,  loss = %.2f\n',       mean(T0), mean(S0), mean(F0));
    fprintf(1, 'LS-II:   time = %.3fs,  sparsity = %.2f,  loss = %.2f\n\n\n\n', mean(T1), mean(S1), mean(F1));
end
