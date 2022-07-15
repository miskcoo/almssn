addpath ./OptM;
addpath ./almssn;

mkdir('log');
dfile=['log/ortho_', datestr(now,'yyyy-mm-dd-HHMMSS'),'.log'];
diary(dfile);
diary on;

seed = 1;
rng(seed);

RunRep(2000, 50, 5,  1.0, 1, 20);
RunRep(2000, 50, 10, 1.0, 1, 20);
RunRep(2000, 50, 15, 1.0, 1, 20);
RunRep(2000, 50, 25, 1.0, 1, 20);
RunRep(2000, 50, 30, 1.0, 1, 20);

RunRep(500,  50, 20, 1.0, 1, 20);
RunRep(1000, 50, 20, 1.0, 1, 20);
RunRep(2000, 50, 20, 1.0, 1, 20);
RunRep(4000, 50, 20, 1.0, 1, 20);
RunRep(6000, 50, 20, 1.0, 1, 20);

diary off;

function RunRep(n, m, r, lambda, start, reps)

    for num = start:reps 

        fprintf(1, '\n   --------- #%d/%d, n = %d, r = %d, mu = %.2f ---------\n', num, reps, n, r, lambda);
        
        B = randn(50, n);
        B = B - repmat(mean(B, 1), 50, 1);
        B = normc(B);

        pca_cpav(num) = sum(eigs(B' * B, r)) / sum(sum(B .* B)) * 100;
                
        global_tol = 5.0e-8;
        Delta = 1.0e-8;

        option_almssn.mu = lambda;
        option_almssn.n = n; option_almssn.r = r;
        option_almssn.Delta = Delta;
        option_almssn.tau = 0.25; %0.99;
        option_almssn.sigma_factor = 10; %1.05;
        option_almssn.maxinner_iter = 2000; %100;
        option_almssn.maxnewton_iter = 10;
        option_almssn.maxcg_iter = 1000;
        option_almssn.gradnorm_decay = 0.1; %0.9;
        option_almssn.gradnorm_min = global_tol;
        option_almssn.verbose = 0;
        almssn_solver = SPCAOrthoNewton();
        almssn_solver = almssn_solver.init(B, option_almssn);
        [cpav_almssn(num), time_almssn(num), sparsity_almssn(num), ortho_almssn(num), eigortho_almssn(num), almssn_solver] = almssn_solver.run(global_tol);
    end
    
    fprintf(1, 'Alg: ***  CPU  **** sparsity *** ortho      ***    eigortho  ***  CPAV  \n');
    print_format = ' %s     %1.2f     %1.4f           %1.4e      %1.4e       %2.2f \n';
    fprintf(1, print_format, 'ALMSSN', mean(time_almssn), mean(sparsity_almssn), mean(ortho_almssn), mean(eigortho_almssn), mean(cpav_almssn));
    fprintf(1, 'PCA: cpav = %2.2f\n', mean(pca_cpav));
        
end

