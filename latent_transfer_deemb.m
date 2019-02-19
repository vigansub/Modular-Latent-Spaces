function D_feats = latent_transfer_deemb(L_feats, fd, Wd, numNeighbors_insensor, eps2)
D_feats = [];
for i=1:size(L_feats,2)
    g_inspace = Wd*L_feats(:,i);
    log_ginspace = -log(g_inspace);
    
    feat_Reps = fd;
    
    l = sqrt(sum((g_inspace*ones(1,size(Wd,2)) - Wd).^2));
    best = find(l == min(l),1);
    g_init = fd(:,best); %mean(feat_Reps')';
    %g_init = fd(:,1);
    F.logspace = log_ginspace;
    F.fspace = g_inspace;
    F.feats = feat_Reps;
    F.eps = eps2;
    F.numNeighbors_insensor = numNeighbors_insensor;
    new_fun = @(f) diffusion_space_inverse(f, F);
    %options = optimoptions('fminunc','Display','iter');
    options = optimoptions('fminunc', 'Display', 'none');
    %try
        g = fminunc(new_fun, g_init, options);
    %catch
    %    g = fminunc(new_fun, mean(feat_Reps')', options);
    %end
    g_full = g;
    D_feats = [D_feats g_full];
end
end