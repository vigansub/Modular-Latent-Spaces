function L_feats = latent_transfer_emb(C_feats, fc, Wc, numNeighbors_insensor, eps1)
L_feats = zeros(size(Wc,1), size(C_feats, 2));
for i=1:size(C_feats,2)
    f = C_feats(:,i);
    
    h = f*ones(1,size(fc,2))-fc;
    dists_f = sqrt(sum(h.*h));
    b = sort(dists_f);
    eps_f = mean(b(2:numNeighbors_insensor+1).^2);
    
    f_inspace = exp(-(dists_f.^2)./ sqrt(eps_f*eps1))';
    f_inspace = f_inspace / sum(f_inspace);
    L_feats(:,i) = pinv(Wc)*f_inspace;
end

end