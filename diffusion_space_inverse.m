function err = diffusion_space_inverse(f, F)
%numNeighbors_insensor = 25;
finspace = F.fspace;
feats = F.feats;
eps = F.eps;
numNeighbors_insensor = F.numNeighbors_insensor;
h=(f*ones(1,size(feats,2))-feats);
dists_f = sqrt(sum(h.*h))';
b = sort(dists_f);
if b(1)==0
    eps_f = mean(b(2:numNeighbors_insensor+1).^2);
else
    eps_f = mean(b(1:numNeighbors_insensor).^2);
end
%eps_f = mean(dists_f.^2);
comp = (dists_f.^2)./ sqrt(eps_f*eps');
comp = exp(-comp);
comp = comp / sum(comp);
err = abs(norm(comp-finspace));
end