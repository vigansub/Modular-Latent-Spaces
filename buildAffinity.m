function [Wc, Wd, fc, fd, eps1, eps2] = buildAffinity(feat_C, feat_D, numNeighbors_insensor)
train_Feats = 1:size(feat_C,1);%one could use a subset of given features

num_train_feats = length(train_Feats);
red_dim = min(100,num_train_feats);

[~,Su1,V1] = svd(feat_C(train_Feats,:)');
[~,Su2,V2] = svd(feat_D(train_Feats,:)');
%fc and fd are reduced dictionary representations of features in shapes C
%and D respectively.
fc = Su1(1:red_dim,1:red_dim)*V1(:,1:red_dim)';
fd = Su2(1:red_dim,1:red_dim)*V2(:,1:red_dim)';

d1 = pdist2(feat_C(train_Feats,:),feat_C(train_Feats,:), 'euclidean');
d2 = pdist2(feat_D(train_Feats,:),feat_D(train_Feats,:), 'euclidean');

[~, ~, alg1_K1, eps1]=SimpleDiffusion(d1,1,numNeighbors_insensor);
[~, ~, alg1_K2, eps2]=SimpleDiffusion(d2,1,numNeighbors_insensor);

Wc = alg1_K1;
Wd = alg1_K2;
s = rank_completion(Wc, Wd);

Wc = Wc(s,s);
Wd = Wd(s,s);
eps1 = eps1(s);
eps2 = eps2(s);
fc = fc(:,s);
fd = fd(:,s);

end