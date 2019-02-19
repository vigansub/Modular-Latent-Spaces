function s = rank_completion(Wc, Wd)
[~,s1] = licols(Wc);
[~,s1] = licols(Wc(:,s1)');
[~,s2] = licols(Wd);
[~,s2] = licols(Wd(:,s2)');
s = intersect(s1,s2);
end