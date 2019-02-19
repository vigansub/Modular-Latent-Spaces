function D_feats = latent_transfer(C_feats, fc, fd, Wc, Wd, numNeighbors_insensor, eps1, eps2)
% Inputs
% Shape C has N vertices, shape D has M vertices. Goal is to transfer k descriptors on shape C to shape D using the idea of modular latent spaces
% C_feats - Nxk matrix, N vertices on the first shape, k such descriptors in action
% fc - nxp input descriptors on the first shape C (reduced version of original descriptors - obtained from buildAffinity)
% fd - nxp input descriptors on the second shape D (reduced version of original descriptors - obtained from buildAffinity)
% Wc - pxp transporter from latent shape space to embedding space of C
% Wd - pxp transporter from latent shape space to embedding space of D
% numNeighbors_insensor - numerical value considering the neighborhood of every descriptor
% eps1, eps2 - epsilon values regarding graph structures of C and D respectively
% Outputs
% D_feats - Mxk matrix, M vertices on the second shape, k corresponding descriptors obtained.	

L_feats = latent_transfer_emb(C_feats, fc, Wc, numNeighbors_insensor, eps1);
D_feats = latent_transfer_deemb(L_feats, fd, Wd, numNeighbors_insensor, eps2);
end