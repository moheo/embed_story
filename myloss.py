from __future__ import print_function
import torch


#def my_triplet_margin_loss(anchor,positive, negative, margin=1.0, p=2, eps=1e-6, swap=False):
def my_triplet_margin_loss(x1, x2, margin=1.0, p=2, eps=1e-6):#, swap=False):
    # copy from triplet_marign_loss and modify something
    
    d1 = x1.size()[-1]
    d2 = x2.size()[-1]

    assert d1 == d2, 'input dimension should be matched'
    
    if x1.dim() > 2 and x2.dim() >2:
        x1=x1.view(-1,d1)
        x2=x2.view(-1,d2)

    scores = x1.mm(x2.t())
    #print(scores.size())
    diagonal = torch.diag(scores)
    #print(diagonal.size())

    # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
    cost_x2 = torch.clamp(margin - diagonal.view(1,-1) + scores, min=0.0)
    # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
    cost_x1 = torch.clamp(margin - diagonal.view(-1, 1) + scores, min=0.0)

    #print(diagonal.view(-1, 1).size())
    #print(cost_x2.size(), cost_x1.size())

    #print(scores[:6,:6])
    #print(diagonal[:6])
    #print(cost_x2[:6,:6])
    #print(cost_x1[:6,:6])

    
    # clear diagonals
    for i in range(0,scores.size()[0]):
        cost_x2[i,i] = 0
        cost_x1[i,i] = 0

    return cost_x2.sum() + cost_x1.sum()    
    
    
    #assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    #assert anchor.size() == negative.size(), "Input sizes between anchor and negative must be equal."
    #assert positive.size() == negative.size(), "Input sizes between positive and negative must be equal."
    #assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    ##d_p = F.pairwise_distance(anchor, positive, p, eps)
    ##d_n = F.pairwise_distance(anchor, negative, p, eps)
    #d_p = F.cosine_similarity(anchor, positive, dim=1, eps=eps)
    #d_n = F.cosine_similarity(anchor, negative, dim=1, eps=eps)
    #if swap:
    #    d_s = F.cosine_similarity(positive, negative, dim=1, eps=eps)
    #    d_n = torch.min(d_n, d_s)

    #dist_hinge = torch.clamp(margin + d_p - d_n, min=0.0)
    #loss = torch.mean(dist_hinge)
    #return loss
