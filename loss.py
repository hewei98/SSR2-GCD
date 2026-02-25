import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class TotalCodingRate(nn.Module):
    def __init__(self, eps=0.01):
        super(TotalCodingRate, self).__init__()
        self.eps = eps
        
    def compute_discrimn_loss(self, W):
        """Discriminative Loss."""
        p, m = W.shape  #[d, B]
        I = torch.eye(p,device=W.device)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.
    
    def forward(self,X):
        return - self.compute_discrimn_loss(X.T)


class CompressLoss(nn.Module):
    def __init__(self, eps=0.2):
        super(CompressLoss, self).__init__()
        self.eps = eps
    def compute_compress_loss(self, W, Pi): # W:dxb, Pi:bxb
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p, device=W.device).expand((k, p, p))
        trPi = Pi.sum(2) + 1e-8
        scale = (p / (trPi * self.eps)).view(k, 1, 1)
        W = W.view((1, p, m))
        log_det = torch.logdet(I + scale * W.mul(Pi).matmul(W.transpose(1, 2)))
        compress_loss = (trPi.squeeze() * log_det / (2 * m)).sum()

        return compress_loss

    def forward(self, X, Y, num_classes):
        # This function support Y as label integer or membership probablity.
        if len(Y.shape) == 1:
            # if Y is a label vector
            if num_classes is None:
                num_classes = Y.max() + 1
            Pi = torch.zeros((num_classes, 1, Y.shape[0]), device=Y.device)
            for indx, label in enumerate(Y):
                Pi[label, 0, indx] = 1
        else:
            # if Y is a probility matrix
            if num_classes is None:
                num_classes = Y.shape[1]
            Pi = Y.T.reshape((num_classes, 1, -1))

        W = X.T
        compress_loss = self.compute_compress_loss(W, Pi)

        total_loss =  compress_loss
        return total_loss

def simgcd_distillLoss(student_out, teacher_out):
    total_loss = 0
    n_loss_terms = 0
    for iq, q in enumerate(teacher_out):
        for v in range(len(student_out)):
            if v == iq:
                # we skip cases where student and teacher operate on the same view
                continue
            loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
            total_loss += loss.mean()
            n_loss_terms += 1
    total_loss /= n_loss_terms
    return total_loss


def simgcd_loss(logits_image, logits_text, labels, mask, teacher_temp_schedule, epoch, args):


    sup_labels = torch.cat([labels[mask] for _ in range(2)], dim=0)

    ## image loss
    sup_logits_image = torch.cat([f[mask] for f in (logits_image / args.tau_s).chunk(2)], dim=0)
    loss_cls_image = nn.CrossEntropyLoss()(sup_logits_image, sup_labels)

    # Note: args.tau_u is the student_temp in SimGCD, tau_t_(start & end) is the teacher_temp in SimGCD, please refer to SimGCD/model.py/DistillLoss()
    student_out_image = logits_image / args.tau_u
    student_out_image = student_out_image.chunk(2)
    teacher_out_image = logits_image.detach()
    teacher_out_image = F.softmax(teacher_out_image / teacher_temp_schedule[epoch], dim=-1)
    teacher_out_image = teacher_out_image.chunk(2)
    loss_cluster_image = simgcd_distillLoss(student_out_image, teacher_out_image)

    avg_probs_image = (logits_image / args.tau_u).softmax(dim=1).mean(dim=0)
    me_max_loss_image = - torch.sum(torch.log(avg_probs_image**(-avg_probs_image))) + math.log(float(len(avg_probs_image)))
    loss_cluster_image += args.memax_weight * me_max_loss_image

    loss_base_image = args.lambda_loss * loss_cls_image + (1-args.lambda_loss) * loss_cluster_image

    ## text loss
    sup_logits_text = torch.cat([f[mask] for f in (logits_text / args.tau_s).chunk(2)], dim=0)
    loss_cls_text = nn.CrossEntropyLoss()(sup_logits_text, sup_labels)  

    student_out_text = logits_text / args.tau_u
    student_out_text = student_out_text.chunk(2)
    teacher_out_text = logits_text.detach()
    teacher_out_text = F.softmax(teacher_out_text / teacher_temp_schedule[epoch], dim=-1)
    teacher_out_text = teacher_out_text.chunk(2)
    loss_cluster_text = simgcd_distillLoss(student_out_text, teacher_out_text)

    avg_probs_text = (logits_text / args.tau_u).softmax(dim=1).mean(dim=0)
    me_max_loss_text = - torch.sum(torch.log(avg_probs_text**(-avg_probs_text))) + math.log(float(len(avg_probs_text)))
    loss_cluster_text += args.memax_weight * me_max_loss_text 

    loss_base_text = args.lambda_loss * loss_cls_text + (1-args.lambda_loss) * loss_cluster_text     

    return loss_base_image + loss_base_text


def coteaching_pseudolabel_loss(selected_samples, logits, images_id, args):

    pseudo_labels = []
    selected_logits = []

    # Duplicate the image ID list for 2 views
    doubled_images_id = images_id + images_id
    if selected_samples:
        # Collect pseudo labels and corresponding logits
        for idx, img_id in enumerate(doubled_images_id):
            if img_id in selected_samples:
                pseudo_labels.append(selected_samples[img_id])
                selected_logits.append(logits[idx])

    # Compute cross-entropy loss if there are pseudo labels available
    if pseudo_labels:
        pseudo_labels = torch.tensor(pseudo_labels, device=args.device)
        selected_logits = torch.stack(selected_logits).to(args.device)
        selected_logits /= args.tau_s  # Apply temperature scaling
        loss = nn.CrossEntropyLoss()(selected_logits, pseudo_labels)
    else:
        # Return zero loss if no pseudo labels are present
        loss = torch.zeros((), device=args.device)

    return loss
