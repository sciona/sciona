"""Checkpoint-compatible DSB classifier using reusable differentiable pooling.

Forward recipe adapted from lfz/DSB2017; MIT notice: docs/licenses/DSB2017-MIT.txt.
Architecture, initialization, parameter names and dropout order are unchanged.
"""
import torch
from sciona.dsb_network import CaseNet as OriginalCaseNet, Net
from sciona.atoms.dl.probability_aggregation import noisy_or_with_baseline


class CaseNet(OriginalCaseNet):
    def forward(self, xlist, coordlist):
        xsize = xlist.size()
        corrdsize = coordlist.size()
        xlist = xlist.view(-1, xsize[2], xsize[3], xsize[4], xsize[5])
        coordlist = coordlist.view(-1, corrdsize[2], corrdsize[3], corrdsize[4], corrdsize[5])
        noduleFeat, nodulePred = self.NoduleNet(xlist, coordlist)
        nodulePred = nodulePred.contiguous().view(corrdsize[0], corrdsize[1], -1)
        featshape = noduleFeat.size()
        centerFeat = self.pool(noduleFeat[:, :, featshape[2] // 2 - 1:featshape[2] // 2 + 1, featshape[3] // 2 - 1:featshape[3] // 2 + 1, featshape[4] // 2 - 1:featshape[4] // 2 + 1])
        centerFeat = centerFeat[:, :, 0, 0, 0]
        out = self.dropout(centerFeat)
        out = self.Relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        out = out.view(xsize[0], xsize[1])
        casePred = noisy_or_with_baseline(out, self.baseline)
        return (nodulePred, casePred, out)


def shared_training_models(topk=5):
    detector=Net()
    return detector,CaseNet(topk,nodulenet=detector)
