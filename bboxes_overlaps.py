import torch
from torch import Tensor
def bboxes_overlaps(bboxes1: Tensor, bboxes2: Tensor, mode='iou') -> Tensor:
    """计算iou

    Args:
        bboxes1 (Tensor): [B, row1, 4]
        bboxes2 (Tensor): [B, row2, 4]
        mode (str, optional):Defaults to 'iou'.

    Returns:
        Tensor: iou
    """
    assert (bboxes1.size(-1)==4 or bboxes1.size(0)==0) # 判断最后维度是否是4，bbox的数量是否为0
    assert (bboxes2.size(-1)==4 or bboxes2.size(0)==0)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]  # 判断batch维度是否相同
    
    # 计算单独面积
    areas1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1]) # shape: [B, row1]
    areas2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1]) # shape: [B, row2]
    
    # 计算交集
    lt = torch.max(bboxes1[..., :, None, :2], bboxes2[..., None, :, :2])  # shape: [B, row1, row2, 2]
    rb = torch.min(bboxes1[..., :, None, 2:], bboxes2[..., None, :, 2:]) # shape: [B, row1, row2, 2]
    wh = lt - rb  # shape: [B, row1, row2, 2]
    wh = wh.clamp(min=0, max=None)
    overlaps = wh[..., 0] * wh[..., 1]  # shape: [B, row1, row2]

    
    # 计算并集
    union = areas1[..., :, None] + areas2[..., None, :] -overlaps
    eps = 1e-6
    eps = overlaps.new_tensor(eps)
    union = torch.max(union, eps)
    
    iou = overlaps/union  # overlaps大于0，union非零
    
    if mode in ['iou']:
        return iou
    
    
if __name__ == "__main__":
    bboxes1 = torch.randn(10, 5, 4)
    bboxes2 = torch.randn(10, 29925, 4)
    bboxes1.shape
    bboxes_overlaps(bboxes1, bboxes2)
    
    
    
