import torch
import torchvision.models as models
import model

if __name__ == '__main__':
    dict = torch.load('checkpoint/maskrcnn_resnet50_fpn_coco.pth')
    # mrcnn = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # print(mrcnn)
    # im = torch.zeros((3, 480, 640), dtype=torch.float)
    # mrcnn.eval()
    # x = mrcnn([im])
    # print(x)

    # for key in dict:
    #     print(key)

    conv = {'backbone.body.conv1': 'fpn.C1.0',
            'backbone.body.bn1': 'fpn.C1.1',
            'backbone.body.layer1': 'fpn.C2',
            'backbone.body.layer2': 'fpn.C3',
            'backbone.body.layer3': 'fpn.C4',
            'backbone.body.layer4': 'fpn.C5',
            'backbone.fpn.inner_blocks.0': 'fpn.P2_conv1',
            'backbone.fpn.inner_blocks.1': 'fpn.P3_conv1',
            'backbone.fpn.inner_blocks.2': 'fpn.P4_conv1',
            'backbone.fpn.inner_blocks.3': 'fpn.P5_conv1',
            'backbone.fpn.layer_blocks.0': 'fpn.P2_conv2.0',
            'backbone.fpn.layer_blocks.1': 'fpn.P3_conv2.0',
            'backbone.fpn.layer_blocks.2': 'fpn.P4_conv2.0',
            'backbone.fpn.layer_blocks.3': 'fpn.P5_conv2.0',
            # 'rpn.head.conv': 'rpn.conv_shared',
            # 'rpn.head.cls_logits': 'rpn.conv_class',
            # 'rpn.head.bbox_pred': 'rpn.conv_bbox',
            }
    dict_conv = {}
    for key, val in dict.items():
        for conv_key, conv_val in conv.items():
            if conv_key in key:
                dict_conv[key.replace(conv_key, conv_val)] = val

    for key in dict_conv.keys():
        print(key)
    torch.save(dict_conv, 'checkpoint/resnet50_fpn_coco.pth')
