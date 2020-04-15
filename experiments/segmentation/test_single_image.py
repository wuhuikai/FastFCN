###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os

import torch
import torchvision.transforms as transform

import encoding.utils as utils

from PIL import Image

from encoding.nn import BatchNorm
from encoding.datasets import datasets
from encoding.models import get_model, get_segmentation_model, MultiEvalModule

from .option import Options


def test(args):
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([.485, .456, .406], [.229, .224, .225])])
    # model
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(args.model, dataset = args.dataset,
                                       backbone = args.backbone, dilated = args.dilated,
                                       lateral = args.lateral, jpu = args.jpu, aux = args.aux,
                                       se_loss = args.se_loss, norm_layer = BatchNorm,
                                       base_size = args.base_size, crop_size = args.crop_size)
        # resuming checkpoint
        if args.resume is None or not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    print(model)
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == 'citys' else \
        [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    if not args.ms:
        scales = [1.0]
    num_classes = datasets[args.dataset.lower()].NUM_CLASS
    evaluator = MultiEvalModule(model, num_classes, scales=scales, flip=args.ms).cuda()
    evaluator.eval()

    img = input_transform(Image.open(args.input_path).convert('RGB')).unsqueeze(0)

    with torch.no_grad():
        output = evaluator.parallel_forward(img)[0]
        predict = torch.max(output, 1)[1].cpu().numpy()
    mask = utils.get_mask_pallete(predict, args.dataset)
    mask.save(args.save_path)


if __name__ == "__main__":
    option = Options()
    option.parser.add_argument('--input-path', type=str, required=True, help='path to read input image')
    option.parser.add_argument('--save-path', type=str, required=True, help='path to save output image')
    args = option.parse()

    torch.manual_seed(args.seed)

    test(args)
