from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import mmcv
from matplotlib import pyplot as plt


def main():
    parser = ArgumentParser()
    parser.add_argument('--img', default='/home/airstudio/code/2.3/demo/demo_wpj/tt1.jpg',help='Image file')
    # parser.add_argument('--img', default='/home/airstudio/code/2.3/demo/demo_yyr/2.jpg',help='Image file')

    parser.add_argument('--config', default='/home/airstudio/code/2.3/a_configs/dior/teacher/atss101_dior.py',help='Config file')
    parser.add_argument('--checkpoint', default='/home/airstudio/code/2.3/work_dirs/atss101_dior2/epoch_24.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)
    # test a single image
    result = inference_detector(model, args.img)
    # show the results
    # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)

    print('it is ok')


def show_result_pyplot(model, img, result, score_thr=0.3, fig_size=(15, 15)):
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size, dpi =120)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.axis('off')
    plt.savefig("./vis/2.jpg", bbox_inches = 'tight', pad_inches = 0.0)

if __name__ == '__main__':
    main()
