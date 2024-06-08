from mmseg.apis import init_model, inference_model, show_result_pyplot
import os
config_path = './configs/custom/dmscan_shorelines.py'
checkpoint_path = './work_dirs/dmscan_shorelines/dmscan-b-11-768/iter_160000.pth'
img_path = './data/shorelines/img_dir/inference_2/'
images = [f for f in os.listdir(img_path) if f.endswith(".png")]
out_dir = "work_dirs/inference_results_2/results_dmscan-b-768"
# build the model from a config file and a checkpoint file
model = init_model(config_path, checkpoint_path, device='cuda:0')
for i in range(len(images)):
    print("inference on image: ", images[i])
    # inference on given image
    result = inference_model(model, img_path + images[i])
    
    print("result ready")
    # # save the visualization result, the output image would be found at the path `work_dirs/result.png`
    vis_iamge = show_result_pyplot(model, img_path + images[i], result, save_dir=out_dir, out_file=f'{out_dir}/result_{images[i]}', wait_time=0.1)


