# imports
from mmvp_behavior.options import Options
from master_mmvp.model import Model
import master_mmvp.metrics as metrics
import glob
import os
from PIL import Image
import numpy as np

## predict takes an Options object from options.py
# and computes and stores a series of predicted images for all input data
def predict(opt):
    # validate arguments
    if opt.pretrained_model == '':
        raise Exception("You must specify a trained model to make predictions with.")

    # re-create model
    model = Model(opt)
    model.load_weight()

    # predict images by trial
    folder_glob = glob.glob(os.path.join(opt.vis_raw_input_dir, 'v*', '*', '*', '*', 'lift*'))
    for folder in folder_glob:
        folder = str(folder)
        reformatted_folder= [{'vision': folder}]
        resultlist, _ = model.predict(reformatted_folder)

        # save images within the trial
        trial_path = folder[folder.find("vision"):]
        print(trial_path)
        i = 0
        for block in resultlist:
            vision = block[0]
            for frame in vision:
                array = frame.cpu().detach().numpy().squeeze()
                rgb_last = np.moveaxis(array,0,-1)
                image_array = (rgb_last*255).astype("uint8")
                
                image = Image.fromarray(image_array)
                save_path = os.path.join(opt.output_dir, trial_path)
                if(os.path.exists(save_path) == False):
                    os.makedirs(save_path)
                save_file = os.path.join(save_path, f'image{i}.png')
                image.save(save_file)
                i += 1

## evaluate takes an Options object from options.py
# and computes the SSIM of each predicted image vs its raw counterpart
def evaluate(opt):
    # gather data
    raw_folders = glob.glob(os.path.join(opt.vis_raw_input_dir, 'v*', '*', '*', '*', 'lift*'))
    predict_folders = glob.glob(os.path.join(opt.output_dir, 'v*', '*', '*', '*', 'lift*'))

    for folder_i in range(1):
        raw_folder = raw_folders[folder_i]
        predict_folder = predict_folders[folder_i]

        raw_images = glob.glob(os.path.join(raw_folder,"*")).sort()
        predict_images = glob.glob(os.path.join(predict_folder,"*")).sort()

        for image_i in range(len(predict_images)):
            print(metrics.calc_ssim(raw_images[image_i], predict_images[image_i]))

## visualization routine
if __name__ == "__main__":
    # parse and set normal and visualization specific arguments
    opt = Options()
    opt.parser.add_argument('--vis_raw_input_dir', type=str, default="", help='directory with raw data to visualize output')
    opt.parser.add_argument('--predict', action="store_true", help="use this if you want to generate predicted images")
    opt.parser.add_argument('--evaluate', action="store_true", help="use this if you want to generate plots evaluating predictions")
    opt = opt.parse()
    opt.baseline = False
    opt.sequence_length = 20
    
    if opt.predict == False:
        predict(opt)

    if opt.evaluate == True:
        evaluate(opt)
    
    
