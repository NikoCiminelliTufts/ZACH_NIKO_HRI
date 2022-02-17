# imports
from operator import mul
from mmvp_behavior.options import Options
from master_mmvp.model import Model
from master_mmvp.data.make_data import IMG_SIZE
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
    folder_glob = glob.glob(os.path.join(opt.vis_raw_input_dir, 'v*', '*', '*', '*', '*'))
    if len(folder_glob) == 0:
        print("Error: vis_raw_input_dir not properly set")
        return
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
    if len(raw_folders) == 0:
        print("Error: vis_raw_input_dir not properly set")
        return
    if len(predict_folders) == 0:
        print("Error: output_dir not properly set")
        return

    # extract data
    for folder_i in range(1):
        raw_folder = raw_folders[folder_i]
        predict_folder = predict_folders[folder_i]

        raw_images = glob.glob(os.path.join(raw_folder,"*.jpg"))
        raw_images.sort()
        predict_images = glob.glob(os.path.join(predict_folder,"*.png"))
        predict_images.sort()

        # evaluate ssim
        for image_i in range(len(predict_images)):
            raw_image = Image.open(raw_images[image_i])
            raw_image = raw_image.resize(IMG_SIZE)
            raw_image = raw_image.convert("L")
            predicted_image = Image.open(predict_images[image_i])
            predicted_image = predicted_image.convert("L")
            score, _ = metrics.calc_ssim(np.asarray(raw_image), np.asarray(predicted_image), multichannel=True)
            print(score)

## visualization routine
if __name__ == "__main__":
    # parse and set normal and visualization specific arguments
    # expects arguments as follows:
    #     --vis_raw_input_dir [path to raw data]
    #     --data_dir [path to preprocessed data]
    #     --output_dir [path to predicted images]
    # if doing prediction, also requires
    #     --predict
    #     --pretrained_model [path to the model to predict with]
    # if doing evaluation, also requires
    #     --evaluate
    opt = Options()
    opt.parser.add_argument('--vis_raw_input_dir', type=str, default="", help='directory with raw data to visualize output')
    opt.parser.add_argument('--predict', action="store_true", help="use this if you want to generate predicted images")
    opt.parser.add_argument('--evaluate', action="store_true", help="use this if you want to generate plots evaluating predictions")
    opt = opt.parse()
    opt.baseline = False
    opt.sequence_length = 20
    
    if opt.predict == True:
        predict(opt)

    if opt.evaluate == True:
        evaluate(opt)
    
    
