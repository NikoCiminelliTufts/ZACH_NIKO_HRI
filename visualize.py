# imports
from mmvp_behavior.data.make_data import compute_behavior
from mmvp_behavior.options import Options
from mmvp_behavior.model import Model
from mmvp_behavior.data.make_data import IMG_SIZE
from mmvp_behavior.data.make_data import BEHAVIORS
import master_mmvp.metrics as metrics
import glob
import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

## check version
if sys.version_info[0] < 3:
    raise Exception("You must run visualize.py with python 3. You are currently running " + sys.version)

## predict takes an Options object from options.py
# and computes and stores a series of predicted images for all input data
def predict(opt):
    # validate arguments
    if opt.pretrained_model == '':
        raise Exception("You must specify a trained model to make predictions with.")

    # re-create model
    model = Model(opt)
    model.load_weight()

    # find raw input files
    # looking for directory structure:
    # vision_data*/object name/trial num/exec num/behavior name
    folder_glob = glob.glob(os.path.join(opt.vis_raw_input_dir, 'v*', '*', '*', '*', '*'))
    if len(folder_glob) == 0:
        print("Error: vis_raw_input_dir not properly set")
        return
    for folder in folder_glob:
        
        # bypass unused behaviors
        behavior_in_folder_name = folder.split(os.sep)[-1]
        if behavior_in_folder_name not in BEHAVIORS:
            continue

        # select object
        object_in_folder_name = folder.split(os.sep)[-4]

        # bypass a bad exec
        folder = str(folder)
        relative_folder = folder.split(os.sep)[-4:]
        if not os.access(os.path.join(opt.vis_raw_input_dir,'rc_data',*relative_folder),os.F_OK):
            continue

        # predict images by trial
        reformatted_folder = [{'vision': folder}]
        out_behavior_npys = compute_behavior(BEHAVIORS, behavior_in_folder_name, object_in_folder_name)
        resultlist, _ = model.predict(reformatted_folder, out_behavior_npys)#need to add room for descriptors

        # save images within the trial
        trial_path = folder[folder.find("vision"):]
        #trial_path = os.join(trial_path.split(os.sep)[:-1])
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
            break # skip additional blocks for simplicity

def namesort(a):
    return len(a), a

## evaluate takes an Options object from options.py
# and computes the SSIM of each predicted image vs its raw counterpart
def evaluate(opt, debug=False):
    # gather data
    predict_folders = glob.glob(os.path.join(opt.output_dir, 'v*', '*', '*', '*', '*'))
    if len(predict_folders) == 0:
        print("Error: output_dir not properly set")
        return

    # extract data
    all_scores = []
    for folder_i in range(len(predict_folders)):
        # determine which folder to operate on next and set paths
        relative_folder = predict_folders[folder_i].split(os.sep)[-5:]
        if type(opt.behavior) != type(None) and relative_folder[-1] not in opt.behavior[0]:
            print("skipping " + relative_folder[-1])
            continue
        raw_folder = os.path.join(opt.vis_raw_input_dir, *relative_folder)
        predict_folder = predict_folders[folder_i]
        print(predict_folder)

        # skip folder if not included in desired output
        include = False
        data_filename = "_".join(relative_folder[1:])
        print(data_filename)
        train_path = os.path.join(opt.data_dir,"train",data_filename + "*")
        test_path = os.path.join(opt.data_dir,"test",data_filename + "*")
        
        if opt.use_train and len(glob.glob(train_path)) > 0:
            include = True
        if opt.use_test and len(glob.glob(test_path)) > 0:
            include = True
        if include == False:
            print("skipping")
            continue
        
        # load images making sure ordering is consistent
        raw_images = glob.glob(os.path.join(raw_folder,"*.jpg"))
        raw_images.sort(key=namesort)
        predict_images = glob.glob(os.path.join(predict_folder,"*.png"))
        predict_images.sort(key=namesort)
        #print("\n".join(predict_images))

        # evaluate ssim
        trial_scores = []
        for image_i in range(len(predict_images)):
            raw_image = Image.open(raw_images[image_i])
            raw_image = raw_image.resize(IMG_SIZE)
            raw_image = raw_image.convert("L")
            predicted_image = Image.open(predict_images[image_i])
            predicted_image = predicted_image.convert("L")
            score, _ = metrics.calc_ssim(np.asarray(raw_image), np.asarray(predicted_image), multichannel=False)
            trial_scores.append(score)

            if debug == True:
                fig, ax = plt.subplots(1,2)
                ax[0].imshow(raw_image)
                ax[1].imshow(predicted_image)
                ax[0].set(title=score)
                plt.show()
                input()

        all_scores.append(trial_scores)

    all_scores = np.asarray(all_scores)
    #print(all_scores)
    print(np.nanmean(all_scores,0))

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
    # if the input data has descriptors, must prepare raw data with descriptors using
    #     --use_descriptor
    opt = Options()
    opt.parser.add_argument('--vis_raw_input_dir', type=str, default="", help='directory with raw data to visualize output')
    opt.parser.add_argument('--predict', action="store_true", help="use this if you want to generate predicted images")
    opt.parser.add_argument('--evaluate', action="store_true", help="use this if you want to generate plots evaluating predictions")
    opt.parser.add_argument('--behavior', nargs="+", action="append", default=None, help="")
    opt.parser.add_argument('--debug', action="store_true", help="")
    opt.parser.add_argument('--use_train', action="store_true", help="Include training data in evaluation.")
    opt.parser.add_argument('--use_test', action="store_true", help="Include test data in evaluation.")
    opt = opt.parse()
    opt.baseline = False
    opt.sequence_length = 20
    
    if opt.predict == True and opt.evaluate == False:
        predict(opt)

    elif opt.evaluate == True and opt.predict == False:
        if not opt.use_train and not opt.use_test:
            raise Exception("You must specify either --use_train, --use_test, or both.")
        evaluate(opt, opt.debug)

    else:
        raise Exception("You must specify exactly one of --predict or --evaluate.")
    
    
