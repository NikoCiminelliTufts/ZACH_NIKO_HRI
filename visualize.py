# imports
from mmvp_behavior.options import Options
from master_mmvp.model import Model
import glob
import os
from PIL import Image
import numpy as np

# visualization routine
if __name__ == "__main__":
    # parse and set normal and visualization specific arguments
    opt = Options()
    opt.parser.add_argument('--vis_raw_input_dir', type=str, default="", help='directory with raw data to visualize output')
    opt = opt.parse()
    opt.baseline = False
    opt.sequence_length = 20
    
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
        folder_list= [{'vision': folder}]
        resultlist, _ = model.predict(folder_list)

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
