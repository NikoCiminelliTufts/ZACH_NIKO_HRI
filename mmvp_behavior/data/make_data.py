import re
import os
import glob
import random
import numpy as np
import PIL.Image
try:
    from . import plotstft, stft
except:
    from make_spectrogram import plotstft, stft
import argparse
from sklearn.preprocessing import OneHotEncoder

#/Users/ramtin/PycharmProjects/VP/Experiement_object_based_all_behaviors
# DATA_DIR = '../../data/CY101'
# OUT_DIR = '../../data/CY101NPY'
# VIS_DIR = '../../data/VIS/'



STRATEGY = 'object' # object | category | trial

CATEGORIES = ['basket', 'weight', 'smallstuffedanimal', 'bigstuffedanimal', 'metal', 'timber', 'pasta', 'tin', 'pvc',
              'cup', 'can', 'bottle', 'cannedfood', 'medicine', 'tupperware', 'cone', 'noodle', 'eggcoloringcup', 'egg',
              'ball']

OBJECTS = [
    'ball_base', 'can_coke', 'egg_rough_styrofoam', 'noodle_3', 'timber_square', 'ball_basket', 'can_red_bull_large',
    'egg_smooth_styrofoam', 'noodle_4', 'timber_squiggle', 'ball_blue', 'can_red_bull_small', 'egg_wood', 'noodle_5',
    'tin_pokemon',
    'ball_transparent', 'can_starbucks', 'eggcoloringcup_blue', 'pasta_cremette', 'tin_poker', 'ball_yellow_purple',
    'cannedfood_chili',
    'eggcoloringcup_green', 'pasta_macaroni', 'tin_snack_depot', 'basket_cylinder', 'cannedfood_cowboy_cookout',
    'eggcoloringcup_orange',
    'pasta_penne', 'tin_snowman', 'basket_funnel', 'cannedfood_soup', 'eggcoloringcup_pink', 'pasta_pipette', 'tin_tea',
    'basket_green',
    'cannedfood_tomato_paste', 'eggcoloringcup_yellow', 'pasta_rotini', 'tupperware_coffee_beans', 'basket_handle',
    'cannedfood_tomatoes',
    'medicine_ampicillin', 'pvc_1', 'tupperware_ground_coffee', 'basket_semicircle', 'cone_1', 'medicine_aspirin',
    'pvc_2', 'tupperware_marbles',
    'bigstuffedanimal_bear', 'cone_2', 'medicine_bilberry_extract', 'pvc_3', 'tupperware_pasta',
    'bigstuffedanimal_bunny', 'cone_3',
    'medicine_calcium', 'pvc_4', 'tupperware_rice', 'bigstuffedanimal_frog', 'cone_4', 'medicine_flaxseed_oil', 'pvc_5',
    'weight_1',
    'bigstuffedanimal_pink_dog', 'cone_5', 'metal_flower_cylinder', 'smallstuffedanimal_bunny', 'weight_2',
    'bigstuffedanimal_tan_dog',
    'cup_blue', 'metal_food_can', 'smallstuffedanimal_chick', 'weight_3', 'bottle_fuse', 'cup_isu',
    'metal_mix_covered_cup',
    'smallstuffedanimal_headband_bear', 'weight_4', 'bottle_google', 'cup_metal', 'metal_tea_jar',
    'smallstuffedanimal_moose',
    'weight_5', 'bottle_green', 'cup_paper_green', 'metal_thermos', 'smallstuffedanimal_otter', 'bottle_red',
    'cup_yellow', 'timber_pentagon', 'bottle_sobe', 'egg_cardboard', 'noodle_1', 'timber_rectangle', 'can_arizona',
    'egg_plastic_wrap', 'noodle_2', 'timber_semicircle'
]

SORTED_OBJECTS = sorted(OBJECTS)

DESCRIPTORS_BY_OBJECT = {
    "ball_base":["hard","ball","green","small","round","toy"],
    "ball_basket":["squishy","soft","brown","ball","rubber","round","toy"],
    "ball_blue":["ball","blue","plastic","hard","round","toy"],
    "ball_transparent":["ball","blue","transparent","hard","small","round","toy"],
    "ball_yellow_purple":["ball","yellow","purple","multi-colored","soft","small","round","toy"],
    "basket_cylinder":["basket","container","wicker","cylindrical","yellow","light","empty"],
    "basket_funnel":["basket","container","wicker","cylindrical","red","yellow","multi-colored","empty"],
    "basket_green":["basket","green","container","wicker","empty"],
    "basket_handle":["basket","brown","container","wicker","handle","empty"],
    "basket_semicircle":["basket","yellow","container","wicker","empty"],
    "bigstuffedanimal_bear":["squishy","stuffed animal","bear","brown","soft","big","deformable","toy"],
    "bigstuffedanimal_bunny":["squishy","stuffed animal","bunny","brown","soft","big","deformable","toy"],
    "bigstuffedanimal_frog":["squishy","stuffed animal","green","frog","soft","big","deformable","toy"],
    "bigstuffedanimal_pink_dog":["squishy","stuffed animal","pink","dog","soft","big","deformable","toy"],
    "bigstuffedanimal_tan_dog":["squishy","stuffed animal","yellow","dog","soft","big","deformable","toy"],
    "bottle_fuse":["cylindrical","bottle","plastic","empty","container","hard","light","purple"],
    "bottle_google":["cylindrical","water bottle","bottle","plastic","blue","empty","container","hard","light"],
    "bottle_green":["cylindrical","bottle","water bottle","empty","plastic","container","green","hard","light"],
    "bottle_red":["cylindrical","bottle","water bottle","empty","plastic","container","red","squishy","light"],
    "bottle_sobe":["cylindrical","bottle","purple","plastic","hard","container","empty","light","cylindrical"],
    "can_arizona":["green","cylindrical","can","metal","aluminum","large","empty","container","open","cylindrical"],
    "can_coke":["red","cylindrical","can","metal","aluminum","small","empty","container","open","cylindrical"],
    "can_red_bull_large":["blue","cylindrical","can","metal","aluminum","large","empty","container","open","cylindrical"],
    "can_red_bull_small":["blue","cylindrical","can","metal","aluminum","small","empty","container","open","cylindrical"],
    "can_starbucks":["cylindrical","can","metal","aluminum","small","empty","container","open","cylindrical"],
    "cannedfood_chili":["cylindrical","can","full","metal","multicolored"],
    "cannedfood_cowboy_cookout":["cylindrical","can","full","metal","multicolored"],
    "cannedfood_soup":["cylindrical","can","full","metal","multicolored"],
    "cannedfood_tomato_paste":["cylindrical","can","full","metal","multicolored"],
    "cannedfood_tomatoes":["cylindrical","can","full","metal","multicolored"],
    "cone_1":["cone","green","small","short","styrofoam","light"],
    "cone_2":["cone","green","small","short","styrofoam","light"],
    "cone_3":["cone","green","medium","styrofoam","light"],
    "cone_4":["cone","green","tall","styrofoam","light"],
    "cone_5":["cone","green","tall","big","styrofoam"],
    "cup_blue":["cup","blue","plastic","empty"],
    "cup_isu":["cup","red","empty","plastic"],
    "cup_metal":["cup","metal","empty"],
    "cup_paper_green":["cup","paper","green","empty"],
    "cup_yellow":["cup","yellow","plastic","empty"],
    "egg_cardboard":["egg","green","small","cardboard"],
    "egg_plastic_wrap":["egg","plastic","small","green"],
    "egg_rough_styrofoam":["egg","small","styrofoam","green"],
    "egg_smooth_styrofoam":["egg","small","styrofoam","green"],
    "egg_wood":["egg","small","wood","green"],
    "eggcoloringcup_blue":["cup","plastic","small","cylindrical","blue","empty","short","light"],
    "eggcoloringcup_green":["cup","plastic","small","cylindrical","green","empty","short","light"],
    "eggcoloringcup_orange":["cup","plastic","small","cylindrical","orange","empty","short","light"],
    "eggcoloringcup_pink":["cup","plastic","small","cylindrical","pink","empty","short","light"],
    "eggcoloringcup_yellow":["cup","plastic","small","cylindrical","yellow","empty","short","light"],
    "medicine_ampicillin":["medicine","container","full","closed","plastic","pills","hard","transparent","orange","short","small"],
    "medicine_aspirin":["medicine","container","full","closed","plastic","pills","hard","transparent","white","short", "small"],
    "medicine_bilberry_extract":["medicine","container","full","closed","plastic","pills","hard","green","short","small"],
    "medicine_calcium":["medicine","container","full","closed","plastic","pills","hard","transparent","orange","short","small"],
    "medicine_flaxseed_oil":["medicine","container","full","closed","plastic","pills","hard","yellow","short","small"],
    "metal_flower_cylinder":["metal","cylinder","tall","large","empty","container","shiny","closed"],
    "metal_food_can":["metal","cylinder","short","empty","container","shiny","closed"],
    "metal_mix_covered_cup":["metal","can","cylinder","empty","open","shiny"],
    "metal_tea_jar":["metal","can","cylinder","empty","open","shiny"],
    "metal_thermos":["metal","cylinder","bottle","empty","closed","shiny"],
    "noodle_1":["pink","foam","soft","deformable","light","short","toy"],
    "noodle_2":["pink","foam","soft","deformable","light","short","toy"],
    "noodle_3":["pink","foam","soft","deformable","toy","light"],
    "noodle_4":["pink","foam","soft","deformable","toy","tall","light"],
    "noodle_5":["pink","foam","soft","deformable","toy","tall","light","big"],
    "pasta_cremette":["green","multicolored","small","pasta","box","paper","container","full","closed","deformable","rectangular"],
    "pasta_macaroni":["blue","multicolored","pasta","box","paper","container","full","closed","deformable","rectangular"],
    "pasta_penne":["yellow","multicolored","pasta","box","paper","container","full","closed","deformable","large","rectangular"],
    "pasta_pipette":["blue","multicolored","pasta","box","paper","container","full","closed","deformable","large","rectangular"],
    "pasta_rotini":["yellow","multicolored","pasta","box","paper","container","full","closed","deformable","large","rectangular"],
    "pvc_1":["pvc","plastic","cylindrical","round","short","hard","green","pipe","small","light"],
    "pvc_2":["pvc","plastic","cylindrical","round","short","hard","green","pipe","small"],
    "pvc_3":["pvc","plastic","cylindrical","round","short","hard","green","pipe"],
    "pvc_4":["pvc","plastic","cylindrical","round","short","hard","green","pipe","wide"],
    "pvc_5":["pvc","plastic","cylindrical","round","short","hard","green","pipe","wide"],
    "smallstuffedanimal_bunny":["squishy","stuffed animal","soft","small","deformable","toy","light","pink"],
    "smallstuffedanimal_chick":["squishy","stuffed animal","soft","small","deformable","toy","light","green"],
    "smallstuffedanimal_headband_bear":["squishy","stuffed animal","soft","small","deformable","toy","light","brown"],
    "smallstuffedanimal_moose":["squishy","stuffed animal","soft","small","deformable","toy","light","brown"],
    "smallstuffedanimal_otter":["squishy","stuffed animal","soft","small","deformable","toy","light","brown"],
    "timber_pentagon":["tall","wood","brown","stick","block","hard"],
    "timber_rectangle":["tall","wood","brown","stick","block","hard"],
    "timber_semicircle":["tall","wood","brown","stick","block","hard"],
    "timber_square":["tall","wood","brown","stick","block","hard"],
    "timber_squiggle":["tall","wood","brown","stick","block","hard"],
    "tin_pokemon":["box","container","closed","metal","empty","shiny","rectangular","hard","large","tall","yellow","multi-colored"],
    "tin_poker":["box","container","closed","metal","empty","shiny","rectangular","hard","tall","blue","multicolored"],
    "tin_snack_depot":["box","container","closed","metal","empty","shiny","rectangular","hard","large","tall","brown","multicolored"],
    "tin_snowman":["box","container","closed","metal","empty","shiny","rectangular","hard","small","short","blue"],
    "tin_tea":["box","container","closed","metal","empty","shiny","rectangular","hard","small","short","brown"],
    "tupperware_coffee_beans":["red","plastic","container","closed","full","hard"],
    "tupperware_ground_coffee":["red","plastic","container","closed","full","hard"],
    "tupperware_marbles":["red","plastic","container","closed","full","hard"],
    "tupperware_pasta":["red","plastic","container","closed","full","hard"],
    "tupperware_rice":["red","plastic","container","closed","full","hard"],
    "weight_1":["blue","tall","cylindrical","empty","closed","container","plastic"],
    "weight_2":["blue","tall","cylindrical","closed","container","plastic"],
    "weight_3":["blue","tall","cylindrical","full","closed","container","plastic"],
    "weight_4":["blue","tall","cylindrical","full","closed","container","plastic"],
    "weight_5":["blue","tall","cylindrical","full","closed","container","plastic"]
}

DESCRIPTOR_CODES = {'aluminum': 0, 'ball': 1, 'basket': 2, 'bear': 3,
    'big': 4, 'block': 5, 'blue': 6, 'bottle': 7, 'box': 8, 'brown': 9,
    'bunny': 10, 'can': 11, 'cardboard': 12, 'closed': 13, 'cone': 14,
    'container': 15, 'cup': 16, 'cylinder': 17, 'cylindrical': 18, 'deformable': 19,
    'dog': 20, 'egg': 21, 'empty': 22, 'foam': 23, 'frog': 24, 'full': 25, 'green': 26,
    'handle': 27, 'hard': 28, 'large': 29, 'light': 30, 'medicine': 31, 'medium': 32,
    'metal': 33, 'multi-colored': 34, 'multicolor': 35, 'multicolored': 36, 'open': 37,
    'orange': 38, 'paper': 39, 'pasta': 40, 'pills': 41, 'pink': 42, 'pipe': 43,
    'plastic': 44, 'purple': 45, 'pvc': 46, 'rectangular': 47, 'red': 48, 'round': 49,
    'rubber': 50, 'shiny': 51, 'short': 52, 'small': 53, 'soft': 54,
    'squishy': 55, 'stick': 56, 'stuffed animal': 57, 'styrofoam': 58, 'tall': 59,
    'toy': 60, 'transparent': 61, 'water bottle': 62,
    'white': 63, 'wicker': 64, 'wide': 65, 'wood': 66, 'yellow': 67}

BEHAVIORS = ['crush', 'grasp', 'lift_slow', 'shake', 'poke', 'push', 'tap', 'low_drop', 'hold']

TRIALS = ['exec_1', 'exec_2', 'exec_3', 'exec_4', 'exec_5']

crop_stategy = {
    'crush': [16, -5],
    'grasp': [0, -10],
    'lift_slow': [0, -3],
    'shake': [0, -1],
    'poke': [2, -5],
    'push': [2, -5],
    'tap': [0, -5],
    'low_drop': [0, -1],
    'hold': [0, -1],
}


SEQUENCE_LENGTH = 10
STEP = 4
IMG_SIZE = (64, 64)

AUDIO_EACH_FRAME_LENGTH = 8
def read_dir(DATA_DIR):
    visions = glob.glob(os.path.join(DATA_DIR, 'vision*/*/*/*/*'))
    return visions


def convert_audio_to_image(audio_path):
    ims, duration = plotstft(audio_path)
    return ims, duration


def generate_npy_vision(path, behavior, sequence_length):
    """
    :param path: path to images folder,
    :return: numpy array with size [SUB_SAMPLE_SIZE, SEQ_LENGTH, ...]
    """
    files = sorted(glob.glob(os.path.join(path, '*.jpg')))
    img_length = len(files)
    files = files[crop_stategy[behavior][0]:crop_stategy[behavior][1]]
    imglist = []
    for file in files:
        img = PIL.Image.open(file)
        img = img.resize(IMG_SIZE)
        img = np.array(img).transpose([2, 0, 1])[np.newaxis, ...]
        imglist.append(img)
    ret = []
    for i in range(0, len(imglist) - sequence_length, STEP):
        ret.append(np.concatenate(imglist[i:i + sequence_length], axis=0))
    return ret, img_length


def generate_npy_haptic(path1, path2, n_frames, behavior, sequence_length):
    """
    :param path: path to ttrq0.txt, you need to open it before you process
    :param n_frames: # frames
    :return: list of numpy array with size [SEQ_LENGTH, ...]
    :preprocess protocol: 48 bins for each single frame, given one frame, if #bin is less than 48,
                            we pad it in the tail with the last bin value. if #bin is more than 48, we take bin[:48]
    """
    if not os.path.exists(path1):
        return None, None
    haplist1 = open(path1, 'r').readlines()
    haplist2 = open(path2, 'r').readlines()
    haplist = [list(map(float, v.strip().split('\t'))) + list(map(float, w.strip().split('\t')))[1:] for v, w in
               zip(haplist1, haplist2)]
    haplist = np.array(haplist)
    time_duration = (haplist[-1][0] - haplist[0][0]) / n_frames
    bins = np.arange(haplist[0][0], haplist[-1][0], time_duration)
    end_time = haplist[-1][0]
    groups = np.digitize(haplist[:, 0], bins, right=False)

    haplist = [haplist[np.where(groups == idx)][..., 1:][:48] for idx in range(1, n_frames + 1)]
    haplist = [np.pad(ht, [[0, 48 - ht.shape[0]], [0, 0]], mode='edge')[np.newaxis, ...] for ht in haplist]
    haplist = haplist[crop_stategy[behavior][0]:crop_stategy[behavior][1]]
    ret = []
    for i in range(0, len(haplist) - sequence_length, STEP):
        ret.append(np.concatenate(haplist[i:i + sequence_length], axis=0).astype(np.float32)[:, np.newaxis, ...])
    return ret, (bins, end_time)


def generate_npy_audio(path, n_frames_vision_image, behavior, sequence_length):
    """
    :param path: path to audio, you need to open it before you process
    :return: list of numpy array with size [SEQ_LENGTH, ...]
    """
    audio_path = glob.glob(path)
    if len(audio_path) == 0:
        return None
    audio_path = audio_path[0]
    img, duration = convert_audio_to_image(audio_path)
 # create a new dimension

    image_height, image_width = img.shape
    image_width = AUDIO_EACH_FRAME_LENGTH * n_frames_vision_image
    img = PIL.Image.fromarray(img)
    img = img.resize((image_height, image_width))

    img = np.array(img)
    img = img[np.newaxis, ...]
    imglist = []
    for i in range(0, n_frames_vision_image):
        imglist.append(img[:, i * AUDIO_EACH_FRAME_LENGTH:(i + 1) * AUDIO_EACH_FRAME_LENGTH, :])
    imglist = imglist[crop_stategy[behavior][0]:crop_stategy[behavior][1]]
    ret = []
    for i in range(0, len(imglist) - sequence_length, STEP):
        ret.append(np.concatenate(imglist[i:i + sequence_length], axis=0).astype(np.float32)[:, np.newaxis, ...])
    return ret


def generate_npy_vibro(path, n_frames, bins, behavior, sequence_length):
    """
    :param path: path to .tsv, you need to open it before you process
    :return: list of numpy array with size [SEQ_LENGTH, ...]
    """
    path = glob.glob(path)
    if not path and not bins:
        return None
    path = path[0]
    vibro_list = open(path).readlines()
    vibro_list = [list(map(int, vibro.strip().split('\t'))) for vibro in vibro_list]

    vibro_list = np.array(vibro_list)
    vibro_time = vibro_list[:, 0]
    vibro_data = vibro_list[:, 1:]
    bins, end_time = bins
    end_time -= bins[0]
    bins -= bins[0]

    v_h_ratio = vibro_time[-1] / end_time
    bins = bins * v_h_ratio

    groups = np.digitize(vibro_time, bins, right=False)

    vibro_data = [vibro_data[np.where(groups == idx)] for idx in range(1, n_frames + 1)]

    vibro_data = [np.vstack([np.resize(vibro[:, 0], (128,)),
                             np.resize(vibro[:, 1], (128,)),
                             np.resize(vibro[:, 2], (128,))]).T[np.newaxis, ...]
                  for vibro in vibro_data]
    # haplist = [np.pad(ht, [[0, 48 - ht.shape[0]], [0, 0]], mode='edge')[np.newaxis, ...] for ht in haplist]
    vibro_data = vibro_data[crop_stategy[behavior][0]:crop_stategy[behavior][1]]

    ret = []
    for i in range(0, len(vibro_data) - sequence_length, STEP):
        ret.append(np.concatenate(vibro_data[i:i + sequence_length], axis=0).astype(np.float32)[:, np.newaxis, ...])
    return ret


# splits words on objects with balanced categories to prepare for
# 5-fold cross validation
# assumes objects are in groupings/categories of exactly 5 with unique prefixes
def split():
    # test assumptions
    if len(SORTED_OBJECTS) != 100:
        raise Exception("split is intended to work for exactly 100 objects")

    # semi-randomly split data
    splits = [set([]),set([]),set([]),set([]),set([])]
    # for each of 20 categories of objects
    for category_i in range(len(SORTED_OBJECTS)//5):
        low_ind = 5*category_i
        random_list = np.random.permutation(5)

        # for each of the 5 objects in that category
        for object_i in range(5):
            ind = low_ind + random_list[object_i]
            if SORTED_OBJECTS[ind][0] != SORTED_OBJECTS[low_ind][0]:
                raise Exception("each grouping must have exactly 5 objects with identical prefix")
            else:
                splits[object_i].add(SORTED_OBJECTS[ind])

    return splits

# create vector encoding descriptors of an object
def switch_words_on(object, descriptor_codes, descriptors_by_object):
    encoded_output = np.zeros(len(descriptor_codes))
    if type(object) != type(None):
        for descriptor in descriptors_by_object[object]:
            word_index = descriptor_codes[descriptor]
            encoded_output[word_index] = 1
    return encoded_output


def process(visions, chosen_behaviors, OUT_DIR):

    for split_num in range(5):
        train_subdir = 'train'
        test_subdir = 'test'
        # vis_subdir = 'vis'
        if not os.path.exists(os.path.join(OUT_DIR, str(split_num), train_subdir)):
            os.makedirs(os.path.join(OUT_DIR, str(split_num), train_subdir))

        if not os.path.exists(os.path.join(OUT_DIR, str(split_num), test_subdir)):
            os.makedirs(os.path.join(OUT_DIR, str(split_num), test_subdir))

        splits = split()

        fail_count = 0
        for vision in visions:
            print("processing " + vision)

            # The path is object/trial/exec/behavior/file_name (visions does not include file names)
            vision_components = vision.split(os.sep)
            object_name = vision_components[-4]
            behavior_name = vision_components[-1]

            # validate behavior
            if behavior_name not in BEHAVIORS:
                continue      

            # validate object and split
            if object_name not in OBJECTS:
                continue
            if object_name in splits[split_num]:
                subdir = test_subdir
            else:
                subdir = train_subdir
            out_sample_dir = os.path.join(OUT_DIR, str(split_num), subdir, '_'.join(vision.split(os.sep)[-4:]))

            haptic1 = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'proprioception', 'ttrq0.txt')
            haptic2 = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'proprioception', 'cpos0.txt')
            audio = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'hearing', '*.wav')
            vibro = os.path.join(re.sub(r'vision_data_part[1-4]', 'rc_data', vision), 'vibro', '*.tsv')

            out_vision_npys, n_frames = generate_npy_vision(vision, behavior_name, SEQUENCE_LENGTH)
            out_audio_npys = generate_npy_audio(audio, n_frames, behavior_name, SEQUENCE_LENGTH)
            out_haptic_npys, bins = generate_npy_haptic(haptic1, haptic2, n_frames, behavior_name, SEQUENCE_LENGTH)
            out_vibro_npys = generate_npy_vibro(vibro, n_frames, bins, behavior_name, SEQUENCE_LENGTH)

            if out_audio_npys is None or out_haptic_npys is None or out_vibro_npys is None:
                fail_count += 1
                continue
            out_behavior_npys = compute_behavior(chosen_behaviors, behavior_name, object_name)

            for i, (out_vision_npy, out_haptic_npy, out_audio_npy, out_vibro_npy) in enumerate(zip(
                    out_vision_npys, out_haptic_npys, out_audio_npys, out_vibro_npys)):
                ret = {
                    'behavior': out_behavior_npys,
                    'vision': out_vision_npy,
                    'haptic': out_haptic_npy,
                    'audio': out_audio_npy,
                    'vibro': out_vibro_npy
                }
                np.save(out_sample_dir + '_' + str(i), ret)
        print("fail: ", fail_count)

def compute_behavior(CHOSEN_BEHAVIORS, behavior, object):
    out_behavior_npys = np.zeros(len(CHOSEN_BEHAVIORS))
    out_behavior_npys[CHOSEN_BEHAVIORS.index(behavior)] = 1
    descriptors = switch_words_on(object, DESCRIPTOR_CODES, DESCRIPTORS_BY_OBJECT)
    out_behavior_npys = np.hstack([out_behavior_npys, descriptors])
    return out_behavior_npys

def run(chosen_behavior, data_dir, out_dir):
    print("start making data")
    visons = read_dir(data_dir)
    process(visons, chosen_behavior, out_dir)
    print("done!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--behavior', nargs="+", action="append", default=[], help='which behavior?')
    parser.add_argument('--data_dir', default='/media/ramtin/ramtin/data/CY101Dataset', help='source data directory') # '../../data/CY101'
    parser.add_argument('--out_dir', default='/media/ramtin/ramtin/data/CY101NPY', help='target data directory') # '../../data/CY101NPY'
    args = parser.parse_args()

    # validate behavior argument
    if len(args.behavior) == 0:
        args.behavior = BEHAVIORS
    else:
        for behavior in args.behavior:
            if behavior not in BEHAVIORS:
                raise Exception("requested unknown behavior: " + behavior)
    print("behavior: ", args.behavior)

    # initiate data processing
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    run(chosen_behavior=args.behavior, data_dir=args.data_dir, out_dir=args.out_dir)