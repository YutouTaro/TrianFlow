def package_version_compare():
 # compare packages in requirements are installed and version are correct
 # the packages installed in the virt env have to be saved in piplist.txt in advance
    p_dict = dict()
    with open('piplist.txt', 'r') as fin:
        fin.readline()
        fin.readline()
        for line in fin.readlines():
            key, version = line.strip().split()
            p_dict[key] = version

    with open('requirements.txt', 'r') as fin2:
        for line in fin2.readlines():
            if '==' not in line:
                key = line.strip()
                if key not in p_dict.keys():
                    print('{} not found'.format(key))
            else:
                key, version = line.strip().split('==')
                if key not in p_dict.keys():
                    print('{} not found'.format(key))
                else:
                    if not version == p_dict[key]:
                        print('{} {} {}'.format(key, version, p_dict[key]))

    print('\ndone.')

def copy_data_from_gdrive():
### copy data files to google drive
### only image_02 of the img folders is copyed
    from os import listdir, mkdir
    from os.path import join, isdir, isfile
    from shutil import copy
    from tqdm import tqdm
    def copy_files_in_folder(dir_src, dir_dst, force_copy=False):
        numFile = [0, 0] # copied, skipped
        for item in listdir(dir_src):
            if isfile(join(dir_src, item)):
                # check exists in dst
                if force_copy or not isfile(join(dir_dst, item)):
                    copy(join(dir_src, item), join(dir_dst,item))
            else: # isdir
                if not isdir(join(dir_dst, item)):
                    mkdir(join(dir_dst, item))
                copy_files_in_folder(join(dir_src,item), join(dir_dst, item), force_copy)

    dir_data = r'D:\Datasets\kitti\raw_data'
    dir_gdrive = r'D:\GoogleDrive\datasets\kitti\raw_data'
    folders_nid = ['image_02', 'velodyne_points'] # folders needed for trianflow
    assert isdir(dir_data), 'dir_data {} not found'.format(dir_data)
    assert isdir(dir_gdrive), 'dir_gdrive {} not found'.format(dir_gdrive)

    dates = [date for date in listdir(dir_data) if isdir(join(dir_data,date))]
    dates.sort()
    for date in dates:
        print(date)
        dir_data_date = join(dir_data, date)
        dir_gdrive_date = join(dir_gdrive, date)
        if not isdir(dir_gdrive_date):
            mkdir(dir_gdrive_date)
        for item in tqdm(listdir(dir_data_date)):
            if isfile(join(dir_data_date, item)):
                copy(join(dir_data_date, item), join(dir_gdrive_date, item))
            else: # isdir
                drive = item
                print('\t{}'.format(drive))
                dir_data_date_drive = join(dir_data_date, drive)
                dir_gdrive_date_drive = join(dir_gdrive_date, drive)
                if not isdir(dir_gdrive_date_drive):
                    mkdir(dir_gdrive_date_drive)
                for folder in folders_nid:
                    print('\t\t{}'.format(folder))
                    if not isdir(join(dir_gdrive_date_drive, folder)):
                        mkdir(join(dir_gdrive_date_drive, folder))
                    dir_data_folder = join(dir_data_date_drive, folder)
                    dir_gdrive_folder = join(dir_gdrive_date_drive, folder)
                    copy_files_in_folder(dir_data_folder, dir_gdrive_folder)

def gt_NTU2kitti():
### NTU dataset ground truth to kitti format (flattened 3*4 matrix)
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    # r = R.from_quat([1,0,0,0])
    # print(r.as_matrix())

    gt_txt = r'C:\Users\kxhyu\Google Drive\datasets\Pioneer\NTU\DJI_0017\dataset_all.txt'
    kitti_txt = r'C:\Users\kxhyu\Google Drive\datasets\Pioneer\NTU\DJI_0017\dataset_kitti.txt'
    # # result_txt = r'C:\Users\kxhyu\Google Drive\datasets\Pioneer\NTU\traj_save\DJI_0017.txt'
    #
    # #load gt data
    # gt_pose = []
    f_kitti = open(kitti_txt,'w')
    with open(gt_txt, 'r') as f_gt:
        for line in f_gt.readlines():
            if 'DJI_0017' not in line:
                continue
            pose = [float(v) for v in line.strip().split(' ')[1:]]
            # print(pose)
            xyz = np.array(pose[:3]).reshape(-1,1)
            wpqr = np.append(pose[4:], pose[3])
            # print(xyz, wpqr)
            r = R.from_quat(wpqr)
            mat_r = r.as_matrix()
            # print(mat_r)
            mat_T = np.concatenate((mat_r, xyz), 1).flatten()
            # print(mat_T)
            t_str = ' '.join([str(v) for v in mat_T])
            # print(t_str)
            f_kitti.write(t_str + '\n')
            # break
    f_kitti.close()

def plot_from_kitti():
### plot traj from kitti format log
    from matplotlib import pyplot as plt
    import numpy as np
    # file = r'C:\Users\kxhyu\Google Drive\datasets\Pioneer\NTU\DJI_0017\00_kitti_pose.txt'
    # file = r'C:\Users\kxhyu\Google Drive\datasets\Pioneer\NTU\DJI_0017\dataset_kitti.txt'
    file = r'C:\Users\kxhyu\Google Drive\datasets\Pioneer\NTU\traj_save\DJI_0017.txt'

    fontsize_ = 20
    fig = plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal')
    pose =[]
    with open(file, 'r') as f:
        for line in f.readlines():
            p = [float(v) for v in line.strip().split()]
            pose.append([p[3], p[7], p[11]])
            # plt.plot(pose[3], pose[7], label='gt')
            # print(pose[3], pose[7])
    pose = np.array(pose)
    plt.plot(pose[:,0], pose[:,2], label='est')
    plt.legend(loc="upper right", prop={'size': fontsize_})
    plt.xticks(fontsize=fontsize_)
    plt.yticks(fontsize=fontsize_)
    plt.xlabel('x (m)', fontsize=fontsize_)
    plt.ylabel('y (m)', fontsize=fontsize_)
    fig.set_size_inches(10, 10)
    # plt.savefig(r"C:\Users\kxhyu\Google Drive\datasets\Pioneer\NTU\DJI_0017\d_xz_plot.pdf", bbox_inches='tight', pad_inches=0)
    plt.show()

# def generate_image_from_video():
#     ### generate image sequence from video (imcompleted) use the code in playAroundImg.py
#     import cv2
#
#     path_video = r'D:\GoogleDrive\datasets\Pioneer\videos\DJI_0017.MOV'
#     dir_img = r'D:\Datasets\Pioneer\dataset\sequences\DJI_0017\image_2'
#     vidcap = cv2.VideoCapture(path_video)
#     success, image = vidcap.read()
#     assert success, 'video not found!\n\t{}'.format(path_video)
#
#     starting_frame = 30
#
#     frame_count = 0 # counting the number of frames of the video
#     image_count = 0 # counting the number of images saved
#     while success:
#
#         if frame_count >= starting_frame:
#
#
#         frame_count += 1
#         success, image = vidcap.read()

def downsample_traj():
    # downsample the predicted traj
    # as the image freq is 50, while pose is 10
    downsample_file = r'C:\Users\kxhyu\Google Drive\datasets\Pioneer\NTU\traj_save\DJI_0017_2000_50fps.txt'
    traj_file = r'C:\Users\kxhyu\Google Drive\datasets\Pioneer\NTU\traj_save\DJI_0017.txt'
    with open(downsample_file, 'r') as fin:
        lines = [line.strip() for line in fin.readlines() if '=' not in line]
    print('found {} lines in \n\t{}.'.format(len(lines), downsample_file))

    idx_start = 0
    idx_end = len(lines)
    freq = 5
    lines = lines[idx_start:idx_end:freq]
    with open(traj_file, 'w') as fout:
        for line in lines:
            fout.write(line+'\n')
    print("wrote {} lines to \n\t{}".format(len(lines), traj_file))

if __name__ == '__main__':
    # generate_image_from_video()
    plot_from_kitti()
    # downsample_traj()