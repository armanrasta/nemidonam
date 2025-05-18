from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMenu, QAction
from main_win.win import Ui_mainWindow
from PyQt5.QtCore import Qt, QPoint, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QIcon

import sys
import os
import json
import numpy as np
import torch

import torch.backends.cudnn as cudnn
import os
import time
import cv2

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadWebcam
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.CustomMessageBox import MessageBox
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
# from utils.plots import colors, plot_one_box, plot_one_box_PIL
from utils.plots import Annotator, colors, save_one_box

from utils.torch_utils import select_device
from utils.capnums import Camera
from dialog.rtsp_win import Window

from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

import time

class DetThread(QThread):
    send_img = pyqtSignal(np.ndarray)
    send_raw = pyqtSignal(np.ndarray)
    send_statistic = pyqtSignal(dict)
    # emit：detecting/pause/stop/finished/error msg
    send_msg = pyqtSignal(str)
    send_percent = pyqtSignal(int)
    send_fps = pyqtSignal(str)

    def __init__(self):
        super(DetThread, self).__init__()
        self.weights = './yolov5s.pt'
        self.current_weight = './yolov5s.pt'
        self.source = '0'
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.jump_out = False                   # jump out of the loop
        self.is_continue = True                 # continue/pause
        self.percent_length = 1000              # progress bar
        self.rate_check = True                  # Whether to enable delay
        self.rate = 100
        self.save_fold = './result/video'

    @torch.no_grad()
    def run(self,
            imgsz=1280,  # inference size (pixels)
            max_det=1000,  # maximum detections per image
            device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=True,  # show results
            save_txt=False,  # save results to *.txt
            save_conf=False,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=(0),  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project='runs/detect',  # save results to project/name
            name='exp',  # save results to project/name
            exist_ok=False,  # existing project/name ok, do not increment
            line_thickness=2,  # bounding box thickness (pixels)
            hide_labels=False,  # hide labels
            hide_conf=True,  # hide confidences
            half=False,  # use FP16 half-precision inference
            ):

        # Initialize
        try:
            device = select_device(device)
            half &= device.type != 'cpu'  # half precision only supported on CUDA

            # Load model
            model = attempt_load(self.weights, map_location=device)  # load FP32 model
            num_params = 0
            for param in model.parameters():
                num_params += param.numel()
            stride = int(model.stride.max())  # model stride
            imgsz = check_img_size(imgsz, s=stride)  # check image size
            names = model.module.names if hasattr(model, 'module') else model.names  # get class names
            if half:
                model.half()  # to FP16
                
            #load face detection and recognition medels
            probability = 0.7
            min_distance = 0.7
            min_face_size = 10
            mtcnn = MTCNN(image_size=160, keep_all=True, min_face_size = min_face_size, device=device)
            resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
            load_data = torch.load('data.pt')
            embedding_list = load_data[0]
            name_list = load_data[1]

            # Dataloader
            if self.source.isnumeric() or self.source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadWebcam(self.source, img_size=imgsz, stride=stride)
                # bs = len(dataset)  # batch_size
            elif self.source.endswith('.txt'):
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(self.source, img_size=imgsz, stride=stride)
                bs = len(dataset)  # batch_size
            else:
                dataset = LoadImages(self.source, img_size=imgsz, stride=stride)

            # Run inference
            if device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            count = 0
            jump_count = 0
            start_time = time.time()
            dataset = iter(dataset)

            while True:
                if self.jump_out:
                    self.vid_cap.release()
                    self.send_percent.emit(0)
                    self.send_msg.emit('Stop')
                    if hasattr(self, 'out'):
                        self.out.release()
                    break
                # change model
                if self.current_weight != self.weights:
                    # Load model
                    model = attempt_load(self.weights, map_location=device)  # load FP32 model
                    num_params = 0
                    for param in model.parameters():
                        num_params += param.numel()
                    stride = int(model.stride.max())  # model stride
                    imgsz = check_img_size(imgsz, s=stride)  # check image size
                    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
                    if half:
                        model.half()  # to FP16
                    # Run inference
                    if device.type != 'cpu':
                        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
                    self.current_weight = self.weights
                if self.is_continue:
                    path, img, im0s, self.vid_cap = next(dataset)
                    # jump_count += 1
                    # if jump_count % 5 != 0:
                    #     continue
                    count += 1
                    if count % 30 == 0 and count >= 30:
                        fps = int(30/(time.time()-start_time))
                        self.send_fps.emit('fps：'+str(fps))
                        start_time = time.time()
                    if self.vid_cap:
                        percent = int(count/self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)*self.percent_length)
                        self.send_percent.emit(percent)
                    else:
                        percent = self.percent_length

                    statistic_dic = {name: 0 for name in names}
                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()  # uint8 to fp16/32
                    img /= 255.0  # 0 - 255 to 0.0 - 1.0
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    pred = model(img, augment=augment)[0]

                    # Apply NMS
                    pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes, agnostic_nms, max_det=max_det)
                    # Process detections
                    for i, det in enumerate(pred):  # detections per image
                        results=[]
                        num_worker=0
                        num_vest=0
                        num_hardhat=0
                        num_no_hardhat=0
                        num_no_vest=0
                        num_glove=0
                        num_no_glove=0
                        num_glasses=0
                        num_no_glasses=0
                        num_cigarette=0
                        worker_center=[]
                        hardhat_center=[]
                        no_hardhat_center=[]
                        vest_center=[]
                        no_vest_center=[]
                        glove_center=[]
                        no_glove_center=[]
                        glasses_center=[]
                        no_glasses_center=[]
                        cigarette_center=[]
                        worker_xyxy=[]
                        glasses_xyxy=[]
                        no_glasses_xyxy=[]
                        workers_names = []
                        im0 = im0s.copy()
                        im1 = im0.copy()
                        annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                
                                c = int(cls)  # integer class
                                statistic_dic[names[c]] += 1
                                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                                if names[c]== 'worker':
                                    worker_xyxy.append(xyxy)
                                    worker_center.append(xyxy)
                                    num_worker +=1
                                                        
                                if names[c]== 'no hardhat':
                                    no_hardhat_center.append(xyxy)
                                    num_no_hardhat +=1
                                                        
                                if names[c]== 'no vest':
                                    no_vest_center.append(xyxy)
                                    num_no_vest +=1
                                                        
                                if names[c]== 'hardhat':
                                    hardhat_center.append(xyxy)
                                    num_hardhat += 1
                                                        
                                if names[c]== 'vest':
                                    vest_center.append(xyxy)
                                    num_vest += 1
                                                        
                                if names[c]== 'glove':
                                    glove_center.append(xyxy)
                                    num_glove += 1
                                                        
                                if names[c]== 'no glove':
                                    no_glove_center.append(xyxy)
                                    num_no_glove += 1
                                                        
                                if names[c]== 'glasses':
                                    glasses_center.append(xyxy)
                                    num_glasses += 1
                                    glasses_xyxy.append(xyxy)
                                                        
                                if names[c]== 'no glasses':
                                    no_glasses_center.append(xyxy)
                                    num_no_glasses += 1
                                    no_glasses_xyxy.append(xyxy)    
                                if names[c]== 'cigarette':
                                    label = ''
                                
                                
                                
                                if names[c] == 'person':
                                    
                                    #label = ''
                                    crop_glasses1 = save_one_box(xyxy, im1,BGR=True,save=False)
                                    original_frame = crop_glasses1.copy()
                                    if (crop_glasses1.shape[0]<min_face_size or crop_glasses1.shape[1]<min_face_size):
                                        
                                        crop_glasses1= cv2.resize(crop_glasses1, (min_face_size+1,min_face_size+1))
                                    img = Image.fromarray(crop_glasses1)
                                    img_cropped_list , prob_list = mtcnn(img, return_prob = True)
                                    if img_cropped_list is not None:
                                        boxes, _ = mtcnn.detect(img)
                                        for i, prob in enumerate(prob_list):
                                            if prob>probability:
                                                aligned1 = torch.squeeze(img_cropped_list[i]).to(device)
                                                emb = resnet(aligned1.unsqueeze(0)).detach().cpu()
                                                dist_list = []
                                                for idx, emb_db in enumerate(embedding_list):
                                                    dist = torch.dist(emb, emb_db).item()
                                                    dist_list.append(dist)
                                                min_dist = min(dist_list)
                                                min_dist_idx = dist_list.index(min_dist)
                                                name = name_list[min_dist_idx]
                                                box = boxes[i]
                                                #original_frame = cv2.resize(original_frame,(450,300))
                                                if min_dist<min_distance:
                                                    #original_frame = cv2.putText( original_frame, name, org, fontFace, fontScale, color)
                                                    label = name
                                                    workers_names.append(label)
                                                else:
                                                    workers_names.append('')
                                            else:
                                                workers_names.append('')
                                    else:
                                        workers_names.append('')
                
                                                
                                annotator.box_label(xyxy, label, color=colors(c, True))
                            if (num_worker>0):
                                print('num_worker',num_worker)
                                print(workers_names)
                                results=[]
                                for workers in range(num_worker):
                                    worker_name = workers_names[workers]
                                    worker_hardhat,worker_glasses,worker_vest,worker_glove, face_cordinate = worker_ppe_scan(worker_center,workers,
                                        hardhat_center,no_hardhat_center,num_hardhat,num_no_hardhat,
                                        glasses_center,no_glasses_center,num_glasses,num_no_glasses,glasses_xyxy,no_glasses_xyxy,
                                        vest_center,no_vest_center,num_vest,num_no_vest,
                                        glove_center,no_glove_center,num_glove,num_no_glove,cigarette_center,num_cigarette)
                                    save_results(worker_name,worker_center[workers],im1,worker_hardhat,worker_glasses,worker_vest,worker_glove)
                                    print(worker_name,worker_hardhat,worker_glasses,worker_vest,worker_glove)

                    if self.rate_check:
                        time.sleep(1/self.rate)
                    im0 = annotator.result()
                    self.send_img.emit(im0)
                    self.send_raw.emit(im0s if isinstance(im0s, np.ndarray) else im0s[0])
                    self.send_statistic.emit(statistic_dic)
                    if self.save_fold:
                        os.makedirs(self.save_fold, exist_ok=True)
                        if self.vid_cap is None:
                            save_path = os.path.join(self.save_fold,
                                                     time.strftime('%Y_%m_%d_%H_%M_%S',
                                                                   time.localtime()) + '.jpg')
                            cv2.imwrite(save_path, im0)
                        else:
                            if count == 1:
                                ori_fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))
                                if ori_fps == 0:
                                    ori_fps = 25
                                # width = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                # height = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                width, height = im0.shape[1], im0.shape[0]
                                save_path = os.path.join(self.save_fold, time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()) + '.mp4')
                                self.out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), ori_fps,
                                                           (width, height))
                            self.out.write(im0)
                    if percent == self.percent_length:
                        print(count)
                        self.send_percent.emit(0)
                        self.send_msg.emit('finished')
                        if hasattr(self, 'out'):
                            self.out.release()
                        break

        except Exception as e:
            self.send_msg.emit('%s' % e)
            
def worker_ppe_scan(worker_center,worker_number,
    hardhat_center,no_hardhat_center,num_hardhat,num_no_hardhat,
    glasses_center,no_glasses_center,num_glasses,num_no_glasses,glasses_xyxy,no_glasses_xyxy,
    vest_center,no_vest_center,num_vest,num_no_vest,
    glove_center,no_glove_center,num_glove,num_no_glove,
    cigarette_center,num_cigarette):
        
    workers=worker_number
    worker_without_hardhat=None
    worker_without_vest=None
    worker_without_glove=None
    worker_without_glasses=None
    worker_with_no_hardhat=None
    worker_with_no_vest=None
    worker_with_no_glasses=None
    worker_with_no_glove=None
    worker_with_cigarette=None
    face_cordinate=None
    #worker without hardhat
    for hardhats in range (num_hardhat):
        if not(hardhat_center[hardhats][2] > (worker_center[workers][0]) and \
               hardhat_center[hardhats][3] > ((worker_center[workers][1])) and \
                   hardhat_center[hardhats][2] < ((worker_center[workers][2])) and \
                       hardhat_center[hardhats][3] < ((worker_center[workers][3]))):
            worker_without_hardhat=True
        #crop_worker=save_one_box(worker_xyxy[workers], imc, file=save_dir / 'crops'  /names[c]/'hardhat'/ f'{p.stem}.jpg', BGR=True) 
    #worker without vest
    for vests in range (num_vest):
        if not(vest_center[vests][0] > (worker_center[workers][0]) and \
               vest_center[vests][1] > ((worker_center[workers][1])) and \
                   vest_center[vests][0] < ((worker_center[workers][2])) and \
                       vest_center[vests][1] < ((worker_center[workers][3]))):
            worker_without_vest=True
    
        #crop_worker=save_one_box(worker_xyxy[workers], imc, file=save_dir / 'crops'  /names[c]/'vest'/ f'{p.stem}.jpg', BGR=True) 
    #worker without glove
    for gloves in range (num_glove):
        if not(glove_center[gloves][0] > (worker_center[workers][0]) and \
               glove_center[gloves][1] > ((worker_center[workers][1])) and \
                   glove_center[gloves][0] < ((worker_center[workers][2])) and \
                       glove_center[gloves][1] < ((worker_center[workers][3]))):
            worker_without_glove=True
    
        #crop_worker=save_one_box(worker_xyxy[workers], imc, file=save_dir / 'crops'  /names[c]/'glove'/ f'{p.stem}.jpg', BGR=True) 
    #worker without glasses
    for glasses in range (num_glasses):
        if not(glasses_center[glasses][0] > (worker_center[workers][0]) and \
               glasses_center[glasses][1] > ((worker_center[workers][1])) and \
                   glasses_center[glasses][0] < ((worker_center[workers][2])) and \
                       glasses_center[glasses][1] < ((worker_center[workers][3]))):
            worker_without_glasses=True
        else:
            face_cordinate = glasses_center
            
    


#crop_worker=save_one_box(worker_xyxy[workers], imc, file=save_dir / 'crops'  /names[c]/'glasses'/ f'{p.stem}.jpg', BGR=True) 
#worker with NO-HardHat
    for no_hardhats in range (num_no_hardhat):
        if (no_hardhat_center[no_hardhats][2] > (worker_center[workers][0]) and \
            no_hardhat_center[no_hardhats][3] > (worker_center[workers][1]) and \
                no_hardhat_center[no_hardhats][2] < (worker_center[workers][2]) and \
                    no_hardhat_center[no_hardhats][3] < (worker_center[workers][3])):
            worker_with_no_hardhat=True
            
            
            #cv2.imshow('worker',crop_worker)
        #worker with no_vest
    for no_vests in range (num_no_vest):
        if (no_vest_center[no_vests][0] > (worker_center[workers][0]) and \
            no_vest_center[no_vests][1] > (worker_center[workers][1]) and \
                no_vest_center[no_vests][0] < (worker_center[workers][2]) and \
                    no_vest_center[no_vests][1] < (worker_center[workers][3])):
            worker_with_no_vest=True
            
        

    #worker with no_glasses
    for no_glasses in range (num_no_glasses):
        if (no_glasses_center[no_glasses][0] > (worker_center[workers][0]) and \
            no_glasses_center[no_glasses][1] > (worker_center[workers][1]) and \
                no_glasses_center[no_glasses][0] < (worker_center[workers][2]) and \
                    no_glasses_center[no_glasses][1] < (worker_center[workers][3])):
            worker_with_no_glasses=True
            face_cordinate = no_glasses_center
    

    #worker with no_glove
    for no_gloves in range (num_no_glove):
        if (no_glove_center[no_gloves][0] > (worker_center[workers][0]) and \
            no_glove_center[no_gloves][1] > (worker_center[workers][1]) and \
                no_glove_center[no_gloves][0] < (worker_center[workers][2]) and \
                    no_glove_center[no_gloves][1] < (worker_center[workers][3])):
            worker_with_no_glove=True
        

    #worker with cigarette
    for cigarette in range (num_cigarette):
        if (cigarette_center[cigarette][0] > (worker_center[workers][0]) and \
            cigarette_center[cigarette][1] > (worker_center[workers][1]) and \
                cigarette_center[cigarette][0] < (worker_center[workers][2]) and \
                    cigarette_center[cigarette][1] < (worker_center[workers][3])):
            worker_with_cigarette=True
        
    if (not(worker_without_hardhat) and worker_with_no_hardhat):
        worker_hardhat=False
    elif ((worker_without_hardhat== True) or (num_hardhat==0)):
        worker_hardhat=None
    else:
        worker_hardhat=True
    if (not(worker_without_vest) and worker_with_no_vest):
        worker_vest=False
    elif ((worker_without_vest == True) or (num_vest== 0)):
        worker_vest=None
    else:
        worker_vest=True
    if (not(worker_without_glasses) and worker_with_no_glasses):
        worker_glasses=False
    elif ((worker_without_glasses==True) or (num_glasses == 0)):
        worker_glasses=None
    else:
        worker_glasses=True
    if (not(worker_without_glove) and worker_with_no_glove):
        worker_glove=False
    elif ((worker_without_glove == True) or (num_glove == 0)):
        worker_glove=None        
    else:
        worker_glove=True
    if (worker_with_cigarette):
        worker_cigarette=True

    else:
        worker_cigarette=False
    return worker_hardhat,worker_glasses,worker_vest,worker_glove,face_cordinate  

def save_results(worker_name,worker_center,im1,worker_hardhat,worker_glasses,worker_vest,worker_glove):
    # #timestr = time.strftime("%Y%m%d-%H%M%S")
    if worker_hardhat == False:
        save_fold = './result' + '/' + worker_name + '/' + 'No_Hardhat'
        os.makedirs(save_fold, exist_ok=True)
        save_path = os.path.join(save_fold,
                                    time.strftime('%Y_%m_%d_%H_%M_%S',
                                                   time.localtime()) + '.jpg')
        crop = save_one_box(worker_center, im1 ,BGR=True,save=False) 
        cv2.imwrite(save_path, crop)
    elif worker_hardhat == True:
        save_fold = './result' + '/' + worker_name + '/' + 'Hardhat'
        os.makedirs(save_fold, exist_ok=True)
        save_path = os.path.join(save_fold,
                                    time.strftime('%Y_%m_%d_%H_%M_%S',
                                                   time.localtime()) + '.jpg')
        crop = save_one_box(worker_center, im1 ,BGR=True,save=False) 
        cv2.imwrite(save_path, crop)
    if worker_glasses == False:
        save_fold = './result' + '/' + worker_name + '/' + 'No_Glasses'
        os.makedirs(save_fold, exist_ok=True)
        save_path = os.path.join(save_fold,
                                    time.strftime('%Y_%m_%d_%H_%M_%S',
                                                 time.localtime()) + '.jpg')
        crop =  save_one_box(worker_center, im1,BGR=True,save=False)
        cv2.imwrite(save_path, crop)
    elif worker_glasses == True:
        save_fold = './result' + '/' + worker_name + '/' + 'Glasses'
        os.makedirs(save_fold, exist_ok=True)
        save_path = os.path.join(save_fold,
                                    time.strftime('%Y_%m_%d_%H_%M_%S',
                                                 time.localtime()) + '.jpg')
        crop =  save_one_box(worker_center, im1,BGR=True,save=False)
        cv2.imwrite(save_path, crop)

    if worker_vest == False:
        save_fold = './result' + '/' + worker_name + '/' + 'No_Vest'
        os.makedirs(save_fold, exist_ok=True)
        save_path = os.path.join(save_fold,
                                    time.strftime('%Y_%m_%d_%H_%M_%S',
                                                 time.localtime()) + '.jpg')
        crop =save_one_box(worker_center, im1,BGR=True,save=False)
        cv2.imwrite(save_path, crop)
    elif worker_vest == True:
        save_fold = './result' + '/' + worker_name + '/' + 'Vest'
        os.makedirs(save_fold, exist_ok=True)
        save_path = os.path.join(save_fold,
                                    time.strftime('%Y_%m_%d_%H_%M_%S',
                                                 time.localtime()) + '.jpg')
        crop =save_one_box(worker_center, im1,BGR=True,save=False)
        cv2.imwrite(save_path, crop)


    if worker_glove == False:
        save_fold = './result' + '/' + worker_name + '/' + 'No_Glove'
        os.makedirs(save_fold, exist_ok=True)
        save_path = os.path.join(save_fold,
                                    time.strftime('%Y_%m_%d_%H_%M_%S',
                                                 time.localtime()) + '.jpg')
        crop = save_one_box(worker_center, im1,BGR=True,save=False)
        cv2.imwrite(save_path, crop)
    elif worker_glove == True:
        save_fold = './result' + '/' + worker_name + '/' + 'Glove'
        os.makedirs(save_fold, exist_ok=True)
        save_path = os.path.join(save_fold,
                                    time.strftime('%Y_%m_%d_%H_%M_%S',
                                                 time.localtime()) + '.jpg')
        crop = save_one_box(worker_center, im1,BGR=True,save=False)
        cv2.imwrite(save_path, crop)
        
        
class MainWindow(QMainWindow, Ui_mainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.m_flag = False

        # style 1: window can be stretched
        # self.setWindowFlags(Qt.CustomizeWindowHint | Qt.WindowStaysOnTopHint)

        # style 2: window can not be stretched
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint
                            | Qt.WindowSystemMenuHint | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint)
        # self.setWindowOpacity(0.85)  # Transparency of window

        self.minButton.clicked.connect(self.showMinimized)
        self.maxButton.clicked.connect(self.max_or_restore)
        # show Maximized window
        self.maxButton.animateClick(10)
        self.closeButton.clicked.connect(self.close)

        self.qtimer = QTimer(self)
        self.qtimer.setSingleShot(True)
        self.qtimer.timeout.connect(lambda: self.statistic_label.clear())

        # search models automatically
        self.comboBox.clear()
        self.pt_list = os.listdir('./pt')
        self.pt_list = [file for file in self.pt_list if file.endswith('.pt')]
        self.pt_list.sort(key=lambda x: os.path.getsize('./pt/'+x))
        self.comboBox.clear()
        self.comboBox.addItems(self.pt_list)
        self.qtimer_search = QTimer(self)
        self.qtimer_search.timeout.connect(lambda: self.search_pt())
        self.qtimer_search.start(2000)

        # yolov5 thread
        self.det_thread = DetThread()
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.det_thread.source = '0'
        self.det_thread.percent_length = self.progressBar.maximum()
        self.det_thread.send_raw.connect(lambda x: self.show_image(x, self.raw_video))
        self.det_thread.send_img.connect(lambda x: self.show_image(x, self.out_video))
        self.det_thread.send_statistic.connect(self.show_statistic)
        self.det_thread.send_msg.connect(lambda x: self.show_msg(x))
        self.det_thread.send_percent.connect(lambda x: self.progressBar.setValue(x))
        self.det_thread.send_fps.connect(lambda x: self.fps_label.setText(x))

        self.fileButton.clicked.connect(self.open_file)
        self.cameraButton.clicked.connect(self.chose_cam)
        self.rtspButton.clicked.connect(self.chose_rtsp)

        self.runButton.clicked.connect(self.run_or_continue)
        self.stopButton.clicked.connect(self.stop)

        self.comboBox.currentTextChanged.connect(self.change_model)
        self.confSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'confSpinBox'))
        self.confSlider.valueChanged.connect(lambda x: self.change_val(x, 'confSlider'))
        self.iouSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'iouSpinBox'))
        self.iouSlider.valueChanged.connect(lambda x: self.change_val(x, 'iouSlider'))
        self.rateSpinBox.valueChanged.connect(lambda x: self.change_val(x, 'rateSpinBox'))
        self.rateSlider.valueChanged.connect(lambda x: self.change_val(x, 'rateSlider'))

        self.checkBox.clicked.connect(self.checkrate)
        self.saveCheckBox.clicked.connect(self.is_save)
        self.load_setting()

    def search_pt(self):
        pt_list = os.listdir('./pt')
        pt_list = [file for file in pt_list if file.endswith('.pt')]
        pt_list.sort(key=lambda x: os.path.getsize('./pt/' + x))

        if pt_list != self.pt_list:
            self.pt_list = pt_list
            self.comboBox.clear()
            self.comboBox.addItems(self.pt_list)

    def is_save(self):
        if self.saveCheckBox.isChecked():
            self.det_thread.save_fold = './result'
        else:
            self.det_thread.save_fold = None

    def checkrate(self):
        if self.checkBox.isChecked():
            self.det_thread.rate_check = True
        else:
            self.det_thread.rate_check = False

    def chose_rtsp(self):
        self.rtsp_window = Window()
        config_file = 'config/ip.json'
        if not os.path.exists(config_file):
            ip = "rtsp://admin:admin888@192.168.1.67:555"
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            ip = config['ip']
        self.rtsp_window.rtspEdit.setText(ip)
        self.rtsp_window.show()
        self.rtsp_window.rtspButton.clicked.connect(lambda: self.load_rtsp(self.rtsp_window.rtspEdit.text()))

    def load_rtsp(self, ip):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='Tips', text='Loading rtsp stream', time=1000, auto=True).exec_()
            self.det_thread.source = ip
            new_config = {"ip": ip}
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open('config/ip.json', 'w', encoding='utf-8') as f:
                f.write(new_json)
            self.statistic_msg('Loading rtsp：{}'.format(ip))
            self.rtsp_window.close()
        except Exception as e:
            self.statistic_msg('%s' % e)

    def chose_cam(self):
        try:
            self.stop()
            MessageBox(
                self.closeButton, title='Tips', text='Loading camera', time=2000, auto=True).exec_()
            # get the number of local cameras
            _, cams = Camera().get_cam_num()
            popMenu = QMenu()
            popMenu.setFixedWidth(self.cameraButton.width())
            popMenu.setStyleSheet('''
                                            QMenu {
                                            font-size: 16px;
                                            font-family: "Microsoft YaHei UI";
                                            font-weight: light;
                                            color:white;
                                            padding-left: 5px;
                                            padding-right: 5px;
                                            padding-top: 4px;
                                            padding-bottom: 4px;
                                            border-style: solid;
                                            border-width: 0px;
                                            border-color: rgba(255, 255, 255, 255);
                                            border-radius: 3px;
                                            background-color: rgba(200, 200, 200,50);}
                                            ''')

            for cam in cams:
                exec("action_%s = QAction('%s')" % (cam, cam))
                exec("popMenu.addAction(action_%s)" % cam)

            x = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).x()
            y = self.groupBox_5.mapToGlobal(self.cameraButton.pos()).y()
            y = y + self.cameraButton.frameGeometry().height()
            pos = QPoint(x, y)
            action = popMenu.exec_(pos)
            if action:
                self.det_thread.source = action.text()
                self.statistic_msg('Loading camera：{}'.format(action.text()))
        except Exception as e:
            self.statistic_msg('%s' % e)

    def load_setting(self):
        config_file = 'config/setting.json'
        if not os.path.exists(config_file):
            iou = 0.26
            conf = 0.33
            rate = 10
            check = 0
            savecheck = 0
            new_config = {"iou": iou,
                          "conf": conf,
                          "rate": rate,
                          "check": check,
                          "savecheck": savecheck
                          }
            new_json = json.dumps(new_config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(new_json)
        else:
            config = json.load(open(config_file, 'r', encoding='utf-8'))
            if len(config) != 5:
                iou = 0.26
                conf = 0.33
                rate = 10
                check = 0
                savecheck = 0
            else:
                iou = config['iou']
                conf = config['conf']
                rate = config['rate']
                check = config['check']
                savecheck = config['savecheck']
        self.confSpinBox.setValue(conf)
        self.iouSpinBox.setValue(iou)
        self.rateSpinBox.setValue(rate)
        self.checkBox.setCheckState(check)
        self.det_thread.rate_check = check
        self.saveCheckBox.setCheckState(savecheck)
        self.is_save()

    def change_val(self, x, flag):
        if flag == 'confSpinBox':
            self.confSlider.setValue(int(x*100))
        elif flag == 'confSlider':
            self.confSpinBox.setValue(x/100)
            self.det_thread.conf_thres = x/100
        elif flag == 'iouSpinBox':
            self.iouSlider.setValue(int(x*100))
        elif flag == 'iouSlider':
            self.iouSpinBox.setValue(x/100)
            self.det_thread.iou_thres = x/100
        elif flag == 'rateSpinBox':
            self.rateSlider.setValue(x)
        elif flag == 'rateSlider':
            self.rateSpinBox.setValue(x)
            self.det_thread.rate = x * 10
        else:
            pass

    def statistic_msg(self, msg):
        self.statistic_label.setText(msg)
        # self.qtimer.start(3000)

    def show_msg(self, msg):
        self.runButton.setChecked(Qt.Unchecked)
        self.statistic_msg(msg)
        if msg == "Finished":
            self.saveCheckBox.setEnabled(True)

    def change_model(self, x):
        self.model_type = self.comboBox.currentText()
        self.det_thread.weights = "./pt/%s" % self.model_type
        self.statistic_msg('Change model to %s' % x)

    def open_file(self):

        config_file = 'config/fold.json'
        # config = json.load(open(config_file, 'r', encoding='utf-8'))
        config = json.load(open(config_file, 'r', encoding='utf-8'))
        open_fold = config['open_fold']
        if not os.path.exists(open_fold):
            open_fold = os.getcwd()
        name, _ = QFileDialog.getOpenFileName(self, 'Video/image', open_fold, "Pic File(*.mp4 *.mkv *.avi *.flv *.txt "
                                                                          "*.jpg *.png)")
        if name:
            self.det_thread.source = name
            self.statistic_msg('Loaded file：{}'.format(os.path.basename(name)))
            config['open_fold'] = os.path.dirname(name)
            config_json = json.dumps(config, ensure_ascii=False, indent=2)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_json)
            self.stop()

    def max_or_restore(self):
        if self.maxButton.isChecked():
            self.showMaximized()
        else:
            self.showNormal()

    def run_or_continue(self):
        self.det_thread.jump_out = False
        if self.runButton.isChecked():
            self.saveCheckBox.setEnabled(False)
            self.det_thread.is_continue = True
            if not self.det_thread.isRunning():
                self.det_thread.start()
            source = os.path.basename(self.det_thread.source)
            source = 'camera' if source.isnumeric() else source
            self.statistic_msg('Detecting >> model：{}，file：{}'.
                               format(os.path.basename(self.det_thread.weights),
                                      source))
        else:
            self.det_thread.is_continue = False
            self.statistic_msg('Pause')

    def stop(self):
        self.det_thread.jump_out = True
        self.saveCheckBox.setEnabled(True)

    def mousePressEvent(self, event):
        self.m_Position = event.pos()
        if event.button() == Qt.LeftButton:
            if 0 < self.m_Position.x() < self.groupBox.pos().x() + self.groupBox.width() and \
                    0 < self.m_Position.y() < self.groupBox.pos().y() + self.groupBox.height():
                self.m_flag = True

    def mouseMoveEvent(self, QMouseEvent):
        if Qt.LeftButton and self.m_flag:
            self.move(QMouseEvent.globalPos() - self.m_Position)

    def mouseReleaseEvent(self, QMouseEvent):
        self.m_flag = False

    @staticmethod
    def show_image(img_src, label):
        try:
            ih, iw, _ = img_src.shape
            w = label.geometry().width()
            h = label.geometry().height()
            # keep original aspect ratio
            if iw/w > ih/h:
                scal = w / iw
                nw = w
                nh = int(scal * ih)
                img_src_ = cv2.resize(img_src, (nw, nh))

            else:
                scal = h / ih
                nw = int(scal * iw)
                nh = h
                img_src_ = cv2.resize(img_src, (nw, nh))

            frame = cv2.cvtColor(img_src_, cv2.COLOR_BGR2RGB)
            img = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[2] * frame.shape[1],
                         QImage.Format_RGB888)
            label.setPixmap(QPixmap.fromImage(img))

        except Exception as e:
            print(repr(e))

    def show_statistic(self, statistic_dic):
        try:
            self.resultWidget.clear()
            statistic_dic = sorted(statistic_dic.items(), key=lambda x: x[1], reverse=True)
            statistic_dic = [i for i in statistic_dic if i[1] > 0]
            results = [' '+str(i[0]) + '：' + str(i[1]) for i in statistic_dic]
            self.resultWidget.addItems(results)

        except Exception as e:
            print(repr(e))

    def closeEvent(self, event):
        self.det_thread.jump_out = True
        config_file = 'config/setting.json'
        config = dict()
        config['iou'] = self.confSpinBox.value()
        config['conf'] = self.iouSpinBox.value()
        config['rate'] = self.rateSpinBox.value()
        config['check'] = self.checkBox.checkState()
        config['savecheck'] = self.saveCheckBox.checkState()
        config_json = json.dumps(config, ensure_ascii=False, indent=2)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_json)
        MessageBox(
            self.closeButton, title='Tips', text='Closing the program', time=2000, auto=True).exec_()
        sys.exit(0)
        



if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MainWindow()
    myWin.show()
    # myWin.showMaximized()
    sys.exit(app.exec_())
