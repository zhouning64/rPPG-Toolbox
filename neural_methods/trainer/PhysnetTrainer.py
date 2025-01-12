"""PhysNet Trainer."""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
import cv2
from evaluation.metrics import calculate_metrics
from exp.record_video import load_mpg_into_tensor
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet import PhysNet_padding_Encoder_Decoder_MAX
from neural_methods.model.PhysNet import AppearanceBranch
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm
from evaluation.post_process import *


class PhysnetTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        # self.device = torch.device(config.DEVICE)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_epoch_num = config.TRAIN.EPOCHS
        # self.model_dir = config.MODEL.MODEL_DIR
        self.model_dir = "saved_model"
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        # EZ: Added
        self.appearance_branch = AppearanceBranch(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]
        # EZ

        self.model = PhysNet_padding_Encoder_Decoder_MAX(
            frames=config.MODEL.PHYSNET.FRAME_NUM).to(self.device)  # [3, T, 128,128]

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_model = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR)
            # See more details on the OneCycleLR scheduler here: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS,
                steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test" or config.TOOLBOX_MODE == "run":
            pass
        else:
            raise ValueError("PhysNet trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)

                # EZ: Added
                outputs = self.appearance_branch(batch[0].to(torch.float32).to(self.device))
                # EZ

                # EZ: Added "outputs" as an additional parameter
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    batch[0].to(torch.float32).to(self.device), outputs)
                # EZ

                BVP_label = batch[1].to(
                    torch.float32).to(self.device)
                # rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                # BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                #            torch.std(BVP_label)  # normalize
                rPPG = ((rPPG - torch.mean(rPPG)) / torch.std(rPPG)).clone()

                BVP_label = ((BVP_label - torch.mean(BVP_label)) / torch.std(BVP_label)).clone()

                loss = self.loss_model(rPPG, BVP_label)
                loss.backward()
                running_loss += loss.item()
                if idx % 100 == 99:  # print every 100 mini-batches
                    print(
                        f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())

                # Append the current learning rate to the list
                lrs.append(self.scheduler.get_last_lr())

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item())

            # Append the mean training loss for the epoch
            mean_training_losses.append(np.mean(train_loss))

            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """ Runs the model on valid sets."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        valid_step = 0
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(
                    torch.float32).to(self.device)

                # EZ: Added
                outputs = self.appearance_branch(valid_batch[0].to(torch.float32).to(self.device))
                # EZ

                # EZ: Added outputs as an additional parameter
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    valid_batch[0].to(torch.float32).to(self.device), outputs)
                # EZ

                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)  # normalize
                BVP_label = (BVP_label - torch.mean(BVP_label)) / \
                            torch.std(BVP_label)  # normalize
                loss_ecg = self.loss_model(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                valid_step += 1
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def load_mpg_into_tensor(self, filepath):
        """
        Reads a .mpg (MPEG) video file using OpenCV,
        converts each frame to a torch tensor,
        and returns a single tensor of shape (num_frames, channels, height, width).
        """
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {filepath}")

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                # No more frames or error reading
                break

            # frame is in BGR format by default (OpenCV)
            # Convert to RGB to maintain consistent color channel ordering
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert NumPy array (H, W, C) to a PyTorch tensor with shape (C, H, W)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)  # (3, H, W)

            frames.append(frame_tensor)
        cap.release()

    def getHearRate(self, image_file, model_path):
        image = load_mpg_into_tensor(image_file)

        print("opening model file: " , model_path)
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()

        with torch.no_grad():
            outputs = self.appearance_branch(image)
            prediction, _, _, _ = self.model(image, outputs)

            prediction = prediction.flatten()

            video_frame_size = prediction.shape[0]
            window_frame_size = video_frame_size
            pred_window = prediction[0 : window_frame_size]
            diff_flag_test = False
            hr = calculate_hr_from_video(
                pred_window, diff_flag=diff_flag_test, use_bandpass=False, fs=30, hr_method='FFT')
            print("inference heart rate: {}".format(hr))
            return hr

    def test(self, data_loader):
        """ Runs the model on test sets."""
        if data_loader["test"] is None:
            raise ValueError("No data for test")
        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE)

                outputs = self.appearance_branch(data)
                pred_ppg_test, _, _, _ = self.model(data, outputs)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]

        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR:  # saving test outputs
            self.save_test_outputs(predictions, labels, self.config)

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)
