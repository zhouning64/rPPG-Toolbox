import math
import pdb
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
import torch
import torchvision.transforms as transforms
from PIL import Image
import os


class PhysNet_padding_Encoder_Decoder_MAX(nn.Module):
    def __init__(self, frames=128):
        super(PhysNet_padding_Encoder_Decoder_MAX, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=False),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=False),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),  # [1, 128, 32]
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    # Frame = 3x72x72 Image
    # Given a frame, return a skin mask (0 = not skin, 1 = skin)
    def skin_mask_v1(self, frame):
        frame_np = frame.permute(1, 2, 0).cpu().numpy()  # Convert to Shape [72, 72, 3]
        # np.save("/content/saved_files/frame_np.npy", frame_np)
        temp = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        # frame1_np = frame_np
        # print("Frame1", np.min(frame1_np), np.max(frame1_np))
        cv2.imwrite("saved_files/temp.png", temp)  # Save in BGR format
        img = cv2.imread("saved_files/temp.png")
        # converting from gbr to hsv color space
        img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # skin color range for hsv color space
        HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
        HSV_mask = cv2.morphologyEx(HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # converting from gbr to YCbCr color space
        # img_YCrCb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2YCrCb)
        # skin color range for hsv color space
        # YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255,180,135))
        # YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        # merge skin detection (YCbCr and hsv)
        # global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask) #EZ: Get areas that were detected as skin in both HSV and YCbCr
        # global_result = global_result
        # print("Mask", np.max(HSV_mask), np.min(HSV_mask))
        cv2.imwrite("saved_files/mask.png", HSV_mask)  # Save in BGR format
        global_mask = HSV_mask
        global_mask = global_mask / 255
        # print(global_mask)
        # Convert the mask back to a PyTorch tensor
        global_mask = torch.tensor(global_mask, dtype=torch.float32).unsqueeze(0)
        return global_mask


    def skin_mask_v2(self, frame):
        frame_np = frame.permute(1, 2, 0).cpu().numpy()  # Convert to Shape [72, 72, 3]
        # np.save("/content/saved_files/frame_np.npy", frame_np)
        temp = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        # frame1_np = frame_np
        # print("Frame1", np.min(frame1_np), np.max(frame1_np))
        cv2.imwrite("saved_files/temp.png", temp)  # Save in BGR format
        img = cv2.imread("saved_files/temp.png")

        # Convert to HSV and YCbCr color spaces
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

        # Extract RGB and HSV channels
        R, G, B = img[:, :, 2], img[:, :, 1], img[:, :, 0]
        H, S, _ = cv2.split(img_hsv)
        Y, Cr, Cb = cv2.split(img_ycbcr)

        # Apply the skin detection algorithm with the specified thresholds
        # Algorithm: https://arxiv.org/pdf/1708.02694
        skin_mask_1 = ((H >= 0) & (H <= 50) &
                       (S >= 0.23 * 255) & (S <= 0.68 * 255) &
                       (R > 95) & (G > 40) & (B > 20) &
                       (R > G) & (R > B) &
                       (np.abs(R - G) > 15))

        skin_mask_2 = ((R > 95) & (G > 40) & (B > 20) &
                       (R > G) & (R > B) &
                       (np.abs(R - G) > 15) &
                       (Cr > 135) & (Cb > 85) & (Y > 80) &
                       (Cr <= (1.5862 * Cb + 20)) &
                       (Cr >= (0.3448 * Cb + 76.2069)) &
                       (Cr >= (-4.5652 * Cb + 234.5652)) &
                       (Cr <= (-1.15 * Cb + 301.75)) &
                       (Cr <= (-2.2857 * Cb + 432.85)))

        # Combine both conditions to create the final skin mask
        skin_mask = skin_mask_1 | skin_mask_2

        # Convert to a binary mask (0 and 1) and return as a PyTorch tensor
        skin_mask = skin_mask.astype(np.float32)
        skin_mask_tensor = torch.tensor(skin_mask)

        return skin_mask_tensor


    # Avg the RGB values
    def compressByAvg(self, array):
        return array.mean(dim=0)

    # For Model 4
    # Avg the spatial value (sum / (# of non-zero pixels))
    def spatialAvg(self, array):
        # return array.mean(dim=0)
        # Get all non-zero values from the tensor
        non_zero_values = array[array != 0]
        # print("Original Mean", array.mean())
        # print("Filtered Mean", non_zero_values.mean())
        # Calculate the mean of the non-zero values
        if non_zero_values.numel() == 0:  # Check if there are no non-zero values
            return 0.0
        else:
            mean_value = non_zero_values.mean()
            return mean_value

    # For Model 3
    def compressByShift(self, array):
        array = array.int()
        # Apply bitwise shift and combine operations across the entire tensor
        compressed_tensor = (array[0] << 16).float() + (array[1] << 8).float() + array[2].float()
        # Initialize a 72x72 array to store the result
        # compressed_array = [[float(0)] * 72 for _ in range(72)]

        # Iterate over each position in the 72x72 grid
        # for i in range(72):
        #    for j in range(72):
        #        # Apply the bitwise shift and OR operations
        #        compressed_array[i][j] = float((int(array[0][i][j]) << 16) | (int(array[1][i][j]) << 8) | int(array[2][i][j]))
        # compressed_tensor = torch.tensor(compressed_array, dtype=torch.float)
        # print("=======Shape After Compress======== ", compressed_tensor.shape)
        return compressed_tensor

    def save_tensor_as_png(self, tensor, filename):
        os.chdir("saved_files")

        # Ensure the tensor has 3 channels and dimensions 72x72
        assert tensor.shape == (3, 72, 72), "Tensor should have shape 3x72x72."

        # Scale the tensor to the range [0, 255] if it's not already in that range
        if tensor.max() <= 1:
            tensor = tensor * 255

        # Convert to uint8 type
        tensor = tensor.to(torch.uint8)

        # Use torchvision to convert the tensor to a PIL Image
        transform = transforms.ToPILImage()
        image = transform(tensor)

        # Save the image as a PNG file
        image.save(filename)
        # print(f"Image saved as {filename}")

    def forward(self, x, outputs):  # Batch_size*[3, T, 72,72]
        x = x.permute(0, 2, 1, 3, 4)
        x_visual = x
        [batch, channel, length, width, height] = x.shape
        # print("MAIN BRANCH - X.SHAPE", x.shape)
        # ================EZ: Skin Segmentation=================
        # Create a tensor to store the frame w/ skin segmented out, same dimensions as X
        device = x.device
        # Loop through frames, compute skin mask, multiply w/ frame, store in tensor

        segmented_frames = torch.zeros_like(x)
        for b in range(0, batch):
            for t in range(0, length):
                frame = x[b, :, t, :, :]
                frame = frame.to(device)
                mask = self.skin_mask_v2(frame)
                mask = mask.to(device)
                seg_frame = frame * mask
                segmented_frames[b, :, t, :, :] = seg_frame
        x = segmented_frames
        #for b in range(0, batch):
        #    for t in range(0, length):
        #        frame = x[b, :, t, :, :]  # Get 3x128x128 frame
        #        frame = frame.to(device)
        #        mask = self.skin_mask_v2(frame)  # Get mask of 0s and 1s
        #        mask = mask.to(device)
        #        seg_frame = frame * mask  # 3x72x72
        #        x[b, :, t, :, :] = seg_frame

                # Save the imgs after applying skin mask
                # file_name_og = "original_frames/image" + str(b) + "_" + str(t) + ".png"
                # self.save_tensor_as_png(frame, file_name_og) #Save the img after applying mask as a png image

                # file_name_seg = "seg_frames/image" + str(b) + "_" + str(t) + ".png"
                # self.save_tensor_as_png(seg_frame, file_name_seg) #Save the img after applying mask as a png image
        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = x * outputs[0]
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x = x * outputs[1]
        x_visual6464 = self.ConvBlock3(x)  # x [32, T, 64,64]
        x_visual6464 = x_visual6464 * outputs[2]
        # x [32, T/2, 32,32]    Temporal halve
        x = self.MaxpoolSpaTem(x_visual6464)

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x = x * outputs[3]

        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x_visual3232 = x_visual3232 * outputs[4]
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16,16]

        x = self.ConvBlock6(x)  # x [64, T/4, 16,16]
        x_visual1616 = self.ConvBlock7(x)  # x [64, T/4, 16,16]
        x = self.MaxpoolSpa(x_visual1616)  # x [64, T/4, 8,8]

        x = self.ConvBlock8(x)  # x [64, T/4, 8, 8]
        x = self.ConvBlock9(x)  # x [64, T/4, 8, 8]
        x = self.upsample(x)  # x [64, T/2, 8, 8]
        x = self.upsample2(x)  # x [64, T, 8, 8]

        # x [64, T, 1,1]    -->  groundtruth left and right - 7
        x = self.poolspa(x)
        x = self.ConvBlock10(x)  # x [1, T, 1,1]

        rPPG = x.view(-1, length)

        return rPPG, x_visual, x_visual3232, x_visual1616


# GENERATE ATTENTION MAPS and save them in "outputs"
class AppearanceBranch(nn.Module):
    def __init__(self, frames=128):
        super(AppearanceBranch, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),  # 16
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=False),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=False),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=False),
        )

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)

        # self.poolspa = nn.AdaptiveMaxPool3d((frames,1,1))    # pool only spatial space
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def forward(self, x):  # Batch_size*[3, T, 128,128]
        x = x.permute(0, 2, 1, 3, 4)
        x_visual = x
        [batch, channel, length, width, height] = x.shape
        outputs = []
        sigm = nn.Sigmoid()

        # print("APPEARANCE BRANCH - X SHAPE", x.shape)
        x = self.ConvBlock1(x)  # x [3, T, 128,128]
        x = sigm(x)
        outputs.append(x)
        x = self.MaxpoolSpa(x)  # x [16, T, 64,64]

        x = self.ConvBlock2(x)  # x [32, T, 64,64]
        x = sigm(x)
        outputs.append(x)

        x_visual6464 = self.ConvBlock3(x)  # x [32, T, 64,64]
        x = sigm(x_visual6464)
        outputs.append(x_visual6464)
        # x [32, T/2, 32,32]    Temporal halve
        x = self.MaxpoolSpaTem(x_visual6464)

        x = self.ConvBlock4(x)  # x [64, T/2, 32,32]
        x = sigm(x)
        outputs.append(x)

        x_visual3232 = self.ConvBlock5(x)  # x [64, T/2, 32,32]
        x = sigm(x_visual3232)
        outputs.append(x_visual3232)
        x = self.MaxpoolSpaTem(x_visual3232)  # x [64, T/4, 16,16]

        return outputs
