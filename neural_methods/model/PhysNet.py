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
        # Input [1, T]
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)


    # Frame = 3x72x72 Image
    # Given a frame, return a skin mask (0 = not skin, 1 = skin)
    def skin_mask_v1(self, frame):
        frame_np = frame.permute(1, 2, 0).cpu().numpy()  # Convert to Shape [72, 72, 3]
        # np.save("/content/saved_files/frame_np.npy", frame_np)
        temp = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        # frame1_np = frame_np
        # print("Frame1", np.min(frame1_np), np.max(frame1_np))
        cv2.imwrite("C:\\Users\\zhoun\\prj\\rPPG-Toolbox\\dataset\\saved_files\\temp.png", temp)  # Save in BGR format
        img = cv2.imread("C:\\Users\\zhoun\\prj\\rPPG-Toolbox\\dataset\\saved_files\\temp.png")
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
        cv2.imwrite("C:\\Users\\zhoun\\prj\\rPPG-Toolbox\\dataset\\saved_files\\mask.png", HSV_mask)  # Save in BGR format
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
        cv2.imwrite("C:\\Users\\zhoun\\prj\\rPPG-Toolbox\\dataset\\saved_files\\temp.png", temp)  # Save in BGR format
        img = cv2.imread("C:\\Users\\zhoun\\prj\\rPPG-Toolbox\\dataset\\saved_files\\temp.png")

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
        os.chdir("C:\\Users\\zhoun\\prj\\rPPG-Toolbox\\dataset\\saved_files\\")

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
        print(f"Image saved as {filename}")


    def forward(self, x):  # Batch_size*[3, T, 72,72]
        x_visual = x
        [batch, channel, length, width, height] = x.shape
        # ================EZ: Skin Segmentation=================
        # Create a tensor to store the frame w/ skin segmented out, same dimensions as X
        noisyRPPG = []  # Store the noisyRPPG waveform [batch, T]
        device = x.device
        # Loop through frames, compute skin mask, multiply w/ frame, store in tensor
        for b in range(0, batch):
            batchAverages = []
            for t in range(0, length):
                frame = x[b, :, t, :, :]  # Get 3x128x128 frame
                frame = frame.to(device)
                mask = self.skin_mask_v2(frame)  # Get mask of 0s and 1s
                mask = mask.to(device)
                seg_frame = frame * mask  # 3x72x72

                # file_name_og = "C:\\Users\\zhoun\\prj\\rPPG-Toolbox\\dataset\\saved_files\\original_frames\\image" + str(b) + "_" + str(t) + ".png"
                # self.save_tensor_as_png(frame, file_name_og)  # Save the img after applying mask as a png image

                # file_name_seg = "C:\\Users\\zhoun\\prj\\rPPG-Toolbox\\dataset\\saved_files\\seg_frames\\image" + str(b) + "_" + str(t) + ".png"
                # self.save_tensor_as_png(seg_frame, file_name_seg)  # Save the img after applying mask as a png image

                # grayscale_frame = compressByAvg(seg_frame) #Average the RGB values, so dimensions become 72x72
                grayscale_frame = self.compressByAvg(seg_frame)  # Average the RGB values, so dimensions become 72x72
                frame_average = self.spatialAvg(grayscale_frame)  # CHANGED
                batchAverages.append(frame_average)
            noisyRPPG.append(batchAverages)
        # Current dimensions: [batch][T]
        noisyRPPG = torch.tensor(noisyRPPG).to(device)  # Turn into a tensor
        noisyRPPG = noisyRPPG.unsqueeze(1)
        # Current dimension: [batch][1][T]
        # ===============EZ: Skin Segmentation END ===============
        # Expected Shape: [batch, channels, time]
        # x shape: [batch, 1, T]
        noisyRPPG = self.relu1(self.conv1(noisyRPPG))  # Output: batch[16][T]
        noisyRPPG = self.relu2(self.conv2(noisyRPPG))  # Output: batch[32][T]
        noisyRPPG = self.conv3(noisyRPPG)  # Output: batch[1][T]
        noisyRPPG = noisyRPPG.squeeze(1)  # Output: [batch][T]
        noisyRPPG = noisyRPPG.unsqueeze(1).unsqueeze(3).unsqueeze(4)  # Output: batch[1][t][1][1]
        rPPG = noisyRPPG.view(-1, length)

        #return rPPG, x_visual, x_visual3232, x_visual1616
        return rPPG, x_visual
        # Final Return Dimensions:
        # rPPG [1, T, 1, 1]
        # x_visual [3, T, 128, 128]
        # x_visual3232 [64, T/2, 32, 32]
        # x_visual1616 [64, T/4, 16, 16]