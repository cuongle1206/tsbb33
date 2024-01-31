import numpy as np
import matplotlib.pyplot as plt
import random
import time 
import cv2
import lab1

from pathlib import Path
from scipy.optimize import least_squares
from scipy.linalg import inv
from PIL import Image
 
import torch
from kornia.geometry import axis_angle_to_rotation_matrix
from kornia.geometry import rotation_matrix_to_axis_angle
from micro_bundle_adjustment.api import projection, optimize_calibrated

def projection(X, r, t):
    if len(X.shape) == 1:
        X = X[None]
    N, D = X.shape
    if len(r.shape) == 1:
        r = r.expand(N,D)
        t = t.expand(N,D)
    R = axis_angle_to_rotation_matrix(r)
    x = (R @ X[...,None]) + t[...,None]
    x = x[...,0]
    return x[...,:2]/x[...,[2]]

def gold_standard_residuals(X, r, t, x_a, x_b):
    r_a = x_a - projection(X, torch.zeros_like(r), torch.zeros_like(t))
    r_b = x_b - projection(X, r, t)
    return torch.cat((r_a, r_b), dim=1)


if __name__ == "__main__":
    
    LAB1_IMAGE_DIRECTORY = Path('tsbb33-datasets/bacchus')
    K           = np.array([[1660.076384971925*1e-3, 0, 763.9663384634628*1e-3],
                            [0, 1656.285062933074*1e-3, 986.3176281647676*1e-3],
                            [0, 0, 1]]) 
    dist_coeffs = np.array([0.260831, -1.67584, -0.00265474, 0.00115126, 3.48227])
    mask        = np.asarray(Image.open('tsbb33-datasets/bacchus/bacchus_mask.png').convert('L'))
    
    a = []
    for i in range(1,2):
        img1        = lab1.load_image_grayscale(LAB1_IMAGE_DIRECTORY / f'frame{(i+1):02d}.png')
        img2        = lab1.load_image_grayscale(LAB1_IMAGE_DIRECTORY / f'frame{(i+2):02d}.png')
        # w, h        = img1.shape
        masked_img1 = cv2.bitwise_and(img1, img1, mask=mask)
        masked_img2 = cv2.bitwise_and(img2, img2, mask=mask)
        
        print("... SIFT Feature Detection ... ")
        sift        = cv2.SIFT_create()
        kp1         = sift.detect(masked_img1, None)
        kp2         = sift.detect(masked_img2, None)
        p1          = cv2.KeyPoint_convert(kp1).astype(int)
        p2          = cv2.KeyPoint_convert(kp2).astype(int)
        
        roi1        = np.array(lab1.cut_out_rois(img1, p1[:,0], p1[:,1], 35))
        roi2        = np.array(lab1.cut_out_rois(img2, p2[:,0], p2[:,1], 35))
        vec1        = roi1.reshape(roi1.shape[0], -1).T
        vec2        = roi2.reshape(roi2.shape[0], -1).T
        vec1        = (vec1 - np.mean(vec1, axis=0, keepdims=1)) / np.std(vec1, axis=0)
        vec2        = (vec2 - np.mean(vec2, axis=0, keepdims=1)) / np.std(vec2, axis=0)

        vec1_ex     = np.expand_dims(vec1, axis=-1)
        vec2_ex     = np.expand_dims(vec2, axis=1)
        corr        = np.sum((vec1_ex-vec2_ex)**2, axis=0)
        _, row, col = lab1.joint_min(corr)
        pl          = np.array(p1[row, :]).T
        pr          = np.array(p2[col, :]).T
        # fig         = lab1.show_corresp(img1, img2, pl, pr)
        
        print("... RANSAC ... ")
        n           = pl.shape[1]
        nPtCor      = 8
        max_iter    = 50000
        d           = np.zeros(n)
        In_max      = 0
        for k in range(max_iter):
            rnb_idx     = random.sample(range(n), nPtCor)
            F           = K.T @ lab1.fmatrix_stls(pl[:,rnb_idx], pr[:,rnb_idx]) @ K
            
            eps         = 2
            d           = np.mean(lab1.fmatrix_residuals(F, pl, pr)**2, axis=0) 
            In_ID       = np.where(d < eps)

            if In_ID[0].shape[0] > In_max:
                print("Num. of corrs: " + str(In_ID[0].shape[0]))
                In_max      = In_ID[0].shape[0]
                Best_F      = F
                pl_inline   = pl[:,In_ID[0]];
                pr_inline   = pr[:,In_ID[0]];

        inliers_a   = pl_inline
        inliers_b   = pr_inline

        fig2        = lab1.show_corresp(img1, img2, inliers_a, inliers_b)
        
        # Gold Standard
        print("Optimizing ... ")
        F_hat       = Best_F
        Cl, Cr      = lab1.fmatrix_cameras(F_hat)

        X0          = np.zeros((3, pr_inline.shape[1])).T
        for i in range(pr_inline.shape[1]):
            X0[i,:]     = lab1.triangulate_optimal(Cl, Cr, pl_inline[:,i], pr_inline[:,i])

        X_hat, r_hat, t_hat = optimize_calibrated(gold_standard_residuals, torch.Tensor(X0), 
                                            rotation_matrix_to_axis_angle(torch.Tensor(Cl[:3,:3])), torch.Tensor(Cl[:,-1]), 
                                            torch.Tensor(inliers_a.T), torch.Tensor(inliers_b.T))
        
        a.append(X_hat)
        
    a = torch.cat(a, dim=0)
    print(a.shape)
        
    fig3 = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(a[:,1], -a[:,2], -a[:,0])

    # F_new       = lab1.fmatrix_from_cameras(Cl_new, Cr)

    # fig3 = plt.figure()
    # plt.subplot(1,2,1)
    # lab1.imshow(img1, cmap='gray')
    # lab1.plot_eplines(F_new, inliers_b, img1.shape)
    # plt.plot(inliers_a[0,:], inliers_a[1,:], '*r')

    # plt.subplot(1,2,2)
    # lab1.imshow(img2, cmap='gray')
    # lab1.plot_eplines(F_new.T, inliers_a, img2.shape)
    # plt.plot(inliers_b[0,:], inliers_b[1,:], '*r')
    
    print('End.')
    
    plt.show()
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 