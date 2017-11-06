import cv2
import numpy as np
import sys

def add_salt_pepper_noise(srcArr, pa, pb):
    rows = srcArr.shape[0]
    cols = srcArr.shape[1]
    amount1 = int(rows * cols * pa)
    amount2 = int(rows * cols * pb)

    for i in range(0, amount1):
        srcArr[np.random.randint(0, rows), np.random.randint(0, cols)] = 0

    for  i in range(0, amount2):
        srcArr[np.random.randint(0, rows), np.random.randint(0, cols)] = 255

def add_gaussian_noise(srcArr, mean, sigma):
    noiseArr = np.copy(srcArr)

    cv2.randn(noiseArr, mean, sigma)
        
    return np.add(srcArr, noiseArr)

if __name__ == '__main__':
    image = cv2.imread(sys.argv[1], 0)
    cv2.imshow('Original image', image)

    means = [0, 5, 10, 20]
    sigmas = [0, 20, 50, 100]

    pas = [0.01, 0.03, 0.05, 0.4]
    pbs = [0.01, 0.03, 0.05, 0.4]

    ksizes = [3, 5, 7]

    for x in range(0, 3):
        for i in range(0, 4):
            for j in range(0, 4):

                ksize = ksizes[x]
                
                noise_img = np.copy(image)
                mean = means[i]
                sigma = sigmas[j]
                noise_img = add_gaussian_noise(noise_img, mean, sigma)
                cv2.imshow('Gaussian Noise', noise_img)
                curr = 'gaussiannoise_mean_' + str(mean) + '_sigma_' + str(sigma) + '.png'
                cv2.imwrite(curr, noise_img)

                noise_dst = np.copy(noise_img)
                noise_dst = cv2.blur(noise_dst, (ksize, ksize))
                cv2.imshow('Box filter on Gaussian Noise', noise_dst)
                curr = 'boxfilter_gaussiannoise_mean_' + str(mean) + '_sigma_' + str(sigma) + '_ksize_' + str(ksize) + '.png'
                cv2.imwrite(curr, noise_dst)

                noise_dst1 = np.copy(noise_img)
                noise_dst1 = cv2.GaussianBlur(noise_dst1, (ksize, ksize), 1.5)
                cv2.imshow('Gaussian filter on Gaussian Noise', noise_dst1)
                curr = 'gaussianfilter_gaussiannoise_mean_' + str(mean) + '_sigma_' + str(sigma) + '_ksize_' + str(ksize) + '.png'
                cv2.imwrite(curr, noise_dst1)

                noise_dst2 = np.copy(noise_img)
                noise_dst2 = cv2.medianBlur(noise_dst2, ksize)
                cv2.imshow('Median filter on Gaussian Noise', noise_dst2)
                curr = 'medianfilter_gaussiannoise_mean_' + str(mean) + '_sigma_' + str(sigma) + '_ksize_' + str(ksize) + '.png'
                cv2.imwrite(curr, noise_dst2)

                noise_img2 = np.copy(image)
                pa = pas[i]
                pb = pas[j]
                t = str(pa)            
                pa_str = t[0:t.index('.')] + 'point' + t[t.index('.') + 1:len(t)]
                t = str(pb)
                pb_str = t[0:t.index('.')] + 'point' + t[t.index('.') + 1:len(t)]
                add_salt_pepper_noise(noise_img2, pa, pb)
                cv2.imshow('Salt and Pepper Noise', noise_img2)
                curr = 'saltpepper_pa_' + str(pa_str) + '_pb_' + str(pb_str) + '.png'
                cv2.imwrite(curr, noise_img2)

                noise_dst3 = np.copy(noise_img2)
                noise_dst = cv2.blur(noise_dst3, (ksize, ksize))
                cv2.imshow('Box filter on Salt and Pepper Noise', noise_dst3)
                curr = 'boxfilter_saltpepper_pa_' + str(pa_str) + '_pb_' + str(pb_str) + '_ksize_' + str(ksize) + '.png'
                cv2.imwrite(curr, noise_dst3)

                noise_dst4 = np.copy(noise_img2)
                noise_dst4 = cv2.GaussianBlur(noise_dst4, (ksize, ksize), 1.5)
                cv2.imshow('Gaussian filter on Salt and Pepper Noise', noise_dst4)
                curr = 'gaussianfilter_saltpepper_pa_' + str(pa_str) + '_pb_' + str(pb_str) + '_ksize_' + str(ksize) + '.png'
                cv2.imwrite(curr, noise_dst4)

                noise_dst5 = np.copy(noise_img2)
                noise_dst5 = cv2.medianBlur(noise_dst5, ksize)
                cv2.imshow('Median filter on Salt and Pepper Noise', noise_dst5)
                curr = 'medianfilter_saltpepper_pa_' + str(pa_str) + '_pb_' + str(pb_str) + '_ksize_' + str(ksize) + '.png'
                cv2.imwrite(curr, noise_dst5)

                cv2.waitKey(0)
    
    
