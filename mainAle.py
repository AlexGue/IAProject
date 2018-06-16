import os
import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import time
from threading import Thread


def main(INPUTPATH, K, T, H, it, n, OUTPUTPATH):
    INPUTPATH = "C:/Users/Alejandro/Desktop/Universidad/TERCERO/IA/prueba2"
    OUTPUTPATH = "C:/Users/Alejandro/Desktop/Universidad/TERCERO/IA/results"
    image_names = os.listdir(INPUTPATH)
    print('Image names:')
    print(image_names)
    print('-------------------')
    image_histr = calc_histr(INPUTPATH, T, image_names, H)
    print('Calculado histogramas de cada imagen')
    print('-------------------')
    k_means(K, image_histr, int(sum(image_histr['1.jpg']['b'])), H, it, n)



def calc_histr(inputpath,T,image_names,H):
    image_histr = []
    lastPercent = 0
    for i in range(0, len(image_names), T):
        if lastPercent != math.trunc(i/len(image_names)*100):
            lastPercent = math.trunc(i/len(image_names)*100)
            print('Calculando histogramas: ' + str(lastPercent) + '%')
        input_path = os.path.join(inputpath, image_names[i])

        img = cv2.imread(input_path)
        # cv2.imshow('scene', img)
        # width, heigth, chanels = img.shape
        color = ('b', 'g', 'r')

        histr = dict()
        for j, col in enumerate(color):
            histr[col] = list(map(lambda x: x[0], cv2.calcHist([img], [j], None, [H], [0, H])))
            #plt.plot(histr[col])
        #plt.title('IMAGE' + image_names[i])
        #plt.show()
        #     plt.plot(histr[col], color=col)
        #     plt.xlim([0, H])
        #
        # plt.show()
        # print(str(histr) + ' ' + str(i))
        image_histr.append((image_names[i], histr))

    return dict(image_histr)


def k_means(K, image_histr, size, H, iterations, n, centroids = None):

    color = ('b', 'g', 'r')
    for it in range(0, iterations):
        if centroids is None:
            centroids = dict()
            s = math.trunc(len(image_histr)/K)
            for i in range(0, K):
                centroids[i] = list(image_histr.values())[i*s]
                for col in color:
                    plt.plot(centroids[i][col])
                plt.show()

        classified_images = {k:[] for k, k_histr in centroids.items()}
        print('Iteration number: '+str(it))
        for name, img_histr in image_histr.items():
            min_histr = []
            min_k = -1
            min_value = float('inf')
            for k, k_histr in centroids.items():
                value_acumulated = 0
                for j, col in enumerate(color):
                    value_acumulated += sum(np.sqrt(np.power([x1 - x2 for (x1, x2) in zip(img_histr[col], k_histr[col])], 2)))

                if value_acumulated < min_value:
                    min_value = value_acumulated
                    min_k = k
                    min_histr = k_histr
            classified_images[min_k].append(name)
        print('Classified images: ' + str(classified_images))
        newCentroids = dict()
        for k, classes in classified_images.items():
            if classes:
                histr = None
                for name in classified_images[k]:
                    if histr is None:
                        histr = {key: [int(values)/len(classified_images[k]) for values in image_histr[name][key]] for key in image_histr[name]}
                    else:
                        histr = {key_h: [sum(x)/len(classified_images[k]) for x in zip(*[histr[key_h], image_histr[name][key_h]])] for key_h in histr}
                newCentroids[k] = histr
            else:
                newCentroids[k] = centroids[k]
        print('New centroids : '+ str(centroids))
    return centroids


if __name__ == '__main__':
    main(None, 2, 1, 256, 50, 3, None)
