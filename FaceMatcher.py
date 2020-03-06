import numpy as np
import cv2
import scipy
import math
from matplotlib import pyplot as plt


def find_mean_face(num):
    path = "face_images/user0"
    meanface = np.zeros(shape=(30, 30))
    i = 1
    while i < 11:
        if i == 10:
            im_path = path[:-1]+str(i)+"_0"
        else:
            im_path = path+str(i)+"_0"
        j = 1
        while j < num+1:
            cur_path = im_path+str(j)+".bmp"
            im = cv2.imread(cur_path, 0)
            array = np.array(im)
            meanface = np.add(meanface, array)
            j += 1
        i += 1
    meanface = np.divide(meanface, 30)
    return meanface


def get_covariance_matrix(meanface, num):
    path = "face_images/user0"
    i = 1
    col = []
    while i < 11:
        if i == 10:
            im_path = path[:-1]+str(i)+"_0"
        else:
            im_path = path+str(i)+"_0"
        j = 1
        while j < num+1:
            cur_path = im_path+str(j)+".bmp"
            im = cv2.imread(cur_path, 0)
            array = np.array(im)
            x = np.subtract(array, meanface).flatten()
            col.append(x)
            j += 1
        i += 1
    x = np.asarray(col)
    return np.cov(x.T)


def find_eigen_cofs(meanface, eigenVecs):
    path = "face_images/user0"
    all_weights = []
    i = 1
    while i < 11:
        if i == 10:
            im_path = path[:-1]+str(i)+"_0"
        else:
            im_path = path+str(i)+"_0"
        j = 1
        while j < 6:
            cur_path = im_path+str(j)+".bmp"
            im = cv2.imread(cur_path, 0)
            array = np.array(im)
            mean_sub = np.subtract(array, meanface).flatten()
            weight = []
            t = 0
            while t < len(eigenVecs):
                weight.append(np.dot(mean_sub, eigenVecs[t]))
                t += 1
            j += 1
            all_weights.append(weight)
        i += 1
    return all_weights


def compute_distance(wa, wb):
    dsum = 0
    t = 0
    while t < len(wa):
        value = wa[t]-wb[t]
        dsum += pow(value, 2)
        t += 1
    return math.sqrt(dsum)


def compute_genuine(start, end, ws, ls):
    while start < end:
        s = start+1
        while s < end:
            ls.append(compute_distance(ws[start], ws[s]))
            s += 1
        start += 1


def compute_impostor(start, end, ws, ls):
    while start < end:
        s = end
        while s < len(ws):
            ls.append(compute_distance(ws[start], ws[s]))
            s += 1
        start += 1


mean = find_mean_face(3)
cv2.imwrite("mean_face.png", mean)
cov = get_covariance_matrix(mean, 3)
val, vec = scipy.linalg.eigh(cov, eigvals=(900-50, 899))


acs = np.argsort(np.abs(val))
dcs = acs[::-1]
new_val = val[dcs]
new_vec = vec[:, dcs].T

name = "Eigen_Face_"
i = 1
for item in new_vec:
    r = np.reshape(item, (30, 30))
    nme = name+str(i)+".png"
    #plt.imsave(nme, r, cmap=plt.cm.gray)
    #plt.imshow(r, cmap=plt.cm.gray)
    #plt.show()
    i += 1

weights = find_eigen_cofs(mean, new_vec)
i = 0
genuine = []
impostor = []
while i < len(weights):
    compute_genuine(i, i+5, weights, genuine)
    compute_impostor(i, i+5, weights, impostor)
    i += 5

cap = max(genuine) if max(genuine) > max(impostor) else max(impostor)
bins = np.linspace(0, cap, 100)
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.title("Genuine Vs. Impostor Using Top 50 Eigen Faces")
plt.hist(impostor, bins, alpha=0.5, label='Impostor', color="#fc0303")
plt.hist(genuine, bins, alpha=0.5, label='Genuine', color="#03fcfc")
plt.legend(loc='upper right')
plt.savefig("50EigHist.png")


############
# 10 Faces
############
val, vec = scipy.linalg.eigh(cov, eigvals=(900-10, 899))

acs = np.argsort(np.abs(val))
dcs = acs[::-1]
new_val = val[dcs]
new_vec = vec[:, dcs].T

weights = find_eigen_cofs(mean, new_vec)
i = 0
genuine = []
impostor = []
while i < len(weights):
    compute_genuine(i, i+5, weights, genuine)
    compute_impostor(i, i+5, weights, impostor)
    i += 5

plt.clf()
cap = max(genuine) if max(genuine) > max(impostor) else max(impostor)
bins = np.linspace(0, cap, 100)
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.title("Genuine Vs. Impostor Using Top 10 Eigen Faces")
plt.hist(impostor, bins, alpha=0.5, label='Impostor', color="#fc039d")
plt.hist(genuine, bins, alpha=0.5, label='Genuine', color="#0345fc")
plt.legend(loc='upper right')
plt.savefig("10EigHist.png")


############
# 20 Faces
############
val, vec = scipy.linalg.eigh(cov, eigvals=(900-20, 899))

acs = np.argsort(np.abs(val))
dcs = acs[::-1]
new_val = val[dcs]
new_vec = vec[:, dcs].T

weights = find_eigen_cofs(mean, new_vec)
i = 0
genuine = []
impostor = []
while i < len(weights):
    compute_genuine(i, i+5, weights, genuine)
    compute_impostor(i, i+5, weights, impostor)
    i += 5

plt.clf()
cap = max(genuine) if max(genuine) > max(impostor) else max(impostor)
bins = np.linspace(0, cap, 100)
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.title("Genuine Vs. Impostor Using Top 20 Eigen Faces")
plt.hist(impostor, bins, alpha=0.5, label='Impostor', color="#fc0303")
plt.hist(genuine, bins, alpha=0.5, label='Genuine', color="#03fc24")
plt.legend(loc='upper right')
plt.savefig("20EigHist.png")


############
# 40 Faces
############
val, vec = scipy.linalg.eigh(cov, eigvals=(900-40, 899))

acs = np.argsort(np.abs(val))
dcs = acs[::-1]
new_val = val[dcs]
new_vec = vec[:, dcs].T

weights = find_eigen_cofs(mean, new_vec)
i = 0
genuine = []
impostor = []
while i < len(weights):
    compute_genuine(i, i+5, weights, genuine)
    compute_impostor(i, i+5, weights, impostor)
    i += 5

plt.clf()
cap = max(genuine) if max(genuine) > max(impostor) else max(impostor)
bins = np.linspace(0, cap, 100)
plt.ylabel('Frequency')
plt.xlabel('Score')
plt.title("Genuine Vs. Impostor Using Top 40 Eigen Faces")
plt.hist(impostor, bins, alpha=0.5, label='Impostor', color="#fc9403")
plt.hist(genuine, bins, alpha=0.5, label='Genuine', color="#0b90b8")
plt.legend(loc='upper right')
plt.savefig("40EigHist.png")



