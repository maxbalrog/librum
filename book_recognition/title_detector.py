import pytesseract

from .book_segmentation import *
from .comnist_train import *

# class Model(nn.Module):
#     def __init__(self, n_classes=34, n_filters=15):
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d(1, n_filters, 5, padding=0)
#         self.conv2 = nn.Conv2d(n_filters, n_filters*2, 5, padding=0)
#         self.conv3 = nn.Conv2d(n_filters*2, n_filters*4, 5, padding=0)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.fc1 = nn.Linear(n_filters*4 * 4 * 4, 160)
#         self.fc2 = nn.Linear(160, 100)
#         self.fc3 = nn.Linear(100, n_classes)
#
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout()
#
#     def forward(self, x):
#         bs, _, _, _ = x.shape
#         x = self.pool(self.relu(self.conv1(x)))
#         x = self.pool(self.relu(self.conv2(x)))
#         x = self.pool(self.relu(self.conv3(x)))
#         x = x.view(bs, -1)
#         x = self.dropout(x)
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)
#
#         return x

def find_edges(img, sigma=1, thr=20):
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = feature.canny(imgray, sigma=1, low_threshold=20)

    return edges

def sort_contours(img, contours, area_thr=50, width_thr=10, height_thr=40):
    h,w = img.shape[:2]
    contours_let = []
    for cnt in contours:
        M = cv2.moments(cnt)

        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])

            x, y, w1, h1 = cv2.boundingRect(cnt)
            bound_rect_area = w1 * h1
    #         cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            if bound_rect_area > w*h//area_thr:
                continue

            w_l = w//width_thr
            w_r = w - w_l
            h_l = h//height_thr
            h_r = h - h_l
            if ((cx < w_l) | (cx > w_r)) | ((cy < h_l) | (cy > h_r)):
                continue
            else:
                contours_let.append(cnt)

    return contours_let

def find_max(flat, relax=5):
    i = 0
    j = len(flat) - 1
    left_max, right_max = flat[i], flat[j]
    i += 1
    j -= 1

    # fl1, fl2 = True, True
    idx_left, idx_right = 0, len(flat)-1

    while i<=len(flat)//2:
        if flat[i] > flat[i-1]:
            i += 1
            continue
        elif flat[i-1] > left_max:
            idx_left = i-1
            left_max = flat[i-1]
        i += 1

    while j>=len(flat)//2:
        if flat[j] > flat[j+1]:
            j -= 1
            continue
        elif flat[j+1] > right_max:
            idx_right = j+1
            right_max = flat[j+1]
        j -= 1

    idx_left -= relax
    idx_right += relax + 1

    return idx_left, idx_right

def find_title_region(image, kernlen=5, std=1):
    img = image.copy().astype(np.uint8)
    h,w = img.shape[:2]

    edges = find_edges(img, sigma=1, thr=20)

    contours, hierarchy = cv2.findContours(edges.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    contours_let = sort_contours(img, contours)

    flat = np.zeros(w)
    for cnt in contours_let:
        x, y, w1, h1 = cv2.boundingRect(cnt)
#         cv2.rectangle(img1,(x,y),(x+w1,y+h1),(0,255,0),2)
        flat[x] += 1
        flat[x+w1] += 1

    kernel = signal.gaussian(kernlen, std=std)
    flat = np.convolve(flat, kernel)

    idx_left, idx_right = find_max(flat, relax=15)
#     print(idx_left, idx_right)

    return img[:, idx_left:idx_right]

def single_words(img, thr=60):
    img = img.astype(np.uint8)
    img_title = find_title_region(img)
    h,w = img_title.shape[:2]

    edges = find_edges(img_title, sigma=1, thr=20)

    kernel = np.ones((5,3))
    mask = cv2.dilate(edges.astype(np.uint8), kernel)

    # plot_img(mask)

    contours_words, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    img1 = img_title.copy()
    words = []
    for cnt in contours_words:
#         M = cv2.moments(cnt)
        x, y, w1, h1 = cv2.boundingRect(cnt)
        if w1*h1 > h*w//thr:
            cv2.rectangle(img1,(x,y),(x+w1,y+h1),(0,255,0),2)
            words.append(img_title[y:y+h1,x:x+w1])
    # plot_img(img1)

    return words

def single_letters(word, ups_factor=3, sigma=1, thr=10, area_thr=10):
    h,w = word.shape[:2]
    word_ups = scipy.ndimage.zoom(word, (ups_factor,ups_factor,1), order=1)
    edges = find_edges(word_ups, sigma=sigma, thr=thr)
    kernel = np.ones((3,3))
    edges = cv2.dilate(edges.astype(np.uint8), kernel, iterations=1)

    # plot_img(edges)

    contour_letters, hierarchy = cv2.findContours(edges.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    letters = []
    img1 = word_ups.copy()
    for cnt in contour_letters:
#         M = cv2.moments(cnt)
        x, y, w1, h1 = cv2.boundingRect(cnt)
        if w1*h1 > h*w//area_thr:
            cv2.rectangle(img1,(x,y),(x+w1,y+h1),(0,255,0),1)
            letters.append(word_ups[y:y+h1,x:x+w1])

    # plot_img(img1)

    return letters

def image2letters(img):
    words = single_words(img)
    words_let = {}
    for i,word in enumerate(words):
        letters = single_letters(word)
        words_let[i] = letters
        plt.figure(dpi=200)
        for i,letter in enumerate(letters):
            plt.subplot(1,len(letters),i+1)
            plt.imshow(cv2.cvtColor(letter, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        plt.show()

    return words, words_let

#different title segmentation algorithm

def filter_cnts(contours, mask):
    h,w = mask.shape[:2]

    mask1 = mask.copy()
    contours_filtered = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            x0, y0, w0, h0 = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area
            ratio = np.max([w0,h0]) / np.min([w0,h0])

            fl1 = h0 > int(0.9 * w)
            fl2 = (w0 > int(0.5 * w)) & (ratio > 5)
            fl3 = area < (w/35)**2
            fl4 = abs(area - h0*w0) < area*0.1
#             fl4 = (0.4 < ratio < 1.7) & (solidity > 0.9)

            if fl1 | fl2 | fl3 | fl4:
                cv2.drawContours(mask1, [cnt], 0, 0, -1)
            else:
                contours_filtered.append(cnt)

    return contours_filtered, mask1

def book2mask(book, window_size=40, diff_thr=50, gap=5, verbose=False, return_windows=False):
    h,w = book.shape[:2]
    imgray = cv2.cvtColor(book, cv2.COLOR_BGR2GRAY)

    n_windows = h // window_size
    windows = []

    book_mask = np.zeros_like(imgray)

    for i in range(n_windows):
        window_img = book[window_size*i:window_size*(i+1)]
        window = imgray[window_size*i:window_size*(i+1)]
        if np.max(window) - np.min(window) > diff_thr:
            ret, thr = cv2.threshold(window, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            idx_black = np.where(thr==0)
            idx_black = [(x,y) for x,y in zip(idx_black[0], idx_black[1])]
            idx_black = sorted(idx_black, key=lambda x: x[1])

            idx_white = np.where(thr==255)
            idx_white = [(x,y) for x,y in zip(idx_white[0], idx_white[1])]
            idx_white = sorted(idx_white, key=lambda x: x[1])

            fl1 = (idx_black[0][0] > idx_white[0][0]) | (idx_black[-1][0] < idx_white[-1][0])

            left_avg = np.mean(thr[:,:gap] / 255)
            right_avg = np.mean(thr[:,-gap:] / 255)

            if left_avg > 0.5 and right_avg > 0.5:
                thr = 255 - thr

        else:
            thr = np.zeros_like(window, dtype=np.uint8)

        windows.append((window_img,thr))
        book_mask[window_size*i:window_size*(i+1)] = thr

        if verbose:
            plt.figure(dpi=100)
            plt.subplot(1,2,1)
            plt.imshow(cv2.cvtColor(book[window_size*i:window_size*(i+1)], cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.subplot(1,2,2)
            plt.imshow(thr, cmap='gray')
            plt.axis('off')
            plt.show()

        contours, hierarchy = cv2.findContours(book_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_filtered, mask = filter_cnts(contours, book_mask)

    if return_windows:
        return book_mask, mask, contours_filtered, windows
    else:
        return book_mask, mask, contours_filtered

def find_cnt_dist(cnt1, cnt2, angle=False):
    n_pts1 = cnt1.shape[0]
    n_pts2 = cnt2.shape[0]

    min_dist = 10000
    pts_min = (0,0)
    for i in range(n_pts2):
        pt0 = cnt2[i,0,:]
        dist = np.array([np.linalg.norm(pt-pt0) for pt in cnt1])
        if dist.min() < min_dist:
            min_dist = dist.min()
            idx = np.argmin(dist)
            pts_min = (idx, i)

    if angle:
        pt1 = cnt1[pts_min[0],0]
        pt2 = cnt2[pts_min[1],0]

        M1 = cv2.moments(cnt1)
        M2 = cv2.moments(cnt2)

        cx1 = int(M1['m10']/M1['m00'])
        cy1 = int(M1['m01']/M1['m00'])

        cx2 = int(M2['m10']/M2['m00'])
        cy2 = int(M2['m01']/M2['m00'])

        if abs(cx2 - cx1) > abs(cy2 - cy1):
            if (cx2-cx1) > 0:
                min_angle = np.arctan(abs(cy2-cy1) / abs(cx2-cx1))
            else:
                min_angle = 0
        else:
            if (cy2-cy1) > 0:
                min_angle = np.arctan(abs(cx2-cx1) / abs(cy2-cy1))
            else:
                min_angle = 0

        assert 0 <= min_angle <= np.pi/2

        return min_dist, min_angle
    else:
        return min_dist

def find_neib(cnt, neibs):
    min_dist = 10000
    idx = -1
    for i,neib in enumerate(neibs):
        dist = find_cnt_dist(cnt, neib)
        if dist < min_dist:
            min_dist = dist
            idx = i

    neib = neibs[idx]
    neibs.pop(idx)

    return neib, neibs, min_dist

def check_relax(x,y,relax):
    if (y-relax) > 0:
        y_start = y - relax
    else:
        y_start = y
    if (x-relax) > 0:
        x_start = x - relax
    else:
        x_start = x

    return x_start, y_start

def check_orientation(words):
    if len(words) > 1:
        start = words[1]
        end = words[-1]

        M1 = cv2.moments(start)
        M2 = cv2.moments(end)

        cx1 = int(M1['m10']/M1['m00'])
        cy1 = int(M1['m01']/M1['m00'])

        cx2 = int(M2['m10']/M2['m00'])
        cy2 = int(M2['m01']/M2['m00'])

        if abs(cx2-cx1) > abs(cy2-cy1):
            fl_rot = False
        else:
            fl_rot = True
    else:
        fl_rot = False

    return fl_rot

def get_word_mask(words, img, mask, relax=0):
    words_mask = []

    words_rot = [check_orientation(word) for word in words]

    for i,word in enumerate(words):
        word_mask = {'cnt': [], 'mask': []}
        word_mask['cnt'] = word
        for symb in word:
            x,y,w,h = cv2.boundingRect(symb)
            x_start, y_start = check_relax(x,y,relax)
            symb_mask = mask[y_start:(y+h+relax),x_start:(x+w+relax)]
            if words_rot[i]:
                symb_mask = np.rot90(symb_mask)
            word_mask['mask'].append(symb_mask)
        words_mask.append(word_mask)

    words_connected = []
    for word in words:
        word_con = word[0]
        for i in range(1,len(word)):
            word_con = np.vstack((word_con,word[i]))

        words_connected.append(word_con)

    words_connected_mask = {'cnt': words_connected, 'mask':[], 'img':[]}
    for i,word in enumerate(words_connected):
        x,y,w,h = cv2.boundingRect(word)
        x_start, y_start = check_relax(x,y,relax)
        mask_ = mask[y_start:(y+h+relax),x_start:(x+w+relax)]
        img_ = img[y_start:(y+h+relax),x_start:(x+w+relax), :]
        if words_rot[i]:
            mask_ = np.rot90(mask_)
            img_ = np.rot90(img_, axes=(0,1))
        words_connected_mask['mask'].append(mask_)
        words_connected_mask['img'].append(img_)

    return words_mask, words_connected_mask


def calc_stats(words):
    word_stats = []
    for i,word in enumerate(words):
        stats = {}
        n_symb = len(word)
        if n_symb > 2:
            dist, angle = [], []
            for j in range(n_symb-1):
                d, alpha = find_cnt_dist(word[j], word[j+1], angle=True)
                dist.append(d)
                angle.append(alpha)
            dist = np.array(dist)
            angle = np.array(angle)
            stats['dist_mean'] = np.mean(dist)
            stats['dist_std'] = np.std(dist)
            stats['angle_mean'] = np.mean(angle)
        elif n_symb == 2:
            d, alpha = find_cnt_dist(word[0], word[-1], angle=True)
            stats['dist_mean'] = d
            stats['dist_std'] = d*0.7
            stats['angle_mean'] = alpha

        word_stats.append(stats)

    return word_stats


def connect_components(words):
    words_comb = []
    words_merged = [word for word in words]
    word_stats = calc_stats(words_merged)
    fl_merged = True
    while fl_merged:
        fl_merged = False

        for i in range(len(words_merged)-1):
            word1 = words_merged[i]
            word2 = words_merged[i+1]
            d, alpha = find_cnt_dist(word1[-1], word2[0], angle=True)
            fl = False

            if (len(word1) >= 2) and (len(word2) >= 2):
                m1, std1 = word_stats[i]['dist_mean'], word_stats[i]['dist_std']
                m_alpha1, std_alpha1 = word_stats[i]['angle_mean'], 15 / 180 * np.pi

                m2, std2 = word_stats[i+1]['dist_mean'], word_stats[i+1]['dist_std']
                m_alpha2, std_alpha2 = word_stats[i+1]['angle_mean'], 15 / 180 * np.pi

                fl1 = (m1 - std1 < d < m1 + std1) or (m2 - std2 < d < m2 + std2)
                fl2 = (m_alpha1 - std_alpha1 < alpha < m_alpha1 + std_alpha1) or (m_alpha2 - std_alpha2 < alpha < m_alpha2 + std_alpha2)

                fl = fl1 and fl2
            elif (len(word1) >= 2):
                m, std = word_stats[i]['dist_mean'], word_stats[i]['dist_std']
                m_alpha, std_alpha = word_stats[i]['angle_mean'], 15 / 180 * np.pi

                fl = (m - std < d < m + std) and (m_alpha - std_alpha < alpha < m_alpha + std_alpha)
            elif (len(word2) >= 2):
                m, std = word_stats[i+1]['dist_mean'], word_stats[i+1]['dist_std']
                m_alpha, std_alpha = word_stats[i+1]['angle_mean'], 15 / 180 * np.pi

                fl = (m - std < d < m + std) and (m_alpha - std_alpha < alpha < m_alpha + std_alpha)

            if fl:
                fl_merged = True
                new_word = word1 + word2
                words_comb.append(new_word)
                i += 1
            else:
                words_comb.append(words_merged[i])

        words_merged = [word for word in words_comb]
        word_stats = calc_stats(words_merged)
        words_comb = []


    return words_merged

def cnt2words(mask, img, contours, thr=1.6, relax=3, connect=False, verbose=False):
    h,w = mask.shape

    words = []

    contours = sorted(contours, key=lambda cnt: cv2.moments(cnt)['m01'] / cv2.moments(cnt)['m00'])

    path = [contours[0]]
    neibs = contours[1:]
    dists = []

    for i in range(len(contours)-1):
        next_stop, neibs, min_dist = find_neib(path[i], neibs)
        path.append(next_stop)
        dists.append(min_dist)

    if verbose:
        img1 = img.copy()
        for i in range(1,len(path)):
            M1 = cv2.moments(path[i-1])
            M2 = cv2.moments(path[i])

            cx1 = int(M1['m10']/M1['m00'])
            cy1 = int(M1['m01']/M1['m00'])

            cx2 = int(M2['m10']/M2['m00'])
            cy2 = int(M2['m01']/M2['m00'])

            cv2.circle(img1, (cx1,cy1), radius=5, color=(255, 0, 0), thickness=-1)
            cv2.circle(img1, (cx2,cy2), radius=5, color=(255, 0, 0), thickness=-1)

            cv2.line(img1, (cx1,cy1), (cx2,cy2), color=(0, 0, 255), thickness=3)

        plot_img(img1)

    word = [path[0]]
    fl = False
    for i in range(1,len(dists)):

        if (dists[i-1] > thr*dists[i]) & (not fl):
            words.append(word)
            word = [path[i]]
        elif dists[i] > thr*dists[i]:
            word.append(path[i])
            words.append(word)
            word = [path[i+1]]
            fl = True
#             i += 1
        else:
            word.append(path[i])
            fl = False

    if (not fl) and (dists[-1] > dists[-2]*thr):
        word.append(path[i+1])
        words.append(word)
    else:
        words.append(word)
        words.append([path[i+1]])

    if connect:
        words = connect_components(words)

    words_mask, words_connected_mask = get_word_mask(words, img, mask, relax=relax)

    return words_mask, words_connected_mask


def load_model(model_path):
    model = Model()
    model = model.double()
    model.load_state_dict(torch.load(model_path))

    return model


def tesseract_predict(img, lang='rus'):
    pred = pytesseract.image_to_string(img, lang=lang)

    return pred


def book_predict(book_img, labels, model_path='comnist_cls.pth'):
    book_pred_tes = tesseract_predict(book_img)

    book_img = find_title_region(book_img)
    book_mask, mask, contours_filtered = book2mask(book_img, window_size=30)
    words_mask, words_connected_mask = cnt2words(mask, book_img, contours_filtered, verbose=True)

    word_pred_tes = []
    for word,mask in zip(words_connected_mask['img'], words_connected_mask['mask']):
        pred = (tesseract_predict(word), tesseract_predict(mask))
        word_pred_tes.append(pred)

    model = load_model(model_path)
    model.eval()
    book_pred_cls = []
    for word in words_mask:
        pred_cls = ''
        for symb in word['mask']:
            input = cv2.resize(symb.astype(np.uint8), (64,64)).reshape((1,1,64,64))
            input = np.array(input, dtype=np.float64)
            input = torch.tensor(input, dtype=torch.double)
            output = model(input.double())
            idx_pred = np.argmax(output.detach().reshape(-1))
            pred_cls += labels[idx_pred]
        book_pred_cls.append(pred_cls)

    return (book_pred_tes, word_pred_tes), book_pred_cls


def read_title(img, model_path='comnist_cls_2.pth', verbose=False):
    if np.max(img.shape) >= 1000:
        resc_factor = np.max(img.shape) // 1000
    else:
        resc_factor = 1

    img_resc, img_rotated, lines_filtered, lines_rotated = rotate_img(img, resc_factor=resc_factor, kernel=(5,5))
    books = extract_books_warped(img_rotated, lines_rotated)

    if verbose:
        plt.figure(dpi=200)
        for i,book in enumerate(books):
            plt.subplot(1,len(books),i+1)
            plt.imshow(cv2.cvtColor(book, cv2.COLOR_BGR2RGB))
            plt.axis('off')
        plt.show()

    idx = -2
    book_img = books[idx]

    labels = '| а б в г д е ж з и к л м н о п р с т у ф х ц ч ш щ ъ ы ь э ю я'.split()
    (book_pred_tes, word_pred_tes), book_pred_cls = book_predict(book_img, labels, model_path)

    return (book_pred_tes, word_pred_tes), book_pred_cls
