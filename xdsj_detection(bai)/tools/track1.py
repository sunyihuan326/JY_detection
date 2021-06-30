import cv2.cv as cv

if __name__ == '__main__':
    capture = cv.CaptureFromCAM(0)
    while True:
        im = cv.QueryFrame(capture)
        # 在此处添加图片名称
        tmp = cv.LoadImage('part.jpg')

        w, h = cv.GetSize(im)
        W, H = cv.GetSize(tmp)

        width = w - W + 1
        height = h - H + 1
        result = cv.CreateImage((width, height), 32, 1)
        cv.MatchTemplate(im, tmp, result, cv.CV_TM_SQDIFF)
        min_val, max_val, min_loc, max_loc = cv.MinMaxLoc(result)

        pt1 = min_loc
        pt2 = (int(min_loc[0]) + W, int(min_loc[1]) + H)
        cv.Rectangle(im, pt1, pt2, cv.CV_RGB(0, 255, 0), 1)
        cv.NamedWindow("test", 1)
        cv.ShowImage("test", im)
        if cv.WaitKey(10) == 27:
            break
