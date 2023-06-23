import cv2
import numpy as np
import argparse

def help():
    print("\nThis program demonstrates GrabCut segmentation -- select an object in a region")
    print("and then grabcut will attempt to segment it out.")
    print("Call:")
    print("python grabcut.py <image_name>")
    print("\nSelect a rectangular area around the object you want to segment\n")
    print("Hot keys:")
    print("\tESC - quit the program")
    print("\tr - restore the original image")
    print("\tn - next iteration")
    print("\n")
    print("\tleft mouse button - set rectangle")
    print("\n")
    print("\tCTRL+left mouse button - set GC_BGD pixels")
    print("\tSHIFT+left mouse button - set GC_FGD pixels")
    print("\n")
    print("\tCTRL+right mouse button - set GC_PR_BGD pixels")
    print("\tSHIFT+right mouse button - set GC_PR_FGD pixels\n")

class GCApplication:
    def __init__(self):
        self.NOT_SET = 0
        self.IN_PROCESS = 1
        self.SET = 2
        self.radius = 2
        self.thickness = -1
        self.rectState = self.NOT_SET
        self.lblsState = self.NOT_SET
        self.prLblsState = self.NOT_SET
        self.isInitialized = False
        self.rect = (0, 0, 1, 1)
        self.bgdPxls = []
        self.fgdPxls = []
        self.prBgdPxls = []
        self.prFgdPxls = []
        self.iterCount = 0

    def reset(self):
        self.rectState = self.NOT_SET
        self.lblsState = self.NOT_SET
        self.prLblsState = self.NOT_SET
        self.isInitialized = False
        self.bgdPxls = []
        self.fgdPxls = []
        self.prBgdPxls = []
        self.prFgdPxls = []
        self.iterCount = 0

    def setImageAndWinName(self, image, winName):
        self.image = image
        self.winName = winName
        self.mask = np.zeros(image.shape[:2], dtype=np.uint8)

    def showImage(self):
        res = self.image.copy()
        binMask = np.zeros(res.shape[:2], dtype=np.uint8)
        if self.isInitialized:
            binMask[self.mask == 2] = 255
            res = cv2.addWeighted(res, 0.5, cv2.cvtColor(binMask, cv2.COLOR_GRAY2BGR), 0.5, 0.0)
        for pt in self.bgdPxls:
            cv2.circle(res, pt, self.radius, (255, 0, 0), self.thickness)
        for pt in self.fgdPxls:
            cv2.circle(res, pt, self.radius, (0, 0, 255), self.thickness)
        for pt in self.prBgdPxls:
            cv2.circle(res, pt, self.radius, (160, 255, 255), self.thickness)
        for pt in self.prFgdPxls:
            cv2.circle(res, pt, self.radius, (255, 130, 230), self.thickness)
        if self.rectState == self.IN_PROCESS or self.rectState == self.SET:
            cv2.rectangle(res, (self.rect[0], self.rect[1]), (self.rect[0] + self.rect[2], self.rect[1] + self.rect[3]), (0, 255, 0), 2)
        cv2.imshow(self.winName, res)

    def setRectInMask(self):
        self.mask = cv2.GC_BGD * np.ones(self.image.shape[:2], dtype=np.uint8)
        self.mask[self.rect[1]:self.rect[1] + self.rect[3], self.rect[0]:self.rect[0] + self.rect[2]] = cv2.GC_PR_FGD

    def setLblsInMask(self, flags, p, isPr):
        bpxls, fpxls = (self.bgdPxls, self.fgdPxls) if not isPr else (self.prBgdPxls, self.prFgdPxls)
        bvalue, fvalue = (cv2.GC_BGD, cv2.GC_FGD) if not isPr else (cv2.GC_PR_BGD, cv2.GC_PR_FGD)
        if flags & cv2.EVENT_FLAG_CTRLKEY:
            bpxls.append(p)
            cv2.circle(self.mask, p, self.radius, bvalue, self.thickness)
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            fpxls.append(p)
            cv2.circle(self.mask, p, self.radius, fvalue, self.thickness)

    def mouseClick(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            isb = flags & cv2.EVENT_FLAG_CTRLKEY
            isf = flags & cv2.EVENT_FLAG_SHIFTKEY
            if self.rectState == self.NOT_SET and not isb and not isf:
                self.rectState = self.IN_PROCESS
                self.rect = (x, y, 1, 1)
            if (isb or isf) and self.rectState == self.SET:
                self.lblsState = self.IN_PROCESS
        elif event == cv2.EVENT_RBUTTONDOWN:
            isb = flags & cv2.EVENT_FLAG_CTRLKEY
            isf = flags & cv2.EVENT_FLAG_SHIFTKEY
            if (isb or isf) and self.rectState == self.SET:
                self.prLblsState = self.IN_PROCESS
        elif event == cv2.EVENT_LBUTTONUP:
            if self.rectState == self.IN_PROCESS:
                if self.rect[0] == x or self.rect[1] == y:
                    self.rectState = self.NOT_SET
                else:
                    self.rect = (self.rect[0], self.rect[1], x - self.rect[0], y - self.rect[1])
                    self.rectState = self.SET
                    self.setRectInMask()
                    assert len(self.bgdPxls) == 0 and len(self.fgdPxls) == 0 and len(self.prBgdPxls) == 0 and len(self.prFgdPxls) == 0
            self.showImage()
            if self.lblsState == self.IN_PROCESS:
                self.lblsState = self.SET
                self.setLblsInMask(flags, (x, y), False)
                self.nextIter()
                self.showImage()
            else:
                if self.rectState == self.SET:
                    self.nextIter()
                    self.showImage()
        elif event == cv2.EVENT_RBUTTONUP:
            if self.prLblsState == self.IN_PROCESS:
                self.setLblsInMask(flags, (x, y), True)
                self.prLblsState = self.SET
            if self.rectState == self.SET:
                self.nextIter()
                self.showImage()
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.rectState == self.IN_PROCESS:
                self.rect = (self.rect[0], self.rect[1], x - self.rect[0], y - self.rect[1])
                assert len(self.bgdPxls) == 0 and len(self.fgdPxls) == 0 and len(self.prBgdPxls) == 0 and len(self.prFgdPxls) == 0
                self.showImage()
            elif self.lblsState == self.IN_PROCESS:
                self.setLblsInMask(flags, (x, y), False)
                self.showImage()
            elif self.prLblsState == self.IN_PROCESS:
                self.setLblsInMask(flags, (x, y), True)
                self.showImage()

    def nextIter(self):
        if self.isInitialized:
            cv2.grabCut(self.image, self.mask, self.rect, self.bgdModel, self.fgdModel, 1, cv2.GC_INIT_WITH_MASK)
        else:
            if self.rectState != self.SET:
                return self.iterCount
            if self.lblsState == self.SET or self.prLblsState == self.SET:
                cv2.grabCut(self.image, self.mask, self.rect, self.bgdModel, self.fgdModel, 1, cv2.GC_INIT_WITH_MASK)
            else:
                cv2.grabCut(self.image, self.mask, self.rect, self.bgdModel, self.fgdModel, 1, cv2.GC_INIT_WITH_RECT)
                self.isInitialized = True
        self.iterCount += 1
        self.bgdPxls = []
        self.fgdPxls = []
        self.prBgdPxls = []
        self.prFgdPxls = []
        return self.iterCount


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GrabCut segmentation')
    parser.add_argument('image_name', type=str, help='path to the input image')
    args = parser.parse_args()

    gcapp = GCApplication()
    gcapp.setImageAndWinName(cv2.imread(args.image_name), "image")
    gcapp.showImage()

    while True:
        c = cv2.waitKey(0)
        if c == 27:
            print("Exiting...")
            break
        elif c == ord('r'):
            print()
            gcapp.reset()
            gcapp.showImage()
        elif c == ord('n'):
            iterCount = gcapp.getIterCount()
            print("<{}... ".format(iterCount), end='')
            newIterCount = gcapp.nextIter()
            if newIterCount > iterCount:
                gcapp.showImage()
                print("{}>".format(iterCount))
            else:
                print("rect must be determined>")
