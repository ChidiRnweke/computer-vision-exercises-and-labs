"""Skeleton code for python script to process a video using OpenCV package
:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import cv2
import numpy as np
import textwrap


# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper


def subtitle(frame, text):
    wrapped_text = textwrap.wrap(text, width=35)

    x, y = 10, 40
    font_size = 1
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, line in enumerate(wrapped_text):
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]

        gap = textsize[1] + 10

        y = int((frame.shape[0] + textsize[1]) / 2) + i * gap
        x = int((frame.shape[1] - textsize[0]) / 2)

        cv2.putText(
            frame,
            line,
            (x, y),
            font,
            font_size,
            (0, 0, 0),
            font_thickness,
            lineType=cv2.LINE_AA,
        )


def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord("q"):
                break
            if between(cap, 0, 4000):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                subtitle(frame, "This high production movie starts with gray scale")

            if between(cap, 4000, 5500):
                frame = cv2.GaussianBlur(frame, (21, 21), 0)
                subtitle(frame, "...We proceed with gaussian blur as our second sfx")

            if between(cap, 5500, 6500):
                frame = cv2.GaussianBlur(frame, (49, 49), 0)
                subtitle(
                    frame, "Increasing the kernel means we get a more blurred image"
                )

            if between(cap, 6500, 8000):
                frame = cv2.GaussianBlur(frame, (49, 49), 100)
                subtitle(
                    frame,
                    "We make it even worse by expanding the standard deviation with the same kernel",
                )

            if between(cap, 6500, 8000):
                frame = cv2.GaussianBlur(frame, (49, 49), 100)
                subtitle(
                    frame,
                    "We make it even worse by expanding the standard deviation with the same kernel",
                )

            if between(cap, 8000, 9000):
                frame = cv2.bilateralFilter(frame, 9, 2000, 2000)
                subtitle(
                    frame,
                    "A bilateral filter is a non-linear filter that is more expensive to compute than a gaussian",
                )
            if between(cap, 9000, 10000):
                frame = cv2.bilateralFilter(frame, 9, 2000, 2000)
                subtitle(
                    frame,
                    "It takes an average of filtering in the neighbourhood of the pixel in physical and color space",
                )
            if between(cap, 10000, 11000):
                subtitle(
                    frame,
                    "The effect is subtle, it is currently turned off for a second",
                )
            if between(cap, 11000, 12000):
                frame = cv2.bilateralFilter(frame, 9, 2000, 2000)
                subtitle(
                    frame,
                    "The main advantage is that it preserves edges. It does this by upweighting similar colors. Closer to the edge the other side will be downweighted and not smooth as much",
                )
            if between(cap, 12000, 14000):

                ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)

                subtitle(
                    frame,
                    "Not all the information on this page is interesting... Lets try and only get the boxes with thresholding",
                )

            if between(cap, 14000, 16000):

                ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
                kernel = np.ones((3, 3), np.uint8)
                frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
                subtitle(
                    frame,
                    "removing noise with erosion and dilation makes it slightly better.",
                )

            if between(cap, 16000, 20000):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, frame = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
                kernel = np.ones((3, 3), np.uint8)
                frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
                subtitle(
                    frame,
                    "Lets do it in gray scale for the best result, now we only study what is relevant",
                )
            if between(cap, 20000, 22000):
                subtitle(
                    frame,
                    "Lets try and find some edges next! We'll be using the sobel filter. First we blur the image and then look for gradients in the horizontally and vertically",
                )
                frame = cv2.GaussianBlur(frame, (3, 3), 0)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                grad_x = cv2.Sobel(
                    gray,
                    cv2.CV_16S,
                    1,
                    0,
                    ksize=3,
                    scale=1,
                    delta=0,
                    borderType=cv2.BORDER_DEFAULT,
                )
                grad_y = cv2.Sobel(
                    gray,
                    cv2.CV_16S,
                    0,
                    1,
                    ksize=3,
                    scale=1,
                    delta=0,
                    borderType=cv2.BORDER_DEFAULT,
                )
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                frame = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

            if between(cap, 22000, 23500):
                subtitle(frame, "increasing the kernel")
                frame = cv2.GaussianBlur(frame, (3, 3), 0)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                grad_x = cv2.Sobel(
                    gray,
                    cv2.CV_16S,
                    1,
                    0,
                    ksize=5,
                    scale=1,
                    delta=0,
                    borderType=cv2.BORDER_DEFAULT,
                )
                grad_y = cv2.Sobel(
                    gray,
                    cv2.CV_16S,
                    0,
                    1,
                    ksize=5,
                    scale=1,
                    delta=0,
                    borderType=cv2.BORDER_DEFAULT,
                )
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                frame = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

            if between(cap, 23500, 25000):
                subtitle(frame, "increasing the derivative from 1 to 3")
                frame = cv2.GaussianBlur(frame, (3, 3), 0)

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                grad_x = cv2.Sobel(
                    gray,
                    cv2.CV_16S,
                    3,
                    0,
                    ksize=5,
                    scale=1,
                    delta=0,
                    borderType=cv2.BORDER_DEFAULT,
                )
                grad_y = cv2.Sobel(
                    gray,
                    cv2.CV_16S,
                    0,
                    3,
                    ksize=5,
                    scale=1,
                    delta=0,
                    borderType=cv2.BORDER_DEFAULT,
                )
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                frame = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

            if between(cap, 25000, 30000):
                subtitle(
                    frame,
                    "Time to find circles with the Hough transform: small radius only finds these OOOOOOOO",
                )
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                gray = cv2.medianBlur(gray, 5)

                rows = gray.shape[0]
                circles = cv2.HoughCircles(
                    gray,
                    cv2.HOUGH_GRADIENT,
                    1,
                    rows / 8,
                    param1=100,
                    param2=30,
                    minRadius=1,
                    maxRadius=30,
                )

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        center = (i[0], i[1])
                        # circle center
                        cv2.circle(frame, center, 1, (0, 100, 100), 3)
                        # circle outline
                        radius = i[2]
                        cv2.circle(frame, center, radius, (255, 0, 255), 3)

            if between(cap, 30000, 35000):
                subtitle(frame, "A bigger radius actually finds circles")
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                gray = cv2.medianBlur(gray, 5)

                rows = gray.shape[0]
                circles = cv2.HoughCircles(
                    gray,
                    cv2.HOUGH_GRADIENT,
                    1,
                    rows / 8,
                    param1=200,
                    param2=30,
                    minRadius=50,
                    maxRadius=150,
                )

                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for i in circles[0, :]:
                        center = (i[0], i[1])
                        # circle center
                        cv2.circle(frame, center, 1, (0, 100, 100), 3)
                        # circle outline
                        radius = i[2]
                        cv2.circle(frame, center, radius, (255, 0, 255), 3)

            if between(cap, 35000, 40000):
                subtitle(frame, "Time to find our headphones")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                template = cv2.imread("template.jpg", 0)
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                width, height = template.shape[::-1]
                res = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF_NORMED)
                _, _, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = min_loc
                top_left = max_loc      
                bottom_right = (top_left[0] + width, top_left[1] + height)
                cv2.rectangle(frame, top_left, bottom_right, 255, 2)

            if between(cap, 40000, 60000):
                subtitle(frame, "Sadly the fun stops here!")

            cv2.imshow("Frame", frame)

            # write frame that you processed to output
            out.write(frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main("input_video.mp4", "output_video.mp4")
