% pose estimation it works on image only
open simple-pose-estimation.prj;
detector = posenet.PoseEstimator;
I = imread('visionteam1.jpg');  % image to work

bbox = [182 74 303 404];
% max size is ;256 192 3]

Iin = imresize(imcrop(I,bbox),detector.InputSize(1:2));
keypoints = detectPose(detector,Iin);
J = detector.visualizeKeyPoints(Iin,keypoints);
imshow(J);