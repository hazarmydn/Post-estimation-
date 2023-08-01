        % Pose estimation and gesture classification by SG
open simple-pose-estimation.prj;
pause;
detector=posenet.PoseEstimator;

% Collection of 4 gestures .. database to be extended
% 1 hands are down
% 2 hands are up
% 3 right hand is up
% 4 left hand is up
% proposal e.g.     hands crossed?
% 
% [y1,Fs] = audioread('gest1.m4a');
% [y2,Fs] = audioread('gest2.m4a');
% [y3,Fs] = audioread('gest3.m4a');
% [y4,Fs] = audioread('gest4.m4a');
% [y5,Fs] = audioread('gest5.m4a');
%???
gesture=[1 1 1 2 2 2 3 3 3 4 4 4 5 5 5 6 6 6];
classes=size(unique(gesture),2); % number of classes
    
%concept tab(:,:,i)=marks(:,1:2);
tab(:,:,1)=[88 40 ; 96 28 ; 80 32 ; 108 40; 68 40; 124 92; 44 88; 144 160; 24 156; 148 220; 12 220; 100 220]; %1      
tab(:,:,2)=[84 116; 88 112; 80 112; 92 120; 72 116; 104 144; 60 144; 112 184; 44 184; 108 220; 48 232; 96 224]; %1
tab(:,:,3)=[44 12; 52 8; 36 4; 72 12 ; 24 16; 100 64; 28 72; 132 160; 28 156; 124 228; 32 216; 96 212]; %1
tab(:,:,4)=[72 40; 84 36; 68 32; 92 44; 52 44; 108 92; 32 92; 152 136; 16 148; 140 80; 4 72; 96 216]; %2
tab(:,:,5)=[60 144; 64 144; 56 144; 72 144; 48 148; 84 168; 44 164; 188 128; 16 168; 104 228; 24 144; 80 228]; %2 Tosia
tab(:,:,6)=[92 60; 100 52; 84 52; 112 60; 76 60; 128 100; 64 100; 168 104; 28 108; 148 56; 40 60; 120 204]; %2
tab(:,:,7)=[72 148; 76 144; 68 144; 80 144; 60 144; 92 168; 56 164; 104 204; 28 164; 112 232; 36 140; 80 228]; %3 Tosia
tab(:,:,8)=[96 48; 104 44; 84 44; 116 56; 76 52; 132 100; 60 96; 152 164; 20 132; 160 224; 44 72; 112 216]; %3
tab(:,:,9)=[100 32; 104 28; 92 28; 116 32; 84 36; 132 84; 64 80; 152 148; 12 92; 156 200; 32 36; 120 200]; %3
tab(:,:,10)=[76 68; 80 64; 72 64; 88 68; 60 68; 100 112; 44 108; 136 140; 28 160; 128 100; 20 212; 92 212]; %4
tab(:,:,11)=[84 80; 88 72; 76 76; 96 80; 68 80; 116 108; 52 116; 160 100; 44 168; 128 64; 40 212; 108 208]; %4
tab(:,:,12)=[88 64; 96 60; 80 56; 108 68; 68 60; 120 96; 40 104; 168 60; 20 168; 152 8; 8 236; 100 240]; %4

 tab(:,:,13) = [96 120; 100 112; 92 112; 108 116; 88 116; 112 148; 76 144; 160 160; 32 148; 168 128; 48 248; 116 252]; % New hand gesture (Gesture 5)
 tab(:,:,14)=[88 68; 92 64; 84 64; 96 68; 80 68; 104 104; 72 104; 120 112; 48 112; 116 36; 32 120; 104 120]; %5 hands crossed
 tab(:,:,15)=[92 64; 96 60; 88 60; 100 64; 84 64; 108 96; 76 96; 128 96; 56 96; 124 16; 40 136; 112 136]; %5 hands crossed
 
 
 tab(:,:,16) = [72 200; 76 196; 68 196; 80 200; 60 200; 88 224; 56 220; 104 260; 24 220; 112 288; 40 196; 88 296]; % New leg gesture (Gesture 6)
 tab(:,:,17)=[80 180; 84 176; 76 176; 88 180; 72 180; 96 200; 68 200; 112 220; 56 220; 120 180; 48 240; 100 240]; %6 dancing
 tab(:,:,18)=[88 180; 92 176; 84 176; 96 180; 80 180; 104 200; 76 200; 128 180; 64 180; 128 140; 56 200; 108 200]; %6 dancing


no_of_cases=size(tab,3)
cam = webcam;
player = vision.DeployableVideoPlayer;
I = zeros(256,192,3,'uint8');
player(I);
while player.isOpen
   % Read an image from web camera 
    I = snapshot(cam);
    
    % Crop the image fitting the network input size of 256x192 
    Iinresize = imresize(I,[256 nan]);
    Itmp = Iinresize(:,(size(Iinresize,2)-192)/2:(size(Iinresize,2)-192)/2+192-1,:);
    Icrop = Itmp(1:256,1:192,1:3);
    
    % Predict pose estimation
    heatmaps = detector.predict(Icrop);
    keypoints = detector.heatmaps2Keypoints(heatmaps);
    % Visualize key points
    Iout = detector.visualizeKeyPoints(Icrop,keypoints);
    player(Iout);
    pause(1);
    if ~isOpen(player)
        break
    end
    % only upper part of the body is considered, i.e. 11 first landmarks
    marks=keypoints(1:11,:)
    test=true;
% uncomment below section for teaching    
%       keypoints(1:11,1:2)
%       test=false %for teaching
%     if all(marks(:,3))
%     end
 %end of section
    if all(marks(:,3)) & test
        % goto next phase if all current markers have status '1'
    for i=1:no_of_cases
  break
     % we use only selected landmarks no. 6 to 11
        distance(i)=norm(marks(6:11,1:2)-tab(6:11,:,i))
    end
[minodl_s,index]=sort(distance);
k=4; % k is number of nearest neighbours
clear membership;
j(1:classes)=0; membership(1:classes)=0; 
for no=1:k,
   membership(no)=gesture(index(no));
  if membership(no)==1
   j(1)=j(1)+1
  end
  if membership(no)==2
   j(2)=j(2)+1
  end
  if membership(no)==3
   j(3)=j(3)+1
  end
  if membership(no)==4
   j(4)=j(4)+1
  end
  %???
  if membership(no)==5
   j(5)=j(5)+1
  end
   if membership(no)==6
   j(6)=j(6)+1
  end

end
disp('Gesture classified as: ')
if max(j) == j(1)
'gesture no. 1'
% sound(y1,Fs);
pause(0.5);
elseif max(j) == j(2)
'gesture no. 2'
% sound(y2,Fs);
pause(0.5);
elseif max(j) == j(3)
'gesture no. 3'
% sound(y3,Fs);
pause(0.5);
elseif max(j) == j(4)
'gesture no. 4'
% sound(y4,Fs);
pause(0.5);
% ??? 
elseif max(j) == j(5)
'gesture no. 5'
% sound(y5,Fs);
pause(0.5);
elseif max(j) == j(6)
'gesture no. 6'
% sound(y6,Fs);
pause(0.5);


end

 else
 %do nothing
 end
  
end
imshow(Iout);
clear cam
release(player)