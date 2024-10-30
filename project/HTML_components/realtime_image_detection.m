% Connect to the webcam
cam = webcam("Integrated Webcam_FHD");
cam.Resolution = '1920x1080';

% Create a video player object for real-time display
video_Player = vision.VideoPlayer('Position',[400 100 900 900]);

% Used figure handles on the player object to automatically press the
% "Maintain fit to window" button in order to have the webcam view centered.
set(0,'showHiddenHandles','on')
fig_handle = gcf ;  
fig_handle.findobj 
ftw = fig_handle.findobj ('TooltipString', 'Maintain fit to window');   
ftw.ClickedCallback()  

% Initialize variables used for the main loop
run_loop = true;
frame_Count = 0;
flag_condensatore = 0;
flag_resistore = 0;
flag_induttanza = 0;

% Initialize maps to store identified labels from the
% neural network as key-value pairs, where each label is associated
% with the number of times the network detects it.
objectDetectorLabelOccurrenceMap = containers.Map;

% Download and load the pre-trained YOLOv2 Indoor Object Detector
% pretrainedURL = "https://www.mathworks.com/supportfiles/vision/data/yolov2IndoorObjectDetector23b.zip";
% pretrainedFolder = fullfile(tempdir,"pretrainedNetwork");
% pretrainedNetworkZip = fullfile(pretrainedFolder, "yolov2IndoorObjectDetector23b.zip"); 
% 
% if ~exist(pretrainedNetworkZip,"file")
%     mkdir(pretrainedFolder);
%     disp("Downloading pretrained network (6 MB)...");
%     websave(pretrainedNetworkZip, pretrainedURL);
% end
% 
% unzip(pretrainedNetworkZip, pretrainedFolder)
% 
% pretrainedNetwork = fullfile(pretrainedFolder, "yolov2IndoorObjectDetector.mat");

trained = load("../yolov4_trained.mat");
detector = trained.detector;


fig = uifigure('Position',[10 100 400 570]);
h = uihtml(fig);
h.Position = [10 10 380 550];

% Main loop that executed the real time object detection and also the real
% time identification frame by frame
while run_loop && frame_Count < 50
    % Capture a frame from the webcam
    video_Frame = snapshot(cam);
    frame_Count = frame_Count + 1;

    % Detect objects in the frame using YOLOv2
    [bbox, score, label]  = detect(detector, video_Frame);

    % Annotate the frame with bounding boxes and labels
    annotatedImage = insertObjectAnnotation(video_Frame, "rectangle", bbox, label, 'LineWidth', 4, 'FontSize', 24);
    
    % Display the annotated frame in real-time
    step(video_Player, annotatedImage);

    % Dispaying the HTML card that shows the charateristic of the object
    if flag_condensatore == 0 && label=="capacitor"
        h.HTMLSource = fullfile(pwd,'condensatore.html');
        flag_condensatore = flag_condensatore+1;
    end

    if flag_resistore == 0 && label=="resistor"
        h.HTMLSource = fullfile(pwd,'resistore.html');
        flag_resistore = flag_resistore+1;
    end

    if flag_induttanza == 0 && label=="inductor"
        h.HTMLSource = fullfile(pwd,'induttanza.html');
        flag_induttanza = flag_induttanza+1;
    end

    % Update the occurrence count of labels detected by the object detector
    updateLabelOccurrence(objectDetectorLabelOccurrenceMap, label);

    % Pause for a short duration to control the display rate
    pause(0.1);
 
end 

close(fig);

% Close the video_Player window
release(video_Player);

% Extract keys and values from objectDetectorLabelOccurrenceMap for the second bar graph
objectDetectorKeysArray = keys(objectDetectorLabelOccurrenceMap);
objectDetectorValuesArray = cell2mat(values(objectDetectorLabelOccurrenceMap));

% Create a new figure
figure('Position', [140 100 1400 900]);

% Subplot for labels detected by object detector
subplot(1, 2, 2);
bar(objectDetectorValuesArray, 'FaceColor', 'flat', 'EdgeColor', 'none', 'CData',[.9 .7 .8], 'FaceAlpha', 0.7,'LineStyle','-','EdgeAlpha', 0.8, 'LineWidth', 0.8);
xticklabels(objectDetectorKeysArray);
xlabel('Labels Detected by Object Detector');
ylabel('Occurrence');
title('Labels Detected by Object Detector');

% Clear webcam connection
clear cam;

function updateLabelOccurrence(labelMap, labels)
    % Update the occurrence count for labels in a given map.

    if iscell(labels)
        % If labels are in a cell array, indicating multiple labels per frame
        for i = 1:length(labels)
            currentLabels = labels{i};
            % Iterate over each label in the cell array
            for j = 1:numel(currentLabels)
                % Convert and check the validity of the current label
                currentLabel = convertAndCheckLabel(char(currentLabels(j)));
                % Update the occurrence count for the single label in the map
                updateSingleLabelOccurrence(labelMap, currentLabel);
            end
        end
    else
        % If labels are not in a cell array, indicating a single label per frame
        currentLabels = labels;
        % Iterate over each label
        for i = 1:numel(currentLabels)
            % Convert and check the validity of the current label
            currentLabel = convertAndCheckLabel(char(currentLabels(i)));
            % Update the occurrence count for the single label in the map
            updateSingleLabelOccurrence(labelMap, currentLabel);
        end
    end
end

function currentLabel = convertAndCheckLabel(label)
    % Convert the label to a string and check for empty labels
    if isempty(label)
        currentLabel = "no object detected";
    else
        currentLabel = label;
    end
end










