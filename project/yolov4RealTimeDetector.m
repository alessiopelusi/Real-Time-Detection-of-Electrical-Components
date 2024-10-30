addpath('src');

% Connect to the webcam
cam = webcam("Integrated Webcam_FHD");
cam.Resolution = '1920x1080';
video_Player = vision.VideoPlayer('Position',[400 -50 900 900]);

% Used figure handles on the player object to automatically press the
% "Maintain fit to window" button in order to have the webcam view centered.
set(0,'showHiddenHandles','on')
fig_handle = gcf ;  
fig_handle.findobj 
ftw = fig_handle.findobj ('TooltipString', 'Maintain fit to window');   
ftw.ClickedCallback()

fig = uifigure('Position',[10 100 400 570]);
h0 = uihtml(fig);
h0.Position = [10 10 380 550];

fig1 = uifigure('Position',[1270 100 400 570]);
h1 = uihtml(fig1);
h1.Position = [10 10 380 550];

fig2 = uifigure('Position',[400 680 900 400]);
h2 = uihtml(fig2);
h2.Position = [10 10 880 380];

%% Inizialize loop && Initialize variables used for the main loop
flag_condensatore = 0;
flag_resistore = 0;
flag_induttanza = 0;

%% Load Network
load("yolov4_fineTuning/yolov4_trained.mat");
% Specify the classNames to be used in the training.
classNames = {'capacitor', 'resistor', 'inductor'};

while true
    % Capture a frame from the webcam
    frame = snapshot(cam);
    
    % Run the detector.
    executionEnvironment = 'auto';
    [bboxes, scores, labels] = detectYOLOv4(net, frame, anchors, classNames, executionEnvironment);

    disp(labels);
    
    if isnumeric(labels)
        firstLabel = 'Unknown';
    else
        if iscell(labels) && ~isempty(labels)
            firstLabel = char(labels{1});
        else
            firstLabel = char(labels);
        end
    end
    
    % Check for capacitor
    if (flag_condensatore == 0 && any(strcmpi(firstLabel, 'capacitor'))) && scores > 0.65
        updateHtmlSource(h0, 'condensatore.html');
        flag_condensatore = 1;
    end
    
    % Check for resistor
    if (flag_resistore == 0 && any(strcmpi(firstLabel, 'resistor'))) && scores > 0.65
        updateHtmlSource(h1, 'resistore.html');
        flag_resistore = 1;
    end
    
    % Check for inductor
    if (flag_induttanza == 0 && any(strcmpi(firstLabel, 'inductor'))) && scores > 0.65
        updateHtmlSource(h2, 'induttanza.html');
        flag_induttanza = 1;
    end
    
    % Display the annotated image if detected
    disp(scores)
    if scores > 0.65
        annotatedImage = insertObjectAnnotation(frame, "rectangle", bboxes, labels, 'LineWidth', 4, 'FontSize', 24);
        step(video_Player, annotatedImage);
    else
        step(video_Player, frame);
    end
    
end

function updateHtmlSource(h, htmlFilename)
    % Update HTMLSource for the specified uihtml object
    h.HTMLSource = fullfile(pwd, 'HTML_components', htmlFilename);
end