baseFolder = 'DOE_Project';
participant_list = {'Danny', 'Selim', 'Ruba', 'Sophie'};
participants = {'Danny/04152025', 'Selim/04152025', 'Ruba/04152025', 'Sophie/04152025'};
numTrials = 27;
g = 9.81;  % gravity in m/s^2
verbose = true;

for p = 1
    participant = participants{p};
    participant_name = participant_list{p};
    participantFolder = fullfile(baseFolder, participant);
    results = NaN(numTrials, 4);  % Store [Impulse, Airtime, JumpHeight_impulse, JumpHeight_flight]
    
    processTrials = 1:numTrials;
    if p == 3 
        processTrials = [1:12 15:29];
    end
    for t = processTrials
        
        fileName = fullfile(participantFolder, sprintf('Trial %d.csv', t));
        try
            % Read matrix, skipping first 4 header rows
            data = readmatrix(fileName, 'NumHeaderLines', 4);

            % Sum Fy columns: indices are 4, 19, 34, 43, 52, 61 (1-based)
            fyCols = [4, 19, 34, 43, 52, 61] + 1;
            Fy = sum(data(:, fyCols), 2);
            time = (0:length(Fy)-1)' / 1200;  % assuming 1200 Hz

            % 1. Landing detection
            [~, peakIdx] = min(Fy);
            region = 65;
            flightBand = abs(Fy) < region;
            landingIdx = NaN;
            for i = peakIdx:-1:101
                if all(flightBand(i-99:i))
                    landingIdx = i - 99;
                    break;
                end
            end
            if isnan(landingIdx)
                warning("Stable landing not found in Trial %d of %s", t, participant);
                continue;
            end
            landingIdx = landingIdx + 100;

            % 2. Liftoff detection
            liftoffIdx = landingIdx - 100;
            while liftoffIdx > 1 && flightBand(liftoffIdx)
                liftoffIdx = liftoffIdx - 1;
            end
            liftoffIdx = liftoffIdx + 1;

            % 3. Pushoff detection
            excludeWindow = 250;
            if liftoffIdx <= excludeWindow
                warning("Not enough data before liftoff in Trial %d of %s", t, participant);
                continue;
            end
            searchEnd = liftoffIdx - excludeWindow;
            [~, pushoffIdx] = max(Fy(1:searchEnd));

            % 4. Compute physics quantities
            Fy = Fy*-1;
            dt = mean(diff(time));
            bodyweight = mean(Fy(2:100));
            mass = bodyweight / g;
            netForce = Fy(pushoffIdx:liftoffIdx) - bodyweight;
            impulse = trapz(netForce) * dt;

            v_takeoff = impulse / mass;
            jumpHeight_impulse = v_takeoff^2 / (2 * g);

            flightTime = (landingIdx - liftoffIdx) * dt;
            jumpHeight_flight = g * flightTime^2 / 8;

            % 5. Save results
            results(t, :) = [impulse, flightTime, jumpHeight_impulse, jumpHeight_flight];

            Fy = Fy;
            % 6. Plot if verbose
            if verbose
                figure;
                set(gcf,'Color','white')
                plot(time, Fy, 'b-', 'LineWidth', 1.2); hold on;
                xline(time(pushoffIdx), 'b--', 'LineWidth', 1.5, 'DisplayName', 'Pushoff');
                xline(time(liftoffIdx), 'r--', 'LineWidth', 1.5, 'DisplayName', 'Liftoff');
                xline(time(landingIdx), 'g--', 'LineWidth', 1.5, 'DisplayName', 'Landing');
                title(sprintf('Participant: %s | Trial %d', participant_name, t), 'Interpreter', 'none');
                xlabel('Time (s)');
                ylabel('Total Vertical Force (Fy, N)');
                legend('show');
                grid on;
            end

        catch ME
            warning("Error processing Trial %d for %s: %s", t, participant_name, ME.message);
        end
    end

    % 7. Write to Excel with headers
    headers = {'Impulse (Ns)', 'Airtime (s)', 'Jump Height (Impulse, m)', 'Jump Height (Flight, m)'};
    outputFile = fullfile(baseFolder, sprintf('%s_JumpResults.xlsx', erase(participant_name, '/')));
    writecell([headers; num2cell(results)], outputFile);
end
%%
% --- Combine all participant data and run regression ---
allData = table();
participant_list = {'Danny', 'Selim', 'Ruba', 'Sophie'};

for p = 1:length(participant_list)
    name = participant_list{p};
    filePath = fullfile(baseFolder, [name '_JumpResults.xlsx']);
    if isfile(filePath)
        T = readtable(filePath);
        T.Participant = repmat({name}, height(T), 1);
        T.Trial = (1:height(T))';
        allData = [allData; T];
    else
        warning('Missing file for %s', name);
    end
end

% Rename for easier reference
ImpulseJump = allData.("JumpHeight_Impulse_M_");
FlightJump = allData.("JumpHeight_Flight_M_");

% Remove any NaNs
validRows = ~isnan(ImpulseJump) & ~isnan(FlightJump);
ImpulseJump = ImpulseJump(validRows);
FlightJump = FlightJump(validRows);

% Regression
mdl = fitlm(FlightJump, ImpulseJump);
r2 = mdl.Rsquared.Ordinary;
xFit = linspace(min(FlightJump), max(FlightJump), 100);
yFit = predict(mdl, xFit');

% Plot
figure;
set(gcf,'Color','white')
gscatter(FlightJump, ImpulseJump, allData.Participant(validRows), 'krgb', 'o^sd', 8, 'on');
set(gca, 'Color', 'w');
hold on;
plot(xFit, yFit, 'k--', 'LineWidth', 2);
xlabel('Jump Height (Flight-based, m)');
ylabel('Jump Height (Impulse-based, m)');
title(sprintf('Impulse vs Flight Jump Height (R^2 = %.3f)', r2));
legend('Location', 'best');
grid on;
