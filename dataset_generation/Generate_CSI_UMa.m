% ------------------------------------------------------------
% QuaDRiGa Narrowband MU-MISO Dataset Generation (UMa)
%
% Generates independent train, validation, or test datasets
% for the considered TDD predictive beamforming setup.
%
% Each sample contains:
%   hist_len = 16 historical channel snapshots
%   fut_len = 4 future channel snapshots
%
% Channel dimensions:
%   H_hist : [S, N, hist_len, Pol, Mv, Mh]
%   H_fut  : [S, N, fut_len, Pol, Mv, Mh]
%
% Mobility information:
%   Pxy     : [S, hist_len, N, 2]
%   Vxy     : [S, hist_len, N, 2]
%   UEspeed : [S, N]
% ------------------------------------------------------------

clc; clear; close all;
%% ================== MODE ==================
mode = 'test';   % 'train' / 'val' / 'test'

switch lower(mode)
    case 'train'
        rng(1);
        S = 35000;
    case 'val'
        rng(2);
        S = 3500;
    case 'test'
        rng(3);
        S = 3500;
    otherwise
        error('mode must be: train / val / test');
end

fprintf('Generating dataset for mode = %s\n', mode);
%% ================== SIMULATION PARAMETERS ==================
s = qd_simulation_parameters;
s.center_frequency = 2.4e9;
%% ================== BS ANTENNA ==================
Mv = 4;
Mh = 4;
Mg_BS = 1;
Ng_BS = 1;
ElcTltAgl_BS = 7;
Hspc = 0.5 * s.wavelength;
Vspc = 0.5 * s.wavelength;

BSAntArray = qd_arrayant.generate('3gpp-mmw', Mh, Mv, s.center_frequency, 2, ElcTltAgl_BS,Vspc / s.wavelength, ...
    Mg_BS, Ng_BS,Vspc / s.wavelength * Mv, Hspc / s.wavelength * Mh);

%% ================== UE ANTENNA ==================
UEAntArray = qd_arrayant.generate('3gpp-mmw', 1, 1, s.center_frequency, 1, ElcTltAgl_BS, ...
    Vspc / s.wavelength, Mg_BS, Ng_BS, Vspc / s.wavelength * Mv, Hspc / s.wavelength * Mh);

%% ================== DATASET PARAMETERS ==================
N = 4;
SpeedSet_kmh = 10:0.1:100;
TimeInterval = 0.5e-3;
hist_len = 16;
fut_len = 4;
SegLen = hist_len + fut_len;     % one sample = 20 snapshots
SegmentsPerTrack = 200;          % non-overlapping samples per track
SnapNum = SegmentsPerTrack * SegLen;
NumTracks = ceil(S / SegmentsPerTrack);
TimeLength = (SnapNum - 1) * TimeInterval;
scenario_name = '3GPP_38.901_UMa_NLOS';
BW = 17280e3;

%% ================== GEOMETRY ==================
BSlocation = [0;0;30];
rho_min = 20;
rho_max = 50;
%% ================== PREALLOCATE ==================
Pol = 2;
Mtot = Pol * Mv * Mh;
H_hist = complex(zeros(S, N, hist_len, Pol, Mv, Mh, 'single'));
H_fut  = complex(zeros(S, N, fut_len, Pol, Mv, Mh, 'single'));
Pxy = zeros(S, hist_len, N, 2, 'single');
Vxy = zeros(S, hist_len, N, 2, 'single');
UEspeed_scene = zeros(S, N, 'single');
scene_idx = 0;

%% ================== MAIN LOOP OVER TRACKS ==================
for tr = 1:NumTracks

    fprintf('Track %d / %d\n', tr, NumTracks);

    %% random speeds for this track
    UESpeeds_kmh = SpeedSet_kmh(randi(numel(SpeedSet_kmh), 1, N));
    UESpeeds_ms = UESpeeds_kmh / 3.6;

    %% scene parameters
    s1 = qd_simulation_parameters;
    s1.center_frequency = 2.4e9;
    s1.use_random_initial_phase = true;
    s1.set_speed(max(UESpeeds_ms), TimeInterval);
    UEcenter = [200; 0; 1.5];

    %% initial UE positions
    rho = rho_min + (rho_max - rho_min) * rand(1, N);
    phi = 120 * rand(1, N) - 60;
    
    UElocation = zeros(3, N);
    for u = 1:N
        UElocation(:,u) = [-rho(u)*cosd(phi(u)); rho(u)*sind(phi(u)); 0] + UEcenter;
    end

    %% build user tracks manually
    UEtrack = qd_track.empty;
    positions = cell(1, N);
    velocities = cell(1, N);

    for u = 1:N

        TrackLength = UESpeeds_ms(u) * TimeLength;
        step_len = UESpeeds_ms(u) * TimeInterval;

        UEtrack(1,u) = qd_track.generate('linear', TrackLength);
        UEtrack(1,u).name = sprintf('%strack%duser%d', mode, tr, u);
        pos_rel = zeros(3, SnapNum);

        % smooth non-linear motion:
        % theta evolves with a slowly varying angular rate
        theta0 = 2*pi*rand;

        % moderate curvature
        omega0 = deg2rad(6 * (2*rand - 1));   % [rad/s]
        omega1 = deg2rad(6 * (2*rand - 1));   % [rad/s]

        theta = zeros(1, SnapNum);
        theta(1) = theta0;

        for t = 1:SnapNum-1
            alpha = (t-1) / max(SnapNum-2, 1);
            omega_t = (1 - alpha) * omega0 + alpha * omega1;
            theta(t+1) = theta(t) + omega_t * TimeInterval;
            pos_rel(:,t+1) = pos_rel(:,t) + [step_len * cos(theta(t));step_len * sin(theta(t));0];
        end

        UEtrack(1,u).positions = pos_rel;
        UEtrack(1,u).no_snapshots = SnapNum;
        pos_abs = (UElocation(1:2,u) + pos_rel(1:2,:)).';   % [SnapNum, 2]
        v_xy_all = zeros(SnapNum, 2);

        for t = 1:SnapNum-1
            v_xy_all(t,:) = (pos_abs(t+1,:) - pos_abs(t,:)) / TimeInterval;
        end
        v_xy_all(SnapNum,:) = v_xy_all(SnapNum-1,:);
        positions{u} = pos_abs;
        velocities{u} = v_xy_all;
    end

    %% layout
    l = qd_layout(s1);
    l.no_tx = 1;
    l.tx_array = BSAntArray;
    l.tx_position = BSlocation;
    l.no_rx = N;
    l.rx_array = UEAntArray;
    l.rx_track = UEtrack;
    l.rx_position = UElocation;
    l.set_scenario(scenario_name);

    %% channel generation over full track
    [BS2UE_channel, ~] = l.get_channels();
    H_track_users = complex(zeros(N, SnapNum, Pol, Mv, Mh, 'single'));

    for u = 1:N
        raw = BS2UE_channel(u).fr(BW, 1);
        h = squeeze(raw);
        if size(h,1) ~= Mtot
            h = h.';
        end

        Tcur = size(h,2);
        if Tcur ~= SnapNum
            error('Bad channel snapshot count.');
        end

        h = reshape(h, [Pol, Mv, Mh, SnapNum]);
        h = permute(h, [4 1 2 3]);   % [SnapNum, Pol, Mv, Mh]
        % Normalize each channel snapshot to unit average coefficient power
        for t = 1:SnapNum
            h_vec = reshape(h(t,:,:,:), [], 1);
            power = sum(abs(h_vec).^2);
            scale = sqrt(length(h_vec) / (power + 1e-12));
            h_vec = h_vec * scale;
            h(t,:,:,:) = reshape(h_vec, [Pol, Mv, Mh]);
        end
        H_track_users(u,:,:,:,:) = complex(single(h));
    end

    %% cut non-overlapping segments and save as scenes
    for seg = 1:SegmentsPerTrack
        if scene_idx >= S
            break;
        end
        scene_idx = scene_idx + 1;
        start_idx = (seg-1) * SegLen + 1;
        mid_idx   = start_idx + hist_len - 1;
        end_idx   = start_idx + SegLen - 1;

        for u = 1:N
            pos_abs = positions{u};
            v_xy_all = velocities{u};
            h_user = squeeze(H_track_users(u,:,:,:,:));   % [SnapNum, Pol, Mv, Mh]
            UEspeed_scene(scene_idx, u) = UESpeeds_ms(u);
            Pxy(scene_idx,:,u,1) = single(pos_abs(start_idx:mid_idx,1));
            Pxy(scene_idx,:,u,2) = single(pos_abs(start_idx:mid_idx,2));
            Vxy(scene_idx,:,u,1) = single(v_xy_all(start_idx:mid_idx,1));
            Vxy(scene_idx,:,u,2) = single(v_xy_all(start_idx:mid_idx,2));
            H_hist(scene_idx,u,:,:,:,:) = h_user(start_idx:mid_idx,:,:,:);
            H_fut(scene_idx,u,:,:,:,:)  = h_user(mid_idx+1:end_idx,:,:,:);
        end
    end

    if scene_idx >= S
        break;
    end
end

H_hist = H_hist(1:scene_idx,:,:,:,:,:);
H_fut  = H_fut(1:scene_idx,:,:,:,:,:);
Pxy    = Pxy(1:scene_idx,:,:,:);
Vxy    = Vxy(1:scene_idx,:,:,:);
UEspeed_scene = UEspeed_scene(1:scene_idx,:);
fprintf('Final number of scenes: %d\n', scene_idx);

%% ================== BUILD VARIABLE NAMES ==================
hist_var_name = sprintf('H_hist_%s', mode);
fut_var_name  = sprintf('H_fut_%s', mode);
pxy_var_name  = sprintf('Pxy_%s', mode);
vxy_var_name  = sprintf('Vxy_%s', mode);
spd_var_name  = sprintf('UEspeed_%s', mode);

eval([hist_var_name ' = H_hist;']);
eval([fut_var_name  ' = H_fut;']);
eval([pxy_var_name  ' = Pxy;']);
eval([vxy_var_name  ' = Vxy;']);
eval([spd_var_name  ' = UEspeed_scene;']);

%% ================== SAVE ==================
if strcmpi(mode, 'test')
    output_dir = fullfile('data', 'UMa', 'test');
else
    output_dir = fullfile('data', 'UMa', 'train_val');
end

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

save(fullfile(output_dir, sprintf('H_hist_%s.mat', mode)),hist_var_name,'-v7.3');
save(fullfile(output_dir, sprintf('H_fut_%s.mat', mode)), fut_var_name,'-v7.3');
save(fullfile(output_dir, sprintf('Pxy_hist_%s.mat', mode)), pxy_var_name, '-v7.3');
save(fullfile(output_dir, sprintf('Vxy_hist_%s.mat', mode)), vxy_var_name,'-v7.3');
save(fullfile(output_dir, sprintf('UEspeed_%s.mat', mode)),spd_var_name,'-v7.3');
fprintf('Dataset saved to: %s\n', output_dir);
disp('UMa dataset generation complete.');