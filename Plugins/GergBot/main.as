bool isLogging = false;
Net::Socket@ socket = null;
float sendInterval = 1.0f/60.0f;
float sendTimer = 0.0f;

void Main() {
    print("GregBot loaded successfully!");
}

void RenderMenu() {
    if (UI::BeginMenu("GregBot")){
        if (UI::MenuItem(isLogging ? "Stop Sending": "Start Sending")) {
            isLogging = !isLogging;
            print(isLogging ? "Started sending car state." : "Stopped sending car state.");
            if (!isLogging && socket !is null) {
                socket.WriteRaw("STOP\n");
                socket.Close();
                @socket = null;
            }
        }
        UI::EndMenu();
    }
}

void Update(float dt) {
    if (!isLogging) {
        return;
    }
    // sendTimer += dt;
    // if (sendTimer < sendInterval) {
    //     return;
    // }
    // sendTimer -= sendInterval;

    auto app = cast<CTrackMania>(GetApp());
    auto playground = cast<CSmArenaClient>(app.CurrentPlayground);
    if (playground is null) {
        print("Current playground is not a valid arena client.");
        return;
    }

    auto raceData = MLFeed::GetRaceData_V4();
    //auto cpCount = MLFeed::HookRaceStatsEventsBase();
    //print(raceData.CPsToFinish);
    // const string localName = raceData.SortedPlayers_Race[0];
    // if (localName == "") {
    //     print("Local player name is empty.");
    //     return;
    // }

    auto LeadinRace = cast<MLFeed::PlayerCpInfo_V4>(raceData.SortedPlayers_Race[0]);
    if (LeadinRace is null) {
        print("LeadinRace is null for local player.");
        return;
    }
    //auto playerInfo = raceData.GetPlayer_V4(localName);
    // if (playerInfo is null) {
    //     print("Player info for local player is null.");
    //     return;
    // }


    auto player = cast<CSmPlayer>(playground.GameTerminals[0].GUIPlayer);
    auto controlled = cast<CSmScriptPlayer>(player.ScriptAPI);
    if (player is null) {
        print("Player is not a valid script player.");
        return;
    }
    if (controlled is null) {
        print("Controlled player is not a valid script player.");
        return;
    }

    if (socket is null){
        @socket = Net::Socket();
        if (!socket.Connect("127.0.0.1", 5055)) {
            print("Failed to connect to the server.");
            @socket = null;
            return;
        }
    }
    //socket.SetNoDelay(true);
    // PlayerState::sTMData@ TMData = PlayerState::GetRaceData();
    // print(TMData.dEventInfo.CheckpointChange);
    // print(PlayerState::GetRaceData().dPlayerInfo.NumberOfCheckpointsPassed);
    // print(player.CurrentLaunchedRespawnLandmarkIndex);
    // print(LeadinRace.cpCount);
    // print(controlled.AccelCoef);
    // print(controlled.DisplaySpeed);
    // print(controlled.Distance);
    // print(controlled.AimDirection);
    //print(LeadinRace.CpTimes);
    float forwardVel = Math::Dot(controlled.Velocity, controlled.AimDirection);
    //print(forwardVel);
    // print(controlled.Speed);
    uint64 ts = Time::Now;
    string msg = ts + ',' + forwardVel + ',' + controlled.Position.x + ',' + controlled.Position.y + ',' + controlled.Position.z + ',' + LeadinRace.cpCount + "\n";
    // string msg = "TS=" + ts + "\n";
    // msg += "Speed=" + controlled.Speed + "\n";
    // msg += "Position=" + "x:" + controlled.Position.x + " y:" + controlled.Position.y + " z:" + controlled.Position.z + "\n";
    // msg += "Velocity=" + "x:" + controlled.Velocity.x + " y:" + controlled.Velocity.y + " z:" + controlled.Velocity.z + "\n";
    // msg += "Checkpoint=" + LeadinRace.cpCount  + "\n";

    socket.WriteRaw(msg);   
}