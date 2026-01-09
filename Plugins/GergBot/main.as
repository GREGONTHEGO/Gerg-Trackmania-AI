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
    auto LeadinRace = cast<MLFeed::PlayerCpInfo_V4>(raceData.SortedPlayers_Race[0]);
    if (LeadinRace is null) {
        print("LeadinRace is null for local player.");
        return;
    }

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
    float forwardVel = Math::Dot(controlled.Velocity, controlled.AimDirection);
    uint64 ts = Time::Now;
    string msg = ts + ',' + forwardVel + ',' + controlled.Position.x + ',' + controlled.Position.y + ',' + controlled.Position.z + ',' + LeadinRace.cpCount + ',' + controlled.EngineCurGear + ',' + controlled.EngineRpm + ',' + LeadinRace.cpTimes.Length + "\n";

    socket.WriteRaw(msg);   
}