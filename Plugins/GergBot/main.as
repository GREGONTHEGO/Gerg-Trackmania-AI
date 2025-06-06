bool isLogging = false;
Net::Socket@ socket = null;
float sendTimer = 0.0;
const float sendInterval = 1.0 / 10.0;

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
    
    auto app = cast<CTrackMania>(GetApp());
    auto playground = cast<CSmArenaClient>(app.CurrentPlayground);
    if (playground is null) {
        print("Current playground is not a valid arena client.");
        return;
    }

    auto player = cast<CSmPlayer>(playground.GameTerminals[0].GUIPlayer);
    auto controlled = cast<CSmScriptPlayer>(player.ScriptAPI);

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

    string msg = "Speed=" + controlled.Speed + "\n";
    msg += "Position=" + "x:" + controlled.Position.x + " y:" + controlled.Position.y + " z:" + controlled.Position.z + "\n";
    socket.WriteRaw(msg);   
}