// Maniaplanet engine classes documentation for 2025-07-04 07:15:08
// Generated with Openplanet 1.28.0 (next, Public, 2025-08-16)
// https://openplanet.dev/

using namespace MwFoundations;
using namespace Game;
using namespace Graphic;
using namespace Function;
using namespace Hms;
using namespace Control;
using namespace Plug;
using namespace Scene;
using namespace System;
using namespace Vision;
using namespace Audio;
using namespace Script;
using namespace Net;
using namespace Input;
using namespace Xml;
using namespace TrackMania;
using namespace ShootMania;
using namespace GameData;
using namespace Meta;
using namespace MetaNotPersistent;

namespace MwFoundations {

struct CMwNod {
  CMwNod();

  string IdName;
  const MwId Id; // Maniascript
};

struct CMwEngine : public CMwNod {
};

struct CMwEngineMain : public CMwEngine {
};

struct CMwCmd : public CMwNod {
};

struct CMwCmdFastCall : public CMwCmd {
};

struct CMwCmdBuffer : public CMwNod {
  CMwCmdBuffer();

  const MwFastBufferCat<CMwCmd*> Cmds;
  const uint CmdCount;
};

struct CMwCmdFiber : public CMwCmd {
};

struct CMwCmdBufferCore : public CMwNod {
  CMwCmdBufferCore();

  const bool IsEnabled;
  const uint ComputerTime;
  const uint HumanTime;
  const uint GameTime;
  const uint SimulationTime;
  const float SimulationRelativeSpeed;
  float PeriodEstimated;
  float DeltaSmoothed;
  float PeriodSmoothing; // Range: 0 - 1
  float DeltaSmoothing; // Range: 0 - 1
  const bool IsLagForbidden;
};

struct CMwClassInfoViewer : public CMwNod {
  CMwClassInfoViewer();

  const uint ClassId;
};

// File extension: 'RefBuffer.Gbx'
struct CMwRefBuffer : public CMwNod {
  CMwRefBuffer();

  const uint Count;
  const MwFastBuffer<CMwNod*> Nods;
  const bool UseAddRefRelease;
  uint NodClassId;
};

struct CMwStatsValue : public CMwNod {
  uint NbMaxSamples;
  bool ComputeDeviatedMean;
  float StdDevRatio;
  bool ComputeMedian;
  uint ReductionRatio;
  bool ComputeBuckets;
  bool AutoBuckets;
  MwFastArray<float> BucketsRanges;
  void Log();
  const uint NbSamples;
  const string Summary;
  const float MeanInv;
  const float Mean;
  const float StdDev;
  const float Min;
  const float Max;
  const float Latest;
  const float Median;
  const float MedianStdDev;
  const float DeviatedMean;
  MwFastArray<float> BucketsRatio;
};

struct GmSimi2 {
  vec2 Pos;
  float Rot;
  vec2 Scale;
};

struct GmMat3 {
  vec3 m_column_0_;
  vec3 m_column_1_;
  vec3 m_column_2_;
};

struct MwTimeNs {
  bool Ns1;
  bool Ns2;
};

struct NatAdr {
};

struct SConstString {
};

struct SConstStringInt {
};

struct TSafePtr {
  const uint PoolIndex;
  const uint BucketIndex;
  const uint ChangeCounter;
  const uint IndexInsideBucket;
  SMetaPtr const Ptr;
};

struct CMwCmdContainer : public CMwNod {
  CMwCmdContainer();

};

struct CMwCmdFastCallUser : public CMwCmd {
};

struct CMwCmdFastCallStatic : public CMwCmd {
};

struct CMwCmdFastCallStaticParam : public CMwCmd {
};

} // namespace MwFoundations

namespace Game {

struct CGameEngine : public CMwEngine {
  CGameEngine();

};

// File extension: 'Title.Gbx'
struct CGameManiaTitle : public CGameNod {
  CGameManiaTitle();

  const string TitleId; // Maniascript
  const string AuthorLogin; // Maniascript
  const wstring AuthorName; // Maniascript
  const wstring Name; // Maniascript
  const wstring Desc; // Maniascript
  const string InfoUrl; // Maniascript
  const string DownloadUrl; // Maniascript
  const string TitleVersion; // Maniascript
  const string MakerTitleId; // Maniascript
  const string BaseTitleId; // Maniascript
  CPlugFileImg* LogosFid;
  string CollectionGroup;
  int SortIndex;
  CGameSkinnedNod* SkinnedBoxCase;
  CGameSkinnedNod* SkinnedLogosStyle;
  string StationManialinkUrl;
  string BoxCaseManialinkUrl;
  bool DisableStationQuickEnter;
  bool ScriptCloud_Enabled;
  CGameManiaTitleCore* TitleCoreFid;
  uint SmGameplayVersion;
  bool InternalAllowLegacyNonScriptModes;
  bool UnlockAdvancedCollectors;
  const MwFastBuffer<CGameCtnCollection*> CollectionFids;
  MwId VehicleIdentId;
  MwId VehicleIdentAuthor;
  MwId VehicleIdentCollection;
  const MwFastBuffer<CPlugFileSnd*> MusicFids;
  const MwFastBuffer<CGameCtnChallenge*> EditorSimpleChallengeFids;
  CPlugFileFidContainer* ModFid;
  string MenuMainManialinkUrl;
  CPlugFileImg* MenuBgFid;
  CSystemFidsFolder* MenuBgFolder;
  CGameCtnReplayRecord* MenuBgReplayFid;
  CPlugFileImg* MenuHeaderFid;
  CPlugFileSnd* MenuMusicFid;
  CGameManialink3dStyle* MenuStyle3dFid;
  CPlugFileImg* MenuSkyGradFid;
  CPlugFont* FallbackFontFid;
  CPlugFont* Hud3dFontFid;
  CPlugFileText* SoloModeFid;
  CPlugFileZip* SoloDataPackFid;
  CGameCtnCampaign* SoloCampaignFid;
  CPlugFileText* SoloPlaylistFid;
  CPlugFileText* NetModeFid;
  CPlugFileText* NetPlaylistFid;
  CPlugFileText* SplitScreenModeFid;
  CPlugFileText* SplitScreenPlaylistFid;
  CPlugFileText* EditorMapTypeFid;
  CPlugFileSnd* EditorMusicFid;
  CGameCtnChallenge* AnimEditor_DefaultChallengeFid;
  uint ModelSkinInit;
};

struct CGamePlayer : public CMwNod {
  CGamePlayerInfo* const User;
};

struct CGameTerminal : public CMwNod {
  CGameTerminal();

  CGamePlayer* ControlledPlayer;
  CGamePlayer* GUIPlayer;
  CGameCtnMediaClipPlayer* const MediaClipPlayer;
  CGameCtnMediaClipPlayer* const MediaAmbianceClipPlayer;
  ESpectatorCameraType SpectatorWantedCameraType;
  ESpectatorTargetMode SpectatorWantedCameraTarget;
  SGamePlaygroundUIConfig::EUISequence UISequence_Current;
  uint UISequence_LatestChangeHumanTime;
  uint8 UISequence_Num;
};

struct CGameApp : public CMwNod {
  CGameApp();

  void Start();
  void TerminateGame();
  CHmsViewport* const Viewport;
  CAudioPort* const AudioPort;
  CInputPort* const InputPort;
  CGameDialogs* const BasicDialogs;
  CGameSystemOverlay* const SystemOverlay;
  const MwFastBuffer<CGameMenu*> ActiveMenus;
  void ShowMenu(CGameMenu* Menu);
  void HideMenu(CGameMenu* Menu);
  void HeavyUpdate();
  const string OSUTCDate;
  const string OSLocalDate;
  const string OSUTCTime;
  const string OSLocalTime;
  const uint TimeSinceInitMs;
  NGameLoadProgress_SMgr* const LoadProgress;
  string CmdLineUrlMptp;
  wstring BenchLoadingFolderName;
  CGameScriptNotificationsConsumer* const NotificationsConsumer;
  CGameScriptNotificationsProducer* const NotificationsProducer;
  CSystemPlatformScript* const SystemPlatform;
};

struct CGameMasterServer : public CNetMasterServer {
  const wstring SubscribeNickName;
  const wstring SubscribeNickNameNew;
  const uint ReturnedError;
  const string ReturnedIP;
  const uint InboxMessagesCount;
  const uint OutboxMessagesCount;
  const MwFastBuffer<CGameRemoteBufferPool*> Pools;
  const MwFastBuffer<CGameRemoteBufferDataInfo*> RemoteDataInfos;
  void CheckDownloadToLaunch();
  const MwFastBuffer<CMwNod*> Downloads;
  const MwFastBuffer<CMwNod*> CurrentDownloads;
  CGameLeagueManager* const LeaguesManager;
  CGameLeagueManager* const SubscribedGroupsManager;
};

struct CGameModuleEditorBase : public CGameManiaApp {
  CGameManialinkScriptHandler* const ScriptHandler;
};

// File extension: ''
struct CGameNod : public CMwNod {
};

// File extension: 'Menu.Gbx'
struct CGameMenu : public CMwNod {
  CGameMenu();

  enum class CGameMenu::EMenuOrder {
    Menu = 2,
    InGameMenu = 3,
    GameDialog = 5,
    SystemDialog = 8,
    BasicDialog = 11,
  };
  MwFastArray<CGameMenuFrame*> Frames;
  CGameMenuFrame* MainFrame;
  CGameMenuFrame* const CurrentFrame;
  CControlBase* const CurrentFocusedControl;
  CControlStyle* const CurrentFocusedControlStyle;
  CPlugSound* SoundDisplayed;
  CPlugSound* SoundHidden;
  CPlugSound* SoundPageChanged;
  CControlStyleSheet* StyleSheet;
  bool EnableFrameStack;
  void Back();
  void RedrawAll();
  vec3 FrustumCenter;
  vec3 FurstrumHfDiag;
  CScene2d* const Overlay;
  CGameMenu::EMenuOrder MenuOrder;
  vec2 GridStep;
};

struct CGameNetFormPlaygroundSync : public CNetNod {
  CGameNetFormPlaygroundSync();

};

// File extension: 'Frame.Gbx'
struct CGameMenuFrame : public CControlFrame {
  CGameMenuFrame();

  MwId AutoBackButtonId;
  CPlugSound* UpDownSound;
  CPlugSound* ShowSound;
  CGameMenu* const Menu;
  CScene* FrameScene;
  bool AllowBgCamera;
  uint NavigationDepth_Manialink3d_Unused;
};

struct CGameSystemOverlay : public CMwNod {
  CScene2d* const Overlay;
  CScene2d* const PluginsMenuOverlayOverlay;
  uint HideDelay;
  void SwitchFullscreen();
  void MinimizeApp();
  void CloseApp();
  void CloseApp_ActionYes();
  void CloseApp_ActionDelay();
  void OpenGraphicSettings();
  void OpenStereoscopySettings();
  void OpenInputSettings();
  void OpenInterfaceSettings();
  float MasterSoundVolume; // Range: -40 - 0
  float MasterMusicVolume; // Range: -40 - 0
  void ThrobberOnAction();
  bool IconManiaPlanet_ShowHome;
  const bool ToolBarIsActive;
  wstring ToolTip;
  const wstring OSLocalTimeOrTimeLeft;
  CGameManiaPlanetMenuStations* const MenuStations;
  CGameManialinkBrowser* const ManialinkBrowser;
  CGameScriptDebugger* const ScriptDebugger;
};

struct CGamePlayground : public CGameSwitcherModule {
  CGamePlaygroundInterface* const Interface; // Maniascript
  const MwFastBuffer<CGamePlayer*> Players;
  const MwFastBuffer<CGameTerminal*> GameTerminals;
  const MwFastBuffer<CGamePlaygroundUIConfig*> UIConfigs;
  CGameCtnReplayRecord* const ReplayRecord;
  CGameCtnReplayRecord* PrevReplayRecord;
  bool GameTerminals_IsBlackOut;
  bool GameTerminals_BlackOut_Auto2dVr;
  CGameAnalyzer* Analyzer;
};

struct CGameNetPlayerInfo : public CMwNod {
  enum class CGameNetPlayerInfo::EPlayerType {
    Human = 0,
    Fake = 1, // Fake (not net)
    Net = 2,
    Replay = 3,
  };
  enum class CGameNetPlayerInfo::ESpectatorMode {
    Void = 0,
    Watcher = 1,
    LocalWatcher = 2,
    Target = 3,
  };
  const uint DbgClientUId;
  bool Live_IsRegisteredToMasterServer;
  bool Live_HasRetrieveTimeLeft;
  const bool Live_Updating;
  const bool Live_RetrievingTimeLeft;
  const uint Live_UpdateLastTime;
  const uint LiveUpdate_Counter;
  const CGameNetPlayerInfo::EPlayerType PlayerType;
  const uint State;
  const uint ChallengeSequenceNumber;
  const uint LatestNetUpdate;
  const uint DownloadRate;
  const uint UploadRate;
  bool CustomDataDeactivated;
  bool EnableHomologation;
  const uint NbSpectators;
  CGameNetPlayerInfo::ESpectatorMode SpectatorMode;
};

struct CGameNetwork : public CMwNod {
  CGameNetServerInfo* ServerInfo;
  CNetMasterServer* const MasterServer;
  CNetFileTransfer* const FileTransfer;
  const MwFastBuffer<CSystemPackDesc*> PackDescs;
  CSystemFidsFolder* const LocationPacksContent;
  const bool IsMultiInternet;
  const bool IsEnabled;
  const bool IsServer;
  CNetServer* const Server;
  CNetClient* const Client;
  const MwFastBuffer<CGameNetServerInfo*> OnlineServers;
  const MwFastBuffer<CGameNetServerInfo*> OnlinePlayers;
  const MwSArray<CGameNetPlayerInfo*> PlayerInfos;
  void FindServers();
  CGamePlaygroundUIConfigMgrScript* const UIConfigMgr_Rules;
  CGamePlaygroundUIConfigMgrScript* const UIConfigMgr_ServerPlugin;
  float VoteDefaultRatio;
  uint CallVoteTimeOut;
  uint CallVotePercent;
  const bool InCallvote;
  const uint VoteNbYes;
  const uint VoteNbNo;
  const MwFastBuffer<CSystemData*> ManialinkDataCache;
  const uint RecvNetRate;
  const uint SendNetRate;
  const float PacketLossRate;
  const string RecvNetRatePretty;
  const string SendNetRatePretty;
  const uint CanDoUDP;
  const int LatestGamePing;
  const int LatestHumanPing;
  const float LatestEpsilon;
  const float SmoothedEpsilon;
  const uint TotalSendingSize;
  const uint TotalReceivingSize;
  const uint TotalHttpReceivingSize;
  const uint TotalTcpUdpReceivingSize;
  const uint TcpReceivingSize;
  const uint UdpReceivingSize;
  const uint TcpSendingSize;
  const uint UdpSendingSize;
  const uint NbrTotalConnection;
  const uint NbrConnectionsDone;
  const uint NbrConnectionsDisconnecting;
  const uint NbrConnectionsInProgress;
  const uint NbrConnectionsPending;
  const uint NbrAcceptPerSecond;
  const uint NbrNewConnectionPerSecond;
};

struct CGameNetFormTunnel : public CGameNetForm {
  CGameNetFormTunnel();

};

struct CGameManiaPlanetNetwork : public CGameCtnNetwork {
};

struct CGameManiaTitleCore : public CGameNod {
  const wstring DisplayName;
  CGameManiaPlanet* const ManiaPlanet;
  CGameCtnCollection* CollectionCommonFid;
  wstring ModeScriptFolderName;
  bool WithLocalMultiplayer;
};

// File extension: 'ManiaPlanet.Gbx'
struct CGameManiaPlanet : public CGameCtnApp {
  CGameManiaPlanet();

  CSystemFidsFolder* ManiaTitlesFolder;
  CGameCoverFlowDesc* CoverFlowDesc;
  CPlugFileImg* TitleMainDefaultBgFid;
  void ScanDiskForManiaTitles();
  const MwFastBuffer<CGameManiaTitle*> ManiaTitles;
  const MwFastBuffer<CGameStation*> Stations;
  CGameManiaTitle* const LoadedManiaTitle;
  CGameManiaTitleCore* const LoadedCore;
  CGameManiaPlanetScriptAPI* const ManiaPlanetScriptAPI;
  CGameManiaTitleControlScriptAPI* const ManiaTitleControlScriptAPI;
  CGameManiaTitleEditionScriptAPI* const ManiaTitleEditionScriptAPI;
  string StartupManiaTitle;
  string StartupManiaTitle_SingleTitleMode;
  string StartupManiaTitle_TestMode;
  void Operation_Abort();
  const bool Operation_InProgress;
  CGameCtnMenusManiaPlanet* MenuManager;
  void BackToMainMenu();
  bool DisableReplayRecording;
};

// Description: "Station info & manipulation."
struct CGameStation : public CGameNod {
  enum class CGameStation::EEchelon {
    None = 0,
    Bronze1 = 1,
    Bronze2 = 2,
    Bronze3 = 3,
    Silver1 = 4,
    Silver2 = 5,
    Silver3 = 6,
    Gold1 = 7,
    Gold2 = 8,
    Gold3 = 9,
  };
  CGameManiaTitle* const Title; // Maniascript
  const uint AudienceRegisteredUsers; // Maniascript
  const uint AudienceInstalls;
  const uint AudienceOnlineUsers;
  const uint CampaignMedalsMax; // Maniascript
  const uint CampaignMedalsCurrent; // Maniascript
  const uint CampaignMedalsRanking; // Maniascript
  const float LadderPoints; // Maniascript
  const uint LadderRank; // Maniascript
  const CGameStation::EEchelon Echelon; // Maniascript
  const uint NextEchelonPercent; // Maniascript
  bool DisableQuickEnter; // Maniascript
  bool IsLogoVisible; // Maniascript
  float GhostAlpha; // Maniascript
  vec3 FocusLightColor; // Maniascript
  vec3 NormalLightColor; // Maniascript
  const bool IsEditable; // Maniascript
  const string ManiaApp_Url;
  CGameManialinkPage* const ManialinkViewer_Page;
  CGameManiaApp* const ManiaApp;
};

// Description: "Manialink entry."
struct CGameManialinkEntry : public CGameManialinkControl {
  enum class CGameManialinkEntry::ETextFormat {
    Basic = 0,
    Script = 1,
    Password = 2,
    Newpassword = 3,
  };
  enum class CGameManialinkEntry::EValueType {
    Ml_Unknown = 0,
    Ml_Natural = 1,
    Ml_Integer = 2,
    Ml_Real = 3,
    Ml_String = 4,
    Ml_TimeMmSsCc = 5,
    Ml_TimeHhMmSs = 6,
    Ml_RealFormated = 7,
    Ml_TimeMmSs = 8,
    Ml_Ascii7bit = 9,
    Ml_Real3Decimals = 10,
    Ml_TimeHhMmSs_24 = 11,
    Ml_TimeHhMm = 12,
    Ml_Percent = 13,
    Ml_Hexa = 14,
    Ml_TimeHhMmSsOrMmSs = 15,
    Ml_TimeHhMmFromSeconds = 16,
    Ml_TimeHhMmSsMil = 17,
  };
  wstring Value; // Maniascript
  wstring HackValueWithEvent;
  void StartEdition(); // Maniascript
  void SetText(wstring NewText, bool SendSubmitEvent); // Maniascript
  CGameManialinkEntry::ETextFormat TextFormat; // Maniascript
  float Opacity; // Maniascript
  vec3 TextColor; // Maniascript
  float TextSizeReal; // Maniascript
  uint TextSize; // Maniascript
  int MaxLength; // Maniascript
  int MaxLine; // Maniascript
  bool AutoNewLine; // Maniascript
  CGameManialinkEntry::EValueType GetValueType(); // Maniascript
  void SetValueType(CGameManialinkEntry::EValueType ValueType); // Maniascript
};

// File extension: 'ColorEffect.Gbx'
struct CGameMenuColorEffect : public CControlEffect {
  CGameMenuColorEffect();

  MwFastArray<CFuncFullColorGradient*> ColorsBeam;
  MwFastArray<float> ColorPeriods;
  MwFastArray<float> BeamWidths;
  MwFastArray<float> InterWidths;
  float Period;
  bool IsBeamColorEvolving;
  bool IsBeamMoving;
  bool IsMoveHalf;
  bool IsMoveInverse;
  bool IsColorEvolveHalf;
  bool IsColorEvolveInverse;
  bool ForceFirstColorWord;
  bool ForceNoMove;
};

struct CGameCtnCollectorList : public CMwNod {
  CGameCtnCollectorList();

  const uint Bob;
  const MwFastBuffer<MwId> BlockStockIds;
  MwFastBuffer<uint> BlockStockNbAvailables;
};

struct CGameCtnChapter : public CMwNod {
  CGameCtnChapter();

  CSystemFidFile* const CollectionFid;
  const wstring LongDesc;
  const MwFastBuffer<CGameCtnArticle*> Articles;
  CSystemFidFile* const Icon;
  const bool Unlocked;
};

struct CGameCtnCatalog : public CMwNod {
  CGameCtnCatalog();

  const MwFastBuffer<CGameCtnChapter*> Chapters;
};

struct CGameCtnArticle : public CMwNod {
  CGameCtnArticle();

  const wstring Name;
  const wstring DisplayName;
  const wstring NameOrDisplayName;
  CSystemFidFile* const CollectorFid;
  const bool IsLoaded;
  void Preload();
  void Purge();
  CGameCtnArticle* const BlockItem_ItemModelArticle;
  CMwNod* const LoadedNod;
  const MwId IdentId;
  UnnamedEnum CollectionId;
  wstring CollectionId_Text;
  const MwId IdentAuthor;
  CPlugBitmap* const BitmapIcon;
  CPlugGameSkin* const GameSkin;
  uint NbAvailableCurrent;
  uint NbAvailableMax;
  UnnamedEnum ArticleDataLocation;
  const wstring PageName;
  const string AmountString;
};

struct CGameLaunchedCheckpoint : public CMwNod {
  CGameLaunchedCheckpoint();

};

struct CGameCtnPlayground : public CGamePlayground {
  ISceneVis* const GameScene;
  CGameCtnGhost* const PlayerRecordedGhost;
  CGameCtnGhost* const PlayerBestGhost;
};

struct CWebServicesTask_PostConnect_PlugInList : public CWebServicesTaskSequence {
};

// File extension: 'GameCtnMediaBlock3dStereo.gbx'
struct CGameCtnMediaBlock3dStereo : public CGameCtnMediaBlock {
  CGameCtnMediaBlock3dStereo();

  float Separation;
  float ScreenDist;
};

struct CGameMasterServerRequest : public CNetMasterServerRequest {
};

struct CGameAvatar : public CMwNod {
  CSystemPackDesc* const PackDesc;
};

struct CGameNetOnlineMessage : public CMwNod {
  CGameNetOnlineMessage();

  string ReceiverLogin;
  string SenderLogin;
  wstring Subject;
  wstring Message;
  uint Donation;
};

// File extension: 'GameCtnMediaBlockTriangles.gbx'
struct CGameCtnMediaBlockTriangles : public CGameCtnMediaBlock {
  CSceneMobil* const Mobil;
};

struct CGameRemoteBuffer : public CMwNod {
  CMwNod* const LocalData;
  const string LastServerUpdate;
  const bool UseRefs;
  enum class CGameRemoteBuffer::EMode {
    None = 0,
    Get = 1,
    Set = 2,
    Get_Set = 3, // Get/Set
  };
  const CGameRemoteBuffer::EMode Mode;
  const uint TotalCount;
  const uint SpecificCount;
  const uint PerPageCount;
  const uint CacheDuration;
  const uint RegisteredUsersCount;
  MwFastBuffer<CMwNod*> Datas;
};

struct CGameRemoteBufferPool : public CMwNod {
  CGameRemoteBufferDataInfo* const DataInfo;
  const uint BuffersCount;
};

struct CGameRemoteBufferDataInfo : public CMwNod {
  uint Default_CacheDuration;
  uint Default_PerPageCount;
};

// File extension: 'GameResources.Gbx'
struct CGameResources : public CMwNod {
  CGameResources();

  MwFastArray<CPlugSound*> Sounds;
  const MwFastArray<CAudioSource*> AudioSources;
  CSystemFidsFolder* HymnsFolder;
  CSystemFidsFolder* MenuBackgroundsFolder;
  CSystemFidsFolder* FolderGlobalEnvBanners;
  CMwNod* EditorSpawnHelper;
  CPlugBitmap* BitmapCubeMenuManiaPlanet3d;
  CPlugMaterial* MaterialSkyDomeMenu;
  CPlugShader* InterfaceBg;
  CControlStyleSheet* StyleSheetFid;
  CControlStyle* StyleUrlLink;
  CControlStyle* StyleDefaultTitleLogos;
  CPlugFont* InterfaceFont;
  CScene2d* InterfaceEditorFid;
  CScene2d* InterfaceEditorSimpleFid;
  CControlFrame* InterfaceEditorInventoryModelFid;
  CScene2d* StoryInterfaceFid;
  CControlFrame* EditorAnimInterfaceFid;
  CControlFrame* EditorAnimCharInterfaceFid;
  CControlFrame* EditorItemInterfaceFid;
  CControlFrame* EditorVehicleInterfaceFid;
  CControlFrame* EditorClassInterfaceFid;
  CControlFrame* EditorManialinkInterfaceFid;
  CControlFrame* EditorModuleInterfaceFid;
  CControlFrame* EditorAnimSetInterfaceFid;
  CPlugBitmap* EditorMeshGradientVBitmapFid;
  CPlugFileTextScript* ModuleMenuBaseScriptFid;
  CPlugFileTextScript* ModuleMenuServerBrowserScriptFid;
  CPlugFileTextScript* ModulePlaygroundStoreScriptFid;
  CPlugFileTextScript* ModulePlaygroundInventoryScriptFid;
  CPlugFileTextScript* ModulePlaygroundSpeedMeterScriptFid;
  CGameModulePlaygroundHudModel* ModulePlaygroundDefaultHudFid;
  CGameModulePlaygroundInventoryModel* ModulePlaygroundDefaultInventoryFid;
  CGameModulePlaygroundStoreModel* ModulePlaygroundDefaultStoreFid;
  CGameModulePlaygroundChronoModel* ModulePlaygroundDefaultChronoFid;
  CGameModulePlaygroundScoresTableModel* ModulePlaygroundDefaultScoresTableFid;
  CGameModulePlaygroundSpeedMeterModel* ModulePlaygroundDefaultSpeedMeterFid;
  CGameModulePlaygroundPlayerStateModel* ModulePlaygroundDefaultPlayerStateFid;
  CGameModulePlaygroundTeamStateModel* ModulePlaygroundDefaultTeamStateFid;
  CGameModulePlaygroundPlayerStateModel* ModulePlaygroundFullPlayerStateFid;
  CPlugFileText* MaterialsLibrariesFid;
  CSystemFidsFolder* EditorMeshMaterialFolder;
  CGameMenu* MenusFid;
  CGameMenu* InGameDialogsFid;
  CGameMenu* DialogsFid;
  CGameMenu* SystemDialogsFid;
  CGameMenu* BasicDialogsFid;
  CScene2d* ProgressOverlayFid;
  CScene2d* SystemOverlayFid;
  CScene2d* SystemMenuFid;
  CScene2d* ManialinkBrowserOverlayFid;
  CScene2d* PluginsMenuOverlayFid;
  CPlugSolid* StationGoldFid;
  CPlugSolid* StationSilverFid;
  CPlugSolid* StationBronzeFid;
  CPlugSolid* StationServerFid;
  CPlugGameSkin* StationGameSkin;
  CPlugSolid* PlanetSolidFid;
  CPlugSound* ManiaPlanetMenuMusic;
  CControlFrame* CardStationMainFid;
  CControlFrame* CardStationSummaryFid;
  CControlFrame* CardPlanetMenu1Fid;
  CControlFrame* CardPlanetMenu2Fid;
  CControlFrame* CardPlanetServerMainFid;
  CControlFrame* CardPlanetServerSummaryFid;
  CControlFrame* CardPlanetServerRackSummaryFid;
  CGameManialink3dWorld* MenuManialink3dFid;
  CGameManialink3dStyle* MenuManialink3d_BaseStyleFid;
  CGameManialink3dStyle* MenuManialink3d_StationsStyleFid;
  CPlugSolid2Model* MenuManialink3dBox_Box;
  CPlugSolid2Model* MenuManialink3dBox_Title;
  CPlugSolid2Model* MenuManialink3dBox_Button;
  CPlugSolid2Model* MenuManialink3dBox_Window;
  CPlugSolid2Model* MenuManialink3dBox_WindowH;
  CPlugSolid2Model* MenuManialink3dBox_BoxCase;
  CPlugSolid2Model* MenuManialink3dBox_Station1x1;
  CPlugSolid2Model* MenuManialink3dBox_Station2x2;
  CPlugSolid2Model* MenuManialink3dBox_Station3x3;
  CSystemFidsFolder* TitleTemplateFolder;
  CPlugBitmap* DefaultAvatarBitmapFid;
  CPlugFileImg* DefaultMTColorGradingImageFid;
  CPlugBitmap* DefaultMTDecal2dBitmapFid;
  CPlugBitmap* DefaultMTBlockBitmapFid;
  CPlugSound* DefaultMTBlockSoundFid;
  CPlugBitmap* DefaultLeagueLogoBitmapFid;
  CPlugBitmap* DefaultOnlineNewsIconBitmapFid;
  CPlugBitmap* DefaultTeamLogoBitmapFid;
  CPlugBitmap* DefaultTagBitmapFid;
  CPlugMusic* DefaultMusicFid;
  CPlugBitmap* BitmapTeamEmblems;
  CPlugFileSnd* EmptyChallengeCustomMusicFid;
  CGameCtnPainterSetting* PainterSetting;
  CGameControlCardManager* GeneralCardManager;
  CSceneFxNod* SceneFxNodRoot;
  CPlugFileTextScript* DefaultSkillScoreComputerFid;
  CPlugFileImg* ImageTurboRoulette;
  CPlugBitmap* EditorModelSpritesBitmapFid;
  CPlugMaterial* EditorModelDefaultMaterialFid;
  CPlugFileImg* EditorPixelCheckerImageFid;
  CGameCtnMediaClip* DefaultTransitionInClipFid;
  CGameCtnMediaClip* DefaultTransitionOutClipFid;
  CPlugAudioBalance* AudioBalance_Base;
  CPlugAudioBalance* AudioBalance_Menus;
  CPlugAudioBalance* AudioBalance_Editor;
  CPlugShaderApply* OffZoneSurfaceShaderFid;
  CPlugShaderApply* OffZoneLayerShaderFid;
  CGameEditorModel* DefaultEditorMeshFid;
  CGameEditorModel* DefaultEditorEditorFid;
  CGameEditorModel* DefaultEditorMediaTrackerFid;
  CGameEditorModel* DefaultEditorSkinFid;
  CGameEditorModel* DefaultEditorVehicleFid;
  CGameEditorModel* DefaultEditorModuleFid;
  CSystemFidsFolder* AdnFolder;
  CSystemFidsFolder* TrafficFolder;
  CSceneLayout* IconShooterSceneLayoutFid;
  CGameCtnDecorationMood* IconShooterMoodFid;
  CGameShootIconSetting* IconShooterDefaultSettingsFid;
  CSystemFidFile* BadWordDict;
};

struct CGameNetServerInfo : public CNetMasterHost {
  const bool IsIdle_ForMasterServer;
  const bool IsOnline;
  const bool IsServer;
  const bool IsPrivate;
  const bool IsPrivateForSpectator;
  const uint LadderMatchId;
  const uint DownloadRate;
  const uint UploadRate;
  enum class CGameNetServerInfo::EPingEnum {
    ____ = 0, //     
    A___ = 1, // *   
    AA__ = 2, // **  
    AAA_ = 3, // *** 
    AAAA = 4, // ****
  };
  const CGameNetServerInfo::EPingEnum PingEnum;
  const uint Ping;
  string ServerHostName;
  const uint State;
  const uint ChallengeSequenceNumber;
  const bool DetailedPingInfoReceived;
  const uint RoundTrip;
  bool CallVoteEnabled;
  string AdvertisingSuffix;
};

struct CGameNetForm : public CNetNod {
  CGameNetForm();

};

struct CGameDialogs : public CMwNod {
  enum class CGameDialogs::EDialog {
    None = 0,
    Message = 1,
    WaitMessage = 2,
  };
  const bool EnableCustomSystemDialogs; // Maniascript
  const CGameDialogs::EDialog Dialog; // Maniascript
  const wstring Message_LabelText; // Maniascript
  const wstring Message_ButtonText; // Maniascript
  const bool Message_FocusOnButton; // Maniascript
  void Message_Ok(); // Maniascript
  const wstring WaitMessage_LabelText; // Maniascript
  const wstring WaitMessage_ButtonText; // Maniascript
  const float WaitMessage_Progress; // Range: 0 - 1 // Maniascript
  const bool WaitMessage_ShowProgressBar; // Maniascript
  const bool WaitMessage_ShowAbortButton; // Maniascript
  void WaitMessage_Ok(); // Maniascript
  CGameMenu* Dialogs;
  void HideDialogs();
  void DoMessage_Ok();
  void AskYesNo_No();
  void AskYesNo_Yes();
  void AskYesNo_Cancel();
  wstring String;
  void DialogSaveAs_HierarchyUp();
  const wstring DialogSaveAs_Path;
  const wstring DialogSaveAs_PathToDisplay;
  void DialogSaveAs_OnRefresh();
  void DialogSaveAs_OnValidate();
  void DialogSaveAs_OnCancel();
  const MwFastBuffer<CGameFid*> DialogSaveAs_Files;
  void DbgTestDoMessage();
};

// File extension: 'ScaleEffect.Gbx'
struct CGameMenuScaleEffect : public CControlEffect {
  CGameMenuScaleEffect();

  float LeftBorderScale;
  float RightBorderScale;
  float UpBorderScale;
  float DownBorderScale;
  float Shift;
  uint Period;
  uint MaxLetterScaling;
  bool AllowDecalage;
  bool AllowHideBeforeEffect;
  bool IsHalf;
  bool IsInverse;
  bool ReCenter;
};

// File extension: 'Collection.Gbx'
struct CGameCtnCollection : public CMwNod {
  CGameCtnCollection();

  UnnamedEnum CollectionId;
  wstring CollectionId_Text;
  MwFastBuffer<CGameCtnZone*> CompleteZoneList;
  wstring DisplayName;
  CGameCtnZone* DefaultZone;
  float SquareSize;
  float SquareHeight;
  float BoardSquareHeight;
  float BoardSquareBorder;
  float TerrainHeightOffset;
  const string WaterG;
  CPlugBitmap* WaterG_BitmapNormal;
  float WaterG_BumpSpeedUV;
  float WaterG_BumpScaleUV;
  float WaterG_BumpScale;
  float WaterG_RefracPertub;
  bool WaterG_FogUseRefractRay;
  CPlugMaterialWaterArray* WaterArray;
  NPlugMaterial_SWater const Water0;
  bool IsWaterMultiHeight;
  float WaterFogClampAboveDist;
  float CameraMinHeight;
  bool NeedUnlock;
  UnnamedEnum BackgroundShadow;
  float ShadowSoftSizeInWorld;
  UnnamedEnum VertexLighting;
  float ColorVertexMin; // Range: 0 - 1
  float ColorVertexMax; // Range: 0 - 1
  UnnamedEnum LightMapMapper;
  float VisMeshLodDistScale;
  vec3 Tech3TunnelSpecularExpScaleMax;
  CGameCtnDecoration* DefaultDecoration;
  MwId VehicleName;
  MwId VehicleAuthor;
  UnnamedEnum VehicleCollection;
  wstring VehicleCollection_Text;
  MwId VehicleTransform_CarSnow_Name;
  MwId VehicleTransform_CarSnow_Author;
  UnnamedEnum VehicleTransform_CarSnow_Collection;
  wstring VehicleTransform_CarSnow_Collection_Text;
  MwId VehicleTransform_CarRally_Name;
  MwId VehicleTransform_CarRally_Author;
  UnnamedEnum VehicleTransform_CarRally_Collection;
  wstring VehicleTransform_CarRally_Collection_Text;
  MwId VehicleTransform_CarDesert_Name;
  MwId VehicleTransform_CarDesert_Author;
  UnnamedEnum VehicleTransform_CarDesert_Collection;
  wstring VehicleTransform_CarDesert_Collection_Text;
  bool IsEditable;
  CPlugBitmap* IconFid;
  CPlugBitmap* LoadScreenFid;
  int SortIndex;
  uint DecalFade_cBlock_FullDensity;
  CSystemFidsFolder* FolderMenusIcons;
  CSystemFidsFolder* FolderDecoration;
  CSystemFidsFolder* FolderBlockInfo;
  CSystemFidsFolder* FolderItem;
  CSystemFidsFolder* FolderAdditionalItem1;
  CSystemFidsFolder* FolderAdditionalItem2;
  CSystemFidsFolder* FolderAdditionalItem3;
  CSystemFidsFolder* FolderMacroblock;
  CSystemFidsFolder* FolderDecalModels;
  CSystemFidsFolder* FolderDefaultActions;
  CSystemFidsFolder* FolderMacroDecals;
  CSystemFidsFolder* FolderSpectators;
  CSystemFidsFolder* FolderMaterialModifiers;
  CSystemDependenciesList* AdditionalDepsForUGC;
  MwFastBuffer<wstring> BaseZoneStrings;
  MwFastBuffer<wstring> ReplacementZoneStrings;
  MwFastBuffer<CPlugGameSkinAndFolder*> ReplacementTerrainModifiers;
  void AddReplacementZone();
  void RemoveReplacementZone();
  CPlugGameSkinAndFolder* ColorBlindnessModifier;
  const bool ColorBlindnessEnabled;
  const MwFastBuffer<CPlugParticleEmitterModel*> ParticleEmitterModelsFids;
  MwFastBuffer<MwId> DecalsTypesId;
  CSceneVehicleCarMarksModel* MarksModel;
  CFuncShaderLayerUV* FidFuncShaderCloudsX2;
  CPlugBitmap* FidPlugBitmapCloudsX2;
  CPlugMediaClipList* DefaultSpawnClipList;
  CPlugBitmap* VehicleEnvLayer_FidBitmap;
  UnnamedEnum VehicleEnvLayer;
  CPlugFogMatter* OffZone_FogMatter;
  const MwFastBuffer<CGameCtnZone*> ZoneList;
  CPlugVehicleVisStyles* VehicleStyles;
  CMwNod* ItemPlacementGroups;
  CPlugAdnRandomGenList* AdnRandomGenList;
  CGameBlockInfoGroups* FidBlockInfoGroups;
  CGameBlockInfoTreeRoot* FidBlockInfoInventory;
  CGameItemModelTreeRoot* FidItemModelInventory;
  CGameItemModelTreeRoot* FidMacroBlockInfoInventory;
  NGameCollection_SCustomizableDeco CustomDeco;
  RGBAColor TurboColor_Turbo;
  RGBAColor TurboColor_Turbo2;
  RGBAColor TurboColor_Roulette1;
  RGBAColor TurboColor_Roulette2;
  RGBAColor TurboColor_Roulette3;
  CPlugBitmap* BitmapDisplayControlDefaultTVProgram_64x10A;
  CPlugBitmap* BitmapDisplayControlDefaultTVProgram_64x10B;
  CPlugBitmap* BitmapDisplayControlDefaultTVProgram_64x10C;
  CPlugBitmap* BitmapDisplayControlDefaultTVProgram_2x3;
  CPlugBitmap* BitmapDisplayControlDefaultTVProgram_155;
  CPlugMaterialColorTargetTable* FidColorTargetTable;
};

struct CGameCtnMediaBlockEditor : public CMwNod {
};

// Description: "A 1-square-sized part of a block model."
struct CGameCtnBlockUnitInfo : public CMwNod {
  CGameCtnBlockUnitInfo();

  enum class CGameCtnBlockUnitInfo::ECardinalDirEnum {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
  };
  nat3 OffsetE;
  uint ClipCount_North;
  uint ClipCount_East;
  uint ClipCount_South;
  uint ClipCount_West;
  uint ClipCount_Top;
  uint ClipCount_Bottom;
  MwFastBuffer<CGameCtnBlockInfoClip*> AllClips;
  MwFastBuffer<CGameCtnBlockInfoClip*> Clips_North;
  MwFastBuffer<CGameCtnBlockInfoClip*> Clips_East;
  MwFastBuffer<CGameCtnBlockInfoClip*> Clips_South;
  MwFastBuffer<CGameCtnBlockInfoClip*> Clips_West;
  MwFastBuffer<CGameCtnBlockInfoClip*> Clips_Top;
  MwFastBuffer<CGameCtnBlockInfoClip*> Clips_Bottom;
  MwFastBuffer<CGameCtnBlockUnitInfo::ECardinalDirEnum> TopClipDirs;
  MwFastBuffer<CGameCtnBlockUnitInfo::ECardinalDirEnum> BottomClipDirs;
  MwFastBuffer<uint> ClipBakedMobilIndexesForPreview;
  void Clear_ClipBakedMobilIndexesForPreview();
  void Init_ClipBakedMobilIndexesForPreview();
  MwFastArray<CGameCtnBlockInfoPylon*> Pylons;
  uint AcceptPylons;
  uint PlacePylons;
  bool Underground;
  MwId TerrainModifierId;
  CGameCtnBlockUnitInfo* OriginalBlockUnitInfoCopy;
  const int PlacedPillarIndex;
  const int ReplacedPillarIndex;
  const nat3 Offset; // Maniascript
  const nat3 RelativeOffset; // Maniascript
  const MwFastArray<CGameCtnBlockInfoClip*> Clips; // Maniascript
};

struct CGameFid : public CMwNod {
  const wstring Name; // Maniascript
  const wstring Path; // Maniascript
  const wstring FileName; // Maniascript
  CSystemFidFile* const Fid;
  CSystemFidsFolder* const Fids;
  const bool Selected;
};

// File extension: 'Decoration.Gbx'
struct CGameCtnDecoration : public CGameCtnCollector {
  CGameCtnDecoration();

  CPlugFileFidContainer* DefaultSkinFid;
  DataRef DefaultSkinFileRef;
  CGameCtnDecorationSize* DecoSize;
  CGameCtnDecorationAudio* DecoAudio;
  CGameCtnDecorationMood* DecoMood;
  CGameCtnDecorationMaterialModifiers* DecoMaterialModifiers;
  CPlugGameSkinAndFolder* TerrainModifierBase;
  CPlugGameSkinAndFolder* TerrainModifierCovered;
  bool IsWaterOutsidePlayField;
  CPlugGameSkin* VehicleFxSkin;
  CSystemFidsFolder* VehicleFxFolder;
  CPlugSound* DecoAudio_Ambient;
  MwId DecorationZoneFrontierId;
  CGameCtnChallenge* DecoMap;
  CPlugFileZip* DecoMapLightmapFid;
  void InitWithNoSkin();
  void ClearWithNoSkin();
  CPlugDecoratorSolid* DecoratorSolidWarp;
};

// File extension: 'DecorationAudio.Gbx'
struct CGameCtnDecorationAudio : public CMwNod {
  CGameCtnDecorationAudio();

  MwFastArray<CPlugSound*> Sounds;
  MwFastArray<CPlugSound*> Musics;
  float CameraWooshVolumedB; // Range: -60 - 0
  float CameraWooshMinSpeedKmh;
  CPlugSound* DefaultWooshDecoItems;
  CPlugAudioEnvironment* AudioEnvOutsideOpen;
  CPlugAudioEnvironment* AudioEnvOutsideEnclosed;
  CPlugAudioEnvironment* AudioEnvUndergroundOpen;
  CPlugAudioEnvironment* AudioEnvUndergroundEnclosed;
  CPlugAudioEnvironment* AudioEnvUnderwater;
  CPlugAudioBalance* AudioBalance_Podium;
  CPlugAudioBalance* AudioBalance_ReplaySoft;
  CPlugAudioBalance* AudioBalance_ReplayLoud;
  CPlugAudioBalance* AudioBalance_PlaygroundSoft;
  CPlugAudioBalance* AudioBalance_PlaygroundLoud;
  CPlugAudioBalance* AudioBalance_Overlay_Underground;
  CPlugAudioBalance* AudioBalance_Overlay_Far;
  CPlugAudioBalance* AudioBalance_TM_EvtStartLine;
  CPlugAudioBalance* AudioBalance_TM_EvtCheckpoint;
  CPlugAudioBalance* AudioBalance_TM_EvtRespawn;
  CPlugAudioBalance* AudioBalance_TM_EvtCrash;
  CPlugAudioBalance* AudioBalance_TM_EvtFlying;
  CPlugAudioBalance* AudioBalance_TM_EvtSwimming;
  CPlugAudioBalance* AudioBalance_SM_EvtSpawn;
  CPlugAudioBalance* AudioBalance_SM_EvtUnspawn;
  CPlugAudioBalance* AudioBalance_SM_EvtHit;
  CPlugAudioBalance* AudioBalance_SM_EvtHitEliminated;
  CPlugAudioBalance* AudioBalance_SM_EvtFire;
  CPlugAudioBalance* AudioBalance_SM_EvtBulletTime;
  CPlugAudioBalance* AudioBalance_SM_EvtBoost;
  float ReverbMinBlockDist;
  float ReverbMaxBlockDist;
  MwFastBuffer<float> ReverbMaterialGains;
  CPlugFileText* ModifierXmlFile;
};

// File extension: 'DecorationMood.Gbx'
struct CGameCtnDecorationMood : public CMwNod {
  CGameCtnDecorationMood();

  uint ShadowCountCarHuman;
  uint ShadowCountCarOpponent;
  float ShadowCarIntensity; // Range: 0 - 1
  bool ShadowScene;
  bool BackgroundIsLocallyLighted;
  bool SolidLightAreSkinned;
  float Latitude;
  float Longitude;
  float DeltaGMT;
  float SunMoonIntensity;
  float LocalLightScale;
  float EditorHelperHdrScale;
  string TimeSunRise;
  string TimeSunFall;
  bool EnableStars;
  bool WaterReflectFakeCube;
  CPlugCloudsSolids* CloudsSolids;
  const string ToneMapping;
  float ToneMapExposureStaticBase;
  UnnamedEnum ToneMapFilmCurve;
  vec3 Tech3Bloom;
  vec4 Tech3ToneMapAutoExp;
  vec2 Tech3SpecularLocal;
  vec3 Tech3SpecularFake_ExpScaleMax;
  CPlugCurveSimpleNod* ToneMapAutoExp_FidAvgLumiToKeyValue;
  CPlugCurveSimpleNod* FxBloom_FidFuncIntensAtHdrNorm;
  CPlugFxLightning* FxLightning;
  CPlugFxWindOnDecal* FxWindOnDecal;
  CPlugFxWindOnTreeSprite* FxWindOnTreeSprite;
  CPlugFxHdrScales_Tech3* FxHdrScalesT3;
  CPlugMoodBlender* MoodBlender;
  float RemappedStartDayTime; // Range: 0 - 1
  bool IsNight;
  CPlugGameSkin* Remapping;
  CSystemFidsFolder* RemapFolder;
  CHmsLightMap* HmsLightMap;
  CHmsAmbientOcc* HmsAmbientOcc;
};

// File extension: 'DecorationSize.Gbx'
struct CGameCtnDecorationSize : public CMwNod {
  CGameCtnDecorationSize();

  CSceneLayout* Scene;
  uint SizeX;
  uint SizeY;
  uint SizeZ;
  uint BaseHeightBase;
  uint BaseHeightOffset;
  bool OffsetBlockY;
  vec2 EditionZoneMin;
  vec2 EditionZoneMax;
  uint VertexCount;
};

struct CGameGhost : public CMwNod {
  CGameGhost();

};

struct CGameControlCameraFirstPerson : public CGameControlCamera {
  CGameControlCameraFirstPerson();

};

struct CGameControlCameraThirdPerson : public CGameControlCamera {
  CGameControlCameraThirdPerson();

};

// File extension: 'Map.Gbx'
struct CGameCtnChallenge : public CMwNod {
  CGameCtnChallenge();

  enum class CGameCtnChallenge::ETrafficDensity {
    Null = 0,
    Low = 1,
    Normal = 2,
    High = 3,
    Very_High = 4, // Very High
  };
  enum class CGameCtnChallenge::EBotPathTypeEnum {
    Recast = 0,
    Recorded = 1,
  };
  enum class CGameCtnChallenge::ENPlugMaterial_EMapElemColorPalette {
    Classic = 0,
    Stunt = 1,
    Red = 2,
    Orange = 3,
    Yellow = 4,
    Lime = 5,
    Green = 6,
    Cyan = 7,
    Blue = 8,
    Purple = 9,
    Pink = 10,
    White = 11,
    Black = 12,
  };
  const string EdChallengeId;
  CGameCtnChallengeInfo* const MapInfo; // Maniascript
  wstring MapName; // Maniascript
  wstring Comments; // Maniascript
  const string AuthorZoneIconUrl; // Maniascript
  const string CollectionName; // Maniascript
  const string DecorationName; // Maniascript
  const string AuthorLogin; // Maniascript
  const wstring AuthorNickName; // Maniascript
  const wstring AuthorZonePath; // Maniascript
  const wstring MapType; // Maniascript
  const wstring MapStyle; // Maniascript
  const MwId TitleId;
  uint TMObjective_AuthorTime; // Maniascript
  uint TMObjective_GoldTime; // Maniascript
  uint TMObjective_SilverTime; // Maniascript
  uint TMObjective_BronzeTime; // Maniascript
  const uint TMObjective_NbLaps; // Maniascript
  const bool TMObjective_IsLapRace; // Maniascript
  wstring ObjectiveTextAuthor; // Maniascript
  wstring ObjectiveTextGold; // Maniascript
  wstring ObjectiveTextSilver; // Maniascript
  wstring ObjectiveTextBronze; // Maniascript
  const uint CopperPrice; // Maniascript
  const nat3 Size; // Maniascript
  const bool HasCustomIntro; // Maniascript
  const bool HasCustomMusic; // Maniascript
  CGameCtnChallengeParameters* ChallengeParameters;
  CGameCtnCollectorList* BlockStock;
  CPlugBitmap* const AuthorZoneIcon;
  const wstring MapTypeOrLegacyMode;
  string HashedPassword;
  MwId VehicleName;
  MwId VehicleAuthor;
  UnnamedEnum VehicleCollection;
  wstring VehicleCollection_Text;
  bool NeedUnlock;
  UnnamedEnum Kind;
  UnnamedEnum Difficulty;
  CGameCtnDecoration* Decoration;
  CSystemPackDesc* ModPackDesc;
  CSystemPackDesc* CustomMusicPackDesc;
  CPlugSound* CustomMusic;
  CGameCtnCollection* Collection;
  const MwFastBuffer<CGameCtnBlock*> Blocks;
  const MwFastBuffer<CGameCtnBlock*> BakedBlocks;
  const MwFastBuffer<CGameCtnChallenge::EBotPathTypeEnum> BotPathTypes;
  void CheckPlayField();
  const uint VertexCount;
  void ComputeCrc32(CMwNod* Nod, uint Crc32);
  CGameCtnMediaClip* ClipIntro;
  CGameCtnMediaClipGroup* ClipGroupInGame;
  CGameCtnMediaClipGroup* ClipGroupEndRace;
  CGameCtnMediaClip* ClipAmbiance;
  vec2 MapCoordTarget;
  vec2 MapCoordOrigin;
  MwFastBuffer<CSceneVehicleCarMarksSamples*> CarMarksBuffer;
  void TmpBlockOffsetMoulinette();
  uint TmpBlockOffsetMoulinetteY;
  const uint LightMapCacheSmall_KBytes;
  const uint Thumbnail_KBytes;
  MwFastBuffer<CGameCtnAnchoredObject*> AnchoredObjects;
  CPlugAnimFile* AnimLibrary;
  CScriptTraitsMetadata* ScriptMetadata;
  float ThumbnailCam_TransX;
  float ThumbnailCam_TransY;
  float ThumbnailCam_TransZ;
  float ThumbnailCam_Yaw;
  float ThumbnailCam_Pitch;
  float ThumbnailCam_Roll;
  float ThumbnailCam_FovY;
  string ThumbnailCam_XmlStr;
  const uint DecoBaseHeightOffset;
  CGameCtnChallenge::ENPlugMaterial_EMapElemColorPalette ColorPalette;
  uint ColorPaletteAsNat;
};

struct CGameCtnChallengeInfo : public CGameFid {
  const string MapUid; // Maniascript
  const wstring Comments; // Maniascript
  const uint CopperPrice; // Maniascript
  const string CollectionName; // Maniascript
  const string AuthorLogin; // Maniascript
  const wstring AuthorNickName; // Maniascript
  const wstring AuthorZonePath; // Maniascript
  const wstring AuthorZoneFlagUrl; // Maniascript
  const wstring AuthorCountryFlagUrl; // Maniascript
  const wstring MapType; // Maniascript
  const wstring MapStyle; // Maniascript
  const bool IsPlayable; // Maniascript
  const bool CreatedWithSimpleEditor; // Maniascript
  const bool CreatedWithPartyEditor; // Maniascript
  const bool CreatedWithGamepadEditor; // Maniascript
  const uint TMObjective_AuthorTime; // Maniascript
  const uint TMObjective_GoldTime; // Maniascript
  const uint TMObjective_SilverTime; // Maniascript
  const uint TMObjective_BronzeTime; // Maniascript
  const uint TMObjective_NbLaps; // Maniascript
  const bool TMObjective_IsLapRace; // Maniascript
  const wstring NameForUi;
  const string CopperString;
  const vec2 MapCoordOrigin;
  const vec2 MapCoordTarget;
  const UnnamedEnum Kind;
  const bool LapRace;
};

struct CGameOutlineBox : public CMwNod {
  CGameOutlineBox();

  CSceneMobil* const Mobil;
  bool AdditiveElseBlendSrcA;
  bool IsShowQuads;
  bool IsShowLines;
  bool IsShowPrisms;
  float HardLinesColorCoef;
  float SoftLinesColorCoef;
  void Show();
  void Hide();
};

// File extension: 'ParticleParam.Gbx'
struct CGameCtnParticleParam : public CMwNod {
  CGameCtnParticleParam();

  MwId ParticleModelId;
};

struct CGameHighScore : public CMwNod {
  CGameHighScore();

  enum class CGameHighScore::EMedal {
    None = 0,
    Finished = 1,
    Bronze = 2,
    Silver = 3,
    Gold = 4,
    Author = 5,
  };
  const wstring Name; // Maniascript
  const uint Time; // Maniascript
  const wstring Score; // Maniascript
  const uint Rank; // Maniascript
  const uint Count; // Maniascript
  const string GhostUrl; // Maniascript
  const CGameHighScore::EMedal Medal; // Maniascript
  const string GhostName; // Maniascript
};

// File extension: 'PainterSetting.Gbx'
struct CGameCtnPainterSetting : public CMwNod {
  CGameCtnPainterSetting();

  vec2 MinMaxScaleFill;
  vec2 MinMaxScaleSticker;
  vec2 MinMaxScaleStickerText;
  vec2 MinMaxScaleBrush;
  vec2 MinMaxScaleBrushText;
  CGameControlCameraOrbital3d* const Camera;
  void SetDefaultCameraSettings();
  CPlugMaterial* MaterialPaint;
  CPlugMaterial* MaterialFillColor;
  CPlugMaterial* MaterialLayerBlend;
  CPlugMaterial* MaterialLayerModulate;
  CPlugShader* ShaderRasterRgbMask;
  CPlugShader* ShaderRasterAlphaMask;
  CPlugBitmap* BitmapBrushFade;
  CPlugBitmap* BitmapStickerFade;
  CPlugFileImg* ImageSubObjectAllIcon;
  const MwFastBuffer<CMwNod*> ScenesFids;
  float MouseZDeltaRot;
  float MouseZDeltaScale;
  float MouseZDeltaTransparency;
  float MouseZDeltaShininess;
  float BrushDtRefSc;
  bool BrushUseDt;
  bool CameraBottomClipGeometry;
  float CameraBottomIn_m1p1;
  CPlugGameSkin* Remapping;
  CSystemFidsFolder* RemapFolder;
  CPlugCurveSimpleNod* FxBloom_FidFuncIntensAtHdrNorm;
};

struct CGameLeagueManager : public CMwNod {
  CGameLeagueManager();

  uint CacheDuration;
  MwFastBuffer<CGameLeague*> Leagues;
};

struct CGameCtnMediaBlockEditorTriangles : public CMwNod {
  CGameCtnMediaBlockEditorTriangles();

  CGameCtnMediaBlockEditorTriangles::EEditMode EditMode;
  void ActionImportSvg();
  vec3 Position;
  wstring Anchor;
  vec3 Color;
  float Opacity; // Range: 0 - 1
  float VertPosX;
  float VertPosY;
  float VertPosZ;
  void ModeMoveVertexs();
  void ModeCreateTriangles();
  void ModeDeleteVertexs();
};

// File extension: 'GameCtnMediaBlockTriangles2D.gbx'
struct CGameCtnMediaBlockTriangles2D : public CGameCtnMediaBlockTriangles {
  CGameCtnMediaBlockTriangles2D();

};

// File extension: 'GameCtnMediaBlockTriangles3D.gbx'
struct CGameCtnMediaBlockTriangles3D : public CGameCtnMediaBlockTriangles {
  CGameCtnMediaBlockTriangles3D();

};

// Description: "Script API for graphic settings"
struct CGameDisplaySettingsWrapper : public CMwNod {
  enum class CGameDisplaySettingsWrapper::EPreset {
    None = 0,
    VeryFast = 1,
    Fast = 2,
    Nice = 3,
    VeryNice = 4,
  };
  enum class CGameDisplaySettingsWrapper::EDisplayMode {
    FullscreenExclusive = 0,
    Windowed = 1,
    WindowedFull = 2,
  };
  enum class CGameDisplaySettingsWrapper::EShaderQuality {
    VeryFast = 0,
    Fast = 1,
    Nice = 2,
    VeryNice = 3,
  };
  enum class CGameDisplaySettingsWrapper::EShadows {
    None = 0,
    Minimum = 1,
    Medium = 2,
    High = 3,
    VeryHigh = 4,
  };
  enum class CGameDisplaySettingsWrapper::EGpuSync {
    None = 0,
    _3Frames = 1,
    _2Frames = 2,
    _1Frames = 3,
    Immediate = 4,
  };
  enum class CGameDisplaySettingsWrapper::EDisplaySync {
    None = 0,
    VBlank1 = 1,
    VBlank2 = 2,
    VBlank3 = 3,
  };
  enum class CGameDisplaySettingsWrapper::ETripleBuffer {
    Off = 0,
    On = 1,
  };
  enum class CGameDisplaySettingsWrapper::EConsoleResolution {
    UltraPerformance = 0,
    Performance = 1,
    Quality = 2,
  };
  enum class CGameDisplaySettingsWrapper::EShowPerformance {
    None = 0,
    Minimal = 1,
    Fps = 2,
    Fps_Bars = 3,
    Fps_Bars_Legends = 4,
  };
  enum class CGameDisplaySettingsWrapper::EForwardAA {
    None = 0,
    _2x = 1,
    _4x = 2,
    _6x = 3,
    _8x = 4,
    _16x = 5,
  };
  enum class CGameDisplaySettingsWrapper::EDeferredAA {
    None = 0,
    FXAA = 1,
    UBI_TXAA = 2,
  };
  enum class CGameDisplaySettingsWrapper::ETextureFilter {
    Bilinear = 0,
    Trilinear = 1,
    Aniso_2x = 2,
    Aniso_4x = 3,
    Aniso_8x = 4,
    Aniso_16x = 5,
    Aniso_16x_Everywhere = 6,
  };
  enum class CGameDisplaySettingsWrapper::EFxBloomHdr {
    None = 0,
    Medium = 1,
    High = 2,
  };
  enum class CGameDisplaySettingsWrapper::EFxMotionBlur {
    None = 0,
    Enabled = 1,
  };
  enum class CGameDisplaySettingsWrapper::EFxBlur {
    None = 0,
    Enabled = 1,
  };
  enum class CGameDisplaySettingsWrapper::EEverywhereReflect {
    None = 0,
    Enabled = 1,
  };
  enum class CGameDisplaySettingsWrapper::EWaterReflect {
    VeryFast = 0,
    Fast = 1,
    Nice = 2,
    VeryNice = 3,
  };
  enum class CGameDisplaySettingsWrapper::EVehicleReflect {
    Low = 0,
    HighInReplay = 1,
    High = 2,
  };
  enum class CGameDisplaySettingsWrapper::EScreenshotExt {
    JPEG = 0,
    WebP = 1,
    TGA = 2,
  };
  enum class CGameDisplaySettingsWrapper::ERealtimeSSAA {
    Disabled = 0,
    _2Renders = 1,
    _3Renders = 2,
    _4Renders = 3,
    _5Renders = 4,
    _6Renders = 5,
    _7Renders = 6,
    _8Renders = 7,
    _9Renders = 8,
  };
  enum class CGameDisplaySettingsWrapper::ERealtimeSSAA_Motion {
    Disabled = 0,
    _1Frame = 1,
    _2Frames = 2,
    _3Frames = 3,
    _4Frames = 4,
  };
  const uint DirtyCounter; // Maniascript
  const MwFastBuffer<wstring> Devices; // Maniascript
  const MwFastBuffer<wstring> DevicesNames; // Maniascript
  wstring Device_NextStart; // Maniascript
  const wstring Device_Current; // Maniascript
  const wstring DeviceInfo; // Maniascript
  CGameDisplaySettingsWrapper::EDisplayMode DisplayMode; // Maniascript
  bool Customize_NextStart; // Maniascript
  const bool Customize_Current; // Maniascript
  CGameDisplaySettingsWrapper::EPreset Preset_NextStart; // Maniascript
  const CGameDisplaySettingsWrapper::EPreset Preset_Current; // Maniascript
  CGameDisplaySettingsWrapper::EShaderQuality ShaderQuality_NextStart; // Maniascript
  const CGameDisplaySettingsWrapper::EShaderQuality ShaderQuality_Current; // Maniascript
  CGameDisplaySettingsWrapper::EShadows Shadows_NextStart; // Maniascript
  const CGameDisplaySettingsWrapper::EShadows Shadows_Current; // Maniascript
  uint TexturesQuality_NextStart; // Maniascript
  const uint TexturesQuality_Current; // Maniascript
  int2 WindowSize; // Maniascript
  bool WindowBorderless; // Maniascript
  const int2 WindowFullSize; // Maniascript
  int2 FullscreenSize; // Maniascript
  const MwFastBuffer<nat2> FullscreenSizes; // Maniascript
  CGameDisplaySettingsWrapper::EConsoleResolution ConsoleResolution; // Maniascript
  float ConsoleOutputScale; // Maniascript
  CGameDisplaySettingsWrapper::EShowPerformance ShowPerformance; // Maniascript
  bool ConsoleResolution_IsAvailable(CGameDisplaySettingsWrapper::EConsoleResolution ConsoleResolution); // Maniascript
  const bool Support_VRR; // Maniascript
  bool Allow_VRR; // Maniascript
  bool Automatic_Enabled; // Maniascript
  uint Automatic_MinFps; // Maniascript
  uint Automatic_MaxFps; // Maniascript
  bool Automatic_Antialiasing; // Maniascript
  const bool IsDeferred; // Maniascript
  CGameDisplaySettingsWrapper::EForwardAA Antialias_Forward; // Maniascript
  CGameDisplaySettingsWrapper::EDeferredAA Antialias_Deferred; // Maniascript
  CGameDisplaySettingsWrapper::ETextureFilter TextureFiltering; // Maniascript
  uint MaxFps; // Maniascript
  CGameDisplaySettingsWrapper::EGpuSync GpuSync; // Maniascript
  CGameDisplaySettingsWrapper::EDisplaySync DisplaySync; // Maniascript
  CGameDisplaySettingsWrapper::ETripleBuffer TripleBuffer; // Maniascript
  const float AgpUseFactor_Current; // Range: 0 - 1 // Maniascript
  float AgpUseFactor_NextStart; // Range: 0 - 1 // Maniascript
  CGameDisplaySettingsWrapper::EEverywhereReflect EverywhereReflect; // Maniascript
  CGameDisplaySettingsWrapper::EWaterReflect WaterReflect; // Maniascript
  CGameDisplaySettingsWrapper::EVehicleReflect VehicleReflect; // Maniascript
  CGameDisplaySettingsWrapper::EFxBloomHdr FxBloomHdr; // Maniascript
  CGameDisplaySettingsWrapper::EFxMotionBlur FxMotionBlur; // Maniascript
  float FxMotionBlurIntens; // Range: 0 - 1 // Maniascript
  CGameDisplaySettingsWrapper::EFxBlur FxBlur; // Maniascript
  CGameDisplaySettingsWrapper::EScreenshotExt ScreenshotExt; // Maniascript
  CGameDisplaySettingsWrapper::ERealtimeSSAA RealtimeSSAA; // Maniascript
  CGameDisplaySettingsWrapper::ERealtimeSSAA_Motion RealtimeSSAA_Motion; // Maniascript
  uint RealtimeSSAA_MinimumFps; // Maniascript
};

// Description: "The model of a map block"
struct CGameCtnBlockInfo : public CGameCtnCollector {
  enum class CGameCtnBlockInfo::EBaseType {
    None = 0,
    Conductor = 1,
    Generator = 2,
    Collector = 3,
  };
  enum class CGameCtnBlockInfo::EWayPointType {
    Start = 0,
    Finish = 1,
    Checkpoint = 2,
    None = 3,
    StartFinish = 4,
    Dispenser = 5,
  };
  enum class CGameCtnBlockInfo::EMultiDirEnum {
    SameDir = 0,
    SymmetricalDirs = 1,
    AllDir = 2,
    OpposedDirOnly = 3,
    PerpendicularDirsOnly = 4,
    NextDirOnly = 5,
    PreviousDirOnly = 6,
  };
  CGameCtnBlockInfoVariantGround* VariantBaseGround;
  CGameCtnBlockInfoVariantAir* VariantBaseAir;
  const MwFastBuffer<CGameCtnBlockInfoVariantGround*> AdditionalVariantsGround;
  const MwFastBuffer<CGameCtnBlockInfoVariantAir*> AdditionalVariantsAir;
  bool IsPillar;
  bool IsMultiHeightPillarOrVFC;
  CGameCtnBlockInfo::EWayPointType EdWaypointType;
  CGameCtnBlockInfo::EBaseType BaseType;
  MwId SymmetricalBlockInfoId;
  UnnamedEnum Dir;
  CGameCtnBlockInfo* SymmetricalBlockInfoConnected;
  void ClearSymmetricalBlockInfo();
  CGameCtnBlockInfo::EMultiDirEnum PillarShapeMultiDir;
  CGameCtnBlockInfo* BlockInfoModelFid;
  CPlugGameSkinAndFolder* MaterialModifier;
  CPlugGameSkinAndFolder* MaterialModifier2;
  NPlugItemPlacement_STag MatModifierPlacementTag;
  void CopyFromBlockInfoModel();
  void CleanInstanceBeforeSave();
  bool EdNoRespawn;
  bool IconAutoUseGround;
  CGameCtnMacroBlockInfo* IconMacroBlockInfo;
  bool CharPhySpecialPropertyCustomizable;
  CPlugCharPhySpecialProperty* CharPhySpecialProperty;
  CPlugMediaClipList* PodiumClipList;
  CPlugMediaClipList* IntroClipList;
  CPlugFogVolumeBox* FogVolumeBox;
  CPlugSound* Sound1;
  iso4 Sound1Loc;
  CPlugSound* Sound2;
  iso4 Sound2Loc;
  const wstring Name; // Maniascript
  const bool IsRoad; // Maniascript
  const bool IsTerrain; // Maniascript
  const bool IsPodium; // Maniascript
  const CGameCtnBlockInfo::EWayPointType WaypointType; // Maniascript
  const CGameCtnBlockInfo::EWayPointType WayPointType; // Maniascript
  const bool NoRespawn; // Maniascript
  const bool IsClip; // Maniascript
  CGameCtnBlockInfoVariantGround* const VariantGround; // Maniascript
  CGameCtnBlockInfoVariantAir* const VariantAir; // Maniascript
};

// File extension: 'EDFlat.Gbx'
struct CGameCtnBlockInfoFlat : public CGameCtnBlockInfo {
  CGameCtnBlockInfoFlat();

};

// File extension: 'EDFrontier.Gbx'
struct CGameCtnBlockInfoFrontier : public CGameCtnBlockInfo {
  CGameCtnBlockInfoFrontier();

};

// File extension: 'EDClassic.Gbx'
struct CGameCtnBlockInfoClassic : public CGameCtnBlockInfo {
  CGameCtnBlockInfoClassic();

  const MwId BlockInfoGroupId_IsInit;
  const MwId BlockInfoGroupId;
};

// File extension: 'EDRoad.Gbx'
struct CGameCtnBlockInfoRoad : public CGameCtnBlockInfo {
  CGameCtnBlockInfoRoad();

  CGameCtnBlockInfoSlope* Slope;
};

// File extension: 'EDClip.Gbx'
struct CGameCtnBlockInfoClip : public CGameCtnBlockInfo {
  CGameCtnBlockInfoClip();

  enum class CGameCtnBlockInfoClip::EnumClipType {
    ClassicClip = 0,
    FreeClipSide = 1,
    FreeClipTop = 2,
    FreeClipBottom = 3,
  };
  enum class CGameCtnBlockInfoClip::EMultiDirEnum {
    SameDir = 0,
    SymmetricalDirs = 1,
    AllDir = 2,
    OpposedDirOnly = 3,
    PerpendicularDirsOnly = 4,
    NextDirOnly = 5,
    PreviousDirOnly = 6,
  };
  MwId SymmetricalClipId;
  MwId ClipGroupId;
  MwId ClipGroupId2;
  MwId SymmetricalClipGroupId;
  MwId SymmetricalClipGroupId2;
  CGameCtnBlockInfoClip::EnumClipType ClipType;
  bool IsFullFreeClip;
  bool IsExclusiveFreeClip;
  bool CanBeDeletedByFullFreeClip;
  bool IsAlwaysVisibleFreeClip;
  bool IsFCTOrFCBIgnoredByVFC;
  bool IsAntiClip;
  CGameCtnBlockInfoClip::EMultiDirEnum TopBottomMultiDir;
};

// File extension: 'EDSlope.Gbx'
struct CGameCtnBlockInfoSlope : public CGameCtnBlockInfo {
  CGameCtnBlockInfoSlope();

};

// File extension: 'EDPylon.Gbx'
struct CGameCtnBlockInfoPylon : public CGameCtnBlockInfo {
  CGameCtnBlockInfoPylon();

  float PylonOffset;
  UnnamedEnum PylonAmount;
  UnnamedEnum PylonPlacement;
  int BlockHeightOffset;
};

// File extension: 'EDRectAsym.Gbx'
struct CGameCtnBlockInfoRectAsym : public CGameCtnBlockInfo {
  CGameCtnBlockInfoRectAsym();

};

struct CGameCtnBlock : public CMwNod {
  CGameCtnBlock();

  enum class CGameCtnBlock::ECardinalDirections {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
  };
  enum class CGameCtnBlock::EMapElemColor {
    Default = 0,
    White = 1,
    Green = 2,
    Blue = 3,
    Red = 4,
    Black = 5,
  };
  enum class CGameCtnBlock::EMapElemLightmapQuality {
    Normal = 0,
    High = 1,
    VeryHigh = 2,
    Highest = 3,
    Low = 4,
    VeryLow = 5,
    Lowest = 6,
  };
  const MwId DescId;
  const UnnamedEnum CollectionId;
  const wstring CollectionId_Text;
  const MwId DescAuthor;
  uint CoordX;
  uint CoordY;
  uint CoordZ;
  CGameCtnBlock::ECardinalDirections BlockDir;
  const bool IsGround;
  const uint BlockInfoVariantIndex;
  const uint MobilIndex;
  const uint MobilVariantIndex;
  CGameCtnBlock::EMapElemColor MapElemColor;
  CGameCtnBlock::EMapElemLightmapQuality MapElemLmQuality;
  MwFastArray<CGameCtnBlockUnit*> BlockUnitsE;
  CGameCtnBlockSkin* const Skin;
  bool IsEditableInPuzzleOrSimpleEditor;
  CGameCtnBlockInfo* const BlockInfo;
  CPlugCharPhySpecialProperty* CharPhySpecialProperty;
  CGameWaypointSpecialProperty* WaypointSpecialProperty;
  const bool CanHaveAnchor; // Maniascript
  void UseDefaultAnchor(); // Maniascript
  void UseCustomAnchor(); // Maniascript
  const nat3 Coord; // Maniascript
  enum class CGameEditorPluginMap::ECardinalDirections {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
  };
  const CGameEditorPluginMap::ECardinalDirections Dir; // Maniascript
  const CGameCtnBlock::ECardinalDirections Direction; // Maniascript
  const MwFastArray<CGameCtnBlockUnit*> BlockUnits; // Maniascript
  CGameCtnBlockInfo* const BlockModel; // Maniascript
  bool IsGhostBlock(); // Maniascript
};

// Description: "A 1-square-sized part of a block instance."
struct CGameCtnBlockUnit : public CMwNod {
  CGameCtnBlockUnit();

  uint PlacePylons;
  uint AcceptPylons;
  const nat3 Offset; // Maniascript
  const nat3 AbsoluteOffset; // Maniascript
  CGameCtnBlockUnitInfo* const BlockUnitModel; // Maniascript
  CGameCtnBlock* const Block; // Maniascript
};

struct CGameCtnBlockSkin : public CMwNod {
  CGameCtnBlockSkin();

  CSystemPackDesc* PackDesc;
  CSystemPackDesc* ParentPackDesc;
  CSystemPackDesc* ForegroundPackDesc;
};

struct CGameCtnPylonColumn : public CMwNod {
  CGameCtnPylonColumn();

};

struct CGameCtnChallengeParameters : public CMwNod {
  CGameCtnChallengeParameters();

  const uint AuthorScore;
  const uint AuthorTime;
  const uint GoldTime;
  const uint SilverTime;
  const uint BronzeTime;
  wstring MapType;
  wstring MapStyle;
  const wstring Type;
  const wstring Style;
  string Tip;
  CGameCtnGhost* RaceValidateGhost;
  const bool IsValidatedForScriptModes;
};

// File extension: 'Zone.Gbx'
struct CGameCtnZone : public CMwNod {
  CGameCtnZone();

  MwId ZoneId;
  MwId SurfaceId;
  MwId WaterId;
  bool IsLargeZone;
  float VisualTopGroundHeight;
  MwId ForcedParentZoneFrontierId;
  const uint Height;
};

// File extension: 'ZoneFlat.Gbx'
struct CGameCtnZoneFlat : public CGameCtnZone {
  CGameCtnZoneFlat();

  bool GroundOnly;
  bool AutoSimplifyGenealogy;
  CGameCtnBlockInfoFlat* BlockInfoFlat;
  CGameCtnBlockInfoClip* BlockInfoClip;
  CGameCtnBlockInfoRoad* BlockInfoRoad;
  CGameCtnBlockInfoPylon* BlockInfoPylon;
};

// File extension: 'ZoneFrontier.Gbx'
struct CGameCtnZoneFrontier : public CGameCtnZone {
  CGameCtnZoneFrontier();

  int ParentToChildHeightChange;
  const int BlockYOffsetFromParent;
  MwId ParentZoneId;
  MwId ChildZoneId;
  CGameCtnBlockInfoFrontier* BlockInfoFrontier;
  bool FrontierParentBorder_AcceptPylons;
  bool FrontierChildBorder_AcceptPylons;
  bool FrontierTransitionMiddle_AcceptPylons;
  bool FrontierStraightMiddle_AcceptPylons;
  MwFastBuffer<CGameCtnZoneFusionInfo*> CompatibleZones;
};

// File extension: 'GameSkinnedNod.gbx'
struct CGameSkinnedNod : public CGameNod {
  CGameSkinnedNod();

  CMwNod* BaseNodFid;
  CPlugGameSkin* GameSkin;
  MwFastArray<CMwNod*> RemapFids;
};

struct CGameCtnMediaShootParams : public CMwNod {
  CGameCtnMediaShootParams();

  uint VideoFps;
  float Duration;
  uint SizeX;
  uint SizeY;
  bool Hq;
  uint HqSampleCountPerAxe;
  bool HqSoftShadows;
  bool HqAmbientOcc;
  UnnamedEnum Stereo3d;
  bool IsAudioStream;
};

struct CGameScoreLoaderAndSynchronizer : public CMwNod {
};

// userName: 'ActionFxPhy'
struct CGameActionFxPhy : public CMwNod {
};

// Description: "Task result containing a season."
struct CWebServicesTaskResult_SeasonScript : public CWebServicesTaskResult_Season {
  CSeason* const Season; // Maniascript
};

struct CWebServicesTaskResult_SeasonList : public CWebServicesTaskResult {
};

// Description: "Task result containing a list of season."
struct CWebServicesTaskResult_SeasonListScript : public CWebServicesTaskResult_SeasonList {
  const MwFastBuffer<CSeason*> SeasonList; // Maniascript
};

// Description: "Internal API to the manilink browser."
struct CGameManialinkBrowser : public CMwNod {
  CControlFrame* const FramePage;
  CControlFrame* const FrameHeader;
  CGameManialinkPage* const CurrentPage;
  CGameManiaApp* const CurrentManiaApp;
  bool Manialink_Enabled;
  bool Manialink_Active; // Maniascript
  wstring ManialinkBrowser_Link; // Maniascript
  MwId ManialinkBrowser_Frame3dFocused; // Maniascript
  void ManialinkBrowser_OnHome(); // Maniascript
  void ManialinkBrowser_OnQuit(); // Maniascript
  void ManialinkBrowser_OnBack(); // Maniascript
  void ManialinkBrowser_OnForward(); // Maniascript
  void GoToHomeAndResetHistory(); // Maniascript
  void GoToCurUrl(); // Maniascript
};

struct CGameNetFormAdmin : public CNetNod {
  CGameNetFormAdmin();

};

struct CGameNetFileTransfer : public CNetFileTransfer {
};

struct CGameNetFormTimeSync : public CNetFormTimed {
  CGameNetFormTimeSync();

};

struct CGameNetFormCallVote : public CGameNetForm {
  CGameNetFormCallVote();

};

struct CGameControlCamera : public CMwNod {
  CGameControlCamera();

  vec3 Pos;
};

struct CGameControlCameraFree : public CGameControlCamera {
  CGameControlCameraFree();

  vec3 m_FreeVal_Loc_Translation;
  float m_Pitch;
  float m_Yaw;
  float m_Roll;
  float m_Radius;
  vec3 m_RelativeFollowedPos;
  vec3 m_TargetPos;
  bool m_TargetIsEnabled;
  float m_Fov;
  float m_NearZ;
  float m_FarZ;
  float m_FreeVal_Lens_DofFocusZ;
  float m_FreeVal_Lens_DofLensSize;
  float m_Acceleration;
  float m_StartMoveSpeed;
  uint m_MoveSpeedCoef;
  float m_MoveSpeed;
  float m_MoveInertia;
  float m_RotateSpeed;
  float m_RotateInertia;
  bool m_UseForcedRoll;
  float m_ForcedRoll;
  bool m_DisableMouseZ;
};

struct CGameControlCameraOrbital3d : public CGameControlCameraTarget {
  CGameControlCameraOrbital3d();

  bool m_CanUseRelativeTargetLocation;
  iso4 m_RelativeTargetLocation;
  float m_Fov;
  float m_MinFov;
  float m_MaxFov;
  bool m_CanCameraMove;
  bool m_UseAutoRadiusFromTargets;
  vec2 m_OrbitalParams_m_RotateSpeed;
  vec3 m_OrbitalParams_m_RadiusScale;
  bool m_OcclusionIsEnable;
  bool m_UsingBorderToRotate;
  float m_MouseBorderMoveSize;
  float m_OcclusionTargetRadius;
  float m_OcclusionDistFromHit;
  float m_Radius;
  float m_Latitude;
  float m_Longitude;
  float m_OrbitalParams_m_WheelSensitivity;
  float m_OrbitalParams_m_FovKeySensitivity;
  float m_OrbitalParams_m_ZoomKeySensitivity;
  float m_OrbitalParams_m_DefaultRadius;
  float m_OrbitalParams_m_RadiusMin;
  float m_OrbitalParams_m_RadiusMax;
  float m_OrbitalParams_m_LatitudeMin;
  float m_OrbitalParams_m_LatitudeMax;
  bool m_FixedTargetLocation;
  bool m_UseNonLinearZoom;
};

struct CGameControlCameraEffect : public CMwNod {
  CGameControlCameraEffect();

  void Reset();
  void Install();
  void Uninstall();
  const bool IsInstalled;
};

struct CGameControlCameraEffectShake : public CGameControlCameraEffect {
  CGameControlCameraEffectShake();

  const vec3 OffsetPos;
  float Yaw;
  float Pitch;
  float Roll;
  float Speed;
  float Intensity;
};

struct CGameControlCameraTarget : public CGameControlCamera {
  CGameControlCameraTarget();

};

struct CGameLadderRanking : public CMwNod {
  wstring Name;
  wstring Path;
  string Login;
  string ScoreStr;
  uint Ranking;
  uint Ranking2;
  uint Score;
  uint Stars;
  uint ChildsCount;
  bool IsFolder;
};

struct CGameCtnMediaBlock : public CMwNod {
  const bool IsInstalled;
  const bool IsActive;
  void SwitchOn();
  const float Start; // Maniascript
  const float End; // Maniascript
  float GetKeyTime(uint KeyIndex); // Maniascript
  uint GetKeysCount(); // Maniascript
};

// File extension: 'GameCtnMediaTrack.gbx'
struct CGameCtnMediaTrack : public CMwNod {
  CGameCtnMediaTrack();

  wstring Name; // Maniascript
  bool IsReadOnly;
  bool IsCycling;
  const MwFastBuffer<CGameCtnMediaBlock*> Blocks; // Maniascript
};

// File extension: 'Clip.gbx'
struct CGameCtnMediaClip : public CMwNod {
  CGameCtnMediaClip();

  wstring Name; // Maniascript
  const MwFastBuffer<CGameCtnMediaTrack*> Tracks; // Maniascript
  uint LocalPlayerClipEntIndex;
  bool StopOnRespawn;
  bool StopWhenRespawn; // Maniascript
  bool StopWhenLeave; // Maniascript
  bool TriggersBeforeRaceStart; // Maniascript
  bool Compat_IntroRelativeToBlock;
};

// File extension: 'GameCtnMediaClipGroup.gbx'
struct CGameCtnMediaClipGroup : public CMwNod {
  CGameCtnMediaClipGroup();

  const MwFastBuffer<CGameCtnMediaClip*> Clips; // Maniascript
};

struct CGameCtnMediaBlockCamera : public CGameCtnMediaBlock {
  const bool IsInstalled;
  const bool IsActive;
};

// File extension: 'GameCtnMediaBlockUi.gbx'
struct CGameCtnMediaBlockUi : public CGameCtnMediaBlock {
  CGameCtnMediaBlockUi();

  CControlContainer* UserInterface;
};

struct CGameCtnMediaBlockFx : public CGameCtnMediaBlock {
  const float FadeDuration;
  CSceneFx* const SceneFx;
};

struct CGameCtnMediaBlockFxBlur : public CGameCtnMediaBlockFx {
};

// File extension: 'GameCtnMediaBlockFxColors.gbx'
struct CGameCtnMediaBlockFxColors : public CGameCtnMediaBlockFx {
  CGameCtnMediaBlockFxColors();

  void SwitchOn();
};

// File extension: 'GameCtnMediaBlockFxBlurDepth.gbx'
struct CGameCtnMediaBlockFxBlurDepth : public CGameCtnMediaBlockFxBlur {
  CGameCtnMediaBlockFxBlurDepth();

  void SwitchOn();
};

// File extension: 'GameCtnMediaBlockFxBlurMotion.gbx'
struct CGameCtnMediaBlockFxBlurMotion : public CGameCtnMediaBlockFxBlur {
  CGameCtnMediaBlockFxBlurMotion();

};

// File extension: 'GameCtnMediaBlockFxBloom.gbx'
struct CGameCtnMediaBlockFxBloom : public CGameCtnMediaBlockFx {
  CGameCtnMediaBlockFxBloom();

  void SwitchOn();
};

// File extension: 'GameCtnMediaBlockCamGame.gbx'
struct CGameCtnMediaBlockCameraGame : public CGameCtnMediaBlockCamera {
  CGameCtnMediaBlockCameraGame();

  enum class CGameCtnMediaBlockCameraGame::EGameCam {
    Default = 0,
    Internal = 1,
    External = 2,
    Helico = 3,
    Free = 4,
    Spectator = 5,
    External_2 = 6, // External 2
  };
  const CGameCtnMediaBlockCameraGame::EGameCam GameCam;
  const uint ClipEntId;
  const wstring TargetClipEntName;
  const CGameCtnMediaBlockCameraGame::EGameCam GameCamDesign;
};

struct CGameCtnMediaBlockTime : public CGameCtnMediaBlock {
  CGameCtnMediaBlockTime();

  void SwitchOn();
};

struct CGameCtnMediaClipPlayer : public CMwNod {
  CGameCtnMediaClip* const Clip;
  CScene2d* const Scene2d;
  const MwFastBuffer<CGameCtnMediaTrack*> EdMediaTracks;
  const uint LocalPlayerEntityUId;
};

struct CGameCtnMediaBlockEvent_deprecated : public CGameCtnMediaBlock {
};

// Description: "Task result containing a part of the history of account trophy gain."
struct CWebServicesTaskResult_AccountTrophyGainHistoryScript : public CWebServicesTaskResult_AccountTrophyGainHistory {
  const MwFastBuffer<CAccountTrophyGainForHistory*> AccountTrophyGainList; // Maniascript
  const uint AccountTrophyGainTotalCount; // Maniascript
};

struct CGameManiaNetResource : public CMwNod {
  const string Url;
};

// Description: "User profile."
struct CGamePlayerInfo : public CGameNetPlayerInfo {
  CGamePlayerInfo();

  enum class CGamePlayerInfo::EEchelon {
    None = 0,
    Bronze1 = 1,
    Bronze2 = 2,
    Bronze3 = 3,
    Silver1 = 4,
    Silver2 = 5,
    Silver3 = 6,
    Gold1 = 7,
    Gold2 = 8,
    Gold3 = 9,
  };
  enum class CGamePlayerInfo::ETagType {
    Bronze = 0,
    Silver = 1,
    Gold = 2,
    Nadeo = 3,
  };
  enum class CGamePlayerInfo::EStereoDisplayMode {
    None = 0,
    Stereo = 1,
    HMD = 2,
  };
  const string Login; // Maniascript
  const string WebServicesUserId; // Maniascript
  const wstring Name; // Maniascript
  const bool IsFirstPartyDisplayName; // Maniascript
  const string AvatarUrl; // Maniascript
  const wstring ZonePath; // Maniascript
  const string ZoneFlagUrl; // Maniascript
  const string CountryFlagUrl; // Maniascript
  const wstring CountryPath; // Maniascript
  const MwFastBuffer<wstring> ZoneIdPath; // Maniascript
  const string Language; // Maniascript
  const wstring Description; // Maniascript
  const vec3 Color; // Maniascript
  const string ClubLink; // Maniascript
  const string Trigram; // Maniascript
  const string BroadcastTVLogin; // Maniascript
  const string SteamUserId; // Maniascript
  const uint FameStars; // Maniascript
  const CGamePlayerInfo::EEchelon Echelon; // Maniascript
  const uint NextEchelonPercent; // Maniascript
  const bool IsBeginner; // Maniascript
  const uint LadderRank; // Maniascript
  const uint LadderTotal; // Maniascript
  const float LadderPoints; // Maniascript
  const wstring LadderZoneName; // Maniascript
  const string LadderZoneFlagUrl; // Maniascript
  const float ReferenceScore; // Maniascript
  const bool IsFakeUser; // Maniascript
  const MwFastBuffer<uint> Tags_Favored_Indices; // Maniascript
  const MwFastBuffer<string> Tags_Id; // Maniascript
  const MwFastBuffer<CGamePlayerInfo::ETagType> Tags_Type; // Maniascript
  const MwFastBuffer<wstring> Tags_Comments; // Maniascript
  const MwFastBuffer<wstring> Tags_Deliverer; // Maniascript
  const wstring ClubTag; // Maniascript
  const CGamePlayerInfo::EStereoDisplayMode StereoDisplayMode; // Maniascript
  const bool ColorblindModeEnabled; // Maniascript
  bool HackCamHmdDisabled; // Maniascript
  const wstring AvatarDisplayName; // Maniascript
  const wstring HornDisplayName; // Maniascript
  const wstring Model_CarSport_SkinName; // Maniascript
  const string Model_CarSport_SkinUrl; // Maniascript
  const wstring Model_CharacterPilot_SkinName; // Maniascript
  const string Model_CharacterPilot_SkinUrl; // Maniascript
  const string Prestige_SkinOptions; // Maniascript
  const string Character_SkinOptions; // Maniascript
  CGameUserVoiceChat* const VoiceChat; // Maniascript
  const uint RequestedClan; // Maniascript
  const bool RequestsSpectate; // Maniascript
  const bool IsConnectedToMasterServer; // Maniascript
  const MwFastBuffer<string> AlliesConnected; // Maniascript
  CPlugBitmap* const ZoneBitmap;
  CGameLeague* const ZoneLeague;
  const bool IsSpectator;
  uint PlaygroundTeamRequested;
  float LightTrailLinearHue; // Range: 0 - 1
  const wstring StrLadderRanking;
  const wstring StrLadderRankingSimple;
  const string StrLadderScore;
  const string StrLadderLastPoints;
  const string StrLadderWins;
  const string StrLadderDraws;
  const string StrLadderLosses;
  const string StrLadderTeamName;
  const string StrLadderTeamRanking;
  const string StrLadderTeamRankingSimple;
  const string StrLadderNbrTeams;
  const string StrLadderScoreRounded;
  const uint PlaygroundRoundNum;
  const string GameStateName;
};

// Description: "User privileges manager."
struct CGameClientTrackingScript : public CMwNod {
  enum class CGameClientTrackingScript::EEventColorLevel {
    White = 0,
    Green = 1,
    Blue = 2,
    Red = 3,
    Black = 4,
  };
  enum class CGameClientTrackingScript::EEventMedalLevel {
    Finished = 0,
    Bronze = 1,
    Silver = 2,
    Gold = 3,
    Author = 4,
  };
  enum class CGameClientTrackingScript::EEventSeason {
    Winter = 0,
    Spring = 1,
    Summer = 2,
    Fall = 3,
  };
  enum class CGameClientTrackingScript::EPlayTimeContext {
    None = 0,
    Club = 1,
    Create = 2,
    Live = 3,
    Local = 4,
    Solo = 5,
  };
  const MwFastBuffer<CWebServicesTaskResult*> TaskResults; // Maniascript
  void TaskResult_Release(MwId TaskId); // Maniascript
  void Track_ContextMenuStart(MwId UserId, string MenuName); // Maniascript
  void Track_ContextMenuStop(MwId UserId, string MenuName); // Maniascript
  void Track_ContextGameModeStart(MwId UserId, string GameMode); // Maniascript
  void Track_ContextGameModeStop(MwId UserId, string GameMode); // Maniascript
  void Track_ContextMapStart(MwId UserId, string MapUid, string Environment); // Maniascript
  void Track_ContextMapStop(MwId UserId, string MapUid); // Maniascript
  void Track_ContextPlayStart(MwId UserId, string Type); // Maniascript
  void Track_ContextPlayStop(MwId UserId, string Type, string Reason, uint NbRespawns); // Maniascript
  void Track_Context_PlayTime(MwId UserId, CGameClientTrackingScript::EPlayTimeContext PlayTimeContext); // Maniascript
  void Track_Context_MenuStart(MwId UserId, string MenuName); // Maniascript
  void Track_Context_MenuStop(MwId UserId, string MenuName); // Maniascript
  void Track_Context_GameModeStart(MwId UserId, string GameMode); // Maniascript
  void Track_Context_GameModeStop(MwId UserId, string GameMode); // Maniascript
  void Track_Context_MapStart(MwId UserId, string MapUid, string Environment); // Maniascript
  void Track_Context_MapStop(MwId UserId, string MapUid); // Maniascript
  void Track_Context_PlayStart(MwId UserId, string Type); // Maniascript
  void Track_Context_PlayStop(MwId UserId, string Type, string Reason, uint NbRespawns); // Maniascript
  void Track_NewMapCreated(MwId UserId, string Environment, bool IsRandomlyGenerated); // Maniascript
  void Track_Create_NewMapCreated(MwId UserId, string Environment, bool IsRandomlyGenerated); // Maniascript
  void Track_Live_COTDPlayed(MwId UserId, uint Rank, bool Win); // Maniascript
  void Track_Live_MultiplayerPlayed(MwId UserId, uint Rank, bool Win); // Maniascript
  void Track_Live_RankedPlayed(MwId UserId, uint Rank, bool Win); // Maniascript
  void Track_Live_RoyalPlayed(MwId UserId, uint Rank, bool Win); // Maniascript
  void Track_Live_RoyalPlayed_V2(MwId UserId, uint Rank, bool Win, bool IsSuperRoyal, string Division); // Maniascript
  void Track_Live_RoyalSectionFinished(MwId UserId, uint ColorLevel); // Maniascript
  void Track_Live_RoyalSectionFinished_V2(MwId UserId, CGameClientTrackingScript::EEventColorLevel ColorLevel); // Maniascript
  void Track_Local_HotseatPlayed(MwId UserId, uint Rank, bool Win); // Maniascript
  void Track_Local_HotseatPlayed_V2(MwId UserId); // Maniascript
  void Track_Local_SplitScreenPlayed(MwId UserId, uint Rank, bool Win); // Maniascript
  void Track_Local_SplitScreenPlayed_V2(MwId UserId); // Maniascript
  void Track_NewsClick(MwId UserId, string NewsId); // Maniascript
  void Track_NewsImpression(MwId UserId, string NewsId); // Maniascript
  void Track_News_PlayerAction(MwId UserId, string NewsId, string Placement, string Action); // Maniascript
  void Track_News_PlayerImpression(MwId UserId, string NewsId, string Placement, uint Duration); // Maniascript
  void Track_Player_MedalEarned(MwId UserId, uint Finished, uint BronzeMedal, uint SilverMedal, uint GoldMedal, uint AuthorMedal, bool IsOfficialCampaign, bool IsTOTD); // Maniascript
  void Track_Player_OfficialCampaign10TrackCompleted(MwId UserId, uint Year, CGameClientTrackingScript::EEventSeason Season, CGameClientTrackingScript::EEventMedalLevel MedalLevel); // Maniascript
  void Track_Player_OfficialCampaignAllTrackCompleted(MwId UserId, uint Year, uint Season, uint MedalLevel); // Maniascript
  void Track_Player_OfficialCampaignAllTrackCompleted_V2(MwId UserId, uint Year, CGameClientTrackingScript::EEventSeason Season, CGameClientTrackingScript::EEventMedalLevel MedalLevel); // Maniascript
  void Track_Player_TrackOfTheDayWeekAllTrackCompleted(MwId UserId, uint Year, uint Week, uint MedalLevel); // Maniascript
  void Track_Player_TrackOfTheDayWeekAllTrackCompleted_V2(MwId UserId, uint Year, uint Week, CGameClientTrackingScript::EEventMedalLevel MedalLevel); // Maniascript
  void Track_Player_TrophyEarned(MwId UserId, uint T1CountPtr, uint T2CountPtr, uint T3CountPtr, uint T4CountPtr, uint T5CountPtr, uint T6CountPtr, uint T7CountPtr, uint T8CountPtr, uint T9CountPtr); // Maniascript
  void Track_Player_StartActivity(MwId UserId, string ActivityId); // Maniascript
  void Track_Player_EndActivity(MwId UserId, string ActivityId); // Maniascript
};

struct CGamePlayerProfile : public CMwNod {
  CGamePlayerProfile();

  string ProfileName;
  const wstring DisplayProfileName;
  const uint TotalPlayTime;
  const MwFastBuffer<CGamePlayerProfileChunk*> Chunks;
  CGamePlayerProfileChunk_AccountSettings* AccountSettings;
  CGamePlayerProfileChunk_GameSettings* GameSettings;
  CGamePlayerProfileChunk_GameStats* GameStats;
  CGamePlayerProfileChunk_GlobalInterfaceSettings* GlobalInterfaceSettings;
  CGamePlayerProfileChunk_InterfaceSettings* InterfaceSettings;
  CGamePlayerProfileChunk_VehiclesSettings* VehiclesSettings;
  CGamePlayerProfileChunk_ManiaPlanetStations* ManiaPlanetStations;
  const MwFastBuffer<CGamePlayerProfileChunk_InputBindingsConfig*> InputBindingsConfigs;
  const MwFastBuffer<CGamePlayerProfileChunk_PackagesInfos*> PackagesInfosChunks;
  CScriptTraitsPersistent* const ScriptPersistentTraits_NoTitle;
};

struct CGameScriptDebugger : public CMwNod {
  enum class CGameScriptDebugger::EVisibility {
    Hidden = 0,
    Editor = 1,
    EditorAndLogs = 2,
    LogsFull = 3,
    LogsReduced = 4,
  };
  CGameScriptDebugger::EVisibility Visibility;
  wstring DialogEditScript_TempText;
  wstring DialogEditScript_TempSearchText;
  uint DialogEditScript_TempLineNumber;
  wstring DialogEditScript_TempReplaceText;
  wstring DialogEditScript_TempSearchInclude;
  void DialogEditScript_Ok();
  void DialogEditScript_RequestCancel();
  void DialogEditScript_Cancel();
  void DialogEditScript_Compile();
  void DialogEditScript_Save();
  void DialogEditScript_SaveNew();
  void DialogEditScript_Save_Done();
  void DialogEditScript_SaveAllFiles_Step2();
  void DialogEditScript_AddNewFileStep2();
  void DialogEditScript_Find();
  void DialogEditScript_ToggleIncludes();
  void DialogEditScript_NextSelection();
  void DialogEditScript_FileOpened();
  void DialogEditScript_SearchInclude();
};

// File extension: 'League.Gbx'
struct CGameLeague : public CMwNod {
  CGameLeague();

  wstring Path;
  wstring Name;
  wstring Description;
  string Login;
  bool IsGroup;
};

struct CGameCtnChallengeGroup : public CMwNod {
  CGameCtnChallengeGroup();

  wstring Name;
  void CleanChallenges();
  void EmptyChallenges();
  const MwFastBuffer<CGameCtnChallengeInfo*> MapInfos; // Maniascript
};

// File extension: 'GameCtnCampaign.Gbx'
struct CGameCtnCampaign : public CMwNod {
  CGameCtnCampaign();

  enum class CGameCtnCampaign::EType {
    None = 0,
    Race = 1,
    Puzzle = 2,
    Survival = 3,
    Platform = 4,
    Stunts = 5,
    Training = 6,
  };
  enum class CGameCtnCampaign::ERequiredPlayersCount {
    Solo_Only = 0, // Solo Only
    Multi_Only = 1, // Multi Only
    Duo_Only = 2, // Duo Only
    Trio_Only = 3, // Trio Only
    Quatuor_Only = 4, // Quatuor Only
    All = 5,
  };
  enum class CGameCtnCampaign::EUnlockType {
    By_Row = 0, // By Row(Nations)
    By_Column = 1, // By Column(United)
    Custom = 2,
  };
  enum class CGameCtnCampaign::EMedal {
    None = 0,
    Finished = 1,
    Bronze = 2,
    Silver = 3,
    Gold = 4,
    Author = 5,
  };
  const string CampaignId; // Maniascript
  MwId CollectionId;
  MwId IconId;
  wstring Name;
  CGameCtnCampaign::EType Type;
  uint Index;
  CGameCtnCampaign::EUnlockType UnlockType;
  CGameCtnCampaign::ERequiredPlayersCount RequiredPlayersCount;
  wstring UnlockedByCampaign;
  MwFastBuffer<CGameCtnChallengeGroup*> MapGroups; // Maniascript
  void AddChallengeGroup();
  wstring ModeScriptName;
  uint GetMapGroupCount(); // Maniascript
  CGameCtnChallengeGroup* GetMapGroup(uint Index); // Maniascript
  CGameCtnChallengeInfo* GetNextMap(CGameCtnChallengeInfo* CurrentMapInfo); // Maniascript
  string ScoreContext; // Maniascript
  bool OfficialRecordEnabled; // Maniascript
};

struct CGameCtnGhost : public CGameGhost {
  CGameCtnGhost();

  uint Duration;
  uint Size;
  MwId ModelIdentName;
  MwId ModelIdentAuthor;
  EDummyCollectionIdent ModelIdentCollection;
  SConstString ModelIdentCollection_Text;
  string GhostLogin;
  string GhostTrigram;
  wstring GhostCountryPath;
  wstring GhostNickname;
  EGhostNameLogoType m_GhostNameLogoType;
  wstring GhostAvatarName;
  string RecordingContext;
  vec3 LightTrailColor;
  uint RaceTime;
  uint NbRespawns;
  uint StuntsScore;
  MwId Validate_ChallengeUid;
  SConstString Validate_ScopeType;
  string Validate_ScopeId;
  string Validate_GameMode;
  string Validate_GameModeCustomData;
  string Validate_ExeVersion;
  uint Validate_ExeChecksum;
  string Validate_TitleId;
  string Validate_ExtraTool_Info;
  uint Validate_OsKind;
  uint Validate_CpuKind;
};

// userName: 'TrackManiaReplay'
// File extension: 'Replay.Gbx'
struct CGameCtnReplayRecord : public CMwNod {
  CGameCtnReplayRecord();

  CGameCtnMediaClip* Clip;
  MwFastBuffer<CGameCtnGhost*> Ghosts;
  CGameCtnChallenge* Challenge;
  uint Duration;
  CFuncSegment* HumanTimeToGameTimeFunc;
};

struct CGameCtnReplayRecordInfo : public CGameFid {
  const MwId ChallengeId;
  const string MapUid; // Maniascript
  const MwId CollectionId;
  const uint BestTime;
  const wstring PlayerNickname;
  const string PlayerLogin;
};

struct CGameLadderRankingLeague : public CGameLadderRanking {
};

struct CGameLadderRankingPlayer : public CGameLadderRanking {
};

struct CGameLadderRankingSkill : public CGameLadderRanking {
  const wstring DynamicLeagueName;
  CSystemData* const DynamicLeagueLogo;
};

// File extension: 'Frame.Gbx'
struct CGameControlCard : public CControlFrame {
  CGameControlCard();

  CMwNod* DisplayedNod;
  void ForceReconfig();
  void ConnectChilds();
  void DisconnectChilds();
  bool UseDelays;
  bool UseOwnData;
  bool SelectionEnabled;
  bool CardFocused;
  bool CardSelected;
};

// File extension: 'CtnCtlCardMng.Gbx'
struct CGameControlCardManager : public CMwNod {
  CGameControlCardManager();

  const MwFastBuffer<CGameControlDataType*> DataTypes;
};

// File extension: 'CtnDispType.Gbx'
struct CGameControlDataType : public CMwNod {
  CGameControlDataType();

  string Name;
  CGameControlCard* CardTemplate;
};

struct CGameCtnMediaBlockCameraSimple : public CGameCtnMediaBlockCamera {
  CGameCtnMediaBlockCameraSimple();

  const uint KeyCache;
};

struct CGameCtnMediaBlockCameraOrbital : public CGameCtnMediaBlockCamera {
  CGameCtnMediaBlockCameraOrbital();

};

// File extension: 'CtnMediaBlockCamPath.gbx'
struct CGameCtnMediaBlockCameraPath : public CGameCtnMediaBlockCamera {
  CGameCtnMediaBlockCameraPath();

  const bool IsPathDirty;
  CFuncKeysTrans* Path;
  const MwFastArray<float> Lengths;
  const float CurveLength;
  const float TotalWeight;
};

// File extension: 'CtnMediaBlockCamCustom.gbx'
struct CGameCtnMediaBlockCameraCustom : public CGameCtnMediaBlockCamera {
  CGameCtnMediaBlockCameraCustom();

  const uint KeyCache;
};

struct CGameCtnMediaBlockCameraEffect : public CGameCtnMediaBlock {
};

// File extension: 'CtnMediaBlockCamFxShake.gbx'
struct CGameCtnMediaBlockCameraEffectShake : public CGameCtnMediaBlockCameraEffect {
  CGameCtnMediaBlockCameraEffectShake();

  CGameControlCameraEffect* const Effect;
};

// File extension: 'GameCtnMediaBlockImage.gbx'
struct CGameCtnMediaBlockImage : public CGameCtnMediaBlock {
  CGameCtnMediaBlockImage();

  CPlugBitmap* Bitmap;
  CControlEffectSimi* const Effect;
  CControlBase* const Mobil;
};

// File extension: 'CtnMediaBlockMusicFx.gbx'
struct CGameCtnMediaBlockMusicEffect : public CGameCtnMediaBlock {
  CGameCtnMediaBlockMusicEffect();

  uint Dummy;
};

// File extension: 'GameCtnMediaBlockSound.gbx'
struct CGameCtnMediaBlockSound : public CGameCtnMediaBlock {
  CGameCtnMediaBlockSound();

  CPlugSound* Sound;
  CAudioSource* AudioSource;
  const bool IsBlockPlaying;
  bool IsLooping;
  bool IsMusic;
  bool StopWithClip;
  uint PlayCount;
  void OnParamsModified();
};

// File extension: 'GameCtnMediaBlockText.gbx'
struct CGameCtnMediaBlockText : public CGameCtnMediaBlock {
  CGameCtnMediaBlockText();

  wstring Text;
  CControlBase* const Mobil;
  CControlEffectSimi* const Effect;
};

// File extension: 'GameCtnMediaBlockTrails.gbx'
struct CGameCtnMediaBlockTrails : public CGameCtnMediaBlock {
  CGameCtnMediaBlockTrails();

};

struct CGameCtnMediaBlockTransition : public CGameCtnMediaBlock {
};

// File extension: 'CtnMediaBlockTransFade.gbx'
struct CGameCtnMediaBlockTransitionFade : public CGameCtnMediaBlockTransition {
  CGameCtnMediaBlockTransitionFade();

  CControlBase* const Mobil;
  vec3 Color;
};

struct CGameCtnMediaClipViewer : public CGameSwitcherModule {
  CGameCtnMediaClipViewer();

  const bool DoInputs;
  enum class CGameCtnMediaClipViewer::EStatus {
    Running = 0,
    Exit = 1,
    Next = 2,
  };
  const CGameCtnMediaClipViewer::EStatus ViewerStatus;
  CGameCtnMediaClip* const Clip;
  CGameCtnMediaClipPlayer* const ClipPlayer;
  float PlaySpeed;
  CGameCtnMediaClipGroup* const ClipGroup;
  CGameCtnMediaClip* const ClipGroupClip;
  CGameCtnMediaClipPlayer* const ClipGroupPlayer;
  uint ClipIndexInGroup;
  float ClipGroupStartTime;
};

struct CGameCursorBlock : public CMwNod {
  CGameCursorBlock();

  enum class CGameCursorBlock::ECardinalDirEnum {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
  };
  enum class CGameCursorBlock::EAdditionalDirEnum {
    P0deg = 0, // +0deg
    P15deg = 1, // +15deg
    P30deg = 2, // +30deg
    P45deg = 3, // +45deg
    P60deg = 4, // +60deg
    P75deg = 5, // +75deg
  };
  nat3 Coord;
  nat3 Subdiv;
  nat3 SubdivFactors;
  CGameCursorBlock::ECardinalDirEnum Dir;
  CGameCursorBlock::EAdditionalDirEnum AdditionalDir;
  float Pitch;
  float Roll;
  bool UseFreePos;
  vec3 FreePosInMap;
  bool UseSnappedLoc;
  vec3 SnappedLocInMap_Trans;
  float SnappedLocInMap_Yaw;
  float SnappedLocInMap_Pitch;
  float SnappedLocInMap_Roll;
  vec3 Color;
  float BrightnessFactor;
  CGameOutlineBox* const CursorBox;
  CPlugTree* const CursorBoxShadow;
  CPlugTree* const DirectionalArrow;
  vec3 CanPlaceColor;
  vec3 CanChangeVariantColor;
  vec3 CanJoinColor;
  vec3 CannotPlaceNorJoinColor;
  vec3 NothingToDoColor;
  vec3 GhostBlockColor;
  vec3 VariantForcedColor;
};

struct CGameCtnEditor : public CGameEditorParent {
  CControlContainer* FrameRoot;
  CGameEditorMainPlugin* const MainPLugin;
};

struct CGameCtnEdControlCam : public CMwNod {
};

struct CGameCtnEdControlCamCustom : public CGameCtnEdControlCam {
  CGameCtnEdControlCamCustom();

};

struct CGameCtnEdControlCamPath : public CGameCtnEdControlCam {
  CGameCtnEdControlCamPath();

};

struct CGameControlGrid : public CControlGrid {
  CGameControlGrid();

  uint MaxPerColumn;
  uint MaxPerRow;
  uint FastNextPageCount;
  uint FastPreviousPageCount;
  uint ForcedPageCount;
  bool ForceHideArrows;
  bool ForceHidePageCounter;
  bool HaveLocalData;
  bool HideLocalDataIfNone;
  uint CurrentPage;
  uint PageCount;
  void UpdatePageCount();
  void UpdatePageCounter();
  void UpdateNavigationButtons();
  void UpdateLocalData();
  void OnCurrentPageChanged();
  CControlButton* ButtonFirstPage;
  CControlButton* ButtonFastPreviousPage;
  CControlButton* ButtonPreviousPage;
  CControlEntry* EntryPageCounter;
  CControlButton* ButtonNextPage;
  CControlButton* ButtonFastNextPage;
  CControlButton* ButtonLastPage;
  CControlBase* BaseLocalData;
  void OnFirstPage();
  void OnFastPreviousPage();
  void OnPreviousPage();
  void OnNextPage();
  void OnFastNextPage();
  void OnLastPage();
  string StrPageCounter;
  const uint Remote_TotalCount;
  const string Remote_SpecificOverTotalCount;
  CGameRemoteBufferPool* const Remote_Pool;
};

// File extension: 'GameGridCards.Gbx'
struct CGameControlGridCard : public CGameControlGrid {
  CGameControlGridCard();

  CGameControlCardManager* CardManager;
  bool PushByColumns;
  bool LeftToRight;
  bool UpToDown;
  bool CacheAtCreation;
  bool AreCardsReadOnly;
  bool UseCustomSelection;
  bool FillWithDefault;
  string DefaultCardName;
  MwFastBuffer<CMwNod*> NodsToDisplay;
  MwFastBuffer<CGameControlCard*> NodCards;
  void UpdateFromDatas();
  void UpdateOnlyCards();
  void PrecacheAll();
  void CleanCaches();
};

struct CGameCtnNetServerInfo : public CGameNetServerInfo {
  const wstring ServerName; // Maniascript
  const string ServerLogin; // Maniascript
  const string JoinLink; // Maniascript
  const wstring Comment; // Maniascript
  const string ServerVersionBuild; // Maniascript
  const uint PlayerCount; // Maniascript
  const uint MaxPlayerCount; // Maniascript
  const uint SpectatorCount; // Maniascript
  const uint MaxSpectatorCount; // Maniascript
  const float PlayersLevelMin; // Maniascript
  const float PlayersLevelAvg; // Maniascript
  const float PlayersLevelMax; // Maniascript
  const wstring ModeName; // Maniascript
  const float LadderServerLimitMax; // Maniascript
  const float LadderServerLimitMin; // Maniascript
  const MwFastBuffer<wstring> PlayerNames; // Maniascript
  const MwFastBuffer<wstring> ChallengeNames; // Maniascript
  const MwFastBuffer<wstring> ChallengeIds; // Maniascript
  const uint NbChallenges; // Maniascript
  const bool HasBuddies; // Maniascript
  const bool IsFavourite; // Maniascript
  const bool IsLobbyServer; // Maniascript
  const bool IsPrivate; // Maniascript
  const bool IsPrivateForSpectator; // Maniascript
  const bool IsMapDownloadAllowed; // Maniascript
  const bool IsTeamMode; // Maniascript
  const bool IsDisableChangeTeams; // Maniascript
  const string SendToServerAfterMatchUrl; // Maniascript
  const uint ViewerCount; // Maniascript
  wstring ServerName_Menu;
  wstring Comment_Menu;
  const string PlayerCountOverMax;
  const UnnamedEnum LadderMode;
  const wstring PlayerName;
  const bool IsBuddy_HackForServerInfoUsedAsPlayer;
  bool AllowDownload;
  bool DisableHorns;
  bool DisableServiceAnnounces;
  bool DisableProfileSkins;
  string ClientUIRootModuleUrl;
  uint HideServer;
  bool IsWarmUp;
  const uint PlaygroundRoundNum;
  const uint PlaygroundRoundNumToPlay;
  const string GameStateName;
  void TeamProfilesAvoidHueOverlap();
  CGameTeamProfile* TeamProfile0;
  CGameTeamProfile* TeamProfile1;
  CGameTeamProfile* TeamProfile2;
  string ForcedClubLinkUrl1;
  string ForcedClubLinkUrl2;
  uint NextMaxPlayerCount;
  uint NextMaxSpectatorCount;
  UnnamedEnum NextLadderMode;
  enum class CGameCtnNetServerInfo::EVehicleNetQuality {
    Low = 0,
    High = 1,
  };
  CGameCtnNetServerInfo::EVehicleNetQuality NextVehicleNetQuality;
  string FirstPartySessionId;
};

// File extension: 'Frame.Gbx'
struct CGameControlCardCtnChallengeInfo : public CGameControlCard {
  CGameControlCardCtnChallengeInfo();

  const MwFastBuffer<CGameLeague*> Groups;
  uint LeagueNameMaxCharsCount;
  const wstring StrFullLeagueName;
  const wstring StrChallengeName;
  const wstring StrChallengeAuthor;
  const wstring StrChallengeComments;
  const wstring StrCollectionName;
  const wstring StrCopperPrice;
  const wstring StrPlayerName;
  const string StrCopperString;
  const wstring StrLeagueName;
  const uint PlayerLeagueRanking;
  CPlugBitmap* const BmpLeagueLogo;
  CPlugBitmap* const BmpMood;
  CPlugBitmap* const BmpMod;
  CPlugBitmap* const BmpBannerGrey;
  CPlugBitmap* const BmpBanner;
  void OnRemoveChallenge();
};

// File extension: 'Frame.Gbx'
struct CGameControlCardGeneric : public CGameControlCard {
  CGameControlCardGeneric();

  uint Type;
  string Str1;
  string Str2;
  string Str3;
  string Str4;
  string Str5;
  string Str6;
  string Str7;
  const wstring StrInt1;
  const wstring StrInt2;
  const wstring StrInt3;
  const wstring StrInt4;
  const wstring StrInt5;
  const wstring StrInt6;
  const wstring StrInt7;
  uint Nat1;
  uint Nat2;
  uint Nat3;
  uint Nat4;
  uint Nat5;
  uint Nat6;
  float Real1;
  float Real2;
  float Real3;
  CMwNod* Nod1;
  CMwNod* Nod2;
  CMwNod* Nod3;
};

// File extension: 'Frame.Gbx'
struct CGameControlCardLeague : public CGameControlCard {
  CGameControlCardLeague();

  const wstring StrPath;
  const wstring StrName;
  const string StrLogoUrl;
  CPlugBitmap* Logo;
};

// File extension: 'Frame.Gbx'
struct CGameControlCardCtnNetServerInfo : public CGameControlCard {
  CGameControlCardCtnNetServerInfo();

  const wstring StrPath;
  const wstring StrPlayerName;
  const wstring StrServerName;
  const wstring StrHostName;
  const wstring StrPureServerName;
  const wstring StrComment;
  const string StrPlayersCount;
  const string StrPlayersCountMax;
  const string StrSpectatorsCount;
  const string StrSpectatorsCountMax;
  const wstring StrLadderServerLimitMin;
  const wstring StrLadderServerLimitMax;
  const uint LadderMode;
  const uint VehicleQuality;
  const uint RaceType;
  const bool HaveBuddies;
  const bool IsBuddy;
  const bool IsFull;
  const bool IsFullSpectator;
  const bool IsAllowingDownload;
  const bool IsPrivate;
  const bool IsPrivateForSpectator;
  CPlugBitmap* Logo;
  const float PlayersCountRatio;
  const float SpectatorsCountRatio;
  const uint Level;
  const uint State;
  void OnChangeBuddyState();
  void OnChangeBuddyState_SetBuddy();
  void OnChangeBuddyState_SetNotBuddy();
  void OnChangeFavouriteState();
  void OnChangeFavouriteState_SetFavourite();
  void OnChangeFavouriteState_SetNotFavourite();
};

// File extension: 'GameCtnMediaBlockLightmap.gbx'
struct CGameCtnMediaBlockLightmap : public CGameCtnMediaBlock {
  CGameCtnMediaBlockLightmap();

};

// File extension: 'Frame.Gbx'
struct CGameControlCardLadderRanking : public CGameControlCard {
  CGameControlCardLadderRanking();

  const uint ChildsCount;
  const uint Ranking;
  const uint Ranking2;
  const uint Medals;
  const uint Medals2;
  const uint Score2;
  const wstring StrName;
  const wstring StrPath;
  const wstring StrScore;
  const float ManiaStarsRatio;
  const string StrLogin;
  const string StrLogoUrl;
  const string StrSubGroupLogoUrl;
  CPlugBitmap* Logo;
  CPlugBitmap* SubGroupLogo;
  bool DoSelectionOnChildsCount;
  bool UseTop3Medals;
};

// File extension: 'Frame.Gbx'
struct CGameControlCardMessage : public CGameControlCard {
  CGameControlCardMessage();

  string StrSender;
  string StrReceiver;
  const string StrSendDate;
  wstring StrSubject;
  wstring StrMessage;
  uint Donation;
  void OnCheckLogin();
};

// userName: 'ActionFxVis'
struct CGameActionFxVis : public CMwNod {
};

struct NGameMgrMap_SBlockInstance {
};

// File extension: 'Frame.Gbx'
struct CGameControlCardProfile : public CGameControlCard {
  CGameControlCardProfile();

  const wstring StrLogin;
  wstring StrNickName;
  CPlugBitmap* Avatar;
  void OnChooseProfile();
  void OnRemoveProfile();
  void OnConnectProfile();
};

// File extension: 'Frame.Gbx'
struct CGameControlCardCtnReplayRecordInfo : public CGameControlCard {
  CGameControlCardCtnReplayRecordInfo();

  const wstring StrName;
  const uint Time;
  bool ShortName;
  void OnRemoveReplayRecord();
};

struct CGameCtnMenus : public CGameSwitcherModule {
  CGameMenu* Menus;
  CGameMenu* InGameDialogs;
  CGameMenu* Dialogs;
  CGameMenu* SystemDialogs;
  CGameManialinkScriptHandler* const ManialinkScriptHandlerMenus;
  string PasswordString;
  string PasswordString2;
  string PasswordString3;
  float RankingScrollDelay;
  float RankingScrollSpeed;
  string SelectedName;
  wstring SelectedNickname;
  wstring SelectedAvatarName;
  CGameCtnChallengeInfo* const SelectedChallengeInfo;
  const MwFastBuffer<CGameCtnChapter*> EnvironmentChapters;
  void Dialog_OnValidate_InGame();
  void Dialog_OnCancel_InGame();
  void Dialog_OnValidate_Normal();
  void Dialog_OnCancel_Normal();
  void DialogJoin_OnJoinAsSpectator();
  const MwFastBuffer<CMwNod*> DialogList_Nods;
  void DialogCardGrid_OnOk();
  void DialogCardGrid_OnCancel();
  void DialogCardGrid_OnRefresh();
  wstring DialogChooseSkin_Name;
  bool DialogChooseSkin_SaveAsDds;
  uint DialogChooseSkin_SelectIndex;
  uint DialogChooseSkin_SelectIndex_Foreground;
  uint DialogChooseSkin_LatestClickedIndex;
  CGameCtnArticle* DialogChooseSkin_Article;
  const MwFastBuffer<CControlListItem*> DialogChooseSkin_SkinsItem;
  string TempLogin;
  string TempPassword;
  bool CheckPassword;
  string TempEMail;
  void DialogConnect_OnRememberPassword();
  void DialogConnect_OnForgotPassword();
  void DialogConnect_OnConnect();
  void DialogConnect_OnCancel();
  void DialogMailAccount_OnOk();
  void DialogMailAccount_OnCancel();
  void DialogGrid_OnSelect();
  void DialogGrid_OnNextPage();
  void DialogGrid_OnPrevPage();
  void DialogGrid_OnOk();
  void DialogGrid_OnCancel();
  void DialogGrid_OnRemove();
  void DialogGrid_OnAdd();
  uint DialogGrid_CurPage;
  void DialogChoosePackDesc_OnNextPage();
  void DialogChoosePackDesc_OnPrevPage();
  void DialogChoosePackDesc_DrawCurPage();
  void DialogChoosePackDesc_OnAddItem();
  uint DialogChoosePackDesc_CurPage;
  void DialogConfirmRequest_OnOk();
  void DialogConfirmRequest_OnCancel();
  void DialogInputSettings();
  void DialogInputSettings_OnClose();
  void DialogInputSettings_OnDeviceSettings();
  void DialogInputSettings_OnPlayerInputs();
  void DialogInputSettings_OnStandardInputs();
  void DialogInputSettings_OnDeviceSettingsApply();
  bool DialogInputSettingsApplyOnlyToThisVehicle;
  void DialogInputSettings_OnBindingsUnbindKey();
  void DialogInputSettings_OnBindingsResetToDefaults();
  float DialogInputSettings_AnalogDeadZone; // Range: 0 - 0.9
  float DialogInputSettings_AnalogSensitivity; // Range: 1 - 10
  float DialogInputSettings_RumbleIntensity; // Range: 0 - 2
  float DialogInputSettings_CenterSpringIntensity; // Range: 0 - 1
  bool DialogInputSettings_MouseLookInvertY;
  UnnamedEnum DialogInputSettings_MouseReleaseKey;
  bool DialogInputSettings_MouseAccel;
  float DialogInputSettings_MouseScaleY; // Range: 0.1 - 3
  float DialogInputSettings_MouseScaleFreeLook; // Range: 0.1 - 3
  float DialogInputSettings_MouseAccelQuantity_NormalizedLog; // Range: 0 - 3
  bool DialogInputSettings_MouseSensitivities_EnableSpecific;
  float DialogInputSettings_MouseSensitivity_Default_Normalized; // Range: 0.1 - 10
  float DialogInputSettings_MouseSensitivity_Laser_Normalized; // Range: 0.1 - 10
  float DialogInputSettings_MouseSensitivity_Default_NormalizedLog; // Range: -1 - 1
  float DialogInputSettings_MouseSensitivity_Laser_NormalizedLog; // Range: -1 - 1
  void DialogInterfaceSettings_OnOk();
  void DialogInterfaceSettings_OnCancel();
  void DialogInterfaceSettings_OnCrosshairNameSelect();
  void DialogInterfaceSettings_OnCrosshairNameSelect_Yes();
  void DialogInterfaceSettings_OnCrosshairNameUseDefault();
  bool DialogInterfaceSettings_CrosshairCustomColor;
  float DialogInterfaceSettings_CrosshairSaturation; // Range: 0 - 1
  float DialogInterfaceSettings_CrosshairSize; // Range: 0.1 - 2
  float DialogInterfaceSettings_CrosshairOpacity; // Range: 0.1 - 1
  const wstring DialogInterfaceSettings_CrosshairName;
  bool DialogInterfaceSettings_AllyCustomColor;
  bool DialogInterfaceSettings_OppoCustomColor;
  float DialogInterfaceSettings_PlayerShieldScale; // Range: 0.1 - 2
  float DialogInterfaceSettings_BeaconOpacity; // Range: 0 - 1
  float DialogInterfaceSettings_BeaconSize; // Range: 0.1 - 5
  float DialogInterfaceSettings_BeaconDuration; // Range: 0.5 - 15
  bool DialogInterfaceSettings_BeaconAllowCustom;
  void DialogReportAbuse_OnOk();
  void DialogReportAbuse_OnCancel();
  bool DialogReportAbuse_UploadReplay;
  wstring DialogReportAbuse_Reason;
  void DialogStereoscopySettings_OnQuit();
  void DialogStereoscopySettings_Enable();
  void DialogStereoscopySettings_Disable();
  void DialogStereoscopySettings_DefaultValues();
  float DialogStereoscopySettings_DisplaySize_cm;
  float DialogStereoscopySettings_DisplaySize_inch;
  float DialogStereoscopySettings_Strength; // Range: 0 - 1
  uint DialogStereoscopySettings_StrengthPercent; // Range: 0 - 100
  float DialogStereoscopySettings_ScreenDist; // Range: 1 - 10
  float DialogStereoscopySettings_ColorFactor; // Range: 0 - 1
  UnnamedEnum DialogStereoscopySettings_Mode;
  bool DialogStereoscopySettings_HmdResetOnRaceReset;
  bool DialogStereoscopySettings_HmdGraphicOptions;
  float DialogStereoscopySettings_HmdWorldScale;
  void DialogStereoscopySettings_HmdDetect();
  void DialogChooseEnvironment_OnCancel();
  void DialogEditorMenu();
  void DialogEditorHelp_OnHelpers();
  void DialogEditorMenu_OnShootCollectorIcons();
  void DialogEditorMenu_OnSaveAllBlocks();
  void DialogEditorMenu_OnAutoAddTopBottomFreeClipsForAllBlockInfos();
  void DialogEditorMenu_OnAutoRemoveTopBottomFreeClipsForAllBlockInfos();
  void DialogEditorMenu_OnReturn();
  void DialogEditorAdditionalMenu();
  void DialogEditorAdditionalMenu_OnEditComments();
  void DialogEditorAdditionalMenu_OnChallengeType();
  void DialogEditorAdditionalMenu_OnPlayMap();
  void DialogEditorAdditionalMenu_OnObjectives();
  void DialogEditorAdditionalMenu_OnEditSnapCamera();
  void DialogEditorAdditionalMenu_OnSetDifficulty();
  void DialogEditorAdditionalMenu_OnChooseCustomMusic();
  void DialogEditorAdditionalMenu_OnComputeShadows();
  void DialogEditorAdditionalMenu_OnSetPassword();
  void DialogEditorAdditionalMenu_OnUnlockExperimentalFeatures();
  void DialogEditorAdditionalMenu_OnReturn();
  void DialogEditorAnimAdditionalMenu();
  void DialogEditorAnimAdditionalMenu_OnTestActionWithMode();
  void DialogEditorAnimAdditionalMenu_OnDefineMap();
  void DialogEditorAnimAdditionalMenu_OnReturn();
  void DialogLightSettings_SetMoodSunrise();
  void DialogLightSettings_SetMoodDay();
  void DialogLightSettings_SetMoodSunset();
  void DialogLightSettings_SetMoodNight();
  void DialogLightSettings_PrevShadowsQuality();
  void DialogLightSettings_NextShadowsQuality();
  void DialogLightSettings_ComputeShadows();
  bool DialogComputeShadowsQuality_CheckSaveBounces;
  void DialogEditorHelp();
  void DialogEditorHelp_OnOk();
  void DialogActionMakerHelp();
  void DialogActionMakerHelp_OnOk();
  void DialogItemEditorHelp();
  void DialogItemEditorHelp_OnOk();
  void DialogEditorManialinkHelp();
  void DialogEditorManialinkHelp_OnOk();
  void DialogEditorManialinkOptions();
  void DialogEditorManialinkOptions_OnOk();
  void DialogEditorManialinkOptions_OnCancel();
  bool DialogEditorManialinkOptions_ButtonGrid;
  bool DialogEditorManialinkOptions_ButtonSnap;
  wstring DialogEditorManialinkOptions_GridStep;
  wstring DialogEditorManialinkOptions_RotationStep;
  void DialogEditorShiftTrack_OnDown();
  void DialogEditorShiftTrack_OnUp();
  void DialogEditorShiftTrack_OnLeft();
  void DialogEditorShiftTrack_OnRight();
  void DialogEditorShiftTrack_OnCancel();
  void DialogEditorShiftTrack_OnApply();
  int LastShiftX;
  int LastShiftZ;
  void DialogAskPassword();
  void DialogAskPassword_OnOk();
  void DialogAskPassword_OnCancel();
  wstring DialogAskPassword_String;
  void DialogDisplayChallengeCard();
  void DialogDisplayChallengeCard_OnOk();
  CPlugBitmap* ModBitmap;
  void DialogDisplayAnimCard();
  void DialogDisplayAnimCard_OnOk();
  void MenuMain();
  void MenuMain_OnQuit();
  void MenuEditors();
  void MenuEditors_OnLoadChallenge();
  void MenuEditors_OnLoadChallenge_OnSimple();
  void MenuEditors_OnLoadChallenge_OnAdvanced();
  void MenuTitleTools();
  void MenuCreateChallenge();
  void MenuCreateChallenge_OnSimple();
  void MenuCreateChallenge_OnAdvanced();
  void MenuLoadSimpleChallenge_OnSimple();
  void MenuLoadSimpleChallenge_OnAdvanced();
  void MenuChooseChallenge_OnSelect();
  void MenuChooseChallenge_OnBack();
  void MenuChooseChallenge_OnRefresh();
  void MenuChooseChallenge_OnOk();
  void MenuChooseChallenge_OnPathUp();
  void MenuChooseChallenge_OnOpenInExplorer();
  void MenuChooseChallenge_OnChangeLeague();
  void MenuChooseChallenge_OnChallengeRemoved();
  void MenuChooseChallenge_OnChallengeRemovedConfirmed();
  void MenuChooseChallenge_SelectOrUnselectAll();
  const uint MenuChooseChallenge_ChallengesCount;
  const bool MenuSoloLadderBoth_ShowSkillPointsOrElseMedals;
  bool MenuChooseChallenge_Flatten;
  bool MenuChooseChallenge_SortByName;
  bool MenuChooseChallenge_SortOrderAsc;
  bool SelectAll;
  const wstring HierarchyPath;
  wstring MenuChooseChallenge_FilterString;
  void MenuMultiLocal();
  void MenuProfileAdvanced();
  void MenuProfile();
  void MenuProfile_Launch();
  void MenuProfile_OnChooseAvatar();
  void MenuProfile_OnChangeHorn();
  void MenuProfile_OnReceiveNews();
  void MenuProfile_OnGroupSelected();
  void MenuProfile_OnAddGroup();
  void MenuProfile_OnChangeZone();
  void MenuProfile_OnConvertAccount();
  void MenuProfile_OnDisconnectAccount();
  void MenuProfile_OnDisconnectAccountConfirmed();
  void MenuProfile_OnCheckModifications();
  void MenuProfile_OnChangePass();
  void MenuProfile_OnShowGameKeyInSteam();
  void MenuProfile_OnEditPodiumAnimWin();
  void MenuProfile_OnEditPodiumAnimLose();
  float MenuProfile_SoundPitch; // Range: 0.5 - 2
  bool MenuProfile_ParentalLock;
  CGameAvatar* const MenuProfile_Avatar;
  CGameLeague* const Menu_GroupToManage;
  const uint MenuProfile_GroupsCount;
  wstring MenuProfile_HornName;
  void DialogManageGroup_Create();
  void DialogManageGroup_ViewPlayers();
  void DialogManageGroup_Invite();
  void DialogManageGroup_Unsubscribe();
  void DialogManageGroup_SetDefault();
  void DialogViewPlayersInGroup_Ok();
  void DialogInviteToGroup_Add();
  void DialogInviteToGroup_Cancel();
  wstring DialogManageGroup_Name;
  string DialogManageGroup_Login;
  string DialogManageGroup_Pass;
  string DialogManageGroup_PassConfirm;
  void DialogRegisterAccountChoice_CreateNewAccount();
  void DialogRegisterAccountChoice_UseExistingAccount();
  void DialogRegisterAccountChoice_Offline();
  string DialogEula_AgeString;
  uint DialogEula_Age;
  bool DialogEula_EulaValidated;
  bool DialogEula_PrivacyPolicyValidated;
  void DialogEula_OnOk();
  void DialogEula_OnCancel();
  void DialogOnlineAccount_OnCancel();
  void DialogOnlineAccount_OnOk();
  void DialogOnlineAccount_OnCheckLogin();
  void DialogOnlineAccountInfo_OnCopyToClipboard();
  void DialogOnlineAccountError_OnOk();
  void DialogOnlineAccountError_OnCancel();
  void DialogOnlineAccountError_OnMailAccount();
  void DialogOnlineAccount_OnRememberPassword();
  void DialogOnlineAccountPersonnal_OnReceiveNews();
  bool DialogOnlineAccount_ChooseLogin1;
  bool DialogOnlineAccount_ChooseLogin2;
  bool DialogOnlineAccount_ChooseLogin3;
  CGameLeague* const DialogOnlineAccount_SubscriptionLeague;
  void DialogOnlineAccountPersonnal_OnCancel();
  void DialogOnlineAccountPersonnal_OnOk();
  void DialogConvertAccount_OnConvert();
  void DialogConvertAccount_OnCancel();
  string ConvertAccount_Key;
  string OnlineAccount_Key;
  string OnlineAccount_Login;
  string OnlineAccount_Pass;
  string OnlineAccount_PassConfirm;
  string OnlineAccount_PassNew;
  uint OnlineAccount_Age;
  string OnlineAccount_EMail;
  string OnlineAccount_EMailConfirm;
  wstring OnlineAccount_NickName;
  wstring OnlineAccount_Path;
  CGameLeague* OnlineAccount_NewLeague;
  bool OnlineAccount_AcceptNews;
  string DialogAccountChina_AccountId;
  string DialogAccountChina_VerificationCode;
  string DialogAccountChina_IDNumber;
  wstring DialogAccountChina_Name;
  bool DialogAccountChina_LegalAccepted;
  void DialogAccountChina_AgreeToPrivacy();
  void DialogAccountChina_OnRequestSendCode();
  void DialogAccountChina_OnOk();
  void DialogResetPassword_OnRequestSendCode();
  void DialogResetPassword_OnOk();
  void DialogBuyTitlePackages_OnPay();
  void DialogBuyTitlePackages_OnEnterKey();
  void DialogBuyTitlePackages_OnBuyAndEnterKey();
  void DialogBuyTitlePackages_OnBuyKeyAndRefresh();
  void DialogBuyTitlePackages_OnBuyKeyAndQuit();
  void DialogBuyTitlePackages_OnCancel();
  void DialogBuyManiaPlanetStations_OnEnterKey();
  void DialogBuyManiaPlanetStations_OnPay();
  void DialogBuyManiaPlanetStations_OnCancel();
  void DialogGeneratedGameKeys_OnCopyToClipboard();
  const MwFastBuffer<CControlListItem*> DialogPlayers;
  const MwFastBuffer<CControlListItem*> ConfirmFiles;
  MwFastBuffer<CGameCtnReplayRecordInfo*> ReplayInfos;
  void MenuMultiPlayerNetworkLan_OnRefresh();
  void MenuMultiPlayerNetworkLan_OnSel();
  void MenuMultiPlayerNetworkLan_OnCreate();
  void MenuMultiPlayerNetworkLan_OnBack();
  const uint MenuMultiPlayerNetworkLan_ServersCount;
  const uint MenuMultiPlayerNetworkLan_PlayersCount;
  CGameCtnNetServerInfo* MenuMultiPlayerNetworkCreate_ServerInfo;
  void MenuMultiPlayerNetworkCreate();
  void MenuMultiPlayerNetworkCreate_OnStart();
  void MenuMultiPlayerNetworkCreate_OnBack();
  void MenuMultiPlayerNetworkCreate_OnAdvanced();
  void MenuMultiPlayerNetworkCreate_OnLoadSettings();
  void MenuMultiPlayerNetworkCreate_OnLoadSettings_OnYes();
  wstring DialogRemoteBrowser_ToFind;
  void DialogRemoteBrowser_OnClose();
  void DialogRemoteBrowser_OnFind();
  void MenuInternetLeague_OnAll();
  void MenuInternetLeague_OnFavorites();
  void MenuInternetLeague_OnRankings();
  void MenuInternetLeague_OnCreate();
  void MenuInternetLeague_OnBack();
  void MenuInternetLeague_OnRefresh();
  void MenuInternetLeague_OnRefreshSimple();
  void MenuInternetLeague_OnHierarchyUp();
  void MenuInternetLeague_OnHierarchyItemSelected();
  void MenuInternetLeague_OnServerSelected();
  void MenuInternetLeague_OnPlayerSelected();
  void MenuInternetLeague_BrowseServers();
  void MenuInternetLeague_BrowsePlayers();
  void MenuInternetLeague_OnChangeBuddyState();
  void MenuInternetLeague_OnLadderHierarchyUp();
  void MenuInternetLeague_SwitchServersPlayers();
  void MenuInternetLeague_SwitchLeaguesPlayersLadder();
  wstring MenuInternetLeague_Today;
  wstring MenuInternet_Path;
  const wstring MenuInternet_LadderCurrentLeagueName;
  const wstring MenuInternetLeague_CurrentLeagueName;
  const wstring MenuInternetLeague_CurrentLeagueDescription;
  void MenuHotSeatCreate();
  void MenuHotSeatCreate_Start();
  void MenuHotSeatCreate_LoadSettings();
  void MenuHotSeatCreate_OnBack();
  void MenuHotSeatCreate_OnOk();
  uint HotSeatTime;
  uint HotSeatTries;
  UnnamedEnum HotSeatGameMode;
  void MenuSplitScreenControls_OnScreen1();
  void MenuSplitScreenControls_OnScreen2();
  void MenuSplitScreenControls_OnScreen3();
  void MenuSplitScreenControls_OnScreen4();
  bool SplitScreen_2Players;
  bool SplitScreen_3Players;
  bool SplitScreen_4Players;
  bool SplitScreen_ScoreLeft;
  bool SplitScreen_ScoreMiddle;
  bool SplitScreen_ScoreRight;
  bool SplitScreen_SplitVertical;
  bool SplitScreen_SplitHorizontal;
  uint MenuSplitScreenMode_CompetitionPoints;
  void DialogChooseInputDevice_OnCancel();
  void DialogChooseInputDevice_OnCardSelected();
  void DialogChooseInputDevice_OnSet();
  void DialogSetInputDevice_OnOk();
  void MenuReplay_OnBack();
  void MenuReplay_OnRefresh();
  void MenuReplay_OnSelectDir();
  void MenuReplay_OnSelectReplay();
  void MenuReplay_OnOk();
  void MenuReplay_OnSelectAll();
  void MenuReplay_FilterAndRedraw();
  void MenuReplay_OnPathUp();
  void MenuReplay_OnOpenInExplorer();
  bool MenuReplay_Flatten;
  bool MenuReplay_SortByName;
  bool MenuReplay_SortOrderAsc;
  void MenuReplay_OnReplayRemovedConfirmed();
  const wstring MenuReplay_CurPath;
  const wstring MenuReplay_CurPathToDisplay;
  const uint MenuReplay_ReplaysCount;
  const MwFastBuffer<CGameCtnReplayRecordInfo*> ReplayList;
  const MwFastBuffer<CGameCtnReplayRecordInfo*> ReplayDirsList;
  void DialogReplayLoaded_OnPlayAgainst();
  void DialogReplayLoaded_OnWatch();
  void DialogReplayLoaded_OnEdit();
  void DialogReplayLoaded_OnCancel();
  void DialogViewReplay_OnView();
  void DialogViewReplay_OnEdit();
  void DialogViewReplay_OnSimpleEdit();
  void DialogViewReplay_OnValidate();
  void DialogViewReplay_OnPlay();
  void DialogViewReplay_OnBench();
  void DialogViewReplay_OnShootVideo();
  void DialogViewReplay_OnConcatenate();
  void DialogViewReplay_OnBack();
  void DialogViewReplay_OnExportToValidate();
  void DialogViewReplay_OnExportChallengeAndReplay();
  const uint DialogViewReplay_ReplayTime;
  const wstring DialogViewReplay_ReplayAuthorLogin;
  const wstring DialogViewReplay_ReplayMapName;
  const wstring DialogViewReplay_ReplayScriptShortName;
  void DialogViewModule_OnHud();
  void DialogViewModule_OnInventory();
  void DialogViewModule_OnMinimap();
  void DialogViewModule_OnStore();
  void DialogViewModule_OnScoresTable();
  void DialogViewModule_OnMenuBase();
  void DialogViewModule_OnMenuPage();
  void DialogViewModule_OnBack();
  void DialogViewModule_OnChrono();
  void DialogViewModule_OnSpeedMeter();
  void DialogViewModule_OnPlayerState();
  void DialogViewModule_OnTeamState();
  void DialogViewModuleMenuComponent_OnLadderRankings();
  void DialogViewModuleMenuComponent_OnServerBrowser();
  void DialogViewModuleMenuComponent_OnMenuBrowser();
  void DialogViewModuleMenuComponent_OnBack();
  void DialogViewScoresTableColumn_OnAvatar();
  void DialogViewScoresTableColumn_OnPlayerName();
  void DialogViewScoresTableColumn_OnManiaStars();
  void DialogViewScoresTableColumn_OnTools();
  void DialogViewScoresTableColumn_OnTags();
  void DialogViewScoresTableColumn_OnSMPoints();
  void DialogViewScoresTableColumn_OnSMRoundPoints();
  void DialogViewScoresTableColumn_OnTMBestTime();
  void DialogViewScoresTableColumn_OnTMPrevTime();
  void DialogViewScoresTableColumn_OnTMBestLapTime();
  void DialogViewScoresTableColumn_OnTMStunts();
  void DialogViewScoresTableColumn_OnTMRespawns();
  void DialogViewScoresTableColumn_OnTMCheckpoints();
  void DialogViewScoresTableColumn_OnTMPoints();
  void DialogViewScoresTableColumn_OnTMPrevRaceDeltaPoints();
  void DialogViewScoresTableColumn_OnCustomReal();
  void DialogViewScoresTableColumn_OnCustomString();
  void DialogViewScoresTableColumn_OnCustomNatural();
  void DialogViewScoresTableColumn_OnCustomInteger();
  void DialogViewScoresTableColumn_OnCustomTime();
  void DialogViewScoresTableColumn_OnBack();
  void DialogViewModulePlayerStateComponent_OnWeaponGauge();
  void DialogViewModulePlayerStateComponent_OnArmorGauge();
  void DialogViewModulePlayerStateComponent_OnStaminaGauge();
  void DialogViewModulePlayerStateComponent_OnActionGauge();
  void DialogViewModulePlayerStateComponent_OnWeaponList();
  void DialogViewModulePlayerStateComponent_OnBack();
  void DialogViewPixelArtSize_OnBack();
  void DialogChooseEnumValue_OnBack();
  void InputsList_ResetToDefaults_OnYes();
  void InputsList_UnbindDevice_OnYes();
  const MwFastBuffer<CControlListItem*> InputsList_Actions;
  void MenuConfigureInputs();
  void MenuConfigureInputs_OnBack();
  void MenuConfigureInputs_SetDefaults();
  void MenuConfigureInputs_OnPlayerInputs();
  void MenuConfigureInputs_OnStandardInputs();
  void MenuConfigureInputs_OnSplitScreenInputs();
  void MenuConfigureInputs_OnUnbindKey();
  void DialogChooseProfile_OnAdd();
  void DialogChooseProfile_OnSelect();
  void DialogChooseProfile_OnCancel();
  bool DialogChooseProfile_PrivateSession;
  void DialogChooseProfileWelcome_OnConnect();
  void DialogChooseProfileWelcome_OnChangeAccount();
  void DialogChooseProfileWelcome_OnPrivateSession();
  void DialogChooseProfileWelcome_OnStayOffline();
  void DialogInGameMenu_OnAdvanced();
  void DialogInGameMenu_OnServerSettings();
  void DialogInGameMenu_OnResume();
  void DialogInGameMenu_Spectator();
  void DialogInGameMenu_Buddy();
  void DialogInGameMenu_Abuse();
  void DialogInGameMenu_Kick();
  void DialogInGameMenu_Ban();
  void DialogInGameMenu_OnQuit();
  void DialogInGameMenu_OnChangeTeam();
  void DialogInGameMenu_SwitchFavourite();
  void DialogInGameMenuAdvanced_OnPlayerProfile();
  void DialogInGameMenuAdvanced_OnCancel();
  void DialogInGameMenuAdvanced_OnInputSettings();
  void DialogInGameMenuAdvanced_OnInterfaceSettings();
  void DialogInGameMenuAdvanced_OnStereoscopySettings();
  void DialogInGameMenuAdvanced_OnSaveReplay();
  void DialogInGameMenuAdvanced_OnSavePrevReplay();
  void DialogInGameMenuAdvanced_OnSaveChallenge();
  void DialogInGameMenuAdvanced_OnVote();
  void DialogChangeTeam();
  void DialogChangeTeam_OnJoinTeam1();
  void DialogChangeTeam_OnJoinTeam2();
  void DialogChangeTeam_OnCancel();
  const MwFastBuffer<CControlListItem*> DialogChangeTeam_Players1;
  const MwFastBuffer<CControlListItem*> DialogChangeTeam_Players2;
  void DialogPlayerProfile_OnOk();
  void DialogPlayerProfile_OnPrevPlayer();
  void DialogPlayerProfile_OnNextPlayer();
  void DialogPlayerProfile_OnHorn();
  void DialogPlayerProfile_Buddy();
  void DialogPlayerProfile_Kick();
  void DialogPlayerProfile_Ban();
  void DialogPlayerProfile_Abuse();
  void DialogVote_OnVoteYes();
  void DialogVote_OnVoteNo();
  void DialogVote_OnCancel();
  void DialogInGameAskYesNo_Yes();
  void DialogInGameAskYesNo_No();
  CScene2d* const DialogPlayerProfile_VehicleOverlayScene;
  void DialogAskIncreaseCacheSize_OnYes();
  void DialogAskIncreaseCacheSize_OnNo();
  void DialogAskIncreaseCacheSize_OnNever();
  void DialogChooseLeague();
  void DialogChooseLeague_Clean();
  void DialogChooseLeague_UpdateThisLevel();
  void DialogChooseLeague_OnSelect();
  void DialogChooseLeague_OnOk();
  void DialogChooseLeague_OnCancel();
  const wstring DialogChooseLeague_CurrentPath;
  const wstring DialogChooseLeague_DisplayableCurrentPath;
  void DialogCreateProfile_OnOk();
  void DialogCreateProfile_OnCancel();
  void DialogCreateProfile_OnAvatar();
  void DialogChooseAvatar_OnCancel();
  void DialogChooseAvatar_OnAddAvatar();
  bool NeverAskAgain;
  const MwFastBuffer<CControlListItem*> DataInfos;
  bool DialogConnect_RememberOnlinePassword;
  bool DialogOnlineAccount_RememberOnlinePassword;
  bool DialogOnlineAccountPersonnal_ReceiveNews;
  void DialogAddOrInviteBuddy_OnAdd();
  CGameLeague* const DialogAddOrInviteBuddy_SelectedGroup;
  bool DialogAddOrInviteBuddy_InviteToJoinAGroup;
  void DialogAddOrInviteBuddy_OnMail();
  void DialogAddOrInviteBuddy_OnCancel();
  string DialogAddOrInviteBuddy_Login;
  string DialogAddOrInviteBuddy_EMail;
  void DialogSelectGroup_OnSelect();
  void DialogSelectGroup_CreateNew();
  void DialogSelectGroup_Cancel();
  uint DialogGraphicSettings_DisplayMode;
  const string DialogGraphicSettings_Resolution;
  uint DialogGraphicSettings_WindowSizeX;
  uint DialogGraphicSettings_WindowSizeY;
  bool DialogGraphicSettings_WindowBorder;
  UnnamedEnum DialogGraphicSettings_DisplayRatio;
  uint DialogGraphicSettings_Antialias;
  uint DialogGraphicSettings_Filter;
  uint DialogGraphicSettings_MaxFps;
  uint DialogGraphicSettings_GpuSync;
  uint DialogGraphicSettings_DisplaySync;
  uint DialogGraphicSettings_TripleBuffer;
  uint DialogGraphicSettings_WaterReflect;
  uint DialogGraphicSettings_CarReflect;
  uint DialogGraphicSettings_Bloom;
  uint DialogGraphicSettings_MotionBlur;
  float DialogGraphicSettings_MotionBlurIntens; // Range: 0 - 1
  uint DialogGraphicSettings_SSAO;
  uint DialogGraphicSettings_ScreenShotExt;
  void DialogGraphicSettings_OnApply();
  void DialogGraphicSettings_OnCancel();
  void DialogGraphicSettings_OnAdvanced();
  void DialogGraphicSettings_OnBroadcast();
  uint DialogBroadcast_Provider;
  string DialogBroadcast_Login;
  string DialogBroadcast_Password;
  uint DialogBroadcast_Resolution;
  uint DialogBroadcast_VideoMaxKbps;
  uint DialogBroadcast_VideoTargetFps;
  uint DialogBroadcast_VideoCpuUsage;
  bool DialogBroadcast_AudioEnable;
  float DialogBroadcast_AudioVolumePlayback; // Range: 0 - 1
  float DialogBroadcast_AudioVolumeRecorder; // Range: 0 - 1
  void DialogBroadcast_StartStop();
  const wstring DialogBroadcast_UserOutput;
  uint MenuProfile_TagsAdmin_CurTag;
  const uint MenuProfile_TagsAdmin_TagCount;
  bool MenuProfile_TagsAdmin_CurTagIsAvailableForConsultation;
  void DialogSetChallengeType();
  void DialogSetChallengeType_OnOk();
  void DialogSetChallengeType_ShowPickType();
  void DialogSetChallengeType_ShowPickStyle();
  void DialogSetChallengeType_TypePicked();
  void DialogSetChallengeType_StylePicked();
  uint ChallengeType_NbLaps;
  CGameCtnChallenge* const CurGameChallenge;
  void DialogSetDifficulty();
  void DialogSetDifficulty_OnOk();
  UnnamedEnum ChallengeDifficulty;
  void DialogUnlockExperimentalFeatures();
  void DialogUnlockExperimentalFeatures_OnOk();
  void DialogCreateObjectives();
  void DialogCreateObjectives_OnValidate();
  uint CreateObjectives_GoldTime;
  uint CreateObjectives_SilverTime;
  uint CreateObjectives_BronzeTime;
  void DialogEditCutScenes_OnIntroEdit();
  void DialogEditCutScenes_OnIntroRemove();
  void DialogEditCutScenes_OnPodiumEdit();
  void DialogEditCutScenes_OnPodiumRemove();
  void DialogEditCutScenes_OnInGameEdit();
  void DialogEditCutScenes_OnInGameRemove();
  void DialogEditCutScenes_OnEndRaceEdit();
  void DialogEditCutScenes_OnEndRaceRemove();
  void DialogEditCutScenes_OnAmbianceEdit();
  void DialogEditCutScenes_OnAmbianceRemove();
  void DialogEditCutScenes_OnRecordMediaTrackerGhost();
  CPlugCharPhySpecialProperty* DialogEditBlockSpecialProperty_Property;
  void DialogEditBlockSpecialProperty_OnOk();
  void DialogEditBlockSpecialProperty_OnCancel();
  void DialogEditBlockSpecialProperty_OnEditSettings();
  void DialogEditBlockSpecialProperty_OnCustom();
  void DialogEditBlockSpecialProperty_OnDefault();
  void DialogChooseMapTypeForEditorSimple_OnMapTypeChosen();
  void DialogChooseMapTypeForEditorSimple_OnCancel();
  void DialogUpdateFiles_Abort();
  const float DialogUpdateFiles_OverallProgress; // Range: 0 - 1
  const float DialogUpdateFiles_CurrentProgress; // Range: 0 - 1
  NGameMenuSkinChooser_SMgr* const MenuSkinChooser;
};

struct CGameLadderRankingCtnChallengeAchievement : public CGameLadderRanking {
};

struct CGameCtnNetForm : public CGameNetForm {
  CGameCtnNetForm();

};

struct CGameRemoteBufferDataInfoFinds : public CGameRemoteBufferDataInfo {
  uint RefreshPlayerDuration;
  uint RefreshServerDuration;
  uint CountPlayerPerPage;
  uint CountServerPerPage;
};

// Description: "Vehicle or Character settings."
struct CGameUserProfileWrapper_VehicleSettings : public CMwNod {
  const wstring ModelDisplayName; // Maniascript
  const string ModelName; // Maniascript
  const wstring SkinName; // Maniascript
  const string SkinUrl; // Maniascript
  float AnalogSensitivity; // Range: 0.1 - 10 // Maniascript
  float AnalogDeadZone; // Range: 0 - 0.9 // Maniascript
  bool AnalogSteerV2; // Maniascript
  bool InvertSteer; // Maniascript
  bool AccelIsToggleMode; // Maniascript
  bool BrakeIsToggleMode; // Maniascript
  float RumbleIntensity; // Range: 0 - 2 // Maniascript
  bool HapticFeedbackEnabled; // Maniascript
  float CenterSpringIntensity; // Range: 0 - 1 // Maniascript
};

struct CGameRemoteBufferDataInfoSearchs : public CGameRemoteBufferDataInfo {
  uint RefreshLeaguesDuration;
  uint RefreshServersDuration;
  uint RefreshServersSuggestedDuration;
  uint RefreshServersFavouritesDuration;
  uint RefreshPlayersDuration;
  uint LeaguesPerPageCount;
  uint ServersPerPageCount;
  uint ServersSuggestedPerPageCount;
  uint ServersFavouritesPerPageCount;
  uint PlayersPerPageCount;
};

struct CGameMgrActionFxPhy : public CMwNod {
  const uint InstancesCount;
};

// Description: "This is the stations Manialink interface."
struct CGameScriptHandlerStation : public CGameManialinkScriptHandler {
  CGameManiaPlanetScriptAPI* const ManiaPlanet; // Maniascript
  CGameManiaAppStation* const ParentApp; // Maniascript
  CGameStation* const Station; // Maniascript
  void EnterStation(); // Maniascript
  void BrowserReload(); // Maniascript
};

struct CGameCtnMasterServer : public CGameMasterServer {
  const bool IsTransfering;
};

struct CGameCtnNetwork : public CGameNetwork {
  const uint NextChallengeIndex;
  const bool Spectator;
  void GetFilesToSubmit();
  void GetManiaNetResource_OnCancel();
  CGameManiaAppPlayground* ClientManiaAppPlayground;
  CGameScriptHandlerPlaygroundInterface* PlaygroundInterfaceScriptHandler;
  CGamePlaygroundClientScriptAPI* PlaygroundClientScriptAPI;
  void RequestChangeSpectator_ToSpec();
  void RequestChangeSpectator_ToPlayer();
  CGameServerPlugin* ServerPlugin;
  CGameScriptServerAdmin* GameScriptServerAdminAPI;
};

// File extension: 'GameCtnApp.Gbx'
struct CGameCtnApp : public CGameApp {
  CGameCtnApp();

  CGameResources* Resources;
  CControlStyleSheet* DefaultStyleSheet;
  CGameCtnNetwork* Network;
  CGamePlaygroundScript* const PlaygroundScript;
  CGameSwitcher* const Switcher;
  CGameCtnChallenge* RootMap;
  UnnamedEnum VehicleCollectionId;
  wstring VehicleCollectionId_Text;
  ISceneVis* GameScene;
  CGameEditorBase* const EditorBase;
  CGameCtnEditor* const Editor;
  CGamePlayground* const CurrentPlayground;
  CGameCtnCatalog* const GlobalCatalog;
  const MwFastBuffer<CMwNod*> AdditionalSkinsFids;
  MwFastBuffer<CGameCtnChallengeInfo*> ChallengeInfos;
  MwFastBuffer<CGameCtnReplayRecordInfo*> ReplayRecordInfos;
  CGamePlayerInfo* const LocalPlayerInfo;
  CSystemConfig* const SystemConfig;
  CGameUserProfileWrapper* CurrentProfile;
  const MwFastBuffer<CGamePlayerProfile*> PlayerProfiles;
  MwFastBuffer<CGameCtnCampaign*> BaseOfficialCampaigns;
  MwFastBuffer<CGameCtnCampaign*> OfficialCampaigns;
  CGameCtnCampaign* const CurrentCampaign;
  bool MenuBackground_MustLoopIntro;
  const uint MessagesCount;
  const wstring Money;
  bool StereoscopyEnable;
  CGameScriptChatManager* const ChatManagerScript;
  NGameScriptChat_SManager* const ChatManagerScriptV2;
  CGameMatchSettingsManagerScript* const MatchSettingsManagerScript;
  CGameUserManagerScript* const UserManagerScript;
  CGameWebServicesNotificationManagerScript* const WSNotificationManagerScript;
};

// File extension: 'Frame.Gbx'
struct CGameControlCardCtnArticle : public CGameControlCard {
  CGameControlCardCtnArticle();

  CPlugBitmap* const Icon;
  const wstring StrName;
};

// Description: "Internal API for Maniaplanet."
struct CGameManiaPlanetScriptAPI : public CMwNod {
  enum class CGameManiaPlanetScriptAPI::EContext {
    MenuStartUp = 0,
    MenuManiaPlanet = 1,
    MenuManiaTitleMain = 2,
    MenuProfile = 3,
    MenuSolo = 4,
    MenuLocal = 5,
    MenuMulti = 6,
    MenuEditors = 7,
    EditorPainter = 8,
    EditorTrack = 9,
    EditorMediaTracker = 10,
    Solo = 11,
    SoloLoadScreen = 12,
    Multi = 13,
    MultiLoadScreen = 14,
    MenuCustom = 15,
    Unknown = 16,
  };
  enum class CGameManiaPlanetScriptAPI::ELinkType {
    ExternalBrowser = 0,
    ManialinkBrowser = 1,
  };
  enum class CGameManiaPlanetScriptAPI::EBuyTitleMode {
    OpenStore = 0,
    BuyIfNeeded = 1,
    Ask = 2,
  };
  enum class CGameManiaPlanetScriptAPI::EMenuStationsPage {
    Channels = 0,
    Play = 1,
    Options = 2,
    Help = 3,
  };
  enum class CGameManiaPlanetScriptAPI::ESystemPlatform {
    None = 0,
    Steam = 1,
    UPlay = 2,
    PS4 = 3,
    XBoxOne = 4,
    PS5 = 5,
    XBoxSeries = 6,
    Stadia = 7,
    Luna = 8,
  };
  enum class CGameManiaPlanetScriptAPI::ESystemSkuIdentifier {
    Unknown = 0,
    EU = 1,
    US = 2,
    JP = 3,
    CN = 4,
  };
  enum class CGameManiaPlanetScriptAPI::EHmdWearingState {
    Dismount = 0,
    Mount = 1,
    Unknown = 2,
  };
  enum class CGameManiaPlanetScriptAPI::EHmdTrackingState {
    NotStarted = 0,
    Calibrating = 1,
    NotTracking = 2,
    Tracking = 3,
  };
  enum class CGameManiaPlanetScriptAPI::EInputsListFilter {
    All = 0,
    OnlyGeneral = 1,
    OnlyPlayer = 2,
  };
  enum class CGameManiaPlanetScriptAPI::EPlayerInfoDisplayType {
    Name = 0,
    Avatar = 1,
    AvatarAndName = 2,
  };
  enum class CGameManiaPlanetScriptAPI::ENetworkSpeed {
    Custom = 0,
    _100Kbps = 1,
    _1Mbps = 2,
    _10Mbps = 3,
    _100Mbps = 4,
  };
  const int Now; // Maniascript
  const string CurrentLocalDateText; // Maniascript
  const wstring CurrentTimezone; // Maniascript
  CGamePlayerInfo* const LocalUser; // Maniascript
  CGameManiaTitle* const LoadedTitle; // Maniascript
  const MwFastBuffer<CGameManiaTitle*> TitlesAvailable; // Maniascript
  const MwFastBuffer<CGameManiaTitle*> TitlesBases; // Maniascript
  const MwFastBuffer<string> TitleIdsInstalled; // Maniascript
  const MwFastBuffer<string> TitleIdsPayed; // Maniascript
  const uint EmptyStationsCount; // Maniascript
  CSystemPlatformScript* const System; // Maniascript
  const CGameManiaPlanetScriptAPI::ESystemPlatform SystemPlatform; // Maniascript
  const CGameManiaPlanetScriptAPI::ESystemSkuIdentifier SystemSkuIdentifier; // Maniascript
  const string ExeVersion; // Maniascript
  CGameCtnChallenge* const CurrentMap; // Maniascript
  const CGameManiaPlanetScriptAPI::EContext ActiveContext; // Maniascript
  const string ActiveContext_MenuFrame; // Maniascript
  const string ActiveContext_InGameDialogFrame; // Maniascript
  const string ActiveContext_DialogFrame; // Maniascript
  const string ActiveContext_SystemDialogFrame; // Maniascript
  const bool ActiveContext_ClassicDialogDisplayed; // Maniascript
  const uint ActiveContext_GameWill; // Maniascript
  const bool ActiveContext_ManialinkBrowserDisplayed; // Maniascript
  const bool ActiveContext_MenuStationsDisplayed; // Maniascript
  const bool ActiveContext_InGameMenuDisplayed; // Maniascript
  const CGameManiaPlanetScriptAPI::EMenuStationsPage ActiveContext_MenuStationsPage; // Maniascript
  const CGameManiaPlanetScriptAPI::EMenuStationsPage ActiveContext_MenuStationsClickedPage; // Maniascript
  const bool ActiveContext_IsProfileEditable; // Maniascript
  const string CurrentServerLogin; // Maniascript
  const wstring CurrentServerName; // Maniascript
  const wstring CurrentServerModeName; // Maniascript
  const string CurrentServerJoinLink; // Maniascript
  const MwFastBuffer<CGamePlayerInfo*> CurrentServerPlayers; // Maniascript
  const uint CurrentAppId; // Maniascript
  bool IsPrivateSession; // Maniascript
  const bool IsKioskMode; // Maniascript
  const bool ParentalLock_IsLocked; // Maniascript
  const uint TotalTimePlay; // Maniascript
  const uint AddictionLimiter_TimeLeftToday; // Maniascript
  const uint Messenger_MessagesCount; // Maniascript
  const uint AccountPlanets; // Maniascript
  const bool IsTitleTimeLimited; // Maniascript
  const bool IsTitleTimeLimitActivated; // Maniascript
  const bool CanUnlockTitleTimeLimitOnStore; // Maniascript
  const uint TitleTimeLeft; // Maniascript
  const uint CurConnectionDate; // Maniascript
  const uint PrevConnectionDate; // Maniascript
  const bool TmTurbo_IsDemo; // Maniascript
  const bool TmTurbo_IsPartialInstall; // Maniascript
  const bool TmTurbo_IsBeta; // Maniascript
  const bool TmTurbo_IsSlowInstall; // Maniascript
  bool Settings_StationsManualLayout; // Maniascript
  const bool PersistentIsReady; // Maniascript
  void ClipboardSet(wstring ClipboardText); // Maniascript
  void OpenLink(string Url, CGameManiaPlanetScriptAPI::ELinkType LinkType); // Maniascript
  void QuitGameAndOpenLink(string Url); // Maniascript
  void InstallTitle(string TitleId); // Maniascript
  void InstallTitleFromUrl(string Url, wstring DisplayName); // Maniascript
  void InstallTitleFromUrlEx1(string Url, wstring DisplayName, bool EnterAfterInstall); // Maniascript
  void InstallTitleFromUrlEx2(string Url, wstring DisplayName, wstring OpenLinkAfterInstall); // Maniascript
  void ShowTitle(string TitleId, bool ShowStation); // Maniascript
  void SelectTitle(string TitleId); // Maniascript
  void EnterTitle(string TitleId); // Maniascript
  void ShowCurMapCard(); // Maniascript
  void SetMenuStationsActive(bool Activate); // Maniascript
  void Dialog_Message(wstring Message); // Maniascript
  void Dialog_EnterGameKey(); // Maniascript
  void Dialog_BuyTitleDialog(string TitleId, string OverrideBuyUrl, int OverrideActionAfterBuy); // Maniascript
  void Dialog_BuyTitleEx(string TitleId, CGameManiaPlanetScriptAPI::EBuyTitleMode Mode, string OverrideBuyUrl, int OverrideActionAfterBuy); // Maniascript
  void Dialog_GenerateGameKeys(string TitleId, uint FeaturesLevel, uint NbToGenerate, bool AddBuddyOnRegistration); // Maniascript
  void Dialog_BuyStations(); // Maniascript
  void Dialog_SetServerFavourite(string ServerLogin, bool Favorite, bool Silent); // Maniascript
  void Dialog_JoinServer(string ServerLogin, string ServerPassword, bool LAN, bool Spectate, string TitleId, bool Silent, bool ChannelServer); // Maniascript
  void Dialog_QuitGame(); // Maniascript
  void Dialog_ChangeAvatar(); // Maniascript
  void Dialog_ChangeHorn(); // Maniascript
  void Dialog_ChangeCrosshair(); // Maniascript
  void Dialog_ChangeZone(); // Maniascript
  void Dialog_ChangePassword(); // Maniascript
  void Dialog_CommitProfileChanges(); // Maniascript
  void Dialog_CleanCache(); // Maniascript
  void Dialog_BindInput(int ActionIndex, CInputScriptPad* Device); // Maniascript
  void Dialog_UnbindInputDevice(CInputScriptPad* Device); // Maniascript
  void Dialog_DefaultInputBindings(CInputScriptPad* Device); // Maniascript
  const bool Dialog_IsFinished; // Maniascript
  const bool Dialog_Success; // Maniascript
  const bool Dialog_Aborted; // Maniascript
  void CustomMenu_Request(wstring Type, MwFastBuffer<wstring>& Data); // Maniascript
  bool ParentalLock_Lock(string Password); // Maniascript
  bool ParentalLock_Unlock(string Password, bool UnlockOnce); // Maniascript
  void SetLocalUserClubLink(string ClubLink); // Maniascript
  void SetLocalUserNickName(wstring NickName); // Maniascript
  void SetLocalUserNickNameAndTrigram(wstring NickName, string Trigram); // Maniascript
  void FlashWindow(); // Maniascript
  void CreateShortcut(string ShortcutName, string Url); // Maniascript
  void HideSystemLoadingScreen(); // Maniascript
  bool IsLiveStreamingEnabled; // Maniascript
  bool IsRemotePlayEnabled; // Maniascript
  void TmTurbo_LoadingOpportunity(); // Maniascript
  void LogToSessionTrace(wstring Text); // Maniascript
  CGameScriptChatManager* const ChatManager; // Maniascript
  NGameScriptChat_SContext* const ChatContext; // Maniascript
  const string ServerChatLog; // Maniascript
  CGameUserManagerScript* const UserMgr; // Maniascript
  CGameUserManagerScript* const WSNotificationMgr; // Maniascript
  void Authentication_GetToken(MwId UserId, string AppLogin); // Maniascript
  const bool Authentication_GetTokenResponseReceived; // Maniascript
  const uint Authentication_ErrorCode; // Maniascript
  const string Authentication_Token; // Maniascript
  CWebServicesTaskResult_String* Authentication_GetToken_v2(MwId UserId, string Service); // Maniascript
  const MwFastBuffer<CWebServicesTaskResult*> MasterServer_MSTaskResults; // Maniascript
  void MasterServer_ReleaseMSTaskResult(MwId TaskId); // Maniascript
  CGameMasterServerUserInfo* const MasterServer_MainMSUser; // Maniascript
  const MwFastBuffer<CGameMasterServerUserInfo*> MasterServer_MSUsers; // Maniascript
  CWebServicesTaskResult_PlanetsTransaction_Bill* PlanetsTransaction_GivePlanets(string LoginPayee, uint Cost, wstring Label); // Maniascript
  CWebServicesTaskResult* BuyFullGame(MwId UserId); // Maniascript
  void HMD_Activate(bool Enable); // Maniascript
  const bool HMD_IsActive; // Maniascript
  const vec3 HMD_HeadTranslation; // Maniascript
  const float HMD_HeadYaw; // Maniascript
  const float HMD_HeadPitch; // Maniascript
  const float HMD_HeadRoll; // Maniascript
  void HMD_ResetTracking(); // Maniascript
  void HMD_SelectUser(); // Maniascript
  CGameUserScript* const HMD_CurrentUser; // Maniascript
  float HMD_WorldScale; // Maniascript
  float HMD_MoveScale; // Range: 1e-05 - 10000 // Maniascript
  float HMD_HUD_Distance; // Range: 0.2 - 100 // Maniascript
  const CGameManiaPlanetScriptAPI::EHmdWearingState HMD_WearingState; // Maniascript
  const CGameManiaPlanetScriptAPI::EHmdTrackingState HMD_TrackingState; // Maniascript
  const bool HMD_IsUserWarnBySystem_OutOfPlayArea; // Maniascript
  bool Cameras_BlackOut; // Maniascript
  void DisplaySettings_LoadCurrent(); // Maniascript
  void DisplaySettings_Unload(); // Maniascript
  void DisplaySettings_Apply(); // Maniascript
  CGameDisplaySettingsWrapper* const DisplaySettings; // Maniascript
  void AudioSettings_LoadCurrent(); // Maniascript
  void AudioSettings_Unload(); // Maniascript
  void AudioSettings_Apply(); // Maniascript
  CGameAudioSettingsWrapper* const AudioSettings; // Maniascript
  bool AudioSettings_EnableAudio; // Maniascript
  float AudioSettings_MasterSoundVolume; // Range: -40 - 0 // Maniascript
  float AudioSettings_SoundVolume_Scene; // Range: -40 - 0 // Maniascript
  float AudioSettings_SoundVolume_Ui; // Range: -40 - 0 // Maniascript
  float AudioSettings_MasterMusicVolume; // Range: -40 - 0 // Maniascript
  bool AudioSettings_MuteWhenAppUnfocused; // Maniascript
  bool AudioSettings_DontMuteWhenAppUnfocused; // Maniascript
  float MasterSoundVolume; // Range: -40 - 0 // Maniascript
  float MasterMusicVolume; // Range: -40 - 0 // Maniascript
  const string Config_GameScriptConfigJson; // Maniascript
  void InputBindings_UpdateList(CGameManiaPlanetScriptAPI::EInputsListFilter Filter, CInputScriptPad* Device); // Maniascript
  const MwFastBuffer<wstring> InputBindings_ActionNames; // Maniascript
  const MwFastBuffer<wstring> InputBindings_Bindings; // Maniascript
  const MwFastBuffer<string> InputBindings_BindingsRaw; // Maniascript
  const uint InputBindings_PlayerInputsCount; // Maniascript
  const MwFastBuffer<string> Language_AvailableChoices; // Maniascript
  const MwFastBuffer<wstring> Language_AvailableChoices_Names; // Maniascript
  const string Language_Current; // Maniascript
  string Language_NextStart; // Maniascript
  void Language_Apply(); // Maniascript
  CGameManiaPlanetScriptAPI::ENetworkSpeed NetworkSpeed; // Maniascript
  uint NetworkSpeed_CustomDownload; // Maniascript
  uint NetworkSpeed_CustomUpload; // Maniascript
  const uint NetworkSpeed_Download; // Maniascript
  const uint NetworkSpeed_Upload; // Maniascript
  bool Network_EnableMapModLocators; // Maniascript
  bool Network_EnableMapSkinLocators; // Maniascript
  bool Network_EnableVehicleSkinLocators; // Maniascript
  CGameManiaPlanetScriptAPI::EPlayerInfoDisplayType Interface_PlayerInfoDisplayType; // Maniascript
  int Interface_PlayerInfoDisplaySize; // Range: 1 - 5 // Maniascript
  bool Interface_PlayerInfoHidden; // Maniascript
  const bool SkipIntroAndEpilepsyWarning; // Maniascript
  bool SystemSettings_SkipIntro; // Maniascript
  const bool SystemSettings_SkipIntro_ReadOnly; // Maniascript
  CGameDirectLinkScript* const DirectLink; // Maniascript
  void DirectLink_Clear(); // Maniascript
  const CGameManiaPlanetScriptAPI::ENetworkSpeed NetworkSpeed_Current; // Maniascript
  CGameManiaPlanetScriptAPI::ENetworkSpeed NetworkSpeed_NextStart; // Maniascript
  const uint NetworkSpeed_CustomDownload_Current; // Maniascript
  uint NetworkSpeed_CustomDownload_NextStart; // Maniascript
  const uint NetworkSpeed_CustomUpload_Current; // Maniascript
  uint NetworkSpeed_CustomUpload_NextStart; // Maniascript
  const uint NetworkSpeed_Download_NextStart; // Maniascript
  const uint NetworkSpeed_Upload_NextStart; // Maniascript
};

// File extension: 'Frame.Gbx'
struct CGameControlCardCtnChapter : public CGameControlCard {
  CGameControlCardCtnChapter();

  CPlugBitmap* const Icon;
  CPlugBitmap* const BannerChallenge;
  const wstring StrName;
};

// File extension: 'Frame.Gbx'
struct CGameControlCardCtnGhostInfo : public CGameControlCard {
  CGameControlCardCtnGhostInfo();

  const wstring StrName;
  const uint Time;
};

// File extension: 'Frame.Gbx'
struct CGameControlCardCtnVehicle : public CGameControlCard {
  CGameControlCardCtnVehicle();

  CPlugBitmap* const Icon;
  const wstring StrName;
};

// File extension: 'ScriptWorkspace.Gbx'
struct CGameScriptDebuggerWorkspace : public CMwNod {
  CGameScriptDebuggerWorkspace();

};

struct CGameAnalyzer : public CMwNod {
  const int Mode;
  CMwStatsValue* const FrameRate;
};

struct CGamePlaygroundInterface : public CMwNod {
  CControlContainer* InterfaceRoot;
  NGameHud3d_SMgr* const Hud3d;
  CGameCtnNetServerInfo* const ServerInfo;
  CGameCtnChallenge* const Challenge;
  wstring ChatEntry;
  void ChatNextPage();
  void ChatPreviousPage();
  void ChatToggleDisplayMode();
  bool ChatRestrictionMessageDisplayedForAllUsersEnabled;
  CControlFrame* const ManialinkPage;
  CGameScriptHandlerPlaygroundInterface* const ManialinkScriptHandler;
  CGameManialinkScriptHandler* const ManialinkScriptHandlerBasic;
  void ShowChallengeCard();
  void ShowModeHelp();
  void CopyServerLink();
};

struct CGamePlaygroundSpectating : public CMwNod {
};

struct CWebServicesTaskResult_AccountTrophyGainList : public CWebServicesTaskResult {
};

// File extension: 'GameCtnMediaBlockGhostOld.gbx'
struct CGameCtnMediaBlockGhostTM : public CGameCtnMediaBlock {
  CGameCtnGhost* GhostModel;
  float StartOffset;
};

struct CGameEnvironmentManager : public CMwNod {
};

struct CGameDialogShootParams : public CMwNod {
  float Duration;
  uint VideoFps;
  uint Width;
  uint Height;
  uint TileCountX;
  uint TileCountY;
  bool TileSplit;
  bool VideoHq;
  UnnamedEnum VideoHqSampleCount;
  bool Sharpen;
  float Remanence;
  UnnamedEnum MotionBlur;
  bool VideoHqDOF;
  UnnamedEnum DofSampleCount;
  bool VideoHqCarReflects;
  bool VideoReflectSubSample;
  bool VideoReflectRayCast;
  UnnamedEnum VideoStereo3d;
  bool IsAudioStream;
  UnnamedEnum ExtScreen;
  UnnamedEnum ExtVideo;
  const string EstimatedTime;
  string ShootName;
  bool Hud3d;
  UnnamedEnum WebmVideoMode;
  UnnamedEnum WebmVideoAutoBirate;
  uint WebmVideoBitrate;
  uint WebmVideoCQ_Level;
  float WebmAudioVBR_Quality;
  void OnOk();
  void OnCancel();
  void OnAdvanced();
  void SetQualityPreset_Low();
  void SetQualityPreset_Medium();
  void SetQualityPreset_High();
};

// Description: "Manialink filename entry."
struct CGameManialinkFileEntry : public CGameManialinkEntry {
  void OnFileChoosen();
  bool OnlyUserFiles; // Maniascript
  void ClearFileNames(); // Maniascript
  const wstring FullFileName; // Maniascript
  const wstring RelativeNameToRoot; // Maniascript
  const wstring RelativeNameToTypeFolder; // Maniascript
  const wstring RelativeFileName; // Maniascript
};

struct CGameNetDataDownload : public CMwNod {
  CGameNetDataDownload();

  const uint ReturnedError;
  const bool CheckUpToDate;
  const bool PauseOnError;
  const bool IsPaused;
  const bool IsFinished;
};

struct CGameCtnMediaBlockSpectators : public CGameCtnMediaBlock {
  CGameCtnMediaBlockSpectators();

};

struct CGameNetFormBuddy : public CNetNod {
  CGameNetFormBuddy();

};

// Description: "API for GameModes Manialinks"
struct CGameScriptHandlerPlaygroundInterface : public CGameManialinkScriptHandler {
  CGameScriptHandlerPlaygroundInterface();

  enum class CGameScriptHandlerPlaygroundInterface::EInGameMenuResult {
    Resume = 0,
    Quit = 1,
    NormalMenu = 2,
    AdvancedMenu = 3,
    ServerSettings = 4,
  };
  enum class CGameScriptHandlerPlaygroundInterface::EUISound {
    Default = 0,
    Silence = 1,
    StartMatch = 2,
    EndMatch = 3,
    StartRound = 4,
    EndRound = 5,
    PhaseChange = 6,
    TieBreakPoint = 7,
    TiePoint = 8,
    VictoryPoint = 9,
    Capture = 10,
    TimeOut = 11,
    Notice = 12,
    Warning = 13,
    PlayerEliminated = 14,
    PlayerHit = 15,
    Checkpoint = 16,
    Finish = 17,
    Record = 18,
    ScoreProgress = 19,
    RankChange = 20,
    Bonus = 21,
    FirstHit = 22,
    Combo = 23,
    PlayersRemaining = 24,
    Custom1 = 25,
    Custom2 = 26,
    Custom3 = 27,
    Custom4 = 28,
  };
  uint GameTime; // Maniascript
  CGamePlaygroundClientScriptAPI* const Playground; // Maniascript
  CGameManiaPlanetScriptAPI* const ManiaPlanet; // Maniascript
  CGamePlaygroundUIConfig* const UI; // Maniascript
  CGamePlaygroundUIConfig* const ClientUI; // Maniascript
  bool IsSpectator; // Maniascript
  bool IsSpectatorClient; // Maniascript
  bool UseClans; // Maniascript
  bool UseForcedClans; // Maniascript
  CGameManiaAppPlaygroundCommon* const ParentApp; // Maniascript
  CGameCtnChallenge* const Map; // Maniascript
  void ShowCurChallengeCard(); // Maniascript
  void ShowModeHelp(); // Maniascript
  void CopyServerLinkToClipBoard(); // Maniascript
  void JoinTeam1(); // Maniascript
  void JoinTeam2(); // Maniascript
  const MwFastBuffer<CGameTeamProfile*> Teams; // Maniascript
  void RequestSpectatorClient(bool Spectator); // Maniascript
  void SetSpectateTarget(string Player); // Maniascript
  void ShowProfile(string Player); // Maniascript
  void ShowInGameMenu(); // Maniascript
  void CloseInGameMenu(CGameScriptHandlerPlaygroundInterface::EInGameMenuResult Result); // Maniascript
  void CloseScoresTable(); // Maniascript
  bool IsInGameMenuDisplayed; // Maniascript
  string CurrentServerLogin; // Maniascript
  wstring CurrentServerName; // Maniascript
  wstring CurrentServerDesc; // Maniascript
  string CurrentServerJoinLink; // Maniascript
  wstring CurrentServerModeName; // Maniascript
  uint SplitScreenNum; // Maniascript
  uint SplitScreenCount; // Maniascript
  bool SplitScreenIsHorizontal; // Maniascript
  void PlayUiSound(CGameScriptHandlerPlaygroundInterface::EUISound Sound, int SoundVariant, float Volume); // Maniascript
  void Spectate(string Player); // Maniascript
  bool IsTeamMode_deprec_; // Maniascript
  bool IsForcedTeams_deprec_; // Maniascript
  bool IsSpectatorMode; // Maniascript
  CGamePlayerInfo* const LocalPlayerInfo_deprec_; // Maniascript
};

// Description: "Station mania app."
struct CGameManiaAppStation : public CGameManiaAppMinimal {
  CGameStation* const Station; // Maniascript
  void EnterStation(); // Maniascript
  void Maker_EditTitle(string EditedTitleId); // Maniascript
  void Maker_EditNewTitle(wstring EditedTitleName); // Maniascript
  const MwFastBuffer<CGamePackCreator_TitleInfoScript*> Maker_EditedTitles; // Maniascript
};

// Description: "This is the base Manialink page interface."
struct CGameManialinkScriptHandler : public CMwNod {
  enum class CGameManialinkScriptHandler::ELinkType {
    ExternalBrowser = 0,
    ManialinkBrowser = 1,
    Goto = 2,
    ExternalFromId = 3,
    ManialinkFromId = 4,
    GotoFromId = 5,
  };
  enum class CGameManialinkScriptHandler::ESystemPlatform {
    None = 0,
    Steam = 1,
    UPlay = 2,
    PS4 = 3,
    XBoxOne = 4,
    PS5 = 5,
    XBoxSeries = 6,
    Stadia = 7,
    Luna = 8,
  };
  enum class CGameManialinkScriptHandler::ESystemSkuIdentifier {
    Unknown = 0,
    EU = 1,
    US = 2,
    JP = 3,
    CN = 4,
  };
  CGameManialinkPage* const Page; // Maniascript
  const bool PageIsVisible; // Maniascript
  bool PageAlwaysUpdateScript; // Maniascript
  const uint Now; // Maniascript
  const uint Period; // Maniascript
  const uint CurrentTime; // Maniascript
  const string CurrentTimeText; // Maniascript
  const string CurrentLocalDateText; // Maniascript
  const wstring CurrentTimezone; // Maniascript
  CGamePlayerInfo* const LocalUser; // Maniascript
  CGameManiaTitle* const LoadedTitle; // Maniascript
  const CGameManialinkScriptHandler::ESystemPlatform SystemPlatform; // Maniascript
  const CGameManialinkScriptHandler::ESystemSkuIdentifier SystemSkuIdentifier; // Maniascript
  const MwFastBuffer<CGameManialinkScriptEvent*> PendingEvents; // Maniascript
  void Dbg_SetProcessed(CGameManialinkScriptEvent* Event); // Maniascript
  bool Dbg_WarnOnDroppedEvents; // Maniascript
  const float MouseX; // Maniascript
  const float MouseY; // Maniascript
  const bool MouseLeftButton; // Maniascript
  const bool MouseRightButton; // Maniascript
  const bool MouseMiddleButton; // Maniascript
  const bool KeyUp; // Maniascript
  const bool KeyDown; // Maniascript
  const bool KeyLeft; // Maniascript
  const bool KeyRight; // Maniascript
  const bool KeyReturn; // Maniascript
  const bool KeySpace; // Maniascript
  const bool KeyDelete; // Maniascript
  bool IsKeyPressed(int KeyCode); // Maniascript
  bool EnableMenuNavigationInputs; // Maniascript
  void EnableMenuNavigation(bool EnableInputs, bool WithAutoFocus, CGameManialinkControl* AutoBackControl, int InputPriority); // Maniascript
  void EnableMenuNavigation2(bool EnableInputs, bool WithAutoFocus, bool WithManualScroll, CGameManialinkControl* AutoBackControl, int InputPriority); // Maniascript
  const bool IsMenuNavigationForeground; // Maniascript
  void OpenLink(string Url, CGameManialinkScriptHandler::ELinkType LinkType); // Maniascript
  void TriggerPageAction(string ActionString); // Maniascript
  CXmlScriptParsingManager* const Xml; // Maniascript
  CNetScriptHttpManager* const Http; // Maniascript
  CGameVideoScriptManager* const Video; // Maniascript
  CAudioScriptManager* const Audio; // Maniascript
  CInputScriptManager* const Input; // Maniascript
  CGameDataFileManagerScript* const DataFileMgr; // Maniascript
  CGameScoreAndLeaderBoardManagerScript* const ScoreMgr; // Maniascript
  CGameZoneManagerScript* const ZoneMgr; // Maniascript
  CGameUserPrivilegesManagerScript* const PrivilegeMgr; // Maniascript
  CGameMasterServerRichPresenceManagerScript* const PresenceMgr; // Maniascript
  CGameUserManagerScript* const UserMgr; // Maniascript
  CGameManialinkAnimManager* const AnimMgr; // Maniascript
  CGameMenuSceneScriptManager* const MenuSceneMgr; // Maniascript
  CSystemPlatformScript* const System; // Maniascript
  CGameWebServicesNotificationManagerScript* const WSNotificationMgr; // Maniascript
  void SendCustomEvent(wstring Type, MwFastBuffer<wstring>& Data); // Maniascript
  void PreloadImage(string ImageUrl); // Maniascript
  void PreloadAll(); // Maniascript
  wstring Dbg_DumpDeclareForVariables(CMwNod* Nod, bool StatsOnly); // Maniascript
  wstring FilterProfanities(wstring TextToFilter); // Maniascript
};

// File extension: 'Frame.Gbx'
struct CGamePlaygroundControlScores : public CControlFrame {
  CGamePlaygroundControlScores();

  uint ListLineCount;
  uint ListColumnCount;
  CControlFrame* CardModelPlayer;
  float PlayerScale;
  float HorizontalMargin;
  float VerticalMargin;
  float CenterMargin;
  bool IsCentered;
  uint Page;
  uint PageCount;
  void PrevPage();
  void NextPage();
  bool LabelHelpEnabled;
  bool LabelMessageEnabled;
  bool HidePureSpectators;
  bool Rank;
  bool Avatar;
  bool IsLocalPlayer;
  bool IsSpectator;
  bool Score;
  bool ScoreInc;
  bool LapScore;
  bool LadderRank;
  bool LadderPointsGain;
  bool CupModeResult;
  bool NotPlaying;
};

struct CGameManialinkMediaPlayer : public CGameManialinkControl {
  bool IsInitPlay; // Maniascript
  bool Music; // Maniascript
  bool IsLooping; // Maniascript
  float Volume; // Maniascript
  string Url; // Maniascript
  void Play(); // Maniascript
  void Stop(); // Maniascript
  void StopAndRewind(); // Maniascript
};

struct CGameNetFormPlayground : public CGameNetForm {
  CGameNetFormPlayground();

};

struct CGameCtnArticleNode : public CMwNod {
  wstring Name;
  const wstring NodeName; // Maniascript
  const bool IsDirectory; // Maniascript
  CGameCtnArticleNodeDirectory* const ParentNode; // Maniascript
  CMwNod* GetCollectorNod(); // Maniascript
};

struct CGameSwitcher : public CMwNod {
  int FocusDialogCount;
  const MwFastBuffer<CGameSwitcher*> ModuleStack;
};

struct CGameManialinkOldTable : public CGameManialinkControl {
};

struct CGameCtnMenusManiaPlanet : public CGameCtnMenus {
  const MwFastBuffer<CGameCtnChallengeInfo*> ChallengeInfosCampaign;
  CGameCtnCampaign* const CurrentCampaign;
  CGameManialink3dWorld* const Manialink3dWorld;
  CGameManialink3dStyle* const Manialink3d_BaseStyle;
  CGameManiaAppTitle* const MenuCustom_CurrentManiaApp;
  CGameManiaPlanetMenuStations* const MenuManiaPlanet_MenuStations;
  CGameManialink3dStyle* const MenuManiaPlanet3d_BrowserManialink3dStyle;
  iso4 CurFocusedCamLoc;
  vec2 CurFocusedCamFovMaxMinY;
  bool DebugBlockBackground;
};

// Description: "Manialink entry."
struct CGameManialinkLabel : public CGameManialinkControl {
  enum class CGameManialinkLabel::EBlendMode {
    Default = 0,
    Add = 1,
  };
  enum class CGameManialinkLabel::EFilterProfanities {
    Never = 0,
    OnlyIfNotTranslated = 1,
    Always = 2,
  };
  string Style; // Maniascript
  string Substyle; // Maniascript
  wstring TextFont; // Maniascript
  void SetText(wstring NewText); // Maniascript
  wstring Value; // Maniascript
  const uint ValueLineCount; // Maniascript
  int MaxLine; // Maniascript
  float LineSpacing; // Maniascript
  float ItalicSlope; // Maniascript
  bool AppendEllipsis; // Maniascript
  bool AutoNewLine; // Maniascript
  float Opacity; // Maniascript
  wstring TextPrefix; // Maniascript
  vec3 TextColor; // Maniascript
  float TextSizeReal; // Maniascript
  uint TextSize; // Maniascript
  CGameManialinkLabel::EBlendMode Blend; // Maniascript
  float ComputeWidth(wstring Text); // Maniascript
  float ComputeWidth2(wstring Text, bool Translated); // Maniascript
  float ComputeHeight(wstring Text); // Maniascript
  wstring TTS_AltText; // Maniascript
  bool TTS_AltText_Translate; // Maniascript
  void TTS_Focus(); // Maniascript
  void TTS_Unfocus(); // Maniascript
  CGameManialinkLabel::EFilterProfanities FilterProfanities; // Maniascript
};

struct CGameUILayer : public CMwNod {
  enum class CGameUILayer::EUILayerType {
    Normal = 0,
    ScoresTable = 1,
    ScreenIn3d = 2,
    AltMenu = 3,
    Markers = 4,
    CutScene = 5,
    InGameMenu = 6,
    EditorPlugin = 7,
    ManiaplanetPlugin = 8,
    ManiaplanetMenu = 9,
    LoadingScreen = 10,
  };
  enum class CGameUILayer::EUILayerAnimation {
    None = 0,
    DownFast = 1,
    DownSlow = 2,
    LeftFast = 3,
    LeftSlow = 4,
    RightFast = 5,
    RightSlow = 6,
    ScaleFast = 7,
    ScaleSlow = 8,
    UpFast = 9,
    UpSlow = 10,
  };
  bool IsVisible; // Maniascript
  const bool AnimInProgress; // Maniascript
  CGameUILayer::EUILayerType Type; // Maniascript
  CGameUILayer::EUILayerAnimation InAnimation; // Maniascript
  CGameUILayer::EUILayerAnimation OutAnimation; // Maniascript
  CGameUILayer::EUILayerAnimation InOutAnimation; // Maniascript
  string AttachId; // Maniascript
  wstring ManialinkPage; // Maniascript
  string ManialinkPageUtf8;
  CGameManialinkPage* const LocalPage; // Maniascript
  const bool IsLocalPageScriptRunning; // Maniascript
};

struct CGamePlaygroundCommon : public CGameCtnPlayground {
};

struct CGameCtnAnchoredObject : public CMwNod {
  CGameCtnAnchoredObject();

  enum class CGameCtnAnchoredObject::ECardinalDirections {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
  };
  enum class CGameCtnAnchoredObject::EMapElemColor {
    Default = 0,
    White = 1,
    Green = 2,
    Blue = 3,
    Red = 4,
    Black = 5,
  };
  enum class CGameCtnAnchoredObject::EPhaseOffset {
    None = 0,
    One8th = 1,
    Two8th = 2,
    Three8th = 3,
    Four8th = 4,
    Five8th = 5,
    Six8th = 6,
    Seven8th = 7,
  };
  enum class CGameCtnAnchoredObject::EMapElemLightmapQuality {
    Normal = 0,
    High = 1,
    VeryHigh = 2,
    Highest = 3,
    Low = 4,
    VeryLow = 5,
    Lowest = 6,
  };
  nat3 BlockUnitCoord;
  MwId AnchorTreeId;
  vec3 AbsolutePositionInMap;
  float Yaw;
  float Pitch;
  float Roll;
  bool IsFlying;
  bool IsAutoPiloted;
  float Scale;
  uint IVariant;
  CGameCtnAnchoredObject::EMapElemColor MapElemColor;
  CGameCtnAnchoredObject::EPhaseOffset AnimPhaseOffset;
  CGameCtnAnchoredObject::EMapElemLightmapQuality MapElemLmQuality;
  CGameItemModel* const ItemModel;
  bool IsLocationInitialised;
  iso4 BlockLocation;
  iso4 LocationInBlock;
  CGameWaypointSpecialProperty* WaypointSpecialProperty;
};

struct CGameSwitcherModule : public CMwNod {
};

// Description: "This object handles the interface."
struct CGamePlaygroundUIConfig : public CMwNod {
  enum class CGamePlaygroundUIConfig::EUISequence {
    None = 0,
    Playing = 1,
    Intro = 2,
    Outro = 3,
    Podium = 4,
    CustomMTClip = 5,
    EndRound = 6,
    PlayersPresentation = 7,
    UIInteraction = 8,
    RollingBackgroundIntro = 9,
    CustomMTClip_WithUIInteraction = 10,
    Finish = 11,
  };
  enum class CGamePlaygroundUIConfig::EUIStatus {
    None = 0,
    Normal = 1,
    Warning = 2,
    Error = 3,
    Official = 4,
  };
  enum class CGamePlaygroundUIConfig::EVisibility {
    None = 0,
    Normal = 1,
    Manual = 2,
    ForcedHidden = 3,
    ForcedVisible = 4,
  };
  enum class CGamePlaygroundUIConfig::ELabelsVisibility {
    None = 0,
    Never = 1,
    Always = 2,
    WhenInFrustum = 3,
    WhenVisible = 4,
    WhenInMiddleOfScreen = 5,
  };
  enum class CGamePlaygroundUIConfig::EAvatarVariant {
    Default = 0,
    Sad = 1,
    Happy = 2,
  };
  enum class CGamePlaygroundUIConfig::EUISound {
    Default = 0,
    Silence = 1,
    StartMatch = 2,
    EndMatch = 3,
    StartRound = 4,
    EndRound = 5,
    PhaseChange = 6,
    TieBreakPoint = 7,
    TiePoint = 8,
    VictoryPoint = 9,
    Capture = 10,
    TimeOut = 11,
    Notice = 12,
    Warning = 13,
    PlayerEliminated = 14,
    PlayerHit = 15,
    Checkpoint = 16,
    Finish = 17,
    Record = 18,
    ScoreProgress = 19,
    RankChange = 20,
    Bonus = 21,
    FirstHit = 22,
    Combo = 23,
    PlayersRemaining = 24,
    Custom1 = 25,
    Custom2 = 26,
    Custom3 = 27,
    Custom4 = 28,
  };
  enum class CGamePlaygroundUIConfig::ENoticeLevel {
    Default = 0,
    PlayerInfo = 1,
    PlayerWarning = 2,
    MapInfo = 3,
    MapWarning = 4,
    MatchInfo = 5,
    MatchWarning = 6,
  };
  enum class CGamePlaygroundUIConfig::EMessageDisplay {
    Chat = 0,
    Small = 1,
    Status = 2,
    Big = 3,
  };
  enum class CGamePlaygroundUIConfig::EObserverMode {
    Default = 0,
    Forced = 1,
    Forbidden = 2,
    Manual = 3,
  };
  enum class CGamePlaygroundUIConfig::EHudVisibility {
    Nothing = 0,
    Everything = 1,
    MarkersOnly = 2,
    Default = 3,
  };
  enum class CGamePlaygroundUIConfig::EMediaClipStyle {
    None = 0,
    TM = 1,
    SM = 2,
  };
  enum class CGamePlaygroundUIConfig::ECutSceneStyle {
    None = 0,
    TM = 1,
    SM = 2,
  };
  CGamePlaygroundUIConfig::EUISequence UISequence; // Maniascript
  const bool UISequenceIsCompleted; // Maniascript
  wstring UISequence_CustomMTClip; // Maniascript
  uint UISequence_CustomMTRefTime; // Maniascript
  bool UISequence_CanSkipIntroMT; // Maniascript
  string UISequence_PodiumPlayersWin; // Maniascript
  string UISequence_PodiumPlayersLose; // Maniascript
  CGamePlaygroundUIConfig::EMediaClipStyle MediaClipStyle; // Maniascript
  CGamePlaygroundUIConfig::ECutSceneStyle UISequence_CutSceneStyle; // Maniascript
  bool DisableZoomTransitions; // Maniascript
  wstring ManialinkPage; // Maniascript
  wstring BigMessage; // Maniascript
  string BigMessageAvatarLogin; // Maniascript
  CGamePlaygroundUIConfig::EAvatarVariant BigMessageAvatarVariant; // Maniascript
  CGamePlaygroundUIConfig::EUISound BigMessageSound; // Maniascript
  uint BigMessageSoundVariant; // Maniascript
  wstring StatusMessage; // Maniascript
  wstring GaugeMessage; // Maniascript
  float GaugeRatio; // Maniascript
  uint GaugeClan; // Maniascript
  string Hud3dMarkers; // Maniascript
  string MarkersXML; // Maniascript
  bool OverrideMarkers; // Maniascript
  const MwFastBuffer<CGameHud3dMarkerConfig*> Markers; // Maniascript
  const MwFastBuffer<CGameUILayer*> UILayers; // Maniascript
  bool OverlayHideAll; // Maniascript
  bool OverlayHideNotices; // Maniascript
  bool OverlayHideMapInfo; // Maniascript
  bool OverlayHideOpponentsInfo; // Maniascript
  bool OverlayHideChat; // Maniascript
  bool OverlayHideCheckPointList; // Maniascript
  bool OverlayHideRoundScores; // Maniascript
  bool OverlayHideCountdown; // Maniascript
  bool OverlayHideCrosshair; // Maniascript
  bool OverlayHideGauges; // Maniascript
  bool OverlayHideConsumables; // Maniascript
  bool OverlayHide321Go; // Maniascript
  bool OverlayMute321Go; // Maniascript
  bool OverlayHideBackground; // Maniascript
  bool OverlayHideChrono; // Maniascript
  bool OverlayHideSpeedAndDist; // Maniascript
  bool OverlayHidePersonnalBestAndRank; // Maniascript
  bool OverlayHidePosition; // Maniascript
  bool OverlayHideCheckPointTime; // Maniascript
  bool OverlayHideEndMapLadderRecap; // Maniascript
  bool OverlayHideMultilapInfos; // Maniascript
  bool OverlayHideSpectatorControllers; // Maniascript
  bool OverlayHideSpectatorInfos; // Maniascript
  bool OverlayChatHideAvatar; // Maniascript
  uint OverlayChatLineCount; // Range: 0 - 40 // Maniascript
  vec2 OverlayChatOffset; // Maniascript
  float OverlayChatWidthCoef; // Range: 0.1 - 10 // Maniascript
  MwId ForcedCrosshairAction; // Maniascript
  vec2 CountdownCoord; // Maniascript
  bool NoticesFilter_HidePlayerInfo; // Maniascript
  bool NoticesFilter_HidePlayerWarning; // Maniascript
  bool NoticesFilter_HidePlayerInfoIfNotMe; // Maniascript
  bool NoticesFilter_HidePlayerWarningIfNotMe; // Maniascript
  bool NoticesFilter_HideMapInfo; // Maniascript
  bool NoticesFilter_HideMapWarning; // Maniascript
  bool NoticesFilter_HideMatchInfo; // Maniascript
  bool NoticesFilter_HideMatchWarning; // Maniascript
  CGamePlaygroundUIConfig::ENoticeLevel NoticesFilter_LevelToShowAsBigMessage; // Maniascript
  string ScoreTable; // Maniascript
  CGamePlaygroundUIConfig::EVisibility ScoreTableVisibility; // Maniascript
  string SmallScoreTable; // Maniascript
  CGamePlaygroundUIConfig::EVisibility SmallScoreTableVisibility; // Maniascript
  bool ScoreTableOnlyManialink; // Maniascript
  bool AltMenuNoDefaultScores; // Maniascript
  bool AltMenuNoCustomScores; // Maniascript
  bool OverlayScoreSummary; // Maniascript
  MwId ScoreSummary_Player1; // Maniascript
  int ScoreSummary_Points1; // Maniascript
  int ScoreSummary_RoundPoints1; // Maniascript
  int ScoreSummary_MatchPoints1; // Maniascript
  float ScoreSummary_Gauge1; // Maniascript
  MwId ScoreSummary_Player2; // Maniascript
  int ScoreSummary_Points2; // Maniascript
  int ScoreSummary_RoundPoints2; // Maniascript
  int ScoreSummary_MatchPoints2; // Maniascript
  float ScoreSummary_Gauge2; // Maniascript
  bool ScreenIn3dHideScoreSummary; // Maniascript
  bool ScreenIn3dHideVersus; // Maniascript
  bool CameraDisplay_IsActive; // Maniascript
  bool DisplayControl_UseLiveCamera; // Maniascript
  bool DisplayControl_UseEsportsProgrammation; // Maniascript
  int CountdownEndTime; // Maniascript
  CGamePlaygroundUIConfig::EUIStatus UIStatus; // Maniascript
  CGamePlaygroundUIConfig::EHudVisibility LabelsVisibility; // Maniascript
  bool LabelsVisibility_SkipMarkersOnly; // Maniascript
  CGamePlaygroundUIConfig::ELabelsVisibility AlliesLabelsVisibility; // Maniascript
  CGamePlaygroundUIConfig::EVisibility AlliesLabelsShowGauges; // Maniascript
  CGamePlaygroundUIConfig::EVisibility AlliesLabelsShowNames; // Maniascript
  uint AlliesLabelsMaxCount; // Range: 0 - 250 // Maniascript
  CGamePlaygroundUIConfig::ELabelsVisibility TeamLabelsVisibility; // Maniascript
  CGamePlaygroundUIConfig::EVisibility TeamLabelsShowGauges; // Maniascript
  CGamePlaygroundUIConfig::EVisibility TeamLabelsShowNames; // Maniascript
  CGamePlaygroundUIConfig::ELabelsVisibility OpposingTeamLabelsVisibility; // Maniascript
  CGamePlaygroundUIConfig::EVisibility OpposingTeamLabelsShowGauges; // Maniascript
  CGamePlaygroundUIConfig::EVisibility OpposingTeamLabelsShowNames; // Maniascript
  bool PlayerDisableFreeCam; // Maniascript
  bool ForceSpectator; // Maniascript
  uint SpectatorForceCameraType; // Maniascript
  float SpectatorCamAutoLatitude; // Maniascript
  float SpectatorCamAutoLongitude; // Maniascript
  float SpectatorCamAutoRadius; // Maniascript
  CGamePlaygroundUIConfig::EObserverMode SpectatorObserverMode; // Maniascript
  void Spectator_SetAutoTarget_Clear(); // Maniascript
  void Spectator_SetAutoTarget_All(); // Maniascript
  void Spectator_SetAutoTarget_User(CGamePlayerInfo* User); // Maniascript
  void Spectator_SetForcedTarget_Clear(); // Maniascript
  void Spectator_SetForcedTarget_AllPlayers(); // Maniascript
  void Spectator_SetForcedTarget_AllMap(); // Maniascript
  void Spectator_SetForcedTarget_Entity(CGameScriptEntity* Entity); // Maniascript
  void Spectator_SetForcedTarget_Landmark(CGameScriptMapLandmark* Landmark); // Maniascript
  void Spectator_SetForcedTarget_Ghost(MwId GhostInstanceId); // Maniascript
  void Spectator_SetForcedTarget_Clan(int ClanNumber); // Maniascript
  void SendChat(wstring Text); // Maniascript
  void SendNotice(wstring Text, CGamePlaygroundUIConfig::ENoticeLevel Level, CGamePlayerInfo* Avatar, CGamePlaygroundUIConfig::EAvatarVariant AvatarVariant, CGamePlaygroundUIConfig::EUISound Sound, int SoundVariant); // Maniascript
  wstring GetLayerManialinkAction(CGameUILayer* Layer); // Maniascript
  void ClearLayerManialinkAction(CGameUILayer* Layer); // Maniascript
  CGameHud3dMarkerConfig* AddMarkerPosition(vec3 Position); // Maniascript
  CGameHud3dMarkerConfig* AddMarkerEntity(CGameScriptEntity* Entity); // Maniascript
  CGameHud3dMarkerConfig* AddMarkerLandmark(CGameScriptMapLandmark* Landmark); // Maniascript
  CGameHud3dMarkerConfig* AddMarkerGhost(MwId GhostInstanceId); // Maniascript
  void RemoveMarker(CGameHud3dMarkerConfig* Marker); // Maniascript
  void ClearMarkers(); // Maniascript
  void QueueMessage1(int Duration, int Priority, CGamePlaygroundUIConfig::EMessageDisplay Level, wstring Message, CGamePlayerInfo* Avatar, CGamePlaygroundUIConfig::EAvatarVariant AvatarVariant, CGamePlaygroundUIConfig::EUISound Sound, int SoundVariant); // Maniascript
  void QueueMessage2(int Duration, int Priority, CGamePlaygroundUIConfig::EMessageDisplay Level, wstring Message, CGamePlaygroundUIConfig::EUISound Sound, int SoundVariant); // Maniascript
  void QueueMessage3(int Duration, int Priority, CGamePlaygroundUIConfig::EMessageDisplay Level, wstring Message); // Maniascript
  void ClearMessages(); // Maniascript
  MwId SpectatorAutoTarget; // Maniascript
  MwId SpectatorForcedTarget; // Maniascript
  int SpectatorForcedClan; // Maniascript
};

// Description: ""
struct CGameManialinkFrame : public CGameManialinkControl {
  const MwFastBuffer<CGameManialinkControl*> Controls; // Maniascript
  const MwFastBuffer<CGameManialinkControl*> ControlsCache; // Maniascript
  CGameManialinkControl* GetFirstChild(string ControlId); // Maniascript
  bool ClipWindowActive; // Maniascript
  vec2 ClipWindowRelativePosition; // Maniascript
  vec2 ClipWindowSize; // Maniascript
  bool ScrollActive; // Maniascript
  vec2 ScrollOffset; // Maniascript
  const vec2 ScrollAnimOffset; // Maniascript
  vec2 ScrollMax; // Maniascript
  vec2 ScrollMin; // Maniascript
  bool ScrollGridSnap; // Maniascript
  vec2 ScrollGrid; // Maniascript
  void Scroll(vec2 DeltaInGridUnits); // Maniascript
  void ScrollBumpTop(); // Maniascript
  void ScrollBumpBottom(); // Maniascript
  void ScrollBumpLeft(); // Maniascript
  void ScrollBumpRight(); // Maniascript
  bool DisablePreload; // Maniascript
};

// Description: ""
struct CGameManialinkPage : public CMwNod {
  const string Url;
  CMwNod* const ScriptHandler;
  CGameManialinkFrame* const MainFrame; // Maniascript
  CGameManialinkControl* GetFirstChild(string ControlId); // Maniascript
  const MwFastBuffer<CGameManialinkControl*> ControlsCache; // Maniascript
  CGameManialinkControl* const FocusedControl; // Maniascript
  bool LinksInhibited; // Maniascript
  void GetClassChildren(string Class, CGameManialinkFrame* Frame, bool Recursive); // Maniascript
  const MwFastBuffer<CGameManialinkControl*> GetClassChildren_Result; // Maniascript
  void ScrollToControl(CGameManialinkControl* Control); // Maniascript
};

// Description: "Manialink control."
struct CGameManialinkControl : public CMwNod {
  enum class CGameManialinkControl::EAlignHorizontal {
    Left = 0,
    HCenter = 1,
    Right = 2,
    None = 3,
  };
  enum class CGameManialinkControl::EAlignVertical {
    Top = 0,
    VCenter = 1,
    Bottom = 2,
    None = 3,
    VCenter2 = 4,
  };
  CControlBase* const Control;
  CGameManialinkFrame* const Parent; // Maniascript
  const string ControlId; // Maniascript
  const MwSArray<string> ControlClasses; // Maniascript
  bool HasClass(string Class); // Maniascript
  vec2 Size; // Maniascript
  CGameManialinkControl::EAlignHorizontal HorizontalAlign; // Maniascript
  CGameManialinkControl::EAlignVertical VerticalAlign; // Maniascript
  bool Visible; // Maniascript
  vec2 RelativePosition_V3; // Maniascript
  float ZIndex; // Maniascript
  float RelativeScale; // Maniascript
  float RelativeRotation; // Maniascript
  const vec2 AbsolutePosition_V3; // Maniascript
  const vec3 AbsolutePosition; // Maniascript
  const float AbsoluteScale; // Maniascript
  const float AbsoluteRotation; // Maniascript
  bool DataAttributeExists(string DataName); // Maniascript
  string DataAttributeGet(string DataName); // Maniascript
  void DataAttributeSet(string DataName, string DataValue); // Maniascript
  wstring ToolTip; // Maniascript
  void Show(); // Maniascript
  void Hide(); // Maniascript
  void Unload(); // Maniascript
  void Focus(); // Maniascript
  void ScriptEvents_Disable(); // Maniascript
  void ScriptEvents_Restore(); // Maniascript
  const bool IsFocused; // Maniascript
  vec3 RelativePosition; // Maniascript
  float PosnX; // Maniascript
  float PosnY; // Maniascript
  float PosnZ; // Maniascript
  float Scale; // Maniascript
};

// Description: ""
struct CGameManialinkQuad : public CGameManialinkControl {
  enum class CGameManialinkQuad::EKeepRatioMode {
    Inactive = 0,
    Clip = 1,
    Fit = 2,
  };
  enum class CGameManialinkQuad::EBlendMode {
    Default = 0,
    Add = 1,
  };
  void ChangeImageUrl(string fieldName); // Maniascript
  CPlugBitmap* Image; // Maniascript
  string ImageUrl; // Maniascript
  string ImageUrlFocus; // Maniascript
  string AlphaMaskUrl; // Maniascript
  string Style; // Maniascript
  string Substyle; // Maniascript
  bool StyleSelected; // Maniascript
  const bool DownloadInProgress; // Maniascript
  bool SuperSample; // Maniascript
  vec3 Colorize; // Maniascript
  vec3 ModulateColor; // Maniascript
  vec3 BgColor; // Maniascript
  vec3 BgColorFocus; // Maniascript
  float Opacity; // Maniascript
  CGameManialinkQuad::EKeepRatioMode KeepRatio; // Maniascript
  CGameManialinkQuad::EBlendMode Blend; // Maniascript
  void RefreshImages(); // Maniascript
  void TTS_Focus(); // Maniascript
  void TTS_Unfocus(); // Maniascript
  wstring TTS_AltText; // Maniascript
  bool TTS_AltText_Translate; // Maniascript
  bool Video_IsPlaying; // Maniascript
  bool Video_IsLooping; // Maniascript
  void Video_Rewind(); // Maniascript
};

struct CGameManiaPlanetMenuStations : public CMwNod {
  bool IsActive;
  CMwCmdFiber* const MenuManiaPlanetCmd;
  CScene2d* const OverlayMenuStations;
  CGameManialinkPage* const MainPage;
  const wstring MainPageManialinkViewerStartUrl;
  CGameManiaTitle* const MainStationFullManiaTitle;
  const bool IsBeginnerMenuDisplay;
  bool IsEditingStations;
  CGameScriptHandlerStation* const ScriptHandlerStations;
};

// File extension: 'AnchorPoint.gbx'
struct CGameCtnAnchorPoint : public CMwNod {
  CGameCtnAnchorPoint();

  const MwId TreeId;
  iso4 LocationInBlock;
  CGameCtnBlock* const Block;
};

// File extension: 'MacroBlock.Gbx'
struct CGameCtnMacroBlockInfo : public CGameCtnCollector {
  CGameCtnMacroBlockInfo();

  bool Connected;
  bool Initialized;
  CGameCtnBlockInfo* const GeneratedBlockInfo;
  const bool IsGround; // Maniascript
  const bool HasStart; // Maniascript
  const bool HasFinish; // Maniascript
  const bool HasCheckpoint; // Maniascript
  const bool HasMultilap; // Maniascript
  CGameCtnBlockInfo* const GeneratedBlockModel; // Maniascript
  CScriptTraitsMetadata* ScriptMetadata;
  void ClearScriptMetadata(); // Maniascript
  const wstring Name; // Maniascript
};

struct CGameCtnEditorCommon : public CGameCtnEditor {
  enum class CGameCtnEditorCommon::ETrafficDensityEnum {
    Null = 0,
    Low = 1,
    Normal = 2,
    High = 3,
    Very_High = 4, // Very High
  };
  CGameCtnChallenge* const Challenge;
  CGameCursorBlock* const Cursor;
  CGameCursorItem* const ItemCursor;
  vec3 GetMapCenter(); // Maniascript
  bool GhostBlockForcedGroundElseAir;
  uint GhostBlockForcedVariantIndex;
  CGameCtnEditorCommonInterface* EditorInterface;
  const float MouseMoveDist;
  CGameCtnBlockInfo* const CurrentBlockInfo;
  CGameCtnBlockInfo* const CurrentGhostBlockInfo;
  CGameItemModel* const CurrentItemModel;
  CGameItemModel* const CurrentTrafficItemModel;
  CGameCtnMacroBlockInfo* const CurrentMacroBlockInfo;
  CGameCtnMacroBlockInfo* const CopyPasteMacroBlockInfo;
  CGameCtnBlock* CurrentTerrainBlock;
  CGameCtnBlock* CurrentBlock;
  int PluginOffsetX;
  int PluginOffsetY;
  int PluginOffsetZ;
  CGameEditorPluginMapMapType* const PluginMapType;
  void ReloadPlugins();
  void DeactivateAllPlugins();
  bool UseNewTerraforming;
  bool UseNewPillars;
  bool PasteAsFreeMacroBlock;
  bool HackExternalMbIconsHD;
  bool HackInternalMbIconsHD;
  bool HackForceTerrainBulldozeForbidden;
  bool EnableGhostMode;
  bool EmbedCustomItems;
  bool HideAlwaysCursorDirectionalArrow;
  bool CanDisplayHelpAtStart;
  uint IterationsRecursive;
  CGameCtnBlock* const PickedBlock;
  CGameCtnAnchoredObject* const PickedObject;
  CGameCtnBlock* BlockEditor2_Block;
  CGameCtnBlockInfo* BlockEditor2_BlockInfo;
  CPlugSolid* BlockEditor2_BlockSolid;
  uint CopperPriceThreshhold_WarningHigh;
  uint CopperPriceThreshhold_WarningVeryHigh;
  bool HasCopperPriceDemoLimit;
  uint CopperPriceThreshhold_DemoWarning;
  uint CopperPriceThreshhold_DemoLimit;
  string ColoredCopperPrice;
  wstring SkinText;
  CGameControlCameraEditorOrbital* OrbitalCameraControl;
  const float CameraTargetMinX;
  const float CameraTargetMaxX;
  const float CameraTargetMinZ;
  const float CameraTargetMaxZ;
  const float CameraAngle;
  CSceneMobil* const Grid;
  CGameOutlineBox* const UndergroundBox;
  CGameOutlineBox* const SelectionBox;
  CGameOutlineBox* const ItemSelectionBox;
  CGameOutlineBox* const CustomSelectionBox;
  CGameOutlineBox* const OffZoneSelectionBox;
  CGameOutlineBox* const ConstraintsBox;
  CGameOutlineBox* const SectorConstraintsBox;
  CGameOutlineBox* const ConstructibleZoneBorderBox;
  CGameOutlineBox* const SectorsOutlineBox;
  CGameOutlineBox* const CurrentSectorOutlineBox;
  CGameOutlineBox* const AnimatedElemOutlineBox_Block;
  CGameOutlineBox* const AnimatedElemOutlineBox_Items;
  vec3 GridColor;
  float GridColorAlpha; // Range: 0 - 1
  string BlockEditor2_Text;
  void ButtonComputeShadowsOnClick();
  void ButtonComputeShadowsDialogOnClick();
  void ButtonComputeDecalsOnClick();
  void ButtonHelper1OnClick();
  void SweepBlocksAndSave();
  void SweepFreeBlocksAndSave();
  void SweepTerrainAndSave();
  void SweepOffZoneAndSave();
  void SweepConstraintsAndSave();
  void SweepSectorsAndSave();
  void SweepObjectsAndSave();
  void SuperSweepAndSave();
  void SweepSelectionAndSave();
  void BlockViewerOnClick();
  void ButtonTestOnClick();
  void ButtonOffZoneOnClick();
  void ButtonConstraintsOnClick();
  void ButtonSectorOnClick();
  void ButtonBackOnClick();
  void ButtonSaveOnClick();
  void ButtonSaveAsOnClick();
  void ButtonLoadOnClick();
  void ButtonValidateOnClick();
  void ButtonCursorRaiseOnClick();
  void ButtonCursorLowerOnClick();
  void ButtonCursorTurnClockwiseOnClick();
  void ButtonCursorTurnAnticlockwiseOnClick();
  void ButtonCursorUpOnClick();
  void ButtonCursorDownOnClick();
  void ButtonCursorLeftOnClick();
  void ButtonCursorRightOnClick();
  void ButtonUndoOnClick();
  void ButtonRedoOnClick();
  void ButtonZoomInOnClick();
  void ButtonZoomOutOnClick();
  void ButtonEraserModeOnClick();
  void ButtonUndergroundModeOnClick();
  void ButtonAirMappingModeOnClick();
  void ButtonFreelookModeOnClick();
  void ButtonPickerModeOnClick();
  void ButtonSelectionBoxAddModeOnClick();
  void ButtonSelectionBoxSubModeOnClick();
  void ButtonChooseSkinModeOnClick();
  void ButtonSetCardEventModeOnClick();
  void ButtonInventoryDecalsOnClick();
  void ButtonInventoryTrafficOnClick();
  void ButtonInventoryFlyingTrafficOnClick();
  void ButtonInventoryBlocksOnClick();
  void ButtonInventoryGhostBlocksOnClick();
  void ButtonInventoryTerraformOnClick();
  void ButtonInventoryObjectsOnClick();
  void ButtonInventoryPluginsOnClick();
  void ButtonInventoryMacroBlocksOnClick();
  void ButtonCreateDeckOnClick();
  void ButtonBlockPropertyModeOnClick();
  void ButtonBlockStockOnClick();
  void ButtonCopyPasteOnClick();
  void DeleteArticle_OnRemoveInstancesOnly();
  void DeleteArticle_OnRemoveInstancesAndModel();
  void DeleteArticle_OnYes();
  void RemoveItemCollectionFromFavorites_OnYes();
  void RemoveItemCollectionFromFavorites_OnCancel();
  void ButtonItemEditModeOnClick();
  void ButtonItemNewModeOnClick();
  void ButtonItemCreateFromBlockModeOnClick();
  void ButtonBlockItemEditModeOnClick();
  void ButtonBlockItemCreateModeOnClick();
  void ButtonLightOnClick();
  void ButtonNormalBlockModeOnClick();
  void ButtonAirBlockModeOnClick();
  void ButtonGhostBlockModeOnClick();
  void ButtonFreeBlockModeOnClick();
  void ButtonNormalItemModeOnClick();
  void ButtonFreeGroundItemModeOnClick();
  void ButtonFreeItemModeOnClick();
  void ButtonNormalMacroblockModeOnClick();
  void ButtonFreeMacroblockModeOnClick();
  void ButtonChallengeTypeOnClick();
  void ButtonObjectivesOnClick();
  void ButtonHelpOnClick();
  void ButtonShowChallengeDetailsOnClick();
  void ButtonEditToolsOnClick();
  void ButtonAdditionalToolsOnClick();
  void ButtonChooseMapTypeOnClick();
  void ButtonSelectionBoxCopyOnClick();
  void ButtonSelectionBoxCutOnClick();
  void ButtonSelectionBoxSaveNewOnClick();
  void ButtonSelectionBoxResetOnClick();
  void ButtonSelectionBoxSelectAllOnClick();
  void ButtonSelectionBoxSymmetriseOnClick();
  void EditorPluginCreateOnClick();
  void EditorPluginEditOnClick();
  void ButtonNewTerrainEditorOnClick();
  void ButtonNewPillarsOnClick();
  void ButtonHackExternalMbIconsHDOnClick();
  void ButtonHackInternalMbIconsHDOnClick();
  void ButtonHackCreateItemGroupFromMbOnClick();
  void ButtonEditEndRaceReplay();
  void ButtonBackStepOnClick();
  bool EnableMapProcX2;
  float Radius;
  int Offset;
  void BlockEditor_OnRotateIcon();
  void BlockEditor_OnCancelIcon();
  void BlockEditor_OnSaveIcon();
  void SaveMacroBlockFromScript_OnSave();
  void SaveMacroBlockFromScript_OnCancel();
  void OnPluginOperationCancelled();
  void OnPluginOperationCancelled_OnYes();
  void OnPluginOperationCancelled_OnNo();
  void SaveChallengeFromScript_OnSave();
  void ComputeShadowsFromScript_OnOk();
  void SetObjectivesFromScript_OnOk();
  void HideInterfaceFromScript_OnOk();
  void HideInterfaceFromScript_OnCancel();
  void SwitchToMTFromScript_OnOk();
  void SwitchToPlaygroundFromScript_OnOk();
  void SwitchToValidationFromScript_OnOk();
  void SwitchToTestWithMapTypeFromScript_OnOk();
  void SwitchToTestWithMapTypeFromScript_OnCancel();
  void QuitFromScript_OnOk();
  void QuickQuitFromScript_OnOk();
  void ChangePlaceModeFromScript_OnOk();
  string DebugShootIconName;
  void OnMapRulesModeSelected();
  void TooManyCoppers_OnOk();
  void TooManyCoppers_OnNeverShowAgain();
  MwFastBuffer<wstring> ForcedPluginsNames;
  MwFastBuffer<string> ForcedPluginsSettings;
  bool MoodIsDynamicTime;
  float MoodTimeOfDay01; // Range: 0 - 1
  string MoodTimeOfDayStr;
  string MoodDayDurationStr;
  void Test_ShowEarthSquares();
  void Test_HideEarthSquares();
  UnnamedEnum CamMode;
  void UpdateBlockInfoGroups();
  NGameEditorMap_SExperimentalFeatures* const ExperimentalFeatures;
  bool HideBlockHelpers;
  void TurnIntoAirMb_Unsafe();
  void TurnIntoGroundMb_Unsafe();
  bool TurnIntoGroundMb_UseGroundNPB;
};

struct CGameCtnEditorFree : public CGameCtnEditorCommon {
};

struct CGameCtnEditorPuzzle : public CGameCtnEditorFree {
};

struct CGamePlaygroundScore : public CMwNod {
  CGamePlayerInfo* const User; // Maniascript
  const bool IsRegisteredForLadderMatch; // Maniascript
  const float LadderScore; // Maniascript
  int LadderRankSortValue; // Maniascript
  float LadderMatchScoreValue; // Maniascript
  uint LadderClan; // Maniascript
};

struct CGameCtnEditorCommonInterface : public CMwNod {
  CGameCtnEditorCommonInterface();

  CGameCtnArticleNodeDirectory* const BlockIconsRoot;
  CGameCtnArticleNodeDirectory* const TerraformIconsRoot;
  CGameCtnArticleNodeDirectory* const MacroBlockIconsRoot;
  CGameCtnArticleNodeDirectory* const ObjectIconsRoot;
  CGameCtnArticleNodeDirectory* const DecalIconsRoot;
  CGameCtnArticleNodeDirectory* const MacroDecalIconsRoot;
  CGameCtnArticleNodeDirectory* const TrafficObjectIconsRoot;
  CGameCtnArticleNodeDirectory* const PluginsIconsRoot;
  CScene2d* const InterfaceScene;
  CControlContainer* const InterfaceRoot;
  CGameCtnEditorCommon* const Editor;
  void ToggleBlockRotation();
  wstring CurrentToolTip;
  const string Allocated;
  const wstring m_LastMacroBlockSelected;
  void EditSnapCamera_OnOk();
  void EditSnapCamera_OnCancel();
  CPlugBitmap* const EditSnapCamera_BitmapSnap;
  void HideInterface();
  void ShowInterface();
  void ButtonPrevSubMenu();
  void ButtonNextSubMenu();
};

// File extension: 'MacroDecals.Gbx'
struct CGameCtnMacroDecals : public CGameCtnCollector {
  CGameCtnMacroDecals();

  bool Connected;
};

struct CGameCtnEditorSimple : public CGameCtnEditorPuzzle {
};

struct CGameCtnArticleNodeDirectory : public CGameCtnArticleNode {
  const MwFastBuffer<CGameCtnArticleNode*> ChildNodes;
  void CreateNewDirectory();
  const MwFastBuffer<CGameCtnArticleNode*> Children; // Maniascript
  const bool HasChildDirectory; // Maniascript
  const bool HasChildArticle; // Maniascript
};

struct CGameCtnArticleNodeArticle : public CGameCtnArticleNode {
  CGameCtnArticle* const Article;
};

struct CGameManialinkArrow : public CGameManialinkControl {
  CGameManialinkArrow();

};

// File extension: 'ZoneGenealogy.Gbx'
struct CGameCtnZoneGenealogy : public CMwNod {
  CGameCtnZoneGenealogy();

  MwId CurrentZoneId;
  uint CurrentIndex;
  UnnamedEnum Dir;
  MwFastBuffer<MwId> ZoneIds;
  CGameCtnZone* const CurrentZone;
  const MwFastBuffer<CGameCtnZone*> Zones;
  const MwFastBuffer<int> ZoneHeights;
  const int BaseHeight;
  const int BottomHeight;
  const int TopHeight;
  void RemoveLastZoneId();
  void AddZoneId();
};

// Description: "Server plugin"
struct CGameServerPlugin : public CMwNod {
  CPlugFileTextScript* Script;
  CGameManiaTitle* const LoadedTitle; // Maniascript
  CGameCtnChallengeInfo* const MapInfo; // Maniascript
  CGameConnectedClient* GetClientFromLogin(string Login); // Maniascript
  CGameConnectedClient* GetClientFromUI(CGamePlaygroundUIConfig* UI); // Maniascript
  CGameConnectedClient* GetClientFromUser(CGamePlayerInfo* Login); // Maniascript
  CGameConnectedClient* GetClientFromWebServicesUserId(string WebServicesUserId); // Maniascript
  const MwFastBuffer<CGameConnectedClient*> Clients; // Maniascript
  const MwFastBuffer<CGameConnectedClient*> Spectators; // Maniascript
  const MwFastBuffer<CGameConnectedClient*> Players; // Maniascript
  CGameScriptPlayer* GetPlaygroundPlayerFromLogin(string Login); // Maniascript
  const MwFastBuffer<CGameScriptPlayer*> PlaygroundPlayers; // Maniascript
  const MwFastBuffer<CGamePlaygroundScore*> Scores; // Maniascript
  const MwFastBuffer<CGamePlayerInfo*> Users; // Maniascript
  const MwFastBuffer<CGameTeamProfile*> Teams; // Maniascript
  string NeutralEmblemUrl; // Maniascript
  string ForcedClubLinkUrl1; // Maniascript
  string ForcedClubLinkUrl2; // Maniascript
  void TweakTeamColorsToAvoidHueOverlap(); // Maniascript
  const MwFastBuffer<uint> ClansNbPlayers; // Maniascript
  const MwFastBuffer<uint> ClanScores; // Maniascript
  const uint Now; // Maniascript
  const MwFastBuffer<CGameServerPluginEvent*> PendingEvents; // Maniascript
  void TriggerModeScriptEvent2(wstring Type, MwFastBuffer<wstring>& Data); // Maniascript
  void SendModeScriptCommandBoolean(string CommandName, bool BoolVal); // Maniascript
  void SendModeScriptCommandInteger(string CommandName, int IntVal); // Maniascript
  void SendModeScriptCommandReal(string CommandName, float RealVal); // Maniascript
  void SendModeScriptCommandText(string CommandName, wstring TextVal); // Maniascript
  void SendModeScriptCommandVec2(string CommandName, vec2 Vec2Val); // Maniascript
  void SendModeScriptCommandVec3(string CommandName, vec3 Vec3Val); // Maniascript
  void SendModeScriptCommandInt2(string CommandName, int2 Int2Val); // Maniascript
  void SendModeScriptCommandInt3(string CommandName, int3 Int3Val); // Maniascript
  const bool MapLoaded; // Maniascript
  const bool MapUnloadRequested; // Maniascript
  const MwFastBuffer<CGameCtnChallengeInfo*> MapList; // Maniascript
  const uint CurMapIndex; // Maniascript
  uint NextMapIndex; // Maniascript
  const uint CurMapTreeCount; // Maniascript
  void RestartMap(); // Maniascript
  void NextMap(); // Maniascript
  bool HoldMapUnloadRequest; // Maniascript
  bool Client_ComputeMinimap; // Maniascript
  CGamePlaygroundUIConfigMgrScript* const UIManager; // Maniascript
  CGameScriptServerAdmin* const ServerAdmin; // Maniascript
  CGameServerScriptXmlRpc* const XmlRpc; // Maniascript
  CXmlScriptParsingManager* const Xml; // Maniascript
  CNetScriptHttpManager* const Http; // Maniascript
  CSystemPlatformScript* const System; // Maniascript
  wstring Dbg_DumpDeclareForVariables(CMwNod* Nod, bool StatsOnly); // Maniascript
  const MwFastBuffer<CWebServicesTaskResult*> TaskResults; // Maniascript
  void TaskResult_Release(MwId TaskResultId); // Maniascript
  const MwFastBuffer<CGameGhostScript*> Ghosts; // Maniascript
  CWebServicesTaskResult_GhostScript* Ghost_Download(string Url); // Maniascript
  void GhostDriver_Playlist_Clear(CGameScriptPlayer* Player); // Maniascript
  void GhostDriver_Playlist_Add(CGameScriptPlayer* Player, CGameGhostScript* Ghost); // Maniascript
};

// Description: "ServerPlugin Event"
struct CGameServerPluginEvent : public CScriptBaseConstEvent {
  enum class CGameServerPluginEvent::EType {
    Unknown = 0,
    ClientConnected = 1,
    ClientDisconnected = 2,
    MapLoaded = 3,
    BeginMatch = 4,
    BeginRound = 5,
    EndRound = 6,
    EndMatch = 7,
    MapUnloadRequested = 8,
    MapUnloaded = 9,
    ChatCommand = 10,
    ChatMessage = 11,
    ModeCallback = 12,
  };
  enum class CGameServerPluginEvent::EChatOption {
    Default = 0,
    ToSpectatorCurrent = 1,
    ToSpectatorAll = 2,
    ToTeam = 3,
  };
  const CGameServerPluginEvent::EType Type; // Maniascript
  CGameConnectedClient* const Client; // Maniascript
  const wstring ChatText; // Maniascript
  const CGameServerPluginEvent::EChatOption ChatOption; // Maniascript
  const wstring ChatCommandType; // Maniascript
  const MwFastBuffer<wstring> ChatCommandData; // Maniascript
  const wstring ModeCallbackType; // Maniascript
  const MwFastBuffer<wstring> ModeCallbackData; // Maniascript
  const MwFastBuffer<CGamePlaygroundScore*> EndMatchScores; // Maniascript
  const MwFastBuffer<uint> EndMatchRanks; // Maniascript
};

struct CGameCtnAutoTerrain : public CMwNod {
  CGameCtnAutoTerrain();

  int OffsetX;
  int OffsetY;
  int OffsetZ;
  CGameCtnZoneGenealogy* const Genealogy;
};

struct CGameCtnSolidDecals : public CMwNod {
  CGameCtnSolidDecals();

  string Name;
  MwId TypeId;
  uint TypeIntensity;
  uint DecalFrequency;
  const uint DecalsCount;
};

struct CGameCtnBlockInfoMobil : public CMwNod {
  CGameCtnBlockInfoMobil();

  CSceneMobil* const OldMobil;
  CPlugSolid* SolidFid;
  CPlugPrefab* PrefabFid;
  CPlugSolid* OldSolidAggreg;
  CPlugSolid2Model* Solid2FromBlockItem;
  CPlugSurface* SurfaceFromBlockItem;
  CPlugSolid* const SolidCache;
  const MwFastBuffer<CGameCtnBlockInfoMobilLink*> DynaLinks;
  uint SolidFrequency;
  const MwFastBuffer<CGameCtnSolidDecals*> SolidDecals;
  uint NoDecalFrequency;
  const vec3 GeomTranslation;
  const vec3 GeomRotation;
  void UpdateDevName();
  CGameObjectModel* Cache_ObjectModelWithoutClips;
  CGameObjectModel* Cache_ObjectModelWithClips;
};

// Description: "A currently connected CUser"
struct CGameConnectedClient : public CMwNod {
  CGamePlayerInfo* const User; // Maniascript
  CGamePlaygroundUIConfig* const UI; // Maniascript
  const bool IsConnectedToMasterServer; // Maniascript
  const string ClientVersion; // Maniascript
  const string ClientTitleVersion; // Maniascript
  const bool IsSpectator; // Maniascript
  const uint IdleDuration; // Maniascript
};

struct CGameScoreTask_GetSeasonListFromUser : public CWebServicesTaskSequence {
};

struct CGameControlCameraEditorOrbital : public CGameControlCamera {
  CGameControlCameraEditorOrbital();

  float m_CameraToTargetDistance;
  float m_CurrentVAngle;
  float m_CurrentHAngle;
  vec3 m_TargetedPosition;
  float m_MinDistance;
  float m_MaxDistance;
  float m_MinVAngle;
  float m_MaxVAngle;
  float m_ParamTurnCameraDistance_X;
  float m_ParamTurnCameraDistance_Y;
  float m_ParamScrollAreaStart;
  float m_ParamScrollAreaMax;
  float m_ParamScrollLowerLimitStart;
  float m_ParamScrollLowerLimitEnd;
  float m_ParamScrollSpeed0_OnZoomMin;
  float m_ParamScrollSpeed0_OnZoomMax;
  float m_ParamScrollSpeed1_OnZoomMin;
  float m_ParamScrollSpeed1_OnZoomMax;
  float m_ParamPanSpeed_OnZoomMin;
  float m_ParamPanSpeed_OnZoomMax;
  bool m_ParamPanInvertIfTargetIsHigher;
  float m_ParamDefaultSmoothDuration;
  float m_ParamMiddleButZoomMinPower;
  float m_ParamMiddleButZoomMaxPower;
  float m_ParamScrollZoomPower;
  float m_ParamHeightSpeed_OnZoomMin;
  float m_ParamHeightSpeed_OnZoomMax;
  float m_ParamHeightSpeed_PgUpPgDown;
  float m_ParamButtonZoomPower;
  float m_ParamButtonZoomMinPower;
  float m_ParamSpeedBuildUpTime;
  float m_ParamFov; // Range: 10 - 170
};

// File extension: 'GameCtnMediaBlockDOF.gbx'
struct CGameCtnMediaBlockDOF : public CGameCtnMediaBlock {
  CGameCtnMediaBlockDOF();

  float FocusZ;
  float LensSize;
};

// File extension: 'GameCtnMediaBlockToneMapping.gbx'
struct CGameCtnMediaBlockToneMapping : public CGameCtnMediaBlock {
  CGameCtnMediaBlockToneMapping();

  float Exposure;
  float MaxHDR;
  UnnamedEnum FilmCurve;
};

// File extension: 'GameCtnMediaBlockBloomHdr.gbx'
struct CGameCtnMediaBlockBloomHdr : public CGameCtnMediaBlock {
  CGameCtnMediaBlockBloomHdr();

  float Intensity;
  float StreaksIntensity;
  float StreaksAttenuation;
};

struct CGameCtnMediaBlockTimeSpeed : public CGameCtnMediaBlock {
  CGameCtnMediaBlockTimeSpeed();

};

// File extension: 'GameCtnMediaBlockManialink.Gbx'
struct CGameCtnMediaBlockManialink : public CGameCtnMediaBlock {
  CGameCtnMediaBlockManialink();

  wstring ManialinkURL;
};

struct CGamePlayerProfileChunk : public CMwNod {
  const string ChunkName;
  const string GameName;
  const string Checksum;
  const uint TimeStamp;
  const bool IsSynchronizedFromMS;
  const bool IsModified;
  const bool IsLoaded;
  const bool IsOnlineSave;
};

struct CGamePlayerProfileChunk_AccountSettings : public CGamePlayerProfileChunk {
  CGamePlayerProfileChunk_AccountSettings();

  const string OnlineLogin;
  string OnlinePassword;
  string OnlineValidationCode;
  string OnlineSupportKey;
  wstring NickName;
  string Trigram;
  wstring Description;
  string ClubLinkUrl;
  bool LoginValidated;
  bool RememberOnlinePassword;
  bool AutoConnect;
  bool AskForAccountConversion;
  bool ReceiveNews;
  const uint OnlineRemainingNickNamesChangesCount;
  const uint OnlinePlanets;
  string RSAPrivateKey;
  string LastUsedMSAddress;
  string LastUsedMSPath;
  string LastSessionId;
  const wstring OnlinePath;
  CGameLeague* League;
  MwFastBuffer<wstring> LeagueSteps;
  wstring AvatarName;
  const MwFastBuffer<CGameNetOnlineMessage*> InboxMessages;
  const MwFastBuffer<CGameNetOnlineMessage*> ReadMessages;
  const MwFastBuffer<CGameNetOnlineMessage*> OutboxMessages;
  bool UnlockAllCheat;
  bool FriendsCheat;
  uint EulaVersion;
  uint PrivacyPolicyVersion;
  uint Age;
  const uint FameStars;
};

struct CGamePlayerProfileChunk_GameSettings : public CGamePlayerProfileChunk {
  CGamePlayerProfileChunk_GameSettings();

  float StereoscopyStrength01;
  float StereoscopyAdvancedScreenDist;
  CGameCtnMediaShootParams* ShootParamsVideo;
  wstring ServerName;
  wstring ServerComment;
  uint NetworkGameMode;
  uint MaxPlayerCount;
  uint MaxSpectatorCount;
  uint NetStateQuality;
  uint LadderMode;
  bool AdvancedCreate;
  bool AllowDownload;
  bool EnablePlayerSkinGeom;
  bool EnableUnlimitedHorns;
  bool AutoSaveReplayOnMulti;
  bool SaveRoundsSeparately;
  enum class CGamePlayerProfileChunk_GameSettings::EPlayerVisibility {
    Hidden = 0,
    Ghost = 1,
    Opaque = 2,
  };
  CGamePlayerProfileChunk_GameSettings::EPlayerVisibility DefaultOpponentVisibility;
  CGamePlayerProfileChunk_GameSettings::EPlayerVisibility InternalCamLocalPlayerVisibility;
  enum class CGamePlayerProfileChunk_GameSettings::ERoadsideSpectatorVisibility {
    Never = 0,
    SpectatorOnly = 1,
    Always = 2,
  };
  CGamePlayerProfileChunk_GameSettings::ERoadsideSpectatorVisibility RoadsideSpectatorVisibility;
  bool UseOldInternalCam;
  bool UseAlternateCam1;
  bool UseAlternateCam2;
  bool ProposeSimpleEditor;
  uint EditorHelp;
  uint UnlockClickCount;
  bool UnlockTalkedToFriend;
  bool UnlockInvitedBuddy;
  bool UnlockHadBuddy;
  bool UnlockSponsoredBuddy;
  bool UnlockCreatedTrack;
  bool UnlockPlayedHotSeat;
  bool UnlockPlayedOnline;
  bool UnlockSentScore;
  float MouseAccelQuantity;
  float MouseSensitivity_Default;
  float MouseSensitivity_Laser;
};

struct CGamePlayerProfileChunk_InterfaceSettings : public CGamePlayerProfileChunk {
  CGamePlayerProfileChunk_InterfaceSettings();

  wstring MenuLeagueFilter;
  uint MenuInternetView;
};

struct CGamePlayerProfileChunk_InputBindingsConfig : public CGamePlayerProfileChunk {
  CGamePlayerProfileChunk_InputBindingsConfig();

  CInputBindingsConfig* Config;
};

struct CGamePlayerProfileChunk_VehiclesSettings : public CGamePlayerProfileChunk {
  CGamePlayerProfileChunk_VehiclesSettings();

  vec3 LightTrailColor;
  const string PrestigeSkinOptions;
};

struct CGamePlayerProfileChunk_OldChallenge : public CGamePlayerProfileChunk {
  CGamePlayerProfileChunk_OldChallenge();

  wstring ChallengeName;
};

struct CGamePlayerProfileChunk_OldCampaign : public CGamePlayerProfileChunk {
  CGamePlayerProfileChunk_OldCampaign();

  CGameCtnCampaign* Campaign;
};

// File extension: 'GameCtnMediaBlockVehicleLight.gbx'
struct CGameCtnMediaBlockVehicleLight : public CGameCtnMediaBlock {
  CGameCtnMediaBlockVehicleLight();

};

struct CGameScoreTask_GetSeasonList : public CWebServicesTaskSequence {
};

struct CGamePlaygroundUIConfigMgrScript : public CMwNod {
  void ResetAll(); // Maniascript
  CGamePlaygroundUIConfig* const UIAll; // Maniascript
  const MwNodPool<CGamePlaygroundUIConfig*> UI; // Maniascript
  CGamePlaygroundUIConfig* GetUI_Player(CGameScriptPlayer* Player); // Maniascript
  CGamePlaygroundUIConfig* GetUI_User(CGamePlayerInfo* User); // Maniascript
  CGamePlaygroundUIConfig* GetUI_ConnectedClient(CGameConnectedClient* Client); // Maniascript
  const MwFastBuffer<CGameUILayer*> UILayers; // Maniascript
  CGameUILayer* UILayerCreate(); // Maniascript
  void UILayerDestroy(CGameUILayer* Layer); // Maniascript
  void UILayerDestroyAll(); // Maniascript
  const MwFastBuffer<CGameUILayer*> UIReplayLayers; // Maniascript
  uint UISequenceMaxDuration; // Maniascript
  bool HoldLoadingScreen; // Maniascript
  const MwNodPool<CGamePlaygroundUIConfigEvent*> PendingEvents; // Maniascript
  CGamePlaygroundUIConfig* const LocalPlayerConfig;
  CGamePlaygroundUIConfig* const LocalPlayerConfig1;
  CGamePlaygroundUIConfig* const LocalPlayerConfig2;
  CGamePlaygroundUIConfig* const LocalPlayerConfig3;
};

// File extension: 'Frame.Gbx'
struct CGamePlaygroundControlSmPlayers : public CControlFrame {
  CGamePlaygroundControlSmPlayers();

  uint ListLineCount;
  uint ListColumnCount;
  CControlFrame* CardModelPlayer;
  void EdClean();
};

// File extension: 'Frame.Gbx'
struct CGamePlaygroundControlMessages : public CControlFrame {
  CGamePlaygroundControlMessages();

  CControlContainer* MessageModel;
  uint LineCount;
  float LineHeight;
  uint DisplayDuration;
  uint MinDuration;
  float LastMessageBaseScale;
  uint ShowEffectDuration;
  float ShowEffectScale;
  uint HideEffectDuration;
  float HideEffectScale;
  float RatioElimination;
};

// Description: "Rules API for gamemodes."
struct CGamePlaygroundScript : public CMwNod {
  enum class CGamePlaygroundScript::EMedal {
    None = 0,
    Finished = 1,
    Bronze = 2,
    Silver = 3,
    Gold = 4,
    Author = 5,
  };
  enum class CGamePlaygroundScript::ESystemPlatform {
    None = 0,
    Steam = 1,
    UPlay = 2,
    PS4 = 3,
    XBoxOne = 4,
    PS5 = 5,
    XBoxSeries = 6,
    Stadia = 7,
    Luna = 8,
  };
  enum class CGamePlaygroundScript::ESystemSkuIdentifier {
    Unknown = 0,
    EU = 1,
    US = 2,
    JP = 3,
    CN = 4,
  };
  const MwFastBuffer<CWebServicesTaskResult*> TaskResults; // Maniascript
  void TaskResult_Release(MwId TaskId); // Maniascript
  CPlugFileTextScript* Script;
  wstring ModeStatusMessage; // Maniascript
  CGameManiaTitle* const LoadedTitle; // Maniascript
  const string ServerLogin; // Maniascript
  const wstring ServerName; // Maniascript
  const wstring ServerModeName; // Maniascript
  const wstring MapName; // Maniascript
  CGameCtnChallenge* const Map; // Maniascript
  const wstring MapPlayerModelName; // Maniascript
  const bool HasPodium; // Maniascript
  const MwFastBuffer<CGamePlayerInfo*> Users; // Maniascript
  const MwFastBuffer<CGameTeamProfile*> Teams; // Maniascript
  string NeutralEmblemUrl; // Maniascript
  const string ForcedClubLinkUrl1; // Maniascript
  const string ForcedClubLinkUrl2; // Maniascript
  void TweakTeamColorsToAvoidHueOverlap(); // Maniascript
  string ClientManiaAppUrl; // Maniascript
  const uint Now; // Maniascript
  const uint Period; // Maniascript
  bool MatchEndRequested; // Maniascript
  const bool ServerShutdownRequested; // Maniascript
  const bool MapLoaded; // Maniascript
  void RequestLoadMap(); // Maniascript
  void RequestUnloadMap(); // Maniascript
  const MwFastBuffer<CGameCtnChallengeInfo*> MapList; // Maniascript
  uint NextMapIndex; // Maniascript
  CGamePlaygroundUIConfigMgrScript* const UIManager; // Maniascript
  void Hud_Load(wstring ModuleName); // Maniascript
  CGamePlaygroundModuleServerHud* const Hud; // Maniascript
  void PassOnModuleEvent(CGamePlaygroundUIConfigEvent* EventToPassOn); // Maniascript
  void DiscardModuleEvent(CGamePlaygroundUIConfigEvent* EventToDiscard); // Maniascript
  void Ladder_OpenMatch_Request(); // Maniascript
  void Ladder_AddPlayer(CGamePlaygroundScore* PlayerScore); // Maniascript
  void Ladder_OpenMatch_BeginRequest(); // Maniascript
  void Ladder_OpenMatch_AddPlayer(CGamePlaygroundScore* PlayerScore); // Maniascript
  void Ladder_OpenMatch_EndRequest(); // Maniascript
  void Ladder_CloseMatchRequest(); // Maniascript
  void Ladder_CancelMatchRequest(); // Maniascript
  const bool Ladder_RequestInProgress; // Maniascript
  void Ladder_SetResultsVersion(uint Version); // Maniascript
  void Ladder_SetMatchMakingMatchId(uint MatchId); // Maniascript
  void Ladder_EnableChallengeMode(bool Enable); // Maniascript
  void Trophy_CompetitionMatch_AddResult(string WebServicesUserId, uint MatchRank, uint TrophyRanking); // Maniascript
  void Trophy_CompetitionMatch_ClearResultList(); // Maniascript
  CWebServicesTaskResult_AccountTrophyGainListScript* Trophy_CompetitionMatch_SendResultList(string CompetitionName, string CompetitionStage, string CompetitionStageStep, wstring CompetitionMatchInfo); // Maniascript
  void Trophy_LiveTimeAttackAchievement_AddResult(string WebServicesUserId, uint MatchRank, uint TrophyRanking); // Maniascript
  void Trophy_LiveTimeAttackAchievement_ClearResultList(); // Maniascript
  CWebServicesTaskResult_AccountTrophyGainListScript* Trophy_LiveTimeAttackAchievement_SendResultList(uint Duration); // Maniascript
  bool Admin_KickUser(CGamePlayerInfo* User, wstring Reason); // Maniascript
  void Admin_SetLobbyInfo(bool IsLobby, int LobbyPlayerCount, int LobbyMaxPlayerCount, float LobbyPlayersLevel); // Maniascript
  CGameScriptServerAdmin* const ServerAdmin; // Maniascript
  void AutoTeamBalance(); // Maniascript
  void Solo_SetNewRecord(CGamePlaygroundScore* PlayerScore, CGamePlaygroundScript::EMedal PlayerScore); // Maniascript
  const bool Solo_NewRecordSequenceInProgress; // Maniascript
  CGameServerScriptXmlRpc* const XmlRpc; // Maniascript
  CXmlScriptParsingManager* const Xml; // Maniascript
  CNetScriptHttpManager* const Http; // Maniascript
  CInputScriptManager* const Input; // Maniascript
  CGameDataFileManagerScript* const DataFileMgr; // Maniascript
  CGameScoreAndLeaderBoardManagerScript* const ScoreMgr; // Maniascript
  CSystemPlatformScript* const System; // Maniascript
  const CGamePlaygroundScript::ESystemPlatform SystemPlatform; // Maniascript
  const CGamePlaygroundScript::ESystemSkuIdentifier SystemSkuIdentifier; // Maniascript
  CGameUserManagerScript* const UserMgr; // Maniascript
  int Synchro_AddBarrier(); // Maniascript
  bool Synchro_BarrierReached(int Barrier); // Maniascript
  bool Users_AreAllies(CGamePlayerInfo* User1, CGamePlayerInfo* User2); // Maniascript
  void Users_RequestSwitchToSpectator(CGamePlayerInfo* User); // Maniascript
  CGamePlayerInfo* Users_CreateFake(wstring NickName, int RequestedTeam); // Maniascript
  void Users_DestroyFake(CGamePlayerInfo* User); // Maniascript
  void Users_SetNbFakeUsers(int NbTeam1, int NbTeam2); // Maniascript
  void Users_DestroyAllFakes(); // Maniascript
  uint Users_EdNbFakeUsers;
  void ItemList_Begin(); // Maniascript
  bool ItemList_Begin2(); // Maniascript
  MwId ItemList_Add(wstring ModelName); // Maniascript
  MwId ItemList_AddWithSkin(wstring ModelName, wstring SkinNameOrUrl); // Maniascript
  MwId ItemList_AddWithSkin2(wstring ModelName, wstring SkinNameOrUrl, string SkinOptions); // Maniascript
  void ItemList_End(); // Maniascript
  void DemoToken_StartUsingToken(); // Maniascript
  void DemoToken_StopUsingToken(); // Maniascript
  void DemoToken_GetAndUseToken(CGamePlayerInfo* User); // Maniascript
  void ActionList_Begin(); // Maniascript
  bool ActionList_Begin2(); // Maniascript
  MwId ActionList_Add(wstring ActionName); // Maniascript
  void ActionList_End(); // Maniascript
  bool UseMinimap; // Maniascript
  bool Replay_AutoStart; // Maniascript
  void Replay_Start(); // Maniascript
  void Replay_Stop(); // Maniascript
  CGameScriptMgrTurret* const TurretsManager; // Maniascript
  CGameScriptMgrVehicle* const VehiclesManager; // Maniascript
  CGameMgrAction* const ActionsManager; // Maniascript
  void Activity_Match_Create_Begin(string ActivityId); // Maniascript
  void Activity_Match_Create_AddPlayer(string WebServicesUserId, string TeamName); // Maniascript
  void Activity_Match_Create_AddTeam(string TeamName); // Maniascript
  void Activity_Match_Create_End(); // Maniascript
  void Activity_Match_ReportResult_Begin(); // Maniascript
  void Activity_Match_ReportResult_SetPlayerResult(string WebServicesUserId, uint Rank, uint Score); // Maniascript
  void Activity_Match_ReportResult_SetTeamResult(string TeamName, uint Rank, uint Score); // Maniascript
  void Activity_Match_ReportResult_End(); // Maniascript
  bool EnableGhostRecording; // Maniascript
  const MwFastBuffer<CGameGhostScript*> Ghosts; // Maniascript
  void Ghost_Release(MwId GhostId); // Maniascript
  void GhostDriver_Playlist_Clear(CGameScriptPlayer* Player); // Maniascript
  void GhostDriver_Playlist_Add(CGameScriptPlayer* Player, CGameGhostScript* Ghost); // Maniascript
  void GhostDriver_UploadLimits_Begin(); // Maniascript
  void GhostDriver_UploadLimits_AddLevel(int TeamLevel); // Maniascript
  CWebServicesTaskResult_GhostDriver_UploadLimits* GhostDriver_UploadLimits_End(); // Maniascript
  void GhostDriver_Upload_Begin(int TeamLevel); // Maniascript
  void GhostDriver_Upload_AddTeamMember(CGameGhostScript* Ghost); // Maniascript
  void GhostDriver_Upload_TeamMember_Begin(); // Maniascript
  void GhostDriver_Upload_AddGhost(CGameGhostScript* Ghost); // Maniascript
  void GhostDriver_Upload_TeamMember_End(); // Maniascript
  CWebServicesTaskResult* GhostDriver_Upload_End(); // Maniascript
  void GhostDriver_Download_Begin(); // Maniascript
  void GhostDriver_Download_AddRange(int TeamLevelMin, int TeamLevelMax, int Count); // Maniascript
  CWebServicesTaskResult_GhostDriver_Download* GhostDriver_Download_End(); // Maniascript
  CWebServicesTaskResult_MapRecordListScript* MapRecord_GetListByMapAndPlayerList(MwId UserId, MwFastBuffer<wstring>& WebServicesUserIdList, string MapUid, string ScopeType, string ScopeId, string GameMode, string GameModeCustomData); // Maniascript
  void GameScene_ResetAll(); // Maniascript
  wstring Dbg_DumpDeclareForVariables(CMwNod* Nod, bool StatsOnly); // Maniascript
  const uint UiUpdatePeriod; // Maniascript
};

// File extension: 'GameCtnMediaBlockFxCameraMap.gbx'
struct CGameCtnMediaBlockFxCameraMap : public CGameCtnMediaBlock {
  CGameCtnMediaBlockFxCameraMap();

  wstring FileName;
};

struct CGameScoreTask_GetAccountTrophyGainHistory : public CWebServicesTaskSequence {
};

struct CGameControlCameraTrackManiaRace : public CGameControlCamera {
  CGameControlCameraTrackManiaRace();

};

struct CGamePlaygroundBasic : public CGamePlayground {
};

// Description: "This is the Maniaplanet plugins Manialink interface."
struct CGameScriptHandlerManiaPlanetPlugin : public CGameManialinkScriptHandler {
  CGameManiaPlanetScriptAPI* const ManiaPlanet; // Maniascript
  CGameManiaplanetPlugin* const ParentApp; // Maniascript
  CGamePluginInterfacesScript* const Plugins; // Maniascript
};

struct CGamePlayerProfileChunk_PackagesInfos : public CGamePlayerProfileChunk {
  CGamePlayerProfileChunk_PackagesInfos();

  const uint PackagesInfosCount;
};

struct CGameCtnMediaBlockShoot : public CGameCtnMediaBlock {
  CGameCtnMediaBlockShoot();

};

struct CGamePlayerProfileChunk_GameStats : public CGamePlayerProfileChunk {
  CGamePlayerProfileChunk_GameStats();

  const uint TotalTimePlay;
  const uint TotalTimeInSolo;
  const uint TotalTimeInSoloRace;
  const uint TotalTimeInSoloPuzzle;
  const uint TotalTimeInSoloPlatform;
  const uint TotalTimeInSoloScript;
  const uint TotalTimeInSplitScreen;
  const uint TotalTimeInHotSeat;
  const uint TotalTimeInNetwork;
  const uint TotalTimeInNetworkTimeAttack;
  const uint TotalTimeInNetworkRounds;
  const uint TotalTimeInNetworkLaps;
  const uint TotalTimeInNetworkStunts;
  const uint TotalTimeInNetworkCup;
  const uint TotalTimeInNetworkScript;
  const uint TotalTimeInEditChallenge;
  const uint TotalTimeInEditReplay;
  const uint TotalTimeInEditSkin;
  const uint TotalTimeInManiaLink;
  const uint TotalNbReset;
  const uint TotalNbFinish;
  const uint TotalNbChallenges;
  const uint AverageTimePlay;
  const uint AverageTimeInSolo;
  const uint AverageTimeInSoloRace;
  const uint AverageTimeInSoloPuzzle;
  const uint AverageTimeInSoloPlatform;
  const uint AverageTimeInSoloScript;
  const uint AverageTimeInSplitScreen;
  const uint AverageTimeInHotSeat;
  const uint AverageTimeInNetwork;
  const uint AverageTimeInNetworkTimeAttack;
  const uint AverageTimeInNetworkRounds;
  const uint AverageTimeInNetworkLaps;
  const uint AverageTimeInNetworkStunts;
  const uint AverageTimeInNetworkCup;
  const uint AverageTimeInNetworkScript;
  const uint AverageTimeInEditChallenge;
  const uint AverageNbReset;
  const uint AverageNbFinish;
  const uint MaxTimePlay;
  const uint MaxTimeInSolo;
  const uint MaxTimeInSoloRace;
  const uint MaxTimeInSoloPuzzle;
  const uint MaxTimeInSoloPlatform;
  const uint MaxTimeInSoloScript;
  const uint MaxTimeInSplitScreen;
  const uint MaxTimeInHotSeat;
  const uint MaxTimeInNetwork;
  const uint MaxTimeInNetworkTimeAttack;
  const uint MaxTimeInNetworkRounds;
  const uint MaxTimeInNetworkLaps;
  const uint MaxTimeInNetworkStunts;
  const uint MaxTimeInNetworkCup;
  const uint MaxTimeInNetworkScript;
  const uint MaxTimeInEditChallenge;
  const uint MaxNbReset;
  const uint MaxNbFinish;
  const wstring MostPlayed;
  const wstring MostRaced;
  const wstring MostEdited;
  const wstring MostNetted;
  const MwFastBuffer<CGamePlayerProfileChunk_ChallengesStats*> ChallengesStatsChunks;
};

struct CGamePlayerProfileChunk_ChallengesStats : public CGamePlayerProfileChunk {
  CGamePlayerProfileChunk_ChallengesStats();

  const uint ChallengesStatsCount;
};

struct CGameCtnMediaBlockSkel : public CGameCtnMediaBlock {
  CGameCtnMediaBlockSkel();

};

struct CGameEditorAnimClip : public CMwNod {
  CGameEditorAnimChar_Interface* const CharEditorInterface;
  CGameEditorAnimChar* const CharEditor;
  float ParticlesPosX;
  float ParticlesPosY;
  float ParticlesPosZ;
  float ParticlesVelX;
  float ParticlesVelY;
  float ParticlesVelZ;
  float ParticlesAnglesYaw;
  float ParticlesAnglesPitch;
  float ParticlesAnglesRoll;
  bool ParticlesOnMesh;
  bool UseCustomColor;
  float ParticlesScale; // Range: 0 - 5
  bool CustomKeyLinearHue;
  bool CustomKeyPos;
  bool CustomKeyAngles;
  bool CustomKeyVel;
  bool KeyVelInLocal;
  bool CustomKeyVolume;
  bool CustomKeyPitch;
  uint SoundVariant;
  float SoundVolume; // Range: 0 - 5
  float Pitch; // Range: 0 - 4
  bool Preview;
  float SoundPosX;
  float SoundPosY;
  float SoundPosZ;
  const wstring SpriteTextureFid;
  float SpritesPosX;
  float SpritesPosY;
  float SpritesPosZ;
  float SpritesSize; // Range: 0 - 25
  bool SpritesIsFree;
  bool UseRandomSoundVar;
  const string SoundVarName;
  wstring ActionDescription;
};

// File extension: 'EDTransition.Gbx'
struct CGameCtnBlockInfoTransition : public CGameCtnBlockInfo {
  CGameCtnBlockInfoTransition();

};

// File extension: 'ZoneTransition.Gbx'
struct CGameCtnZoneTransition : public CGameCtnZone {
  CGameCtnZoneTransition();

  enum class CGameCtnZoneTransition::EZoneTransitionType {
    Custom = 0,
    TriZone = 1,
    Overlap = 2,
  };
  CGameCtnBlockInfoTransition* BlockInfoTransition;
  CGameCtnZoneTransition::EZoneTransitionType TransitionType;
  MwId FrontierId_0To1;
  MwId FrontierId_0To2;
  MwId FrontierId_1To2;
  MwId FrontierId_Base;
  MwId FrontierId_Overlap;
  MwId FrontierId_BaseToOverlap;
  MwId ReplacementZoneId;
  UnnamedEnum Border_North;
  CGameCtnZoneGenealogy* const Genealogy_North;
  UnnamedEnum Border_East;
  CGameCtnZoneGenealogy* const Genealogy_East;
  UnnamedEnum Border_South;
  CGameCtnZoneGenealogy* const Genealogy_South;
  UnnamedEnum Border_West;
  CGameCtnZoneGenealogy* const Genealogy_West;
};

struct CGameEditorTrigger : public CMwNod {
  CGameEditorTrigger();

};

// File extension: ''
struct CGameCtnZoneFusionInfo : public CMwNod {
  CGameCtnZoneFusionInfo();

  MwId CompatibleZoneId;
  MwId MergedZoneId;
  UnnamedEnum FusionType;
};

// File extension: 'Frame.Gbx'
struct CGameControlCardBuddy : public CGameControlCard {
  CGameControlCardBuddy();

  const wstring StrLoginOrNickName;
  const wstring StrStatus;
  CPlugBitmap* AvatarOrLeague;
};

struct CGameBuddy : public CMwNod {
  CGameBuddy();

  string Login;
  wstring NickName;
  wstring Path;
  uint SkillsRank;
  uint SkillsPoints;
  uint LadderRank;
  uint LadderPoints;
  bool Invited;
  bool WaitingConfirmation;
  bool CanReceiveMessages;
  bool IsUsed;
  bool IsOnline;
  CGameNetServerInfo* Server;
  CGameAvatar* Avatar;
  CGameLeague* League;
};

struct CGameEditorPluginMapMapType : public CGameEditorPluginMap {
  enum class CGameEditorPluginMapMapType::EValidationStatus {
    NotValidable = 0,
    Validable = 1,
    Validated = 2,
  };
  bool CustomEditAnchorData; // Maniascript
  void ClearMapMetadata(); // Maniascript
  CGameEditorPluginMapMapType::EValidationStatus ValidationStatus; // Maniascript
  wstring ValidabilityRequirementsMessage; // Maniascript
  bool ValidationEndRequested; // Maniascript
  bool ValidationEndNoConfirm; // Maniascript
  void RequestEnterPlayground(); // Maniascript
  void RequestLeavePlayground(); // Maniascript
  const bool IsSwitchedToPlayground; // Maniascript
  CGamePlaygroundUIConfigMgrScript* const UIManager; // Maniascript
  const MwFastBuffer<CGamePlayerInfo*> Users; // Maniascript
};

struct CGameCursorItem : public CMwNod {
  CGameCursorItem();

  CSceneMobil* const HelperMobil;
  const vec3 CurrentPos;
  float MagnetSnapping_LocalRotation_Deg;
};

struct CGameCtnMacroBlockJunction : public CMwNod {
  CGameCtnMacroBlockJunction();

  enum class CGameCtnMacroBlockJunction::ECardDir {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
  };
  int OffsetX;
  int OffsetY;
  int OffsetZ;
  CGameCtnMacroBlockJunction::ECardDir Dir;
  bool CanBeEntry;
  bool CanBeExit;
};

struct CGameActionMaker : public CGameEditorAsset {
  void Undo();
  void Redo();
  void Exit();
  void ModelLoad();
  void ActionNew();
  void ActionOpen();
  void ActionSaveAs();
  void SwitchFullScreen();
  void BlockSoundInsert();
  void BlockParticleInsert();
  void BlockSpriteInsert();
  void BlockRemove();
  void AdditionalTools();
  void SwitchToBlockMode();
  void SwitchToAnimMode();
  void SwitchToScriptMode();
  void AnimEdit();
  void AnimValidate();
  void BlocAnimValidate();
  void ScriptParamProjAdd();
  void ScriptParamProjSuppr();
  void ScriptParamProjEdit();
  void ScriptParamAnimEdit();
  void ScriptParamProjNext();
  void ScriptParamProjPrev();
  void ScriptParamUpdate();
  void ScriptParamCompil();
  void ScriptAdvancedParamCompil();
  void SoundTypePrev();
  void SoundTypeNext();
  void SoundVarPrev();
  void SoundVarNext();
  void ImportBullet();
  void ExportBullet();
  void Help();
  void ScriptAddAnimEffect();
  void ScriptAddProjEffect();
  void ScriptAddShieldEffect();
  void CreateResources();
  void ImportResources();
  void LibraryValidate();
  void AutoGeneratedScript();
  void ShowScript();
  void ShowBulletEditor();
  void TestMode();
  void SwitchBulletState();
  void SubModelShowList();
  void SwitchSubModelVisibility();
  void SoloMode();
  void ModelValidate();
  void NextBulletPattern();
  void PrevBulletPattern();
  void NextBulletType();
  void PrevBulletType();
  void NextLaserType();
  void PrevLaserType();
  void EditModeBullet();
  void EditModeExplosion();
  void EditModeSound();
  void EditModeGameplay();
  void EditModeGauge();
  void SubModelTypeCore();
  void SubModelTypeTrail();
  void SubModelTypePower();
  void SubModelTypeSmokeDebris();
  void ProjectileSoundModeShooting();
  void ProjectileSoundModeExplosion();
  void ProjectileSoundModeProjectile();
  void ProjectileSoundModeRebound();
  void ProjectileSoundModeHoming();
  void ParamGameplayModeBullet();
  void ParamGameplayModeExplosion();
  void ParamLoadMesh();
  void DrawHelp();
  void SpriteTextureImport();
  CGameEditorAnimClip* const ClipEditor;
  CGameEditorBullet* const BulletEditor;
  CPlugAnimFile* const AnimFile;
  uint ClipIndex;
  CGameActionModel* Action;
  void TestMode_Start();
};

struct CGameCtnEditorScriptAnchoredObject : public CMwNod {
  enum class CGameCtnEditorScriptAnchoredObject::ECardinalDirections {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
  };
  const vec3 Position; // Maniascript
  CGameItemModel* const ItemModel; // Maniascript
};

// Description: "Landmark of a map."
struct CGameCtnEditorScriptSpecialProperty : public CMwNod {
  enum class CGameCtnEditorScriptSpecialProperty::EWaypointType {
    Start = 0,
    Finish = 1,
    Checkpoint = 2,
    None = 3,
    StartFinish = 4,
    Dispenser = 5,
  };
  enum class CGameCtnEditorScriptSpecialProperty::EMapElemColor {
    Default = 0,
    White = 1,
    Green = 2,
    Blue = 3,
    Red = 4,
    Black = 5,
  };
  const string DefaultTag; // Maniascript
  const uint DefaultOrder; // Maniascript
  string Tag; // Maniascript
  uint Order; // Maniascript
  string RuntimeCustomMarker; // Maniascript
  CGameCtnBlock* const Block; // Maniascript
  CGameCtnEditorScriptAnchoredObject* const Item; // Maniascript
  const CGameCtnEditorScriptSpecialProperty::EWaypointType WaypointType; // Maniascript
  const vec3 Position; // Maniascript
};

struct CGameCtnBlockInfoVariant : public CMwNod {
  enum class CGameCtnBlockInfoVariant::ECardinalDirEnum {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
  };
  enum class CGameCtnBlockInfoVariant::EMultiDirEnum {
    SameDir = 0,
    SymmetricalDirs = 1,
    AllDir = 2,
    OpposedDirOnly = 3,
    PerpendicularDirsOnly = 4,
    NextDirOnly = 5,
    PreviousDirOnly = 6,
  };
  enum class CGameCtnBlockInfoVariant::EVariantBaseTypeEnum {
    Inherit = 0,
    None = 1,
    Conductor = 2,
    Generator = 3,
  };
  void ResetVariantCompletely();
  MwFastBuffer<CGameCtnBlockUnitInfo*> BlockUnitInfos;
  const MwId BlockInfoIdForLog;
  string Name; // Maniascript
  const bool IsAllUnderground; // Maniascript
  const bool IsPartUnderground; // Maniascript
  const nat3 Size; // Maniascript
  CPlugPrefab* HelperPrefabFid;
  CPlugPrefab* const HelperPrefabCache;
  CPlugSolid* HelperSolidFid;
  CPlugSolid* FacultativeHelperSolidFid;
  CPlugSurface* WaypointTriggerShape;
  CPlugSurface* ScreenInteractionTriggerShape;
  CMwNod* DeprecWaypointTriggerSolid;
  CMwNod* DeprecScreenInteractionTriggerSolid;
  CGameGateModel* Gate;
  CGameTeleporterModel* Teleporter;
  CGameTurbineModel* Turbine;
  CPlugFlockModel* FlockModel;
  MwFastBuffer<CPlugEntitySpawner*> EntitySpawners;
  float FlockEmitterRadius;
  float FlockEmitterPower;
  uint FlockEmitterSpawnCount;
  bool FlockEmitterIsRepulsor;
  bool FlockEmitterIsLandingArea;
  iso4 FlockEmitterLoc;
  CPlugSpawnModel* SpawnModel;
  MwId DefaultPlacementItem;
  CPlugProbe* Probe;
  bool HasManualSymmetryH;
  bool HasManualSymmetryV;
  bool HasManualSymmetryD1;
  bool HasManualSymmetryD2;
  const bool HasVolumeSymmetryH;
  const bool HasVolumeSymmetryV;
  const bool HasVolumeSymmetryD1;
  const bool HasVolumeSymmetryD2;
  const bool HasFreeClips;
  int SymmetricalVariantIndex;
  CGameCtnBlockInfoVariant::ECardinalDirEnum CardinalDir;
  int NoPillarBelowIndex;
  bool IsObsoleteVariant;
  bool IsFakeReplacement;
  bool AutoChangeVariantOff;
  bool DontPlaySound1;
  bool DontPlaySound2;
  bool IsNoPillarBelowVariant;
  CGameCtnBlockInfoVariant::EMultiDirEnum MultiDir;
  vec3 SpawnTrans;
  float SpawnTransX;
  float SpawnTransY;
  float SpawnTransZ;
  float SpawnYaw; // Range: -180 - 180
  float SpawnPitch; // Range: -90 - 90
  float SpawnRoll; // Range: -180 - 180
  const int3 OffsetBoundingBoxMin; // Maniascript
  const int3 OffsetBoundingBoxMax; // Maniascript
  MwFastBuffer<CMwNod*> Mobils00;
  MwFastBuffer<CMwNod*> Mobils01;
  MwFastBuffer<CMwNod*> Mobils02;
  MwFastBuffer<CMwNod*> Mobils03;
  MwFastBuffer<CMwNod*> Mobils04;
  MwFastBuffer<CMwNod*> Mobils05;
  MwFastBuffer<CMwNod*> Mobils06;
  MwFastBuffer<CMwNod*> Mobils07;
  MwFastBuffer<CMwNod*> Mobils08;
  MwFastBuffer<CMwNod*> Mobils09;
  MwFastBuffer<CMwNod*> Mobils10;
  MwFastBuffer<CMwNod*> Mobils11;
  MwFastBuffer<CMwNod*> Mobils12;
  MwFastBuffer<CMwNod*> Mobils13;
  MwFastBuffer<CMwNod*> Mobils14;
  MwFastBuffer<CMwNod*> Mobils15;
  CGameCtnBlockInfoVariant::EVariantBaseTypeEnum VariantBaseType;
  void AddNewReplacedPillar();
  MwFastBuffer<CGameCtnBlockInfo*> ReplacedPillarBlockInfo_List;
  MwFastBuffer<nat3> ReplacedPillarOffset2D_List;
  MwFastBuffer<CGameCtnBlockInfoVariant::EMultiDirEnum> ReplacedPillarMultiDir_List;
  MwFastBuffer<bool> ReplacedPillarIsOnFlyingBase_List;
  const MwFastBuffer<bool> IsNewPillarPlacedBelow_List;
  void AddNewPlacedPillar();
  MwFastBuffer<CGameCtnBlockInfo*> PlacedPillarBlockInfo_List;
  MwFastBuffer<nat3> PlacedPillarOffset_List;
  MwFastBuffer<CGameCtnBlockInfoVariant::ECardinalDirEnum> PlacedPillarDir_List;
  CGameObjectPhyCompoundModel* CompoundModel;
  iso4 CompoundLoc;
  const MwFastBuffer<CGameCtnBlockUnitInfo*> BlockUnitModels; // Maniascript
};

struct CGameCtnBlockInfoVariantGround : public CGameCtnBlockInfoVariant {
  CGameCtnBlockInfoVariantGround();

  enum class CGameCtnBlockInfoVariantGround::EnumAutoTerrainPlaceType {
    Auto = 0,
    Force = 1,
    DoNotPlace = 2,
    DoNotDestroy = 3,
    ForceIfNecessary = 4,
  };
  const MwFastBuffer<CGameCtnAutoTerrain*> AutoTerrains;
  bool AutoTerrainWithFrontiers;
  int AutoTerrainHeightOffset;
  CGameCtnBlockInfoVariantGround::EnumAutoTerrainPlaceType AutoTerrainPlaceType;
};

struct CGameCtnBlockInfoVariantAir : public CGameCtnBlockInfoVariant {
  CGameCtnBlockInfoVariantAir();

  CGameCtnBlockInfo* PillarsForCompatibility;
};

struct CGameTeamProfile : public CMwNod {
  wstring Name; // Maniascript
  wstring ZonePath; // Maniascript
  wstring City; // Maniascript
  string EmblemUrl; // Maniascript
  string PresentationManialinkUrl; // Maniascript
  string ClubLinkUrl; // Maniascript
  float ColorLinearHuePrimary; // Range: 0 - 1
  const vec3 ColorPrimaryAsColor;
  vec3 ColorPrimary; // Maniascript
  float ColorLinearHueSecondary; // Range: 0 - 1
  const vec3 ColorSecondaryAsColor;
  vec3 ColorSecondary; // Maniascript
  vec3 ColorUI; // Maniascript
  const wstring ColorText; // Maniascript
  const wstring ColorizedName; // Maniascript
  void UpdateCache();
};

// Description: "Events for Manialink page scripts."
struct CGameManialinkScriptEvent : public CScriptBaseConstEvent {
  enum class CGameManialinkScriptEvent::EType {
    KeyPress = 0,
    MouseClick = 1,
    MouseRightClick = 2,
    MouseOver = 3,
    MouseOut = 4,
    EntrySubmit = 5,
    MenuNavigation = 6,
    PluginCustomEvent = 7,
  };
  enum class CGameManialinkScriptEvent::EMenuNavAction {
    Up = 0,
    Right = 1,
    Left = 2,
    Down = 3,
    Select = 4,
    Cancel = 5,
    PageUp = 6,
    PageDown = 7,
    AppMenu = 8,
    Action1 = 9,
    Action2 = 10,
    Action3 = 11,
    Action4 = 12,
    ScrollUp = 13,
    ScrollDown = 14,
  };
  const CGameManialinkScriptEvent::EType Type; // Maniascript
  const uint KeyCode; // Maniascript
  const string KeyName; // Maniascript
  const string CharPressed; // Maniascript
  const string ControlId; // Maniascript
  CGameManialinkControl* const Control; // Maniascript
  const CGameManialinkScriptEvent::EMenuNavAction MenuNavAction; // Maniascript
  const bool IsActionAutoRepeat; // Maniascript
  const wstring CustomEventType; // Maniascript
  const MwFastBuffer<wstring> CustomEventData; // Maniascript
  const wstring PluginCustomEventType; // Maniascript
  const MwFastBuffer<wstring> PluginCustomEventData; // Maniascript
};

// Description: "API for the plugins of the map editor."
struct CGameEditorPluginMap : public CGameManiaApp {
  enum class CGameEditorPluginMap::ECardinalDirections {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
  };
  enum class CGameEditorPluginMap::ECardinalDirections8 {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
    NorthEast = 4,
    SouthEast = 5,
    SouthWest = 6,
    NorthWest = 7,
  };
  enum class CGameEditorPluginMap::ERelativeDirections {
    Forward = 0,
    RightForward = 1,
    Right = 2,
    RightBackward = 3,
    Backward = 4,
    LeftBackward = 5,
    Left = 6,
    LeftForward = 7,
  };
  enum class CGameEditorPluginMap::EPlaceMode {
    Unknown = 0,
    Terraform = 1,
    Block = 2,
    Macroblock = 3,
    Skin = 4,
    CopyPaste = 5,
    Test = 6,
    Plugin = 7,
    CustomSelection = 8,
    OffZone = 9,
    BlockProperty = 10,
    Path = 11,
    GhostBlock = 12,
    Item = 13,
    Light = 14,
    FreeBlock = 15,
    FreeMacroblock = 16,
  };
  enum class CGameEditorPluginMap::EditMode {
    Unknown = 0,
    Place = 1,
    FreeLook = 2,
    Erase = 3,
    Pick = 4,
    SelectionAdd = 5,
    SelectionRemove = 6,
  };
  enum class CGameEditorPluginMap::EShadowsQuality {
    NotComputed = 0,
    VeryFast = 1,
    Fast = 2,
    Default = 3,
    High = 4,
    Ultra = 5,
  };
  enum class CGameEditorPluginMap::EValidationStatus {
    NotValidable = 0,
    Validable = 1,
    Validated = 2,
  };
  enum class CGameEditorPluginMap::EMapElemColor {
    Default = 0,
    White = 1,
    Green = 2,
    Blue = 3,
    Red = 4,
    Black = 5,
  };
  enum class CGameEditorPluginMap::EPhaseOffset {
    None = 0,
    One8th = 1,
    Two8th = 2,
    Three8th = 3,
    Four8th = 4,
    Five8th = 5,
    Six8th = 6,
    Seven8th = 7,
  };
  enum class CGameEditorPluginMap::EMapElemLightmapQuality {
    Normal = 0,
    High = 1,
    VeryHigh = 2,
    Highest = 3,
    Low = 4,
    VeryLow = 5,
    Lowest = 6,
  };
  enum class CGameEditorPluginMap::EMapElemColorPalette {
    Classic = 0,
    Stunt = 1,
    Red = 2,
    Orange = 3,
    Yellow = 4,
    Lime = 5,
    Green = 6,
    Cyan = 7,
    Blue = 8,
    Purple = 9,
    Pink = 10,
    White = 11,
    Black = 12,
  };
  const MwNodPool<CGameEditorPluginMapScriptEvent*> PendingEvents; // Maniascript
  CGameCtnChallenge* const Map; // Maniascript
  const wstring MapName; // Maniascript
  const wstring MapFileName; // Maniascript
  const bool IsEditorReadyForRequest; // Maniascript
  const bool BackToMainMenuRequested; // Maniascript
  bool HoldLoadingScreen; // Maniascript
  void HideOtherPlugins(); // Maniascript
  void ShowOtherPlugins(); // Maniascript
  void ComputeShadows(); // Maniascript
  void ComputeShadows1(CGameEditorPluginMap::EShadowsQuality ShadowsQuality); // Maniascript
  const CGameEditorPluginMap::EShadowsQuality CurrentShadowsQuality; // Maniascript
  const bool IsUltraShadowsQualityAvailable; // Maniascript
  void DisplayDefaultSetObjectivesDialog(); // Maniascript
  bool Undo(); // Maniascript
  bool Redo(); // Maniascript
  void Help(); // Maniascript
  void Validate(); // Maniascript
  void AutoSave(); // Maniascript
  void Quit(); // Maniascript
  void QuickQuit(); // Maniascript
  void QuitAndSetResult(wstring Type, MwFastBuffer<wstring>& Data); // Maniascript
  void QuickQuitAndSetResult(wstring Type, MwFastBuffer<wstring>& Data); // Maniascript
  void TestMapFromStart(); // Maniascript
  void TestMapFromCoord(int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  void TestMapFromMacroblockInstance(CGameEditorMapMacroBlockInstance* MbInstance); // Maniascript
  void TestMapWithMode(wstring RulesModeName); // Maniascript
  void TestMapWithMode2(wstring RulesModeName, string SettingsXml); // Maniascript
  void TestMapWithMode_SplitScreen(wstring RulesModeName); // Maniascript
  void TestMapWithMode_SplitScreen_n(wstring RulesModeName, uint ScreenCount); // Maniascript
  void TestMapWithMode_SplitScreen_n_m(wstring RulesModeName, uint ScreenCount, uint FakeCount, string SettingsXml); // Maniascript
  void StartTestMapWithMode(wstring RulesModeName); // Maniascript
  bool EnableMapTypeStartTest; // Maniascript
  void SaveMap(wstring FileName); // Maniascript
  void SaveMapGamepadEditor(wstring FileName); // Maniascript
  void SaveMapCompat(wstring FileName, wstring Path); // Maniascript
  CGameEditorPluginMap::EPlaceMode PlaceMode; // Maniascript
  CGameEditorPluginMap::EditMode EditMode; // Maniascript
  CGameEditorPluginMap::EMapElemColor NextMapElemColor; // Maniascript
  CGameEditorPluginMap::EMapElemColorPalette MapElemColorPalette; // Maniascript
  void SetNextMapElemColorPalette(); // Maniascript
  const bool IsColorBlindModeActive; // Maniascript
  const MwFastBuffer<CGameEditorPluginMap::EMapElemColorPalette> MapElemColorPalettes; // Maniascript
  bool ForceMacroblockColor; // Maniascript
  CGameEditorPluginMap::EMapElemColor GetMapElemColorBlock(CGameCtnBlock* Block); // Maniascript
  CGameEditorPluginMap::EMapElemColor GetMapElemColorItem(CGameCtnEditorScriptAnchoredObject* AnchoredObject); // Maniascript
  CGameEditorPluginMap::EMapElemColor GetMapElemColorAnchorData(CGameCtnEditorScriptSpecialProperty* Anchor); // Maniascript
  void SetMapElemColorBlock(CGameCtnBlock* Block, CGameEditorPluginMap::EMapElemColor Color); // Maniascript
  void SetMapElemColorItem(CGameCtnEditorScriptAnchoredObject* AnchoredObject, CGameEditorPluginMap::EMapElemColor Color); // Maniascript
  void SetMapElemColorAnchorData(CGameCtnEditorScriptSpecialProperty* Anchor, CGameEditorPluginMap::EMapElemColor Color); // Maniascript
  wstring GetColorPaletteName(CGameEditorPluginMap::EMapElemColorPalette EColorPalette); // Maniascript
  vec3 GetColorPaletteCurrentColor(CGameEditorPluginMap::EMapElemColorPalette EColorPalette, CGameEditorPluginMap::EMapElemColor EColor); // Maniascript
  vec3 GetColorPaletteColorblindColor(CGameEditorPluginMap::EMapElemColorPalette EColorPalette, CGameEditorPluginMap::EMapElemColor EColor); // Maniascript
  vec3 GetColorPaletteNotColorblindColor(CGameEditorPluginMap::EMapElemColorPalette EColorPalette, CGameEditorPluginMap::EMapElemColor EColor); // Maniascript
  CGameEditorPluginMap::EPhaseOffset NextItemPhaseOffset; // Maniascript
  CGameEditorPluginMap::EPhaseOffset NextMbAdditionalPhaseOffset; // Maniascript
  CGameEditorPluginMap::EMapElemLightmapQuality NextMapElemLightmapQuality; // Maniascript
  bool ForceMacroblockLightmapQuality; // Maniascript
  bool UndergroundMode; // Maniascript
  bool BlockStockMode; // Maniascript
  nat3 CursorCoord; // Maniascript
  CGameEditorPluginMap::ECardinalDirections CursorDir; // Maniascript
  CGameCtnBlockInfo* CursorBlockModel; // Maniascript
  CGameCtnBlockInfo* CursorTerrainBlockModel; // Maniascript
  CGameCtnMacroBlockInfo* CursorMacroblockModel; // Maniascript
  CGameEditorGenericInventory* const Inventory; // Maniascript
  bool HideInventory; // Maniascript
  float CameraVAngle; // Maniascript
  float CameraHAngle; // Maniascript
  float CameraToTargetDistance; // Maniascript
  vec3 CameraTargetPosition; // Maniascript
  const vec3 CameraPosition; // Maniascript
  vec3 TargetedPosition; // Maniascript
  float ThumbnailCameraVAngle; // Maniascript
  float ThumbnailCameraHAngle; // Maniascript
  float ThumbnailCameraRoll; // Maniascript
  float ThumbnailCameraFovY; // Maniascript
  vec3 ThumbnailCameraPosition; // Maniascript
  bool GetRaceCamera(vec3 Position, float Yaw, float Pitch, float Roll, float FovY); // Maniascript
  bool EnableAirMapping; // Maniascript
  bool EnableMixMapping; // Maniascript
  void RemoveAllBlocks(); // Maniascript
  void RemoveAllTerrain(); // Maniascript
  void RemoveAllOffZone(); // Maniascript
  void RemoveAllObjects(); // Maniascript
  void RemoveAll(); // Maniascript
  void RemoveAllBlocksAndTerrain(); // Maniascript
  void ShowCustomSelection(); // Maniascript
  void HideCustomSelection(); // Maniascript
  void CopyPaste_Copy(); // Maniascript
  void CopyPaste_Cut(); // Maniascript
  void CopyPaste_Remove(); // Maniascript
  void CopyPaste_SelectAll(); // Maniascript
  void CopyPaste_ResetSelection(); // Maniascript
  void OpenToolsMenu(); // Maniascript
  void EditMediatrackIngame(); // Maniascript
  void PreloadAllBlocks(); // Maniascript
  void PreloadAllItems(); // Maniascript
  bool CanPlaceBlock(CGameCtnBlockInfo* BlockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir, bool OnGround, uint VariantIndex); // Maniascript
  bool PlaceBlock(CGameCtnBlockInfo* BlockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool CanPlaceGhostBlock(CGameCtnBlockInfo* BlockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool PlaceGhostBlock(CGameCtnBlockInfo* BlockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool CanPlaceBlock_NoDestruction(CGameCtnBlockInfo* BlockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir, bool OnGround, uint VariantIndex); // Maniascript
  bool PlaceBlock_NoDestruction(CGameCtnBlockInfo* BlockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool CanPlaceRoadBlocks(CGameCtnBlockInfo* BlockModel, int3 StartCoord, int3 EndCoord); // Maniascript
  bool PlaceRoadBlocks(CGameCtnBlockInfo* BlockModel, int3 StartCoord, int3 EndCoord); // Maniascript
  bool CanPlaceTerrainBlocks(CGameCtnBlockInfo* BlockModel, int3 StartCoord, int3 EndCoord); // Maniascript
  bool PlaceTerrainBlocks(CGameCtnBlockInfo* BlockModel, int3 StartCoord, int3 EndCoord); // Maniascript
  bool PlaceTerrainBlocks_NoDestruction(CGameCtnBlockInfo* BlockModel, int3 StartCoord, int3 EndCoord); // Maniascript
  bool CanPlaceMacroblock(CGameCtnMacroBlockInfo* MacroblockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool PlaceMacroblock(CGameCtnMacroBlockInfo* MacroblockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool PlaceMacroblock_AirMode(CGameCtnMacroBlockInfo* MacroblockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool CanPlaceMacroblock_NoDestruction(CGameCtnMacroBlockInfo* MacroblockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool PlaceMacroblock_NoDestruction(CGameCtnMacroBlockInfo* MacroblockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool CanPlaceMacroblock_NoTerrain(CGameCtnMacroBlockInfo* MacroblockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool PlaceMacroblock_NoTerrain(CGameCtnMacroBlockInfo* MacroblockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool RemoveMacroblock(CGameCtnMacroBlockInfo* MacroblockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool RemoveMacroblock_NoTerrain(CGameCtnMacroBlockInfo* MacroblockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  CGameCtnBlock* GetBlock(int3 Coord); // Maniascript
  CGameCtnBlock* GetBlockSafe(CGameCtnBlockInfo* BlockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  CGameCtnBlock* GetGhostBlock(CGameCtnBlockInfo* BlockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool IsBlockModelSkinnable(CGameCtnBlockInfo* BlockModel); // Maniascript
  uint GetNbBlockModelSkins(CGameCtnBlockInfo* BlockModel); // Maniascript
  wstring GetBlockModelSkin(CGameCtnBlockInfo* BlockModel, uint SkinIndex); // Maniascript
  bool IsItemModelSkinnable(CGameItemModel* ItemModel); // Maniascript
  uint GetNbItemModelSkins(CGameItemModel* ItemModel); // Maniascript
  wstring GetItemModelSkin(CGameItemModel* ItemModel, uint SkinIndex); // Maniascript
  wstring GetSkinDisplayName(wstring SkinFileName); // Maniascript
  wstring GetBlockSkin(CGameCtnBlock* Block); // Maniascript
  wstring GetBlockSkinBg(CGameCtnBlock* Block); // Maniascript
  wstring GetBlockSkinFg(CGameCtnBlock* Block); // Maniascript
  void SetBlockSkin(CGameCtnBlock* Block, wstring SkinFileName); // Maniascript
  void SetBlockSkins(CGameCtnBlock* Block, wstring BgSkinFileName, wstring FgSkinFileName); // Maniascript
  wstring GetItemSkinBg(CGameCtnEditorScriptAnchoredObject* AnchoredObject); // Maniascript
  wstring GetItemSkinFg(CGameCtnEditorScriptAnchoredObject* AnchoredObject); // Maniascript
  void SetItemSkin(CGameCtnEditorScriptAnchoredObject* AnchoredObject, wstring SkinFileName); // Maniascript
  void SetItemSkins(CGameCtnEditorScriptAnchoredObject* AnchoredObject, wstring BgSkinFileName, wstring FgSkinFileName); // Maniascript
  bool IsSkinForeground_Block(CGameCtnBlockInfo* BlockModel, wstring SkinFileName); // Maniascript
  bool IsSkinForeground_Item(CGameItemModel* ItemModel, wstring SkinFileName); // Maniascript
  bool IsMacroblockModelSkinnable(CGameCtnMacroBlockInfo* BlockModel); // Maniascript
  bool SetMacroblockSkin(CGameEditorMapMacroBlockInstance* Macroblock, wstring SkinFileName); // Maniascript
  bool OpenBlockSkinDialog(CGameCtnBlock* Block); // Maniascript
  bool RemoveBlock(int3 Coord); // Maniascript
  bool RemoveBlockSafe(CGameCtnBlockInfo* BlockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool RemoveGhostBlock(CGameCtnBlockInfo* BlockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool RemoveTerrainBlocks(int3 StartCoord, int3 EndCoord); // Maniascript
  uint GetBlockGroundHeight(CGameCtnBlockInfo* BlockModel, int CoordX, int CoordZ, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  uint GetGroundHeight(int CoordX, int CoordZ); // Maniascript
  int3 GetMouseCoordOnGround(); // Maniascript
  int3 GetMouseCoordAtHeight(uint CoordY); // Maniascript
  CGameCtnBlock* GetStartLineBlock(); // Maniascript
  bool RemoveItem(CGameCtnEditorScriptSpecialProperty* Item); // Maniascript
  void CopyPaste_AddOrSubSelection(int3 StartCoord, int3 EndCoord); // Maniascript
  bool CopyPaste_Symmetrize(); // Maniascript
  uint CopyPaste_GetSelectedCoordsCount(); // Maniascript
  void SaveMacroblock(CGameCtnMacroBlockInfo* MacroblockModel); // Maniascript
  bool CopyPaste_ApplyColorToSelection(CGameEditorPluginMap::EMapElemColor Color); // Maniascript
  bool CopyPaste_IncreaseSelectionPhaseOffset(CGameEditorPluginMap::EPhaseOffset Offset); // Maniascript
  bool CopyPaste_DecreaseSelectionPhaseOffset(CGameEditorPluginMap::EPhaseOffset Offset); // Maniascript
  bool CopyPaste_SetSelectionPhaseOffset(CGameEditorPluginMap::EPhaseOffset Offset); // Maniascript
  bool CopyPaste_ApplyLightmapQualityToSelection(CGameEditorPluginMap::EMapElemLightmapQuality LightmapQuality); // Maniascript
  void CopyPaste_GetLightmapQualityInSelection(); // Maniascript
  const MwFastBuffer<CGameEditorPluginMap::EMapElemLightmapQuality> CopyPaste_GetLightmapQualityInSelection_Results; // Maniascript
  CGameCtnMacroBlockInfo* GetMacroblockModelFromName(wstring MacroblockModelName); // Maniascript
  CGameCtnMacroBlockInfo* GetMacroblockModelFromFilePath(wstring MacroblockModelFilePath); // Maniascript
  CGameCtnBlockInfo* GetTerrainBlockModelFromName(wstring TerrainBlockModelName); // Maniascript
  CGameCtnBlockInfo* GetBlockModelFromName(wstring BlockModelName); // Maniascript
  CGameEditorMapScriptClipList* CreateFrameClipList(); // Maniascript
  CGameEditorMapScriptClipList* CreateFixedClipList(); // Maniascript
  void UnvalidateMetadata(); // Maniascript
  void UnvalidateGameplayInfo(); // Maniascript
  void UnvalidatePlayfield(); // Maniascript
  bool RemoveMacroblock_NoTerrain_NoUnvalidate(CGameCtnMacroBlockInfo* MacroblockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool PlaceMacroblock_NoTerrain_NoUnvalidate(CGameCtnMacroBlockInfo* MacroblockModel, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  void ResetAutoRepeat(); // Maniascript
  void ComputeItemsForMacroblockInstance(CGameEditorMapMacroBlockInstance* MacroBlockInstance); // Maniascript
  const MwFastBuffer<CGameCtnEditorScriptAnchoredObject*> MacroblockInstanceItemsResults; // Maniascript
  void GetConnectResults(CGameCtnBlock* ExistingBlock, CGameCtnBlockInfo* NewBlock); // Maniascript
  void GetConnectResultsBlockToMacroBlock(CGameCtnBlock* ExistingBlock, CGameCtnMacroBlockInfo* NewBlock); // Maniascript
  void GetConnectResultsMacroBlockToBlock(CGameEditorMapMacroBlockInstance* ExistingBlock, CGameCtnBlockInfo* NewBlock); // Maniascript
  void GetConnectResultsMacroBlockToMacroBlock(CGameEditorMapMacroBlockInstance* ExistingBlock, CGameCtnMacroBlockInfo* NewBlock); // Maniascript
  int GetStartBlockCount(bool IncludeMultilaps); // Maniascript
  int GetFinishBlockCount(bool IncludeMultilaps); // Maniascript
  int GetMultilapBlockCount(); // Maniascript
  int GetCheckpointBlockCount(); // Maniascript
  uint GetItemsCountResult(); // Maniascript
  MwFastBuffer<wstring> GetItemsCountRequest; // Maniascript
  CGameEditorMapMacroBlockInstance* CreateMacroblockInstance(CGameCtnMacroBlockInfo* MacroblockModel, nat3 Coord, CGameEditorPluginMap::ECardinalDirections Dir, CGameEditorPluginMap::EMapElemColor Color, bool ForceMacroblockColor); // Maniascript
  CGameEditorMapMacroBlockInstance* CreateMacroblockInstanceWithUserData(CGameCtnMacroBlockInfo* MacroblockModel, nat3 Coord, CGameEditorPluginMap::ECardinalDirections Dir, CGameEditorPluginMap::EMapElemColor Color, bool ForceMacroblockColor, int UserData); // Maniascript
  CGameEditorMapMacroBlockInstance* CreateMacroblockInstanceWithClipList(CGameCtnMacroBlockInfo* MacroblockModel, nat3 Coord, CGameEditorPluginMap::ECardinalDirections Dir, CGameEditorMapScriptClipList* DefaultClipList, CGameEditorPluginMap::EMapElemColor Color, bool ForceMacroblockColor); // Maniascript
  CGameEditorMapMacroBlockInstance* CreateMacroblockInstanceWithClipListAndUserData(CGameCtnMacroBlockInfo* MacroblockModel, nat3 Coord, CGameEditorPluginMap::ECardinalDirections Dir, CGameEditorMapScriptClipList* DefaultClipList, CGameEditorPluginMap::EMapElemColor Color, bool ForceMacroblockColor, int UserData); // Maniascript
  CGameEditorMapMacroBlockInstance* GetMacroblockInstanceFromOrder(uint Order); // Maniascript
  CGameEditorMapMacroBlockInstance* GetMacroblockInstanceFromUnitCoord(int3 Coord); // Maniascript
  CGameEditorMapMacroBlockInstance* GetLatestMacroblockInstance(); // Maniascript
  CGameEditorMapMacroBlockInstance* GetLatestMacroblockInstanceWithOffset(uint Offset); // Maniascript
  CGameEditorMapMacroBlockInstance* GetMacroblockInstanceConnectedToClip(CGameEditorMapScriptClip* Clip); // Maniascript
  bool RemoveMacroblockInstance(CGameEditorMapMacroBlockInstance* MacroblockInstance); // Maniascript
  bool RemoveMacroblockInstanceFromOrder(uint Order); // Maniascript
  bool RemoveMacroblockInstanceFromUnitCoord(uint Order); // Maniascript
  bool RemoveMacroblockInstancesByUserData(int UserData); // Maniascript
  void ResetAllMacroblockInstances(); // Maniascript
  uint GetMaxOrder(); // Maniascript
  bool SetMapType(wstring MapType); // Maniascript
  wstring GetMapType(); // Maniascript
  void SetMapStyle(wstring MapStyle); // Maniascript
  wstring GetMapStyle(); // Maniascript
  void SetMapIsCreatedWithPartyEditor(bool IsCreatedWithGamepadEditor); // Maniascript
  wstring GetAvailableMapName(); // Maniascript
  bool IsMapFileNameAvailable(wstring MapName); // Maniascript
  vec3 GetVec3FromCoord(nat3 Coord); // Maniascript
  CGameEditorPluginCameraAPI* const Camera; // Maniascript
  CGameEditorPluginCursorAPI* const Cursor; // Maniascript
  const MwNodPool<CGameCtnEditorScriptAnchoredObject*> Items; // Maniascript
  const MwFastBuffer<wstring> MediatrackIngameClips; // Maniascript
  uint MediatrackIngameEditedClipIndex; // Maniascript
  const MwFastBuffer<CGameCtnBlock*> Blocks; // Maniascript
  const MwFastBuffer<CGameCtnBlockInfo*> BlockModels; // Maniascript
  const MwFastBuffer<CGameCtnBlockInfo*> TerrainBlockModels; // Maniascript
  const MwFastBuffer<CGameCtnMacroBlockInfo*> MacroblockModels; // Maniascript
  const MwFastBuffer<CGameCtnBlock*> ClassicBlocks; // Maniascript
  const MwFastBuffer<CGameCtnBlock*> TerrainBlocks; // Maniascript
  const MwFastBuffer<CGameCtnBlock*> GhostBlocks; // Maniascript
  const MwNodPool<CGameEditorMapScriptClipList*> FixedClipLists; // Maniascript
  const MwNodPool<CGameEditorMapScriptClipList*> FrameClipLists; // Maniascript
  const MwNodPool<CGameEditorMapScriptClipList*> MacroblockInstanceClipLists; // Maniascript
  const MwNodPool<CGameEditorMapMacroBlockInstance*> MacroblockInstances; // Maniascript
  const MwFastBuffer<CGameEditorPluginMapConnectResults*> ConnectResults; // Maniascript
  const MwNodPool<CGameCtnEditorScriptSpecialProperty*> AnchorData; // Maniascript
  bool DoesAnchorHaveSpawn(CGameCtnEditorScriptSpecialProperty* Anchor); // Maniascript
  MwFastBuffer<nat3> CustomSelectionCoords; // Maniascript
  vec3 CustomSelectionRGB; // Maniascript
  bool EnableEditorInputsCustomProcessing; // Maniascript
  bool EnableCursorShowingWhenInterfaceIsFocused; // Maniascript
  bool HideEditorInterface; // Maniascript
  bool HideBlockHelpers; // Maniascript
  bool ShowPlacementGrid; // Maniascript
  float CursorBrightnessFactor; // Maniascript
  bool HideAlwaysCursorDirectionalArrow; // Maniascript
  const bool IsTesting; // Maniascript
  const bool IsValidating; // Maniascript
  const bool EditorInputIsDown_Menu; // Maniascript
  const bool EditorInputIsDown_SwitchToRace; // Maniascript
  const bool EditorInputIsDown_Undo; // Maniascript
  const bool EditorInputIsDown_Redo; // Maniascript
  const bool EditorInputIsDown_CursorTiltLeft; // Maniascript
  const bool EditorInputIsDown_CursorTiltRight; // Maniascript
  const bool EditorInputIsDown_CursorUp; // Maniascript
  const bool EditorInputIsDown_CursorRight; // Maniascript
  const bool EditorInputIsDown_CursorDown; // Maniascript
  const bool EditorInputIsDown_CursorLeft; // Maniascript
  const bool EditorInputIsDown_CursorRaise; // Maniascript
  const bool EditorInputIsDown_CursorLower; // Maniascript
  const bool EditorInputIsDown_CursorTurn; // Maniascript
  const bool EditorInputIsDown_CursorPick; // Maniascript
  const bool EditorInputIsDown_CursorPlace; // Maniascript
  const bool EditorInputIsDown_CursorDelete; // Maniascript
  const bool EditorInputIsDown_CameraUp; // Maniascript
  const bool EditorInputIsDown_CameraRight; // Maniascript
  const bool EditorInputIsDown_CameraDown; // Maniascript
  const bool EditorInputIsDown_CameraLeft; // Maniascript
  const bool EditorInputIsDown_CameraZoomNext; // Maniascript
  const bool EditorInputIsDown_Camera0; // Maniascript
  const bool EditorInputIsDown_Camera1; // Maniascript
  const bool EditorInputIsDown_Camera3; // Maniascript
  const bool EditorInputIsDown_Camera7; // Maniascript
  const bool EditorInputIsDown_Camera9; // Maniascript
  const bool EditorInputIsDown_PivotChange; // Maniascript
  const bool EditorInputIsDown_CursorTurnSlightly; // Maniascript
  const bool EditorInputIsDown_CursorTurnSlightlyAntiClockwise; // Maniascript
  const bool EditorInputIsDown_IconUp; // Maniascript
  const bool EditorInputIsDown_IconRight; // Maniascript
  const bool EditorInputIsDown_IconDown; // Maniascript
  const bool EditorInputIsDown_IconLeft; // Maniascript
  const bool EditorInputIsDown_RemoveAll; // Maniascript
  const bool EditorInputIsDown_Save; // Maniascript
  const bool EditorInputIsDown_SelectAll; // Maniascript
  const bool EditorInputIsDown_Copy; // Maniascript
  const bool EditorInputIsDown_Cut; // Maniascript
  const bool EditorInputIsDown_Paste; // Maniascript
  const float CollectionSquareSize; // Maniascript
  const float CollectionSquareHeight; // Maniascript
  const uint CollectionGroundY; // Maniascript
  const CGameEditorPluginMap::EValidationStatus ValidationStatus; // Maniascript
  float BleacherSpectatorsFillRatio; // Maniascript
  uint BleacherSpectatorsCount; // Maniascript
  string ManialinkText; // Maniascript
  CGameManialinkPage* const ManialinkPage; // Maniascript
  uint GetInterfaceNumber(CGameCtnCollector* Collector); // Maniascript
  void SetInterfaceNumber(CGameCtnCollector* Collector, uint NewValue); // Maniascript
};

struct CGameCtnMediaBlockCameraEffectScript : public CGameCtnMediaBlockCameraEffect {
  CGameCtnMediaBlockCameraEffectScript();

  const float Now; // Maniascript
  const float A; // Maniascript
  const float B; // Maniascript
  const float C; // Maniascript
  vec3 OffsetPos; // Maniascript
  vec3 OffsetRot; // Maniascript
};

// Description: "API for server control when playing online."
struct CGameScriptServerAdmin : public CMwNod {
  enum class CGameScriptServerAdmin::ESpecMode {
    Selectable = 0,
    SpectatorForced = 1,
    PlayerForced = 2,
    SpectatorSelectable = 3,
    PlayerSelectable = 4,
  };
  CGameCtnNetServerInfo* const ServerInfo; // Maniascript
  void AutoTeamBalance(); // Maniascript
  bool Kick1(CGamePlayerInfo* User, wstring Reason); // Maniascript
  bool Kick2(CGameConnectedClient* Client, wstring Reason); // Maniascript
  bool KickUser(CGamePlayerInfo* User, wstring Reason); // Maniascript
  bool Ban1(CGamePlayerInfo* User, wstring Reason); // Maniascript
  bool Ban2(CGameConnectedClient* Client, wstring Reason); // Maniascript
  bool ForceSpectator1(CGamePlayerInfo* User, CGameScriptServerAdmin::ESpecMode SpecMode); // Maniascript
  bool ForceSpectator2(CGameConnectedClient* Client, CGameScriptServerAdmin::ESpecMode SpecMode); // Maniascript
  bool ForcePlayerRequestedTeam1(CGamePlayerInfo* User, int Team); // Maniascript
  bool ForcePlayerRequestedTeam2(CGameConnectedClient* Client, int Team); // Maniascript
  bool IsDisableChangeTeams; // Maniascript
  bool IsDisableProfileSkins; // Maniascript
  string ConnectFakePlayer(); // Maniascript
  void DisconnectFakePlayer(string Login); // Maniascript
  void SetLobbyInfo(bool IsLobby, int LobbyPlayerCount, int LobbyMaxPlayerCount, float LobbyPlayersLevel); // Maniascript
  void SendToServerAfterMatch(string ServerUrl); // Maniascript
  void CustomizeQuitDialog(string ManialinkPage, string SendToServerUrl, bool ProposeAddToFavorites, uint ForceDelay); // Maniascript
  void Authentication_GetToken(MwId UserId, string AppLogin); // Maniascript
  const bool Authentication_GetTokenResponseReceived; // Maniascript
  const uint Authentication_ErrorCode; // Maniascript
  const string Authentication_Token; // Maniascript
  void SetViewerCount(uint ViewerCount); // Maniascript
  const string PlayerRestrictions; // Maniascript
};

struct CGamePlayerProfileChunk_EditorSettings : public CGamePlayerProfileChunk {
  CGamePlayerProfileChunk_EditorSettings();

  const uint MapEditorCoppersLimit;
};

struct CGameEditorPluginMapScriptEvent : public CGameManiaAppScriptEvent {
  enum class CGameEditorPluginMapScriptEvent::EType {
    LayerCustomEvent = 0,
    KeyPress = 1,
    MenuNavigation = 3,
    CursorSelectionBegin = 4,
    CursorSelectionEnd = 5,
    CursorChange = 6,
    MapModified = 7,
    EditorInput = 8,
    MapSavedOrSaveCancelled = 9,
    EditAnchor = 10,
    EditObjectives = 11,
    StartValidation = 12,
    StartTest = 13,
  };
  enum class CGameEditorPluginMapScriptEvent::EInput {
    Unknown = 0,
    Menu = 1,
    SwitchToRace = 2,
    Undo = 3,
    Redo = 4,
    CursorUp = 5,
    CursorRight = 6,
    CursorDown = 7,
    CursorLeft = 8,
    CursorRaise = 9,
    CursorLower = 10,
    CursorTurn = 11,
    CursorTurnSlightly = 12,
    CursorTurnSlightlyAntiClockwise = 13,
    CursorTiltLeft = 14,
    CursorTiltRight = 15,
    CursorPick = 16,
    CursorPlace = 17,
    CursorDelete = 18,
    CameraUp = 19,
    CameraRight = 20,
    CameraDown = 21,
    CameraLeft = 22,
    CameraZoomNext = 23,
    Camera0 = 24,
    Camera1 = 25,
    Camera3 = 26,
    Camera7 = 27,
    Camera9 = 28,
    PivotChange = 29,
    IconUp = 30,
    IconRight = 31,
    IconDown = 32,
    IconLeft = 33,
    RemoveAll = 34,
    Save = 35,
    SelectAll = 36,
    Copy = 37,
    Cut = 38,
    Paste = 39,
  };
  const CGameEditorPluginMapScriptEvent::EType Type; // Maniascript
  const CGameEditorPluginMapScriptEvent::EInput Input; // Maniascript
  const MwId EditedAnchorDataId; // Maniascript
  const bool IsFromPad; // Maniascript
  const bool IsFromMouse; // Maniascript
  const bool IsFromKeyboard; // Maniascript
  const bool OnlyScriptMetadataModified; // Maniascript
  const bool MapSavedOrSaveCancelled; // Maniascript
};

// File extension: 'GameCtnMediaBlockDirtyLens.gbx'
struct CGameCtnMediaBlockDirtyLens : public CGameCtnMediaBlock {
  CGameCtnMediaBlockDirtyLens();

  void SwitchOn();
};

struct CGameCtnMediaBlockCameraEffectInertialTracking : public CGameCtnMediaBlockCameraEffect {
  CGameCtnMediaBlockCameraEffectInertialTracking();

  bool TrackingEnabled;
  bool AutoFocusEnabled;
  bool AutoZoomEnabled;
  float AutoZoomMinDist;
  float AutoZoomMaxDist;
  float AutoZoomMinFov;
  float AutoZoomMaxFov;
};

struct CGameEditorPacks : public CMwNod {
  const bool PackInfo_IsTitlePack;
  bool PackInfo_IsCreditedPack;
  uint PackInfo_Price;
  const uint PackInfo_PriceIncluded;
  const uint PackInfo_PriceTotal;
  string PackInfo_Url;
  wstring PackInfo_PackName;
  wstring PackInfo_Comment;
  void PackInfo_EditTitle();
  void PackInfo_EditTitle_OnOk();
  void PackInfo_EditContents();
  void PackInfo_TestTitle();
  const wstring Contents_Contents;
  const wstring Contents_IncludedContents;
  void Contents_OnAdd();
  void Contents_OnClear();
  const wstring Title_BaseTitle;
  void Title_ChooseBaseTitle();
  const MwId Title_PlayerModel;
  void Title_OnOpenInExplorer();
  void Title_OnButtonPreview();
  const wstring Title_MusicFolder;
  void Title_ChooseMusicFolder();
  void Title_ClearMusicFolder();
  string Title_CollectionGroup;
  int Title_SortOrder;
  string Title_StationManialinkUrl;
  void Title_ChooseStationManialinkUrl();
  void Title_ClearStationManialinkUrl();
  string Title_BoxCaseManialinkUrl;
  void Title_ChooseBoxCaseManialinkUrl();
  void Title_ClearBoxCaseManialinkUrl();
  const wstring Title_Mod;
  void Title_ChooseMod();
  void Title_ClearMod();
  const wstring Title_FallbackFontFileName;
  const wstring Title_Hud3dFontFileName;
  const wstring TitleMenu_ReplayName;
  void TitleMenu_ChooseReplayName();
  void TitleMenu_ClearReplayName();
  const wstring TitleMenu_MusicName;
  void TitleMenu_ChooseMusicName();
  void TitleMenu_ClearMusicName();
  string TitleMenu_ManialinkUrl;
  void TitleMenu_ChooseManialinkUrl();
  void TitleMenu_ClearManialinkUrl();
  vec3 TitleMenu_Color;
  const wstring TitleEditor_MapType;
  void TitleEditor_ChooseMapType();
  void TitleEditor_ClearMapType();
  const wstring TitleEditor_SimpleEditorMap;
  void TitleEditor_ChooseSimpleEditorMap();
  void TitleEditor_ClearSimpleEditorMap();
  const wstring TitleSolo_Mode;
  void TitleSolo_ChooseMode();
  void TitleSolo_ClearMode();
  const wstring TitleSolo_CampaignName;
  void TitleSolo_ChooseCampaignName();
  void TitleSolo_ClearCampaignName();
  const wstring TitleSolo_PlaylistFileName;
  void TitleSolo_ChoosePlaylistFileName();
  void TitleSolo_ClearPlaylistFileName();
  const wstring TitleNet_Mode;
  void TitleNet_ChooseMode();
  void TitleNet_ClearMode();
  const wstring TitleNet_PlaylistFileName;
  void TitleNet_ChoosePlaylistFileName();
  void TitleNet_ClearPlaylistFileName();
  void OpenDialogChooseName_Yes();
  void OpenDialogChooseFolder_Yes();
};

struct CGameCtnMediaBlockBulletFx_Deprecated : public CGameCtnMediaBlock {
  CGameCtnMediaBlockBulletFx_Deprecated();

};

struct CGameCtnMediaBlockCharVis_Deprecated : public CGameCtnMediaBlock {
  CGameCtnMediaBlockCharVis_Deprecated();

};

struct CGameServerScriptXmlRpc : public CMwNod {
  const MwNodPool<CGameServerScriptXmlRpcEvent*> PendingEvents; // Maniascript
  void SendCallback(wstring Param1, wstring Param2); // Maniascript
  void SendCallbackArray(wstring Type, MwFastBuffer<wstring>& Data); // Maniascript
  void SendCallback_BeginRound(); // Maniascript
  void SendCallback_EndRound(); // Maniascript
};

// File extension: 'GameCtnMediaBlockColoringCapturable.gbx'
struct CGameCtnMediaBlockColoringCapturable : public CGameCtnMediaBlock {
  CGameCtnMediaBlockColoringCapturable();

  uint CapturableIndex;
};

// File extension: 'GameCtnMediaBlockFxCameraBlend.gbx'
struct CGameCtnMediaBlockFxCameraBlend : public CGameCtnMediaBlock {
  CGameCtnMediaBlockFxCameraBlend();

  float Intensity;
};

// File extension: 'GameCoverFlow.Gbx'
struct CGameCoverFlowDesc : public CMwNod {
  CGameCoverFlowDesc();

  CPlugGameSkin* GameSkin;
  CSystemFidsFolder* GameSkinFolder;
  float TimeRemapped;
  CGameManialink3dStyle* Style;
  float SpeedChangeGroup;
  float SpeedZ;
  float TransitionRotateAngleDegPrev;
  float TransitionRotateAngleDegNext;
  const MwFastBuffer<CSceneLocationCamera*> CameraLocs;
  iso4 BoxesGroup_RelLoc;
  iso4 StationsGroup_HomeManialinkRelLoc;
  iso4 StationsGroup_MainRelLoc;
  iso4 StationsGroup_ListRelLoc;
  const string LinearSeparator0;
  float LinearOffsetX;
  float LinearOffsetY;
  float LinearOffsetZ;
  float LinearStepX;
  float LinearStepY;
  float LinearStepZ0;
  float LinearStepZ1;
  float LinearAngleMax0;
  float LinearAngleMax1;
  float LinearSpeedMove;
  float LinearSpeedFocus;
  float LinearBumpY;
  float LinearZoomDistance;
  uint LinearTransitionDuration0;
  uint LinearTransitionDuration1;
  uint LinearTransitionDuration2;
  const string GridSeparator0;
  float GridOffsetX;
  float GridOffsetY;
  float GridOffsetZ;
  float GridStepX;
  float GridStepY;
  float GridStepZ0;
  float GridStepZ1;
  float GridAngleMax0;
  float GridAngleMax1;
  float GridSpeedMove;
  float GridSpeedFocus;
  float GridBumpY;
  float GridZoomDistance;
  uint GridTransitionDuration0;
  uint GridTransitionDuration1;
  uint GridTransitionDuration2;
  const string FreeSeparator0;
  float FreeOffsetX;
  float FreeOffsetY;
  float FreeOffsetZ;
  float FreeStepX;
  float FreeStepY;
  float FreeStepZ0;
  float FreeStepZ1;
  float FreeAngleMax0;
  float FreeAngleMax1;
  float FreeSpeedMove;
  float FreeSpeedFocus;
  float FreeBumpY;
  float FreeZoomDistance;
  uint FreeTransitionDuration0;
  uint FreeTransitionDuration1;
  uint FreeTransitionDuration2;
  const string LinearSeparator1;
  CPlugSound* LinearSound_FocusGained;
  CPlugSound* LinearSound_FocusLoop;
  CPlugSound* LinearSound_FocusLost;
  CPlugSound* LinearSound_Select;
  CPlugSound* LinearSound_Edit;
  CPlugSound* LinearSound_Clear;
  CPlugSound* LinearSound_GroupEnter;
  CPlugSound* LinearSound_GroupLoop;
  CPlugSound* LinearSound_GroupLeave;
  CPlugSound* LinearSound_TransitionBump;
  CPlugSound* LinearSound_TransitionZoomIn;
  CPlugSound* LinearSound_TransitionZoomOut;
  const string GridSeparator1;
  CPlugSound* GridSound_FocusGained;
  CPlugSound* GridSound_FocusLoop;
  CPlugSound* GridSound_FocusLost;
  CPlugSound* GridSound_Select;
  CPlugSound* GridSound_Edit;
  CPlugSound* GridSound_Clear;
  CPlugSound* GridSound_GroupEnter;
  CPlugSound* GridSound_GroupLoop;
  CPlugSound* GridSound_GroupLeave;
  CPlugSound* GridSound_TransitionBump;
  CPlugSound* GridSound_TransitionZoomIn;
  CPlugSound* GridSound_TransitionZoomOut;
  const string FreeSeparator1;
  CPlugSound* FreeSound_FocusGained;
  CPlugSound* FreeSound_FocusLoop;
  CPlugSound* FreeSound_FocusLost;
  CPlugSound* FreeSound_Select;
  CPlugSound* FreeSound_Edit;
  CPlugSound* FreeSound_Clear;
  CPlugSound* FreeSound_GroupEnter;
  CPlugSound* FreeSound_GroupLoop;
  CPlugSound* FreeSound_GroupLeave;
  CPlugSound* FreeSound_TransitionBump;
  CPlugSound* FreeSound_TransitionZoomIn;
  CPlugSound* FreeSound_TransitionZoomOut;
};

struct CGamePlayerProfileChunk_ScriptPersistentTraits : public CGamePlayerProfileChunk {
  CGamePlayerProfileChunk_ScriptPersistentTraits();

  CScriptTraitsPersistent* ScriptPersistentTraits;
};

struct CGamePlayerProfileCompatibilityChunk : public CMwNod {
  CGamePlayerProfileCompatibilityChunk();

  const uint ClassId;
  const string ChunkName;
  const string GameName;
  const string Checksum;
  const uint TimeStamp;
};

// File extension: 'GameCtnMediaBlockColoringBase.gbx'
struct CGameCtnMediaBlockColoringBase : public CGameCtnMediaBlock {
  CGameCtnMediaBlockColoringBase();

  uint BaseIndex;
};

struct CGameCtnMediaClipConfigScriptContext : public CMwNod {
  float SmShieldSizeScale; // Maniascript
};

struct CGameServerScriptXmlRpcEvent : public CScriptBaseConstEvent {
  enum class CGameServerScriptXmlRpcEvent::EType {
    Unknown = 0,
    Callback = 1,
    CallbackArray = 2,
  };
  const CGameServerScriptXmlRpcEvent::EType Type; // Maniascript
  const wstring Param1; // Maniascript
  const wstring Param2; // Maniascript
  const wstring ParamArray1; // Maniascript
  const MwFastBuffer<wstring> ParamArray2; // Maniascript
};

// Description: "This is the Manialink browser interface."
struct CGameScriptHandlerBrowser : public CGameManialinkScriptHandler {
  CGameManiaPlanetScriptAPI* const ManiaPlanet; // Maniascript
  CGameManiaAppBrowser* const ParentApp; // Maniascript
  CGameCtnChallenge* const CurMap; // Maniascript
  void ShowCurMapCard(); // Maniascript
  const bool IsInBrowser; // Maniascript
  void BrowserBack(); // Maniascript
  void BrowserQuit(); // Maniascript
  void BrowserHome(); // Maniascript
  void BrowserReload(); // Maniascript
  wstring BrowserFocusedFrameId; // Maniascript
};

struct CGameCtnEditorBody : public CMwNod {
  void StartBodiesPreview();
  void StopBodiesPreview();
  void SwitchToGhostPathMode();
  void SwitchToStraightPathMode();
  void SwitchToRemoveMode();
  void SwitchToPathSpotsMode();
  void SwitchMarkersVisible();
  float BotSpeed;
  uint GhostSamplingPeriod;
  float ShootmanSpeed;
  void SweepPathsAndSave();
  string BotPathFrameTag;
  uint BotPathFrameOrder;
  uint BotPathFrameKind;
  bool BotPathFrameIsAutonomous;
  bool BotPathFrameIsRecorded;
  void RecordItemPath();
  string BotPathCurrentIndexString;
  const uint BotPathCount;
  void PrevBotPath();
  void NextBotPath();
};

struct CGameManialinkPlayerList : public CGameManialinkControl {
};

struct CGamePlayerProfileChunk_GlobalInterfaceSettings : public CGamePlayerProfileChunk {
  CGamePlayerProfileChunk_GlobalInterfaceSettings();

  uint WarnUserForAvatars;
  bool EnableAvatars;
  bool EnableChat;
  bool AllowChatHiding;
  bool ColorblindMode;
  bool OpenWebLinksInSteamOverlay;
  bool SynchonizeSteamWorkshopFiles;
  bool PreferSteamScreenshots;
};

struct CGameAnimSet : public CMwNod {
  CGameAnimSet();

  const string ZipName;
  CGameAnimClipNod* Idle;
  CGameAnimClipNod* RunF;
  CGameAnimClipNod* RunB;
  CGameAnimClipNod* TurnR_90;
  CGameAnimClipNod* TurnL_90;
  CGameAnimClipNod* Jump;
  CGameAnimClipNod* FrontFlip;
  CGameAnimClipNod* BackFlip;
  CGameAnimClipNod* Air;
  CGameAnimClipNod* AirToIdle;
  CGameAnimClipNod* AirToRunF;
  CGameAnimClipNod* AirToRunB;
  CGameAnimClipNod* Slide;
  CGameAnimClipNod* Glide;
  CGameAnimClipNod* SwimI;
  CGameAnimClipNod* SwimF;
  CGameAnimClipNod* SwimB;
  CGameAnimClipNod* SneakF;
  CGameAnimClipNod* SneakI;
  CGameAnimClipNod* SneakB;
  CGameAnimClipNod* Sneak_TurnR_90;
  CGameAnimClipNod* Sneak_TurnL_90;
  CGameAnimClipNod* Fire;
  CGameAnimClipNod* Gesture;
};

// Description: "Manager of buddies instant messaging."
struct CGameScriptChatManager : public CMwNod {
  enum class CGameScriptChatManager::EConnectionStatus {
    Offline = 0,
    Online = 1,
    Connecting = 2,
  };
  enum class CGameScriptChatManager::EDesiredConnectionStatus {
    Offline = 0,
    Online = 1,
  };
  enum class CGameScriptChatManager::EPresenceShow {
    Available = 0,
    WantToChat = 1,
    DoNotDisturb = 2,
    Away = 3,
    ExtendedAway = 4,
  };
  const MwFastBuffer<CGameScriptChatContact*> DebugContacts;
  const CGameScriptChatManager::EConnectionStatus CurrentConnectionStatus; // Maniascript
  CGameScriptChatManager::EDesiredConnectionStatus DesiredConnectionStatus; // Maniascript
  const bool SteamIsEnabled; // Maniascript
  void SendMessage(CGameScriptChatContact* Buddy, wstring Body); // Maniascript
  void SendMessage2(CGameScriptChatContact* Buddy, wstring Body, wstring Link, wstring Metadata); // Maniascript
  void OpenSteamChat(CGameScriptChatContact* Buddy); // Maniascript
  void PresenceSet(CGameScriptChatManager::EPresenceShow Show, wstring Status); // Maniascript
  const CGameScriptChatManager::EPresenceShow Presence_Show; // Maniascript
  const wstring Presence_Status; // Maniascript
  void Room_SendMessage(CGameScriptChatRoom* Room, wstring Body); // Maniascript
  const string CurrentServerLogin; // Maniascript
  const wstring CurrentServerName; // Maniascript
  const string CurrentServerJoinLink; // Maniascript
  const wstring CurrentServerModeName; // Maniascript
  const wstring CurrentServerModeStatus; // Maniascript
  const uint CurrentServerPlayerCount; // Maniascript
  const uint CurrentServerPlayerCountMax; // Maniascript
  const bool CurrentServer_IsFull; // Maniascript
  const bool CurrentServer_IsFullSpec; // Maniascript
  const bool CurrentServer_IsPrivate; // Maniascript
  const bool CurrentServer_IsPrivateSpec; // Maniascript
  const bool CurrentServer_IsHidden; // Maniascript
  const bool CurrentServer_IsLobby; // Maniascript
  const bool CurrentServer_IsRelay; // Maniascript
  const bool CurrentServer_IsFavourite; // Maniascript
  const bool CurrentServer_IsWarmUp; // Maniascript
  CGameScriptChatContact* GetContactFromLogin(string Login); // Maniascript
  wstring Filter_SearchText; // Maniascript
  bool Filter_PlayingOnline; // Maniascript
  const MwFastBuffer<CGameScriptChatContact*> Contacts; // Maniascript
  const MwFastBuffer<CGameScriptChatContact*> Buddies; // Maniascript
  const MwFastBuffer<CGameScriptChatContact*> FilteredBuddies; // Maniascript
  const MwFastBuffer<CGameScriptChatContact*> OnlineBuddies; // Maniascript
  const MwFastBuffer<CGameScriptChatContact*> PlayingOnlineBuddies; // Maniascript
  const MwFastBuffer<CGameScriptChatContact*> OfflineBuddies; // Maniascript
  const MwFastBuffer<CGameScriptChatContact*> XmppBuddies; // Maniascript
  const MwFastBuffer<CGameScriptChatContact*> XmppOnlineBuddies; // Maniascript
  const MwFastBuffer<CGameScriptChatContact*> XmppOfflineBuddies; // Maniascript
  const MwFastBuffer<CGameScriptChatContact*> SteamBuddies; // Maniascript
  const MwFastBuffer<CGameScriptChatContact*> SteamOnlineBuddies; // Maniascript
  const MwFastBuffer<CGameScriptChatContact*> SteamOfflineBuddies; // Maniascript
  const MwFastBuffer<CGameScriptChatContact*> IncomingBuddyRequests; // Maniascript
  const MwFastBuffer<CGameScriptChatContact*> OutgoingBuddyRequests; // Maniascript
  const MwNodPool<CGameScriptChatEvent*> PendingEvents; // Maniascript
  void AddBuddy(string Login); // Maniascript
  void RemoveBuddy(CGameScriptChatContact* Buddy); // Maniascript
  void AcceptBuddy(CGameScriptChatContact* Buddy); // Maniascript
  void AddBuddySlot(); // Maniascript
  const MwFastBuffer<CGameScriptChatHistory*> Histories; // Maniascript
  CGameScriptChatHistory* History_Create(); // Maniascript
  CGameScriptChatHistory* History_Create_Contact(CGameScriptChatContact* Contact); // Maniascript
  CGameScriptChatHistory* History_Create_Room(CGameScriptChatRoom* Room); // Maniascript
  void History_Destroy(CGameScriptChatHistory* History); // Maniascript
  void History_DestroyAll(); // Maniascript
  const MwFastBuffer<CGameScriptChatRoom*> Channels; // Maniascript
  void Channel_Join(string ChannelName); // Maniascript
  void Channel_Leave(CGameScriptChatRoom* Channel); // Maniascript
  CGameScriptChatRoom* const Squad; // Maniascript
  const wstring Squad_PreferredServer; // Maniascript
  void Squad_Invite(string Login); // Maniascript
  void Squad_Leave(); // Maniascript
  const MwFastBuffer<CGameScriptChatSquadInvitation*> Squad_Invitations; // Maniascript
  void Squad_Invitation_Accept(CGameScriptChatSquadInvitation* Invitation); // Maniascript
  void Squad_Invitation_Decline(CGameScriptChatSquadInvitation* Invitation); // Maniascript
};

// Description: "A buddy from the buddy list."
struct CGameScriptChatContact : public CMwNod {
  enum class CGameScriptChatContact::ESubscriptionStatus {
    None = 0,
    RequestFrom = 1,
    RequestTo = 2,
    Both = 3,
  };
  enum class CGameScriptChatContact::ESubscription {
    None = 0,
    Pending = 1,
    Accepted = 2,
  };
  enum class CGameScriptChatContact::EPresenceShow {
    Offline = 0,
    Available = 1,
    WantToChat = 2,
    DoNotDisturb = 3,
    Away = 4,
    ExtendedAway = 5,
  };
  const string Login; // Maniascript
  const wstring Name; // Maniascript
  const wstring Description; // Maniascript
  const wstring ZonePath; // Maniascript
  const CGameScriptChatContact::EPresenceShow PresenceShow; // Maniascript
  const wstring PresenceStatus; // Maniascript
  const CGameScriptChatContact::ESubscriptionStatus SubscriptionStatus; // Maniascript
  const CGameScriptChatContact::ESubscription Subscription_From; // Maniascript
  const CGameScriptChatContact::ESubscription Subscription_To; // Maniascript
  const bool IsXmpp; // Maniascript
  const bool IsSteam; // Maniascript
  const bool IsLegacy; // Maniascript
  const bool IsOnline; // Maniascript
  const bool IsOnlineInLegacy; // Maniascript
  const bool IsOnlineInXmpp; // Maniascript
  const bool IsOnlineInSteam; // Maniascript
  const bool IsBuddyInManiaPlanet; // Maniascript
  bool IsAlly; // Maniascript
  const string CurrentServerLogin; // Maniascript
  const wstring CurrentServerName; // Maniascript
  const string CurrentServerJoinLink; // Maniascript
  const wstring CurrentServerModeName; // Maniascript
  const wstring CurrentServerModeStatus; // Maniascript
  const uint CurrentServerPlayerCount; // Maniascript
  const uint CurrentServerPlayerCountMax; // Maniascript
  const bool CurrentServer_IsFull; // Maniascript
  const bool CurrentServer_IsFullSpec; // Maniascript
  const bool CurrentServer_IsPrivate; // Maniascript
  const bool CurrentServer_IsPrivateSpec; // Maniascript
  const bool CurrentServer_IsHidden; // Maniascript
  const bool CurrentServer_IsLobby; // Maniascript
  const bool CurrentServer_IsRelay; // Maniascript
  const bool CurrentServer_IsFavourite; // Maniascript
  const bool CurrentServer_IsWarmUp; // Maniascript
  void NotifyInteraction(); // Maniascript
  CPlugBitmap* const Avatar; // Maniascript
  const string AvatarUrl; // Maniascript
};

// Description: "Event from the BuddiesManager."
struct CGameScriptChatEvent : public CScriptBaseConstEvent {
  enum class CGameScriptChatEvent::EType {
    Message = 0,
    ChatRoomMessage = 1,
    PresenceChange = 2,
    BuddyChange = 3,
    AddBuddyResult = 4,
    RemoveBuddyResult = 5,
    JoinRoomResult = 6,
    ConnectionStatusUpdate = 7,
    HistoryUpdate = 8,
    ContactListChange = 9,
    AddBuddySlotResult = 10,
    SquadInviteResult = 11,
    SquadInvitationReceived = 12,
  };
  enum class CGameScriptChatEvent::ESubscriptionStatus {
    None = 0,
    RequestFrom = 1,
    RequestTo = 2,
    Both = 3,
  };
  enum class CGameScriptChatEvent::ESubscription {
    None = 0,
    Pending = 1,
    Accepted = 2,
  };
  enum class CGameScriptChatEvent::EPresenceShow {
    Offline = 0,
    Available = 1,
    WantToChat = 2,
    DoNotDisturb = 3,
    Away = 4,
    ExtendedAway = 5,
  };
  enum class CGameScriptChatEvent::EAddBuddySlotStatus {
    Success = 0,
    AlreadyBuddy = 1,
    NotConnected = 2,
    UnknownLogin = 3,
    RosterFull = 4,
    SubscriptionsFull = 5,
    Timeout = 6,
    Invalid = 7,
    ParentalLock = 8,
  };
  const CGameScriptChatEvent::EType Type; // Maniascript
  CGameScriptChatContact* const Sender; // Maniascript
  const string SenderResource; // Maniascript
  const wstring MessageBody; // Maniascript
  CGameScriptChatRoom* const ChatRoom; // Maniascript
  const wstring SenderNickname; // Maniascript
  CGameScriptChatContact* const Buddy; // Maniascript
  const bool PreviousIsOnline; // Maniascript
  const CGameScriptChatEvent::EPresenceShow PreviousShow; // Maniascript
  const wstring PreviousStatus; // Maniascript
  const string PreviousServerLogin; // Maniascript
  const wstring PreviousServerName; // Maniascript
  const string PreviousServerJoinLink; // Maniascript
  const wstring PreviousServerModeName; // Maniascript
  const wstring PreviousServerModeStatus; // Maniascript
  const uint PreviousServerPlayerCount; // Maniascript
  const uint PreviousServerPlayerCountMax; // Maniascript
  const CGameScriptChatEvent::ESubscriptionStatus PreviousSubscriptionStatus; // Maniascript
  const CGameScriptChatEvent::ESubscription PreviousSubscription_From; // Maniascript
  const CGameScriptChatEvent::ESubscription PreviousSubscription_To; // Maniascript
  const wstring PreviousNickname; // Maniascript
  const wstring PreviousDescription; // Maniascript
  const bool ContactList; // Maniascript
  const bool ContactListBuddies; // Maniascript
  const bool ContactListPlayingOnlineBuddies; // Maniascript
  const bool ContactListOnlineBuddies; // Maniascript
  const bool ContactListOfflineBuddies; // Maniascript
  const bool ContactListXmppBuddies; // Maniascript
  const bool ContactListXmppOnlineBuddies; // Maniascript
  const bool ContactListXmppOfflineBuddies; // Maniascript
  const bool ContactListSteamBuddies; // Maniascript
  const bool ContactListSteamOnlineBuddies; // Maniascript
  const bool ContactListSteamOfflineBuddies; // Maniascript
  const bool ContactListIncomingBuddyRequest; // Maniascript
  const bool ContactListOutgoingBuddyRequest; // Maniascript
  const string BuddyLogin; // Maniascript
  const CGameScriptChatEvent::EAddBuddySlotStatus Status; // Maniascript
  const string RoomName; // Maniascript
  const bool Success; // Maniascript
  const wstring ErrorMessage; // Maniascript
  CGameScriptChatHistory* const History; // Maniascript
  CGameScriptChatSquadInvitation* const Invitation; // Maniascript
};

struct CGameManialink3dMood : public CMwNod {
  CGameManialink3dMood();

  vec3 LAmbient_LinearRgb;
  vec3 LDir0_LinearRgb;
  float LDir0_Intens;
  float LDir0_DirTheta; // Range: 0 - 180
  float LDir0_DirPhi; // Range: -180 - 180
  vec3 LDir1_LinearRgb;
  float LDir1_Intens;
  float LDir1_DirTheta; // Range: 0 - 180
  float LDir1_DirPhi; // Range: -180 - 180
  vec3 LBall_LinearRgb;
  float LBall_Intens;
  float LBall_Radius;
  vec3 CloudsRgbMinLinear;
  float CloudsRgbMinScale;
  vec3 CloudsRgbMaxLinear;
  float CloudsRgbMaxScale;
  vec3 FogColorSrgb;
  float SkyGradientV_Scale;
  CPlugFileImg* FidImageSkyGradientV;
  CPlugFileImg* FidImageCloudsMinMaxDayTime;
  vec3 SelfIllumColor;
};

struct CGameManialink3dWorld : public CMwNod {
  CGameManialink3dWorld();

  CGameManialink3dStyle* Style;
};

struct CGamePlayerProfileChunk_ManiaPlanetStations : public CGamePlayerProfileChunk {
  CGamePlayerProfileChunk_ManiaPlanetStations();

  bool IsFirstLaunch;
  bool HasSeenWelcomePage;
  MwId LatestTitleIdLoaded;
  uint StationsCount;
};

struct CGameManialink3dStyle : public CMwNod {
  CGameManialink3dStyle();

  CGameManialink3dMood* Mood;
};

// Description: "API for Maniaplanet plugins."
struct CGameManiaplanetPlugin : public CGameManiaApp {
  enum class CGameManiaplanetPlugin::EContext {
    MenuStartUp = 0,
    MenuManiaPlanet = 1,
    MenuManiaTitleMain = 2,
    MenuProfile = 3,
    MenuSolo = 4,
    MenuLocal = 5,
    MenuMulti = 6,
    MenuEditors = 7,
    EditorPainter = 8,
    EditorTrack = 9,
    EditorMediaTracker = 10,
    Solo = 11,
    SoloLoadScreen = 12,
    Multi = 13,
    MultiLoadScreen = 14,
    MenuCustom = 15,
    Unknown = 16,
  };
  enum class CGameManiaplanetPlugin::EBuyTitleMode {
    OpenStore = 0,
    BuyIfNeeded = 1,
    Ask = 2,
  };
  enum class CGameManiaplanetPlugin::EUISound {
    Alert = 0,
    ShowDialog = 1,
    HideDialog = 2,
  };
  const MwNodPool<CGameManiaAppScriptEvent*> PendingEvents; // Maniascript
  const MwFastBuffer<CGameManiaTitle*> TitlesAvailable; // Maniascript
  const MwFastBuffer<CGameManiaTitle*> TitlesBases; // Maniascript
  const MwFastBuffer<string> TitleIdsInstalled; // Maniascript
  const MwFastBuffer<string> TitleIdsPayed; // Maniascript
  const uint CurrentAppId; // Maniascript
  const string CurrentServerLogin; // Maniascript
  const wstring CurrentServerName; // Maniascript
  const wstring CurrentServerModeName; // Maniascript
  const string CurrentServerJoinLink; // Maniascript
  const MwFastBuffer<CGamePlayerInfo*> CurrentServerPlayers; // Maniascript
  CGameCtnChallenge* const CurrentMap; // Maniascript
  const CGameManiaplanetPlugin::EContext ActiveContext; // Maniascript
  const string ActiveContext_MenuFrame; // Maniascript
  const string ActiveContext_DialogFrame; // Maniascript
  const string ActiveContext_SystemDialogFrame; // Maniascript
  const bool ActiveContext_ClassicDialogDisplayed; // Maniascript
  const uint ActiveContext_GameWill; // Maniascript
  const bool ActiveContext_ManialinkBrowserDisplayed; // Maniascript
  bool PrivateSession; // Maniascript
  const bool PersistentIsReady; // Maniascript
  CGameScriptChatManager* const ChatManager; // Maniascript
  CGameScriptChatManager* const BuddiesManager; // Maniascript
  CAdvertisingManager* const AdvertisingManager; // Maniascript
  CGameManialinkBrowser* const ManialinkBrowser; // Maniascript
  CGameManiaTitleControlScriptAPI* const TitleControl; // Maniascript
  CGamePlaygroundClientScriptAPI* const Playground; // Maniascript
  CGameMatchSettingsManagerScript* const MatchSettingsManager; // Maniascript
  CGameScriptNotificationsProducer* const NotificationsProducer; // Maniascript
  CGameScriptNotificationsConsumer* const Notifications; // Maniascript
  bool ToolBarForceShow; // Maniascript
  const bool ToolBarIsActive; // Maniascript
  bool ToolBarClockVisible; // Maniascript
  bool ToolBarPlanetsVisible; // Maniascript
  bool ToolBarBrowserVisible; // Maniascript
  bool ToolBarSoundVisible; // Maniascript
  bool ToolBarSettingsVisible; // Maniascript
  int PluginZOrder; // Maniascript
  float MusicVolume; // Range: 0 - 1 // Maniascript
  void HackSetCurMenuControlVisible(string ControlId, bool Visible); // Maniascript
  bool HackIsCurMenuControlVisible(string ControlId); // Maniascript
  bool RequestStartUpFreeze; // Maniascript
  void ClipboardSet(wstring ClipboardText); // Maniascript
  void QuitGameAndOpenLink(string Url); // Maniascript
  void ShowTitle(string TitleId, bool ShowStation); // Maniascript
  void EnterTitle(string TitleId); // Maniascript
  void ShowCurMapCard(); // Maniascript
  CGameDialogs* const SystemDialogs; // Maniascript
  void Dialog_EnterGameKey(); // Maniascript
  void Dialog_BuyTitleDialog(string TitleId, string OverrideBuyUrl, int OverrideActionAfterBuy); // Maniascript
  void Dialog_BuyTitleEx(string TitleId, CGameManiaplanetPlugin::EBuyTitleMode Mode, string OverrideBuyUrl, int OverrideActionAfterBuy); // Maniascript
  void Dialog_GenerateGameKeys(string TitleId, uint FeaturesLevel, uint NbToGenerate, bool AddBuddyOnRegistration); // Maniascript
  const bool Dialog_IsFinished; // Maniascript
  const bool Dialog_Success; // Maniascript
  const bool Dialog_Aborted; // Maniascript
  CNetScriptHttpRequest* CreatePostImage(string Url, string TextData, string HeaderTextData, CPlugBitmap* Image, string AdditionnalHeaders); // Maniascript
  const bool UseAllies; // Maniascript
  const uint TotalTimePlay; // Maniascript
  bool LoadingScreenRequireKeyPressed; // Maniascript
  void SetLocalUserClubLink(string ClubLink); // Maniascript
  void SetLocalUserNickName(wstring NickName); // Maniascript
  void FlashWindow(); // Maniascript
  const uint Messenger_MessagesCount; // Maniascript
  void PlaySound(CGameManiaplanetPlugin::EUISound Sound, uint SoundVariant); // Maniascript
  const string ServerChatLog; // Maniascript
  void ServerChatLog_Copy();
  CGameManiaplanetPluginInterface* const Interface;
  void CustomEvent(wstring Type, MwFastBuffer<wstring>& Data); // Maniascript
  void SendExternalCustomEvent(wstring Type, MwFastBuffer<wstring>& Data); // Maniascript
  CGamePluginInterfacesScript* const Plugins; // Maniascript
  void Dialog_ShowEnterGameKeyDialog();
  void Dialog_ShowBuyTitleDialog();
  void Dialog_ShowGenerateGameKeysDialog();
};

struct CGameCtnInterfaceViewer : public CMwNod {
  bool Hit;
  bool Elimination;
  uint Score;
};

struct CGameManialinkGauge : public CGameManialinkControl {
  string Style; // Maniascript
  void SetRatio(float NewRatio); // Maniascript
  void SetClan(uint NewClan); // Maniascript
  float Ratio; // Range: 0 - 1 // Maniascript
  float GradingRatio; // Range: 0 - 1 // Maniascript
  uint Clan; // Maniascript
  vec3 Color; // Maniascript
  bool DrawBackground; // Maniascript
  bool DrawBlockBackground; // Maniascript
  bool CenteredBar; // Maniascript
};

// File extension: 'GameCtnMediaBlockColorGrading.gbx'
struct CGameCtnMediaBlockColorGrading : public CGameCtnMediaBlock {
  CGameCtnMediaBlockColorGrading();

  CPlugFileImg* Image;
};

// Description: "API for plugins to get playground info."
struct CGamePlaygroundClientScriptAPI : public CMwNod {
  enum class CGamePlaygroundClientScriptAPI::ESpectatorCameraType {
    Replay = 0,
    Follow = 1,
    Free = 2,
    StandardCount = 3,
    NotUsed0 = 4,
    NotUsed1 = 5,
    NotUsed2 = 6,
    NotUsed3 = 7,
    NotUsed4 = 8,
    NotUsed5 = 9,
    NotUsed6 = 10,
    NotUsed7 = 11,
    NotUsed8 = 12,
    NotUsed9 = 13,
    FollowForced = 14,
    DontChange = 15,
  };
  enum class CGamePlaygroundClientScriptAPI::ESpectatorTargetType {
    None = 0,
    Single = 1,
    AllPlayers = 2,
    AllMap = 3,
  };
  CGameCtnChallenge* const Map; // Maniascript
  const int GameTime; // Maniascript
  CGamePlayerInfo* const LocalUser; // Maniascript
  CGamePlaygroundUIConfig* const UI; // Maniascript
  CGameCtnNetServerInfo* const ServerInfo; // Maniascript
  const MwId SettingsPlayerModelId; // Maniascript
  const bool HasPodium; // Maniascript
  const bool IsSpectator; // Maniascript
  const bool IsSpectatorClient; // Maniascript
  const bool UseClans; // Maniascript
  const bool UseForcedClans; // Maniascript
  const bool IsLoadingScreen; // Maniascript
  const bool IsServerOrSolo; // Maniascript
  void QuitServer(bool Silent); // Maniascript
  void QuitServerAndSetResult(bool Silent, wstring Type, MwFastBuffer<wstring>& Data); // Maniascript
  const bool IsInGameMenuDisplayed; // Maniascript
  void JoinTeam1(); // Maniascript
  void JoinTeam2(); // Maniascript
  const MwFastBuffer<CGameTeamProfile*> Teams; // Maniascript
  void RequestSpectatorClient(bool Spectator); // Maniascript
  void SetSpectateTarget(string Player); // Maniascript
  void ShowProfile(string Player); // Maniascript
  bool SaveReplay(wstring FileName); // Maniascript
  bool SavePrevReplay(wstring FileName); // Maniascript
  bool SaveMap(wstring FileName); // Maniascript
  void MapList_Request(); // Maniascript
  CGamePlaygroundClientScriptAPI::ESpectatorCameraType GetSpectatorCameraType(); // Maniascript
  CGamePlaygroundClientScriptAPI::ESpectatorTargetType GetSpectatorTargetType(); // Maniascript
  void SetWantedSpectatorCameraType(CGamePlaygroundClientScriptAPI::ESpectatorCameraType CameraType); // Maniascript
  const bool MapList_IsInProgress; // Maniascript
  const MwFastBuffer<wstring> MapList_Names; // Maniascript
  const MwFastBuffer<string> MapList_MapUids; // Maniascript
  const bool Request_IsInProgress; // Maniascript
  const bool Request_Success; // Maniascript
  void RequestRestartMap(); // Maniascript
  void RequestNextMap(); // Maniascript
  void RequestGotoMap(string MapUid); // Maniascript
  void RequestSetNextMap(string MapUid); // Maniascript
  void RequestAutoTeamBalance(); // Maniascript
  void RequestChangeModeScriptSettings(string SettingsXml); // Maniascript
  const wstring Vote_Question; // Maniascript
  const bool Vote_CanVote; // Maniascript
  void Vote_Cast(bool Answer); // Maniascript
};

struct CGameCtnMediaBlockScenery : public CGameCtnMediaBlock {
  CGameCtnMediaBlockScenery();

};

struct CGameControlCameraVehicleInternal : public CGameControlCamera {
  CGameControlCameraVehicleInternal();

  CPlugVehicleCameraInternalModel* m_Model;
};

struct CWebServicesTask_SetServerInfo : public CWebServicesTaskSequence {
};

struct CGameHud3dMarkerConfig : public CMwNod {
  CGameHud3dMarkerConfig();

  enum class CGameHud3dMarkerConfig::EAnchorType {
    Invalid = 0,
    Position = 1,
    Player = 2,
    Entity = 3,
    Landmark = 4,
    GhostInstance = 5,
  };
  enum class CGameHud3dMarkerConfig::EMiniMapVisibility {
    Never = 0,
    Always = 1,
    WhenInFrame = 2,
  };
  enum class CGameHud3dMarkerConfig::EHudVisibility {
    Never = 0,
    Always = 1,
    WhenInFrustum = 2,
    WhenVisible = 3,
    WhenInMiddleOfScreen = 4,
  };
  const CGameHud3dMarkerConfig::EAnchorType Type; // Maniascript
  bool IsTurning; // Maniascript
  bool ShowArmor; // Maniascript
  wstring Label; // Maniascript
  vec3 Box; // Maniascript
  string Icon; // Maniascript
  vec3 Color; // Maniascript
  string ImageUrl; // Maniascript
  string ManialinkFrameId; // Maniascript
  float DistMin; // Maniascript
  float DistMax; // Maniascript
  CGameHud3dMarkerConfig::EMiniMapVisibility MiniMapVisibility; // Maniascript
  CGameHud3dMarkerConfig::EHudVisibility HudVisibility; // Maniascript
  bool HideSelf; // Maniascript
  bool HideOnSelf; // Maniascript
  float Gauge; // Maniascript
};

struct CGamePlaygroundResources : public CMwNod {
  const MwFastBuffer<CGameItemModel*> m_AllGameItemModels;
  const MwFastBuffer<CSystemPackDesc*> EntModelSkins;
  const MwFastBuffer<CGameObjectVisModel*> ObjectLib;
  const MwFastBuffer<CGameCharacterModel*> CharLib;
  CPlugAnimFile* const CinematicLib;
  CPlugVehicleVisStyles* const VehicleStyles;
  CPlugDestructibleFx* const DestructiblesFx;
};

struct CWebServicesTask_GetServerInfo : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_ServerInfo : public CWebServicesTaskResult {
};

struct CGameEditorAsset : public CGameCtnEditor {
};

struct CGameManialinkGraph : public CGameManialinkControl {
  vec2 CoordsMin; // Maniascript
  vec2 CoordsMax; // Maniascript
  CGameManialinkGraphCurve* AddCurve(); // Maniascript
  void RemoveCurve(CGameManialinkGraphCurve* Curve); // Maniascript
  MwFastBuffer<CGameManialinkGraphCurve*> Curves; // Maniascript
};

struct CGameManialinkGraphCurve : public CMwNod {
  MwFastBuffer<vec2> Points; // Maniascript
  vec3 Color; // Maniascript
  void SortPoints(); // Maniascript
  string Style; // Maniascript
  float Width; // Range: 0 - 50 // Maniascript
};

struct CGameCtnBlockInfoMobilLink : public CMwNod {
  CGameCtnBlockInfoMobilLink();

  MwId SocketId;
  CGameObjectModel* Model;
};

struct CGameManiaplanetPluginInterface : public CMwNod {
  const wstring Name; // Maniascript
  void CustomEvent(wstring Type, MwFastBuffer<wstring>& Data); // Maniascript
  const MwNodPool<CGameManiaplanetPluginInterfaceEvent*> PendingEvents; // Maniascript
};

struct CGameManiaplanetPluginInterfaceEvent : public CScriptBaseConstEvent {
  enum class CGameManiaplanetPluginInterfaceEvent::EType {
    PluginCustomEvent = 0,
  };
  const CGameManiaplanetPluginInterfaceEvent::EType Type; // Maniascript
  const wstring EventType; // Maniascript
  const MwFastBuffer<wstring> EventData; // Maniascript
};

// File extension: 'GameCtnMediaBlockInterface.gbx'
struct CGameCtnMediaBlockInterface : public CGameCtnMediaBlock {
  CGameCtnMediaBlockInterface();

  const bool ShowInterface;
  CControlFrame* ManialinkFrame;
  CPlugFileTextScript* ManialinkScript;
};

// File extension: 'GameCtnMediaBlockObject.gbx'
struct CGameCtnMediaBlockObject : public CGameCtnMediaBlock {
  CGameCtnMediaBlockObject();

  const bool ShowObjects;
};

struct CGameScriptCloudManager : public CMwNod {
};

struct CGameEditorBullet : public CMwNod {
  CPlugParticleEmitterModel* ParticleEmitterModel;
  CPlugBeamEmitterModel* BeamEmitterModel;
  CPlugBeamEmitterSubModel* BaseSubModel;
  float SubModelSphereRadMax;
  float SubModelSphereThickness;
  float SubModelViscosity;
  float SubModelFriction;
  float SubModelLength;
  float SubModelLengthVar;
  float SubModelRotation;
  float SubModelRotationVar;
  float SubModelLife;
  float SubModelLifeVar;
  float SubModelParticlesCount;
  float SubModelSize;
  float SubModelSizeSlider; // Range: 0 - 1
  float SubModelSizeVar;
  float SubModelSizeEndScale;
  float SubModelIntensVar;
  float SubModelIntens;
  float SubModelIntensSlider; // Range: 0 - 1
  float SubModelWeight;
  float SubModelPersistence;
  float SubModelEmitterCoef;
  float SubModelDiscMaxZ;
  float SubModelDiscVelRad;
  float SubModelDiscVelLong;
  float SubModelDiscRadMax;
  float SubModelDiscThickness;
  uint SubModelParticlesCountSplash;
  float SubModelLengthPitchMin;
  float SubModelLengthPitchMax;
  float SubModelIntensFadeIn;
  float SubModelIntensFadeOut;
  bool SubModelUseCustomColor;
  bool SubModelCameraSpace;
  string BulletName;
  float BulletScale;
  string BulletSubModelName;
  float BulletSpeed;
  float BulletMass;
  float BulletFluidFriction; // Range: 0 - 2
  float BulletHitboxRadius;
  float BulletLifeTime;
  float BulletLifeTimeAfterFirstImpact;
  wstring BulletPattern;
  bool BulletExplodeOnEndLife;
  uint BulletRebounds;
  bool BulletBounceOnTechWall;
  float BulletImpactBouncingN;
  float BulletImpactBouncingT;
  wstring BulletType;
  float BulletHomingDist;
  float BulletHomingPeriod;
  float BulletHomingAngularSpeed;
  uint BulletHomingLockDuration;
  float BulletGuidedAngularSpeed;
  uint BulletGuidedMinLifeTime;
  bool BulletShowPlayerExplosion;
  bool BulletShowDebris;
  bool BulletShowPlayerLaserExplosion;
  bool BulletShowLaserDebris;
  bool BulletLaserModifyFOV;
  bool BulletModifyFOV;
  uint PatternBulletCount;
  float PatternBulletRadius;
  float PatternBulletSpinSecond;
  float PatternBulletBlendDuration;
  uint NoPatternBulletCount;
  float NoPatternBulletDispersionAngle;
  float NoPatternBulletSpeedCoef;
  float PatternBulletMinApexTime;
  bool PatternBulletApexRegroup;
  bool PatternBulletRandomRotations;
  float BulletEDRResting;
  float BulletEDRFlying;
  float BulletBlowRadius;
  float BulletBlowStrength;
  float BulletBlowVerticalScale;
  float BulletDamageRadius;
  float BulletBlowAttenuation; // Range: 0 - 1
  uint BulletDamageDirectHit;
  uint BulletDamage;
  float BulletDamageAttenuation; // Range: 0 - 1
  bool BulletExplosionOnAllPlayers;
  bool BulletIsFlare;
  float BulletFlareAttractionRadius;
  float BulletFlareExplosionRadius;
  bool BulletIsWard;
  float BulletWardRadius;
  const wstring IsLaser;
  const wstring LaserType;
  float LaserUVScale;
  float LaserUSpeed;
  float LaserVSpeed;
  uint LaserCylinderSideCount;
  bool LaserBackface;
  uint LaserDuration;
  float LaserRadius;
  float LaserRadiusAtOneMeter0;
  float LaserRadiusAtOneMeter1;
  uint LaserDamage;
  uint HelixVerticesCountPerTurn;
  float HelixTurnLength;
  float HelixOutterRadius;
  float HelixInnerRadius;
  float HelixStartAngle;
  uint HelixMaxHelixCount;
  bool HelixClockWise;
  float DoubleHelixStartAngle;
  bool DoubleHelixClockWise;
  uint BWSNbRadialSubDiv;
  uint BWSNbVerticalSubDiv;
  uint BWSNbSpheresPerMeter;
  float BWSRadius;
  float BWSStartDistance;
  float BWSStepAngle;
  float BWSEndDistance;
  float BWSSpeed;
  uint BWCNbSubDiv;
  float BWCStepSize;
  float BWCOutterRadius;
  float BWCInnerRadius;
  float BWCHalfWidth;
  float BWCTimeStartFade;
  float BWCFadeSpeed;
  uint LightningcBeam;
  float LightningDistStep;
  float LightningMaxJitterDistPerStep;
  float LightningJitterRadius;
  float LightningLenMax;
  float LightningFadeTimeScale;
  float LightningFadeTimeOffset;
  float SoundParamVolume; // Range: 0 - 1
  void ParamModePoint();
  void ParamModeSphereFull();
  void ParamModeSphereEmpty();
  void ParamModeDisc();
  void ParamModeCircle();
  const MwFastBuffer<CPlugBulletModel*> BulletModels;
};

// File extension: 'GameCtnMediaBlockFog.gbx'
struct CGameCtnMediaBlockFog : public CGameCtnMediaBlock {
  CGameCtnMediaBlockFog();

  void SwitchOn();
};

struct CGameEditorActionScript : public CMwNod {
  wstring EditorActionScript;
  uint ScriptCooldown;
  float ScriptReloadTime;
  uint ScriptEnergyNbBullets;
  bool ScriptAdvancedMode;
  float ScriptEffectOffset;
  bool ScriptAnimOnPlayer;
  float ScriptAnimPosX;
  float ScriptAnimPosY;
  float ScriptAnimPosZ;
  float ScriptAnimDirX;
  float ScriptAnimDirY;
  float ScriptAnimDirZ;
  bool ScriptProjByPlayer;
  float ScriptProjPosX;
  float ScriptProjPosY;
  float ScriptProjPosZ;
  float ScriptProjDirX;
  float ScriptProjDirY;
  float ScriptProjDirZ;
  string ScriptProjName;
};

struct CGameEditorItem : public CGameEditorAsset {
  void Exit();
  void FileOpen();
  void FileSaveAs();
  void FileNew();
  void Help();
  void SwitchFullScreen();
  void EditCapsuleTrigger();
  void EditDiscTrigger();
  void EditCylinderObjectTrigger();
  void EditTimeTrigger();
  void EditMeshTrigger();
  void EditCapsuleObjectTrigger();
  void CreateAABBShape();
  void CreateSphereShape();
  void CreateCylinderShape();
  void ImportResources();
  void CreateResources();
  void TriggerValidate();
  void SwitchActionMaker();
  void SwitchParticleModel();
  void SwitchVehicleEditor();
  void SwitchMeshModeler();
  void SwitchItemEditor();
  void SwitchScriptEditor();
  void SwitchProcGenEditor();
  void CenterCamera();
  void PrevInput();
  void NextInput();
  void PrevLinkedAction();
  void NextLinkedAction();
  void AddEmptyMesh();
  CGameCtnApp* const Game;
  float TimeTriggerPeriod;
  const string TriggerInput;
  const string LinkedAction;
  float SpritePosX;
  float SpritePosY;
  float SpritePosZ;
  float SpriteSize; // Range: 0 - 25
  bool SpriteIsFree;
  float ShapeAABBWidth;
  float ShapeAABBHeight;
  float ShapeAABBDepth;
  float ShapeSphereRadius;
  float ShapeCylinderHeight;
  float ShapeCylinderRadius;
  bool PlacementParamSwitchPivotManually;
  float PlacementParamFlyStep;
  float PlacementParamFlyOffset;
  float PlacementParamGridHorizontalSize;
  float PlacementParamGridHorizontalOffset;
  float PlacementParamGridVerticalSize;
  float PlacementParamGridVerticalOffset;
  float PlacementParamCubeSize;
  float PlacementParamCubeCenterX;
  float PlacementParamCubeCenterY;
  float PlacementParamCubeCenterZ;
  bool PlacementParamGhostMode;
  bool PlacementParamYawOnly;
  bool PlacementParamNotOnObject;
  bool PlacementParamAutoRotation;
  float PlacementParamPivotSnapDistance;
  CGameItemModel* ItemModel;
};

struct CGameManialinkMiniMap : public CGameManialinkControl {
  vec3 WorldPosition; // Maniascript
  vec2 MapPosition; // Maniascript
  float MapYaw; // Maniascript
  float ZoomFactor; // Range: 0.5 - 10 // Maniascript
  bool Underground; // Maniascript
  bool DisableAutoUnderground; // Maniascript
  void Fog_SetAll(float Value); // Maniascript
  void Fog_ClearDisk(vec3 WorldCenter, float Radius, float FadeSize); // Maniascript
};

struct CWebServicesTaskResult_NadeoServicesSkin : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_NadeoServicesSkinList : public CWebServicesTaskResult {
};

struct CGameEditorManialink : public CGameEditorBase {
  CGameEditorManialink();

  void GenerateMetadataTree();
  void ButExit();
  void ButAddQuad();
  void ButAddFrame();
  void ButAddFrameModel();
  void ButAddFrameInstance();
  void ButAddFormat();
  void ButAddLabel();
  void ButAddAudio();
  void ButAddEntry();
  void ButAddTextEdit();
  void ButAddFileEntry();
  void ButAddVideo();
  void ButAddGauge();
  void ButAddGraph();
  void ButAddColorPicker();
  void ButAddTimeLine();
  void ButDeleteTag();
  void ButLockControl();
  void ButUnlockControl();
  void ButIndent();
  void ButHelp();
  void ButAdvancedOptions();
  void ButUndo();
  void ButRedo();
  void ButToggleStyles();
  void ButTogglePreview();
  void ButReloadResources();
  void ButQuadImage();
  void ButQuadImageFocus();
  void ButQuadStyle();
  void ButTextStyle();
  void ButTextTextFont();
  void ButGaugeStyle();
  void ButMediaData();
  void OnTextFontChoosen();
  void OnMediaChoosen();
  void OnImageChoosen();
  void ButQuadColor();
  void ButTextColorColor();
  void ButBgColorColor();
  void ButColorHelperOk();
  void ButFocusAreaColor1Color();
  void ButFocusAreaColor2Color();
  void ButGaugeColorColor();
  void ButSeparatorColorColor();
  void ButCursorColorColor();
  void ButRulerColorColor();
  void ButRulerLineColorColor();
  void ButKeyColorColor();
  void ButColorHelperCancel();
  wstring Params;
  wstring PosnX;
  wstring PosnY;
  wstring PosnZ;
  wstring Scale;
  wstring Rot;
  wstring ControlId;
  wstring Class;
  bool ButtonHidden;
  wstring SizenWidth;
  wstring SizenHeight;
  wstring Halign;
  wstring Valign;
  wstring FrameModelId;
  wstring ModelId;
  wstring Image;
  wstring ImageFocus;
  wstring Style;
  wstring Substyle;
  wstring Colorize;
  wstring ModulateColor;
  wstring KeepRatio;
  bool ButtonStyleSelected;
  bool ButtonAutoscale;
  bool ButtonAutoscaleFixedWidth;
  wstring StyleText;
  wstring TextColor;
  wstring TextSize;
  wstring TextFont;
  wstring FocusAreaColor1;
  wstring FocusAreaColor2;
  wstring Url;
  wstring Manialink;
  wstring Text;
  wstring TextId;
  wstring Action;
  wstring TextPrefix;
  wstring MaxLine;
  wstring Opacity;
  wstring LineSpacing;
  bool ButtonTextEmboss;
  bool ButtonTranslate;
  wstring Default;
  wstring Name;
  bool ButtonSelectText;
  wstring TextEditDefault;
  wstring TextEditMaxLine;
  bool ButtonShowLineNumbers;
  wstring TextFormat;
  wstring Data;
  wstring Volume;
  bool ButtonLooping;
  bool ButtonPlay;
  bool ButtonMusic;
  bool ButtonScriptEvents;
  wstring BgColor;
  bool ButtonAutoNewLine;
  wstring Folder;
  wstring Type;
  wstring Style3d;
  wstring Grading;
  wstring Ratio;
  wstring StyleGauge;
  wstring GaugeColor;
  wstring Clan;
  bool ButtonCentered;
  bool ButtonDrawBg;
  bool ButtonDrawBlockBg;
  wstring TrackCount;
  wstring DefaultTimeShown;
  wstring SeparatorColor;
  wstring CursorColor;
  wstring RulerColor;
  wstring RulerLineColor;
  wstring KeyColor;
  void ButHalignNextInput();
  void ButHalignPrevInput();
  void ButValignNextInput();
  void ButValignPrevInput();
  void ButKeepRatioNextInput();
  void ButKeepRatioPrevInput();
  void ButClanNextInput();
  void ButClanPrevInput();
  void ButTypeNextInput();
  void ButTypePrevInput();
  void ButXmlTypeNextInput();
  void ButXmlTypePrevInput();
  void ButTextFormatNextInput();
  void ButTextFormatPrevInput();
  void ButActionNext();
  void ButActionPrev();
  void ButUIBaseChoose();
  void ButUIBaseUp();
  void ButUIBaseDown();
  void ButUIBaseDelete();
  void OnUIBaseChoosen();
  void ToggleScriptEditorSize();
  bool ButtonGrid;
  bool ButtonSnap;
  wstring Step;
  wstring RotationStep;
  float ColorHelperOpacity; // Range: 0 - 1
  CControlFrame* const FrameLayoutEditor;
};

// Description: "API for titles in edition mode."
struct CGameManiaTitleEditionScriptAPI : public CMwNod {
  enum class CGameManiaTitleEditionScriptAPI::EDrive {
    TitleReadable = 0,
    Title = 1,
    User = 2,
  };
  CGameManiaTitle* const TitleMaker; // Maniascript
  CGamePackCreatorScript* const PackCreator; // Maniascript
  const string EditedTitleId; // Maniascript
  CGamePackCreator_TitleInfoScript* const EditedTitleInfo; // Maniascript
  void File_ImportFromUser(wstring FileName); // Maniascript
  void File_Move(wstring OrigName, wstring DestNameOrFolder, bool KeepOriginalCopy); // Maniascript
  bool File_Exists(wstring FileName, CGameManiaTitleEditionScriptAPI::EDrive InDrive); // Maniascript
  void File_Delete(wstring Name); // Maniascript
  void File_WriteText(wstring FileName, wstring Text); // Maniascript
  wstring File_ReadText(wstring FileName); // Maniascript
  void Dialog_ImportFiles(); // Maniascript
  const bool Dialog_IsFinished; // Maniascript
  const bool Dialog_Success; // Maniascript
  const bool Dialog_Aborted; // Maniascript
  void OpenTitleFolderInExplorer(); // Maniascript
  void ReloadTitleDesc(); // Maniascript
  void SaveTitleDesc(); // Maniascript
  void SetTitleCampaign(uint CampaignNum, string ScoreContext, wstring MapsFolderNameOrPlayListName, bool OfficialRecordEnabled); // Maniascript
  void SetTitleCampaign2(uint CampaignNum, string ScoreContext, wstring MapsFolderNameOrPlayListName, bool OfficialRecordEnabled, bool DisableUnlockSystem); // Maniascript
};

struct CGameEditorBase : public CMwNod {
};

struct CGameEditorPropertyList : public CMwNod {
};

struct CGameEditorFileToolBar : public CMwNod {
};

// Description: "API for Maniaplanet client scripts."
struct CGameManiaApp : public CMwNod {
  enum class CGameManiaApp::ELinkType {
    ExternalBrowser = 0,
    ManialinkBrowser = 1,
  };
  enum class CGameManiaApp::ESystemPlatform {
    None = 0,
    Steam = 1,
    UPlay = 2,
    PS4 = 3,
    XBoxOne = 4,
    PS5 = 5,
    XBoxSeries = 6,
    Stadia = 7,
    Luna = 8,
  };
  enum class CGameManiaApp::ESystemSkuIdentifier {
    Unknown = 0,
    EU = 1,
    US = 2,
    JP = 3,
    CN = 4,
  };
  const string ManiaAppUrl; // Maniascript
  const string ManiaAppBaseUrl; // Maniascript
  const int Now; // Maniascript
  const bool IsVisible; // Maniascript
  const uint CurrentDate; // Maniascript
  const string CurrentLocalDateText; // Maniascript
  const wstring CurrentTimezone; // Maniascript
  uint LayersDefaultManialinkVersion; // Maniascript
  CGamePlayerInfo* const LocalUser; // Maniascript
  CGameManiaTitle* const LoadedTitle; // Maniascript
  const CGameManiaApp::ESystemPlatform SystemPlatform; // Maniascript
  const CGameManiaApp::ESystemSkuIdentifier SystemSkuIdentifier; // Maniascript
  bool EnableMenuNavigationInputs; // Maniascript
  const float MouseX; // Maniascript
  const float MouseY; // Maniascript
  const MwFastBuffer<CGameUILayer*> UILayers; // Maniascript
  CGameUILayer* UILayerCreate(); // Maniascript
  void UILayerDestroy(CGameUILayer* Layer); // Maniascript
  void UILayerDestroyAll(); // Maniascript
  void LayerCustomEvent(CGameUILayer* Layer, wstring Type, MwFastBuffer<wstring>& Data); // Maniascript
  void OpenLink(string Url, CGameManiaApp::ELinkType LinkType); // Maniascript
  bool OpenFileInExplorer(wstring FileName); // Maniascript
  void Dialog_Message(wstring Message); // Maniascript
  CXmlScriptParsingManager* const Xml; // Maniascript
  CNetScriptHttpManager* const Http; // Maniascript
  CGameVideoScriptManager* const Video; // Maniascript
  CAudioScriptManager* const Audio; // Maniascript
  CInputScriptManager* const Input; // Maniascript
  CGameDataFileManagerScript* const DataFileMgr; // Maniascript
  CGameScoreAndLeaderBoardManagerScript* const ScoreMgr; // Maniascript
  CGameZoneManagerScript* const ZoneMgr; // Maniascript
  CGameUserPrivilegesManagerScript* const PrivilegeMgr; // Maniascript
  CGameMasterServerRichPresenceManagerScript* const PresenceMgr; // Maniascript
  CGameClientTrackingScript* const TrackingMgr; // Maniascript
  CGameUserManagerScript* const UserMgr; // Maniascript
  CGameWebServicesNotificationManagerScript* const WSNotificationMgr; // Maniascript
  CSystemPlatformScript* const System; // Maniascript
  CGameManiaPlanetScriptAPI* const ManiaPlanet; // Maniascript
  wstring Dbg_DumpDeclareForVariables(CMwNod* Nod, bool StatsOnly); // Maniascript
  bool TTS_Disabled; // Maniascript
  MwId TTS_Context_Enter(int Level); // Maniascript
  MwId TTS_Context_Leave(MwId ContextId); // Maniascript
  void TTS_Context_Change_Control(MwId ContextId, CGameManialinkControl* Control); // Maniascript
  void TTS_Context_Change_Text(MwId ContextId, wstring Text, bool Translate); // Maniascript
  void TTS_Context_Read_Lvl(int Level); // Maniascript
  void TTS_Context_Read_Id(MwId ContextId); // Maniascript
  wstring FilterProfanities(wstring TextToFilter); // Maniascript
};

struct CGameManiaAppScriptEvent : public CScriptBaseConstEvent {
  enum class CGameManiaAppScriptEvent::EType {
    LayerCustomEvent = 0,
    KeyPress = 1,
    ExternalCustomEvent = 2,
    MenuNavigation = 3,
  };
  enum class CGameManiaAppScriptEvent::EMenuNavAction {
    Up = 0,
    Right = 1,
    Left = 2,
    Down = 3,
    Select = 4,
    Cancel = 5,
    PageUp = 6,
    PageDown = 7,
    AppMenu = 8,
    Action1 = 9,
    Action2 = 10,
    Action3 = 11,
    Action4 = 12,
    ScrollUp = 13,
    ScrollDown = 14,
  };
  const CGameManiaAppScriptEvent::EType Type; // Maniascript
  CGameUILayer* const CustomEventLayer; // Maniascript
  const wstring CustomEventType; // Maniascript
  const MwFastBuffer<wstring> CustomEventData; // Maniascript
  const wstring ExternalEventType; // Maniascript
  const MwFastBuffer<wstring> ExternalEventData; // Maniascript
  const CGameManiaAppScriptEvent::EMenuNavAction MenuNavAction; // Maniascript
  const bool IsActionAutoRepeat; // Maniascript
  const uint KeyCode; // Maniascript
  const string KeyName; // Maniascript
};

// Description: "API for game interface client scripts"
struct CGameManiaAppPlaygroundCommon : public CGameManiaApp {
  const MwNodPool<CGameManiaAppPlaygroundScriptEvent*> PendingEvents; // Maniascript
  CGamePlaygroundClientScriptAPI* const Playground; // Maniascript
  CGameCtnChallenge* const Map; // Maniascript
  CGameGhostMgrScript* const GhostMgr; // Maniascript
  CGamePlaygroundUIConfig* UI; // Maniascript
  CGamePlaygroundUIConfig* ClientUI; // Maniascript
  const uint SplitScreenCount; // Maniascript
  void SplitScreenAssignLayer(CGameUILayer* UILayer, uint ScreenNum); // Maniascript
  CGameVoiceChatConfigScript* const VoiceChat; // Maniascript
};

struct CGameManialinkCamera : public CGameManialinkControl {
  bool Fullscreen; // Maniascript
};

struct CGameReplayObjectVisData : public CMwNod {
  CGameReplayObjectVisData();

};

// Description: "API for a Title main menu."
struct CGameManiaAppTitle : public CGameManiaApp {
  const MwNodPool<CGameManiaAppScriptEvent*> PendingEvents; // Maniascript
  void Menu_Quit(); // Maniascript
  void Menu_Home(); // Maniascript
  void Menu_Solo(); // Maniascript
  void Menu_Local(); // Maniascript
  void Menu_Internet(); // Maniascript
  void Menu_Editor(); // Maniascript
  void Menu_Profile(); // Maniascript
  void PlayMap(wstring Map); // Maniascript
  bool LoadingScreenRequireKeyPressed; // Maniascript
  bool DontScaleMainMenuForHMD; // Maniascript
  float FillExtraSpaceWithBluredContents_Opacity; // Range: 0 - 1 // Maniascript
  CGameManiaTitleControlScriptAPI* const TitleFlow; // Maniascript
  CGameManiaTitleControlScriptAPI* const TitleControl; // Maniascript
  CGameManiaTitleEditionScriptAPI* const TitleEdition; // Maniascript
  CGameScriptNotificationsConsumer* const Notifications; // Maniascript
  const wstring ExternalRequest_Type; // Maniascript
  const MwFastBuffer<wstring> ExternalRequest_Data; // Maniascript
  void ExternalRequest_Clear(); // Maniascript
  void LogToSessionTrace(wstring Text); // Maniascript
  CGameMatchSettingsManagerScript* const MatchSettingsManager; // Maniascript
  CGameMenuSceneScriptManager* const MenuSceneManager; // Maniascript
  void Authentication_GetToken(MwId UserId, string AppLogin); // Maniascript
  const bool Authentication_GetTokenResponseReceived; // Maniascript
  const uint Authentication_ErrorCode; // Maniascript
  const string Authentication_Token; // Maniascript
  bool Adverts_UsePersonnalData; // Maniascript
  bool Adverts_Enabled; // Maniascript
  CGameVoiceChatConfigScript* const VoiceChat; // Maniascript
};

// File extension: 'GameCtnMediaBlockDecal2d.gbx'
struct CGameCtnMediaBlockDecal2d : public CGameCtnMediaBlock {
  CGameCtnMediaBlockDecal2d();

};

struct CGameCtnMediaBlockEditorDecal2d : public CGameCtnMediaBlockEditor {
  CGameCtnMediaBlockEditorDecal2d();

  void ChooseImage();
  const wstring ImgDiffuseA;
  float Depth; // Range: 0 - 5
  float Opacity; // Range: 0 - 1
  bool FlipU;
  bool ForcePreview;
  CPlugDecalModel* DecalModel;
  CControlFrame* Frame;
};

struct CGameManiaAppBrowser : public CGameManiaApp {
  const MwNodPool<CGameManiaAppScriptEvent*> PendingEvents; // Maniascript
  void BrowserBack(); // Maniascript
  void BrowserQuit(); // Maniascript
  void BrowserHome(); // Maniascript
  void BrowserReload(); // Maniascript
  wstring BrowserFocusedFrameId; // Maniascript
};

// Description: "API for titles menus to control the main loop."
struct CGameManiaTitleControlScriptAPI : public CMwNod {
  enum class CGameManiaTitleControlScriptAPI::ESplitScreenLayout {
    Horizontal = 0,
    Vertical = 1,
    Four = 2,
  };
  enum class CGameManiaTitleControlScriptAPI::EResult {
    Success = 0,
    Error_Internal = 1,
    Error_DataMgr = 2,
    Error_Net_ServerNotFound = 3,
    Error_Net_ServerUnreachable = 4,
    Error_Net_Disconnected = 5,
    Error_Net_WrongPassword = 6,
    Error_Network_OnlineExpired = 7,
    Error_Network_ServerFull = 8,
    Error_Abort = 9,
  };
  enum class CGameManiaTitleControlScriptAPI::EEditorType {
    ActionMaker = 0,
    ChallengeEditor = 1,
    ItemEditor = 2,
    InterfaceDesigner = 3,
    MeshModeler = 4,
  };
  enum class CGameManiaTitleControlScriptAPI::EReplayEditType {
    None = 0,
    Edit = 1,
    View = 2,
    Shoot = 3,
  };
  void Dbg_GetServerInfoObj();
  void Dbg_Join_GetServerInfo_Result();
  void Dbg_PlayMatchSettingsObject();
  void Dbg_PlaySplitScreenObj();
  void Dbg_PlayMultiOnSameScreenObj();
  const bool IsReady; // Maniascript
  const CGameManiaTitleControlScriptAPI::EResult LatestResult; // Maniascript
  const wstring CustomResultType; // Maniascript
  const MwFastBuffer<wstring> CustomResultData; // Maniascript
  void PlayMap(wstring Map, wstring Mode, string SettingsXml); // Maniascript
  void PlayCampaign(CGameCtnCampaign* Campaign, CGameCtnChallengeInfo* MapInfo, wstring Mode, string SettingsXml); // Maniascript
  void PlayMapList(MwFastBuffer<wstring>& MapList, wstring Mode, string SettingsXml); // Maniascript
  void PlayPlaylist(wstring Playlist, wstring OverrideMode, string OverrideSettingsXml); // Maniascript
  void PlayMatchSettingsFile(wstring MatchSettingsFilePath, wstring OverrideMode, string OverrideSettingsXml); // Maniascript
  void PlayMatchSettingsObject(CGameMatchSettingsScript* MatchSettings, wstring OverrideMode, string OverrideSettingsXml); // Maniascript
  void PlayAgainstReplay(wstring Replay, wstring Mode, string SettingsXml); // Maniascript
  void PlaySplitScreen(CGameManiaTitleControlScriptAPI::ESplitScreenLayout LayoutType, MwFastBuffer<wstring>& MapList, wstring Mode, string SettingsXml); // Maniascript
  void PlayMultiOnSameScreen(MwFastBuffer<wstring>& MapList, wstring Mode, string SettingsXml); // Maniascript
  void PlaySplitScreenObj(CGameManiaTitleControlScriptAPI::ESplitScreenLayout LayoutType, CGameMatchSettingsScript* MatchSettings); // Maniascript
  void PlayMultiOnSameScreenObj(CGameMatchSettingsScript* MatchSettings); // Maniascript
  void ViewReplay(wstring Replay); // Maniascript
  void OpenEditor(wstring EditorName, string MainPluginSettings); // Maniascript
  void OpenEditor2(CGameManiaTitleControlScriptAPI::EEditorType EditorType); // Maniascript
  void EditSkins(); // Maniascript
  void EditSkinsFromFileName(wstring SkinFilePath); // Maniascript
  void EditReplay(MwFastBuffer<wstring>& ReplayList); // Maniascript
  void EditReplay2(MwFastBuffer<wstring>& ReplayList, CGameManiaTitleControlScriptAPI::EReplayEditType EditType); // Maniascript
  void EditGhosts(wstring Map); // Maniascript
  void EditAsset(wstring EditorName, string MainPluginSettingsXml, wstring RelativeFileName); // Maniascript
  void EditMap(wstring Map, wstring EditorPluginScript, string EditorPluginArgument); // Maniascript
  void EditMap2(wstring Map, string Decoration, wstring ModNameOrUrl, wstring PlayerModel, wstring EditorPluginScript, string EditorPluginArgument); // Maniascript
  void EditMap3(wstring Map, string Decoration, wstring ModNameOrUrl, wstring PlayerModel, wstring EditorPluginScript, string EditorPluginArgument, bool UpgradeToAdvancedEditor); // Maniascript
  void EditMap4(wstring Map, string Decoration, wstring ModNameOrUrl, wstring PlayerModel, MwFastBuffer<wstring>& EditorPluginsScripts, MwFastBuffer<wstring>& EditorPluginsArguments, bool UpgradeToAdvancedEditor); // Maniascript
  void EditMap5(wstring Map, string Decoration, wstring ModNameOrUrl, wstring PlayerModel, MwFastBuffer<wstring>& EditorPluginsScripts, MwFastBuffer<wstring>& EditorPluginsArguments, bool UpgradeToAdvancedEditor, bool OnlyUseForcedPlugins); // Maniascript
  void EditNewMap1(string Environment, string Decoration, wstring ModNameOrUrl, wstring PlayerModel, wstring MapType, wstring EditorPluginScript, string EditorPluginArgument); // Maniascript
  void EditNewMap2(string Environment, string Decoration, wstring ModNameOrUrl, wstring PlayerModel, wstring MapType, bool UseSimpleEditor, wstring EditorPluginScript, string EditorPluginArgument); // Maniascript
  void EditNewMap3(string Environment, string Decoration, wstring ModNameOrUrl, wstring PlayerModel, wstring MapType, bool UseSimpleEditor, MwFastBuffer<wstring>& EditorPluginsScripts, MwFastBuffer<wstring>& EditorPluginsArguments); // Maniascript
  void EditNewMap4(string Environment, string Decoration, wstring ModNameOrUrl, wstring PlayerModel, wstring MapType, bool UseSimpleEditor, MwFastBuffer<wstring>& EditorPluginsScripts, MwFastBuffer<wstring>& EditorPluginsArguments, bool OnlyUseForcedPlugins); // Maniascript
  void EditNewMapFromBaseMap(wstring BaseMapName, wstring ModNameOrUrl, wstring PlayerModel, wstring MapType, wstring EditorPluginScript, string EditorPluginArgument); // Maniascript
  void EditNewMapFromBaseMap2(wstring BaseMapName, string Decoration, wstring ModNameOrUrl, wstring PlayerModel, wstring MapType, wstring EditorPluginScript, string EditorPluginArgument); // Maniascript
  void EditNewMapFromBaseMap3(wstring BaseMapName, string Decoration, wstring ModNameOrUrl, wstring PlayerModel, wstring MapType, MwFastBuffer<wstring>& EditorPluginsScripts, MwFastBuffer<wstring>& EditorPluginsArguments, bool OnlyUseForcedPlugins); // Maniascript
  const bool CanPublishFiles; // Maniascript
  void PublishFile(wstring FileName); // Maniascript
  void ProcessManiaCodeXml(string ManiaCodeXml); // Maniascript
  const MwFastBuffer<CGameCtnNetServerInfo*> LocalServers; // Maniascript
  const MwFastBuffer<CGameCtnNetServerInfo*> LocalServers_CurrentTitle; // Maniascript
  void DiscoverLocalServers(); // Maniascript
  void CreateServer1(wstring ServerName, wstring ServerComment, uint MaxPlayerCount, string Password, string PasswordSpectators, MwFastBuffer<wstring>& MapList, wstring Mode, string ScriptsSettingsXml); // Maniascript
  void CreateServer2(wstring ServerName, wstring ServerComment, uint MaxPlayerCount, string Password, string PasswordSpectators, wstring MatchSettingsFileName); // Maniascript
  void CreateServer2Obj(wstring ServerName, wstring ServerComment, uint MaxPlayerCount, string Password, string PasswordSpectators, CGameMatchSettingsScript* MatchSettings); // Maniascript
  void CreateServer3(wstring ServerName, wstring ServerComment, uint MaxPlayerCount, string Password, string PasswordSpectators, bool LocalOnly, MwFastBuffer<wstring>& MapList, wstring Mode, string ScriptsSettingsXml); // Maniascript
  void CreateServer4(wstring ServerName, wstring ServerComment, uint MaxPlayerCount, string Password, string PasswordSpectators, bool LocalOnly, wstring MatchSettingsFileName); // Maniascript
  void CreateServer4Obj(wstring ServerName, wstring ServerComment, uint MaxPlayerCount, string Password, string PasswordSpectators, CGameMatchSettingsScript* MatchSettings, bool LocalOnly); // Maniascript
  void GetServerInfo(string ServerLogin); // Maniascript
  void GetServerInfoObj(CGameCtnNetServerInfo* LocalServer); // Maniascript
  void GetServerInfo_Abort(); // Maniascript
  CGameCtnNetServerInfo* const GetServerInfo_Result; // Maniascript
  void Join_GetServerInfo_Result(bool AsSpectator, string Password); // Maniascript
  void JoinServer(string ServerLogin, bool AsSpectator, string Password); // Maniascript
  void JoinServer_Abort(); // Maniascript
  bool JoinServer_DisableSystemDialogs; // Maniascript
  CGameCtnNetServerInfo* const JoinServer_ServerInfo; // Maniascript
  void JoinServer_GetInfo(string ServerLogin); // Maniascript
  void JoinServer_GetInfoPwd(string ServerLogin, string ServerPassword); // Maniascript
  void JoinServer_GetInfoObj(CGameCtnNetServerInfo* Server); // Maniascript
  void JoinServer_GetInfoObjPwd(CGameCtnNetServerInfo* Server, string ServerPassword); // Maniascript
  void JoinServer_Join(bool AsSpectator); // Maniascript
  void Quit(); // Maniascript
};

// Description: "This is the title menus Manialink interface."
struct CGameManiaAppTitleLayerScriptHandler : public CGameManialinkScriptHandler {
  CGameManiaAppTitle* const ParentApp; // Maniascript
  CGameMatchSettingsManagerScript* const MatchSettingsManager; // Maniascript
  CGameManiaTitleControlScriptAPI* const TitleControl; // Maniascript
  CGameManiaTitleControlScriptAPI* const TitleFlow; // Maniascript
};

struct CGameEditorPluginMapLayerScriptHandler : public CGameManialinkScriptHandler {
  CGameEditorPluginMap* const Editor; // Maniascript
};

// Description: "A Notification."
struct CGameScriptNotificationsConsumerNotification : public CMwNod {
  enum class CGameScriptNotificationsConsumerNotification::ENotificationPriority {
    Memo = 0,
    Notice = 1,
    Alarm = 2,
  };
  const wstring Title; // Maniascript
  const wstring Description; // Maniascript
  const wstring ImageUrl; // Maniascript
  const CGameScriptNotificationsConsumerNotification::ENotificationPriority Priority; // Maniascript
  const bool HasBeenRead; // Maniascript
  const bool HasBeenActivated; // Maniascript
  void SetRead(); // Maniascript
  void SetActivated(); // Maniascript
};

// Description: "Notifications Producing manager."
struct CGameScriptNotificationsProducer : public CMwNod {
  enum class CGameScriptNotificationsProducer::ENotificationLifeSpan {
    Game = 0,
    Session = 1,
    NeverEnding = 2,
  };
  enum class CGameScriptNotificationsProducer::ENotificationPriority {
    Memo = 0,
    Notice = 1,
    Alarm = 2,
  };
  const MwFastBuffer<CGameScriptNotificationsProducerEvent*> Events; // Maniascript
  void SendNotification(MwFastBuffer<wstring>& CallbackParams, wstring Title, wstring Description, wstring ImageUrl, CGameScriptNotificationsProducer::ENotificationLifeSpan LifeSpan, CGameScriptNotificationsProducer::ENotificationPriority Priority); // Maniascript
};

// Description: "Manager of buddies instant messaging."
struct CGameScriptNotificationsConsumer : public CMwNod {
  enum class CGameScriptNotificationsConsumer::EFilterPriority {
    All = 0,
    MoreThanMemo = 1,
    MoreThanNotice = 2,
  };
  const MwNodPool<CGameScriptNotificationsConsumerEvent*> Events; // Maniascript
  const MwFastBuffer<CGameScriptNotificationsConsumerNotification*> Notifications; // Maniascript
  const MwFastBuffer<CGameScriptNotificationsConsumerNotification*> FilteredNotifications; // Maniascript
  CGameScriptNotificationsConsumer::EFilterPriority Filter_Priority; // Maniascript
};

// Description: "An event."
struct CGameScriptNotificationsProducerEvent : public CMwNod {
  enum class CGameScriptNotificationsProducerEvent::EType {
    NotificationHasBeenActivated = 0,
  };
  const CGameScriptNotificationsProducerEvent::EType Type; // Maniascript
};

// Description: "An event."
struct CGameScriptNotificationsConsumerEvent : public CMwNod {
  enum class CGameScriptNotificationsConsumerEvent::EType {
    NewNotification = 0,
    NotificationChanged = 1,
  };
  const CGameScriptNotificationsConsumerEvent::EType Type; // Maniascript
  CGameScriptNotificationsConsumerNotification* const Notification; // Maniascript
};

// Description: "Manialink entry."
struct CGameManialinkTextEdit : public CGameManialinkControl {
  enum class CGameManialinkTextEdit::EControlScriptEditorTextFormat {
    Basic = 0,
    Script = 1,
    Password = 2,
    Newpassword = 3,
  };
  wstring Value; // Maniascript
  void StartEdition(); // Maniascript
  int MaxLine; // Maniascript
  bool AutoNewLine; // Maniascript
  bool ShowLineNumbers; // Maniascript
  float LineSpacing; // Maniascript
  const uint ValueLineCount; // Maniascript
  float Opacity; // Maniascript
  vec3 TextColor; // Maniascript
  float TextSizeReal; // Maniascript
  uint TextSize; // Maniascript
  CGameManialinkTextEdit::EControlScriptEditorTextFormat TextFormat; // Maniascript
};

struct CGameDataFileTask_AccountSkin_NadeoServices_GetListForUser : public CWebServicesTaskSequence {
};

struct CGameManialinkStylesheet : public CMwNod {
  CGameManialinkStylesheet();

};

struct CGameWebServicesNotificationTask_BuildVisualNotification : public CWebServicesTaskSequence {
};

struct CGameManiaAppMinimal : public CGameManiaApp {
  const MwNodPool<CGameManiaAppScriptEvent*> PendingEvents; // Maniascript
};

struct CGameVideoScriptManager : public CMwNod {
  CGameVideoScriptVideo* CreateVideo(string Url, bool IsLooping, bool AutoProcessing); // Maniascript
  CGameVideoScriptVideo* CreateVideo2(string Url, bool IsLooping, bool AutoProcessing, bool UseMipMap); // Maniascript
  void DestroyVideo(CGameVideoScriptVideo* Video); // Maniascript
  const MwFastBuffer<CGameVideoScriptVideo*> Videos; // Maniascript
};

// Description: "A ghost model."
struct CGameGhostScript : public CMwNod {
  CGameGhostScript();

  const MwId Id; // Maniascript
  CTmRaceResultNod* const Result; // Maniascript
  wstring Nickname; // Maniascript
  string Trigram; // Maniascript
  wstring CountryPath; // Maniascript
  string OverrideZoneIcon_ImageUrl; // Maniascript
  void SetProfileInfoFromUser(CGamePlayerInfo* User); // Maniascript
};

// Description: "Description of a race run."
struct CTmRaceResultNod : public CMwNod {
  enum class CTmRaceResultNod::ETmRaceResultCriteria {
    Time = 0,
    Stunts = 1,
    NbRespawns = 2,
    CheckpointsProgress = 3,
    None = 4,
  };
  uint Time; // Maniascript
  int Score; // Maniascript
  int StuntsScore; // Maniascript
  uint NbRespawns; // Maniascript
  MwId SpawnLandmarkId; // Maniascript
  MwStridedArray<uint> Checkpoints; // Maniascript
  MwFastBuffer<MwId> CheckpointLandmarkIds; // Maniascript
  int Compare(CTmRaceResultNod* Other, CTmRaceResultNod::ETmRaceResultCriteria Criteria); // Maniascript
};

// Description: ""
struct CGameEditorModule : public CGameCtnEditor {
  enum class CGameEditorModule::EModuleType {
    Undefined = 0,
    Hud = 1,
    Inventory = 2,
    Store = 3,
    ScoresTable = 4,
    Equipment = 5,
    MenuBase = 6,
    MenuPage = 7,
    Chrono = 8,
    SpeedMeter = 9,
    PlayerState = 10,
    TeamState = 11,
  };
  void Exit();
  void FileNew();
  void FileOpen();
  void FileSave();
  void FileSaveAs();
  void SwitchToPreview();
  void SwitchFromPreview();
  void AdvancedParameters();
  void CustomMode();
  void ActionIsPowerup();
  void SwitchInterfaceDesigner();
  CGameCtnApp* const Game;
  void EditHudModule_OnBack();
  void EditSubpart_OnBack();
  void EditScoresTableColumn_OnNextTextHalign();
  void EditScoresTableColumn_OnPrevTextHalign();
  void EditMenuLadderRankings_OnNextMode();
  void EditMenuLadderRankings_OnPrevMode();
  void EditInventoryCategory_LoadIconUrl();
  float EditHudModulePosX;
  float EditHudModulePosY;
  float EditHudModuleZIndex;
  float EditHudModuleScale;
  string EditActionName;
  string EditActionDesc;
  uint EditActionMaxLevel;
  uint EditActionPrice;
  uint EditActionPrice1;
  uint EditActionPrice2;
  uint EditActionPrice3;
  uint EditActionPrice4;
  uint EditActionPrice5;
  string EditItemName;
  string EditItemDesc;
  uint EditItemPrice;
  string EditCategoryName;
  string EditCategoryDesc;
  wstring EditCategoryIconUrl;
  string EditColumnName;
  string EditColumnId;
  string EditColumnDefaultValue;
  float EditColumnWidth;
  string EditColumnTextStyle;
  float EditColumnTextSize;
  string EditPageName;
  MwFastBuffer<string> EditedPageButtonsValues;
  void NewModule(CGameEditorModule::EModuleType ModuleType); // Maniascript
  void OpenModule(wstring Path); // Maniascript
  void Save(); // Maniascript
  void SaveAs(wstring Path); // Maniascript
  void SaveCopyAs(wstring Path); // Maniascript
  void ForceExit(); // Maniascript
  void SetPreviewBackground(wstring Url); // Maniascript
  void ReloadPreview(); // Maniascript
  const CGameEditorModule::EModuleType EditedModuleType; // Maniascript
  CGameModuleMenuModel* const EditedMenu; // Maniascript
  CGameModuleMenuPageModel* const EditedMenuPage; // Maniascript
  CGameModulePlaygroundPlayerStateModel* const EditedPlaygroundPlayerState; // Maniascript
  CGameModulePlaygroundHudModel* const EditedPlaygroundHud; // Maniascript
  void EditedPlaygroundHud_SetPreviewContext(MwId ContextId); // Maniascript
  bool GetControlRect_HudSubModule(CGameModulePlaygroundHudModelModule* SubModule, vec2 Center, vec2 Size); // Maniascript
  bool GetControlRect_PlayerStateComponent(CGameModulePlaygroundPlayerStateComponentModel* Component, vec2 Center, vec2 Size); // Maniascript
  void FileBrowser_Open(); // Maniascript
  void FileBrowser_Save(); // Maniascript
  const bool FileBrowser_IsRunning; // Maniascript
  const wstring FileBrowser_FilePath; // Maniascript
  const MwFastBuffer<CGameEditorPluginModuleScriptEvent*> PendingEvents; // Maniascript
  bool IsPreviewMode; // Maniascript
  bool IsAdvancedParameters; // Maniascript
  bool IsCustomMode; // Maniascript
};

// Description: "Create scenes for use in menus with &lt;Camera&gt; manialink element"
struct CGameMenuSceneScriptManager : public CMwNod {
  MwId SceneCreate(wstring LayoutFileName); // Maniascript
  void SceneDestroy(MwId SceneId); // Maniascript
  void CameraSetLocation0(MwId SceneId, vec3 Position, float AngleDeg); // Maniascript
  void CameraSetLocation1(MwId SceneId, vec3 Position, float AngleDeg, float FovY_Deg); // Maniascript
  void CameraSetFromItem(MwId SceneId, MwId ItemId); // Maniascript
  void LightDir0Set0(MwId SceneId, vec3 sRGB, float Intensity); // Maniascript
  void LightDir0Set1(MwId SceneId, vec3 sRGB, float Intensity, float AltitudeDeg, float AzimutDeg); // Maniascript
  MwId ItemCreate0(MwId SceneId, wstring ModelName, wstring SkinNameOrUrl); // Maniascript
  MwId ItemCreate1(MwId SceneId, wstring ModelName, wstring SkinName, string SkinUrl); // Maniascript
  MwId ItemCreate(MwId SceneId, wstring ModelName, wstring SkinName, string SkinUrl, string SkinOptions); // Maniascript
  void ItemDestroy(MwId SceneId, MwId ItemId); // Maniascript
  void ItemSetLocation(MwId SceneId, MwId ItemId, vec3 Position, float AngleDeg, bool IsTurntable); // Maniascript
  void ItemAttachTo(MwId SceneId, MwId ItemId, MwId ParentItemId); // Maniascript
  void ItemSetVehicleState(MwId SceneId, MwId ItemId, float Steer, bool Brakes, bool FrontLight, uint TurboLvl, uint BoostLvl, bool BurnoutSmoke); // Maniascript
  void ItemSetPlayerState(MwId SceneId, MwId ItemId, vec3 LightrailColor, vec3 DossardColor, string DossardNumber, string DossardTrigram); // Maniascript
  void ItemSetPlayerStateFromUser(MwId SceneId, MwId ItemId, CGamePlayerInfo* User); // Maniascript
  void ItemTriggerPlayerGesture(MwId SceneId, MwId ItemId); // Maniascript
  void ItemSetPivot(MwId SceneId, MwId ItemId, vec3 Pivot); // Maniascript
  void ItemSetVisible(MwId SceneId, MwId ItemId, bool IsVisible); // Maniascript
  void PlaneReflectEnable0(MwId SceneId, float OutOpacity, CGameManialinkQuad* Quad0, CGameManialinkQuad* Quad1, CGameManialinkQuad* Quad2, CGameManialinkQuad* Quad3); // Maniascript
  void PlaneReflectEnable1(MwId SceneId, float OutOpacity, CGameManialinkQuad* Quad0, CGameManialinkQuad* Quad1, CGameManialinkQuad* Quad2, CGameManialinkQuad* Quad3, float HorizonAngleDeg_InCamera); // Maniascript
  void PlaneReflectRefresh(); // Maniascript
  void SetBackgroundQuads(MwId SceneId, CGameManialinkQuad* Quad0, CGameManialinkQuad* Quad1, CGameManialinkQuad* Quad2, CGameManialinkQuad* Quad3); // Maniascript
  void CubeMapSetImage2ds(MwId SceneId, wstring RelName0, wstring RelName1, wstring RelName2, wstring RelName3); // Maniascript
  void CubeMapSetFromEquirect(MwId SceneId, CGameManialinkQuad* Quad); // Maniascript
  void ColorGradingSetImage2ds(MwId SceneId, wstring RelName0, wstring RelName1, wstring RelName2, wstring RelName3); // Maniascript
  void MoodLightDir0Set(MwId SceneId, uint iMood, vec3 sRGB, float Intensity, float AltitudeDeg, float AzimutDeg); // Maniascript
  void PodiumTweakShadingSet(MwId SceneId, float BaseColor, float CubeReflect, float PlaneReflect); // Maniascript
  void PostFxSet(MwId SceneId, float BloomIntensity); // Maniascript
};

struct CGamePlaygroundModuleClientInventory : public CGamePlaygroundModuleClient {
  CGameUILayer* const InventoryLayer; // Maniascript
};

struct CGamePlaygroundModuleClientScoresTable : public CGamePlaygroundModuleClient {
};

struct CGamePlaygroundModuleManagerClient : public CMwNod {
};

// Description: "GameMode ManaiApp event"
struct CGameManiaAppPlaygroundScriptEvent : public CGameManiaAppScriptEvent {
  enum class CGameManiaAppPlaygroundScriptEvent::EType {
    LayerCustomEvent = 0,
    KeyPress = 1,
    MenuNavigation = 3,
    PlaygroundScriptEvent = 4,
    GhostAdded = 5,
    RecordUpdated = 6,
    RecordsUpdated = 7,
    OnPlayerTriggerSpecial = 8,
    OnPlayerTriggerWaypoint = 9,
  };
  enum class CGameManiaAppPlaygroundScriptEvent::EGameplaySpecialType {
    None = 0,
    Turbo = 1,
    Turbo2 = 2,
    TurboRoulette = 3,
    FreeWheeling = 4,
    NoGrip = 5,
    NoSteering = 6,
    ForceAcceleration = 7,
    Reset = 8,
    SlowMotion = 9,
    Bumper = 10,
    Bumper2 = 11,
    ReactorBoost_Legacy = 12,
    Fragile = 13,
    ReactorBoost2_Legacy = 14,
    Bouncy = 15,
    NoBrakes = 16,
    Cruise = 17,
    ReactorBoost_Oriented = 18,
    ReactorBoost2_Oriented = 19,
    VehicleTransform_Reset = 20,
    VehicleTransform_CarSnow = 21,
    VehicleTransform_CarRally = 22,
    VehicleTransform_CarDesert = 23,
    XXX_Null = 24, // XXX Null
  };
  enum class CGameManiaAppPlaygroundScriptEvent::EGameplayTurboRoulette {
    TurboRoulette_None = 0,
    TurboRoulette_1 = 1,
    TurboRoulette_2 = 2,
    TurboRoulette_3 = 3,
  };
  const CGameManiaAppPlaygroundScriptEvent::EType PlaygroundType; // Maniascript
  const wstring PlaygroundScriptEventType; // Maniascript
  const MwFastBuffer<wstring> PlaygroundScriptEventData; // Maniascript
  CGameGhostScript* const Ghost; // Maniascript
  const CGameManiaAppPlaygroundScriptEvent::EGameplaySpecialType GameplaySpecialType; // Maniascript
  const CGameManiaAppPlaygroundScriptEvent::EGameplayTurboRoulette GameplayTurboRoulette; // Maniascript
  const bool IsBoostUpElseDown; // Maniascript
  const uint WaypointLandmarkIndex; // Maniascript
  const uint RaceWaypointTime; // Maniascript
  const int DiffWithBestRace; // Maniascript
  const bool DiffWithBestRace_IsValid; // Maniascript
  const uint LapWaypointTime; // Maniascript
  const int DiffWithBestLap; // Maniascript
  const bool DiffWithBestLap_IsValid; // Maniascript
  const bool IsFinish; // Maniascript
  const bool IsNewLap; // Maniascript
  const uint RaceWaypointCount; // Maniascript
  const uint LapWaypointCount; // Maniascript
  const uint RaceWaypointIndex; // Maniascript
  const uint LapWaypointIndex; // Maniascript
};

struct CGameUserProfile : public CMwNod {
  CGameUserProfile();

  bool Editor_ShowHelp;
};

struct CGameUIAnimManager : public CMwNod {
};

struct CGameControlCameraTrackManiaRace3 : public CGameControlCameraTarget {
  CGameControlCameraTrackManiaRace3();

  CPlugVehicleCameraRace3Model* Model;
};

struct CGameControlCameraTrackManiaRace2 : public CGameControlCameraTarget {
  CGameControlCameraTrackManiaRace2();

  CPlugVehicleCameraRace2Model* Model;
};

struct CGamePlaygroundModuleClientStore : public CGamePlaygroundModuleClient {
  CGameUILayer* const StoreLayer; // Maniascript
};

struct CGamePlaygroundModuleConfig : public CMwNod {
};

// Description: "Masterserver user info."
struct CGameMasterServerUserInfo : public CNetMasterServerUserInfo {
  const bool NeedToChangeZone; // Maniascript
  const uint ZoneLevelCount; // Maniascript
  wstring GetZone(uint ZoneLevel); // Maniascript
  const uint MultiInternetPlayTimeLeft; // Maniascript
};

// File extension: 'StickerSlots.Gbx'
struct CGameBadgeStickerSlots : public CMwNod {
  CGameBadgeStickerSlots();

  const uint MeshCount;
};

struct CGamePlaygroundModuleClient : public CGameManiaAppPlaygroundCommon {
};

// Description: "This is the client ManiaApp for game modes."
struct CGameManiaAppPlayground : public CGameManiaAppPlaygroundCommon {
  void SendCustomEvent(wstring Type, MwFastBuffer<wstring>& Data); // Maniascript
  bool HoldLoadingScreen; // Maniascript
};

// Description: "This is a video playback handle."
struct CGameVideoScriptVideo : public CMwNod {
  enum class CGameVideoScriptVideo::ETextureFilter {
    Default = 0,
    Point = 1,
  };
  CPlugBitmap* const Image; // Maniascript
  bool IsLooping; // Maniascript
  const bool DownloadInProgress; // Maniascript
  const float PlayLength; // Maniascript
  void BeginProcessing(); // Maniascript
  void EndProcessing(); // Maniascript
  const bool IsProcessing; // Maniascript
  bool AutoProcessing; // Maniascript
  void Play(); // Maniascript
  void Pause(); // Maniascript
  void Stop(); // Maniascript
  const bool IsPlaying; // Maniascript
  float PlayCursor; // Maniascript
  CGameVideoScriptVideo::ETextureFilter TextureFilter; // Maniascript
};

// Description: "SystemUser from the platform."
struct CGameUserScript : public CMwNod {
  const wstring DisplayName; // Maniascript
  const vec3 Color; // Maniascript
  const bool PersistentIsReady; // Maniascript
  void PersistentSave(); // Maniascript
  void SavePersistent(); // Maniascript
  CGameUserProfileWrapper* const Config; // Maniascript
};

// Description: "API for local users/profiles."
struct CGameUserManagerScript : public CMwNod {
  enum class CGameUserManagerScript::ECrossPlayState {
    Unknown = 0,
    Disabled = 1,
    Enabled = 2,
  };
  enum class CGameUserManagerScript::EPrestigeMode {
    Unknown = 0,
    Ranked = 1,
    Royal = 2,
    Season = 3,
  };
  enum class CGameUserManagerScript::EUbisoftClubFlow {
    OverView = 0,
    Auth = 1,
    Reward = 2,
    Rewards = 3,
    Challenge = 4,
    Challenges = 5,
    CoreChallenges = 6,
    TimeLimitedChallenges = 7,
  };
  enum class CGameUserManagerScript::EUplayOverlaySection {
    Achievements = 0,
    Actions = 1,
    Challenges = 2,
    Chat = 3,
    Current = 4,
    Friends = 5,
    GameOptions = 6,
    Home = 7,
    Party = 8,
    PendingGameInvites = 9,
    ProductActivation = 10,
    Rewards = 11,
    Shop = 12,
  };
  enum class CGameUserManagerScript::EVoiceChatMuteSetting {
    Muted = 0,
    NotMuted = 1,
  };
  CGameUserProfileWrapper* const MainUserProfile; // Maniascript
  const MwFastBuffer<CWebServicesTaskResult*> TaskResults; // Maniascript
  void TaskResult_Release(MwId TaskId); // Maniascript
  CWebServicesTaskResult* TaskError_ShowSystemDialog(MwId UserId, string ErrorCode); // Maniascript
  CWebServicesTaskResult_StringIntList* GetGroups(MwId UserId); // Maniascript
  const MwFastBuffer<CGameUserScript*> Users; // Maniascript
  CInputScriptPad* const MainUserPad; // Maniascript
  bool MainUserAcceptPressStart; // Maniascript
  const bool MainUserLogged; // Maniascript
  CGameUserScript* const MainUser; // Maniascript
  const bool Intro; // Maniascript
  bool CanSkipIntro; // Maniascript
  void SetSkipIntro(); // Maniascript
  void DevSetSkipIntro(); // Maniascript
  void ShowProfile(MwId UserId, string ProfileLogin); // Maniascript
  CWebServicesTaskResult_GetDisplayNameScriptResult* GetDisplayName(MwId UserId, MwFastBuffer<wstring>& WebServicesUserIdList); // Maniascript
  CWebServicesTaskResult_GetDisplayNameScriptResult* RetrieveDisplayName(MwId UserId, MwFastBuffer<wstring>& WebServicesUserIdList); // Maniascript
  wstring FindDisplayName(string WebServicesUserId); // Maniascript
  wstring FindDisplayName2(string WebServicesUserId, bool IsFirstPartyDisplayName); // Maniascript
  string WebServicesUserIdFromLogin(string Login); // Maniascript
  CGameMasterServerUserInfo* const MainUserWebServicesInfo; // Maniascript
  string ResolveURLShorcut(string ShortCutId); // Maniascript
  CGameMasterServerUserInfo* FindWebServicesUserInfo(MwId UserId); // Maniascript
  CWebServicesTaskResult* CheckNetworkAvailability(MwId UserId); // Maniascript
  const bool IsNetworkAvailable; // Maniascript
  CWebServicesTaskResult_Connect* ConnectUser(MwId UserId); // Maniascript
  CWebServicesTaskResult_Connect* ConnectUser2(MwId UserId, bool ForceUbisoftConnectOverlay); // Maniascript
  CWebServicesTaskResult* ConnectUser3(MwId UserId, bool ForceUbisoftConnectOverlay, bool OfflineMode); // Maniascript
  CWebServicesTaskResult* Commerce_ShowPrimaryStore(MwId UserId); // Maniascript
  CWebServicesTaskResult_Bool* CrossPlay_IsEnabled(MwId UserId); // Maniascript
  bool CrossPlay_Setting_GetEnabled(MwId UserId); // Maniascript
  void CrossPlay_Setting_SetEnabled(MwId UserId, bool Value); // Maniascript
  CGameUserManagerScript::ECrossPlayState CrossPlay_V2_IsEnabled(MwId UserId); // Maniascript
  void CrossPlay_V2_SetEnabled(MwId UserId, bool Enabled); // Maniascript
  bool Blocklist_CanViewUGC(MwId UserId, string WebServicesUserId); // Maniascript
  uint Friend_GetLastChangeIndex(MwId UserId); // Maniascript
  CWebServicesTaskResult_FriendListScript* Friend_GetList(MwId UserId); // Maniascript
  CWebServicesTaskResult_Session_Get* LiveSession_GetInfo(MwId UserId, wstring SessionId); // Maniascript
  void LiveSession_ShowInviteUI(MwId UserId); // Maniascript
  CWebServicesTaskResult* LiveSession_ShowFirstPartyInviteUI(MwId UserId); // Maniascript
  CWebServicesTaskResult_UserNewsListScript* News_GetProfileNewsList(MwId UserId); // Maniascript
  CWebServicesTaskResult_UserNewsListScript* News_GetSpaceNewsList(MwId UserId); // Maniascript
  CWebServicesTaskResult* News_OpenLink(MwId UserId, string Type, string Param); // Maniascript
  CWebServicesTaskResult* Profile_ShowFirstPartyProfile(MwId UserId, string WebServicesUserId); // Maniascript
  CWebServicesTaskResult* Profile_ShowUbisoftConnectProfile(MwId UserId, string WebServicesUserId); // Maniascript
  CWebServicesTaskResult_UserPrestigeListScript* Prestige_GetAccountPrestigeList(MwId UserId); // Maniascript
  CWebServicesTaskResult_UserPrestigeScript* Prestige_GetCurrentAccountPrestige(MwId UserId); // Maniascript
  CWebServicesTaskResult_UserPrestigeScript* Prestige_GetCurrentAccountPrestigeForUser(MwId UserId, string WebServicesUserId); // Maniascript
  CWebServicesTaskResult_PrestigeListScript* Prestige_GetPrestigeList(MwId UserId, CGameUserManagerScript::EPrestigeMode Mode, uint Year, string CategoryType); // Maniascript
  CWebServicesTaskResult_UserPrestigeScript* Prestige_SetCurrentAccountPrestige(MwId UserId, string PrestigeId); // Maniascript
  CWebServicesTaskResult_UserPrestigeScript* Prestige_ResetCurrentAccountPrestige(MwId UserId); // Maniascript
  CWebServicesTaskResult_SquadScript* Squad_AcceptInvitation(MwId UserId, string SquadId); // Maniascript
  CWebServicesTaskResult_SquadScript* Squad_CancelInvitation(MwId UserId, string SquadId, string WebServicesUserId); // Maniascript
  CWebServicesTaskResult_SquadScript* Squad_Create(MwId UserId, wstring SquadName, uint Size); // Maniascript
  CWebServicesTaskResult_SquadScript* Squad_Create_v2(MwId UserId, wstring SquadName, uint Size, string Type); // Maniascript
  CWebServicesTaskResult_SquadScript* Squad_DeclineInvitation(MwId UserId, string SquadId); // Maniascript
  CWebServicesTaskResult_SquadScript* Squad_Get(MwId UserId, string SquadId); // Maniascript
  CWebServicesTaskResult_SquadScript* Squad_GetCurrent(MwId UserId); // Maniascript
  CWebServicesTaskResult_SquadScript* Squad_Invite(MwId UserId, string SquadId, string WebServicesUserId); // Maniascript
  CWebServicesTaskResult* Squad_JoinSession(MwId UserId, string SessionId, bool IsFirstPartySession); // Maniascript
  CWebServicesTaskResult_SquadScript* Squad_Leave(MwId UserId, string SquadId); // Maniascript
  CWebServicesTaskResult_SquadScript* Squad_RemoveMember(MwId UserId, string SquadId, string WebServicesUserId); // Maniascript
  CWebServicesTaskResult_SquadScript* Squad_SetLeader(MwId UserId, string SquadId, string WebServicesUserId); // Maniascript
  void Squad_SetLocked(MwId UserId, string SquadId, bool Locked); // Maniascript
  void Squad_SetType(MwId UserId, string SquadId, string Type); // Maniascript
  void Squad_SetEnabled(MwId UserId, bool Enabled); // Maniascript
  const UnknownType VoiceChat_Events; // Maniascript
  string VoiceChat_Channel; // Maniascript
  CGameUserManagerScript::EVoiceChatMuteSetting VoiceChat_NewRemoteUser_DefaultMuteSetting; // Maniascript
  CGameUserVoiceChat* VoiceChat_UserAdd(string WebServicesUserId); // Maniascript
  void VoiceChat_ClearUsers(); // Maniascript
  const bool VoiceChat_DisplayUI; // Maniascript
  bool VoiceChat_Mute_Myself; // Maniascript
  void VoiceChat_MuteAll(); // Maniascript
  void VoiceChat_UnmuteAll(); // Maniascript
  const MwNodPool<CGameUserVoiceChat*> VoiceChat_Users; // Maniascript
  const MwFastBuffer<CGameUserVoiceChat*> VoiceChat_Users_Local; // Maniascript
  const MwFastBuffer<CGameUserVoiceChat*> VoiceChat_Users_Remote; // Maniascript
  const MwFastBuffer<CGameUserVoiceChat*> VoiceChat_Users_Speaking; // Maniascript
  const MwFastBuffer<CGameUserVoiceChat*> VoiceChat_Users_Muted; // Maniascript
  const MwFastBuffer<CGameUserVoiceChat*> VoiceChat_Users_Remote_Muted; // Maniascript
  const bool MainUser_CanUseVoiceChat; // Maniascript
  CGameUserVoiceChat* VoiceChat_FindUserFromWebServicesUserId(string WebServicesUserId); // Maniascript
  uint Subscription_GetEndTimeStamp(MwId UserId, string SubscriptionName); // Maniascript
  bool Subscription_IsRefreshingInfo(MwId UserId); // Maniascript
  CWebServicesTaskResult_StringInt* Tag_GetClubTag(MwId UserId); // Maniascript
  CWebServicesTaskResult_ClubTagListScript* Tag_GetClubTagList(MwId UserId, MwFastBuffer<wstring>& WebServicesUserIdList); // Maniascript
  CWebServicesTaskResult* Tag_SetClubTag(MwId UserId, wstring ClubTag); // Maniascript
  CWebServicesTaskResult* UbisoftConnect_Show(MwId UserId); // Maniascript
  CWebServicesTaskResult* UbisoftConnect_ShowReward(MwId UserId, string RewardId); // Maniascript
  CWebServicesTaskResult* UbisoftConnect_ShowSocialFriendList(MwId UserId); // Maniascript
  CWebServicesTaskResult* UbisoftConnect_ShowSocialInvitationList(MwId UserId); // Maniascript
  CWebServicesTaskResult* UbisoftClub_Launch(MwId UserId, CGameUserManagerScript::EUbisoftClubFlow UbisoftClubFlow, string RewardId); // Maniascript
  CWebServicesTaskResult* UbisoftClub_LaunchAndCompleteActions(MwId UserId, CGameUserManagerScript::EUbisoftClubFlow UbisoftClubFlow, string RewardId, MwFastBuffer<wstring>& ActionIdList); // Maniascript
  CWebServicesTaskResult* Uplay_OpenOverlay(MwId UserId, CGameUserManagerScript::EUplayOverlaySection OverlaySection); // Maniascript
};

struct CGameDataFileTask_Map_NadeoServices_AddFavorite : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_Map_NadeoServices_RemoveFavorite : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_Map_NadeoServices_GetFavoriteList : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_Map_NadeoServices_GetFavoriteListByUid : public CWebServicesTaskSequence {
};

// Description: ""
struct CGameModuleMenuBase : public CGameManiaAppTitle {
  CGameModuleMenuPage* GetFirstPage(wstring PageId); // Maniascript
  void Menu_Goto(wstring PageId); // Maniascript
  void Menu_Back(); // Maniascript
  void Menu_Previous(); // Maniascript
  void Menu_Quit(); // Maniascript
};

struct CGameModuleMenuBrowser : public CGameModuleMenuComponent {
  enum class CGameModuleMenuBrowser::EFileType {
    Maps = 0,
    Skins = 1,
    Items = 2,
    Blocks = 3,
    Scripts = 4,
    Images = 5,
    Manialinks = 6,
    Packs = 7,
    Actions = 8,
    Modules = 9,
    Meshes = 10,
    Replays = 11,
  };
  enum class CGameModuleMenuBrowser::EFileAction {
    Select = 0,
    Save = 1,
    MultiSelect = 2,
  };
  const bool HasFinished; // Maniascript
  const MwFastBuffer<wstring> Selection; // Maniascript
  void SetFileType(CGameModuleMenuBrowser::EFileType FileType); // Maniascript
  void SetFileAction(CGameModuleMenuBrowser::EFileAction FileAction); // Maniascript
};

struct CGameHapticDevice : public CMwNod {
};

// Description: "Script API to communicate with plugins."
struct CGamePluginInterfacesScript : public CMwNod {
  CGameManiaplanetPluginInterface* GetInterface(wstring Name); // Maniascript
  const MwFastBuffer<CGameManiaplanetPluginInterface*> Interfaces; // Maniascript
};

struct CGameModuleMenuComponent : public CMwNod {
  CGameUILayer* const ComponentLayer; // Maniascript
  void Hide(); // Maniascript
  void Show(); // Maniascript
};

struct CGameScoreComputer_MultiAsyncLevel : public CMwNod {
  CGameScoreComputer_MultiAsyncLevel();

};

// Description: "Task result containing a list of account trophy gain."
struct CWebServicesTaskResult_AccountTrophyGainListScript : public CWebServicesTaskResult_AccountTrophyGainList {
  const MwFastBuffer<CAccountTrophyGain*> AccountTrophyGainList; // Maniascript
};

struct CGameScoreTask_LoadPlayerScore : public CWebServicesTaskSequence {
};

struct CGameScoreTask_SynchronizePlayerScore : public CWebServicesTaskSequence {
};

// Description: "Task result containing a Skin info from NadeoServices."
struct CWebServicesTaskResult_NadeoServicesSkinScript : public CWebServicesTaskResult_NadeoServicesSkin {
  CNadeoServicesSkin* const Skin; // Maniascript
};

// Description: "Task result containing a list of Skin info from NadeoServices."
struct CWebServicesTaskResult_NadeoServicesSkinListScript : public CWebServicesTaskResult_NadeoServicesSkinList {
  const MwFastBuffer<CNadeoServicesSkin*> SkinList; // Maniascript
};

struct CGameEditorPluginModuleScriptEvent : public CGameManiaAppScriptEvent {
  enum class CGameEditorPluginModuleScriptEvent::EType {
    LayerCustomEvent = 0,
    KeyPress = 1,
    MenuNavigation = 3,
    Exit = 4,
    FileNew = 5,
    FileOpen = 6,
    FileSave = 7,
    FileSaveAs = 8,
  };
  const CGameEditorPluginModuleScriptEvent::EType Type; // Maniascript
  void Eat(); // Maniascript
};

struct CWebServicesTaskResult_MapRecordList : public CWebServicesTaskResult {
};

struct CGameScoreTask_SetSeason : public CWebServicesTaskSequence {
};

struct CGameModuleMenuLadderRankings : public CGameModuleMenuComponent {
};

struct CGameScriptHandlerPlaygroundModuleStore : public CGameManialinkScriptHandler {
  const uint CurrentMoney; // Maniascript
  const MwFastBuffer<CGameModuleScriptStoreCategory*> StoreContent; // Maniascript
  void SetCategory(uint CategoryIndex); // Maniascript
  void SelectItem(uint ItemIndex); // Maniascript
  void BuySelectedItem(); // Maniascript
  void ClearItemSelection(); // Maniascript
  void Quit(); // Maniascript
  const MwFastBuffer<wstring> BlockedItemsUrls; // Maniascript
};

struct CGameModuleMenuServerBrowser : public CGameModuleMenuComponent {
};

struct CGamePlaygroundModuleClientHud : public CGameManiaAppPlaygroundCommon {
  void ToggleStore(); // Maniascript
  void ToggleInventory(); // Maniascript
};

struct CGamePlaygroundModuleServer : public CMwNod {
  void Hide(); // Maniascript
  void Hide2(CGamePlaygroundUIConfig* UIConfig); // Maniascript
  void Show(); // Maniascript
  void Show2(CGamePlaygroundUIConfig* UIConfig); // Maniascript
  bool IsVisible(CGamePlaygroundUIConfig* UIConfig); // Maniascript
};

struct CGameScriptHandlerTitleModuleMenu : public CGameManiaAppTitleLayerScriptHandler {
  const MwFastBuffer<CGameModuleMenuComponent*> Components; // Maniascript
  CGameModuleMenuComponent* GetFirstComponent(string Name); // Maniascript
};

struct CGameEditorAnimChar : public CMwNod {
  bool DbgDrawJointCurves;
  bool DbgIKs;
  bool DbgTestIK;
  float DbgIKTolerance;
  int DbgJoint;
  int DbgSkel;
  bool DrawBoneInfos;
  bool DrawRefBox;
  bool Collisions;
  bool DrawJointWeights;
  uint IKChainLength;
};

struct CGameScriptHandlerPlaygroundModuleInventory : public CGameManialinkScriptHandler {
  const MwFastBuffer<CGameModuleInventoryCategory*> InventoryContent; // Maniascript
  void Quit(); // Maniascript
  void DropSelectedSlot(); // Maniascript
  void EquipSelectedSlot(); // Maniascript
  void ClearSlotSelection(); // Maniascript
  void SetCategory(uint CategoryIndex); // Maniascript
  void SelectSlot(uint SlotIndex); // Maniascript
};

struct CGameModuleInventoryCategory : public CMwNod {
  const string Name; // Maniascript
  const string Description; // Maniascript
  const uint ItemClass; // Maniascript
  const wstring IconUrl; // Maniascript
  const MwFastBuffer<CGameModuleScriptItem*> SlotsItems; // Maniascript
  const MwFastBuffer<uint> SlotsItemsQuantity; // Maniascript
};

struct CGameModuleScriptItem : public CMwNod {
  const wstring Url; // Maniascript
  const string IconUrl; // Maniascript
  const string Name; // Maniascript
  const string Description; // Maniascript
  const uint Occupation; // Maniascript
  void IsItem(); // Maniascript
  void HasAction(); // Maniascript
};

struct CGameMasterServerTask_SetBuddies : public CNetMasterServerRequestTask {
};

struct CGameDataFileTask_Skin_NadeoServices_Get : public CWebServicesTaskSequence {
};

struct CGameScoreTask_SetNewMapRecord : public CWebServicesTaskSequence {
};

// Description: "An playground Store module."
struct CGamePlaygroundModuleServerStore : public CGamePlaygroundModuleServer {
  void Reset(); // Maniascript
  void Reset2(CGameScriptPlayer* Player); // Maniascript
  void SetMoney(CGameScriptPlayer* Player, uint Amount); // Maniascript
  uint GetMoney(CGameScriptPlayer* Player); // Maniascript
  bool AddMoney(CGameScriptPlayer* Player, uint Amount); // Maniascript
  bool SubMoney(CGameScriptPlayer* Player, uint Amount); // Maniascript
  void SetActionLevel(CGameScriptPlayer* Player, wstring ActionUrl, uint ActionLevel); // Maniascript
  uint GetActionLevel(CGameScriptPlayer* Player, wstring ActionUrl); // Maniascript
  bool SetPlayerData(CGameScriptPlayer* Player, string Data); // Maniascript
  string GetPlayerData(CGameScriptPlayer* Player); // Maniascript
  void SetItemCanBeBought(CGameScriptPlayer* Player, wstring ActionUrl, bool CanBeBought); // Maniascript
  bool GetItemCanBeBought(CGameScriptPlayer* Player, wstring ActionUrl); // Maniascript
  MwFastBuffer<wstring>& Hack_GetAllItemsUrls(); // Maniascript
};

struct CGamePlaygroundModuleServerHud : public CMwNod {
  CGamePlaygroundModuleServerInventory* Inventory; // Maniascript
  CGamePlaygroundModuleServerStore* Store; // Maniascript
  CGamePlaygroundModuleServerScoresTable* ScoresTable; // Maniascript
  CGamePlaygroundModuleServerChrono* Chrono; // Maniascript
  CGamePlaygroundModuleServerSpeedMeter* SpeedMeter; // Maniascript
  CGamePlaygroundModuleServerPlayerState* PlayerState; // Maniascript
  CGamePlaygroundModuleServerTeamState* TeamState; // Maniascript
  const MwFastBuffer<CGamePlaygroundModuleServer*> Modules; // Maniascript
  void SwitchContext(CGameScriptPlayer* Player, string ContextName); // Maniascript
  void SetDefaultContext(CGameScriptPlayer* Player); // Maniascript
  MwId RetrieveModuleId(string ModuleName); // Maniascript
};

struct CGamePlaygroundModuleManagerServer : public CMwNod {
};

struct CGamePlaygroundModuleServerInventory : public CGamePlaygroundModuleServer {
  uint AddItem(CGameScriptPlayer* Player, wstring Url, uint Quantity); // Maniascript
  bool AddAction(CGameScriptPlayer* Player, wstring Url); // Maniascript
  uint RemoveInventoryItem(CGameScriptPlayer* Player, wstring Url, uint Quantity); // Maniascript
  uint GetInventoryItemQuantity(CGameScriptPlayer* Player, wstring Url); // Maniascript
  bool IsInventoryItemStored(CGameScriptPlayer* Player, wstring Url); // Maniascript
  MwFastBuffer<wstring>& GetStoredItemsList(CGameScriptPlayer* Player); // Maniascript
  MwFastBuffer<wstring>& GetStoredActionsList(CGameScriptPlayer* Player); // Maniascript
};

struct CGamePlaygroundModuleServerScoresTable : public CGamePlaygroundModuleServer {
  enum class CGamePlaygroundModuleServerScoresTable::EColumnType {
    CustomString = 0,
    CustomNatural = 1,
    CustomInteger = 2,
    CustomReal = 3,
    CustomTime = 4,
    Avatar = 5,
    Name = 6,
    ManiaStars = 7,
    Tools = 8,
    Tags = 9,
    TMBestTime = 10,
    TMPrevTime = 11,
    TMBestLapTime = 12,
    TMStunts = 13,
    TMRespawns = 14,
    TMCheckpoints = 15,
    TMPoints = 16,
    TMPrevRaceDeltaPoints = 17,
    SMPoints = 18,
    SMRoundPoints = 19,
  };
  enum class CGamePlaygroundModuleServerScoresTable::EScoreSortOrder {
    Default = 0,
    Name = 1,
    LadderRanking = 2,
    TMPoints = 3,
    TMBestTime = 4,
    TMBestLapTime = 5,
    TMStunts = 6,
    TMRespawns = 7,
    TMCheckpoints = 8,
    TMPrevTime = 9,
    SMPoints = 10,
    SMRoundPoints = 11,
  };
  void SetFooterText(wstring FooterText); // Maniascript
  void ResetTargetCustomColumns(CGamePlaygroundScore* Score); // Maniascript
  void ResetCustomColumns(); // Maniascript
  void Scores_Sort(CGamePlaygroundModuleServerScoresTable::EScoreSortOrder SortOrder); // Maniascript
  void SetColumnValueString(CGamePlaygroundScore* Score, string ColumnId, string ColumnValue); // Maniascript
  void SetColumnValueInteger(CGamePlaygroundScore* Score, string ColumnId, int ColumnValue); // Maniascript
  void SetColumnValueReal(CGamePlaygroundScore* Score, string ColumnId, float ColumnValue); // Maniascript
  void SetColumnVisibility(CGamePlaygroundModuleServerScoresTable::EColumnType Type, bool Visibility); // Maniascript
  void SetCustomColumnVisibility(string ColumnId, bool Visibility); // Maniascript
  void SetColor(CGamePlaygroundScore* Score, vec3 Color); // Maniascript
  void ResetColor(CGamePlaygroundScore* Score); // Maniascript
};

struct CWebServicesTaskResult_Ghost : public CWebServicesTaskResult {
};

struct CGameScoreTask_GetPlayerMapRecordGhost : public CWebServicesTaskSequence {
};

// Description: "Score and leaderboard manager."
struct CGameScoreAndLeaderBoardManagerScript : public CMwNod {
  enum class CGameScoreAndLeaderBoardManagerScript::ELocalScoreStatus {
    None = 0,
    Loading = 1,
    NotLoaded = 2,
    Loaded = 3,
  };
  enum class CGameScoreAndLeaderBoardManagerScript::EMasterServerScoreStatus {
    None = 0,
    Synchronizing = 1,
    NotSynchronized = 2,
    Synchronized = 3,
  };
  const MwFastBuffer<CWebServicesTaskResult*> TaskResults; // Maniascript
  void TaskResult_Release(MwId TaskId); // Maniascript
  void ReleaseTaskResult(MwId TaskId); // Maniascript
  void ReleaseGhost(MwId GhostId); // Maniascript
  CGameGhostScript* Playground_GetPlayerGhost(CGameScriptPlayer* GamePlayer); // Maniascript
  void Score_GetFromCompressedPlatformScore(uint CompressedScore, uint RespawnCount, uint Time); // Maniascript
  uint Map_GetRecord_v2(MwId UserId, string MapUid, string ScopeType, string ScopeId, string GameMode, string GameModeCustomData); // Maniascript
  void Map_GetRecordFullScore(MwId UserId, string MapUid, string ScopeType, string ScopeId, string GameMode, string GameModeCustomData, uint Time, uint Points, uint RespawnCount); // Maniascript
  CWebServicesTaskResult_GhostScript* Map_GetRecordGhost_v2(MwId UserId, string MapUid, string ScopeType, string ScopeId, string GameMode, string GameModeCustomData); // Maniascript
  uint Map_GetMedal(MwId UserId, string MapUid, string ScopeType, string ScopeId, string GameMode, string GameModeCustomData); // Maniascript
  uint Map_GetMultiAsyncLevelRecord_v2(string MapUid, string ScopeType, string ScopeId, string GameMode, string GameModeCustomData, uint MultiAsyncLevel); // Maniascript
  CWebServicesTaskResult_GhostScript* Map_GetMultiAsyncLevelRecordGhost_v2(string MapUid, string ScopeType, string ScopeId, string GameMode, string GameModeCustomData, uint MultiAsyncLevel); // Maniascript
  CWebServicesTaskResult_MapRecordListScript* Map_GetPlayerListRecordList(MwId UserId, MwFastBuffer<wstring>& WebServicesUserIdList, string MapUid, string ScopeType, string ScopeId, string GameMode, string GameModeCustomData); // Maniascript
  CWebServicesTaskResult* Map_LoadPBScoreList(MwId UserId, MwFastBuffer<wstring>& MapUidList, string GameMode, string GameModeCustomData); // Maniascript
  CWebServicesTaskResult_SeasonScript* Season_Create(MwId UserId, wstring SeasonName, string GameMode, string GameModeCustomData, string MapRecordType, uint StartTimeStamp, uint EndTimeStamp, MwFastBuffer<wstring>& MapUidList); // Maniascript
  CWebServicesTaskResult_SeasonScript* Season_Update(MwId UserId, string SeasonId, wstring SeasonName, string GameMode, string GameModeCustomData, string MapRecordType, uint StartTimeStamp, uint EndTimeStamp, MwFastBuffer<wstring>& MapUidList); // Maniascript
  CWebServicesTaskResult_SeasonScript* Season_AddMapList(MwId UserId, string SeasonId, MwFastBuffer<wstring>& MapUidList); // Maniascript
  CWebServicesTaskResult_SeasonScript* Season_RemoveMapList(MwId UserId, string SeasonId, MwFastBuffer<wstring>& MapUidList); // Maniascript
  CWebServicesTaskResult_SeasonScript* Season_Get(MwId UserId, string SeasonId); // Maniascript
  CWebServicesTaskResult_SeasonListScript* Season_GetList(MwId UserId, MwFastBuffer<wstring>& SeasonIdList); // Maniascript
  CWebServicesTaskResult_SeasonListScript* Season_GetListFromUser(MwId UserId, string WebServicesUserId); // Maniascript
  CWebServicesTaskResult* Season_LoadScore(MwId UserId, string SeasonId); // Maniascript
  CWebServicesTaskResult* Season_LoadScoreList(MwId UserId, MwFastBuffer<wstring>& SeasonIdList); // Maniascript
  CWebServicesTaskResult_AccountTrophyGainHistoryScript* Trophy_GetGainHistory(MwId UserId, uint Offset, uint Count); // Maniascript
  CWebServicesTaskResult_AccountTrophyGainHistoryScript* Trophy_GetGainHistoryByType(MwId UserId, uint TrophyType, uint Offset, uint Count); // Maniascript
  CWebServicesTaskResult_AccountTrophyLastYearSummaryScript* Trophy_GetAccountLastYearSummary(MwId UserId); // Maniascript
  CWebServicesTaskResult_AccountTrophyLastYearSummaryScript* Trophy_GetLastYearSummary(MwId UserId); // Maniascript
  CWebServicesTaskResult_AccountTrophyLastYearSummaryScript* Trophy_GetLastYearSummaryForUser(MwId UserId, string WebServicesUserId); // Maniascript
  CWebServicesTaskResult_TrophySoloMedalAchievementSettingsScript* Trophy_GetSoloMedalAchievementSettings(MwId UserId, string SoloMedalAchievementType); // Maniascript
};

// Description: "Results of task requesting the display name of registered login."
struct CWebServicesTaskResult_GetDisplayNameScriptResult : public CWebServicesTaskResult {
  wstring GetDisplayName(string WebServicesUserId); // Maniascript
  wstring GetDisplayName2(string WebServicesUserId, bool IsFirstPartyDisplayName); // Maniascript
};

struct CGameEditorTimeLine : public CMwNod {
  const uint Time;
  void TimeStop();
  void TimeSlowForward();
  void TimeNormalForward();
  void TimeFirstFrame();
  void TimeLastFrame();
  void TimePlayStop();
  void TimeLineShowFull();
  void KeyInsert();
  void KeyRemove();
  void KeysSelectAll();
  void KeyPrev();
  void KeyNext();
  void Copy();
  void KeyCut();
  void Paste();
  void TimeLineNext();
  void TimeLinePrev();
  void KeysRemoveAll();
};

struct CGameEditorAnimChar_Interface : public CMwNod {
  CScene2d* const Scene2d;
  float EntryParamTransitionDuration;
  const float Duration;
};

struct CGameEditorAnimSet : public CGameEditorBase {
  CGameAnimSet* const AnimSet;
};

struct CGameAnimClipNod : public CMwNod {
  CGameAnimClipNod();

};

struct CGameEditorVehicle : public CGameEditorParent {
};

struct CGameEditorParent : public CGameSwitcherModule {
  ISceneVis* const GameScene;
  CGameEditorPluginAPI* const PluginAPI;
};

struct CGameCtnMasterServerTask_GetLeagues : public CNetMasterServerRequestTask {
};

struct CGameMasterServerTask_Connect : public CNetMasterServerTask_Connect {
};

// Description: "Online presence"
struct CGameMasterServerPlayerOnlinePresence : public CMwNod {
  const string Login; // Maniascript
  const wstring DisplayName; // Maniascript
  const string ServerLogin; // Maniascript
  const bool IsOnline; // Maniascript
};

struct CWebServicesTaskResult_OnlinePresenceList : public CWebServicesTaskResult {
};

struct CGameCtnMasterServerTask_GetOnlinePresenceForPlayers : public CNetMasterServerRequestTask {
};

// Description: "Results of task requesting the list of servers where the logins are playing."
struct CGameMasterServerRichPresenceTaskResult_GetOnlinePresenceForPlayersScript : public CGameMasterServerRichPresenceTaskResult_PlayerOnlinePresenceList {
  const MwFastBuffer<CGameMasterServerPlayerOnlinePresence*> OnlinePresences; // Maniascript
};

struct CGameMasterServerRichPresenceManager : public CMwNod {
};

struct CGameMasterServerRichPresenceTask_UpdatePresence : public CWebServicesTaskSequence {
};

// Description: "User privileges manager."
struct CGameUserPrivilegesManagerScript : public CMwNod {
  enum class CGameUserPrivilegesManagerScript::ECommunicationRestrictionLevel {
    Unknown = 0,
    Public = 1,
    Private = 2,
    FriendsOnly = 3,
    FriendsOfFriends = 4,
  };
  enum class CGameUserPrivilegesManagerScript::EPrivilege {
    CommunicationText = 0,
    CommunicationTextRequired = 1,
    CommunicationVoice = 2,
    CommunicationVoiceRequired = 3,
    CrossPlay = 4,
    CrossPlayRequired = 5,
    DownloadUserCreatedContent = 6,
    PlayMultiplayerAsync = 7,
    PlayMultiplayerAsyncWithUGC = 8,
    PlayMultiplayerMode = 9,
    PlayMultiplayerModeWithUGC = 10,
    PlayMultiplayerSession = 11,
    PlayMultiplayerSessionWithUGC = 12,
    UploadUserCreatedContent = 13,
    UseUserCreatedContent = 14,
    UseUserCreatedContentRequired = 15,
    ViewOnlinePresence = 16,
    ViewUserCreatedContent = 17,
    ViewUserCreatedContentRequired = 18,
    ViewUserGameHistory = 19,
    VoiceChat = 20,
    CommunicationsVoiceWithCrossplayUsers = 21,
    CommunicationsTextWithCrossplayUsers = 22,
  };
  const MwFastBuffer<CWebServicesTaskResult*> TaskResults; // Maniascript
  void TaskResult_Release(MwId TaskId); // Maniascript
  void ReleaseTaskResult(MwId TaskId); // Maniascript
  CWebServicesTaskResult* CheckPrivilege(MwId UserId, CGameUserPrivilegesManagerScript::EPrivilege Privilege); // Maniascript
  CWebServicesTaskResult* CheckPrivilegeForAllUsers(CGameUserPrivilegesManagerScript::EPrivilege Privilege); // Maniascript
  CWebServicesTaskResult_CheckTargetedPrivilegeResultScript* CheckTargetedPrivilege(MwId UserId, CGameUserPrivilegesManagerScript::EPrivilege Privilege, MwFastBuffer<wstring>& WebServicesUserIdList); // Maniascript
  CWebServicesTaskResult_CheckTargetedPrivilegeResultScript* CheckTargetedPrivilegeForAllUsers(CGameUserPrivilegesManagerScript::EPrivilege Privilege, MwFastBuffer<wstring>& WebServicesUserIdList); // Maniascript
  uint GetPermissionValue(MwId UserId, string Name); // Maniascript
  bool IsPermissionEnabled(MwId UserId, string Name); // Maniascript
  CWebServicesTaskResult* RefreshPermissions(MwId UserId); // Maniascript
  bool IsSeasonPlayable(MwId UserId, string SeasonId); // Maniascript
  CGameUserPrivilegesManagerScript::ECommunicationRestrictionLevel Communication_GetRestrictionLevel(MwId UserId); // Maniascript
};

// Description: "Results of task requesting if the registered login have a privilege."
struct CWebServicesTaskResult_CheckTargetedPrivilegeResultScript : public CWebServicesTaskResult_CheckTargetedPrivilegeResult {
  bool HasPrivilege(string Login); // Maniascript
  wstring GetDenyReason(string Login); // Maniascript
};

struct CGameMasterServerRichPresenceTask_GetOnlinePresence : public CWebServicesTaskSequence {
};

struct CGameMasterServerRichPresenceTaskResult_PlayerOnlinePresenceList : public CWebServicesTaskResult {
};

struct CGameMasterServerRichPresenceManagerScript : public CMwNod {
  enum class CGameMasterServerRichPresenceManagerScript::ERichPresence {
    Undefined = 0,
    MainMenu = 1,
    Solo = 2,
    Multi = 3,
    Party = 4,
    Create = 5,
  };
  const MwFastBuffer<CWebServicesTaskResult*> TaskResults; // Maniascript
  void TaskResult_Release(MwId TaskId); // Maniascript
  void ReleaseTaskResult(MwId TaskId); // Maniascript
  void SetPresence(MwId UserId, CGameMasterServerRichPresenceManagerScript::ERichPresence RichPresence); // Maniascript
  CGameMasterServerRichPresenceTaskResult_GetOnlinePresenceForPlayersScript* GetOnlinePresenceForPlayers(MwId UserId, MwFastBuffer<wstring>& WebServicesUserIdList); // Maniascript
};

struct CGameCtnMasterServerTask_BuyFullGame : public CWebServicesTaskSequence {
};

struct CGameNetFormVoiceChat : public CGameNetForm {
  CGameNetFormVoiceChat();

};

struct CGameScoreTask_SetTrophyLiveTimeAttackAchievementResults : public CWebServicesTaskSequence {
};

struct CGameScoreTask_UploadNewMapRecord : public CWebServicesTaskSequence {
};

struct CGameZoneManagerScript : public CMwNod {
  const MwFastBuffer<CWebServicesTaskResult*> TaskResults; // Maniascript
  void TaskResult_Release(MwId TaskId); // Maniascript
  void GetPathAndName(wstring ZoneFullPath, wstring ZonePath, wstring ZoneName); // Maniascript
  CWebServicesTaskResult_ZoneListScript* GetZoneList(MwFastBuffer<wstring>& ZoneIdList); // Maniascript
  CWebServicesTaskResult_ZoneListScript* GetChildZoneList(wstring ZoneFullPath); // Maniascript
  CWebServicesTaskResult_UserZoneListScript* RetrieveUserZoneList(MwId UserId, MwFastBuffer<wstring>& WebServicesUserIdList); // Maniascript
  CWebServicesTaskResult* SetUserZone(MwId UserId, wstring ZoneFullPath); // Maniascript
};

struct CGameModuleScriptStoreItem : public CGameModuleScriptItem {
  const string StoreName; // Maniascript
  const string StoreDesc; // Maniascript
  const uint CurrentLevel; // Maniascript
  const MwFastBuffer<uint> Prices; // Maniascript
};

struct CGameModuleScriptStoreCategory : public CMwNod {
  const string Name; // Maniascript
  const MwFastBuffer<CGameModuleScriptStoreItem*> Items; // Maniascript
};

// userName: 'LaunchedCP'
// File extension: 'LaunchedCP.Gbx'
struct CGameSaveLaunchedCheckpoints : public CMwNod {
  CGameSaveLaunchedCheckpoints();

};

struct CGameEditorMainPlugin : public CGameEditorPlugin {
  enum class CGameEditorMainPlugin::EMeshEditorAPI {
    Materials = 0,
    Interactions = 1,
    Display = 2,
    Global = 3,
    Sets = 4,
    Voxel = 5,
    PickInfo = 6,
    UndoRedo = 7,
  };
  void Help_Open(); // Maniascript
  void Help_Close(); // Maniascript
  CGameEditorPluginHandle* GetPluginHandle(string Name); // Maniascript
  void SendPluginEvent(CGameEditorPluginHandle* Handle, wstring Type, MwFastBuffer<wstring>& Data); // Maniascript
  void Context_SetActive(wstring ContextName, bool IsActive); // Maniascript
  bool Context_IsActive(wstring ContextName); // Maniascript
  bool Binding_IsActive(wstring BindingName); // Maniascript
  bool Binding_IsActiveInContext(wstring ContextName, wstring BindingName); // Maniascript
  void Plugin_SetClearance_MeshEditor(CGameEditorPluginHandle* Handle, CGameEditorMainPlugin::EMeshEditorAPI API, bool IsAllowed); // Maniascript
  const MwFastBuffer<CGameEditorPluginHandle*> Plugins; // Maniascript
  void Module_Load(wstring ModuleName); // Maniascript
  const MwFastBuffer<CGameModuleEditorBase*> Modules; // Maniascript
};

struct CGameDataFileTask_Skin_NadeoServices_GetList : public CWebServicesTaskSequence {
};

// Description: "UIConfig Event"
struct CGamePlaygroundUIConfigEvent : public CScriptBaseEvent {
  enum class CGamePlaygroundUIConfigEvent::EType {
    Unknown = 0,
    OnModuleCustomEvent = 1,
    OnModuleShowRequest = 2,
    OnModuleHideRequest = 3,
    OnModuleStorePurchase = 4,
    OnModuleInventoryDrop = 5,
    OnModuleInventoryEquip = 6,
    OnLayerCustomEvent = 7,
  };
  enum class CGamePlaygroundUIConfigEvent::EModuleType {
    Undefined = 0,
    Hud = 1,
    Inventory = 2,
    Store = 3,
    ScoresTable = 4,
    Equipment = 5,
    MenuBase = 6,
    MenuPage = 7,
    Chrono = 8,
    SpeedMeter = 9,
    PlayerState = 10,
    TeamState = 11,
  };
  const CGamePlaygroundUIConfigEvent::EType Type; // Maniascript
  CGamePlaygroundUIConfig* const UI; // Maniascript
  CGamePlaygroundUIConfig* const UIConfig; // Maniascript
  const CGamePlaygroundUIConfigEvent::EModuleType ModuleType; // Maniascript
  const wstring Param1; // Maniascript
  const MwFastBuffer<wstring> Param2; // Maniascript
  CGameUILayer* const CustomEventLayer; // Maniascript
  const wstring CustomEventType; // Maniascript
  const MwFastBuffer<wstring> CustomEventData; // Maniascript
  const wstring ItemUrl; // Maniascript
  const uint Quantity; // Maniascript
};

struct CWebServicesTaskResult_NadeoServicesMap : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_NadeoServicesMapList : public CWebServicesTaskResult {
};

// Description: "Script API to create Packs (can be Titlepacks or plain ManiaCredited data), and generate new builds of these packs."
struct CGamePackCreatorScript : public CMwNod {
  void RegisterPackForEditedTitle(); // Maniascript
  const bool RegisterPack_IsInProgess; // Maniascript
  CGamePackCreator_PackScript* const CurrentPack; // Maniascript
  MwId Build_Begin(CGamePackCreator_PackScript* Pack, CGamePackCreator_TitleInfoScript* TitleInfo); // Maniascript
  void Build_AddFile(MwId BuildId, wstring FileName); // Maniascript
  void Build_AddFolder(MwId BuildId, wstring FolderName); // Maniascript
  void Build_AddFile_Ex(MwId BuildId, wstring FileName, bool IsPublic, bool IsInternal, bool NoAutomaticDeps); // Maniascript
  void Build_AddFolder_Ex(MwId BuildId, wstring FolderName, bool IsPublic, bool IsInternal, bool NoRecursion, bool NoAutomaticDeps); // Maniascript
  void Build_Generate(MwId BuildId, bool Upload); // Maniascript
  bool Build_IsGenerated(MwId BuildId); // Maniascript
  wstring Build_ErrorMessage(MwId BuildId); // Maniascript
  void Build_End(MwId BuildId); // Maniascript
};

// Description: "A pack"
struct CGamePackCreator_PackScript : public CMwNod {
  const MwId PackId; // Maniascript
  const MwId CreatorId; // Maniascript
  const bool IsTitlePack; // Maniascript
  void Recipients_Add(string Login, uint UseCost, uint GetCost); // Maniascript
  void Recipients_Remove(string Login); // Maniascript
  const MwFastBuffer<CGamePackCreator_RecipientScript*> Recipients; // Maniascript
};

// Description: "Script API to create Pack files (can be Titlepack or plain ManiaCredited data)."
struct CGamePackCreator_TitleInfoScript : public CMwNod {
  const MwId TitleId; // Maniascript
  const MwId MakerTitleId; // Maniascript
  wstring DisplayName; // Maniascript
  wstring Description; // Maniascript
  string InfoUrl; // Maniascript
  string DownloadUrl; // Maniascript
  string TitleVersion; // Maniascript
  string AllowedClientTitleVersion; // Maniascript
  string BaseTitleIds; // Maniascript
  wstring ForcedPlayerModel; // Maniascript
  wstring Packaging_ImageFileName; // Maniascript
  wstring Packaging_LogosFileName; // Maniascript
  string Packaging_Group; // Maniascript
  string Station_ManialinkUrl; // Maniascript
  wstring Menus_BgReplayFileName; // Maniascript
  string Menus_ManiaAppFileName; // Maniascript
  wstring Menus_MusicFileName; // Maniascript
  const bool Solo_HasCampaign; // Maniascript
  wstring FallbackFontFileName; // Maniascript
  wstring Hud3dFontFileName; // Maniascript
  wstring MusicFolder; // Maniascript
  wstring Editor_MusicFileName; // Maniascript
};

// Description: "A recipient included in the bill for a Pack."
struct CGamePackCreator_RecipientScript : public CMwNod {
  const string Login; // Maniascript
  const uint GetCost; // Maniascript
  const uint UseCost; // Maniascript
};

// File extension: 'DecorationMaterialModifiers.Gbx'
struct CGameCtnDecorationMaterialModifiers : public CMwNod {
  CGameCtnDecorationMaterialModifiers();

  CPlugGameSkinAndFolder* LowHeightMaterialModifier;
  MwFastBuffer<float> Rules_MinHeights;
  MwFastBuffer<CPlugGameSkinAndFolder*> Rules_MaterialModifiers;
};

struct CGameModuleMenuPage : public CMwNod {
  const wstring Name; // Maniascript
  const MwFastBuffer<CGameModuleMenuComponent*> Components; // Maniascript
};

// Description: "API for editor plugins."
struct CGameEditorPlugin : public CGameManiaApp {
  enum class CGameEditorPlugin::EInteractionStatus {
    Active = 0,
    Closed = 1,
    Aborted = 2,
  };
  const MwNodPool<CGameManiaAppScriptEvent*> PendingEvents; // Maniascript
  CGameEditorModule* const ModuleEditor; // Maniascript
  CGameEditorMesh* const MeshEditor; // Maniascript
  CGameEditorEditor* const EditorEditor; // Maniascript
  CGameEditorMediaTrackerPluginAPI* const MediaTracker; // Maniascript
  CGameEditorSkinPluginAPI* const SkinEditor; // Maniascript
  const CGameEditorPlugin::EInteractionStatus InteractionStatus; // Maniascript
};

struct CGameManialinkAnimManager : public CMwNod {
  enum class CGameManialinkAnimManager::EAnimManagerEasing {
    Linear = 0,
    QuadIn = 1,
    QuadOut = 2,
    QuadInOut = 3,
    CubicIn = 4,
    CubicOut = 5,
    CubicInOut = 6,
    QuartIn = 7,
    QuartOut = 8,
    QuartInOut = 9,
    QuintIn = 10,
    QuintOut = 11,
    QuintInOut = 12,
    SineIn = 13,
    SineOut = 14,
    SineInOut = 15,
    ExpIn = 16,
    ExpOut = 17,
    ExpInOut = 18,
    CircIn = 19,
    CircOut = 20,
    CircInOut = 21,
    BackIn = 22,
    BackOut = 23,
    BackInOut = 24,
    ElasticIn = 25,
    ElasticOut = 26,
    ElasticInOut = 27,
    ElasticIn2 = 28,
    ElasticOut2 = 29,
    ElasticInOut2 = 30,
    BounceIn = 31,
    BounceOut = 32,
    BounceInOut = 33,
  };
  void AddFromXml(CGameManialinkControl* Control, string XmlTarget, uint StartTime, uint Duration, CGameManialinkAnimManager::EAnimManagerEasing EasingFunc); // Maniascript
  void AddChainFromXml(CGameManialinkControl* Control, string XmlTarget, uint Duration, CGameManialinkAnimManager::EAnimManagerEasing EasingFunc); // Maniascript
  void FlushAndAddFromXml(CGameManialinkControl* Control, string XmlTarget, uint Duration, CGameManialinkAnimManager::EAnimManagerEasing EasingFunc); // Maniascript
  void Flush(CGameManialinkControl* Control); // Maniascript
};

struct CGameEditorEditor : public CGameCtnEditor {
  CControlFrame* const UIRoot;
  CMwNod* const EditedNod;
  CGameEditorModel* const EditedEditor;
  void FlushBindings();
  void Bindings_AddContext(wstring ContextName); // Maniascript
  void Bindings_AddBinding(wstring ContextName, wstring BindingScriptId, wstring BindingDisplayName); // Maniascript
  void Bindings_RemoveContext(wstring ContextName); // Maniascript
  void Bindings_RemoveBinding(wstring ContextName, wstring BindingName); // Maniascript
  void Bindings_RequestInput(wstring ContextName, wstring BindingName); // Maniascript
  const bool Bindings_RequestInput_Done; // Maniascript
  void Bindings_SetBindingScriptId(wstring ContextName, wstring BindingScriptId, wstring NewBindingScriptId); // Maniascript
  void Bindings_SetBindingDisplayName(wstring ContextName, wstring BindingScriptId, wstring BindingDisplayName); // Maniascript
  void Bindings_SetContextName(wstring ContextName, wstring NewContextName); // Maniascript
  const MwFastBuffer<wstring> BindingContexts; // Maniascript
  void Bindings_GetContextBindings(wstring ContextName); // Maniascript
  const MwFastBuffer<wstring> RequestedContextBindings; // Maniascript
  wstring Bindings_GetBindingActionName(wstring ContextName, wstring BindingName); // Maniascript
  wstring Bindings_GetBindingDisplayName(wstring ContextName, wstring BindingName); // Maniascript
  const MwFastBuffer<CGameEditorEvent*> PendingEvents; // Maniascript
};

struct CGameControlCameraHmdExternal : public CGameControlCameraTarget {
  CGameControlCameraHmdExternal();

};

// Description: "Task result containing a map info from NadeoServices."
struct CWebServicesTaskResult_NadeoServicesMapScript : public CWebServicesTaskResult_NadeoServicesMap {
  CNadeoServicesMap* const Map; // Maniascript
};

// Description: "Task result containing a list of map info from NadeoServices."
struct CWebServicesTaskResult_NadeoServicesMapListScript : public CWebServicesTaskResult_NadeoServicesMapList {
  const MwFastBuffer<CNadeoServicesMap*> MapList; // Maniascript
};

// Description: "A chat room."
struct CGameScriptChatRoom : public CMwNod {
  const string RoomId; // Maniascript
  const MwFastBuffer<CGameScriptChatContact*> Members; // Maniascript
};

struct NGameWaypoint_STrigger {
};

// userName: 'ObjectItem'
struct CGameObjectItem {
};

struct CGameGatePhy {
};

struct CGameTriggerGate {
};

struct CGameTriggerTeleport {
};

struct CGameShield : public CMwNod {
  CGameShield();

};

struct CGameTriggerScreen : public CMwNod {
};

// Description: "Manager for opening maniaplanet dialogs.
//               Work in progress, unavailable in public version."
struct CGameDialogsScript : public CMwNod {
  CGameDialogsScript();

  enum class CGameDialogsScript::EType {
    None = 0,
    Message = 1,
    Choice = 2,
    FileBrowser = 3,
  };
  enum class CGameDialogsScript::EResult {
    Unknown = 0,
    Ok = 1,
    Cancel = 2,
    Yes = 3,
    No = 4,
  };
  enum class CGameDialogsScript::EFileType {
    All = 0,
    Skin = 1,
    Map = 2,
    Replay = 3,
    Image = 4,
    Audio = 5,
    Video = 6,
    Text = 7,
    Zip = 8,
  };
  const bool IsAvailable; // Maniascript
  const MwFastBuffer<CGameDialogsScriptEvent*> PendingEvents; // Maniascript
  bool OpenMessage(wstring Message); // Maniascript
  bool OpenChoice(wstring Message, bool AllowCancel); // Maniascript
  bool OpenFileBrowser(wstring Folder, CGameDialogsScript::EFileType FileType); // Maniascript
};

// Description: "A filtered history"
struct CGameScriptChatHistory : public CMwNod {
  wstring Filter; // Maniascript
  const bool Filter_IsValid; // Maniascript
  const MwFastBuffer<CGameScriptChatHistoryEntry*> Entries; // Maniascript
};

// Description: "An message from the history."
struct CGameScriptChatHistoryEntry : public CMwNod {
  enum class CGameScriptChatHistoryEntry::EDirection {
    Sent = 0,
    Recieved = 1,
  };
  const CGameScriptChatHistoryEntry::EDirection Direction; // Maniascript
  const bool InRoom; // Maniascript
  const wstring Date; // Maniascript
  const MwFastBuffer<CGameScriptChatHistoryEntryMessage*> Messages; // Maniascript
  const wstring SenderNickname; // Maniascript
  const string SenderAvatarUrl; // Maniascript
  const wstring Nickname; // Maniascript
  const string AvatarUrl; // Maniascript
};

struct CGameEditorCanvas : public CMwNod {
  CGameEditorCanvas();

  MwId Layer_Create(); // Maniascript
  void Layer_Destroy(MwId Handle); // Maniascript
  void Layer_Rename(MwId Handle, string Name); // Maniascript
  string Layer_GetName(MwId Handle); // Maniascript
  void Layer_SetSelected(MwId Handle); // Maniascript
  MwId Layer_GetSelectedLayer(); // Maniascript
  void Layer_SwapLayersPriorities(MwId Handle1, MwId Handle2); // Maniascript
  void Layer_IncreasePriority(MwId Handle); // Maniascript
  void Layer_DecreasePriority(MwId Handle); // Maniascript
  MwFastBuffer<MwId>& Layer_GetLayers(); // Maniascript
  void Layer_SetVisibility(MwId Handle, bool Visible); // Maniascript
  bool Layer_IsVisible(MwId Handle); // Maniascript
};

struct CGameEditorPluginMapManager : public CMwNod {
  const MwFastBuffer<CGameEditorPluginMap*> ActivePluginsCache;
};

// Description: "API for the mesh modeler."
struct CGameEditorMesh : public CGameEditorAsset {
  enum class CGameEditorMesh::EEdgesDisplay {
    Any = 0,
    Borders = 1,
    None = 2,
  };
  enum class CGameEditorMesh::EEdgesConstraint {
    Any = 0,
    Adjacent = 1,
    Closed = 2,
  };
  enum class CGameEditorMesh::EElemType {
    Vertex = 0,
    Edge = 1,
    Face = 2,
    EVoxel = 3,
    Any = 4,
  };
  enum class CGameEditorMesh::EInteraction {
    Creation = 0,
    Pick = 1,
    PickJoint = 2,
    VoxelPickDrag_Base = 3,
    VoxelPickDrag_Creation = 4,
    VoxelPickDrag_Delete = 5,
    VoxelPickDrag_Select = 6,
    VoxelPickDrag_Pick = 7,
    VoxelPickDrag_SelectTranslation = 8,
    VoxelPickDrag_SelectRotation = 9,
    VoxelPickDrag_Paste = 10,
    Selection = 11,
    Translation = 12,
    PickTranslation = 13,
    ExtrudeTranslation = 14,
    Rotation = 15,
    PickRotation = 16,
    Scale = 17,
    Curve2D = 18,
    Merge = 19,
    Split = 20,
    Mirror = 21,
    Paste = 22,
    PasteMaterial = 23,
    BlocTransformation = 24,
    None = 25,
  };
  enum class CGameEditorMesh::ETexCoordLayer {
    Lightmap = 0,
  };
  enum class CGameEditorMesh::EMaterialFilterCriterion {
    IsAutomatic = 0,
    IsBadForHorizontalFaces = 1,
    IsBadForVerticalFaces = 2,
  };
  enum class CGameEditorMesh::EFilterKind {
    NoFilter = 0,
    PassIfMatches = 1,
    CutIfMatches = 2,
  };
  enum class CGameEditorMesh::EUVEditorMode {
    UV = 0,
    Atlas_ApplyOnClic = 1,
    Atlas_SelectOnClic = 2,
  };
  enum class CGameEditorMesh::EUVEditorProjectionType {
    Planar = 0,
    Curve2D = 1,
    Cylindrical2D = 2,
    Cubic = 3,
    Polyedric = 4,
    Cylindrical = 5,
    ApplyOnlyMaterial = 6,
  };
  enum class CGameEditorMesh::ELayerType {
    AddGeometry = 0,
    SubdivideSmooth = 1,
    Translation = 2,
    Rotation = 3,
    Scale = 4,
    Mirror = 5,
    MoveToGround = 6,
    Extrude = 7,
    Subdivide = 8,
    Chaos = 9,
    Smooth = 10,
    BorderTransition = 11,
    BlocTransfo = 12,
    Voxels = 13,
    TriggerShape = 14,
    RespawnPos = 15,
    Sector = 16,
    Light = 17,
    LightModel = 18,
    WaterShape = 19,
    None = 20,
  };
  enum class CGameEditorMesh::ETitleCoreType {
    TrackMania = 0,
    ShootMania = 1,
  };
  void Parts_Repair();
  void UVEditor_UVMode();
  void UVEditor_AtlasMode();
  bool GoToMaterialEditor; // Maniascript
  bool IsCreateMaterial; // Maniascript
  uint Layers_GetCount(); // Maniascript
  MwId Layers_GetLayerIdFromIndex(uint LayerIndex); // Maniascript
  wstring Layers_GetLayerNameFromIndex(uint LayerIndex); // Maniascript
  CGameEditorMesh::ELayerType Layers_GetLayerTypeFromIndex(uint LayerIndex); // Maniascript
  bool Layers_GetLayerGeneratableFromIndex(uint LayerIndex); // Maniascript
  void Layers_SetLayerGeneratableFromIndex(uint LayerIndex, bool LayerGeneratability); // Maniascript
  void Layers_AddLayer(CGameEditorMesh::ELayerType LayerType); // Maniascript
  MwId Layers_GetLayerIdSelected(); // Maniascript
  void Layers_SetLayerIdSelected(MwId LayerId); // Maniascript
  CGameEditorMesh::ELayerType Layers_GetLayerSelectedType(); // Maniascript
  uint Layers_GetLayerSelectedIndex(); // Maniascript
  CGameEditorMesh::ELayerType Layers_GetEditedLayerType(); // Maniascript
  wstring Layers_GetLayerSelectedName(); // Maniascript
  void Layers_EditMask(uint LayerIndex); // Maniascript
  void Layers_EditMaskValidate(uint LayerIndex); // Maniascript
  void Layers_MoveSelectedLayerUp(); // Maniascript
  void Layers_MoveSelectedLayerDown(); // Maniascript
  void Layers_EditSelectedLayer(bool RegenerateSolid); // Maniascript
  void Layers_CloseEditSelectedLayer(); // Maniascript
  void Layers_DeleteSelectedLayer(); // Maniascript
  uint Transitions_GetCount(); // Maniascript
  void Transitions_AddTransition(wstring TransitionName, int TransitionPosition, float TransitionSize); // Maniascript
  void Transitions_DeleteCurrentTransition(); // Maniascript
  int Transitions_GetTransitionSelectedIndex(); // Maniascript
  void Transitions_SetTransitionSelectedIndex(int TransitionIndex); // Maniascript
  uint Transitions_GetTransitionCurrentPage(); // Maniascript
  void Transitions_SetTransitionCurrentPage(uint TransitionCurrentPage); // Maniascript
  wstring Transitions_GetTransitionNameFromIndex(int TransitionIndex); // Maniascript
  void Transitions_SetTransitionNameFromIndex(int TransitionIndex, float TransitionName); // Maniascript
  bool Transitions_GetTransitionVisibilityFromIndex(int TransitionIndex); // Maniascript
  void Transitions_SetTransitionVisibilityFromIndex(int TransitionIndex, bool TransitionVisibility); // Maniascript
  float Transitions_GetTransitionSizeFromIndex(int TransitionIndex); // Maniascript
  void Transitions_SetTransitionSizeFromIndex(int TransitionIndex, float TransitionSize); // Maniascript
  int Transitions_GetTransitionPositionFromIndex(int TransitionIndex); // Maniascript
  void Transitions_SetTransitionPositionFromIndex(int TransitionIndex, int TransitionPosition); // Maniascript
  bool IsEditingLayer; // Maniascript
  uint SubdivideSmooth_NbSteps; // Maniascript
  float LayerValueAxisX; // Maniascript
  float LayerValueAxisY; // Maniascript
  float LayerValueAxisZ; // Maniascript
  bool LayerIndep; // Maniascript
  float LayerValueParam1; // Maniascript
  float LayerValueParam2; // Maniascript
  float LayerValueParam3; // Maniascript
  float LayerValueParam4; // Maniascript
  bool LayerIndexAxisX; // Maniascript
  bool LayerIndexAxisY; // Maniascript
  bool LayerIndexAxisZ; // Maniascript
  bool Layer_IsVisible; // Maniascript
  bool Layer_IsCollidable; // Maniascript
  wstring LayerName; // Maniascript
  bool LayerGeneratable; // Maniascript
  bool AddTransitionInProgress; // Maniascript
  wstring NewTransitionName; // Maniascript
  int NewTransitionPosition; // Maniascript
  float NewTransitionSize; // Maniascript
  uint RotateAxis; // Maniascript
  const bool Tmp_UseParts; // Maniascript
  const bool IsDebug; // Maniascript
  bool CameraEatingInputsScript; // Maniascript
  CControlFrame* const UIRoot;
  CMwNod* const EditedNod;
  CPlugMaterialUserInst* MatUserInstToEdit;
  const uint VertexCount; // Maniascript
  const uint EdgeCount; // Maniascript
  const uint FaceCount; // Maniascript
  void EditionBox_SetStep(float Step); // Maniascript
  const bool EditionBox_IsPlaneOriented; // Maniascript
  const float Scale; // Maniascript
  const float Step; // Maniascript
  const float Size; // Maniascript
  const int RotationStep; // Maniascript
  const float RotationValue; // Maniascript
  const float ScalingStep; // Maniascript
  const float ScalingRatio; // Maniascript
  bool DisplayVertices; // Maniascript
  bool DisplayFaces; // Maniascript
  bool DisplayJoints; // Maniascript
  CGameEditorMesh::EEdgesDisplay DisplayEdges; // Maniascript
  void EditedMesh_Clear(); // Maniascript
  void EditedMesh_Simplify(); // Maniascript
  void EditedMesh_SimplifySelection(); // Maniascript
  void EditedMesh_Lod(float FacesRatio); // Maniascript
  void UVUnwrap(MwId SetHandle, CGameEditorMesh::ETexCoordLayer ETexCoordLayer); // Maniascript
  void Undo(); // Maniascript
  void Redo(); // Maniascript
  void SwitchPlane(); // Maniascript
  void GridSnap_SetActive(bool IsActive); // Maniascript
  const bool GridSnap_IsActive; // Maniascript
  vec3 PickInfo_GetNormal(); // Maniascript
  vec3 PickInfo_GetPosition(); // Maniascript
  MwId PickInfo_GetAnchorId(); // Maniascript
  float PickInfo_GetEdgeLength(); // Maniascript
  vec3 PickInfo_GetNextVertexPosition(); // Maniascript
  MwId PickInfo_GetMaterial(); // Maniascript
  wstring PickInfo_GetError(); // Maniascript
  void Part_SetAnchorPos(vec3 Position); // Maniascript
  void Part_SetIsJoint(bool IsJoint); // Maniascript
  void Part_ClearAnchor(); // Maniascript
  void Joint_Add(vec3 Position); // Maniascript
  void Joint_Link(MwId IdChild, MwId IdParent); // Maniascript
  void Anchor_Remove(MwId Id); // Maniascript
  bool IsUsingPhysicMatLib; // Maniascript
  const int MaterialGameplayIdNumber; // Maniascript
  const int MaterialsUpdateId; // Maniascript
  const MwFastBuffer<CPlugBitmap*> AllBitmaps; // Maniascript
  const MwFastBuffer<MwId> MaterialIds; // Maniascript
  const MwSArray<MwId> MaterialPhysicsIds; // Maniascript
  const MwSArray<MwId> MaterialDynaIds; // Maniascript
  const MwSArray<wstring> MaterialNames; // Maniascript
  const MwSArray<wstring> MaterialPhysicsNames; // Maniascript
  const MwSArray<uint> MaterialPhysics_GameplayRemap; // Maniascript
  const MwSArray<wstring> MaterialDynaNames; // Maniascript
  MwFastArray<vec3> MaterialLastUsedColors; // Maniascript
  const MwFastArray<vec3> MaterialBaseColors; // Maniascript
  vec3 CurrentColorForSpecialMaterials; // Maniascript
  MwId Material_GetMaterialIdSelected(); // Maniascript
  void Material_SetMaterialIdSelected(MwId MaterialEditorId); // Maniascript
  MwId MaterialDyna_GetMaterialIdSelected(); // Maniascript
  void MaterialDyna_SetMaterialIdSelected(MwId DynaMaterialType); // Maniascript
  uint Material_GetSubTexIndexSelected(); // Maniascript
  uint Material_MaterialLibGetCount(); // Maniascript
  void Material_SetDefault(MwId MaterialId); // Maniascript
  MwId Material_GetDefault(); // Maniascript
  MwId MaterialPhysic_GetDefault(); // Maniascript
  uint MaterialPhysic_GetIndex(); // Maniascript
  uint MaterialPhysic_GetGameplayId(MwId MaterialId); // Maniascript
  CPlugBitmap* MaterialPhysic_GetNextBitmap(); // Maniascript
  MwId MaterialDyna_GetDefault(); // Maniascript
  void MaterialDyna_SetDefault(MwId MaterialId); // Maniascript
  CPlugBitmap* Material_GetBitmapBase(MwId MaterialId); // Maniascript
  CPlugBitmap* Material_GetBitmap(MwId MaterialId); // Maniascript
  CPlugBitmap* MaterialDyna_GetBitmap(MwId MaterialId); // Maniascript
  bool Material_IsSpecialColored(MwId MaterialId); // Maniascript
  bool Material_MatchesCriterion(MwId MaterialId, CGameEditorMesh::EMaterialFilterCriterion ResultSetHandle); // Maniascript
  void Material_SetFilter(CGameEditorMesh::EMaterialFilterCriterion Criterion, CGameEditorMesh::EFilterKind FilterKind); // Maniascript
  CGameEditorMesh::EFilterKind Material_GetFilter(CGameEditorMesh::EMaterialFilterCriterion Criterion); // Maniascript
  void Material_ClearFilters(); // Maniascript
  void Material_UVEditor_SetIsRotation(bool IsRotation); // Maniascript
  void Material_UVEditor_SetIsScale(bool IsScale); // Maniascript
  void Material_UVEditor_SetIsScale1D(bool IsScale); // Maniascript
  void Material_UVEditor_Open(MwId MaterialId, CGameManialinkQuad* LocationQuad); // Maniascript
  void Material_UVEditor_Close(); // Maniascript
  bool Material_UVEditor_IsRunning(); // Maniascript
  void Material_UVEditor_SetMode(CGameEditorMesh::EUVEditorMode Mode); // Maniascript
  CGameEditorMesh::EUVEditorMode Material_UVEditor_GetMode(); // Maniascript
  void Material_UVEditor_SetProjectionType(CGameEditorMesh::EUVEditorProjectionType ProjectionType); // Maniascript
  bool Material_IsGameMaterial(); // Maniascript
  bool Material_IsCustomLinkFull(MwId MaterialId); // Maniascript
  void Material_UVEditor_Apply(); // Maniascript
  void Material_CopyMaterial(MwId SetHandle); // Maniascript
  void Material_PasteMaterial(MwId SetHandle); // Maniascript
  const uint Material_Atlas_SelectedSubTexIndex; // Maniascript
  const CGameEditorMesh::EInteraction CurrentInteraction; // Maniascript
  void Interaction_Close(); // Maniascript
  void Interaction_SetPreview(MwId SetToPreview); // Maniascript
  const uint CreationElemsCount; // Maniascript
  bool Interaction_StartCreation(MwId CreationSetHandle, CGameEditorMesh::EElemType ElemType, MwId SetToPickFromHandle); // Maniascript
  void Interaction_Creation_GetElems(MwId ResultSetHandle); // Maniascript
  void Interaction_Creation_ClearParams(); // Maniascript
  void Interaction_Creation_SetEdgesConstraint(CGameEditorMesh::EEdgesConstraint EdgesConstraint); // Maniascript
  void Interaction_Creation_SetAutoMerge(bool AutoMerge); // Maniascript
  bool Interaction_StartPaste(); // Maniascript
  bool Interaction_StartBlocTransformation(MwId TransformationSetHandle); // Maniascript
  bool Interaction_StartCurve2D(MwId BordersSetHandle); // Maniascript
  void Interaction_CloseCurve2D(bool CanDoCurve2D); // Maniascript
  bool Interaction_StartPick(CGameEditorMesh::EElemType ElemType, MwId SetToPickFrom); // Maniascript
  bool Interaction_StartPickJoint(); // Maniascript
  bool Interaction_StartVoxelPickDrag_Base(CGameEditorMesh::EElemType ElemType); // Maniascript
  bool Interaction_StartVoxelPickDrag_Creation(); // Maniascript
  bool Interaction_StartVoxelPickDrag_Delete(); // Maniascript
  bool Interaction_StartVoxelPickDrag_Select(MwId SelectionSetHandle); // Maniascript
  void Interaction_CloseVoxelPickDrag_Select(); // Maniascript
  bool Interaction_StartVoxelPickDrag_Pick(); // Maniascript
  void Interaction_CloseVoxelPickDrag_Pick(); // Maniascript
  bool Interaction_StartVoxelPickDrag_SelectTranslation(); // Maniascript
  bool Interaction_StartVoxelPickDrag_SelectRotation(); // Maniascript
  bool Interaction_StartVoxelPickDrag_Paste(); // Maniascript
  void CutVoxels(); // Maniascript
  void CopyVoxels(); // Maniascript
  bool Interaction_StartMerge(MwId MergeSetHandle); // Maniascript
  bool Interaction_StartMirror(MwId SetHandle); // Maniascript
  void Interaction_Selection_ClearParams(); // Maniascript
  void Interaction_Selection_SetUseParts(bool UseParts); // Maniascript
  void Interaction_Selection_SetCanEnterLeaf(bool CanEnterLeaf); // Maniascript
  bool Interaction_StartSelection(MwId SelectionSetHandle, CGameEditorMesh::EElemType ElemType, MwId SelectionSetToPickFrom, bool IsFromALayer, bool AllowDoubleClick); // Maniascript
  void Interaction_CloseSelection(); // Maniascript
  bool Interaction_StartTranslation(MwId TranslationSetHandle); // Maniascript
  bool Interaction_StartPickTranslation(MwId TranslationSetHandle); // Maniascript
  bool Interaction_StartRotation(MwId RotationSetHandle); // Maniascript
  bool Interaction_StartPickRotation(MwId RotationSetHandle); // Maniascript
  void Interaction_Rotation_SetStep(float RotationStep); // Maniascript
  bool Interaction_StartPickScale(MwId ScalingSetHandle); // Maniascript
  void Interaction_Scale_SetStep(float ScalingStep); // Maniascript
  bool Interaction_StartSplit(); // Maniascript
  void Display_HighlightSet(MwId SetHandle); // Maniascript
  void Display_ClearHighlighting(); // Maniascript
  const bool Display_HideElemsByDistance_IsActive; // Maniascript
  uint Display_HideElemsByDistance_Distance; // Maniascript
  float Display_HideElemsByDistance_Opacity; // Maniascript
  void Display_HideElemsByDistance_Start(MwId SetHandle); // Maniascript
  void Display_HideElemsByDistance_Stop(); // Maniascript
  void Display_HideMap(); // Maniascript
  void Display_ShowMap(); // Maniascript
  void MergeAllSuperposedElements(MwId SetHandle); // Maniascript
  const MwId SelectionSet; // Maniascript
  void Selection_Undo(); // Maniascript
  void Selection_Redo(); // Maniascript
  void Selection_Invert(); // Maniascript
  void Selection_SelectAllByMaterial(); // Maniascript
  MwId SetOfElements_Create(); // Maniascript
  void SetOfElements_CopyFrom(MwId DestinationSet, MwId SourceSet); // Maniascript
  void SetOfElements_Append(MwId DestinationSet, MwId SourceSet); // Maniascript
  void SetOfElements_Destroy(MwId SetHandle); // Maniascript
  void SetOfElements_Empty(MwId SetHandle); // Maniascript
  void SetOfElements_SetAllElements(MwId SetHandle); // Maniascript
  void SetOfElements_SetAllFaces(MwId SetHandle); // Maniascript
  void SetOfElements_DeleteElements(MwId SetHandle); // Maniascript
  void SetOfElements_DeleteElements_NoSpread(MwId SetHandle, bool Spread); // Maniascript
  bool SetOfElements_HasHorizontalFaces(MwId SetHandle); // Maniascript
  bool SetOfElements_HasVerticalFaces(MwId SetHandle); // Maniascript
  uint SetOfElements_GetElemsCount(MwId SetHandle); // Maniascript
  uint SetOfElements_GetVerticesCount(MwId SetHandle); // Maniascript
  uint SetOfElements_GetEdgesCount(MwId SetHandle); // Maniascript
  uint SetOfElements_GetFacesCount(MwId SetHandle); // Maniascript
  void ExtendSelectedSet(MwId SetHandle); // Maniascript
  bool GetBordersSet(MwId SetHandle, MwId SetBordersHandle); // Maniascript
  void GetBordersVertexs(MwId SetHandle, MwId SetVertexHandle); // Maniascript
  void SelectionSet_SelectAll(); // Maniascript
  void Curve2DPolygon(MwId FourVertexSetHandle, MwId Sethandle, uint SubTexIndex); // Maniascript
  void Preview_Clear(); // Maniascript
  void VoxelSpace_SetVec3(vec3 Pos); // Maniascript
  uint VoxelSpace_GetVoxelsCount(); // Maniascript
  void VoxelSpace_SelectAll(); // Maniascript
  void VoxelSpace_DeleteOneVoxel(); // Maniascript
  void VoxelSpace_DeleteSelection(); // Maniascript
  void VoxelSpace_ApplyMaterialToVoxel(); // Maniascript
  void VoxelSpace_GenerateMesh(); // Maniascript
  vec3 VoxelSpaceCenter; // Maniascript
  vec3 VoxelSpaceAngle; // Maniascript
  float VoxelSpaceStep; // Maniascript
  bool VoxelSpaceIsInteractive; // Maniascript
  void SetOfElements_ProjectOnPlane(MwId SetHandle); // Maniascript
  void SetOfElements_ProjectOnGround(MwId SetHandle, float Height); // Maniascript
  void SetOfElements_SplitEdgeWithVertex(MwId SetHandle); // Maniascript
  void SetOfElements_CollapseEdgeWithVertex(MwId SetHandle); // Maniascript
  void SetOfElements_Subdivide(MwId SetHandle); // Maniascript
  void SetOfElements_Subdivide_Interpolation(MwId SetHandle); // Maniascript
  void SetOfVertices_DrawCircle(MwId InputSetHandle, MwId ResultSetHandle); // Maniascript
  void SetOfVertices_DrawDisc(MwId InputSetHandle, MwId ResultSetHandle); // Maniascript
  void SetOfVertices_DrawCircle2(MwId CenterSetHandle, vec3 PointOnCircle, MwId ResultSetHandle); // Maniascript
  void SetOfVertices_DrawIcosahedron(MwId InputSetHandle, MwId ResultSetHandle); // Maniascript
  void SetOfVertices_DrawIcosahedron2(MwId CenterSetHandle, vec3 PointOnCircle, MwId ResultSetHandle); // Maniascript
  void SetOfVertices_DrawIcosahedricSphere(MwId InputSetHandle, MwId ResultSetHandle); // Maniascript
  void SetOfVertices_DrawPoly(MwId InputSetHandle, MwId ResultSetHandle, int VerticesCount); // Maniascript
  void SetOfVertices_DrawPoly2(MwId CenterSetHandle, vec3 PointOnPoly, MwId ResultSetHandle, int VerticesCount); // Maniascript
  void SetOfVertices_DrawSpline(MwId ControlSetHandle, MwId ResultSetHandle); // Maniascript
  void SetOfVertices_Weld(MwId VerticesSetHandle); // Maniascript
  void SetOfVertices_DrawBox(MwId ControlSetHandle, MwId ResultSetHandle); // Maniascript
  void SetOfEdges_Fill(MwId SetHandle); // Maniascript
  void SetOfEdges_Flip(MwId SetHandle, MwId ResultSetHandle); // Maniascript
  void SetOfEdges_BorderExpand(MwId SetHandle); // Maniascript
  void SetOfOneEdge_FaceLoopExpand(MwId SetHandle); // Maniascript
  void SetOfOneEdge_EdgeLoopExpand(MwId SetHandle); // Maniascript
  void SetOfOneFace_CutHole(MwId FaceSetHandle, MwId EdgesSetHandle); // Maniascript
  void SetOfFaces_Extrude(MwId SetHandle, MwId ResultSetHandle); // Maniascript
  void SetOfFaces_QuadsToTriangles(MwId SetHandle, MwId ResultSetHandle); // Maniascript
  void SetOfFaces_TrianglesToQuads(MwId SetHandle, MwId ResultSetHandle); // Maniascript
  void SetOfFaces_ApplyMaterial(MwId SetHandle, MwId MaterialId); // Maniascript
  void SetOfFaces_PlanarExpand(MwId FacesSetHandle); // Maniascript
  void SetOfFaces_ChangeOrientation(MwId FacesSetHandle); // Maniascript
  const MwFastBuffer<wstring> PrefabNames; // Maniascript
  const int PrefabNamesUpdateId; // Maniascript
  uint PrefabListCurrentPage; // Maniascript
  uint Prefab_TotalNb; // Maniascript
  void Prefabs_Reload(); // Maniascript
  void Prefab_Export(); // Maniascript
  void Prefab_Import(uint PrefabIndex); // Maniascript
  bool Parts_CanMergeParts(); // Maniascript
  bool Parts_CanGroupParts(); // Maniascript
  bool Parts_CanUngroupParts(); // Maniascript
  int3 Parts_GetOpsState(); // Maniascript
  void Parts_MergeSelectedParts(); // Maniascript
  void Parts_Group(); // Maniascript
  void Parts_UngroupSelectedParts(); // Maniascript
  void Cut(); // Maniascript
  void Copy(); // Maniascript
  void AddUndoState(); // Maniascript
  bool AutoSave(wstring FileName); // Maniascript
  const MwFastBuffer<CGameEditorEvent*> PendingEvents; // Maniascript
  bool MustClearLastSaveBuffer; // Maniascript
  const bool IsExperimental; // Maniascript
  CGameEditorMesh::ETitleCoreType GetTitleCoreType(); // Maniascript
};

struct CGameEditorEvent : public CGameManiaAppScriptEvent {
  enum class CGameEditorEvent::EType {
    LayerCustomEvent = 0,
    KeyPress = 1,
    MenuNavigation = 3,
    FileChanged = 4,
    Exit = 5,
    FileSave = 6,
    OnUndo = 7,
    OnRedo = 8,
    OnSwitchedBack = 9,
    CameraButtonOn = 10,
    CameraButtonOff = 11,
    VoxelUpdateMaterial = 12,
    OpenUVEditor = 13,
    CloseUVEditor = 14,
    EnableUndo = 15,
    DisableUndo = 16,
    EnableRedo = 17,
    DisableRedo = 18,
    UpdateUI = 19,
    UpdateSliders = 20,
    UpdateMaterialsLibPage = 21,
    MediaTrackerPopUp = 22,
    HideUI = 23,
    ShowUI = 24,
    EnableFullScreen = 25,
    DisableFullScreen = 26,
    Autosave = 27,
  };
  const CGameEditorEvent::EType Type; // Maniascript
};

struct CGameDataFileTask_Skin_NadeoServices_GetAccountList : public CWebServicesTaskSequence {
};

struct CGameTurretPhy : public CMwNod {
};

struct CGameDataFileTask_Map_NadeoServices_GetAccountList : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_Map_NadeoServices_GetList : public CWebServicesTaskSequence {
};

// userName: 'Turret'
struct CGameTurretVis : public CMwNod {
};

struct CGameObjectPhy {
};

struct CGameEditorPluginHandle : public CMwNod {
};

struct CGameCtnMediaBlockTurret : public CGameCtnMediaBlock {
  CGameCtnMediaBlockTurret();

};

struct CGameEditorPluginLayerScriptHandler : public CGameManialinkScriptHandler {
  CGameEditorModule* const ModuleEditor; // Maniascript
  CGameEditorMesh* const MeshEditor; // Maniascript
  CGameEditorEditor* const EditorEditor; // Maniascript
  CGameEditorMediaTrackerPluginAPI* const MediaTracker; // Maniascript
  CGameEditorSkinPluginAPI* const SkinEditor; // Maniascript
  wstring Binding_GetShortcut(wstring ContextName, wstring BindingName); // Maniascript
};

struct CGameDataFileManager : public CMwNod {
};

struct CGameDataFileTask_GhostStoreUserRecord_Maniaplanet : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_GhostLoadUserRecord_Maniaplanet : public CWebServicesTaskSequence {
};

// Description: "A CDialogsManager event.
//               Work in progress, unavailable in public version."
struct CGameDialogsScriptEvent : public CMwNod {
  CGameDialogsScriptEvent();

  enum class CGameDialogsScriptEvent::EType {
    Close = 0,
  };
  const CGameDialogsScriptEvent::EType Type; // Maniascript
  enum class CGameDialogsScript::EType {
    None = 0,
    Message = 1,
    Choice = 2,
    FileBrowser = 3,
  };
  const CGameDialogsScript::EType DialogType; // Maniascript
  enum class CGameDialogsScript::EResult {
    Unknown = 0,
    Ok = 1,
    Cancel = 2,
    Yes = 3,
    No = 4,
  };
  const CGameDialogsScript::EResult Result; // Maniascript
  const wstring FullFileName; // Maniascript
  const wstring RelativeFileName; // Maniascript
  const wstring FileName; // Maniascript
};

// Description: "Task result containing a ghost."
struct CWebServicesTaskResult_GhostScript : public CWebServicesTaskResult_Ghost {
  CGameGhostScript* const Ghost; // Maniascript
};

struct CGameDataFileTask_GhostLoadMedal : public CWebServicesTaskSequence {
};

struct CGameCtnMediaBlockEntity : public CGameCtnMediaBlock {
  CGameCtnMediaBlockEntity();

};

// Description: "Results containing a list of map info."
struct CWebServicesTaskResult_MapListScript : public CWebServicesTaskResult_GameFidList {
  const wstring ParentPath; // Maniascript
  const wstring Path; // Maniascript
  const MwFastBuffer<CGameCtnChallengeInfo*> MapInfos; // Maniascript
  const MwFastBuffer<wstring> SubFolders; // Maniascript
};

struct CGameDataFileTask_GameFidGetGameList : public CWebServicesTaskSequence {
};

// Description: "Script API to manage game data."
struct CGameDataFileManagerScript : public CMwNod {
  enum class CGameDataFileManagerScript::EMediaType {
    Image = 0,
    Sound = 1,
    Script = 2,
    MatchSettings = 3,
    Module = 4,
    Skins = 5,
    ItemCollection = 6,
  };
  const MwFastBuffer<CWebServicesTaskResult*> TaskResults; // Maniascript
  void TaskResult_Release(MwId TaskId); // Maniascript
  void ReleaseTaskResult(MwId TaskId); // Maniascript
  const MwFastBuffer<CGameCtnCampaign*> Campaigns; // Maniascript
  CGameCtnCampaign* Campaign_Get(string CampaignId); // Maniascript
  CWebServicesTaskResult_NadeoServicesItemCollectionScript* ItemCollection_Create(MwId UserId, string ClubId, wstring DisplayName, wstring FileName); // Maniascript
  CWebServicesTaskResult_NadeoServicesItemCollectionScript* ItemCollection_CreateVersion(MwId UserId, string ItemCollectionId, wstring FileName); // Maniascript
  CWebServicesTaskResult_NadeoServicesItemCollectionScript* ItemCollection_Get(MwId UserId, string ItemCollectionId); // Maniascript
  CWebServicesTaskResult_NadeoServicesItemCollectionListScript* ItemCollection_GetList(MwId UserId, MwFastBuffer<wstring>& ItemCollectionIdList); // Maniascript
  CWebServicesTaskResult_NadeoServicesItemCollectionListScript* ItemCollection_GetListByUser(MwId UserId, string WebServicesUserId); // Maniascript
  CWebServicesTaskResult_NadeoServicesItemCollectionScript* ItemCollection_SetActivityId(MwId UserId, string ItemCollectionId, string ActivityId); // Maniascript
  CWebServicesTaskResult_NadeoServicesItemCollectionScript* ItemCollection_Update(MwId UserId, string ItemCollectionId, wstring DisplayName); // Maniascript
  CWebServicesTaskResult* ItemCollection_AddFavorite(MwId UserId, string ItemCollectionId); // Maniascript
  CWebServicesTaskResult_NadeoServicesItemCollectionListScript* ItemCollection_GetFavoriteList(MwId UserId); // Maniascript
  CWebServicesTaskResult* ItemCollection_RemoveFavorite(MwId UserId, string ItemCollectionId); // Maniascript
  void Map_RefreshFromDisk(); // Maniascript
  CWebServicesTaskResult_MapListScript* Map_GetGameList(wstring Path, bool Flatten); // Maniascript
  CWebServicesTaskResult_MapListScript* Map_GetGameListWithFilters(wstring Path, bool Flatten, bool SortByNameElseByDate, bool SortOrderAsc); // Maniascript
  CWebServicesTaskResult_MapListScript* Map_GetFilteredGameList(uint Scope, wstring Path, bool Flatten, bool SortByNameElseByDate, bool SortOrderAsc); // Maniascript
  CGameGhostScript* Map_GetAuthorGhost(CGameCtnChallenge* Map); // Maniascript
  CWebServicesTaskResult_NadeoServicesMapScript* Map_NadeoServices_Get(MwId UserId, string MapId); // Maniascript
  CWebServicesTaskResult_NadeoServicesMapScript* Map_NadeoServices_GetFromUid(MwId UserId, string MapUid); // Maniascript
  CWebServicesTaskResult_NadeoServicesMapListScript* Map_NadeoServices_GetList(MwId UserId, MwFastBuffer<wstring>& MapIdList); // Maniascript
  CWebServicesTaskResult_NadeoServicesMapListScript* Map_NadeoServices_GetListFromUid(MwId UserId, MwFastBuffer<wstring>& MapUidList); // Maniascript
  CWebServicesTaskResult_NadeoServicesMapListScript* Map_NadeoServices_GetListFromUser(MwId UserId, string WebServicesUserId); // Maniascript
  int Map_NadeoServices_GetVote(MwId UserId, string MapUid); // Maniascript
  CWebServicesTaskResult_NadeoServicesMapScript* Map_NadeoServices_Register(MwId UserId, string MapUid); // Maniascript
  void Map_NadeoServices_Vote(MwId UserId, string MapUid, int Vote); // Maniascript
  CWebServicesTaskResult* Map_NadeoServices_AddFavorite(MwId UserId, string MapUid); // Maniascript
  CWebServicesTaskResult_NadeoServicesMapListScript* Map_NadeoServices_GetFavoriteList(MwId UserId, MwFastBuffer<wstring>& MapTypeList, bool SortByDateElseByName, bool SortOrderAsc, bool OnlyPlayable, bool OnlyMine); // Maniascript
  CWebServicesTaskResult_NadeoServicesMapListScript* Map_NadeoServices_GetFavoriteListByUid(MwId UserId, MwFastBuffer<wstring>& MapUidList); // Maniascript
  CWebServicesTaskResult* Map_NadeoServices_RemoveFavorite(MwId UserId, string MapUid); // Maniascript
  CWebServicesTaskResult_NadeoServicesSkinScript* Skin_NadeoServices_Get(MwId UserId, string SkinId); // Maniascript
  CWebServicesTaskResult_NadeoServicesSkinScript* Skin_NadeoServices_GetFromChecksum(MwId UserId, string SkinChecksum); // Maniascript
  CWebServicesTaskResult_NadeoServicesSkinListScript* Skin_NadeoServices_GetList(MwId UserId, MwFastBuffer<wstring>& SkinIdList); // Maniascript
  CWebServicesTaskResult_NadeoServicesSkinListScript* Skin_NadeoServices_GetListFromChecksum(MwId UserId, MwFastBuffer<wstring>& SkinChecksumList); // Maniascript
  CWebServicesTaskResult_NadeoServicesSkinListScript* Skin_NadeoServices_GetListFromUser(MwId UserId, string WebServicesUserId); // Maniascript
  CWebServicesTaskResult_NadeoServicesSkinScript* Skin_NadeoServices_Register(MwId UserId, wstring SkinDisplayName, wstring SkinFileName); // Maniascript
  CWebServicesTaskResult* AccountSkin_NadeoServices_AddFavorite(MwId UserId, string SkinId); // Maniascript
  CWebServicesTaskResult_NadeoServicesSkinListScript* AccountSkin_NadeoServices_GetFavoriteList(MwId UserId); // Maniascript
  CWebServicesTaskResult_NadeoServicesSkinListScript* AccountSkin_NadeoServices_GetList(MwId UserId); // Maniascript
  CWebServicesTaskResult_NadeoServicesSkinListScript* AccountSkin_NadeoServices_GetListForUser(MwId UserId, string WebServicesUserId); // Maniascript
  CWebServicesTaskResult* AccountSkin_NadeoServices_RemoveFavorite(MwId UserId, string SkinId); // Maniascript
  CWebServicesTaskResult_NadeoServicesSkinScript* AccountSkin_NadeoServices_Set(MwId UserId, string SkinId); // Maniascript
  CWebServicesTaskResult* AccountSkin_NadeoServices_Unset(MwId UserId, string SkinType); // Maniascript
  const MwFastBuffer<CGameGhostScript*> Ghosts; // Maniascript
  void Ghost_Release(MwId GhostId); // Maniascript
  CWebServicesTaskResult_GhostScript* Ghost_Download(wstring FileName, string Url); // Maniascript
  CWebServicesTaskResult* Ghost_Upload(string Url, CGameGhostScript* Ghost, string AdditionalHeaders); // Maniascript
  void Replay_RefreshFromDisk(); // Maniascript
  CWebServicesTaskResult_ReplayListScript* Replay_GetGameList(wstring Path, bool Flatten); // Maniascript
  CWebServicesTaskResult_ReplayListScript* Replay_GetGameListWithFilters(wstring Path, bool Flatten, bool SortByNameElseByDate, bool SortOrderAsc); // Maniascript
  CWebServicesTaskResult_ReplayListScript* Replay_GetFilteredGameList(uint Scope, wstring Path, bool Flatten, bool SortByNameElseByDate, bool SortOrderAsc); // Maniascript
  CWebServicesTaskResult_GhostListScript* Replay_Load(wstring Path); // Maniascript
  CWebServicesTaskResult* Replay_Save(wstring Path, CGameCtnChallenge* Map, CGameGhostScript* Ghost); // Maniascript
  CWebServicesTaskResult* Replay_Author_Save(CGameCtnChallenge* Map, CGameGhostScript* Ghost); // Maniascript
  CWebServicesTaskResult_FileListScript* Media_GetGameList(CGameDataFileManagerScript::EMediaType Type, wstring Path, bool Flatten); // Maniascript
  CWebServicesTaskResult_FileListScript* Media_GetGameListWithFilters(CGameDataFileManagerScript::EMediaType Type, wstring Path, bool Flatten, bool SortByNameElseByDate, bool SortOrderAsc); // Maniascript
  CWebServicesTaskResult_FileListScript* Media_GetFilteredGameList(CGameDataFileManagerScript::EMediaType Type, uint Scope, wstring Path, bool Flatten, bool SortByNameElseByDate, bool SortOrderAsc); // Maniascript
  void Media_RefreshFromDisk(CGameDataFileManagerScript::EMediaType Type, uint Scope); // Maniascript
  CWebServicesTaskResult_GameModeListScript* GameMode_GetGameList(uint Scope, wstring Path, bool Flatten); // Maniascript
  CWebServicesTaskResult* Pack_DownloadOrUpdate(wstring DisplayName, string Url); // Maniascript
  CWebServicesTaskResult* UserSave_DeleteFile(wstring Path); // Maniascript
  const uint AvailableSaveSpace; // Maniascript
  const uint UserSave_AvailableSpace; // Maniascript
  const bool UserSave_IsWriting; // Maniascript
};

struct CWebServicesTaskResult_GameFidList : public CWebServicesTaskResult {
};

// Description: "Results containing a list of replay info."
struct CWebServicesTaskResult_ReplayListScript : public CWebServicesTaskResult_GameFidList {
  const wstring ParentPath; // Maniascript
  const wstring Path; // Maniascript
  const MwFastBuffer<CGameCtnReplayRecordInfo*> ReplayInfos; // Maniascript
  const MwFastBuffer<wstring> SubFolders; // Maniascript
};

struct CWebServicesTaskResult_FileList : public CWebServicesTaskResult {
};

// Description: "Results containing a list of media info."
struct CWebServicesTaskResult_FileListScript : public CWebServicesTaskResult_FileList {
  const wstring ParentPath; // Maniascript
  const wstring Path; // Maniascript
  const MwFastBuffer<wstring> Files; // Maniascript
  const MwFastBuffer<wstring> SubFolders; // Maniascript
};

struct CGameDataFileTask_FileGetGameList : public CWebServicesTaskSequence {
};

// userName: 'Object'
struct CGameObjectVis : public CMwNod {
};

struct CGameMapScoreManager : public CMwNod {
};

struct CGameMapScoreManager_MapRecord : public CGameMapScoreManager {
};

struct CGameMapScoreManager_MultiAsyncLevel : public CGameMapScoreManager {
};

struct CGameScriptMgrTurret : public CMwNod {
  void MapTurrets_Reset(); // Maniascript
  CGameScriptTurret* Turret_Create(MwId ModelId, vec3 Position, vec3 Direction, uint Clan, CGameScriptPlayer* OwnerPlayer); // Maniascript
  CGameScriptTurret* Turret_CreateWithOwner(MwId ModelId, vec3 Position, vec3 Direction, vec3 Up, uint Clan, MwId OwnerId); // Maniascript
  CGameScriptTurret* Turret_CreateWithOwner2(MwId ModelId, vec3 Position, vec3 Direction, vec3 Up, uint Clan, MwId OwnerId, bool AutoAimOn, bool AutoTriggerOn); // Maniascript
  vec3 Turret_GetPosition(CGameScriptTurret* Turret); // Maniascript
  void Turret_Destroy(CGameScriptTurret* Turret); // Maniascript
  void Turret_DestroyAll(); // Maniascript
  void Turret_Activate(CGameScriptTurret* Turret); // Maniascript
  void Turret_Deactivate(CGameScriptTurret* Turret); // Maniascript
  void Turret_SetIsAutomatic(CGameScriptTurret* Turret, bool IsAuto); // Maniascript
  void Turret_Auto_SetAimAnticipation(CGameScriptTurret* Turret, float AimAnticipationMs); // Maniascript
  void Turret_Auto_SetFirePeriod(CGameScriptTurret* Turret, uint FirePeriodMs); // Maniascript
  void Turret_Auto_SetTargetDetectionFov(CGameScriptTurret* Turret, float DectectionFOVDeg); // Maniascript
  void Turret_Auto_SetTargetDetectionRadius(CGameScriptTurret* Turret, float DetectionRadius); // Maniascript
  void Turret_Manual_SetTargetPos(CGameScriptTurret* Turret, vec3 TargetPos); // Maniascript
  void Turret_Manual_Fire(CGameScriptTurret* Turret); // Maniascript
  MwFastBuffer<CGameScriptTurret*> Turrets; // Maniascript
};

struct CGameScriptTurret : public CGameScriptEntity {
  uint Armor; // Maniascript
  uint ArmorMax; // Maniascript
  CGameScriptPlayer* const Owner; // Maniascript
};

struct CWebServicesTaskResult_GhostList : public CWebServicesTaskResult {
};

// Description: "Results containing a list of ghost."
struct CWebServicesTaskResult_GhostListScript : public CWebServicesTaskResult_GhostList {
  const MwFastBuffer<CGameGhostScript*> Ghosts; // Maniascript
};

struct CGameMasterServerTask_SetTitlePaid : public CNetMasterServerRequestTask {
};

// Description: "Allows handling of match-settings files"
struct CGameMatchSettingsManagerScript : public CMwNod {
  void Debug_MatchSettings_NewFile();
  void Debug_MatchSettings_NewTemp();
  void Debug_MatchSettings_Save();
  void Debug_MatchSettings_SaveAs();
  void Debug_MatchSettings_EditScriptSettings();
  void Debug_MatchSettings_Remove();
  void MatchSettings_RefreshFiles(); // Maniascript
  CGameMatchSettingsScript* MatchSettings_FindFile(wstring FilePath); // Maniascript
  CGameMatchSettingsScript* MatchSettings_CreateFile(wstring FilePath); // Maniascript
  CGameMatchSettingsScript* MatchSettings_CreateTemp(); // Maniascript
  void MatchSettings_Save(CGameMatchSettingsScript* MatchSettings); // Maniascript
  void MatchSettings_ReloadFromFile(CGameMatchSettingsScript* MatchSettings); // Maniascript
  CGameMatchSettingsScript* MatchSettings_SaveAs(wstring FilePath, CGameMatchSettingsScript* MatchSettings); // Maniascript
  void MatchSettings_EditScriptSettings(CGameMatchSettingsScript* MatchSettings); // Maniascript
  const bool MatchSettings_EditScriptSettings_Ongoing; // Maniascript
  void MatchSettings_Remove(CGameMatchSettingsScript* MatchSettings); // Maniascript
  const MwFastBuffer<CGameMatchSettingsScript*> MatchSettings; // Maniascript
  const MwFastBuffer<CGameMatchSettingsScript*> MatchSettings_File; // Maniascript
  const MwFastBuffer<CGameMatchSettingsScript*> MatchSettings_Temp; // Maniascript
  void MatchSettings_Refresh(); // Maniascript
  CGameMatchSettingsScript* MatchSettings_Create(wstring FilePath); // Maniascript
  CGameMatchSettingsScript* MatchSettings_CreateMem(); // Maniascript
};

// Description: "Represents a match-settings file"
struct CGameMatchSettingsScript : public CMwNod {
  const wstring Name; // Maniascript
  const wstring FileName; // Maniascript
  wstring ScriptModeName; // Maniascript
  bool ScriptModeName_Check(wstring ScriptModeName); // Maniascript
  void ScriptModeName_Set(wstring ScriptModeName); // Maniascript
  void ScriptSettings_SetToDefault(); // Maniascript
  const MwFastBuffer<CGameMatchSettingsPlaylistItemScript*> Playlist; // Maniascript
  bool Playlist_FileExists(wstring File); // Maniascript
  bool Playlist_FileMatchesMode(wstring File); // Maniascript
  void Playlist_Add(wstring File); // Maniascript
  void Playlist_Remove(uint Index); // Maniascript
  void Playlist_SwapUp(uint Index); // Maniascript
  void Playlist_SwapDown(uint Index); // Maniascript
};

struct CGameDataFileTask_PackDownloadOrUpdate : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_Title : public CWebServicesTaskResult {
};

// Description: "Represents a map in the playlist of a matchsetting"
struct CGameMatchSettingsPlaylistItemScript : public CMwNod {
  const wstring Name; // Maniascript
  const bool FileExists; // Maniascript
};

struct CGameScoreTask_GetPlayerPersonalBestMapRecordList : public CWebServicesTaskSequence {
};

// Description: "Task result containing a list of map record info from NadeoServices."
struct CWebServicesTaskResult_MapRecordListScript : public CWebServicesTaskResult_WSMapRecordList {
  const MwFastBuffer<CMapRecord*> MapRecordList; // Maniascript
};

struct CGameScoreTask_SetTrophyCompetitionMatchAchievementResults : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_GameModeGetGameList : public CWebServicesTaskSequence {
};

// Description: "Results containing a list of GameMode info."
struct CWebServicesTaskResult_GameModeListScript : public CWebServicesTaskResult {
  const MwFastBuffer<CGameGameModeInfoScript*> GameModes; // Maniascript
};

// Description: "A GameMode script."
struct CGameGameModeInfoScript : public CMwNod {
  const wstring Name; // Maniascript
  const wstring Path; // Maniascript
  const wstring Description; // Maniascript
  const wstring Version; // Maniascript
  const MwFastBuffer<wstring> CompatibleMapTypes; // Maniascript
};

// Description: "Local profile settings."
struct CGameUserProfileWrapper : public CMwNod {
  enum class CGameUserProfileWrapper::EMapEditorMode {
    Ask = 0,
    Advanced = 1,
  };
  enum class CGameUserProfileWrapper::EMapEditorMood {
    Sunrise = 0,
    Day = 1,
    Sunset = 2,
    Night = 3,
  };
  enum class CGameUserProfileWrapper::EMapEditorDifficulty {
    Simple = 0,
    Advanced = 1,
    Expert = 2,
  };
  enum class CGameUserProfileWrapper::ECustomPlayerModels {
    All = 0,
    OnlyTextures = 1,
    None = 2,
  };
  enum class CGameUserProfileWrapper::EInputMouseReleaseKey {
    LeftAlt = 0,
    RightAlt = 1,
    LeftCtrl = 2,
    RightCtrl = 3,
  };
  enum class CGameUserProfileWrapper::EPlayerVisibility {
    Hidden = 0,
    Ghost = 1,
    Opaque = 2,
  };
  enum class CGameUserProfileWrapper::EDisplayRecords {
    Progressive = 0,
    Always = 1,
    Hide = 2,
  };
  enum class CGameUserProfileWrapper::ERoadsideSpectatorVisibility {
    Never = 0,
    SpectatorOnly = 1,
    Always = 2,
  };
  enum class CGameUserProfileWrapper::EIngameChatBackground {
    Hidden = 0,
    Transparent = 1,
    Opaque = 2,
  };
  enum class CGameUserProfileWrapper::EIngameChatTextSize {
    Medium = 0,
    Small = 1,
    Big = 2,
  };
  CGamePlayerProfile* ProfileOld;
  CGameUserProfile* ProfileNew;
  const bool CanChangePassword; // Maniascript
  const bool CanChangeNickName; // Maniascript
  const bool CanChangeAvatar; // Maniascript
  const bool CanChangeSkin; // Maniascript
  const bool CanChangeZone; // Maniascript
  const bool CanChangeGroups; // Maniascript
  void Account_ResetChanges(); // Maniascript
  wstring Account_Name; // Maniascript
  string Account_EMail; // Maniascript
  bool Account_AcceptNews; // Maniascript
  bool Account_EnableAutoConnect; // Maniascript
  float User_LightTrailHue; // Range: 0 - 1 // Maniascript
  float User_HornPitch; // Range: 0.5 - 2 // Maniascript
  wstring User_Description; // Maniascript
  string User_ClubLinkUrl; // Maniascript
  string User_Trigram; // Maniascript
  bool User_ForceEmptyPilotSkin; // Maniascript
  string User_CharacterSkinOptions; // Maniascript
  bool User_CombinePrestigeAndSkins; // Maniascript
  bool Custom_EnableAvatars; // Maniascript
  bool Custom_EnableChat; // Maniascript
  CGameUserProfileWrapper::ECustomPlayerModels Custom_PlayerModels; // Maniascript
  bool Custom_EnableUnlimitedHorns; // Maniascript
  CGameUserProfileWrapper::EMapEditorMode Editors_MapEditorMode; // Maniascript
  CGameUserProfileWrapper::EMapEditorMode MapEditorMode; // Maniascript
  bool Editors_MapEditorUseQuickstart; // Maniascript
  bool Editors_MapEditorQuickstartUseGamepad; // Maniascript
  bool Editors_MapEditorQuickstartIsAdvanced; // Maniascript
  CGameUserProfileWrapper::EMapEditorDifficulty Editors_MapEditorQuickstartDifficulty; // Maniascript
  CGameUserProfileWrapper::EMapEditorMood Editors_MapEditorQuickstartMood; // Maniascript
  int Editors_MapEditorQuickstartMapType; // Maniascript
  bool Online_AutoSaveReplay; // Maniascript
  bool Online_SaveRoundReplaysSeparately; // Maniascript
  CGameUserProfileWrapper::EPlayerVisibility Online_DefaultOpponentVisibility; // Maniascript
  CGameUserProfileWrapper::ERoadsideSpectatorVisibility Online_RoadsideSpectatorVisibility; // Maniascript
  bool Title_IsForbiddenWithParentalLock; // Maniascript
  uint AddictionLimiter_DailyQuotaMinutes; // Maniascript
  bool Steam_OpenLinksInSteam; // Maniascript
  bool Steam_SynchonizeWorkshopFiles; // Maniascript
  bool Interface_CrosshairEnableCustomColor; // Maniascript
  float Interface_CrosshairSaturation; // Range: 0 - 1 // Maniascript
  float Interface_CrosshairLinearHue; // Range: 0 - 1 // Maniascript
  float Interface_CrosshairSize; // Range: 0.1 - 2 // Maniascript
  float Interface_CrosshairOpacity; // Range: 0.1 - 1 // Maniascript
  wstring Interface_CrosshairName; // Maniascript
  float Interface_PlayerShieldScale; // Range: 0.1 - 2 // Maniascript
  bool Interface_AllyEnableCustomColor; // Maniascript
  float Interface_AllyLinearHue; // Range: 0 - 1 // Maniascript
  bool Interface_OppoEnableCustomColor; // Maniascript
  float Interface_OppoLinearHue; // Range: 0 - 1 // Maniascript
  bool Interface_BeaconEnableCustom; // Maniascript
  bool Interface_BeaconUseProfileColor; // Maniascript
  float Interface_BeaconOpacity; // Range: 0 - 1 // Maniascript
  float Interface_BeaconSize; // Range: 0.1 - 5 // Maniascript
  float Interface_BeaconDuration; // Range: 0.5 - 15 // Maniascript
  CGameUserProfileWrapper::EPlayerVisibility Interface_InternalCamLocalPlayerVisibility; // Maniascript
  bool Interface_UseOldInternalCam; // Maniascript
  bool Interface_UseAlternateCam1; // Maniascript
  bool Interface_UseAlternateCam2; // Maniascript
  bool Interface_ShowSpecialsFeedback; // Maniascript
  bool Interface_AlwaysDisplayRecords; // Maniascript
  CGameUserProfileWrapper::EDisplayRecords Interface_DisplayRecords; // Maniascript
  bool Interface_AllowChatHiding; // Maniascript
  bool Interface_ColorblindMode; // Maniascript
  CGameUserProfileWrapper::EIngameChatBackground Interface_IngameChatBackground; // Maniascript
  CGameUserProfileWrapper::EIngameChatTextSize Interface_IngameChatTextSize; // Maniascript
  bool Inputs_MouseLookInvertY; // Maniascript
  CGameUserProfileWrapper::EInputMouseReleaseKey Inputs_MouseReleaseKey; // Maniascript
  float Inputs_MouseScaleY; // Range: 0.1 - 3 // Maniascript
  float Inputs_MouseScaleFreeLook; // Range: 0.1 - 3 // Maniascript
  float Inputs_MouseAccel; // Range: 0 - 3 // Maniascript
  float Inputs_MouseSensitivityDefault; // Range: -1 - 1 // Maniascript
  float Inputs_MouseSensitivityLaser; // Range: -1 - 1 // Maniascript
  float Inputs_MouseSensitivityDefault_Raw; // Maniascript
  float Inputs_MouseSensitivityLaser_Raw; // Maniascript
  MwFastBuffer<CGameUserProfileWrapper_VehicleSettings*> Inputs_Vehicles; // Maniascript
  bool Adverts_Enabled; // Maniascript
  bool Adverts_UsePersonnalData; // Maniascript
  bool TTS_Enabled; // Maniascript
  bool STT_Enabled; // Maniascript
  bool VoiceChat_Loopback; // Maniascript
  bool VoiceChat_Enabled; // Maniascript
  const bool STT_Available; // Maniascript
  bool VoiceChat_SendTextAsVoice; // Maniascript
  bool VoiceChat_PTT_Enabled; // Maniascript
  uint VoiceChat_SpeakerVolume; // Maniascript
};

struct CGameEditorPluginAPI : public CMwNod {
  enum class CGameEditorPluginAPI::EEditorFileToolBar_QuadType {
    Quit = 0,
    New = 1,
    Open = 2,
    SaveAs = 3,
    Save = 4,
    Import = 5,
    Export = 6,
    Close = 7,
    Help = 8,
  };
  void NewFile(); // Maniascript
  void Undo(); // Maniascript
  void Redo(); // Maniascript
  void FileToolBarSendCmd(wstring CmdName); // Maniascript
  bool SetToolBarButtonVisible(CGameEditorPluginAPI::EEditorFileToolBar_QuadType Type); // Maniascript
};

struct CGameBlockItemVariantChooser : public CMwNod {
  CGameItemModel* const ItemModelToView;
  CGameCtnBlockInfo* const ArchetypeBlockInfo;
  bool ShowOriginalVariant;
  wstring VariantType;
  uint VariantGroupCurrent;
  const uint VariantGroupMax;
  uint RandomVariantCurrent;
  const uint RandomVariantMax;
  void NextVariantType();
  void PreviousVariantType();
  void NextVariantGroup();
  void PreviousVariantGroup();
  void NextRandomVariant();
  void PreviousRandomVariant();
};

struct CGameArenaPlayer : public CMwNod {
};

struct CGameEditorMaterial : public CGameEditorParent {
};

struct CWebServicesTaskResult_UserNewsListScript : public CWebServicesTaskResult_WSNewsList {
  const MwFastBuffer<CNews*> NewsList; // Maniascript
};

struct CGameMgrAction : public CMwNod {
  enum class CGameMgrAction::EActionSlot {
    Slot_A = 0,
    Slot_B = 1,
    Slot_C = 2,
    Slot_D = 3,
    Slot_E = 4,
    Slot_F = 5,
    Slot_G = 6,
    Slot_H = 7,
  };
  CGameScriptAction* Action_GetState_Player(CGameScriptPlayer* Player, CGameMgrAction::EActionSlot Slot); // Maniascript
  CGameScriptAction* Action_GetState_Vehicle(CGameScriptVehicle* Vehicle, int VehicleSlotIndex, CGameMgrAction::EActionSlot Slot); // Maniascript
  CGameScriptAction* Action_GetState_Turret(CGameScriptTurret* Turret, CGameMgrAction::EActionSlot Slot); // Maniascript
  MwFastBuffer<CGameScriptAction*> Actions; // Maniascript
};

struct CGameScriptVehicle : public CGameScriptEntity {
  enum class CGameScriptVehicle::ESlotType {
    Driver = 0,
    Passenger = 1,
  };
  uint Armor; // Maniascript
  uint ArmorMax; // Maniascript
  const vec3 Position; // Maniascript
  const float Pitch; // Maniascript
  const float Roll; // Maniascript
  const vec3 Left; // Maniascript
  const vec3 Up; // Maniascript
  const vec3 Dir; // Maniascript
  const vec3 Velocity; // Maniascript
  uint Clan; // Maniascript
  const float AccelCoef; // Maniascript
  const float AccelInput; // Maniascript
  void SetEngineActivated(bool IsActivated); // Maniascript
};

struct CGameScriptMgrVehicle : public CMwNod {
  enum class CGameScriptMgrVehicle::EArmorUse {
    Self = 0,
    Group = 1,
    Owner = 2,
    Children = 3,
    Mine = 4,
  };
  enum class CGameScriptMgrVehicle::ESlotType {
    Driver = 0,
    Passenger = 1,
  };
  void MapVehicles_Reset(); // Maniascript
  void DestroyAllVehicles(); // Maniascript
  CGameScriptVehicle* Vehicle_Create(MwId ModelId, int Clan, int Armor, CGameScriptMgrVehicle::EArmorUse ArmorUse, CGameScriptMapSpawn* Spawn); // Maniascript
  CGameScriptVehicle* Vehicle_CreateWithOwner(MwId ModelId, vec3 Offset, int Armor, CGameScriptMgrVehicle::EArmorUse ArmorUse, CGameScriptVehicle* Owner); // Maniascript
  void Vehicle_Destroy(CGameScriptVehicle* Vehicle); // Maniascript
  void Vehicle_Assign_AutoPilot(CGameScriptVehicle* Vehicle, string ModelName); // Maniascript
  void Vehicle_Assign_AutoPilot2(CGameScriptVehicle* Vehicle, string ModelName, bool LoopPath); // Maniascript
  void Vehicle_SetTrailVisible(CGameScriptVehicle* Vehicle, bool IsVisible); // Maniascript
  void Vehicle_SetShieldVisible(CGameScriptVehicle* Vehicle, bool IsVisible); // Maniascript
  uint VehicleModel_GetSlotsCount(MwId ModelId, CGameScriptMgrVehicle::ESlotType SlotType); // Maniascript
  MwFastBuffer<CGameScriptVehicle*> Vehicles; // Maniascript
};

struct CGameAction : public CMwNod {
};

struct CGameScriptAction : public CMwNod {
  const uint Cooldown; // Maniascript
  const uint CooldownStartTime; // Maniascript
  const uint Energy; // Maniascript
  const uint EnergyCost; // Maniascript
  const uint EnergyMax; // Maniascript
};

struct CGameDataFileTask_Skin_NadeoServices_Set : public CWebServicesTaskSequence {
};

// Description: ""
struct CGameScriptMapLandmark : public CMwNod {
  const MwId MarkerId; // Maniascript
  const string Tag; // Maniascript
  const uint Order; // Maniascript
  const vec3 Position; // Maniascript
  CGameScriptMapSector* const Sector; // Maniascript
  CGameScriptMapWaypoint* const Waypoint; // Maniascript
  CGameScriptMapSpawn* const PlayerSpawn; // Maniascript
  CGameScriptMapBotPath* const BotPath; // Maniascript
  CGameScriptMapBotSpawn* const BotSpawn; // Maniascript
  CGameScriptMapObjectAnchor* const ObjectAnchor; // Maniascript
};

struct CGameScriptMapSector : public CMwNod {
  const MwFastBuffer<MwId> PlayersIds; // Maniascript
  const string Tag; // Maniascript
};

struct CGameScriptMapSpawn : public CMwNod {
};

struct CGameScriptMapBotPath : public CMwNod {
  const uint Clan; // Maniascript
  const MwFastBuffer<vec3> Path; // Maniascript
  const bool IsFlying; // Maniascript
  const MwId BotModelId; // Maniascript
};

struct CGameScriptMapObjectAnchor : public CMwNod {
  const wstring ItemName; // Maniascript
  const MwId ItemModelId; // Maniascript
};

struct CGameScriptMapBotSpawn : public CMwNod {
  const bool IsFlying; // Maniascript
  const MwId BotModelId; // Maniascript
};

struct CWebServicesTask_Title_GetConfig : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Title_GetPolicyRuleValues : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Title_GetLadderInfo : public CWebServicesTaskSequence {
};

// userName: 'Slot'
struct CGameSlotPhy : public CMwNod {
};

// userName: 'CGameVehiclePhy'
// File extension: 'Vehicle'
struct CGameVehiclePhy : public _0x0A020000 {
};

// userName: 'Slot'
struct CGameSlotVis : public CMwNod {
};

struct CGameGateVis {
};

// Description: ""
struct CGameScriptEntity : public CMwNod {
  const MwId MarkerId; // Maniascript
};

// Description: "A player is the incarnation of the user in the playground."
struct CGameScriptPlayer : public CGameScriptEntity {
  CGamePlayerInfo* const User; // Maniascript
  const string Login; // Maniascript
  const wstring Name; // Maniascript
  const int RequestedClan; // Maniascript
  const bool RequestsSpectate; // Maniascript
  int LandmarkOrderSelector_Race; // Maniascript
};

// Description: "An invitation to join a squad."
struct CGameScriptChatSquadInvitation : public CMwNod {
  CGameScriptChatContact* const Sender; // Maniascript
  const string SquadId;
};

// Description: "An message in the history."
struct CGameScriptChatHistoryEntryMessage : public CMwNod {
  const wstring Body; // Maniascript
  const wstring Link; // Maniascript
  const wstring Metadata; // Maniascript
};

struct CGameZoneTask_UpdateZoneList : public CWebServicesTaskSequence {
};

struct CGameMasterServerTask_GetOnlineProfile : public CNetMasterServerRequestTask {
};

struct CGameMasterServerRichPresenceTaskResult_NextPresence : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_PlanetsTransaction_Bill : public CWebServicesTaskResult {
  enum class CWebServicesTaskResult_PlanetsTransaction_Bill::EState {
    Void = 0,
    CreatingTransaction = 1,
    Issued = 2,
    ValidatingPayement = 3,
    Payed = 4,
    Refused = 5,
    Error = 6,
  };
  const CWebServicesTaskResult_PlanetsTransaction_Bill::EState State; // Maniascript
  const uint TransactionId; // Maniascript
  const string LoginPayer; // Maniascript
  const string LoginPayee; // Maniascript
  const wstring Label; // Maniascript
  const int Cost; // Maniascript
};

struct CGameScriptMapVehicleAnchor : public CMwNod {
  const wstring ItemName; // Maniascript
  const MwId ItemModelId; // Maniascript
};

// userName: 'ShootIconSetting'
// File extension: 'ShootIconSetting.gbx.json'
struct CGameShootIconSetting : public CMwNod {
  CGameShootIconSetting();

  NGameIconShooter_SSceneSetting Scene;
  NGameIconShooter_SCameraSetting Camera_ItemCustom;
};

struct CGameScriptMapWaypoint : public CMwNod {
  const bool IsFinish; // Maniascript
  const bool IsMultiLap; // Maniascript
};

struct CGameControlCameraHelico : public CGameControlCamera {
  CGameControlCameraHelico();

  CPlugVehicleCameraHelicoModel* m_Model;
};

struct CWebServicesTask_PostConnect : public CWebServicesTaskSequence {
};

// userName: 'ShootIconConfig'
// File extension: 'ShootIconConfig.gbx.json'
struct CGameShootIconConfig : public CMwNod {
  CGameShootIconConfig();

  CSystemFidFile* ShootSetting;
  CSystemFidFile* ItemToSkin;
  string SkinOptions;
  uint IconSize;
};

struct CWebServicesTask_PostConnect_BannedCryptedChecksumsList : public CWebServicesTaskSequence {
};

struct CWebServicesTask_PostConnect_UrlConfig : public CWebServicesTaskSequence {
};

struct CGameEditorMapScriptClipList : public CMwNod {
  const MwFastBuffer<CGameEditorMapScriptClip*> Clips; // Maniascript
  const nat3 Size; // Maniascript
  bool SetClipListFromMacroblock(CGameCtnMacroBlockInfo* MacroBlockModel, nat3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool SetClipListFromMacroblockOnly(CGameCtnMacroBlockInfo* MacroBlockModel); // Maniascript
  bool SetClipListFromBlock(CGameCtnBlockInfo* BlockModel, nat3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  bool SetClipListFromBlockOnly(CGameCtnBlockInfo* BlockModel); // Maniascript
  void CreateAndAddClip(wstring Name, int3 Coord, int3 Offset, CGameEditorPluginMap::ECardinalDirections Dir, int ClipId); // Maniascript
  bool SetFromClipList(CGameEditorMapScriptClipList* ClipList, int3 Coord, CGameEditorPluginMap::ECardinalDirections Dir); // Maniascript
  void RemoveClip(CGameEditorMapScriptClip* Clip); // Maniascript
  void ClearClips(); // Maniascript
  void Destroy(); // Maniascript
};

struct CGameEditorMapScriptClip : public CMwNod {
  const wstring Name; // Maniascript
  enum class CGameEditorPluginMap::ECardinalDirections {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
  };
  CGameEditorPluginMap::ECardinalDirections Dir; // Maniascript
  const nat3 Coord; // Maniascript
  const nat3 Offset; // Maniascript
  int ClipId; // Maniascript
  int3 GetConnectableCoord(); // Maniascript
};

struct CWebServicesTask_SynchronizeProfileChunks : public CWebServicesTaskSequence {
};

struct CGamePlaygroundModuleClientChrono : public CGamePlaygroundModuleClient {
  CGameUILayer* const ChronoLayer; // Maniascript
};

struct CGamePlaygroundModuleServerChrono : public CGamePlaygroundModuleServer {
  void StopChrono(); // Maniascript
  void StartChrono(); // Maniascript
  void Reset(); // Maniascript
  void Reset2(CGameScriptPlayer* Player); // Maniascript
};

struct CGamePlaygroundModuleClientSpeedMeter : public CGamePlaygroundModuleClient {
  CGameUILayer* const SpeedMeterLayer; // Maniascript
};

struct CGamePlaygroundModuleServerSpeedMeter : public CGamePlaygroundModuleServer {
  void Reset(CGameScriptPlayer* Player); // Maniascript
  void SetGlobalScale(float Scale); // Maniascript
  void SetGaugeNumber(int Number); // Maniascript
  void SetGaugeAngle(float Angle); // Maniascript
  void SetGaugeBGVisible(bool Visibility); // Maniascript
  void SetSpeedLineVisible(bool Visibility); // Maniascript
  void SetGaugeSpeedMax(float Maximum); // Maniascript
  void SetSpeedValueScale(float Scale); // Maniascript
  void SetSpeedUnitScale(float Scale); // Maniascript
  void SetFooterUnitVisible(bool Visibility); // Maniascript
  void SetFooterLineVisible(bool Visibility); // Maniascript
  void SetDistanceVisible(bool Visibility); // Maniascript
  void SetDistanceFooterLineVisible(bool Visibility); // Maniascript
};

struct CGamePlaygroundModuleClientPlayerState : public CGamePlaygroundModuleClient {
  CGameUILayer* const PlayerStateLayer; // Maniascript
};

struct CGamePlaygroundModuleServerPlayerState : public CGamePlaygroundModuleServer {
  void Reset(CGameScriptPlayer* Player); // Maniascript
};

struct CGamePlaygroundModuleClientTeamState : public CGamePlaygroundModuleClient {
  CGameUILayer* const TeamStateLayer; // Maniascript
};

struct CGamePlaygroundModuleServerTeamState : public CGamePlaygroundModuleServer {
  void Reset(CGameScriptPlayer* Player); // Maniascript
};

struct CGamePlaygroundModuleClientAltimeter : public CGamePlaygroundModuleClient {
  CGameUILayer* const AltimeterLayer; // Maniascript
};

struct CGamePlaygroundModuleServerAltimeter : public CGamePlaygroundModuleServer {
};

struct CGamePlaygroundModuleClientThrottle : public CGamePlaygroundModuleClient {
  CGameUILayer* const ThrottleLayer; // Maniascript
};

struct CGamePlaygroundModuleServerThrottle : public CGamePlaygroundModuleServer {
};

struct CGameEditorUndoSystem_State : public CMwNod {
  CGameEditorUndoSystem_State();

};

struct CWebServicesTask_UploadProfileChunks : public CWebServicesTaskSequence {
};

struct CGameMasterServerTask_GetTitlePackagesInfos : public CNetMasterServerRequestTask {
};

struct CWebServicesTask_GetTitlePackagesInfos : public CWebServicesTaskSequence {
};

struct CGameMasterServerTask_UpdateManiaPlanetStationInfos : public CNetMasterServerRequestTask {
};

struct CWebServicesTask_LoadStation : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Title_GetPlayerInfos : public CWebServicesTaskSequence {
};

struct CGameMasterServerTask_GetPackageUpdateUrl : public CNetMasterServerRequestTask {
};

struct CWebServicesTask_GetPackageUpdateUrl : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_CreditedPackageUpdateUrlList : public CWebServicesTaskResult {
};

struct CGameMasterServerTask_GetAuthenticationToken : public CNetMasterServerRequestTask {
};

struct CGameMasterServerTask_GetAccountFromUplayUser : public CNetMasterServerRequestTask {
};

struct CGameMasterServerTask_GetSubscribedGroups : public CNetMasterServerRequestTask {
};

struct CGameMasterServerTask_UpdateOnlineProfile : public CNetMasterServerRequestTask {
};

struct CWebServicesTask_UnloadStation : public CWebServicesTaskSequence {
};

struct CGameEditorMapMacroBlockInstance : public CMwNod {
  CGameCtnMacroBlockInfo* const MacroblockModel; // Maniascript
  CGameEditorMapScriptClipList* const ClipList; // Maniascript
  enum class CGameEditorPluginMap::ECardinalDirections {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
  };
  CGameEditorPluginMap::ECardinalDirections Dir; // Maniascript
  const int3 Coord; // Maniascript
  uint Order; // Maniascript
  int UserData; // Maniascript
  const MwFastBuffer<nat3> UnitCoords; // Maniascript
  enum class CGameEditorPluginMap::EMapElemColor {
    Default = 0,
    White = 1,
    Green = 2,
    Blue = 3,
    Red = 4,
    Black = 5,
  };
  const CGameEditorPluginMap::EMapElemColor Color; // Maniascript
  const bool ForceMacroblockColor; // Maniascript
  nat3 GetSize(); // Maniascript
};

struct CGameScoreAndLeaderBoardManager : public CMwNod {
};

struct CWebServicesTask_PostConnect_Zone : public CWebServicesTaskSequence {
};

// Description: "Results of task requesting the zone of user."
struct CWebServicesTaskResult_UserZoneListScript : public CWebServicesTaskResult {
  const MwFastBuffer<CZone*> ZoneList; // Maniascript
  CZone* GetUserZone(string WebServicesUserId); // Maniascript
};

struct CWebServicesTaskResult_AccountTrophyGainHistory : public CWebServicesTaskResult {
};

struct CGameMasterServerTask_GetPlayerCreditedPackagesGroups : public CNetMasterServerRequestTask {
};

struct CWebServicesTask_GetPlayerCreditedPackagesGroups : public CWebServicesTaskSequence {
};

struct CGameEditorPluginCameraManager : public CMwNod {
  CGameEditorPluginCameraManager();

};

struct CGameEditorPluginCursorManager : public CMwNod {
  CGameEditorPluginCursorManager();

};

struct CGameEditorPluginCameraAPI : public CMwNod {
  CGameEditorPluginCameraAPI();

  enum class CGameEditorPluginCameraAPI::EZoomLevel {
    Close = 0,
    Medium = 1,
    Far = 2,
  };
  enum class CGameEditorPluginCameraAPI::ECameraVStep {
    Low = 0,
    MediumLow = 1,
    Medium = 2,
    MediumHigh = 3,
    High = 4,
  };
  void ReleaseLock(); // Maniascript
  void CenterOnCursor(); // Maniascript
  void WatchStart(); // Maniascript
  void WatchClosestFinishLine(); // Maniascript
  void WatchClosestCheckpoint(); // Maniascript
  void WatchWholeMap(); // Maniascript
  void WatchCustomSelection(); // Maniascript
  void MoveToMapCenter(); // Maniascript
  bool GetLock(); // Maniascript
  bool CanUse(); // Maniascript
  void IgnoreCameraCollisions(bool IgnoreCameraCollisions); // Maniascript
  void TurnH(bool Clockwise); // Maniascript
  void TurnH(bool Clockwise, bool HalfSteps); // Maniascript
  void TurnH(bool Clockwise, bool HalfSteps, bool Smooth); // Maniascript
  void TurnV(bool UpOrElseDown); // Maniascript
  void TurnV(bool UpOrElseDown, bool Smooth); // Maniascript
  void Move(CGameEditorPluginMap::ERelativeDirections RelativeDir); // Maniascript
  void FollowCursor(bool Follow); // Maniascript
  void CenterOnCursor(bool Smooth); // Maniascript
  void MoveToMapCenter(bool Smooth); // Maniascript
  void Watch(CGameCtnBlock* Block); // Maniascript
  void Watch(CGameCtnBlock* Block, bool Smooth); // Maniascript
  void Watch(CGameEditorMapMacroBlockInstance* Macroblock); // Maniascript
  void Watch(CGameEditorMapMacroBlockInstance* Macroblock, bool Smooth); // Maniascript
  void Watch(CGameEditorMapScriptClip* Clip); // Maniascript
  void Watch(CGameEditorMapScriptClip* Clip, bool Smooth); // Maniascript
  void WatchStart(bool Smooth); // Maniascript
  void WatchClosestFinishLine(bool Smooth); // Maniascript
  void WatchClosestCheckpoint(bool Smooth); // Maniascript
  void WatchWholeMap(bool Smooth); // Maniascript
  void WatchMacroblocks(int UserData); // Maniascript
  void WatchMacroblocks(int UserData, bool Smooth); // Maniascript
  void ZoomIn(bool Loop); // Maniascript
  void ZoomIn(bool Loop, bool Smooth); // Maniascript
  void ZoomOut(bool Loop); // Maniascript
  void ZoomOut(bool Loop, bool Smooth); // Maniascript
  void Zoom(CGameEditorPluginCameraAPI::EZoomLevel Level); // Maniascript
  void Zoom(CGameEditorPluginCameraAPI::EZoomLevel Level, bool Smooth); // Maniascript
  void Look(CGameEditorPluginMap::ECardinalDirections8 Direction); // Maniascript
  void Look(CGameEditorPluginMap::ECardinalDirections8 Direction, bool Smooth); // Maniascript
  void Look(CGameEditorPluginMap::ECardinalDirections Direction); // Maniascript
  void Look(CGameEditorPluginMap::ECardinalDirections Direction, bool Smooth); // Maniascript
  void SetVStep(CGameEditorPluginCameraAPI::ECameraVStep Step); // Maniascript
  void WatchCustomSelection(bool WatchCustomSelection); // Maniascript
  void ActivateScrollRotateMode(bool ActivateScrollRotateMode); // Maniascript
};

struct CGameEditorPluginCursorAPI : public CMwNod {
  CGameEditorPluginCursorAPI();

  void ReleaseLock(); // Maniascript
  nat3 Coord; // Maniascript
  enum class CGameEditorPluginMap::ECardinalDirections {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
  };
  CGameEditorPluginMap::ECardinalDirections Dir; // Maniascript
  CGameCtnBlockInfo* BlockModel; // Maniascript
  CGameCtnBlockInfo* TerrainBlockModel; // Maniascript
  CGameCtnMacroBlockInfo* MacroblockModel; // Maniascript
  float Brightness; // Maniascript
  bool HideDirectionalArrow; // Maniascript
  void MoveToCameraTarget(); // Maniascript
  void ResetCustomRGB(); // Maniascript
  bool GetLock(); // Maniascript
  bool CanUse(); // Maniascript
  bool Raise(); // Maniascript
  bool Lower(); // Maniascript
  void FollowCameraTarget(bool Follow); // Maniascript
  void Rotate(bool CWOrCCW); // Maniascript
  void Move(CGameEditorPluginMap::ECardinalDirections CardinalDir); // Maniascript
  void Move(CGameEditorPluginMap::ECardinalDirections8 CardinalDir8); // Maniascript
  void Move(CGameEditorPluginMap::ERelativeDirections RelativeDir); // Maniascript
  void DisableMouseDetection(bool Disable); // Maniascript
  bool CanPlace(); // Maniascript
  bool IsCustomRGBActivated(); // Maniascript
  bool IsCurrentItemAnimated(); // Maniascript
  bool CurrentMacroblockHasAnimatedItem(); // Maniascript
  bool CurrentSelectionHasAnimatedItem(); // Maniascript
  void SetCustomRGB(vec3 Color); // Maniascript
};

struct CGameEditorPluginMapConnectResults : public CMwNod {
  CGameEditorPluginMapConnectResults();

  bool CanPlace; // Maniascript
  int3 Coord; // Maniascript
  enum class CGameEditorPluginMap::ECardinalDirections {
    North = 0,
    East = 1,
    South = 2,
    West = 3,
  };
  CGameEditorPluginMap::ECardinalDirections Dir; // Maniascript
};

struct CGameEditorGenericInventory : public CMwNod {
  CGameEditorGenericInventory();

  CGameCtnArticleNodeDirectory* const CurrentRootNode; // Maniascript
  CGameCtnArticleNodeDirectory* const CurrentDirectory; // Maniascript
  CGameCtnArticleNode* const CurrentSelectedNode; // Maniascript
  MwFastBuffer<CGameCtnArticleNode*> RootNodes; // Maniascript
  void EnterDirectory(); // Maniascript
  void LeaveDirectory(); // Maniascript
  void SelectArticle(CGameCtnArticleNodeArticle* NodeArticle); // Maniascript
  void OpenDirectory(CGameCtnArticleNodeDirectory* NodeDirectory); // Maniascript
  void SelectNode(CGameCtnArticleNode* Node); // Maniascript
  void OpenBrotherDirectory(bool NextOrElsePrevious); // Maniascript
  void SelectBrotherArticle(bool NextOrElsePrevious); // Maniascript
  void SelectBrotherNode(bool NextOrElsePrevious); // Maniascript
  CGameCtnArticleNodeDirectory* GetDirectoryAfter(CGameCtnArticleNode* Node); // Maniascript
  CGameCtnArticleNodeDirectory* GetDirectoryBefore(CGameCtnArticleNode* Node); // Maniascript
  CGameCtnArticleNodeArticle* GetArticleAfter(CGameCtnArticleNode* Node); // Maniascript
  CGameCtnArticleNodeArticle* GetArticleBefore(CGameCtnArticleNode* Node); // Maniascript
};

struct CGameEditorMediaTracker : public CGameEditorParent {
};

struct CGameEditorSkin : public CGameEditorParent {
};

struct CGameEditorSkinPluginAPI : public CGameEditorPluginAPI {
  CGameEditorSkinPluginAPI();

  enum class CGameEditorSkinPluginAPI::EPainterMode {
    NoOp = 0,
    Fill = 1,
    Brush = 2,
    Sticker = 3,
    Layer = 4,
    BadgeSlots = 5,
    Team = 6,
  };
  enum class CGameEditorSkinPluginAPI::EPainterSolidType {
    Other = 0,
    CarWithPilot = 1,
    Pilot_Male = 2,
    Pilot_Female = 3,
    Helmet = 4,
  };
  enum class CGameEditorSkinPluginAPI::EEditorSkin_IconType {
    Stickers = 0,
    Brushs = 1,
    Layers = 2,
    SubObjects = 3,
  };
  MwSArray<CPlugBitmap*> AllIcons; // Maniascript
  MwSArray<CGameEditorEvent*> PendingEvents; // Maniascript
  MwSArray<vec3> CustomColors; // Maniascript
  CGameEditorSkinPluginAPI::EPainterMode PainterMode; // Maniascript
  CGameEditorSkinPluginAPI::EPainterSolidType PainterSolidType; // Maniascript
  bool IsTextEnabled; // Maniascript
  bool IsPickingColor; // Maniascript
  bool IsErasing; // Maniascript
  bool IsUsingShininessOnly; // Maniascript
  bool IsBrushNormal; // Maniascript
  bool IsTextSymmetry; // Maniascript
  MwSArray<uint> SubObjectsSelected; // Maniascript
  uint SubObjectsCount; // Maniascript
  MwSArray<uint> LayersSelected; // Maniascript
  uint LayersCount; // Maniascript
  MwSArray<uint> StickersSelected; // Maniascript
  uint StickersCount; // Maniascript
  MwSArray<uint> BrushsSelected; // Maniascript
  uint BrushsCount; // Maniascript
  MwSArray<uint> SubFolderElemSelected; // Maniascript
  uint CurSubFolderElemCount; // Maniascript
  float Transparency; // Maniascript
  float Scale; // Maniascript
  float Rough; // Maniascript
  float Metal; // Maniascript
  vec3 Color; // Maniascript
  uint EditionLayersCount; // Maniascript
  bool IsEditingLayer; // Maniascript
  uint EditionLayerSelected; // Maniascript
  uint SubObjectHovered; // Maniascript
  MwSArray<uint> CurrentLayerSubObjectsSelected; // Maniascript
  bool CanUndo; // Maniascript
  bool CanRedo; // Maniascript
  wstring TextToCreateBitmap; // Maniascript
  bool IsFillWithReplacement; // Maniascript
  wstring SkinName; // Maniascript
  bool IsSkinHelmetOk; // Maniascript
  bool IsCameraButtonOn; // Maniascript
  void ToggleIsTextEnabled(); // Maniascript
  void ToggleIsPickingColor(); // Maniascript
  void TogglePaintSymMode(); // Maniascript
  void EditCurrentLayerSelected(); // Maniascript
  void AddAndEditLayer(); // Maniascript
  void CloseEditSelectedLayer(); // Maniascript
  void MoveSelectedLayerUp(); // Maniascript
  void MoveSelectedLayerDown(); // Maniascript
  void DeleteSelectedLayer(); // Maniascript
  void Undo(); // Maniascript
  void Redo(); // Maniascript
  void ExportSkin(); // Maniascript
  void AlignBrushSticker(); // Maniascript
  void AskSwitchEditedNodType(); // Maniascript
  void ToggleHelper(); // Maniascript
  void ToggleLight(); // Maniascript
  void ExportLayerEdition(); // Maniascript
  void ImportLayerEdition(); // Maniascript
  CPlugBitmap* GetAllIconFromId(int Index, CGameEditorSkinPluginAPI::EEditorSkin_IconType IconType); // Maniascript
  CPlugBitmap* GetSubFolderIconFromId(int FolderIndex, int ElemIndex, CGameEditorSkinPluginAPI::EEditorSkin_IconType IconType); // Maniascript
  wstring GetLayerNameFromIndex(int Index); // Maniascript
  bool GetLayerVisibleFromIndex(int Index); // Maniascript
  void SetLayerVisibleFromIndex(bool IsVisible, int Index); // Maniascript
  bool SetPainterMode(CGameEditorSkinPluginAPI::EPainterMode Mode); // Maniascript
  int GetSubObjectMaxPage(int NbPerPage); // Maniascript
  int GetMainFrameMaxPage(int NbPerPage); // Maniascript
  void OnLayerSelected(int CurrentPage, int Index); // Maniascript
  void OnStickerSelected(int CurrentPage, int Index); // Maniascript
  void OnBrushSelected(int CurrentPage, int Index); // Maniascript
  void OnSubFolderElemSelected(int SubFolderCurrentPage, int ElemIndex); // Maniascript
  void SetCurrentLayerSelectedName(wstring Name); // Maniascript
  void AddScale(float Delta); // Maniascript
  void AddAngle(float DeltaRad); // Maniascript
  void SetSubObjectIndexForLayer(int SubObjectIndex, int LayerIndex, bool ResetFirst); // Maniascript
  bool IsElemAtScriptIndexFolder(int ScriptIndex, CGameEditorSkinPluginAPI::EEditorSkin_IconType IconType); // Maniascript
  wstring GetFolderNameFromScriptIndex(int ScriptIndex, CGameEditorSkinPluginAPI::EEditorSkin_IconType IconType); // Maniascript
  bool IsCtrlDown(); // Maniascript
  wstring GetSubObjectNameFromIndex(int ScriptIndex); // Maniascript
  void OpenMediaFolder(wstring SubFolderPath); // Maniascript
  void ReloadResources(bool forSubObjects); // Maniascript
};

struct CGameEditorMediaTrackerPluginAPI : public CGameEditorPluginAPI {
  enum class CGameEditorMediaTrackerPluginAPI::EMediaTrackerBlockType {
    Ghost = 0,
    CameraCustom = 1,
    CameraPath = 2,
    Time = 3,
    FxColors = 4,
    Sound = 5,
    Fog = 6,
    TransitionFade = 7,
    CameraEffectShake = 8,
    CameraEffectScript = 9,
    Stereo3d = 10,
    DOF = 11,
    ToneMapping = 12,
    BloomHdr = 13,
    DirtyLens = 14,
    ColorGrading = 15,
    FxCameraBlend = 16,
    MusicEffect = 17,
    TimeSpeed = 18,
    TextBlock = 19,
    Image = 20,
    ColoringCapturable = 21,
    ColoringBase = 22,
    CameraGame = 23,
    Trails = 24,
    Manialink = 25,
    EditingCut = 26,
    CamFxInertialTracking = 27,
    VehicleLight = 28,
    Interface = 29,
    Triangles2D = 30,
    Triangles3D = 31,
    CameraOrbital = 32,
    OpponentVisibility = 33,
    Spectators = 34,
  };
  enum class CGameEditorMediaTrackerPluginAPI::EMediaTrackerCopyType {
    None = 0,
    Key = 1,
    Block = 2,
  };
  enum class CGameEditorMediaTrackerPluginAPI::EMediaTrackerPasteType {
    None = 0,
    KeyInfo = 1,
    NewKey = 2,
    BlockCurrentTrack = 3,
    BlockNewTrack = 4,
  };
  enum class CGameEditorMediaTrackerPluginAPI::EMediaTrackerGhostRef {
    Author = 0,
    Ghost1 = 1,
    Ghost2 = 2,
    Ghost3 = 3,
  };
  void ImportClip_OnOk();
  void ExportClip_OnOk();
  void ImportGhosts_OnOk();
  float CurrentTimer; // Maniascript
  bool UseOrbitalInsteadOfFreeCam; // Maniascript
  bool UseClipCamWhenAvailable; // Maniascript
  bool CanUseClipCam; // Maniascript
  float PlaySpeed; // Maniascript
  float CameraSpeed; // Maniascript
  float ClipConditionValue; // Maniascript
  void TimePlay(); // Maniascript
  void TimeStop(); // Maniascript
  void TimeToggle(); // Maniascript
  void Rewind(); // Maniascript
  void Quit(); // Maniascript
  void UpdatePropertyListReadOnly(); // Maniascript
  void CreateBlock(); // Maniascript
  void CreateKey(); // Maniascript
  void ToggleTriggersMode(); // Maniascript
  void CreateClip(); // Maniascript
  void RemoveClip(); // Maniascript
  void Undo(); // Maniascript
  void Redo(); // Maniascript
  void ImportClip(); // Maniascript
  void ExportClip(); // Maniascript
  void ImportGhosts(); // Maniascript
  void ToggleRecordGhostMode(); // Maniascript
  void ShootVideo(); // Maniascript
  void ShootScreen(); // Maniascript
  void ComputeShadows(); // Maniascript
  void ToggleDisplayPlayerNames(); // Maniascript
  void InformInterfaceIsHidden(); // Maniascript
  void ToggleGhostRef(); // Maniascript
  void StopGhostRefPreview(); // Maniascript
  void ToggleClipCondition(); // Maniascript
  void Copy(); // Maniascript
  void Paste(); // Maniascript
  void Cut(); // Maniascript
  void DuplicateTrack(); // Maniascript
  void SplitBlock(); // Maniascript
  void MergeTracks(); // Maniascript
  void ToggleAlwaysShowTriggerZone(); // Maniascript
  void ResetTriggerZone(); // Maniascript
  void RemoveAllTracks(); // Maniascript
  CGameCtnMediaClip* const Clip; // Maniascript
  CGameCtnMediaClipGroup* const ClipGroup; // Maniascript
  const MwFastBuffer<CGameEditorEvent*> PendingEvents; // Maniascript
  const wstring PopUpMessage; // Maniascript
  const CGameEditorMediaTrackerPluginAPI::EMediaTrackerBlockType EditMode; // Maniascript
  void SetClip(CGameCtnMediaClip* Clip); // Maniascript
  bool IsPlaying(); // Maniascript
  bool IsDevMode(); // Maniascript
  void SetCameraDrawRect(vec2 Pos, vec2 Size); // Maniascript
  uint GetSelectedClip(); // Maniascript
  uint GetSelectedTrack(); // Maniascript
  uint GetSelectedBlock(); // Maniascript
  uint GetSelectedKey(); // Maniascript
  void SelectItem(uint Track, uint Block, uint Key); // Maniascript
  void CreateTrack(CGameEditorMediaTrackerPluginAPI::EMediaTrackerBlockType Type); // Maniascript
  bool CanCreateTrack(CGameEditorMediaTrackerPluginAPI::EMediaTrackerBlockType Type); // Maniascript
  bool IsKeyStartEnd(uint Track, uint Block, uint Key); // Maniascript
  void RemoveTrack(uint Track); // Maniascript
  void RemoveBlock(uint Track, uint Block); // Maniascript
  void RemoveKey(uint Track, uint Block, uint Key); // Maniascript
  bool CanRemoveAllKeys(uint Track, uint Block); // Maniascript
  void RemoveAllKeys(uint Track, uint Block); // Maniascript
  void SetProcessCamInputs(bool ProcessCamInputs); // Maniascript
  void ToolBarSetVisible(bool Visible); // Maniascript
  bool IsTriggersModeOn(); // Maniascript
  bool IsRecordGhostModeOn(); // Maniascript
  void SetTempHidePropertyList(bool TempHide); // Maniascript
  void SetClipName(uint Index, wstring Name); // Maniascript
  void SetTrackName(uint Index, wstring Name); // Maniascript
  void SetTrackIsCycling(CGameCtnMediaTrack* Track, bool IsCycling); // Maniascript
  void SetTrackIsKeepPlaying(CGameCtnMediaTrack* Track, bool IsKeepPlaying); // Maniascript
  bool GetTrackIsCycling(CGameCtnMediaTrack* Track); // Maniascript
  bool GetTrackIsKeepPlaying(CGameCtnMediaTrack* Track); // Maniascript
  void SetStartIsCycling(CGameCtnMediaTrack* Track); // Maniascript
  void SetStopIsCycling(CGameCtnMediaTrack* Track); // Maniascript
  float GetStartIsCycling(CGameCtnMediaTrack* Track); // Maniascript
  float GetStopIsCycling(CGameCtnMediaTrack* Track); // Maniascript
  wstring GetGhostRefName(); // Maniascript
  wstring GetClipConditionName(); // Maniascript
  bool HasClipCondition(); // Maniascript
  float GetTimeLimit(); // Maniascript
  void SetTimer(string Time); // Maniascript
  void SetCurrentBlockStart(string Start); // Maniascript
  void SetCurrentBlockEnd(string End); // Maniascript
  CGameEditorMediaTrackerPluginAPI::EMediaTrackerCopyType CanCopy(); // Maniascript
  CGameEditorMediaTrackerPluginAPI::EMediaTrackerCopyType CanCut(); // Maniascript
  CGameEditorMediaTrackerPluginAPI::EMediaTrackerPasteType CanPaste(); // Maniascript
  bool HasCopiedItem(); // Maniascript
  bool IsPropertyListReadOnlyAndVisible(); // Maniascript
  void SelectGhostRef(CGameEditorMediaTrackerPluginAPI::EMediaTrackerGhostRef Type); // Maniascript
  void DeleteGhostRef(CGameEditorMediaTrackerPluginAPI::EMediaTrackerGhostRef Type); // Maniascript
  void RecordGhostRef(CGameEditorMediaTrackerPluginAPI::EMediaTrackerGhostRef Type); // Maniascript
  bool GhostRefExists(CGameEditorMediaTrackerPluginAPI::EMediaTrackerGhostRef Type); // Maniascript
  CGameEditorMediaTrackerPluginAPI::EMediaTrackerGhostRef GetSelectedGhostRef(); // Maniascript
  void StartGhostRefPreview(CGameEditorMediaTrackerPluginAPI::EMediaTrackerGhostRef Type); // Maniascript
  void RemoveAllGhostTracksExcept(uint Track); // Maniascript
  void RemoveAllCameraTracksExcept(uint Track); // Maniascript
  bool CanRemoveAllGhostTracks(); // Maniascript
  bool CanRemoveAllCameraTracks(); // Maniascript
  bool IsGhostRefPreview(); // Maniascript
  void SetProfileTheme(uint Theme); // Maniascript
  uint GetProfileTheme(); // Maniascript
};

struct CGameManialinkColorChooser : public CGameManialinkControl {
  CGameManialinkColorChooser();

  vec3 Color; // Maniascript
};

struct CGameManialinkSlider : public CGameManialinkControl {
  CGameManialinkSlider();

  float Value; // Maniascript
  float MinRange; // Maniascript
  float MaxRange; // Maniascript
};

struct CGameManialinkTimeLine : public CGameManialinkControl {
  CGameManialinkTimeLine();

  float Time; // Maniascript
  float TimeMin; // Maniascript
  float TimeMax; // Maniascript
  uint Page; // Maniascript
  uint TracksPerPage; // Maniascript
  const bool IsDraggingCursor; // Maniascript
  const bool IsDraggingBlock; // Maniascript
  const bool IsDraggingKey; // Maniascript
  const bool IsScaling; // Maniascript
  const bool IsPanning; // Maniascript
  void ShowFullTimeLine(); // Maniascript
  void ResetCycling(); // Maniascript
  void SetSimpleMediaTrackerPilot(CGameEditorMediaTrackerPluginAPI* MediaTracker); // Maniascript
  void SetSelection(uint Track, uint Block, uint Key); // Maniascript
  void SetSeparatorColor(string HexColor); // Maniascript
  void SetBgColor(string HexColor); // Maniascript
  void SetCursorColor(string HexColor); // Maniascript
  void SetRulerColor(string HexColor); // Maniascript
  void SetRulerLineColor(string HexColor); // Maniascript
  void SetKeyColor(string HexColor); // Maniascript
  void SetStartIsCycling(float Time); // Maniascript
  void SetStopIsCycling(float Time); // Maniascript
  void SetTimeLimit(float TimeLimit); // Maniascript
};

struct CGameDataFileTask_Map_NadeoServices_Get : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_Map_NadeoServices_Register : public CWebServicesTaskSequence {
};

struct CGameEditorAction : public CGameEditorParent {
  CGameEditorAction();

};

struct CGameScoreTask_GetSeason : public CWebServicesTaskSequence {
};

struct CGameEditorCustomBullet : public CGameEditorParent {
  CGameEditorCustomBullet();

};

struct CWebServicesTaskResult_Season : public CWebServicesTaskResult {
};

struct CGameEditorScript : public CGameEditorParent {
  CGameEditorScript();

};

struct CGameSeasonScoreManager : public CMwNod {
};

struct CGameSeasonScoreManager_MapRecord : public CGameSeasonScoreManager {
};

struct CGameSeasonScoreManager_MultiAsyncLevel : public CGameSeasonScoreManager {
};

struct CGameScoreTask_AddMapListToSeason : public CWebServicesTaskSequence {
};

struct CGameScoreTask_RemoveMapListToSeason : public CWebServicesTaskSequence {
};

struct CGameScoreTask_GetPlayerSeasonMapRecordList : public CWebServicesTaskSequence {
};

// File extension: 'EDVerticalClip.Gbx'
struct CGameCtnBlockInfoClipVertical : public CGameCtnBlockInfoClip {
  CGameCtnBlockInfoClipVertical();

  MwId VerticalClipGroupId;
};

// Description: "The ghosts manager."
struct CGameGhostMgrScript : public CMwNod {
  CGameGhostMgrScript();

  MwId Ghost_Add(CGameGhostScript* Ghost); // Maniascript
  MwId Ghost_Add(CGameGhostScript* Ghost, bool IsGhostLayer); // Maniascript
  MwId Ghost_Add(CGameGhostScript* Ghost, bool IsGhostLayer, int TimeOffset); // Maniascript
  MwId Ghost_AddWaypointSynced(CGameGhostScript* Ghost, bool IsGhostLayer); // Maniascript
  bool Ghost_IsReplayOver(MwId GhostInstanceId); // Maniascript
  bool Ghost_IsVisible(MwId GhostInstanceId); // Maniascript
  void Ghost_Remove(MwId GhostInstanceId); // Maniascript
  void Ghost_RemoveAll(); // Maniascript
  void Ghost_SetDossard(MwId GhostInstanceId, string Dossard); // Maniascript
  void Ghost_SetDossard(MwId GhostInstanceId, string Dossard, vec3 Color); // Maniascript
};

// userName: 'BlockInfoGroups'
// File extension: 'gbx.json'
struct CGameBlockInfoGroups : public CMwNod {
  CGameBlockInfoGroups();

  MwFastBuffer<SGameBlockInfoGroup> Groups;
};

struct CWebServicesTask_GetAccountXp : public CWebServicesTaskSequence {
};

// userName: 'BlockInfoTree'
// File extension: 'gbx.json'
struct CGameBlockInfoTreeRoot : public CMwNod {
  CGameBlockInfoTreeRoot();

  MwSArray<SGameBlockInfoTree> RootChilds;
};

struct CGameDataFileTask_AccountSkin_NadeoServices_Unset : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_AccountTrophyLastYearSummary : public CWebServicesTaskResult {
};

struct CGameScoreTask_GetAccountTrophyLastYearSummary : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_TrophySoloMedalAchievementSettings : public CWebServicesTaskResult {
};

struct CGameScoreTask_GetTrophySoloMedalAchievementSettings : public CWebServicesTaskSequence {
};

// Description: "Task result containing a last year summary of account trophy."
struct CWebServicesTaskResult_AccountTrophyLastYearSummaryScript : public CWebServicesTaskResult_AccountTrophyLastYearSummary {
  CAccountTrophyLastYearSummary* const Summary; // Maniascript
};

struct CGameDataFileTask_AccountSkin_NadeoServices_AddFavorite : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_AccountSkin_NadeoServices_GetFavoriteList : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_AccountSkin_NadeoServices_GetList : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_AccountSkin_NadeoServices_RemoveFavorite : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_AccountSkin_NadeoServices_Set : public CWebServicesTaskSequence {
};

// Description: "Audio device settings."
struct CGameAudioSettingsWrapper : public CMwNod {
  CGameAudioSettingsWrapper();

  uint DirtyCounter; // Maniascript
  MwFastBuffer<string> Devices; // Maniascript
  string Device_Current; // Maniascript
  string Device_NextApply; // Maniascript
  const uint VoiceChat_Devices_DirtyCounter; // Maniascript
  const MwFastBuffer<string> VoiceChat_Devices_In; // Maniascript
  const MwFastBuffer<string> VoiceChat_Devices_Out; // Maniascript
  const string VoiceChat_Device_In_Current; // Maniascript
  const string VoiceChat_Device_Out_Current; // Maniascript
  string VoiceChat_Device_In_NextApply; // Maniascript
  string VoiceChat_Device_Out_NextApply; // Maniascript
  float VoiceChat_SpeakerVolume; // Maniascript
  NSystemConfig::EVoiceDetectionMode VoiceChat_VoiceDetection_Mode; // Maniascript
  float VoiceChat_VoiceDetection_Sensitivity; // Maniascript
  bool VoiceChat_VoiceDetection_Auto; // Maniascript
  float VoiceChat_VoiceDetection; // Maniascript
};

// userName: 'ItemModelTree'
// File extension: 'gbx.json'
struct CGameItemModelTreeRoot : public CMwNod {
  CGameItemModelTreeRoot();

  SConstString Collection;
  EGameCollectorSearchScope SearchIn;
  MwSArray<SGameItemModelTree> RootChilds;
};

// Description: "Task result containing the settings of a solo medal trophy achievement."
struct CWebServicesTaskResult_TrophySoloMedalAchievementSettingsScript : public CWebServicesTaskResult_TrophySoloMedalAchievementSettings {
  CTrophySoloMedalAchievementSettings* const Settings; // Maniascript
};

// File extension: 'EDHorizontalClip.Gbx'
struct CGameCtnBlockInfoClipHorizontal : public CGameCtnBlockInfoClip {
  CGameCtnBlockInfoClipHorizontal();

  MwId HorizontalClipGroupId;
};

struct CWebServicesTaskResult_NadeoServicesItemCollection : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_NadeoServicesItemCollectionList : public CWebServicesTaskResult {
};

// Description: "Task result containing a ItemCollection info from NadeoServices."
struct CWebServicesTaskResult_NadeoServicesItemCollectionScript : public CWebServicesTaskResult_NadeoServicesItemCollection {
  CNadeoServicesItemCollection* const ItemCollection; // Maniascript
};

// Description: "Task result containing a list of ItemCollection info from NadeoServices."
struct CWebServicesTaskResult_NadeoServicesItemCollectionListScript : public CWebServicesTaskResult_NadeoServicesItemCollectionList {
  const MwFastBuffer<CNadeoServicesItemCollection*> ItemCollectionList; // Maniascript
};

struct CGameDataFileTask_ItemCollection_NadeoServices_AddFavorite : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_ItemCollection_NadeoServices_Create : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_ItemCollection_NadeoServices_Get : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_ItemCollection_NadeoServices_GetAccountList : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_ItemCollection_NadeoServices_GetFavoriteList : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_ItemCollection_NadeoServices_GetList : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_ItemCollection_NadeoServices_RemoveFavorite : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_ItemCollection_NadeoServices_Update : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_ItemCollection_NadeoServices_CreateVersion : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_ItemCollection_NadeoServices_SetActivityId : public CWebServicesTaskSequence {
};

struct CGameScoreTask_LoadAndSynchronizeSeasonScoreList : public CWebServicesTaskSequence {
};

struct CGameScoreTask_LoadAndSynchronizePersonalBestScoreList : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_ItemCollection_NadeoServices_RemoveFavoriteFromName : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_ItemCollection_NadeoServices_GetListFromIdentifierList : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_NadeoServicesItemCollectionWithClubInfoList : public CWebServicesTaskResult {
};

struct CGameSessionArchive : public CMwNod {
};

struct CWebServicesTaskResult_PrestigeListScript : public CWebServicesTaskResult_WSPrestigeList {
  const MwFastBuffer<CPrestige*> PrestigeList; // Maniascript
};

// Description: "Task result containing a list of friend info."
struct CWebServicesTaskResult_FriendListScript : public CWebServicesTaskResult_WSFriendList {
  const MwFastBuffer<CFriend*> FriendList; // Maniascript
};

struct CGameUserService : public CMwNod {
};

struct CWebServicesTaskResult_UserPrestigeListScript : public CWebServicesTaskResult_WSUserPrestigeList {
  const MwFastBuffer<CUserPrestige*> UserPrestigeList; // Maniascript
};

struct CWebServicesTaskResult_Squad : public CWebServicesTaskResult {
};

// Description: "Task result containing a Squad"
struct CWebServicesTaskResult_SquadScript : public CWebServicesTaskResult_Squad {
  CSquad* const Squad; // Maniascript
};

struct CGameUserTask_Squad_AcceptInvitation : public CGameUserTask_Squad_AbstractTask {
};

struct CGameUserTask_Squad_CancelInvitation : public CGameUserTask_Squad_AbstractTask {
};

struct CGameUserTask_Squad_Create : public CGameUserTask_Squad_AbstractTask {
};

struct CGameUserTask_Squad_DeclineInvitation : public CGameUserTask_Squad_AbstractTask {
};

struct CGameUserTask_Squad_Get : public CGameUserTask_Squad_AbstractTask {
};

struct CGameUserTask_Squad_GetCurrent : public CGameUserTask_Squad_AbstractTask {
};

struct CGameUserTask_Squad_InviteInto : public CGameUserTask_Squad_AbstractTask {
};

struct CGameUserTask_Squad_RemoveMember : public CGameUserTask_Squad_AbstractTask {
};

struct CGameUserTask_Squad_SetLeader : public CGameUserTask_Squad_AbstractTask {
};

struct CGameWebServicesNotificationService : public CMwNod {
};

// Description: "API for webservices notifications."
struct CGameWebServicesNotificationManagerScript : public CMwNod {
  const MwFastBuffer<CWebServicesTaskResult*> TaskResults; // Maniascript
  void TaskResult_Release(MwId TaskId); // Maniascript
  bool Notification_IsAvailable(MwId UserId); // Maniascript
  CWebServicesTaskResult_WSNotificationScript* Notification_PopNext(MwId UserId); // Maniascript
};

struct CWebServicesTaskResult_WSNotification : public CWebServicesTaskResult {
};

// Description: "Task result containing a Notification"
struct CWebServicesTaskResult_WSNotificationScript : public CWebServicesTaskResult_WSNotification {
  CNotification* const Notification; // Maniascript
};

struct CGameWebServicesNotificationTask_PopNextNotification : public CWebServicesTaskSequence {
};

struct CGameUserTask_Squad_AbstractTask : public CWebServicesTaskSequence {
};

struct CGameManialinkScriptHandler_ReadOnly : public CMwNod {
  enum class CGameManialinkScriptHandler_ReadOnly::ESystemPlatform {
    None = 0,
    Steam = 1,
    UPlay = 2,
    PS4 = 3,
    XBoxOne = 4,
    PS5 = 5,
    XBoxSeries = 6,
    Stadia = 7,
    Luna = 8,
  };
  enum class CGameManialinkScriptHandler_ReadOnly::ESystemSkuIdentifier {
    Unknown = 0,
    EU = 1,
    US = 2,
    JP = 3,
    CN = 4,
  };
  CGameManialinkPage* Page; // Maniascript
  bool PageIsVisible; // Maniascript
  bool PageAlwaysUpdateScript; // Maniascript
  uint Now; // Maniascript
  uint Period; // Maniascript
  uint CurrentTime; // Maniascript
  string CurrentTimeText; // Maniascript
  string CurrentLocalDateText; // Maniascript
  wstring CurrentTimezone; // Maniascript
  CGamePlayerInfo* const LocalUser; // Maniascript
  CGameManiaTitle* const LoadedTitle; // Maniascript
  CGameManialinkScriptHandler_ReadOnly::ESystemPlatform SystemPlatform; // Maniascript
  CGameManialinkScriptHandler_ReadOnly::ESystemSkuIdentifier SystemSkuIdentifier; // Maniascript
  CXmlScriptParsingManager* Xml; // Maniascript
  CGameVideoScriptManager* Video; // Maniascript
  CInputScriptManager* Input; // Maniascript
  CGameManialinkAnimManager* AnimMgr; // Maniascript
};

struct CGameScriptHandlerPlaygroundInterface_ReadOnly : public CGameManialinkScriptHandler_ReadOnly {
  uint GameTime; // Maniascript
  CGamePlaygroundClientScriptAPI* const Playground; // Maniascript
  bool IsSpectator; // Maniascript
  bool IsSpectatorClient; // Maniascript
  bool UseClans; // Maniascript
  bool UseForcedClans; // Maniascript
  CGameCtnChallenge* const Map; // Maniascript
  const MwSArray<CGameTeamProfile*> Teams; // Maniascript
  bool IsInGameMenuDisplayed; // Maniascript
  string CurrentServerLogin; // Maniascript
  wstring CurrentServerName; // Maniascript
  wstring CurrentServerDesc; // Maniascript
  string CurrentServerJoinLink; // Maniascript
  wstring CurrentServerModeName; // Maniascript
  uint SplitScreenNum; // Maniascript
  uint SplitScreenCount; // Maniascript
  bool SplitScreenIsHorizontal; // Maniascript
};

struct CWebServicesTask_PostConnect_Tag : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_UserPrestigeScript : public CWebServicesTaskResult_WSUserPrestige {
  CUserPrestige* const UserPrestige; // Maniascript
  const bool HasUserPrestige; // Maniascript
};

// Description: "Results of task requesting the club tag of users."
struct CWebServicesTaskResult_ClubTagListScript : public CWebServicesTaskResult {
  wstring GetClubTag(string WebServicesUserId); // Maniascript
};

// File extension: 'GameCtnMediaBlockOpponentVisibility.gbx'
struct CGameCtnMediaBlockOpponentVisibility : public CGameCtnMediaBlock {
  CGameCtnMediaBlockOpponentVisibility();

};

// Description: "Results containing a list of zones."
struct CWebServicesTaskResult_ZoneListScript : public CWebServicesTaskResult_WSZonePtrList {
  const MwFastBuffer<CZone*> Zones; // Maniascript
  const MwFastBuffer<CZone*> ZoneList; // Maniascript
  CZone* GetZone(string ZoneId); // Maniascript
};

struct CGameUserVoiceChat : public CMwNod {
  CGameUserVoiceChat();

  enum class CGameUserVoiceChat::EMuteSetting {
    Muted = 0,
    NotMuted = 1,
  };
  CGameUserVoiceChat::EMuteSetting MuteSetting; // Maniascript
  const bool IsLocal; // Maniascript
  const bool IsMuted; // Maniascript
  const bool MuteChangePending; // Maniascript
  const bool IsSpeaking; // Maniascript
  const bool IsConnected; // Maniascript
  const bool Supported; // Maniascript
  CGamePlayerInfo* const MatchingPlayerInfo; // Maniascript
};

// Description: "This is the Manialink browser interface."
struct CGameScriptHandlerMediaTrack : public CGameManialinkScriptHandler {
  CGameCtnChallenge* const CurMap; // Maniascript
  void ShowCurMapCard(); // Maniascript
};

} // namespace Game

namespace Graphic {

struct GxLight : public CMwNod {
  GxLight();

  vec3 Color;
  float Intensity; // Range: 0 - 1
  float DiffuseIntensity; // Range: 0 - 1
  float ShadowIntensity; // Range: 0 - 1
  vec3 ShadowRGB;
  bool DoLighting;
  bool LightMapOnly;
  bool IsInversed;
  bool IsShadowGen;
  bool DoSpecular;
  bool HasLensFlare;
  float FlareIntensity; // Range: 0 - 1
  bool HasSprite;
  bool IgnoreLocalScale;
  bool EnableGroup0;
  bool EnableGroup1;
  bool EnableGroup2;
  bool EnableGroup3;
  CMwNod* const PlugLight;
};

struct CGxLightBall : public GxLightPoint {
  CGxLightBall();

  enum class CGxLightBall::EStaticShadow {
    LightMap_ID_list = 0,
    LightMap_RGB_accumulation = 1,
  };
  CGxLightBall::EStaticShadow StaticShadow;
  const uint StaticShadowAsNat;
  vec3 AmbientRGB;
  float Radius; // Range: 0.01 - 50
  bool CustomRadiusSpecular;
  float RadiusSpecular; // Range: 0.01 - 50
  bool CustomRadiusIndex;
  float RadiusIndex; // Range: 0.01 - 50
  bool CustomRadiusShadow;
  float RadiusShadow; // Range: 0.01 - 50
  bool CustomRadiusFlare;
  float RadiusFlare; // Range: 0.01 - 50
  float EmittingRadius;
  float EmittingCylinderLenZ;
  const UnnamedEnum AttenuationType;
  UnnamedEnum TweakHN2;
  const float AttHTnLR; // Range: 0 - 10
  const float AttHTnLR2; // Range: 0 - 10
  float AttHyper2DerivAt0; // Range: -5 - 0
  float AttHyper2Tension; // Range: 0 - 1
};

struct GxLightPoint : public GxLightNotAmbient {
  GxLightPoint();

  float FlareSize;
  float FlareBiasZ;
};

struct CGxFog : public CMwNod {
  CGxFog();

};

struct GxLightAmbient : public GxLight {
  GxLightAmbient();

  float ShadeMinY;
  float ShadeMaxY;
};

struct GxLightNotAmbient : public GxLight {
  GxLightNotAmbient();

};

struct GxLightDirectional : public GxLightNotAmbient {
  GxLightDirectional();

  vec3 DblSidedRGB;
  vec3 ReverseRGB;
  float ReverseIntens; // Range: 0 - 1
  float EmittAngularSize;
  float FlareAngularSize;
  float FlareIntensPower; // Range: 0 - 1
  bool UseBoundaryHint;
  vec3 BoundaryHintPos;
  float DazzleAngleMax;
  float DazzleIntensity; // Range: 0 - 1
};

struct GxFogBlender : public CMwNod {
  GxFogBlender();

};

struct CGxLightFrustum : public CGxLightBall {
  CGxLightFrustum();

  bool IsOrtho;
  float NearZ;
  float FarZ;
  float FovY;
  float RatioXY;
  float SizeX;
  float SizeY;
  bool DoAttenuation;
  enum class CGxLightFrustum::EApply {
    ModulateAdd = 0,
    Modulate = 1,
    Add = 2,
    ModulateX2 = 3,
  };
  CGxLightFrustum::EApply Apply;
  enum class CGxLightFrustum::ETechnique {
    RenderCube = 0,
    _2dMaskNoClipZ = 1, // 2dMaskNoClipZ
    _2dBallLight = 2, // 2dBallLight
    GenShadowMask = 3,
  };
  CGxLightFrustum::ETechnique Technique;
  uint iShadowGroup;
  bool DoFadeZ;
  float RatioFadeZ; // Range: 0 - 1
  bool UseFacePosX;
  bool UseFaceNegX;
  bool UseFacePosY;
  bool UseFaceNegY;
  bool UseFacePosZ;
  bool UseFaceNegZ;
};

struct CGxLightSpot : public CGxLightBall {
  CGxLightSpot();

  float AngleInner; // Range: 0 - 179
  float AngleOuter; // Range: 0 - 179
  uint8 SubLightCountX;
  uint8 SubLightCountY;
};

} // namespace Graphic

namespace Function {

// File extension: 'FuncKey.Gbx'
struct CFuncKeys : public CFunc {
  MwFastArray<float> Xs;
  void Reset();
};

struct CFuncKeysTrans : public CFuncKeys {
  CFuncKeysTrans();

  MwFastArray<vec3> Trans;
};

struct CFuncPlug : public CFunc {
  float Period;
  float Phase;
  bool AutoCreateMotion;
  bool RandomizePhase;
  MwId InputValId;
};

struct CFuncLightIntensity : public CFuncLight {
  CFuncLightIntensity();

  float Intensity0;
  float Intensity1;
  float FlickerDuration;
};

struct CFuncTreeTranslate : public CFuncTree {
  CFuncTreeTranslate();

  vec3 StartPoint;
  vec3 EndPoint;
  bool IsPingPong;
  bool IsSmooth;
};

struct CFuncEnum : public CFunc {
  CFuncEnum();

};

// File extension: 'Func.Gbx'
struct CFunc : public CMwNod {
};

struct CFuncShader : public CFuncPlug {
};

// File extension: 'FuncShader.Gbx'
struct CFuncShaders : public CFuncShader {
  CFuncShaders();

  MwFastArray<CFuncShader*> FuncShaders;
};

// File extension: 'FuncShader.Gbx'
struct CFuncShaderLayerUV : public CFuncShader {
  CFuncShaderLayerUV();

  MwId LayerName;
  UnnamedEnum LayerEnum;
  enum class CFuncShaderLayerUV::ESignalType {
    TransLinear = 0,
    TransCircular = 1,
    Rotate = 2,
    TransSubTexture = 3,
    TransLinearScale = 4,
    CopySubTexture = 5,
    Scale = 6,
    Reset = 7,
    TransSubTexture2 = 8,
    _GComp = 9, // =>Comp
    TransLinearScaleRotate = 10,
  };
  CFuncShaderLayerUV::ESignalType SignalType;
  vec2 Base;
  vec2 Amplitude;
  vec2 Offset;
  vec2 Scale;
  float Angle; // Range: 0 - 360
  float AngleStart; // Range: 0 - 360
  float AngleEnd; // Range: 0 - 360
  uint NbSubTexture;
  uint NbSubTexturePerLine;
  uint NbSubTexturePerColumn;
  bool TopToBottom;
  bool WriteX;
  bool WriteY;
  bool WriteZ;
  bool WriteW;
  bool EnablePingPong;
  bool EnableSmooth;
  bool EnableMipMapping;
  bool EnableBlending;
  bool EnableSmoothBlend;
  CPlugBitmap* BitmapCopy;
};

struct CFuncLight : public CFuncPlug {
  enum class CFuncLight::EFctType {
    Sinus = 0,
    Flick = 1,
  };
  CFuncLight::EFctType FctType;
  float FlickPeriod;
  uint FlickCount;
};

struct CFuncLightColor : public CFuncLight {
  CFuncLightColor();

  vec3 Color0;
  vec3 Color1;
  CPlugFileImg* Image;
};

struct CFuncTree : public CFuncPlug {
};

// File extension: 'FuncTree.Gbx'
struct CFuncTreeRotate : public CFuncTree {
  CFuncTreeRotate();

  float AngleMin;
  float AngleMax;
};

struct CFuncTreeBend : public CFuncTree {
  CFuncTreeBend();

  float Amplitude;
};

struct CFuncKeysNatural : public CFuncKeys {
  CFuncKeysNatural();

  MwFastArray<uint> Naturals;
};

struct CFuncTreeSubVisualSequence : public CFuncTree {
  CFuncTreeSubVisualSequence();

  CFuncKeysNatural* SubKeys;
  bool SimpleModeIsLooping;
  uint SimpleModeStartIndex;
  uint SimpleModeEndIndex;
};

struct CFuncSegment : public CFunc {
  CFuncSegment();

  const uint KeyCount;
};

// File extension: 'FuncColorGradient.Gbx'
struct CFuncColorGradient : public CFunc {
  CFuncColorGradient();

  UnnamedEnum ColorSpace;
  vec3 KeyFrameValue0;
  vec3 KeyFrameValue1;
  vec3 KeyFrameValue2;
  vec3 KeyFrameValue3;
  float KeyFramePos1;
  float KeyFramePos2;
};

// File extension: 'FuncFullColorGradient.Gbx'
struct CFuncFullColorGradient : public CFunc {
  CFuncFullColorGradient();

  vec3 KeyFrameValue0;
  vec3 KeyFrameValue1;
  vec3 KeyFrameValue2;
  vec3 KeyFrameValue3;
  float KeyFramePos1;
  float KeyFramePos2;
};

} // namespace Function

namespace Hms {

struct CHmsCamera : public CHmsPoc {
  CHmsCamera();

  bool m_IsPickEnable;
  bool m_UseViewDependantRendering;
  CHmsCamera::EViewportRatio m_ViewportRatio;
  iso4 NextLocation;
  bool m_IsOverlay3d;
  bool ClearColorEnable;
  vec3 ClearColor;
  bool m_UseZBuffer;
  bool ScissorRect;
  bool FovRect;
  bool ClearZBuffer;
  vec2 DrawRectMin;
  vec2 DrawRectMax;
  vec2 ScissorMin;
  vec2 ScissorMax;
  vec2 FovRectMin;
  vec2 FovRectMax;
  float NearZ;
  float FarZ;
  float Fov;
  float ClampFovX;
  float ClampFovY;
  bool ClampFovAuto;
  float ClampFovRatioXy;
  float Width_Height;
  CHmsPicker* m_Picker;
  const uint8 m_GroupIndex;
  const uint8 m_IsInternal;
  const float WaterTop;
  const uint WaterIndex;
  uint IsJustAboveWater;
  uint IsInsideWater;
};

struct CHmsCorpus : public CHmsZoneElem {
  CHmsCorpus();

  CHmsItem* const Item;
};

struct CHmsItem : public CMwNod {
  CHmsItem();

  CPlugSolid* Solid;
  CHmsCorpus* const Corpus;
  CMwNod* const SceneMobil;
  bool IsVisionStatic;
  bool IsStatic;
  bool IsBackground;
  bool CopyCameraTranslationXZ;
  bool BackgroundZClipCullBefore;
  bool UseAccurateBBoxTest;
  bool IsForcePointDynamicCollisionResponse;
  const float LightmapQualityScale;
  uint CountShadowTexCasted;
  bool CanSelfShadow;
  bool CanFakeShadow;
  bool CastShadowGrp0;
  bool CastShadowGrp1;
  bool CastShadowGrp2;
  bool CastShadowGrp3;
  bool RecvShadowGrp0;
  bool RecvShadowGrp1;
  bool RecvShadowGrp2;
  bool RecvShadowGrp3;
  bool LightLensFlareEnable;
  bool LightEGroup0;
  bool LightEGroup1;
  bool LightEGroup2;
  bool LightEGroup3;
  bool VIdReflected;
  bool VIdReflectMirror;
  bool VIdRefracted;
  bool VIdWaterNormalDecals;
  bool VIdViewDepOcclusion;
  bool VIdOnlyRefracted;
  bool VIdHideWhenUnderground;
  bool VIdFoilage;
  bool VIdHideAlways;
  bool VIdHideButPick;
  bool VIdBackground;
  bool VIdGrassRGB;
  bool VIdLightGenP;
  bool VIdVehicle;
  bool VIdHideOnlyDirect;
  bool VIdInvisibleStopBounce;
  bool IsVisible;
};

struct CHmsZone : public CMwNod {
  CHmsZone();

  MwFastBufferCat<CHmsCorpus*> CorpusCats;
  bool FogByVertex;
  vec3 FogRGB;
  enum class CHmsZone::EGxFogFormula {
    None = 0,
    Exp = 1,
    Exp2 = 2,
    Linear = 3,
  };
  CHmsZone::EGxFogFormula FogFormula;
  enum class CHmsZone::EGxFogSpace {
    CameraFarZ = 0,
    World = 1,
  };
  CHmsZone::EGxFogSpace FogSpace;
  float FogLinearStart;
  float FogLinearEnd;
  float FogExpDensity;
  MwFastBuffer<CHmsPrecalcRender*> PrecalcRenders;
  const MwFastBuffer<CHmsLightArray*> DynamicLightArrays;
  const MwFastBuffer<CHmsDecalArray*> DynamicDecalArrays;
  const MwFastBuffer<CHmsDecalArray*> StaticDecalArrays;
  bool MRMaskWater;
  float MRTileH;
  vec3 MRPoint;
  vec3 MRNormal;
  MwFastBuffer<vec4> WaterPlaneEqInWs;
  bool IVIdMaskReflected;
  bool IVIdMaskReflectMirror;
  bool IVIdMaskRefracted;
  bool IVIdMaskWaterNormalDecals;
  bool IVIdMaskViewDepOcclusion;
  bool IVIdMaskOnlyRefracted;
  bool IVIdMaskHideWhenUnderground;
  bool IVIdMaskFoilage;
  bool IVIdMaskHideAlways;
  bool IVIdMaskHideButPick;
  bool IVIdMaskBackground;
  bool IVIdMaskGrassRGB;
  bool IVIdMaskLightGenP;
  bool IVIdMaskVehicle;
  bool IVIdMaskHideOnlyDirect;
  bool IVIdMaskInvisibleStopBounce;
  bool IVIdRefReflected;
  bool IVIdRefReflectMirror;
  bool IVIdRefRefracted;
  bool IVIdRefWaterNormalDecals;
  bool IVIdRefViewDepOcclusion;
  bool IVIdRefOnlyRefracted;
  bool IVIdRefHideWhenUnderground;
  bool IVIdRefFoilage;
  bool IVIdRefHideAlways;
  bool IVIdRefHideButPick;
  bool IVIdRefBackground;
  bool IVIdRefGrassRGB;
  bool IVIdRefLightGenP;
  bool IVIdRefVehicle;
  bool IVIdRefHideOnlyDirect;
  bool IVIdRefInvisibleStopBounce;
  bool SVIdMaskReflected;
  bool SVIdMaskReflectMirror;
  bool SVIdMaskRefracted;
  bool SVIdMaskWaterNormalDecals;
  bool SVIdMaskViewDepOcclusion;
  bool SVIdMaskOnlyRefracted;
  bool SVIdMaskHideWhenUnderground;
  bool SVIdMaskFoilage;
  bool SVIdMaskHideAlways;
  bool SVIdMaskHideButPick;
  bool SVIdMaskBackground;
  bool SVIdMaskGrassRGB;
  bool SVIdMaskLightGenP;
  bool SVIdMaskVehicle;
  bool SVIdMaskHideOnlyDirect;
  bool SVIdMaskInvisibleStopBounce;
  bool SVIdRefReflected;
  bool SVIdRefReflectMirror;
  bool SVIdRefRefracted;
  bool SVIdRefWaterNormalDecals;
  bool SVIdRefViewDepOcclusion;
  bool SVIdRefOnlyRefracted;
  bool SVIdRefHideWhenUnderground;
  bool SVIdRefFoilage;
  bool SVIdRefHideAlways;
  bool SVIdRefHideButPick;
  bool SVIdRefBackground;
  bool SVIdRefGrassRGB;
  bool SVIdRefLightGenP;
  bool SVIdRefVehicle;
  bool SVIdRefHideOnlyDirect;
  bool SVIdRefInvisibleStopBounce;
  void TestDynaSpriteMgr();
  CPlugBitmap* BitmapWaterFog;
  CPlugBitmap* BitmapCubeReflectHardSpecA;
  CPlugBitmap* BitmapCubeReflectHdrAlpha2;
  CMwNod* const UserData;
  NHmsZone_SPImp* const PImp;
  CHmsZoneVPacker* const VPacker;
  SHmsZoneVisionCst* VisionCst;
};

struct CHmsPortal : public CMwNod {
  CHmsPortal();

  bool IsActive;
  bool NeedClipping2D;
  bool NeedClipping3D;
  bool CanSeeThrough;
  float SeeThroughOpacity; // Range: 0 - 1
  bool IsVisualVisible;
  bool IsPickingPossible;
  bool CanPassThrough;
  bool SoundCanPassThrough;
  bool IsDirectPathSet;
  int DirectOcclusion;
  float DirectOcclusionSpectralRatio;
  float DirectOcclusionRatio;
  int IndirectOcclusion;
  float IndirectOcclusionSpectralRatio;
  float IndirectOcclusionRatio;
  int IndirectObstruction;
  float IndirectObstructionSpectralRatio;
};

struct CHmsPoc : public CHmsZoneElem {
  CHmsPoc();

  bool IsActive;
};

struct CHmsZoneElem : public CMwNod {
  CHmsZoneElem();

  CHmsZone* const Zone;
  iso4 Location;
  vec3 Vel;
};

struct CHmsZoneOverlay : public CHmsZone {
  CHmsZoneOverlay();

  bool IsVisible;
  CHmsZoneOverlay::EPickEnableMode m_PickEnableMode;
  uint SortOrder;
  EHmsOverlayAdaptRatio m_AdaptRatio;
  float m_AdaptRatio_FillExtraSpace_Opacity;
  bool m_UseZBuffer;
  vec2 m_DrawRect_min;
  vec2 m_DrawRect_max;
  vec3 Frustum_Center;
  vec3 Frustum_HalfDiag;
  vec3 ClearColor;
  MwFastBuffer<CHmsCorpus*> m_CorpusVisibles;
  MwFastBuffer<NHmsZoneOverlay_SClipRect> m_ClipRects;
  CHmsCamera* m_In3d_Camera;
  bool m_StickToHmd;
};

struct CHmsLight : public CHmsPoc {
  CHmsLight();

  UnnamedEnum UpdateType;
  GxLight* const MainGxLight;
  CPlugBitmap* BitmapFlare;
  CPlugBitmap* BitmapSprite;
  CPlugBitmap* BitmapProjector;
  bool ForceShadowGroup;
  uint ForceShadowGroupIndex;
  bool NeverSkipShadow;
};

struct CHmsPortalProperty : public CMwNod {
  CHmsPortalProperty();

  CHmsPortal* const Portal1;
  CHmsPortal* const Portal2;
  bool Visibility;
  bool Audibility;
};

struct CHmsViewport : public CMwNod {
  enum class CHmsViewport::ERenderDevice {
    PC0 = 0,
    PC1 = 1,
    PC2 = 2,
    PC3 = 3,
  };
  enum class CHmsViewport::EPC3Quality {
    VeryFast = 0,
    Fast = 1,
    Nice = 2,
    VeryNice = 3,
    Ultra = 4,
  };
  enum class CHmsViewport::EBitmapQuality {
    PC0 = 0,
    PC1 = 1,
    PC2 = 2,
    PC3 = 3,
  };
  const wstring DisplayName;
  const uint cPixelSwapX;
  const uint cPixelSwapY;
  const uint cPixelSwapHudX;
  const uint cPixelSwapHudY;
  CHmsViewport::ERenderDevice DeviceKind;
  uint iSubDevice;
  CHmsViewport::EPC3Quality PC3_Quality;
  CHmsViewport::EBitmapQuality BitmapQuality;
  UnnamedEnum MultiSampleType;
  UnnamedEnum DeferredAA;
  uint DisplaySyncIntervalCount;
  bool EnableTripleBuffer;
  bool ArePortalsActive;
  uint PortalMaxRecur;
  bool IsPickingPossible;
  vec2 LimitRectMin;
  vec2 LimitRectMax;
  float ConsoleOutputScale; // Range: 0.5 - 1
  const float AverageFps;
  float MipScaleZ_SysConfig;
  float MipScaleZ_Production;
  const float MipScaleZ;
  UnnamedEnum TextureRender;
  UnnamedEnum RenderShadows;
  UnnamedEnum RenderProjectors;
  bool RenderZoneVPacker;
  const MwFastBuffer<wstring> DisplayNames;
  const wstring ScreenShotFullName;
  bool ScreenShotForceRes;
  uint ScreenShotWidth;
  uint ScreenShotHeight;
  uint ScreenShot360_Height;
  bool ScreenShotUseAlpha;
  uint ScreenShotTileX;
  uint ScreenShotTileY;
  UnnamedEnum ScreenShot360;
  UnnamedEnum PixelOutput;
  void ScreenShotDoCaptureWebp();
  void ScreenShotDoCaptureTga();
  void ScreenShotDoCaptureJpg();
  void ScreenShotDoCaptureDDS();
  void ShaderConstantLogBindedValues();
  bool DisableOverlayRender;
  MwFastBuffer<CHmsZoneOverlay*> Underlays;
  MwFastBuffer<CHmsCamera*> Cameras;
  MwFastBuffer<CHmsZoneOverlay*> Overlays;
  CSystemConfig* SystemConfig;
  CSystemWindow* const SystemWindow;
  CHmsPicker* const Picker;
  bool MgrParticle_UseCameraMotion;
  bool MgrParticle_PauseKeepCameraMotion;
};

struct CHmsPrecalcRender : public CMwNod {
  CHmsPrecalcRender();

  CPlugBitmap* BitmapRGB;
  CPlugBitmap* BitmapDepth;
  enum class CHmsPrecalcRender::EBitmapDepth {
    Linear = 0,
    ZBuffer = 1,
  };
  CHmsPrecalcRender::EBitmapDepth BitmapDepthMode;
  string TreeIdDepthGen;
  const bool IsTreeDepthGenFound;
  float ZoomFactor;
  float ScrollPosX; // Range: -1 - 1
  float ScrollPosY; // Range: -1 - 1
};

struct CHmsShadowGroup : public CMwNod {
  CHmsShadowGroup();

  uint TexelCountX;
  uint TexelCountY;
  uint MaxShadowCountGrp0;
  float MappingQuality;
  float MaxTexelPerMeter;
  bool Enable;
  bool EnableLightDir;
  bool EnableLightPos;
  bool GameAllowLightDir;
  bool GameAllowLightPos;
  float LocalIntensity; // Range: 0 - 1
  bool NeedSelfShadow;
  bool ForceShadowMask;
  bool DepthNeed32b;
  bool IsStatic;
  bool IsStaticDirty;
  bool ForceWorldAlign;
  bool UseHqCasterBBox;
  bool AllStaticItemAreCaster;
  bool ShadowInShader;
  bool TmBackgroundReceives;
  bool TmBackgroundCast;
  bool DepthTestMaskRender;
  UnnamedEnum DepthBiasScale;
  uint DepthBiasConst;
  float DepthBiasSlope;
  float DepthBiasConstShaderExtra;
  float ShadeSlope; // Range: 0 - 1
  float Soft2dSlope; // Range: 0 - 2
  vec2 SoftSizeInW;
  float OpacityMipMapLodBias;
  uint MaskBlurTexelCount;
  uint PssmTexelCountX;
  uint PssmTexCount;
  const uint PssmTexCountActive;
  float PssmOverlapIn01; // Range: 0 - 1
  float PssmDistNF0;
  float PssmDistNF1;
  float PssmDistNF2;
  float PssmDistNF3;
  float PssmDistNF4;
  float PssmSkipCmpStaticScale;
};

struct CHmsViewportPerfDbg : public CMwNod {
  bool UseAverage;
  bool ShowMem;
  bool ForceOcclusion;
  UnnamedEnum ShowFps;
  UnnamedEnum Profile;
  UnnamedEnum SumUp;
  UnnamedEnum VisionSkip;
  bool SmallShader;
  bool SmallScissor;
  uint DebugParticle;
  UnnamedEnum FillForced;
  bool IsFillRateZTestEnable;
  bool IsFillRateStencilTestEnable;
  bool IsFillRateNoMipMapEnable;
  bool IsFillRateLegendEnable;
  vec3 WiredLayerColor;
};

struct CHmsMgrVisDyna : public CMwNod {
};

struct CHmsFogPlane : public CHmsZoneElem {
  CHmsFogPlane();

};

struct CHmsPicker : public CMwNod {
  CHmsPicker();

  bool IsEnable;
  const vec2 InputPos;
  CHmsCamera* const Camera;
  CHmsZoneOverlay* const Overlay;
  const vec2 PosRect;
  const vec3 RayDir;
  const vec3 RayPos;
  const uint In3dQuadIndex;
  CMwNod* const PickedSceneMobil;
  CHmsCorpus* const Corpus;
  CPlugTree* const Tree;
  CPlugSolid2Model* const Solid2Model;
  const uint VisDynaHandle;
  const vec3 PickPosV;
  const vec3 PickPosZ;
  const vec3 PickNormalV;
  const vec3 PickNormalZ;
  const float PickZ;
  const float PickZNoBiasZ;
  const float Depth;
};

struct CHmsConfig : public CMwNod {
  CHmsConfig();

  const MwFastArray<CHmsShadowGroup*> ShadowGroups;
};

struct CHmsItemShadow : public CMwNod {
  CHmsItemShadow();

  float Intensity; // Range: 0 - 1
  float FallOffStart;
  float FallOffEnd;
  float VolumeMaxZ;
};

struct CHmsLightMap : public CMwNod {
  CHmsLightMap();

  NHmsLightMap_SPImp* m_PImp;
  NHmsLightMap_SComputePImp* m_CptPImp;
};

struct CHmsLightMapCache : public CMwNod {
  CHmsLightMapCache();

  const MwId m_Id_IdCollection;
  const MwId m_Id_IdDecoration;
  const string Challenge;
  const string MostRecentSolid;
  const string MostRecentBlock;
  const uint m_MapperTexelCountX;
  const CHmsLightMapCache::ESortMode m_SortMode;
  const CHmsLightMapCache::EAllocMode m_AllocMode;
  const EPlugGpuPlatform m_GpuPlatform;
  const uint AmbSamples;
  const uint DirSamples;
  const uint PntSamples;
  const MwSArray<NHmsLightMapCache_SFrame> Frames;
  const EHmsLightMapCompress m_CompressMode;
  const CHmsLightMapCache::EVersion m_Version;
  const EHmsLightMapQuality m_Quality;
  const CHmsLightMapCache::EQualityVer m_QualityVer;
  const EHmsLightMapBump m_Bump;
  const float m_AllocatedTexelByMeter;
  const bool m_SpriteOriginY_WasWronglyTop;
  const uint cDecal2d;
  const uint cDecal3d;
};

// File extension: 'LightMapMood.Gbx'
struct CHmsLightMapMood : public CMwNod {
  CHmsLightMapMood();

};

struct CHmsCorpus2d : public CHmsCorpus {
  CHmsCorpus2d();

  uint ClipRectIndex;
  uint In3dQuadIndex;
};

struct CHmsAmbientOcc : public CMwNod {
  CHmsAmbientOcc();

};

struct CHmsLightProbeGrid : public CMwNod {
};

struct CHmsLightMapCacheSH : public CMwNod {
  CHmsLightMapCacheSH();

};

struct CHmsLightMapParam : public CMwNod {
  CHmsLightMapParam();

};

struct CHmsLightArray : public CMwNod {
  CHmsLightArray();

};

struct CHmsDecalArray : public CMwNod {
  CHmsDecalArray();

};

struct CHmsLightProbePartition : public CMwNod {
};

struct CHmsMgrVisEnvMap : public CMwNod {
};

struct CHmsMgrVisDynaDecal2d : public CMwNod {
};

struct CHmsVisMiniMap : public CMwNod {
};

struct CHmsVolumeShadow : public CMwNod {
};

struct SType_Warp : public CMwNod {
};

struct SType_VehicleVisForBodyPart : public CMwNod {
};

struct CHmsMoodBlender : public CMwNod {
};

struct CHmsMgrVisVolume : public CMwNod {
  CPlugBitmap* const Fog3D_BitmapBaseColor;
};

} // namespace Hms

namespace Control {

struct CControlBase : public CSceneMobil {
  void Draw();
  void Clean();
  UnnamedEnum AlignHorizontal;
  UnnamedEnum AlignVertical;
  CControlStyle* Style;
  bool IsReadOnly;
  bool AddFocusArea;
  bool DrawBackground;
  bool InheritVisualStyle;
  bool KeepFocusWeak;
  const uint ClipRectAllocatedIndex;
  const uint ClipRectCurrentIndex;
  bool DisableI18n;
  const bool HasSolid;
  bool IsSubSolid;
  bool IsDynamic;
  bool IsFocused;
  bool IsSelected;
  bool IsHiddenExternal;
  const bool IsVisualFocus;
  const bool IsVisualSelect;
  CControlContainer* const Parent;
  CMwNod* Nod;
  string StackText;
  wstring ToolTip;
  const bool IsHiddenInternal;
  const bool IsFocusCaptured;
  const bool IsUpdateNodNeeded;
  bool IsCreatedByScript;
  CPlugTree* ControlDisplayTree;
  CPlugTree* ControlDrawTree;
  vec2 BoxMin;
  vec2 BoxMax;
  CControlLayout* Layout;
  const float ClipLength;
  void OnAction();
};

struct CControlContainer : public CControlBase {
  MwFastArray<CControlBase*> Childs;
  bool IsClippingContainer;
  bool AcceptOwnControls;
  CControlBase* AddControl(string Id, vec3 Position, string Label, CMwNod* Nod, string Stack, string Type, CControlStyle* Style);
  CControlBase* AddInstance(CControlBase* Model, string Id, vec3 Position);
  CControlLabel* AddLabel(string Id, vec3 Position, string Label, CControlStyle* Style);
  void RemoveControl(CControlBase* Control);
};

struct CControlEffect : public CMwNod {
};

struct CControlLabel : public CControlText {
  CControlLabel();

  wstring Label;
  CPlugBitmap* Bitmap;
  CPlugShaderApply* ExternalShader;
  vec3 ImageColor;
  float ImageAlpha; // Range: 0 - 1
  bool DontDrawText_IfSolid;
};

struct CControlButton : public CControlText {
  CControlButton();

  wstring Label;
  CPlugSound* ActionSound;
  CFuncEnum* Icons;
  UnnamedEnum DisplayType;
  uint SubIconIndexOff;
  uint SubIconIndexOn;
  string IconId;
};

struct CControlEntry : public CControlText {
  CControlEntry();

  wstring String;
  uint MaxLength;
  UnnamedEnum Type;
  bool IsPassword;
  bool IsNewPassword;
  bool SetValueOnFocusLost;
  bool SelectAllOnFocusGained;
};

struct CControlEnum : public CControlText {
  CControlEnum();

  CFuncEnum* FuncEnum;
  bool IsLooping;
  UnnamedEnum DisplayType;
  void Incr();
  void Decr();
  uint EnumIndex;
  const string EnumString;
};

struct CControlSlider : public CControlBase {
  CControlSlider();

  enum class CControlSlider::EAxis {
    X = 0,
    Y = 1,
  };
  CControlSlider::EAxis Axis;
  float Ratio;
  MwId IconIdBar;
  MwId IconIdCursor;
  bool AutoSize;
};

// File extension: 'ControlLayout.Gbx'
struct CControlLayout : public CMwNod {
  CControlLayout();

  UnnamedEnum AlignHorizontal;
  UnnamedEnum AlignVertical;
  float RatioHorizontal;
  float RatioVertical;
  float PaddingHorizontal;
  float PaddingVertical;
};

// File extension: 'ListItem.Gbx'
struct CControlListItem : public CMwNod {
  CControlListItem();

  const string Str1;
  const string Str2;
  const string Str3;
  const string Str4;
  const string Str5;
  const string Str6;
  const string Str7;
  const wstring StrInt1;
  const wstring StrInt2;
  const wstring StrInt3;
  const wstring StrInt4;
  const uint Nat1;
  const uint Nat2;
  const uint Nat3;
  const uint Nat4;
  const uint Nat5;
  const uint Nat6;
  const uint Time1;
  const uint Time2;
  const uint Time3;
  const float Real1;
  const float Real2;
  const float Real3;
  vec2 MapCoordOrigin;
  vec2 MapCoordTarget;
  CMwNod* const Nod1;
  CMwNod* const Nod2;
  CMwNod* const Nod3;
  bool IsSelected;
};

// File extension: 'ControlEffect.Gbx'
struct CControlEffectSimi : public CControlEffect {
  CControlEffectSimi();

  bool IsInterpolated;
  bool Centered;
  UnnamedEnum ColorBlendMode;
  bool IsContinousEffect;
  const uint KeyCount;
  uint CurrentKey;
  void InsertKey();
  void RemoveKey();
  float Time;
  float Rot;
  vec2 Pos;
  vec2 Scale;
  float Depth;
  float Opacity;
  float ColorBlend;
  vec3 Color;
};

// File extension: 'ControlEffect.Gbx'
struct CControlEffectMotion : public CControlEffect {
  CControlEffectMotion();

  CPlugParticleEmitterModel* ParticleEmitterModel;
  MwId ParticleEmitterId;
  float Period;
  float EnveloppePosStart;
  float EnveloppePosCharged;
  float IntensityStart;
  float IntensityCharged;
  float SpeedStart;
  float SpeedCharged;
  vec3 SpeedDirection;
  float ChargeTime;
  vec3 EmitterPosOffset;
};

struct CControlUiRange : public CControlBase {
  CControlUiRange();

  float BlockOverlapRatio; // Range: 0 - 1
  float BackgroundMargin; // Range: 0 - 1
  bool DrawBackground;
  bool DrawBlockBackground;
  MwId IconId;
  bool CenteredBar;
  vec3 Color;
  float ColorAlpha; // Range: 0 - 1
  vec3 Color2;
  float Color2Alpha; // Range: 0 - 1
  float Ratio; // Range: 0 - 1
  uint UnitMax;
  uint Unit1;
  uint Unit2;
  float GradingRatio;
  float Scaling; // Range: 0.1 - 10
  bool LastGradingIsGhostlike;
  void CreateTranslate(vec3 StartPoint, vec3 EndPoint);
  void CreateRotate(vec3 Axis, float MinAngle, float MaxAngle);
};

struct CControlGrid : public CControlContainer {
  CControlGrid();

  MwFastArray<vec2> ChildsSquares;
  MwFastArray<string> ChildsSquaresParam;
  CControlLayout* MainLayout;
  bool PackEmptyRows;
  float ForceColumnsUniformWidth;
  float ForceRowsUniformHeight;
  MwFastBuffer<float> ForceColumnsWidths;
  float HorizontalSkewOffset;
  float VerticalSkewOffset;
  bool PagedGrid_Enable;
  uint PagedGrid_NbLinesPerPage;
  uint PagedGrid_FastPageStep;
  uint CurrentPage;
  uint PageCount;
  void OnNextPage();
  void OnPrevPage();
  void OnFastPrevPage();
  void OnFastNextPage();
  void OnFirstPage();
  void OnLastPage();
};

// File extension: 'Frame.Gbx'
struct CControlFrame : public CControlContainer {
  CControlFrame();

  MwFastArray<iso4> ChildsRelativeLocations;
};

// File extension: 'ControlStyle.Gbx'
struct CControlStyle : public CPlug {
  CControlStyle();

  CPlugFont* Font;
  float FontHeight;
  float FontRatioXY;
  UnnamedEnum LabelColorFromPalette;
  vec3 LabelColor;
  float LabelColorAlpha; // Range: 0 - 1
  bool LabelForceEmbossed;
  bool LabelDisableSqueeze;
  wstring LabelCharAttributes;
  UnnamedEnum EditableColorFromPalette;
  vec3 EditableColor;
  float EditableColorAlpha; // Range: 0 - 1
  bool EditableForceEmbossed;
  bool EditableDisableSqueeze;
  wstring EditableCharAttributes;
  UnnamedEnum GrayedColorFromPalette;
  vec3 GrayedColor;
  float GrayedColorAlpha; // Range: 0 - 1
  bool GrayedForceEmbossed;
  bool GrayedDisableSqueeze;
  wstring GrayedCharAttributes;
  bool FitTextSize;
  float Skew; // Range: -1 - 1
  CPlugShader* DefaultShader;
  CPlugSound* FocusSound;
  CPlugSound* ActionSound;
  CPlugSound* ActionBackSound;
  CFuncEnum* ButtonDefaultIcons;
  MwId ButtonDefaultIconId;
  float ButtonIconWidth;
  float ButtonIconHeight;
  CPlugShader* EnumSound;
  CPlugShader* EnumListShader;
  uint EnumMaxElemCount;
  float EnumIconWidth;
  float EnumIconHeight;
  UnnamedEnum EnumForceDisplayType;
  CFuncEnum* EnumForceIcons;
  float QuadZ;
  bool QuadIsLines;
  bool QuadIsFill;
  float QuadZLines;
  vec3 QuadGradientColor0;
  float QuadGradientColor0Alpha; // Range: 0 - 1
  vec3 QuadGradientColor1;
  float QuadGradientColor1Alpha; // Range: 0 - 1
  vec3 LineGradientColor0;
  float LineGradientColor0Alpha; // Range: 0 - 1
  vec3 LineGradientColor1;
  float LineGradientColor1Alpha; // Range: 0 - 1
  vec3 QuadLinesColor;
  float QuadLinesColorAlpha; // Range: 0 - 1
  vec2 Quad_UvTopLeft;
  vec2 Quad_UvBottomRight;
  float SliderBarWidth;
  float SliderBarHeight;
  float SliderCursorWidth;
  float SliderCursorHeight;
  CFuncEnum* SliderBarIcons;
  CFuncEnum* SliderCursorIcons;
  CPlugSound* SliderSound;
  bool FocusAreaEnable;
  CPlugMaterial* FocusAreaMaterial;
  CPlugMaterial* FocusAreaMaterialReadOnly;
  CPlugMaterial* FocusAreaMaterialSelected;
  CPlugMaterial* FocusAreaMaterialFocused;
  float FocusAreaMinWidth;
  float FocusAreaMinHeight;
  float FocusAreaXMargin;
  float FocusAreaYMargin;
  float FocusAreaZOffset;
  CPlugSolid* FocusAreaSolid;
  CControlEffectMaster* EffectMaster;
};

// File extension: 'ControlUrlLinks.Gbx'
struct CControlUrlLinks : public CControlBase {
  CControlUrlLinks();

  void ForceDirty();
  uint CurFocusedLink;
};

struct CControlQuad : public CControlBase {
  CControlQuad();

  bool IsLines;
  bool IsFill;
  UnnamedEnum GradientDir;
  MwId IconId;
  vec3 IconVertexColors;
  float IconVertexAlpha; // Range: 0 - 1
};

// File extension: 'ControlEffectMaster.Gbx'
struct CControlEffectMaster : public CMwNod {
  CControlEffectMaster();

  CControlEffect* FocusEffect;
  CControlEffect* FocusGainedEffect;
  CControlEffect* FocusLostEffect;
  CControlEffect* FocusGainedByAnotherEffect;
  CControlEffect* FocusLostByAnotherEffect;
  CControlEffect* SleepingEffect;
  CControlEffect* ShowingEffect;
  CControlEffect* HidingEffect;
  CControlEffect* ActionEffect;
  CControlEffect* ManagedEffect;
  bool UseRefBBox;
  bool ShowActivated;
  bool HideActivated;
  CMwRefBuffer* SpecialEffect;
};

struct CControlColorChooser : public CControlFrame {
  CControlColorChooser();

  UnnamedEnum StyleType;
  vec2 ColorChooserSize;
  vec3 Color;
  const float Hue; // Range: 0 - 1
  CPlugShader* ColorChooserShader;
};

struct CControlColorChooser2 : public CControlFrame {
  CControlColorChooser2();

  vec3 Color;
  CControlColorChooser* const ColorChooserHue;
  CControlColorChooser* const ColorChooserSV;
  wstring HexaString;
  wstring R;
  wstring G;
  wstring B;
};

struct CControlSimi2 : public CMwNod {
  CControlSimi2();

  float PosX;
  float PosY;
  float Rot;
  float ScaleX;
  float ScaleY;
};

struct CControlTimeLine2 : public CControlBase {
  CControlTimeLine2();

  uint TrackCount;
  uint TrackStart;
  float TimeMin;
  float TimeMax;
  float RulerLength;
  float Width;
  float Height;
  float TrackHeight;
  float BlockHeight;
  float RulerHeight;
  float TimeCursorWidth;
  float TimeCursorHeight;
  float KeyHeight;
  float KeyWidth;
  float KeyHighLightHeight;
  float KeyHighLightWidth;
  vec3 RulerLinesColorBig;
  vec3 RulerLinesColorSmall;
  vec3 TrackSeparationColor;
  vec3 BlockColor;
  vec3 TimeCursorLineColor;
  vec3 BgColor;
  vec3 RulerBgColor;
  vec3 BlockHighLightColor;
  vec3 KeyColor;
  vec3 KeyHighLightColor;
  vec3 KeyFocusColor;
  vec3 DeadZoneColor;
  void CreateSampleData();
  float Time;
};

// File extension: 'ControlEffect.Gbx'
struct CControlEffectCombined : public CControlEffect {
  CControlEffectCombined();

  CMwRefBuffer* Effects;
};

// File extension: 'ControlEffectMoveFrame.Gbx'
struct CControlEffectMoveFrame : public CControlEffect {
  CControlEffectMoveFrame();

  vec2 DefaultShift;
  float Period;
  void AddChildShift();
  void RemoveChildShift();
  bool IsInverse;
  vec3 AngleFrom;
};

// File extension: 'FrameStyled.Gbx'
struct CControlFrameStyled : public CControlFrame {
  CControlFrameStyled();

  CControlStyleSheet* StyleSheet;
};

// File extension: 'StyleSheet.Gbx'
struct CControlStyleSheet : public CPlug {
  CControlStyleSheet();

  CMwRefBuffer* Buffer;
  CControlStyle* MasterStyle;
  CPlugFont* const FallbackFont;
  MwFastBuffer<CMwNod*> Overrides;
  CControlStyle* PagerIconStyle;
  CControlStyle* PagerTextStyle;
  CControlStyle* PagerTextBackgroundStyle;
  MwFastArray<vec3> Colors;
  MwFastArray<float> ColorsAlpha;
};

struct CControlMediaPlayer : public CControlFrame {
  CControlMediaPlayer();

  CPlugFileSnd* const MediaAudio;
  CPlugFileVideo* const MediaVideo;
};

struct CControlGraph : public CControlBase {
  CControlGraph();

  vec2 ValuesMin;
  vec2 ValuesMax;
  uint SampleCount;
  bool SampleKeys;
  MwFastBuffer<CPlugCurveSimpleNod*> Curves;
};

// File extension: 'Pager.Gbx'
struct CControlPager : public CControlFrame {
  CControlPager();

  string StackPageCountText;
  bool UseCounter;
  bool UseCounterQuadBackground;
  bool UseUpDown;
  bool UseFastPrevNext;
  bool UseFirstLast;
  uint FastPrevNextIncrement;
  CControlLabel* LabelPageCounter;
  CControlQuad* QuadPageCounter;
  CControlButton* ButtonPrevPage;
  CControlButton* ButtonNextPage;
  CControlButton* ButtonFastPrevPage;
  CControlButton* ButtonFastNextPage;
  CControlButton* ButtonFirstPage;
  CControlButton* ButtonLastPage;
  void OnPrevPage();
  void OnNextPage();
  void OnFastPrevPage();
  void OnFastNextPage();
  void OnFirstPage();
  void OnLastPage();
};

struct CControlText : public CControlBase {
  float ClipLength;
  int MaxLine;
  UnnamedEnum TextMode;
  CPlugTree* const TextTree;
};

// File extension: 'FrameAnimated.Gbx'
struct CControlFrameAnimated : public CControlFrame {
  CControlFrameAnimated();

  float ScrollVerticalDistance;
  float ScrollHorizontalDistance;
  float ScrollPeriod;
  float ScrollCycleTime;
  bool DoScrolling;
  float ScrolledVerticalDistance;
  float ScrolledHorizontalDistance;
  float ScrollVerticalHistory;
  float ScrollHorizontalHistory;
};

struct CControlScriptEditor : public CControlBase {
  CControlScriptEditor();

  wstring Text;
  bool ShowLineNumbers;
  vec3 ShowLineNuKeywordColormbers;
  vec3 OperatorColor;
  vec3 ConstantColor;
  vec3 CommentColor;
  vec3 ClassNameColor;
  vec3 IdentColor;
  vec3 OthersColor;
  vec3 DirectiveColor;
  vec3 UnexpectedColor;
  string Colors;
};

struct CControlScriptConsole : public CControlBase {
  CControlScriptConsole();

  const wstring HistoryText;
};

struct CControlListCard : public CControlFrame {
  CControlListCard();

  uint NbLinesPerPage;
  uint NbColumnsPerPage;
  uint FastPageStep;
  vec2 GridSize;
  bool Transpose;
  bool AllowMultipleSelection;
  uint CurrentPage;
  uint PageCount;
  void OnNextPage();
  void OnPrevPage();
  void OnFastPrevPage();
  void OnFastNextPage();
  void OnFirstPage();
  void OnLastPage();
  MwFastBuffer<CControlBase*> CardsModels;
  const MwFastBuffer<CControlBase*> ListCards;
};

struct CControlMiniMap : public CControlBase {
};

struct CControlCamera : public CControlBase {
};

} // namespace Control

namespace Plug {

struct CPlugAudio : public CPlug {
};

struct CPlugShader : public CPlug {
};

// File extension: 'Crystal.Gbx'
struct CPlugCrystal : public CPlug {
  CPlugCrystal();

  const uint CrystalVertexCount;
  const uint CrystalEdgeCount;
  const uint CrystalFaceCount;
};

struct CPlugShaderGeneric : public CPlugShader {
};

struct CPlugSolid : public CPlug {
  CPlugSolid();

};

// File extension: 'Visual.Gbx'
struct CPlugVisual : public CPlug {
  void Inverse();
  bool IsGeometryStatic;
  const bool IsGeometryDynaPart;
  bool IsIndexationStatic;
  bool OptimizeInVision;
  bool UseVertexNormal;
  bool UseVertexColor;
  const bool UseUvGroup;
};

struct CPlugMapAINode : public CMwNod {
  CPlugMapAINode();

  NPlugMapAI_SNode* Node;
};

struct CPlugBitmapHighLevel : public CSystemNodWrapper {
  CPlugBitmapHighLevel();

  UnnamedEnum Mode;
  uint BlurTexelCount;
  uint Width;
  uint Height;
  iso4 CameraToWorld;
  float CameraFovY;
  float CameraRatioXY;
  float CameraNearZ;
  float CameraFarZ;
};

struct CPlugVisualIndexedLines : public CPlugVisualIndexed {
  CPlugVisualIndexedLines();

};

struct CPlugVisualOctree : public CMwNod {
};

struct CPlugBitmapRenderShadow : public CPlugBitmapRender {
  CPlugBitmapRenderShadow();

};

// userName: 'Shape'
// File extension: 'Shape.Gbx'
struct CPlugSurface : public CPlug {
  CPlugSurface();

  MwFastBuffer<CPlugMaterial*> Materials;
  MwFastBuffer<GmSurfaceIds> MaterialIds;
  CPlugSkel* Skel;
  void UpdateSurfMaterialIdsFromMaterialIndexs();
  void TransformMaterialsToMatIds();
  GmSurf* m_GmSurf;
};

// File extension: 'PlugModelShading.Gbx'
struct CPlugModelShading : public CMwNod {
  CPlugModelShading();

  CPlugFileImg* DefaultImage_Diffuse;
  CPlugFileImg* DefaultImage_DiffuseOpacity;
  CPlugFileImg* DefaultImage_BaseColor;
  CPlugFileImg* DefaultImage_BaseColorOpacity;
  CPlugFileImg* DefaultImage_Specular;
  CPlugFileImg* DefaultImage_Normal;
  CPlugFileImg* DefaultImage_Energy;
  CPlugFileImg* DefaultImage_TeamMask;
  CPlugFileImg* DefaultImage_SelfIllum;
  CPlugFileImg* DefaultImage_Damage;
  CPlugFileImg* DefaultImage_Dirt;
  CPlugFileImg* DefaultImage_Shield;
  CPlugFileImg* DefaultImage_RoughMetal;
  CPlugFileImg* DefaultImage_AmbientOcclusion;
  CPlugFileImg* DefaultImage_DirtMask;
  CPlugFileImg* DefaultImage_MudMask;
  CPlugFileImg* DefaultImage_CoatRough;
  CPlugFileImg* DefaultImage_Transmittance;
  CPlugFileImg* DefaultImage_DamageMask;
  CPlugFileImg* DefaultImage_SubsurfaceScattering;
  CPlugFileImg* DefaultImage_Height;
  CPlugFileImg* DefaultImage_Opacity;
  CPlugFileImg* DefaultImage_SelfIllumRGB;
  CPlugFileImg* DefaultImage_DecalBaseColorOpacity;
  CPlugFileImg* DefaultImage_DecalRoughMetal;
  CPlugFileImg* DefaultImage_MatterId;
  CPlugFileImg* DefaultImage_DecalOpacity;
  CPlugFileImg* DefaultImage_HueMask;
  CPlugFileImg* DefaultImage_FresnelHue;
  CPlugFileImg* DefaultImage_HueMinMax;
  CPlugMaterial* MaterialDyna0_ZOnly_Water;
  CPlugMaterial* MaterialDyna0_TDSNI;
  CPlugMaterial* MaterialDyna0_TI;
  CPlugMaterial* MaterialDyna0_TI_AddModCV;
  CPlugMaterial* MaterialDyna0_TDSNE;
  CPlugMaterial* MaterialDyna0_TE;
  CPlugMaterial* MaterialDyna0_TIce;
  CPlugMaterial* MaterialDyna0_TShield;
  CPlugMaterial* MaterialStatic_TDSN;
  CPlugMaterial* MaterialStatic_TDOSN;
  CPlugMaterial* MaterialStatic_TDOBSN;
  CPlugMaterial* MaterialStatic_TDSNE;
  CPlugMaterial* MaterialStatic_TDSNI;
  CPlugMaterial* MaterialStatic_TDSNI_Night;
  CPlugMaterial* MaterialStatic_TIAdd;
  CPlugMaterial* MaterialChar_TDSNEM;
  CPlugMaterial* MaterialChar_TE;
  CPlugMaterial* MaterialChar_TDOSNEM;
  CPlugMaterial* MaterialChar_TDOSN;
  CPlugMaterial* MaterialChar_TDOSN2Sided;
  CPlugMaterial* MaterialChar_TDOS;
  CPlugMaterial* MaterialChar_TAniso;
  CPlugMaterial* MaterialChar_Part;
  CPlugMaterial* MaterialChar_Body;
  CPlugMaterial* MaterialFid_Char_Shield;
  CPlugMaterial* MaterialFid_Char_Teleport;
  CPlugMaterial* MaterialFid_Char_ActionMakerOtherFrames;
  CPlugMaterial* MaterialFid_Char_ShowEnergy;
  CPlugLight* LightFid_Char_FakeShadowProj;
  CPlugMaterial* MaterialFid_Vehicle_Shield;
  CPlugMaterial* MaterialFid_Vehicle_Teleport;
  CPlugMaterial* MaterialFid_StaticScreen;
  CPlugMaterial* MaterialFid_StaticScreen_Parallax;
  CMwNod* MaterialRemapFid;
  CPlugBitmap* BitmapFid_CarDecalStadium_BaseColorOp;
  CPlugBitmap* BitmapFid_CarDecalStadium_RoughMetal;
  CPlugMaterial* MaterialVehicle_Skin;
  CPlugMaterial* MaterialVehicle_SkinDmg;
  CPlugMaterial* MaterialVehicle_SkinDmgDecal;
  CPlugMaterial* MaterialVehicle_SkinNoSkelDmg;
  CPlugMaterial* MaterialVehicle_Details;
  CPlugMaterial* MaterialVehicle_DetailsDmg;
  CPlugMaterial* MaterialVehicle_DetailsDmgNormal;
  CPlugMaterial* MaterialVehicle_DetailsDmgDecal;
  CPlugMaterial* MaterialVehicle_DetailsNoSkelDmg;
  CPlugMaterial* MaterialVehicle_Glass;
  CPlugMaterial* MaterialVehicle_GlassDmg;
  CPlugMaterial* MaterialVehicle_GlassDmgCrack;
  CPlugMaterial* MaterialVehicle_GlassDmgDecal;
  CPlugMaterial* MaterialVehicle_GlassNoSkelDmg;
  CPlugMaterial* MaterialVehicle_GlassRefract;
  CPlugMaterial* MaterialVehicle_GlassRefractNoSkelDmg;
  CPlugMaterial* MaterialVehicle_Gems;
  CPlugMaterial* MaterialVehicle_GemsNoSkelDmg;
  CPlugMaterial* MenuBox3dMaterial_MenuBox3dStation;
  CPlugMaterial* MenuBox3dMaterial_MenuBox3dBoxCase;
  CPlugMaterial* MenuBox3dMaterial_MenuBox3dMetal;
  CPlugMaterial* MenuBox3dMaterial_MenuBox3dGlass;
  CPlugMaterial* MenuBox3dMaterial_MenuBox3dLight;
  CPlugMaterial* MenuBox3dMaterial_MenuBox3dTransp;
};

struct CPlugSurfaceGeomDeprecated : public CPlug {
  CPlugSurfaceGeomDeprecated();

};

struct CPlugVisualSprite : public CPlugVisual3D {
  CPlugVisualSprite();

  const bool UseTextureAtlas;
  const bool SpriteIndexTC1;
  CPlugSpriteParam* const SpriteParam;
  CPlugBitmapAtlas* const BitmapAtlas;
};

// File extension: 'Texture.Gbx'
struct CPlugBitmap : public CPlug {
  CPlugBitmap();

  enum class CPlugBitmap::ERenderTech {
    Unknown = 0,
    Tech3 = 1,
  };
  enum class CPlugBitmap::ETexFilter {
    Point = 0,
    Bilinear = 1,
    Trilinear = 2,
    Anisotropic = 3,
    AnisoPoint = 4,
  };
  enum class CPlugBitmap::ETexAddress {
    Wrap = 0,
    Mirror = 1,
    Clamp = 2,
    Border = 3,
  };
  enum class CPlugBitmap::EColorSpace {
    Linear = 0,
    sRGB = 1,
  };
  enum class CPlugBitmap::EVideoTimer {
    Game = 0,
    Human = 1,
    Scene = 3,
    Default = 4,
  };
  enum class CPlugBitmap::EGenerateUV {
    NoGenerate = 0,
    CameraVertex = 1,
    WorldVertex = 2,
    WorldVertexXY = 3,
    WorldVertexXZ = 4,
    WorldVertexYZ = 5,
    CameraNormal = 6,
    WorldNormal = 7,
    CameraReflectionVector = 8,
    WorldReflectionVector = 9,
    WorldNormalNeg = 10,
    WaterReflectionVector = 11,
    Hack1Vertex = 12,
    MapTexel_DEPRECATED = 13, // MapTexel DEPRECATED
    FogPlane0 = 14,
    Vsk3SeaFoam = 15,
    ImageSpace = 16,
    LightDir0Reflect = 17,
    EyeNormal = 18,
    ShadowB1Pw01 = 19,
    Tex3AsPosPrCamera = 20,
    FlatWaterReflect = 21,
    FlatWaterRefract = 22,
    FlatWaterFresnel = 23,
    WorldPosXYblendZY = 24,
    DisableVshOutput = 25,
    WorldPos_PyPxz = 26,
  };
  enum class CPlugBitmap::EUsage {
    Color = 0,
    Light = 1,
    HeightNGDuDv = 2, // Height->DuDv
    Render = 3,
    HNGDuDvLumi = 4, // H->DuDvLumi
    HNGNormXYZ = 5, // H->NormXYZ
    HNGNormXY = 6, // H->NormXY
    DepthCmp = 7,
    DispH01 = 8,
    HNGNormPal8b = 9, // H->NormPal8b
    NormXYZ = 10,
    NormXY = 11,
    NormPal8b = 12,
    NormPal16b = 13,
    ColorFloat = 14,
    RenderFloat = 15,
    HNGDuDv1 = 16, // H->DuDv1
    Alpha = 17,
    LightAlpha = 18,
    HNGNormX0ZY = 19, // H->NormX0ZY
    NormX0ZY = 20,
    TexCoord = 21,
    Render16b = 22,
    Vertex = 23,
    HNGBumpTxTy = 24, // H->BumpTxTy
    BumpTxTy = 25,
    HNGNorm0YZX = 26, // H->Norm0YZX
    Norm0YZX = 27,
    Norm_XYZNG0YZX = 28, // Norm:XYZ->0YZX
    Depth = 29,
    SrgbL8NGLinearL16 = 30, // SrgbL8->LinearL16
    NormATI2N = 31,
    NormXYZNGATI2N = 32, // NormXYZ->ATI2N
    Color16b = 33,
    SpecFIENGFI0E = 34, // SpecFIE->FI0E
    SpecFI0E = 35,
    RoughMetal = 36,
    RoughMetalNGBC1 = 37, // RoughMetal->BC1(Metal,Rough,0)
    Mask_BC4 = 38,
    Staging = 39,
  };
  enum class CPlugBitmap::EColorDepth {
    DefaultColorDepth = 0,
    Color16b = 1,
    Color32b = 2,
  };
  enum class CPlugBitmap::ECubeMapFace {
    None = 0,
    XPos = 1,
    XNeg = 2,
    YPos = 3,
    YNeg = 4,
    ZPos = 5,
    ZNeg = 6,
  };
  enum class CPlugBitmap::EPixelUpdate {
    None = 0,
    Render = 1,
    Shader = 2,
    DynaSpecular = 3,
    Clear = 4,
    RenderVideo = 5,
  };
  enum class CPlugBitmap::EDynamic {
    Off = 0,
    On = 1,
    On_2_Buffers = 2, // On 2 Buffers
    On_GPU_Only = 3, // On GPU Only
  };
  enum class CPlugBitmap::ENormalRotate {
    None = 0,
    Px_Pz_Ny = 1, // +x +z -y
  };
  enum class CPlugBitmap::EEdCustomSaveOp {
    None = 0,
    Conv_Diffuse = 1, // Conv.Diffuse
    CubeHdrScaleA2_DXT5 = 2, // CubeHdrScaleA2 DXT5
    CubeHdr = 3,
    Cube_EquirectExr = 4, // Cube EquirectExr
  };
  enum class CPlugBitmap::ECompressor {
    NVidia = 0,
    DirectX = 1,
  };
  enum class CPlugBitmap::ECompressFile {
    dds = 0,
    webp_Y_BC4 = 1, // webp Y BC4
    webp_YY_BC5 = 2, // webp YY BC5
    webp_YCC_BC4PBC5 = 3, // webp YCC BC4+BC5
  };
  enum class CPlugBitmap::EForceBorderSize {
    _1_texel = 0, // 1 texel
    _2_texels = 1, // 2 texels
    _3_texels = 2, // 3 texels
    _4_texels = 3, // 4 texels
  };
  enum class CPlugBitmap::EMipMapAlpha01 {
    HalfBinary = 0,
    ForceBinary = 1,
    ShadeOfGray = 2,
  };
  enum class CPlugBitmap::ECube2d {
    None = 0,
    PX_NX_PY_NY_PZ_NZ = 1, // +X -X +Y -Y +Z -Z
  };
  CPlugBitmap::ERenderTech RenderTech;
  CPlugBitmap::EUsage Usage;
  CPlugBitmap::EColorDepth WantedColorDepth;
  bool AllowR11G11B10F;
  bool AllowStaticNonPowOf2;
  bool IsOneBitAlpha;
  const bool IgnoreImageAlpha01;
  bool NoShaderSetBlendMap;
  bool ShadowCasterIgnoreAlpha;
  bool ShadowCasterAlphaCut;
  uint ShadowCasterAlphaRef; // Range: 0 - 255
  bool AlphaToCoverage;
  const bool IsNonPow2Conditional;
  const bool IsCubeMap;
  const bool IsOriginTop;
  const bool RGB_IsPreModulatedWith_Alpha;
  CPlugBitmap::ETexFilter TexFilter;
  CPlugBitmap::ETexAddress TexAddressU;
  CPlugBitmap::ETexAddress TexAddressV;
  CPlugBitmap::ETexAddress TexAddressW;
  CPlugBitmap::EColorSpace LdrColorSpace;
  CPlugBitmap::ECube2d Cube2d;
  float MipMapLodBiasDefault;
  float DefaultTexCoordSizeXm;
  float DefaultTexCoordSizeYm;
  float DefaultTexCoordTransXm;
  float DefaultTexCoordTransYm;
  vec2 DefaultTexCoordScale;
  vec2 DefaultTexCoordTrans;
  float DefaultTexCoordRotate; // Range: 0 - 360
  CPlugBitmap::EVideoTimer DefaultVideoTimer;
  const uint DefaultMaxMipLevel;
  float HeightInMeters;
  bool Force1stPixelAlpha0;
  bool ForceBorderRGB;
  vec3 BorderRGB;
  bool ForceBorderAlpha;
  float BorderAlpha; // Range: 0 - 1
  CPlugBitmap::EForceBorderSize ForceBorderSize;
  bool BorderLeft;
  bool BorderRight;
  bool BorderTop;
  bool BorderBottom;
  bool WantMipMapping;
  bool IsMipMapLowerAlphaEnable;
  float MipMapLowerAlpha; // Range: 0 - 1
  MwFastArray<float> MipMapFadeAlphas;
  CPlugBitmap::EMipMapAlpha01 MipMapAlpha01;
  bool CanBeDeletedFromSystemMemory;
  bool RenderTexelsMustPersist;
  bool CanBeCompressedInVideoMemory;
  bool CompressInterpolatedAlpha;
  bool CompressSkipDXT1;
  bool CompressUseDithering;
  CPlugBitmap::ECompressFile CompressFile;
  CPlugBitmap::ECompressor Compressor;
  float BumpScaleFactor;
  float BumpScaleMipLevel;
  CPlugBitmap::ENormalRotate NormalRotate;
  bool NormalAreSigned;
  bool NoMipNormalize;
  CPlugFileImg* Image;
  CPlugBitmapAtlas* Atlas;
  CPlugSpriteParam* const SpriteParam;
  CPlugBitmapDecals* Decals;
  MwId GrassId;
  CPlugFileImg* GrassId_ImageFid;
  float GrassId_TcSizeXm;
  float GrassId_TcSizeYm;
  float GrassId_TcRotate;
  CPlugGrassMatterArray* GrassMatterArray;
  CPlugImageArray* ImageArray;
  string ImageArray_Suffix;
  const MwFastBuffer<CPlugFileImg*> ImageArrayFids;
  bool CanUpscaleSliceInArray;
  NPlugBitmap_SAtlasCubeIn* AtlasCubeIn;
  CPlugBitmap* ArrayView_Bitmap;
  wstring ArrayView_ElemName;
  const uint ArrayView_SlicesStart;
  const uint ArrayView_SlicesCount;
  bool MipLevelSkipFromQuality;
  uint MipLevelSkipCountMax;
  int LDExportSkipMip_Racing;
  void SetOriginTopLeft();
  void SetOriginPlugDefault();
  void SetOriginVisionNative();
  bool FloatRequireFiltering;
  bool RenderAutoFitSize;
  bool RenderAutoFitSS;
  uint RenderSizeMul;
  uint RenderSizeDiv;
  bool RenderRequireBlending;
  uint MultiSampleCount_Resolved;
  bool RenderExplicitMip;
  bool RenderCreateClear;
  vec3 DefaultRenderClearRGB;
  float DefaultRenderClearAlpha; // Range: 0 - 1
  bool DepthUseStencil;
  bool DepthCanSwap;
  bool UseUAV;
  CPlugBitmap::EPixelUpdate PixelUpdate;
  CPlugBitmap::EDynamic Dynamic;
  bool Convert_2d_To_1d;
  vec3 SpecularRGB;
  float SpecularExp; // Range: 1 - 50
  vec3 ClearRGB;
  float ClearAlpha; // Range: 0 - 1
  CPlugBitmapRender* Render;
  CPlugBitmapShader* Shader;
  bool ForceShaderBitmapTc;
  bool ForceShaderGenerateUV;
  CPlugBitmap::EGenerateUV GenerateUV;
};

struct CPlugVisualLines : public CPlugVisual3D {
  CPlugVisualLines();

};

struct CPlugVisualLines2D : public CPlugVisual2D {
  CPlugVisualLines2D();

};

struct CPlugTreeVisualMip : public CPlugTree {
  CPlugTreeVisualMip();

  MwFastArray<float> LevelsFarZ;
  MwFastArray<CPlugTree*> LevelsTree;
};

struct CPlugVisualStrip : public CPlugVisual3D {
  CPlugVisualStrip();

};

struct CPlugVisualVertexs : public CPlugVisual3D {
  CPlugVisualVertexs();

};

struct CPlugVoxelResource : public CMwNod {
  CPlugVoxelResource();

};

struct CPlugFilePack : public CPlugFileFidContainer {
  CPlugFilePack();

  const uint Version;
  const string CreationBuildInfo;
  const string AuthorLogin;
  const wstring AuthorNickName;
  const wstring AuthorZonePath;
  const string InfoManialinkUrl;
  const string DownloadUrl;
  const wstring Comment;
  const string XmlHeader;
  const string TitleId;
  const string UsageSubDir;
};

// File extension: 'Sound.Gbx'
struct CPlugSound : public CPlugAudio {
  CPlugSound();

  enum class CPlugSound::EAudioBalanceGroup {
    Auto = 0,
    Music = 1,
    Menus = 2,
    Ambiance = 3,
    Player = 4,
    Bengs = 5,
    Guns = 6,
    BackingDirect = 7,
    Trails = 8,
    GameUI = 9,
    Custom1 = 10,
    Custom2 = 11,
    OtherPlayers = 12,
    ImpactWarning = 13,
    Environment = 14,
  };
  enum class CPlugSound::EAudioRoomFx {
    None = 0,
    Low = 1,
    Mid = 2,
    High = 3,
    Music = 4,
    UI = 5,
  };
  enum class CPlugSound::EAudioPitchFromDistMode {
    Auto = 0,
    CurvePlayer = 1,
    CurveGun = 2,
    Disabled = 3,
  };
  enum class CPlugSound::EAudioIgnoreSourceProperties {
    None = 0,
    Ignore_All = 1, // Ignore All
    Ignore_Pitch = 2, // Ignore Pitch
    Ignore_Volume = 3, // Ignore Volume
  };
  CPlugFileSnd* PlugFile;
  UnnamedEnum Mode;
  float VolumedB; // Range: -60 - 0
  float Priority;
  int MaxDuplicates;
  MwId GroupDuplicate;
  CPlugSound::EAudioBalanceGroup BalanceGroup;
  float RefDistance;
  float MaxDistance;
  float RolloffFactor; // Range: 0 - 10
  CPlugCurveSimpleNod* VolumeFromDistance;
  CPlugCurveSimpleNod* PitchFromDistance;
  CPlugCurveSimpleNod* VolumeFromSpeedKmh;
  float Pitch; // Range: 0.5 - 2
  float DopplerFactor;
  bool EnableDoppler;
  CPlugSound* BackingSound;
  CPlugSound* FocusedSound;
  float AirAbsorptionFactor; // Range: 0 - 10
  CPlugSound::EAudioRoomFx RoomFxSend;
  float RoomRolloffFactor; // Range: 1 - 10
  float LfeSenddB; // Range: -60 - 0
  float FadeStopDuration;
  float FadePlayDuration;
  bool IsLooping;
  UnnamedEnum SoundKind;
  uint InsideConeAngle; // Range: 0 - 360
  uint OutsideConeAngle; // Range: 0 - 360
  float ConeOutsideAttenuation; // Range: 0 - 1
  float Radius;
  float PanAngleDeg; // Range: -180 - 180
  CPlugSound::EAudioPitchFromDistMode PitchFromDistMode;
  CPlugSound::EAudioIgnoreSourceProperties IgnoreSourceProperties;
  bool IsContinuous;
  bool UseLowPassFilter;
  int DuplicatesIntervalMin;
  void SetDirty();
};

// File extension: 'SoundMood.Gbx'
struct CPlugSoundMood : public CPlugSound {
  CPlugSoundMood();

  CPlugFileSnd* InsideFileSnd;
  const MwFastBuffer<CPlugFileSnd*> EventSounds;
  const MwFastBuffer<float> EventPeriods;
};

// File extension: 'Music.Gbx'
struct CPlugMusic : public CPlugMusicType {
  CPlugMusic();

  CSystemFidsFolder* RootFolder;
  CPlugFileText* Settings;
};

// userName: 'Light'
// File extension: 'Light.Gbx'
struct CPlugLight : public CPlug {
  CPlugLight();

  GxLight* m_GxLightModel;
  CFuncLight* m_FuncLight;
  CPlugBitmap* m_BitmapFlare;
  CPlugBitmap* m_BitmapProjector;
  CPlugFileImg* m_ImageAnim;
  CPlugMaterialColorTargetTable* m_ColorTargetTable;
  const vec3 m_DualCenterToLight;
  MwId AnimTimerName;
  float AnimPeriodMin;
  float AnimPeriodMax;
  bool NightOnly;
  bool ReflectByGround;
  bool DuplicateGxLight;
  bool SceneLightOnlyWhenTreeVisible;
  bool SceneLightAlwaysActive;
};

struct CPlugVisualIndexedTriangles : public CPlugVisualIndexed {
  CPlugVisualIndexedTriangles();

};

struct CPlugFile : public CPlug {
  void ReGenerate();
};

struct CPlugBitmapRenderLightFromMap : public CPlugBitmapRender {
  CPlugBitmapRenderLightFromMap();

  uint SampledWidthPerObject;
  uint ObjectCountPerAxisMin;
  uint ObjectCountPerAxisMax;
  const uint ObjectCountPerAxisVision;
  float CameraNearZ_FactorInObject; // Range: 0 - 1
  float CameraFarZ_ToAdd;
  float StartFadeToWhite; // Range: 0 - 1
  float RemapMin_Night; // Range: -0.5 - 1.5
  float RemapMax_Night; // Range: -0.5 - 1.5
  float RemapMin_DayAmb; // Range: -0.5 - 1.5
  float RemapMax_DayAmb; // Range: -0.5 - 1.5
  float RemapMin_DayDir; // Range: -0.5 - 1.5
  float RemapMax_DayDir; // Range: -0.5 - 1.5
  float CameraDovWorldY_MaxDot; // Range: -1 - 1
  CPlugBitmap* BitmapLightHistory;
  uint RayCastHalfTimeMs;
};

struct CPlugFileJpg : public CPlugFileImg {
  CPlugFileJpg();

};

struct CPlugFileTga : public CPlugFileImg {
  CPlugFileTga();

};

struct CPlugFileDds : public CPlugFileImg {
  CPlugFileDds();

};

struct CPlugFileImg : public CPlugFile {
  enum class CPlugFileImg::EDimension {
    _1D = 0, // 1D
    _2D = 1, // 2D
    _3D = 2, // 3D
    Cube = 3,
  };
  const CPlugFileImg::EDimension Dimension;
  const uint Width;
  const uint Height;
  const uint Depth;
  const uint ArraySize;
  const uint NbComp;
  enum class CPlugFileImg::EPlugImageFormat {
    BGRA = 0,
    BC1_DXT1 = 1,
    BC2_DXT2_PM = 2,
    BC2_DXT3 = 3,
    BC3_DXT4_PM = 4,
    BC3_DXT5 = 5,
    RGBA16 = 6,
    RGBA16F = 7,
    RGBA32F = 8,
    FileJpg = 9,
    BC4 = 10, // BC4 (ATI1N)
    BC5yx = 11, // BC5yx (ATI2N)
    BC6U = 12,
    BC6F = 13,
    BC7 = 14,
    R11G11B10F = 15,
    RGBA8 = 16,
    R10G10B10A2 = 17,
    BC5 = 18,
  };
  const CPlugFileImg::EPlugImageFormat Format;
  enum class CPlugFileImg::EPlugChannelType {
    Auto = 0,
    UNorm = 1,
    SNorm = 2,
    UInt = 3,
    SInt = 4,
    sRGB = 5,
    Float = 6,
  };
  const CPlugFileImg::EPlugChannelType ChannelType;
  const uint cMipLevel;
  const uint cMipLevelSkipAtLoad;
  const bool OriginIsTop;
  const bool IsInSystemMemory;
  const uint SystemKb;
  void ScaleToPowerOfTwo(bool Shrink, uint TexFilter);
  const float LdrToHdrScale;
  const float HRoughnessToMipPower;
  const UnnamedEnum MipMapFilter;
  const bool IsSizePowerOf2;
};

struct CPlugShaderApply : public CPlugShaderGeneric {
  CPlugShaderApply();

};

struct CPlugVisualQuads : public CPlugVisual3D {
  CPlugVisualQuads();

};

struct CPlugVisualTriangles : public CPlugVisual3D {
  CPlugVisualTriangles();

};

// userName: 'SoundEngine2'
// File extension: 'SoundEngine.Gbx'
struct CPlugSoundEngine2 : public CPlugSound {
  CPlugSoundEngine2();

  CPlugFileAudioMotors* AudioMotors_Exhaust_Throttle;
  CPlugFileAudioMotors* AudioMotors_Exhaust_Release;
  CPlugFileAudioMotors* AudioMotors_Engine_Throttle;
  CPlugFileAudioMotors* AudioMotors_Engine_Release;
  CPlugFileSnd* AudioMotors_IdleLoop_Exhaust;
  CPlugFileSnd* AudioMotors_LimiterLoop_Exhaust;
  CPlugFileSnd* AudioMotors_IdleLoop_Engine;
  CPlugFileSnd* AudioMotors_LimiterLoop_Engine;
  NPlugCurve_SSimpleCurveInPlace7 AudioMotors_PitchRandomize_Rpm;
  NPlugCurve_SSimpleCurveInPlace7 AudioMotors_PitchRandomize_Rpm;
  NPlugCurve_SSimpleCurveInPlace7 AudioMotors_PitchRandomize_Throttle;
  float AudioMotors_IdleVolume_dB_;
  float AudioMotors_LimiterVolume_dB_;
  CSystemFidsFolder* Loops_Folder;
  bool Loops_UseVolumeCorrection;
  bool Loops_RpmClamp;
  float RpmMaxFromEngine;
  float RpmGamma;
  NPlugCurve_SSimpleCurveInPlace7 Volume_Speed;
  NPlugCurve_SSimpleCurveInPlace7 Volume_Distance;
  NPlugCurve_SSimpleCurveInPlace7 Volume_Rpm;
  NPlugCurve_SSimpleCurveInPlace7 Volume_Throttle;
  NPlugCurve_SSimpleCurveInPlace7 Volume_Gear;
  NPlugCurve_SSimpleCurveInPlace7 RpmFactor_Gear;
  NPlugCurve_SSimpleCurveInPlace7 VolPersp_Rpm_Exhaust;
  NPlugCurve_SSimpleCurveInPlace7 VolPersp_Throttle_Exhaust;
  NPlugCurve_SSimpleCurveInPlace7 VolPersp_Rpm_Engine;
  NPlugCurve_SSimpleCurveInPlace7 VolPersp_Throttle_Engine;
  NPlugCurve_SSimpleCurveInPlace7 VolPersp_Rpm_Interior;
  NPlugCurve_SSimpleCurveInPlace7 VolPersp_Throttle_Interior;
  float Filters_CutoffRatio_Exhaust; // Range: 0 - 1
  float Filters_CutoffRatio_Engine; // Range: 0 - 1
  EAudioFilterType Filters_Type_Exhaust;
  EAudioFilterType Filters_Type_Engine;
  float Mix_FrontBackSpread;
  float Mix_Front____of_engine_; // Range: 0 - 1
  float Mix_Back____of_engine_;
  void UpdateSubEmitters();
  float SubEmitter_Vol_Exhaust;
  float SubEmitter_RelSize_Exhaust;
  vec3 SubEmitter_RelPos_Exhaust;
  vec3 SubEmitter_RelDir_Exhaust;
  float SubEmitter_ConeAttenuation_Exhaust;
  float SubEmitter_ConeAngle_Exhaust;
  float SubEmitter_Vol_Engine;
  float SubEmitter_RelSize_Engine;
  vec3 SubEmitter_RelPos_Engine;
  vec3 SubEmitter_RelDir_Engine;
  float SubEmitter_ConeAttenuation_Engine;
  float SubEmitter_ConeAngle_Engine;
  float SubEmitter_Vol_Interior;
  float SubEmitter_RelSize_Interior;
  vec3 SubEmitter_RelPos_Interior;
  vec3 SubEmitter_RelDir_Interior;
  float SubEmitter_ConeAttenuation_Int;
  float SubEmitter_ConeAngle_Int;
  void _Loops_LoadAndParseFileSnd_();
  float Loops_MinRpmAll;
  float Loops_MaxRpmAll;
};

struct CPlugVisualIndexedStrip : public CPlugVisualIndexed {
  CPlugVisualIndexedStrip();

};

struct CPlug : public CMwNod {
  CPlug();

};

struct CPlugVisual3D : public CPlugVisual {
  bool UseTgtU;
  bool UseTgtV;
  CPlugBlendShapes* const BlendShapes;
  void NegNormals();
  void ComputeFaceCull();
  void ComputeOccBox();
};

struct CPlugFileFont : public CPlugFile {
  CPlugFileFont();

};

struct CPlugFileGen : public CPlugFileImg {
  CPlugFileGen();

  enum class CPlugFileGen::EGenKind {
    Checker = 0,
    LightMap = 1,
    Plain = 2,
    Point = 3,
    Shade = 4,
    Render = 5,
    RenderCube = 6,
    CubeNormals = 7,
    Identity = 8,
    Pixels = 9,
    Depth = 10,
    DepthCube = 11,
    RenderF = 12,
    Iradiance = 13,
    Specular = 14,
    TestNormal = 15,
    RandNormal = 16,
    SpecularCube = 17,
    HemiReflec = 18,
    CubeInvHemiReflec = 19,
    SpecularsLA = 20,
    HueGradient = 21,
    SLGradient = 22,
    VolumeRotate = 23,
    SpecularCubeVect = 24,
    TestMipMap = 25,
    SpecCubeVectRgb = 26,
    DdxInMipMap = 27,
    RandVolume = 28,
    Unalloc = 29,
    PotentialField = 30,
    RenderCubeF = 31,
    TestAnaglyph = 32,
    ACosSmoothInRange = 33,
    PerlinPerm = 34,
    PerlinPerm2D = 35,
    PerlinGrad3 = 36,
    PerlinGrad3OfPerm = 37,
    xxx = 38,
    Cube3x2_2d1RotCube = 39,
    Cube3x2_Ya1 = 40,
    ShClampedCos = 41,
    Random = 42,
    HueSaturation = 43,
    SmEnergyHueIntens = 44,
    PerlinGrad3OfPerm2d = 45,
    PerlinGrad4OfPerm2d = 46,
    Simplex4 = 47,
    InvalidData = 48,
    Callback = 49,
    Noise = 50,
    TransmittanceAccum = 51,
    Power = 52,
    HueMinMaxFromRGB = 53,
  };
  const CPlugFileGen::EGenKind GenKind;
  const MwFastBuffer<CPlugFileImg*> ArrayOfImages;
  const MwFastBuffer<CPlugFileImg*> GenImages;
};

struct CPlugFileSnd : public CPlugFile {
};

struct CPlugFileWav : public CPlugFileSnd {
  CPlugFileWav();

};

// File extension: 'AudioBalance.Gbx'
struct CPlugAudioBalance : public CPlugAudio {
  CPlugAudioBalance();

  bool IsAbsolute;
  float AttackDuration;
  float MinimumDuration;
  float ReleaseDuration;
  void SetDefaultValues();
  void ApplySceneVolume();
  float SceneVolumedB; // Range: -60 - 6
  float UiVolumedB; // Range: -60 - 6
  float SceneLfeSenddB; // Range: -60 - 6
  float MusicLfeSenddB; // Range: -60 - 6
  float GlobalLfeSenddB; // Range: -60 - 6
  CPlugCurveSimpleNod* RolloffFromMusicVolume;
  float RolloffFactor; // Range: 0 - 4
  float RefDistanceFactor; // Range: 0 - 4
  float AutoRadiusFactor; // Range: 0 - 1
  float MaxDistFade_Ratio_Others;
  float MaxDistFade_Rolloff_Others;
  CPlugCurveSimpleNod* PitchFromDistance_Player_Others;
  CPlugCurveSimpleNod* PitchFromDistance_BengsGuns_Others;
  float MaxDistFade_Ratio_Focused;
  float MaxDistFade_Rolloff_Focused;
  CPlugCurveSimpleNod* PitchFromDistance_Player_Focused;
  CPlugCurveSimpleNod* PitchFromDistance_BengsGuns_Focused;
};

struct CPlugFileFidContainer : public CPlugFile {
  CSystemFidsFolder* const Location;
  void UiInstallFidsInSubFolder();
  void UiInstallFids();
  void UninstallAndDeleteFids();
  void EdDumpStatistics();
  void DumpContents();
  const uint NbFolders;
  const uint NbFiles;
  void DbgBenchExpand();
  void DbgExpandToDisk();
};

// File extension: 'TexturePack.Gbx'
struct CPlugBitmapPacker : public CPlug {
  CPlugBitmapPacker();

  uint BitmapSizeMax;
  const MwFastBuffer<CPlugBitmapPack*> Packs;
  CSystemFidsFolder* FidsBrowseMaterials;
  CSystemFidsFolder* FidsBrowseSolids;
  void FindPackListFromPath();
  void PackBitmaps();
  void AddPackInput();
  const MwFastBuffer<CPlugBitmapPackInput*> PackInputs;
  void FindTextureTiling();
  void FidParametersPush();
};

struct CPlugMusicType : public CPlugSound {
};

// File extension: 'AudioEnvironment.Gbx'
struct CPlugAudioEnvironment : public CPlugAudio {
  CPlugAudioEnvironment();

  float DopplerFactor; // Range: 0 - 10
  float RoomFxLowGain; // Range: -60 - 0
  float RoomFxLowGainHF; // Range: -60 - 0
  float RoomFxMidGain; // Range: -60 - 0
  float RoomFxMidGainHF; // Range: -60 - 0
  float RoomFxHighGain; // Range: -60 - 0
  float RoomFxHighGainHF; // Range: -60 - 0
  float RoomFxUIGain; // Range: -60 - 0
  float RoomFxUIGainHF; // Range: -60 - 0
  float RoomFxMusicGain; // Range: -60 - 0
  float RoomFxMusicGainHF; // Range: -60 - 0
  float LowPassGain; // Range: -60 - 0
  float LowPassGainHF; // Range: -60 - 0
  const float Gain;
  const float ReflectionsGain;
  const float LateReverbGain;
};

struct CPlugMaterialCustom : public CPlug {
  CPlugMaterialCustom();

};

struct CPlugVisualGrid : public CPlugVisual3D {
  CPlugVisualGrid();

  uint NbPointX;
  uint NbPointZ;
  float RangeX;
  float RangeZ;
  void Courbificateur();
  float Courbifiant;
  float Courbifiant2;
};

struct CPlugFilePng : public CPlugFileImg {
  CPlugFilePng();

};

struct CPlugBlendShapes : public CPlug {
  CPlugBlendShapes();

  MwFastArray<float> BlendVals;
  bool NormalizeNormals;
  bool BlendNormals;
};

// File extension: 'Text.Gbx'
struct CPlugTreeGenText : public CPlugTreeGenerator {
  CPlugTreeGenText();

  wstring Text;
  CPlugFont* Font;
  vec3 Color;
  float Height;
  float RatioXY;
  UnnamedEnum AlignHorizontal;
  UnnamedEnum AlignVertical;
  float ClipLength;
  uint MaxLine;
  uint ClipLineMin;
  uint ClipLineMax;
};

struct CPlugFileGPU : public CPlugFile {
  CPlugFileGPU();

  const UnnamedEnum GpuStage;
  const uint GpuVersionMajor;
  const uint GpuVersionMajorReq;
  const uint CodeSize;
  CSystemFidFile* const GeneratedCombination_Fid0;
  CSystemFidFile* const GeneratedCombination_Fid1;
};

// File extension: 'txt'
struct CPlugFileText : public CPlugFile {
  CPlugFileText();

  string Text;
};

struct CPlugBitmapPack : public CPlug {
  CPlugBitmapPack();

  uint SizeX;
  uint SizeY;
  UnnamedEnum Format;
  uint NbComp;
  enum class CPlugBitmapPack::EGxTexAddress {
    Wrap = 0,
    Mirror = 1,
    Clamp = 2,
    Border = 3,
  };
  CPlugBitmapPack::EGxTexAddress TexAdrU;
  CPlugBitmap* Bitmap;
  const MwFastBuffer<CPlugBitmapPackElem*> PackElems;
  void LoadBitmap();
};

struct CPlugBitmapPackElem : public CPlug {
  CPlugBitmapPackElem();

  CPlugBitmap* FidBitmapSrc;
  uint TexelStartX;
  uint TexelStartY;
  const uint TexelCountX;
  const uint TexelCountY;
};

struct CPlugBitmapAddress : public CPlugBitmapSampler {
  CPlugBitmapAddress();

  bool UseBitmapTcScale;
  bool DirectTransfo;
  enum class CPlugBitmapAddress::EGxUVGenerate {
    NoGenerate = 0,
    CameraVertex = 1,
    WorldVertex = 2,
    WorldVertexXY = 3,
    WorldVertexXZ = 4,
    WorldVertexYZ = 5,
    CameraNormal = 6,
    WorldNormal = 7,
    CameraReflectionVector = 8,
    WorldReflectionVector = 9,
    WorldNormalNeg = 10,
    WaterReflectionVector = 11,
    Hack1Vertex = 12,
    MapTexel_DEPRECATED = 13, // MapTexel DEPRECATED
    FogPlane0 = 14,
    Vsk3SeaFoam = 15,
    ImageSpace = 16,
    LightDir0Reflect = 17,
    EyeNormal = 18,
    ShadowB1Pw01 = 19,
    Tex3AsPosPrCamera = 20,
    FlatWaterReflect = 21,
    FlatWaterRefract = 22,
    FlatWaterFresnel = 23,
    WorldPosXYblendZY = 24,
    DisableVshOutput = 25,
    WorldPos_PyPxz = 26,
  };
  CPlugBitmapAddress::EGxUVGenerate GenerateUV;
  bool UVTransfoIso3;
  bool UVTransfoMat4;
  uint TexCoordIndex;
  iso3 TransfoIso3;
  vec4 TransfoMat4U;
  vec4 TransfoMat4V;
  vec4 TransfoMat4W;
};

struct CPlugBitmapPackInput : public CPlug {
  CPlugBitmapPackInput();

  uint LayerCount;
  const MwFastBuffer<CPlugBitmap*> FidBitmaps;
};

// File extension: 'ScanCache.Gbx'
struct CPlugFileFidCache : public CPlugFileFidContainer {
  CPlugFileFidCache();

  CSystemFidsFolder* FidsToCreateCacheFrom;
  uint Version;
  const wstring RootEnumFullName;
};

struct CPlugVisual2D : public CPlugVisual {
};

struct CPlugVisualQuads2D : public CPlugVisual2D {
  CPlugVisualQuads2D();

};

// File extension: 'Font.Gbx'
struct CPlugFont : public CPlug {
};

// File extension: 'Font.Gbx'
struct CPlugFontBitmap : public CPlugFont {
  CPlugFontBitmap();

  CPlugFontBitmap* FallbackFont;
  uint NbPages;
  MwFastArray<CMwNod*> PageShaders;
  uint FontHeight;
  uint FontAscent;
  vec2 BBoxCapitalMin;
  vec2 BBoxCapitalMax;
  void LoadAllPages();
  void CreateCharRemap();
  const uint NbCharRemapPages;
  CPlugMaterial* CustomMaterialModel;
  const MwFastArray<CMwNod*> PageTextureFids;
  MwFastArray<wstring> PageTextureDataRefs;
};

struct CPlugTree : public CPlug {
  CPlugTree();

  const MwFastBuffer<CPlugTree*> Childs;
  bool IsVisible;
  bool IsCollidable;
  bool IsRooted;
  bool IsLightVolume;
  bool IsLightVolumeVisible;
  bool UseLocation;
  bool IsShadowCaster;
  bool IsFixedRatio2D;
  bool IsPickable;
  bool IsPickableVisual;
  bool IsPortal;
  const bool HasModel;
  bool TestBBoxVisibility;
  const bool IsCustomBBox;
  bool UseRenderBefore;
  iso4 Location;
  vec3 Translation;
  CPlugVisual* Visual;
  uint SubVisualIndex1;
  uint SubVisualIndex2;
  float SubVisualIndexB; // Range: 0 - 1
  uint SplitVisualIndex;
  uint SplitVisualCount;
  CPlugTreeGenerator* Generator;
  CPlugMaterial* Material;
  CPlug* Shader;
  CPlugSurface* Surface;
  void UpdateBBox();
  void Generate();
  CFuncTree* FuncTree;
  const vec3 BoundingBoxCenter;
  const vec3 BoundingBoxHalfDiag;
  const vec3 BoundingBoxMin;
  const vec3 BoundingBoxMax;
};

struct CPlugTreeGenerator : public CPlug {
  bool IsSaveGenerated;
  bool IsToKeepInSaveAsRelease;
};

// userName: 'SoundGauge'
// File extension: 'SoundGauge.Gbx'
struct CPlugSoundGauge : public CPlugSound {
  CPlugSoundGauge();

  CPlugFileSnd* FileSndFilling;
  CPlugFileSnd* FileSndEmptying;
  CPlugFileSnd* FileSndFillingFast;
  CPlugFileSnd* FileSndEmptyingFast;
  CPlugFileSnd* FileSndOnStopFilling;
  CPlugFileSnd* FileSndOnStopEmptying;
  CPlugFileSnd* FileSndOnStartFilling;
  CPlugFileSnd* FileSndOnStartEmptying;
  CPlugFileSnd* FileSndOnFull;
  CPlugFileSnd* FileSndOnEmpty;
  CPlugFileSnd* FileSndOnGradFull;
  CPlugFileSnd* FileSndOnGradEmpty;
  float FastRate;
  float FadeInDuration;
  float FadeOutDuration;
  bool RestartLoopOnChange;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRateFilling;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRateEmptying;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRateOnStopFilling;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRateOnStopEmptying;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRateOnStartFilling;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRateOnStartEmptying;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRateOnFullOrEmpty;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRateOnGradFull;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRateOnGradEmpty;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRatioFilling;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRatioEmptying;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRatioOnStopFilling;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRatioOnStopEmptying;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRatioOnStartFilling;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRatioOnStartEmptying;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRatioOnFullOrEmpty;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRatioOnGradFull;
  NPlugCurve_SSimpleCurveInPlace7 PitchFromRatioOnGradEmpty;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRateFilling;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRateEmptying;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRateOnStopFilling;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRateOnStopEmptying;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRateOnStartFilling;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRateOnStartEmptying;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRateOnFullOrEmpty;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRateOnGradFull;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRateOnGradEmpty;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRatioFilling;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRatioEmptying;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRatioOnStopFilling;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRatioOnStopEmptying;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRatioOnStartFilling;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRatioOnStartEmptying;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRatioOnFullOrEmpty;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRatioOnGradFull;
  NPlugCurve_SSimpleCurveInPlace7 VolumeFromRatioOnGradEmpty;
};

// File extension: 'GpuCache.Gbx'
struct CPlugGpuCompileCache : public CPlug {
  CPlugGpuCompileCache();

};

// File extension: 'Script.txt'
struct CPlugFileTextScript : public CPlugFileText {
  CPlugFileTextScript();

};

struct CPlugFileI18n : public CPlugFile {
  CPlugFileI18n();

};

struct CPlugVertexStream : public CPlug {
  CPlugVertexStream();

  bool IsStatic;
  bool SkipVision;
  bool IsDirtyVision;
};

struct CPlugIndexBuffer : public CPlug {
  CPlugIndexBuffer();

  const bool IsStatic;
  const UnnamedEnum IndexType;
  const uint IndexCount;
};

struct CPlugBitmapRenderHemisphere : public CPlugBitmapRender {
  CPlugBitmapRenderHemisphere();

  float SpecularPower0;
  float SpecularPower1;
  bool AutoScaleWithPower;
  bool m_UseLightDir;
  UnnamedEnum HemiLayout;
};

struct CPlugFileOggVorbis : public CPlugFileSnd {
  CPlugFileOggVorbis();

};

struct CPlugBitmapRenderPortal : public CPlugBitmapRender {
  CPlugBitmapRenderPortal();

  float DepthFadeOutStart;
  float DepthFadeOutEnd;
};

struct CPlugBitmapRenderPlaneR : public CPlugBitmapRender {
  CPlugBitmapRenderPlaneR();

  UnnamedEnum Plane;
  bool IsPlaneEqValid;
  string TreeId;
  vec4 PlaneEq;
};

struct CPlugSimuDump : public CMwNod {
  CPlugSimuDump();

};

// File extension: 'SoundSurface.Gbx'
struct CPlugSoundSurface : public CPlugSound {
  CPlugSoundSurface();

  float EdMaxSpeedKmh;
  MwFastArray<CMwNod*> Texture;
  MwFastArray<CMwNod*> Skid;
  CPlugCurveSimpleNod* const SkidVolumeFromIntensity;
};

struct CPlugFileBink : public CPlugFile {
  CPlugFileBink();

};

struct CPlugFileVideo : public CPlugFileImg {
  void Play();
  void Pause();
  void Stop();
  void Rewind();
  UnnamedEnum TimeMode;
  bool IsLooping;
  const bool HasSound;
};

struct CPlugLocatedSound : public CMwNod {
  CPlugLocatedSound();

  CPlugSound* Sound;
  iso4 Loc;
};

struct CPlugTreeLight : public CPlugTree {
  CPlugTreeLight();

  CPlugLight* PlugLight;
  GxLight* const Light;
};

// File extension: 'Sound.Gbx'
struct CPlugSoundMulti : public CPlugSound {
  CPlugSoundMulti();

  enum class CPlugSoundMulti::ESoundInputMapping {
    Direct = 0,
    ForceRandom = 1,
    Distance = 2,
    Scale = 3,
  };
  MwFastBuffer<CPlugFileSnd*> AdditionalSounds;
  CPlugSoundMulti::ESoundInputMapping InputMapping;
  MwFastBuffer<vec3> PreferedDistances;
  CPlugCurveSimpleNod* VolumeFromInput;
  bool AvoidDuplicates;
  bool AlternateParity;
  float PitchVarianceNeg; // Range: 0 - 1
  float PitchVariancePos; // Range: 0 - 1
  float VolumeVariance; // Range: 0 - 1
};

struct CPlugSoundVideo : public CPlugSound {
  CPlugSoundVideo();

  CPlugFileVideo* const Video;
  CPlugFileImg* const ImageNonVideo;
};

// File extension: 'PointsInSphere.Gbx'
struct CPlugPointsInSphereOpt : public CPlug {
  CPlugPointsInSphereOpt();

};

struct CPlugShaderPass : public CPlug {
  CPlugShaderPass();

  const uint iPass;
  enum class CPlugShaderPass::EGxBlendFactor {
    _0 = 0, // 0
    _1 = 1, // 1
    SrcColor = 2,
    _1NSrcColor = 3, // 1-SrcColor
    SrcAlpha = 4,
    _1NSrcAlpha = 5, // 1-SrcAlpha
    DstColor = 6,
    _1NDstColor = 7, // 1-DstColor
    DstAlpha = 8,
    _1NDstAlpha = 9, // 1-DstAlpha
    SrcAlphaSat = 10,
    Constant = 11,
    _1NConstant = 12, // 1-Constant
    Src1Color = 13,
    _1NSrc1Color = 14, // 1-Src1Color
    Src1Alpha = 15,
    InvSrc1Alpha = 16,
  };
  CPlugShaderPass::EGxBlendFactor BlendSrc;
  CPlugShaderPass::EGxBlendFactor BlendDst;
  enum class CPlugShaderPass::ECullMode {
    Default = 0,
    Inverse_Culling = 1, // Inverse Culling
    DblSidedLighting = 2,
  };
  CPlugShaderPass::ECullMode CullMode;
  bool WriteZ;
  bool FlatCBufferV;
  bool FlatCBufferP;
  const bool IsValid;
  CPlugFileGPUV* VertexShader;
  CPlugFileGPUP* PixelShader;
  CPlugFileGPU* GeometryShader;
  CPlugFileGPU* HullShader;
  CPlugFileGPU* DomainShader;
  const MwStridedArray<CPlugBitmapSampler*> VertexTextures;
  MwFastArray<MwId> SkipSamplers;
};

struct CPlugVisualIndexed : public CPlugVisual3D {
};

// userName: 'SkelLodSetup'
// File extension: 'SkelLodSetup.Gbx.json'
struct NPlugSkel_SLodSetup : public CMwNod {
  NPlugSkel_SLodSetup();

  MwFastBuffer<NPlugSkel_SJointLodSetup> JointLodSetups;
};

// File extension: '.svg'
struct CPlugFileSvg : public CPlugFile {
  CPlugFileSvg();

};

struct CPlugModelTree : public CMwNod {
  CPlugModelTree();

  const MwFastBuffer<CPlugModelTree*> Childs;
  const MwFastBuffer<CPlugModelLodMesh*> LodMeshes;
  MwFastBuffer<iso4> LodMeshesLocs;
  const MwFastBuffer<CPlugModelMesh*> Surfaces;
  const MwFastBuffer<iso4> SurfaceLocs;
  vec3 RotationPivot;
  vec3 ScalePivot;
  iso4 Location;
  const MwFastBuffer<CMwNod*> ChildGens;
  const MwFastBuffer<iso4> ChildGensLocs;
  const MwFastBuffer<MwId> ChildGensIds;
  const MwFastBuffer<bool> ChildGensDisableSurface;
  bool OptimIsKeepTree;
};

struct CPlugModelMesh : public CMwNod {
  CPlugModelMesh();

  const uint VertexCount;
  const uint InfluenceCount;
  const uint MorphCount;
  const uint FrameCount;
  const uint VertexInfluenceCount;
  const uint PolyCount;
  const uint TriCount;
  const uint QuadCount;
  const bool PolyIsMaterialIndex;
  const bool PolyIsSmoothingGroup;
  const bool PolyIsVertexNormal;
  const bool PolyIsVertexColor;
  const uint PolyVertexUvLayerCount;
  const bool PolyIsVertexTangent;
  bool PolyIsDoubleSide;
  const uint SpriteCount;
  const bool SpriteIsMaterialIndex;
  const bool SpriteIsDiameter;
  const bool SpriteIsColor;
  const bool SpriteIsRotAngle;
  const bool SpriteIsXYRatio;
  const bool SpriteIsTextureAtlas;
  const bool SpriteIsTextureSubId;
  const MwFastBuffer<CMwNod*> Exts;
};

// File extension: 'VHlsl.Txt'
struct CPlugFileVHlsl : public CPlugFileGPUV {
  CPlugFileVHlsl();

};

struct CPlugFileGPUV : public CPlugFileGPU {
};

struct CPlugFileGPUP : public CPlugFileGPU {
};

// File extension: 'PHlsl.Txt'
struct CPlugFilePHlsl : public CPlugFileGPUP {
  CPlugFilePHlsl();

};

// File extension: 'TextureDecals.Gbx'
struct CPlugBitmapDecals : public CPlug {
  CPlugBitmapDecals();

  MwId MatterId;
  const MwFastBuffer<CPlugDecalModel*> DecalModels;
  const uint DecalSetCount;
  uint CellSizeX;
  uint CellSizeY;
  uint CellSizeZ;
};

// File extension: 'Material.Gbx'
struct CPlugMaterial : public CPlug {
  CPlugMaterial();

};

// File extension: 'MaterialFx.Gbx'
struct CPlugMaterialFx : public CPlug {
};

struct CPlugMaterialFxFlags : public CPlugMaterialFx {
  CPlugMaterialFxFlags();

};

// File extension: 'MaterialFx.Gbx'
struct CPlugMaterialFxFur : public CPlugMaterialFx {
  CPlugMaterialFxFur();

  uint ShellCount;
  float ShellThick;
  vec3 ShellLowRGB;
  float ShellLowAlpha;
  vec3 ShellHighRGB;
  float ShellHighAlpha;
  uint FenceCount;
  float FenceHeight;
  CPlugBitmap* FenceBitmap;
};

// File extension: 'MaterialFx.Gbx'
struct CPlugMaterialFxs : public CPlugMaterialFx {
  CPlugMaterialFxs();

  MwFastArray<CPlugMaterialFx*> MaterialFxs;
};

struct CPlugBitmapSampler : public CPlug {
  CPlugBitmapSampler();

  bool IsInternal;
  uint NbCompRequired;
  enum class CPlugBitmapSampler::EAlphaReq {
    None = 0,
    Binary = 1, // Binary(Tested)
    Translucent = 2, // Translucent(Blended)
  };
  CPlugBitmapSampler::EAlphaReq AlphaRequired;
  bool IsSharedByDevices;
  bool SrgbToLinear;
  bool UseBitmapDefaults;
  enum class CPlugBitmapSampler::EGxTexFilter {
    Point = 0,
    Bilinear = 1,
    Trilinear = 2,
    Anisotropic = 3,
    AnisoPoint = 4,
  };
  CPlugBitmapSampler::EGxTexFilter WantedFiltering;
  enum class CPlugBitmapSampler::EGxTexAddress {
    Wrap = 0,
    Mirror = 1,
    Clamp = 2,
    Border = 3,
  };
  CPlugBitmapSampler::EGxTexAddress TexAddressU;
  CPlugBitmapSampler::EGxTexAddress TexAddressV;
  CPlugBitmapSampler::EGxTexAddress TexAddressW;
  vec3 BorderRGB;
  float BorderAlpha; // Range: 0 - 1
  const bool SynchNameWithShader;
  float MipMapLodBias;
  uint MaxMipLevel;
  const uint MaxAnisoRatio;
  CPlugBitmap* Bitmap;
};

struct CPlugBitmapShader : public CPlug {
  CPlugBitmapShader();

  CPlugShader* Shader;
  CPlugBitmap* BitmapToSwap;
};

// File extension: 'MaterialFx.Gbx'
struct CPlugMaterialFxDynaBump : public CPlugMaterialFx {
  CPlugMaterialFxDynaBump();

  bool IsCollidable;
  float SpeedMaxIntens;
  float MaxIntens;
  float Inter1SizeX;
  float Inter1SizeZ;
};

// File extension: 'MaterialFx.Gbx'
struct CPlugMaterialFxDynaMobil : public CPlugMaterialFx {
  CPlugMaterialFxDynaMobil();

};

struct CPlugScriptWithSettings : public CMwNod {
  CPlugScriptWithSettings();

  CPlugFileTextScript* Script;
  const MwFastBuffer<CScriptSetting*> Settings;
  void UpdateSettingsFromScript();
};

// File extension: 'zip'
struct CPlugFileZip : public CPlugFileFidContainer {
  CPlugFileZip();

  bool DisableCrc32Check;
};

struct CPlugFileAudioMotors : public CPlugFile {
  CPlugFileAudioMotors();

};

struct CPlugBitmapRender : public CPlug {
  enum class CPlugBitmapRender::ETrigger {
    None = 0,
    Once = 1,
    EachFrame = 2,
  };
  CPlugBitmapRender::ETrigger TriggerRender;
  uint cFrameToSkip;
  UnnamedEnum RenderPath;
  enum class CPlugBitmapRender::ERenderPathFails {
    Ignore = 0,
    Hide = 1,
  };
  CPlugBitmapRender::ERenderPathFails RenderPathFails;
  bool IVIdMaskReflected;
  bool IVIdMaskReflectMirror;
  bool IVIdMaskRefracted;
  bool IVIdMaskWaterNormalDecals;
  bool IVIdMaskViewDepOcclusion;
  bool IVIdMaskOnlyRefracted;
  bool IVIdMaskHideWhenUnderground;
  bool IVIdMaskFoilage;
  bool IVIdMaskHideAlways;
  bool IVIdMaskHideButPick;
  bool IVIdMaskBackground;
  bool IVIdMaskGrassRGB;
  bool IVIdMaskLightGenP;
  bool IVIdMaskVehicle;
  bool IVIdMaskHideOnlyDirect;
  bool IVIdMaskInvisibleStopBounce;
  bool IVIdRefReflected;
  bool IVIdRefReflectMirror;
  bool IVIdRefRefracted;
  bool IVIdRefWaterNormalDecals;
  bool IVIdRefViewDepOcclusion;
  bool IVIdRefOnlyRefracted;
  bool IVIdRefHideWhenUnderground;
  bool IVIdRefFoilage;
  bool IVIdRefHideAlways;
  bool IVIdRefHideButPick;
  bool IVIdRefBackground;
  bool IVIdRefGrassRGB;
  bool IVIdRefLightGenP;
  bool IVIdRefVehicle;
  bool IVIdRefHideOnlyDirect;
  bool IVIdRefInvisibleStopBounce;
  bool SVIdMaskReflected;
  bool SVIdMaskReflectMirror;
  bool SVIdMaskRefracted;
  bool SVIdMaskWaterNormalDecals;
  bool SVIdMaskViewDepOcclusion;
  bool SVIdMaskOnlyRefracted;
  bool SVIdMaskHideWhenUnderground;
  bool SVIdMaskFoilage;
  bool SVIdMaskHideAlways;
  bool SVIdMaskHideButPick;
  bool SVIdMaskBackground;
  bool SVIdMaskGrassRGB;
  bool SVIdMaskLightGenP;
  bool SVIdMaskVehicle;
  bool SVIdMaskHideOnlyDirect;
  bool SVIdMaskInvisibleStopBounce;
  bool SVIdRefReflected;
  bool SVIdRefReflectMirror;
  bool SVIdRefRefracted;
  bool SVIdRefWaterNormalDecals;
  bool SVIdRefViewDepOcclusion;
  bool SVIdRefOnlyRefracted;
  bool SVIdRefHideWhenUnderground;
  bool SVIdRefFoilage;
  bool SVIdRefHideAlways;
  bool SVIdRefHideButPick;
  bool SVIdRefBackground;
  bool SVIdRefGrassRGB;
  bool SVIdRefLightGenP;
  bool SVIdRefVehicle;
  bool SVIdRefHideOnlyDirect;
  bool SVIdRefInvisibleStopBounce;
  bool CustomFog;
  float FogCustomFarZ;
  bool TreeMipForceLowQ;
  bool RenderShadows;
  bool NoShadowMPass;
  bool RenderProjectors;
  bool RenderZoneFogG;
  bool RenderLightFlares;
  bool InvertY;
  bool OnePixBorder;
  bool UseZBuffer;
  CPlugBitmapRender::ETrigger TriggerClearRGBA;
  vec3 ClearRGB;
  float ClearAlpha; // Range: 0 - 1
  UnnamedEnum ClearFogColor;
  bool IgnoreClearBitmap;
  bool HideSun;
  bool WriteRed;
  bool WriteGreen;
  bool WriteBlue;
  bool WriteAlpha;
  bool WriteTranslAlpha;
  bool ForceAlphaToOne;
  bool ForceAlphaToWrittenZ;
  CPlugBitmap* BitmapClear;
  UnnamedEnum BitmapClearMode;
  vec2 BitmapClearUV;
  uint SuperSampleLevel;
  uint BlurTexelCount;
  bool BlurWRed;
  bool BlurWGreen;
  bool BlurWBlue;
  bool BlurWAlpha;
  uint GutterTexelCount;
  bool GutterKeepAlpha;
  bool HdrToneMap;
  bool RenderMultiLight;
  bool UpdateForEachCamera;
  CPlugBitmapRenderSub* RenderSub;
  void CleanRenderCache();
};

struct CPlugBitmapRenderWater : public CPlugBitmapRender {
  CPlugBitmapRenderWater();

  enum class CPlugBitmap::E__ {
    Refraction = 0,
    Reflection = 1,
  };
  CPlugBitmap::E__ WaterType;
  float FogMaxDepth;
  float FogClampAboveDist;
  float MaxDistPlaneToAlpha;
  bool MirrorGeom;
  float MirrorScaleY; // Range: 0 - 1
  bool UseClipPlane;
  float ClipPlaneHeight;
  bool UseFMargin;
  bool FrustumUseHorizon;
  float FMargin;
  float FHMargin;
  bool RndLDirSpecInA;
  bool MaskWater2d;
  bool UseCameraZClip;
  bool AddWaterFog;
  bool NoSubWaterOptim;
  bool BlitCubeAtFarZ;
  bool DisableGeomOptim;
  UnnamedEnum ReflectNoGeom;
  UnnamedEnum QualityTech3;
  bool DisableConfigQuality;
  bool HqSplitSkyOutDepth;
  bool FogUseSharedParams;
  UnnamedEnum WaterHeight;
  float m_WaterHeightInW;
  bool UseBufferRefract;
  uint DisableRenderSkipOptims;
  float InvisibleRatio;
  uint InvisibleSleepPeriod;
  const float TmpWaterVisibilityRatio; // Range: 0 - 1
  uint MaxUpdatePeriod;
  float MaxCameraDeltaPos;
  float MinCameraDeltaCos;
  const float FrameRenderRatio;
  CPlugBitmap* BitmapSplitSky;
  CPlugBitmap* BitmapDepth;
};

struct CPlugBitmapRenderCubeMap : public CPlugBitmapRender {
  CPlugBitmapRenderCubeMap();

  void SaveInTga();
  uint CubeFaceCount;
  vec3 CenterPos;
  float NearZ;
  float FarZ;
  UnnamedEnum eFarZ;
  float MinDistToUpdate;
  bool AverageReceiverCenters;
  UnnamedEnum Discard;
  bool UseItemShaderFilter;
  CPlugBitmap* BitmapImageSpace;
  float BitmapImageSpaceDistToCenter;
  float BitmapImageSpaceScaleHeight;
};

struct CPlugBitmapRenderCamera : public CPlugBitmapRender {
  CPlugBitmapRenderCamera();

  bool UseCameraDrawRect;
  bool UseCameraScissor;
  UnnamedEnum CameraMode;
  iso4 CameraToVisual;
  float ScaleZRange;
  CMwNod* Camera;
  float DepthBias;
  float DepthBiasSlope;
  float PlaneY_Offset;
};

struct CPlugBitmapRenderVDepPlaneY : public CPlugBitmapRender {
  CPlugBitmapRenderVDepPlaneY();

  CPlugViewDepLocator* ViewDepLocator;
};

struct CPlugFileSndGen : public CPlugFileSnd {
  CPlugFileSndGen();

};

struct CPlugMaterialFxGenCV : public CPlugMaterialFx {
  CPlugMaterialFxGenCV();

  float DeltaYMax;
  float DeltaYMin;
  MwFastArray<CPlugMaterial*> MaterialToRayCasts;
};

// userName: 'SoundEngine'
// File extension: 'SoundEngine.Gbx'
struct CPlugSoundEngine : public CPlugSound {
  CPlugSoundEngine();

  MwFastArray<CPlugSoundComponent*> Components;
  float MaxRpm;
  NPlugCurve_SSimpleCurveInPlace7 Volume_Speed;
  NPlugCurve_SSimpleCurveInPlace7 Volume_Distance;
  NPlugCurve_SSimpleCurveInPlace7 Volume_Rpm;
  NPlugCurve_SSimpleCurveInPlace7 Volume_Accel;
  NPlugCurve_SSimpleCurveInPlace7 Alpha_Speed;
  NPlugCurve_SSimpleCurveInPlace7 Alpha_Distance;
  NPlugCurve_SSimpleCurveInPlace7 Alpha_Rpm;
  NPlugCurve_SSimpleCurveInPlace7 Alpha_Accel;
};

struct CPlugSoundComponent : public CMwNod {
  CPlugSoundComponent();

  CPlugFileSnd* PlugFile;
  float MinVolume;
  float MaxVolume;
  float FadeInStartSpeedKmh;
  float FadeInEndSpeedKmh;
  float FadeOutStartSpeedKmh;
  float FadeOutEndSpeedKmh;
  float MinPitch;
  float MaxPitch;
  float PitchShiftStartSpeedKmh;
  float PitchShiftEndSpeedKmh;
};

struct CPlugBitmapRenderSolid : public CPlugBitmapRender {
  CPlugBitmapRenderSolid();

  UnnamedEnum TriggerBitmap;
  UnnamedEnum TriggerShader;
  UnnamedEnum TriggerSolid;
  UnnamedEnum BitmapFilter;
  CPlugBitmap* Bitmap;
  CPlugShader* Shader;
  const MwFastBuffer<CPlugSolid*> Solids;
  MwFastBuffer<iso4> Locations;
};

struct CPlugBitmapRenderSub : public CPlugBitmapRender {
  CPlugBitmapRenderSub();

  CPlugShader* ShaderToForce;
};

// File extension: 'Model.Gbx'
struct CPlugModel : public CPlugTreeGenerator {
  CPlugModel();

  string Origin;
  float ExportScale;
  CPlugModelTree* const ModelTree;
  CPlugSkel* const Skel;
  CPlugPath* const Path;
  float VertexPositionsQuantize;
};

struct CPlugIconIndex : public CMwNod {
  CPlugIconIndex();

};

struct CPlugVehicleGearBox : public CMwNod {
  CPlugVehicleGearBox();

};

// userName: 'AdnAnimClip'
// File extension: 'AdnAnimClip.gbx'
struct CPlugAdnAnimClip : public CMwNod {
  CPlugAdnAnimClip();

  CPlugAnimClip* AnimClip;
  MwFastBuffer<NPlugAdn_STag> RequiredTags;
  MwFastBuffer<NPlugAdn_STag> Tags;
  float Proba;
};

// userName: 'AdnTagDataBase'
// File extension: 'gbx.json'
struct NPlugAdn_STagDatabase : public CMwNod {
  NPlugAdn_STagDatabase();

  MwSArray<NPlugAdn_STagTypeDef> Types;
  void CompatGenerateDefautHumanDb();
};

struct CPlugFileModel : public CPlugFile {
};

struct CPlugFileModelObj : public CPlugFileModel {
  CPlugFileModelObj();

};

// File extension: 'GenSolid.Gbx'
struct CPlugTreeGenSolid : public CPlugTreeGenerator {
  CPlugTreeGenSolid();

  bool MergeInstances;
  CPlugSolid* Solid;
  CPlugMaterial* SolidReplaceMaterial;
  bool UseCustomFuncTree;
  float CustomFuncTreePhase;
  float CustomFuncTreePeriodScale;
};

struct CPlugFileModel3ds : public CPlugFileModel {
  CPlugFileModel3ds();

};

struct CPlugModelLodMesh : public CMwNod {
  CPlugModelLodMesh();

  MwFastBuffer<NPlugModelLodMesh_SRange> m_Ranges;
  bool SplitGridIsUse;
  vec3 m_OptimFlags_SplitGridCellSize;
  vec3 m_OptimFlags_SplitGridOrigin;
  string OptimGroupId;
};

// File extension: 'ModelFur.Gbx'
struct CPlugModelFur : public CMwNod {
  CPlugModelFur();

  uint RandSeed;
  bool Enabled;
  string MaterialName;
  CPlugMaterial* Material;
  CMwNod* DiffuseMap;
  CMwNod* SpecularMap;
  CMwNod* DensityMap;
  CMwNod* DirAlphaMap;
  CMwNod* DirBetaMap;
  CMwNod* WidthMap;
  CMwNod* LengthMap;
  CMwNod* CurvatureMap;
  uint MapAtlasX;
  uint MapAtlasY;
  bool RandomUDir;
  float DirAlphaDeg;
  float DirAlphaVarDeg;
  float DirBetaDeg;
  float DirBetaVarDeg;
  float CurvatureDeg;
  float CurvatureVarDeg;
  float HelixDeg;
  float HelixVarDeg;
  float Width;
  float WidthVar; // Range: 0 - 1
  float Length;
  float LengthVar; // Range: 0 - 1
  float NormalBendAngleX;
  float NormalBendAngleY;
  bool Debug;
  bool DoubleLayer;
  bool ModulateWidthByTriSurf;
  uint DensityMax;
  bool DensityInAtlasY;
  float DensityRandom;
  uint DensitySampling;
  float FluffPosRandom;
  bool DynaEnabled;
  float DynaK1;
  float DynaK1Var;
  float DynaK2;
  float DynaK2Var;
  float DynaAccel;
  float DynaAccelVar;
  bool DynaZeroBeta;
  float DynaAlphaMin;
  float DynaAlphaMax;
  float DynaBetaMin;
  float DynaBetaMax;
  uint FluffChunkCount;
  uint ChunkVertCount;
  bool FluffShapeCircular;
  float FluffShapeEndWidthCoef;
  uint HairMaxCount;
};

struct CPlugBitmapRenderOverlay : public CPlugBitmapRender {
  CPlugBitmapRenderOverlay();

  bool IsOverlaySelfBlur;
  float TexelByPixel;
  bool CanRenderInSubRect;
};

struct CPlugBitmapRenderLightOcc : public CPlugBitmapRender {
  CPlugBitmapRenderLightOcc();

  float FovY;
  float OpacityLightThrough; // Range: 0 - 1
  float OpacityLensFlare; // Range: 0 - 1
  float FlareThreshold; // Range: 0 - 1
  CPlugBitmap* BitmapToModulate;
};

// File extension: 'ViewDepLocator.Gbx'
struct CPlugViewDepLocator : public CMwNod {
  CPlugViewDepLocator();

  float MinY;
  float MaxY;
  bool UseWaterY;
};

// File extension: 'DecoTree.Gbx'
struct CPlugDecoratorTree : public CMwNod {
  CPlugDecoratorTree();

  MwId TreeId;
  CPlugMaterial* Material;
  CPlugTreeLight* TreeLight;
  enum class CPlugDecoratorTree::EBoolCond {
    Never = 0,
    LowOnly = 1,
    LowAndMedium = 2,
    MediumOnly = 3,
    MediumAndHigh = 4,
    HighOnly = 5,
    Always = 6,
  };
  CPlugDecoratorTree::EBoolCond ExistCond;
  CPlugDecoratorTree::EBoolCond VisibleCond;
  bool VisibleApplyOnChilds;
  CPlugDecoratorTree::EBoolCond ShadowCasterCond;
  bool ShadowCasterApplyOnChilds;
  bool TransformVisualToSurface;
  CPlugDecoratorTree::EBoolCond CollidableCond;
  bool NoLocation;
};

// File extension: 'DecoSolid.Gbx'
struct CPlugDecoratorSolid : public CMwNod {
  CPlugDecoratorSolid();

  const MwFastBuffer<CPlugDecoratorTree*> TreeDecorators;
};

// File extension: 'ModelFences.Gbx'
struct CPlugModelFences : public CMwNod {
  CPlugModelFences();

  float BlockSizeXZ;
  uint BlockFenceCountXZ;
  vec2 FenceRangeY;
  float LodMaxDist;
  uint RandSeed;
  CPlugMaterial* MaterialFences;
  bool IsOrthos;
  bool IsDiags;
  bool OnlyOnePlaneY;
  bool Debug;
  bool DebugRand;
  bool DebugEdges;
};

struct CPlugFileModelFbx : public CPlugFileModel {
  CPlugFileModelFbx();

};

// File extension: 'FurWind.Gbx'
struct CPlugFurWind : public CMwNod {
  CPlugFurWind();

  vec3 WorldDir;
  float Intensity;
};

// File extension: 'Decal.Gbx'
struct CPlugDecalModel : public CPlug {
  CPlugDecalModel();

  bool IsObsolete;
  CPlugBitmap* Icon;
  float TexelByMeter;
  float MaxAngleN; // Range: 0 - 90
  bool FadeNormalAndZ;
  CPlugBitmap* DiffuseA;
  CPlugBitmap* Specular;
  CPlugBitmap* Roughness;
  CPlugBitmap* Normal;
  float ImpactSize;
  DataRef Svg;
  float SvgSize;
  float SvgAlpha;
  CPlugSolid* Solid;
  CPlugBitmap* _3dSpriteBitmap;
  MwId _3dSpriteGroupId;
  bool RandomInstances;
  float MaxAngleN3d; // Range: 0 - 180
  uint MinAngleN3d; // Range: 0 - 180
};

struct CPlugBitmapAtlas : public CPlug {
  CPlugBitmapAtlas();

};

// File extension: 'SphericalHarmonics.Gbx'
struct CPlugSphericalHarmonics : public CPlug {
  CPlugSphericalHarmonics();

};

struct CPlugBitmapArray : public CPlug {
  CPlugBitmapArray();

};

struct CPlugSpriteParam : public CPlug {
  CPlugSpriteParam();

  enum class CPlugVisualSprite::ERenderMode {
    Quad = 0,
    RotatedQuad = 1,
  };
  CPlugVisualSprite::ERenderMode RenderMode;
  UnnamedEnum Usage;
  bool RadiusInScreen;
  bool UseGlobalDir;
  bool SortBackToFront;
  vec3 GlobalDirection;
  vec2 PivotPoint;
  float GlobalDirTiltFactor; // Range: 0 - 1
  float ZBiasFactor; // Range: 0 - 1
  float TextureHeightInWorld;
  float VisibleMaxDistAtFov90;
  float VisibleMinScreenHeight01;
};

struct CPlugFileExr : public CPlugFileImg {
  CPlugFileExr();

};

// File extension: 'PoissonDiscDistribution.Gbx'
struct CPlugPoissonDiscDistribution : public CPlug {
  CPlugPoissonDiscDistribution();

};

// userName: 'AnimFile'
// File extension: 'Anim.Gbx'
struct CPlugAnimFile : public CMwNod {
  CPlugAnimFile();

  MwSArray<SConstString> BaseAssetFolderPaths;
  MwFastBuffer<SMwIdRefSkel> Skels;
  MwFastBuffer<SMwIdRefRig> Rigs;
  MwFastBuffer<SMwIdRefRigToSkel> RigToSkels;
  const MwFastBuffer<CPlugAnimClipBaked*> BakedClips;
  MwFastBuffer<SMwIdRefChannelGroup> ChannelGroups;
  MwFastBuffer<SMwIdRefJointExprGroup> JointExprGroups;
  MwFastBuffer<CPlugAnimVariantGroup*> VariantGroups;
  MwFastBuffer<CPlugAnimPoseGrid*> PoseGrids;
  MwFastBuffer<CPlugAnimPoseGroup*> PoseGroups;
  MwFastBuffer<CPlugAnimClipEdition*> EditionClips;
  MwFastBuffer<SMwIdRefClip> Clips;
  MwArrayInPlaceDyn<MwClassId> GraphContextClassIds;
  MwFastBuffer<CPlugAnimGraphNode_Graph*> NodeGraphs;
  MwFastBuffer<CPlugAnimSpotModel> Spots;
  wstring ImportString;
  wstring UpdateString;
  wstring ExportFullName;
  void Dbg();
  void Log();
};

struct CPlugVisualCelEdge : public CPlugVisual {
  CPlugVisualCelEdge();

};

// userName: 'PlugParticleEmitterSubModel'
// File extension: 'ParticleEmitterSubModel.Gbx'
struct CPlugParticleEmitterSubModel : public CMwNod {
  CPlugParticleEmitterSubModel();

  float ActiveMinSpeedKmh;
  EPlugParticleEmitterSubModel ParticleEmitterSubModel;
  uint MaxParticleCount;
  bool MultiState_IsAsyncLink;
  uint PartChain_MaxCount;
  uint PartChain_MaxPartPerChain;
  bool PartChain_AddLinkOnDestroy;
  CPlugParticleGpuSpawn* GpuSpawn;
  CPlugParticleGpuModel* GpuModel;
  CPlugParticleSplashModel* SplashModel;
  SPlugParticleSpawnModel Spawn;
  SPlugParticleLifeModel Life;
  SPlugParticleRenderModel Render;
  SPlugParticlePhysicsModel Physics;
  SPlugParticlePrecalcModel Precalc;
  SPlugParticleSimulatedSmokeModel SimulatedSmoke;
  SPlugParticleEmitStateFromImpactModel EmitStateFromImpact;
  SPlugParticleLaserEnergyModel LaserEnergyModel;
};

// File extension: 'ParticleModel.Gbx'
struct CPlugParticleEmitterModel : public CMwNod {
  CPlugParticleEmitterModel();

  bool IsSplashMode;
  MwFastBuffer<CPlugParticleEmitterSubModel*> ParticleEmitterSubModels;
  float ShadowMapTexelSize;
  bool EnableVortexEmitter;
  uint VortexCountPerCircle;
  vec3 CircleNormal;
  float Randomness;
  float Radius;
  uint EmitPeriod;
  float Lod_cPixelMin;
  float Strength;
  uint LifeTime;
  float Coef;
};

// File extension: 'BeamModel.Gbx'
struct CPlugBeamEmitterModel : public CMwNod {
  CPlugBeamEmitterModel();

  MwFastBuffer<CPlugBeamEmitterSubModel*> SubModels;
};

struct CPlugParticleSplashModel : public CMwNod {
  CPlugParticleSplashModel();

};

struct CPlugParticleImpactModel : public CMwNod {
  CPlugParticleImpactModel();

};

struct CPlugParticleMaterialImpactModel : public CMwNod {
  CPlugParticleMaterialImpactModel();

};

struct CPlugBitmapApplyArray : public CPlug {
  CPlugBitmapApplyArray();

};

struct CPlugOpModel : public CMwNod {
  CPlugOpModel();

};

// userName: 'Skel'
// File extension: 'Skel.Gbx'
struct CPlugSkel : public CMwNod {
  CPlugSkel();

  uint16 cJoint;
  MwFastBuffer<MwId> JointNames;
  MwFastBuffer<uint16> JointParentIndexs;
  MwFastBuffer<iso4> RefGlobalJoints;
  MwSArray<GmTransQuat> RefGlobalJointsTQ;
  UnknownType LodMaxDists;
  MwFastBuffer<uint8> JointMaxLods;
  MwFastBuffer<uint8> JointFixedTranss;
  MwSArray<uint16> JointChildIndexs;
  MwSArray<uint16> JointChildArrays;
  bool DevNonOrtho;
  CPlugSkelSetup* Setup;
  MwStridedArray<vec3> RefLocalJointsTranss;
  uint8 cLod;
  MwSArray<uint> LodJointCounts;
  MwFastBuffer<SPlugSkelSocket> Sockets;
  void Normalize();
};

struct CPlugSolid2Model : public CMwNod {
  CPlugSolid2Model();

};

struct CPlugFileModelCollada : public CPlugFileModel {
  CPlugFileModelCollada();

};

struct CPlugTimedPixelArray : public CPlug {
  CPlugTimedPixelArray();

};

struct CPlugResource : public CMwNod {
  CPlugResource();

};

// File extension: 'WeatherModel.Gbx'
struct CPlugWeatherModel : public CMwNod {
  CPlugWeatherModel();

  MwFastBuffer<CPlugWeather*> Weathers;
  CPlugMoodSetting* MoodSetting;
  CPlugDayTime* DayTime;
  CPlugBitmap* BitmapSpecularDir;
  CPlugBitmap* BitmapWaterFog;
};

struct CPlugFileFidContainer_SystemUserSaveProxy : public CPlugFileFidContainer {
};

struct CPlugFxLensFlareArray : public CPlug {
  CPlugFxLensFlareArray();

};

struct CPlugParticleEmitterSubModelGpu : public CMwNod {
  CPlugParticleEmitterSubModelGpu();

};

struct CPlugParticleGpuSpawn : public CMwNod {
  CPlugParticleGpuSpawn();

};

struct CPlugParticleGpuModel : public CMwNod {
  CPlugParticleGpuModel();

};

struct CPlugCharVisModel : public CMwNod {
  CPlugCharVisModel();

};

struct CPlugSkelSetup : public CMwNod {
  CPlugSkelSetup();

};

struct CPlugCharPhyModel : public CMwNod {
  CPlugCharPhyModel();

};

struct CPlugCharPhyMaterial : public CMwNod {
  CPlugCharPhyMaterial();

};

struct CPlugFxLensDirtGen : public CMwNod {
  CPlugFxLensDirtGen();

  MwId AtlasId;
  uint Count;
  float ScreenSizeYMin; // Range: 0 - 1
  float ScreenSizeYMax; // Range: 0 - 1
  float Intens_Min; // Range: 0 - 1
  float Intens_Max; // Range: 0 - 1
};

struct CPlugShieldEmitterModel : public CMwNod {
  CPlugShieldEmitterModel();

};

// userName: 'BulletModel'
// File extension: 'Bullet.Gbx'
struct CPlugBulletModel : public CMwNod {
  CPlugBulletModel();

  CPlugBulletPhyModel::EBulletType BulletType;
  float BulletRadius;
  float BulletVsRadiusMargin;
  uint BulletVsMinAgeMs;
  float Speed;
  NPlugCurve_SSimpleCurveInPlace7 SpeedFromHorizonAngle;
  float GunSpeedCoef;
  float GunSpeedCoefRatioMin;
  float GunSpeedCoefRatioMax;
  float RestSpeed;
  float LifeTime;
  float LifeTimeAfterFirstImpact;
  float Mass;
  float FluidFriction;
  float MaxDistance;
  float TrajectoryUTurnDistance;
  float ImpactBouncingN;
  float ImpactBouncingT;
  float TriLaserRadiusAtOneMeter_Zoom0;
  float TriLaserRadiusAtOneMeter_Zoom1;
  float TriLaserNearMissDist;
  float ProjectileNearMissDist;
  float ProjectileNearMissDelay;
  uint ChargeDurationFull;
  uint ChargeDurationStop;
  float SpearLockDist;
  bool DamageAttenuationWithDist;
  NPlugCurve_SSimpleCurveInPlace7 DamageAttenuationFromDist;
  uint ExplosionOnImpactCount;
  bool ExplosionOnEndLife;
  float ExplosionDetectionRadius_Resting;
  float ExplosionDetectionRadius_Flying;
  bool ExplosionOnAllPlayers;
  bool ExplosionBlowOnAllPlayers;
  bool DirectHitDamageValue;
  bool ExplosionDamageValue;
  float ExplosionDamageRadius;
  float ExplosionDamageRadiusAttenuation; // Range: 0 - 1
  float ExplosionBlowValue;
  float ExplosionBlowRadius;
  float ExplosionBlowRadiusAttenuation; // Range: 0 - 1
  float ExplosionElectroPulseValue;
  NPlugCurve_SSimpleCurveInPlace7 ExplosionRadiusCoefFromLifeTime;
  float ExplosionBlowVerticalScale;
  CPlugBulletPhyModel::EExplosionOccultationTest ExplosionOccultationTest;
  float ExplosionEnergyWhenOccluded; // Range: 0 - 1
  NPlugCurve_SSimpleCurveInPlace7 ExplosionEnergyFromLifeTime;
  float HomingDist; // Range: 1 - 100
  float HomingPeriod; // Range: 0.1 - 1
  float HomingMaxAngularSpeed; // Range: 10 - 320
  uint HomingLockDuration;
  float GuidedAngularSpeed; // Range: 10 - 320
  uint GuidedMinLifeTime;
  float GuidedRollAngleMax; // Range: 0 - 3.14159
  bool IsFlare;
  float FlareAttractionRadius;
  float FlareExplosionRadius;
  bool IsWard;
  float WardRadius;
  bool MultiSphereDetection;
  bool Sm_AutoAim;
  float Sm_AimIntertia;
  float Sm_Recoil;
  uint Sm_ThisHitOtherBulletBonusDuration;
  float Sm_ThisHitOtherBulletBonusAmmoCount;
  uint Sm_OtherHitThisBulletBonusDuration;
  float Sm_OtherHitThisBulletBonusAmmoCount;
  bool HoldTriggerToPower;
  NPlugCurve_SSimpleCurveInPlace7 SpeedFromTriggerDuration;
  uint AmmoCostDuringTriggerDuration;
  NPlugCurve_SSimpleCurveInPlace7 SpreadCoefFormChargeDuration;
  uint FirePeriod;
  uint FireEnergyCost;
  uint FireMaxEnergyStorage;
  bool FireVisualRecoil;
  bool NeedReleaseBeforeFireAgain;
  uint EnergyGainDelay;
  uint DefuserFirePeriod;
  uint DefuserMissAmmoLoss;
  uint FireBulletCount;
  float FireBulletDispersionAngleDeg;
  float FireBulletCountSpeedCoef;
  MwFastBuffer<vec3> FireBulletSpreads;
  CPlugBulletPhyModel::EFireBulletPatternMode FireBulletPattern_Mode;
  bool FireBulletPattern_OnlyVisual;
  MwFastBuffer<vec3> FireBulletPattern_GrowingOffset_Offsets;
  float FireBulletPattern_InitialBlendDuration; // Range: 0 - 60
  uint FireBulletPattern_SpiralHatchetNet_BulletCount; // Range: 1 - 255
  float FireBulletPattern_SpiralHatchetNet_Radius; // Range: 0 - 100
  float FireBulletPattern_SpiralHatchetNet_SpinPerSecond; // Range: -1000 - 1000
  bool FireBulletPattern_RotateOddBulletsCounterClockwise;
  bool FireBulletPattern_ApexRegroup;
  float FireBulletPattern_ApexRegroup_MinApexTime; // Range: 0 - 10
  bool FireBulletPattern_ChaosSpiral_Spin;
  MwFastBuffer<float> FireBulletPattern_ChaosSpiral_Offsets;
  CPlugSound* SoundFire;
  CPlugSound* SoundAlive;
  CPlugSound* SoundReloading;
  CPlugSound* SoundGauge;
  CPlugSound* SoundIsHoming;
  vec3 SoundFireOffset;
  NPlugCurve_SSimpleCurveInPlace7 SoundAliveVolumeFromRemainingLifeTime;
  NPlugCurve_SSimpleCurveInPlace7 SoundAlivePitchFromRemainingLifeTime;
  NPlugCurve_SSimpleCurveInPlace7 SoundFireVolumeFromRemainingAmmo;
  NPlugCurve_SSimpleCurveInPlace7 SoundFirePitchFromRemainingAmmo;
  CPlugParticleEmitterModel* ParticleModelFire;
  CPlugParticleEmitterModel* ParticleModelAlive;
  CPlugParticleMaterialImpactModel* ImpactBounce;
  CPlugParticleMaterialImpactModel* ExplosionBullet;
  CPlugParticleMaterialImpactModel* ExplosionSurface;
  CPlugBeamEmitterModel* BeamVisualModel;
  float HiddenDist;
  vec3 VisualOffsetFirstPerson;
  float VisualOffsetDuration;
  float VisualOffsetDerivTimeOffset;
  bool WarmUp;
  wstring Name;
  CPlugBitmap* BitmapLasersight;
  CPlugBitmap* BitmapCrosshair;
  CPlugBitmap* BitmapCrosshairTriLaser;
  float CrossHairSizeXScreenPercent;
  float CrossHairSizeYScreenPercent;
  float CrossHairSizeXScreenPercentMax;
  float CrossHairSizeYScreenPercentMax;
  CPlugBitmap* Icon;
  bool ShowImpactPos;
  MwId Sm_SpModelName;
  uint Sm_SpChargeDuration;
  uint Sm_SpInactivityDelayBeforeCharge;
  bool Sm_FirstPersonCam;
  vec3 Sm_FirstPersonCamOffset;
  float Sm_FirstPersonCamFov;
  uint Sm_ProtectDuration;
};

struct CPlugDataTape : public CMwNod {
  CPlugDataTape();

};

struct CPlugSpline3D : public CMwNod {
  CPlugSpline3D();

  const float Length;
  vec3 Translate;
  CFuncKeysTrans* const Keys;
};

struct CPlugFogMatter : public CPlug {
  CPlugFogMatter();

};

struct CPlugFogVolume : public CPlug {
  CPlugFogVolume();

};

struct CPlugFogVolumeBox : public CPlugFogVolume {
  CPlugFogVolumeBox();

};

// userName: 'PlugDestructibleFx'
// File extension: 'PlugDestructibleFx.Gbx'
struct CPlugDestructibleFx : public CMwNod {
  CPlugDestructibleFx();

  CPlugParticleMaterialImpactModel* Vis_MaterialImpactModel;
  CPlugParticleMaterialImpactModel* Vis_MaterialDestructionModel;
  float Vis_ImpactScaleFromMagnitudeRatio;
  float Vis_ImpactScaleFromMagnitudeMax;
  float Vis_ImpactMaterialSoundAttenuation;
  CPlugSound* Vis_ImpactSound;
  float Vis_SoundClusterMinMagnitude;
  uint Vis_SoundClusterMinNbImpacts;
  float Vis_SoundClusterRadius;
  uint Vis_SoundClusterAccumDuration;
  uint Vis_SoundClusterMaskingDuration;
  float Vis_SoundClusterVolumeFromMagnitudeRatio;
  float Vis_SoundClusterVolumeFromMagnitudeMax;
  CPlugParticleEmitterModel* Vis_CreationEmitterModel;
  CPlugParticleEmitterModel* Vis_HeatEmitterModel;
  CPlugParticleEmitterModel* Vis_SteamEmissionModel;
  CPlugSound* Vis_HeatSound;
  uint Phy_MaxNoticePerFrame;
  float Phy_MinImpactVelocity;
  uint Phy_TimeBetweenImpact;
};

struct CPlugDynaPointModel : public CMwNod {
  CPlugDynaPointModel();

  vec3 Center;
  float Radius;
  float Friction;
  float Restitution;
  const float GravityCoef;
};

struct CPlugFxLightning : public CPlug {
  CPlugFxLightning();

};

struct CPlugFxWindOnDecal : public CPlug {
  CPlugFxWindOnDecal();

};

struct CPlugFileWebP : public CPlugFileImg {
  CPlugFileWebP();

};

struct CPlugMaterialPack : public CPlug {
  CPlugMaterialPack();

};

struct NPlugItemPlacement_SDatabase : public CMwNod {
  NPlugItemPlacement_SDatabase();

  MwSArray<NPlugItemPlacement_STypeDef> Types;
};

struct CPlugCharPhyRecoilModel : public CMwNod {
  CPlugCharPhyRecoilModel();

};

struct CPlugFxWindOnTreeSprite : public CPlug {
  CPlugFxWindOnTreeSprite();

};

// userName: 'FlockModel'
// File extension: 'PlugFlockModel.Gbx'
struct CPlugFlockModel : public CMwNod {
  CPlugFlockModel();

  EPlugFlockType FlockType;
  uint DefSpawnCount;
  CMwNod* BirdModel;
  CSystemFidFile* BirdModelFid;
  CSystemFidFile* AnimFile_Fid;
  CPlugSound* SoundLoop;
  CPlugSound* SoundEventTakeOff;
  float Volatility; // Range: 0 - 1
  float Range;
  float CosViewAngle;
  float MinSpeed;
  float MaxSpeed;
  uint UpdateFrequency;
  float Variance;
  float vAvoidance;
  float kAvoidance;
  float vGrouping;
  float kGrouping;
  float vMatching;
  float kMatching;
  float vGroundAvoid;
  float kGroundAvoid;
  uint StandingDuration;
  uint AnimPeriod;
  uint Anim_StandingStart;
  uint Anim_StandingEnd;
  uint Anim_GlidingStart;
  uint Anim_GlidingEnd;
  uint Anim_FlappingStart;
  uint Anim_FlappingEnd;
};

struct CPlugVehicleVisEmitterModel : public CMwNod {
  CPlugVehicleVisEmitterModel();

};

// userName: 'VehicleVisModel'
// File extension: 'VehicleVisModel.Gbx'
struct CPlugVehicleVisModel : public CMwNod {
  CPlugVehicleVisModel();

};

struct CPlugVehicleVisModelShared : public CMwNod {
  CPlugVehicleVisModelShared();

};

struct CPlugVehicleMaterialGroup : public CMwNod {
  CPlugVehicleMaterialGroup();

};

// userName: 'VehicleCameraRace3Model'
// File extension: 'VehicleCameraRace3Model.gbx'
struct CPlugVehicleCameraRace3Model : public CPlugCamControlModel {
  CPlugVehicleCameraRace3Model();

};

struct CPlugBodyPath : public CMwNod {
  CPlugBodyPath();

};

struct CPlugCharPhySpecialProperty : public CMwNod {
  CPlugCharPhySpecialProperty();

};

struct CPlugParticleGpuVortex : public CMwNod {
  CPlugParticleGpuVortex();

};

struct CPlugGameSkin : public CMwNod {
  CPlugGameSkin();

};

struct CPlugFxHdrScales_Tech3 : public CPlug {
  CPlugFxHdrScales_Tech3();

};

// userName: 'VehicleCameraRace2Model'
// File extension: 'VehicleCameraRace2Model.gbx'
struct CPlugVehicleCameraRace2Model : public CPlugCamControlModel {
  CPlugVehicleCameraRace2Model();

};

struct CPlugVehicleCameraInternalModel : public CPlugCamControlModel {
  CPlugVehicleCameraInternalModel();

  float Fov; // Range: 60 - 110
  bool IsFirstPerson;
  vec3 RelativePos;
  vec3 PitchYawRoll;
  vec3 RelativePos_Hmd;
  bool CamBlendEnabled;
  float PilotHeadCoef;
  bool Hmd_NoPitchRoll;
  uint Hmd_Model;
  float Hmd_Up_DegMax;
  float Hmd_Up_DegPerSecond;
  float BulletTimeFovSmoothDelta_m_Delta;
  uint BulletTimeFovSmoothDelta_m_TimeUp;
  uint BulletTimeFovSmoothDelta_m_TimeDown;
  float SuperBulletTimeFovSmoothMultiplier;
  uint SuperBulletTimeFovSmoothMultiplier_m_TimeUp;
  uint SuperBulletTimeFovSmoothMultiplier_m_TimeDown;
};

// userName: 'AnimLocSimple'
// File extension: 'AnimLocSimple.Gbx'
struct CPlugAnimLocSimple : public CMwNod {
  CPlugAnimLocSimple();

  uint Axis;
  uint RotPeriod;
  uint RotPeriodMax;
  float RotAngle;
  EPlugAnimSimpleFunc RotFunc;
  uint TransPeriod;
  uint TransPeriodMax;
  float TransY;
  bool NightOnly;
};

struct CPlugLightUserModel : public CMwNod {
  CPlugLightUserModel();

  vec3 Color;
  float Intensity;
  float Distance;
  float PointEmissionRadius;
  float PointEmissionLength;
  float SpotInnerAngle;
  float SpotOuterAngle;
  float SpotEmissionSizeX;
  float SpotEmissionSizeY;
  bool NightOnly;
};

struct CPlugCharPhyMaterials : public CMwNod {
  CPlugCharPhyMaterials();

};

struct CPlugCharPhyModelCustom : public CMwNod {
  CPlugCharPhyModelCustom();

  float EyesHeight;
  float Radius;
  float SpeedCoef;
  float JumpCoef;
  float AggroRadius;
  float ShootRadius;
  float TargetMinDistance;
  float DisengageDistance;
  float Accuracy; // Range: 0 - 1
  uint Reaction;
  uint ShootPeriod;
  uint ArmorMax;
  uint Fov;
};

// File extension: 'Mat.Gbx'
struct CPlugMaterialUserInst : public CMwNod {
  CPlugMaterialUserInst();

  enum class CPlugMaterialUserInst::ETexAddress {
    Wrap = 0,
    Mirror = 1,
    Clamp = 2,
    Border = 3,
  };
  enum class CPlugMaterialUserInst::EMaterialModelStatic {
    TDSN = 0,
    TDOSN = 1,
    TDOBSN = 2,
    TDSNE = 3,
    TDSNI = 4,
    TDSNI_Night = 5,
    TIAdd = 6,
  };
  enum class CPlugMaterialUserInst::EMaterialModelDyna0 {
    ZOnly_Water = 0,
    TDSNI = 1,
    TI = 2,
    TI_AddModCV = 3,
    TDSNE = 4,
    TE = 5,
    TIce = 6,
    TShield = 7,
  };
  enum class CPlugMaterialUserInst::EMaterialModelChar {
    TDSNEM = 0,
    TE = 1,
    TDOSNEM = 2,
    TDOSN = 3,
    TDOSN2Sided = 4,
    TDOS = 5,
    TAniso = 6,
    Part = 7,
    Body = 8,
  };
  enum class CPlugMaterialUserInst::EMaterialModelVehicle {
    Skin = 0,
    SkinDmg = 1,
    SkinDmgDecal = 2,
    SkinNoSkelDmg = 3,
    Details = 4,
    DetailsDmg = 5,
    DetailsDmgNormal = 6,
    DetailsDmgDecal = 7,
    DetailsNoSkelDmg = 8,
    Glass = 9,
    GlassDmg = 10,
    GlassDmgCrack = 11,
    GlassDmgDecal = 12,
    GlassNoSkelDmg = 13,
    GlassRefract = 14,
    GlassRefractNoSkelDmg = 15,
    Gems = 16,
    GemsNoSkelDmg = 17,
  };
  MwId _Name;
  string _LinkFull;
  MwId _Link_OldCompat;
  bool IsUsingGameMaterial;
  const string Collection;
  uint MaterialId;
  string LinkFull;
  CPlugMaterialUserInst::EMaterialModelStatic ModelModelStatic;
  CPlugMaterialUserInst::EMaterialModelDyna0 ModelModelDyna0;
  CPlugMaterialUserInst::EMaterialModelChar ModelModelChar;
  CPlugMaterialUserInst::EMaterialModelVehicle ModelModelVehicle;
  MwId Model;
  wstring BaseTexture;
  DataRef TexturesDiffuse;
  DataRef TexturesDiffuseO;
  DataRef TexturesBaseColor;
  DataRef TexturesBaseColorO;
  DataRef TexturesSpecular;
  DataRef TexturesNormal;
  DataRef TexturesEnergy;
  DataRef TexturesTeamMask;
  DataRef TexturesSelfIllum;
  DataRef TexturesDamage;
  DataRef TexturesDirt;
  DataRef TexturesShield;
  DataRef TexturesRoughMetal;
  MwId Link;
  CPlugMaterialUserInst::ETexAddress TilingU;
  CPlugMaterialUserInst::ETexAddress TilingV;
  float TextureSizeInMeters;
  uint PhysicsID;
  uint Dbg_PhysicsID;
  string PhysicsName;
  uint Dbg_PhysicsName;
  uint GameplayID;
  string GameplayName;
  bool IsNatural;
  MwId HidingGroup;
};

struct CPlugMoodSetting : public CPlug {
  CPlugMoodSetting();

};

struct CPlugMoodAtmo : public CPlug {
};

struct CPlugBodyGraph : public CMwNod {
  CPlugBodyGraph();

};

struct CPlugCustomBulletModel : public CMwNod {
  CPlugCustomBulletModel();

  string BulletName;
  EPlugCustomBulletModelType BulletType;
  CPlugBulletPhyModel::EFireBulletPatternMode BulletPattern;
  bool BulletIsFlare;
  bool BulletIsWard;
  float BulletSpeed;
  float BulletGunSpeedCoef; // Range: 0.01 - 1
  float BulletGunSpeedCoefRatioMin; // Range: 0.01 - 1
  float BulletGunSpeedCoefRatioMax; // Range: 0.01 - 1
  float BulletMass;
  float BulletFluidFriction;
  float BulletRecoil; // Range: 0 - 1
  float BulletHitboxRadius;
  float BulletLifeTime;
  float BulletLifeTimeAfterFirstImpact;
  bool BulletExplodeOnEndLife;
  uint BulletRebounds;
  bool BulletBounceOnTechWall;
  float BulletImpactBouncingN; // Range: 0 - 1
  float BulletImpactBouncingT; // Range: 0 - 1
  uint NoPatternBulletCount;
  float NoPatternBulletDispersionAngle;
  float NoPatternBulletSpeedCoef;
  uint PatternBulletCount;
  float PatternBulletRadius;
  float PatternBulletSpinSecond;
  float PatternBulletBlendDuration;
  bool PatternBulletApexRegroup;
  float PatternBulletMinApexTime;
  bool PatternBulletRandomRotations;
  float BulletHomingDist;
  float BulletHomingAngularSpeed;
  float BulletHomingPeriod;
  uint BulletHomingLockDuration;
  uint BulletHomingDamageMinAngle_Deg;
  uint BulletHomingDamageMaxAngle_Deg;
  float BulletGuidedAngularSpeed;
  uint BulletGuidedMinLifeTime;
  float BulletFlareAttractionRadius;
  float BulletFlareExplosionRadius;
  float BulletWardRadius;
  MwId BulletAliveSound;
  MwId BulletExplosionSound;
  MwId BulletShootingSound;
  MwId BulletReboundSound;
  MwId BulletHomingSound;
  float BulletAliveSoundVolume;
  float BulletExplosionSoundVolume;
  float BulletShootingSoundVolume;
  float BulletReboundSoundVolume;
  float BulletHomingSoundVolume;
  MwFastBuffer<CPlugParticleEmitterSubModel*> m_AliveParticleEmitterSubModels;
  MwFastBuffer<CPlugParticleEmitterSubModel*> m_ExplosionParticleEmitterSubModels;
  bool BulletShowPlayerExplosion;
  bool BulletShowDebris;
  bool BulletModifyFOV;
};

struct CPlugTriggerAction : public CMwNod {
  float StartYaw;
};

// File extension: 'BeamModel.Gbx'
struct CPlugBeamEmitterSubModel : public CMwNod {
  CPlugBeamEmitterSubModel();

  bool EdVisible;
  CPlugMaterial* Material;
  const MwFastBuffer<CPlugMaterial*> AdditionalMaterials;
  enum class CPlugBeamEmitterSubModel::EDisplayType {
    LinesWide = 0,
    Cylinder = 1,
    Helix = 2,
    HelixWithCylinder = 3,
    DoubleHelix = 4,
    DoubleHelixWithHelix = 5,
    BeamWithCylinders = 6,
    BeamWithSpheres = 7,
    Lightning = 8,
    Analytic = 9,
    EnergyPrim = 10,
    LinesCurved = 11,
  };
  CPlugBeamEmitterSubModel::EDisplayType DisplayType;
  float UVScale;
  uint CylinderSideCount;
  uint Duration;
  float Radius;
  float USpeed;
  float VSpeed;
  bool BackFace;
  float LightningMaxJitterDistPerStep; // Range: 0 - 1
  float LightningJitterRadius; // Range: 0 - 5
  float LightningDistStep; // Range: 0 - 10
  float LightningLenMax;
  uint LightningBeamCount;
  float FadeTimeOffset;
  float FadeTimeScale;
  uint Helix_VerticesCountPerTurn;
  float Helix_TurnLength;
  float Helix_OutterRadius;
  float Helix_InnerRadius;
  bool Helix_ClockWise;
  float Helix_DistLOD;
  float Helix_CoeffLowDef;
  uint Helix_MaxHelixCount;
  float DoubleHelix_StartAngle;
  bool DoubleHelix_ClockWise;
  uint BeamWithCylinders_NbSubDiv;
  float BeamWithCylinders_StepSize;
  float BeamWithCylinders_InnerRadius;
  float BeamWithCylinders_OutterRadius;
  float BeamWithCylinders_HalfWidth;
  float BeamWithCylinders_FadeSpeed;
  float BeamWithCylinders_TimeStartFade;
  float BeamWithCylinders_DistLOD;
  float BeamWithCylinders_CoeffLowDef;
  uint BeamWithSpheres_NbVerticalSubDiv;
  uint BeamWithSpheres_NbSpheresPerMeter;
  float BeamWithSpheres_Radius;
  float BeamWithSpheres_StartDistance;
  float BeamWithSpheres_EndDistance;
  float BeamWithSpheres_DistLOD;
  uint BeamWithSpheres_MaxDetailedRailsCount;
  float BeamWithSpheres_Speed;
  vec3 ColorCenter;
  vec3 ColorBorder;
  float ColorLerpPow; // Range: 1 - 10
  float ColorCenterMultiplier;
  float ColorBorderMultiplier;
  float LaserRadius;
  vec3 TriColorCenter;
  vec3 TriColorBorder;
  float TriColorLerpPow; // Range: 1 - 10
  float TriColorCenterMultiplier;
  float TriColorBorderMultiplier;
  float TriLaserRadius;
  float StartOffsetAtEndLife;
  bool EnableTriLaser;
  bool EnableCenterLaser;
  float SharpIntensity;
  float SharpFadeEndAtEndLife;
  float SharpStartOffset;
  float SharpScale; // Range: 0.1 - 1
  bool SharpEnabled;
  float SharpEnergy;
  bool SharpFadeOut;
  bool SharpShrinkOverLife;
  float BloomIntensity;
  float BloomFadeEndAtEndLife;
  float BloomStartOffset;
  float BloomScale; // Range: 0.1 - 1
  bool BloomEnabled;
  float BloomEnergy;
  bool BloomFadeOut;
  bool BloomShrinkOverLife;
};

// File extension: 'PlugProbe.Gbx'
struct CPlugProbe : public CPlug {
  CPlugProbe();

};

struct CPlugCustomBeamModel : public CMwNod {
  CPlugCustomBeamModel();

  string BulletName;
  float MaxDistance;
  float LaserRadius;
  float LaserRadiusDamage;
  float BulletVsRadius;
  float LaserDispersionAngle;
  CPlugCustomBeamModel::ECustomBeamType BeamType;
  float Blow_Value;
  float Blow_Radius;
  int LaserDamage;
  bool DamageAttenuationWithDist;
  NPlugCurve_SSimpleCurveInPlace7 DamageAttenuationFromDist;
  bool LaserShowAdvancedCrosshair;
  vec3 VisualOffsetFirstPerson;
};

struct CPlugFileWebM : public CPlugFileVideo {
  CPlugFileWebM();

};

struct CPlugCharVisModelCustom : public CMwNod {
  CPlugCharVisModelCustom();

  const bool IsSprite;
};

// userName: 'CamShakeModel'
// File extension: 'CamShakeModel.Gbx'
struct CPlugCamShakeModel : public CMwNod {
  CPlugCamShakeModel();

  UnknownType Params;
};

struct CPlugCamControlModel : public CMwNod {
  CPlugCamShakeModel* Shake;
  uint LaunchedCheckpointStopped_SecondPartDuration;
  float LaunchedCheckpointStopped_FirstPartCoef;
  float LaunchedCheckpointStopped_SecondPartCoef;
};

struct CPlugImportMeshParam : public CMwNod {
  CPlugImportMeshParam();

  float ForceScale;
  bool ForceSmoothNormals;
  string NodePrefix;
  string JointPrefix;
  string MeshSkipPrefix;
  string MeshOnlyPrefix;
  string MeshSkipSuffix;
  string MeshOnlySuffix;
  MwFastBuffer<CPlugMaterialUserInst*> MatUserModels;
  MwFastBuffer<CPlugLightUserModel*> LightUserModels;
  string Skel_GenericRootName;
  string Skel_DummyPrefix;
  string Skel_SocketPrefix;
  bool Skel_SocketOnly;
  uint TextureParams_MaxSize;
  bool TextureParams_HqDds;
  string CollectionName;
  EImportMeshType ImportMeshType;
  string FileNameSuffix;
  bool GenerateSkel;
  EPlugModelShadingType ModelShadingType;
  uint MaxInfluenceCount;
};

struct CPlugVehicleCarPhyShape : public CMwNod {
  CPlugVehicleCarPhyShape();

};

// File extension: 'Turret.Gbx'
struct CPlugTurret : public CMwNod {
  CPlugTurret();

  enum class CPlugTurret::ETurretFixedAngleSignal {
    Constant = 0,
    Linear = 1,
    PingPong = 2,
  };
  enum class CPlugTurret::EOnArmorEmtpy {
    Destroy = 0,
    Disable = 1,
  };
  bool AimEnabled;
  float AimDetectRadius;
  float AimDetectFOVDeg;
  float AimMaxTrackDist;
  float AimAnticipation;
  uint AimKeepAimingDurationMs;
  uint AimFireTargetChangeDelayMs;
  float AimFireMaxAngleDeg;
  float AimFireMaxDist;
  CPlugTurret::ETurretFixedAngleSignal FixedAngleSignal;
  uint FixedAnglePeriodMs;
  float FixedAngleMinDeg;
  float FixedAngleMaxDeg;
  uint LifeArmorMax;
  float LifeDisabledDuration;
  CPlugTurret::EOnArmorEmtpy LifeOnArmorEmtpy;
  bool IsControllable;
  uint FirePeriodMs;
  MwId Joint0Name;
  vec3 Joint0LocalAxis;
  float Joint0MinAngleDeg;
  float Joint0MaxAngleDeg;
  float Joint0SpeedDegPerS;
  float Joint0NextJointUpdateAngleMaxDeg;
  MwId Joint1Name;
  vec3 Joint1LocalAxis;
  float Joint1MinAngleDeg;
  float Joint1MaxAngleDeg;
  float Joint1SpeedDegPerS;
  float Joint1NextJointUpdateAngleMaxDeg;
  MwId JointFireName;
  vec3 JointFireLocalAxis;
  MwId JointRadarName;
  CPlugParticleEmitterModel* OnFireParticle;
  DataRef Skel;
  DataRef Shape;
  DataRef BulletModel;
  DataRef Mesh;
  DataRef RotateSound1;
  DataRef VisEntFx;
  bool IsDyna;
  CPlugDynaModel* DynaModel;
};

struct CPlugVehicleCamInternalVisOffset : public CMwNod {
  CPlugVehicleCamInternalVisOffset();

};

// File extension: 'Shield.Gbx'
struct CPlugShieldModel : public CMwNod {
  CPlugShieldModel();

  bool IsBouncing;
  bool NeedActivation;
  uint ShieldArmor;
  uint ShieldDuration;
  vec3 RelativePos;
  uint TextureNotches;
  DataRef Shape;
  DataRef ShieldVisModel;
  DataRef ShieldActiveSound;
  DataRef ShieldTouchSound;
  DataRef ShieldDestroySound;
  DataRef ShieldTouchParticle;
};

struct CPlugVehicleCameraRaceModel : public CPlugCamControlModel {
  CPlugVehicleCameraRaceModel();

};

struct CPlugVehicleCameraHmdExternalModel : public CPlugCamControlModel {
  CPlugVehicleCameraHmdExternalModel();

};

struct CPlugVehicleVisGeomModel : public CMwNod {
  CPlugVehicleVisGeomModel();

};

// userName: 'VisEntFx'
// File extension: 'VisEntFx.Gbx'
struct CPlugVisEntFxModel : public CMwNod {
  CPlugVisEntFxModel();

  CPlugParticleEmitterModel* Part_Deactivated;
  CPlugParticleEmitterModel* Part_DeactivatedShot;
  CPlugParticleEmitterModel* Part_TeleportSpawn;
  CPlugParticleEmitterModel* Part_TeleportUnspawn;
  NPlugCurve_SSimpleCurveInPlace7 EliminatedTeleportIntens;
  NPlugCurve_SSimpleCurveInPlace7 EliminatedHighlight;
  vec3 EliminatedHighlightRgb;
  CGxLightBall* TacticalLight;
};

struct CPlugGpuBuffer : public CMwNod {
  CPlugGpuBuffer();

};

// userName: 'PolyLine3'
struct CPlugPolyLine3 : public CMwNod {
  CPlugPolyLine3();

  MwFastBuffer<vec3> Poss;
  MwFastBuffer<vec3> Lefts;
};

struct CPlugPath : public CMwNod {
  CPlugPath();

};

struct CPlugMoodBlender : public CPlug {
  CPlugMoodBlender();

};

// userName: 'TrainModel'
// File extension: 'Train.Gbx'
struct CPlugTrainModel : public CMwNod {
  CPlugTrainModel();

  MwFastBuffer<CPlugTrainWagonModel*> Wagons;
};

// userName: 'WagonModel'
// File extension: 'Wagon.gbx'
struct CPlugTrainWagonModel : public CMwNod {
  CPlugTrainWagonModel();

  CPlugSurface* Shape;
  CPlugSolid2Model* Mesh;
  CPlugSkel* Skel;
  CPlugAnimFile* AnimFile;
  CPlugSound* Sound_Engine;
  CPlugSound* Sound_Brake;
  CPlugSound* Sound_RailContact;
  CPlugSound* Sound_Collision;
  CPlugParticleEmitterModel* SmokeEmitterModel;
  CPlugParticleEmitterModel* DustEmitterModel;
  CPlugParticleEmitterModel* SparkleParticle;
  bool IsLoco;
  bool IsAutopiloted;
  NPlugCurve_SSimpleCurveInPlace7 AccelCurve;
  float LinearMass;
  bool Use_TM_Simulation;
  vec3 CenterOfMass;
  float WagonLength;
  float WagonColOffset;
  bool GenLengthFromShape;
};

struct CPlugMoodCurve : public CPlug {
};

// userName: 'VehiclePhyModelCustom'
// File extension: 'VehiclePhyModelCustom.Gbx'
struct CPlugVehiclePhyModelCustom : public CMwNod {
  CPlugVehiclePhyModelCustom();

  const float AccelCoef;
  const float ControlCoef;
  const float GravityCoef;
};

struct CPlugEntRecordData : public CMwNod {
  CPlugEntRecordData();

};

// userName: 'WagonModelCustom'
// File extension: 'WagonCustom.Gbx'
struct CPlugTrainWagonModelCustom : public CMwNod {
  CPlugTrainWagonModelCustom();

  float Speed;
  float Accel;
  float Mass;
};

// File extension: 'EntitySpawner.Gbx'
struct CPlugEntitySpawner : public CMwNod {
  CPlugEntitySpawner();

  UnnamedEnum EntityType;
  CMwNod* Model;
  CMwNod* ModelRef;
  iso4 Loc;
};

struct CPlugWeather_DayTimeElem_Compat : public CMwNod {
  CPlugWeather_DayTimeElem_Compat();

};

struct CPlugWeather_WindBlockerElem : public CMwNod {
  CPlugWeather_WindBlockerElem();

};

struct CPlugDynaConstraintModel : public CMwNod {
  CPlugDynaConstraintModel();

  NPlugDyna_SConstraintModel Data;
};

struct CPlugRoadChunk : public CMwNod {
  CPlugRoadChunk();

  NPlugRoadChunk_SParams Params;
  MwFastBuffer<vec3> CenterVerts;
  MwFastBuffer<vec3> RightVerts;
  MwFastBuffer<vec3> LeftVerts;
  GmBoxAligned AABB;
  MwFastBuffer<vec3> LinkVerts;
  SPlugTrafficLight TrafficLight;
};

struct CPlugAnimGraph : public CMwNod {
  CPlugAnimGraph();

  MwId Name;
  MwFastBuffer<CPlugAnimGraphState> States;
  MwFastBuffer<CPlugAnimGraphTransition> Transitions;
  MwId DefaultStateName;
  EPlugAnimGraphBlending Blending;
};

struct SPlugVehicleOccupantSpawn {
  vec3 Pos;
  uint iSlot;
};

struct SPlugVehicleOccupantSlot {
};

struct CPlugVehicleWheelPhyModel {
  MwId Id;
  bool IsDriving;
  bool IsSteering;
};

struct CPlugDynaModel : public CMwNod {
  CPlugDynaModel();

  float LinearMass;
  float MaxDistPerStep;
  vec3 CenterOfMass;
  float AngularSpeedClamp;
  bool Use_TM_Simulation;
  EPlugDynaSleepingMethod SleepingMethod;
  bool EnableSubStepping;
  CPlugDynaWaterModel* WaterModel;
};

// userName: 'AdnPart'
// File extension: 'AdnPart.gbx'
struct CPlugAdnPart : public CMwNod {
  CPlugAdnPart();

  CPlugSolid2Model* Mesh;
  MwFastBuffer<NPlugAdn_STag> RequiredTags;
  MwFastBuffer<NPlugAdn_STag> Tags;
  CPlugMetaData* BaseShader;
  MwFastBuffer<CPlugMetaData*> DefaultShaders;
  float Proba;
  CPlugSkel* Skel;
};

struct CPlugAnimClipBaked : public CMwNod {
  CPlugAnimClipBaked();

  CPlugAnimTimingFixedPeriod Timing;
  CPlugAnimChannelGroup* ChannelGroup;
  CPlugAnimClipFlags Flags;
  MwFastBuffer<uint8> Data;
  MwFastBuffer<GmTransYaw> RootMotionFrames;
};

struct CPlugAnimChannelGroup : public CMwNod {
  CPlugAnimChannelGroup();

  MwSArray<MwId> ChannelNames;
  MwSArray<float> ChannelWeights;
  UnknownType ChannelTypeRanges;
  CPlugSkel* OldSkel;
  MwFastBuffer<uint16> OldJointIndexs;
};

struct CPlugAnimClipEdition : public CMwNod {
  CPlugAnimClipEdition();

  CPlugSkel* Skel;
};

// userName: 'AnimClip'
// File extension: 'AnimClip.Gbx'
struct CPlugAnimClip : public CMwNod {
  CPlugAnimClip();

  CPlugAnimClipBaked* Baked;
  CPlugAnimClipEdition* Edition;
  CPlugAnimClipFlags Flags;
};

struct CPlugAnimPoseGrid : public CMwNod {
  CPlugAnimPoseGrid();

  MwId Name;
  CPlugAnimChannelGroup* ChannelGroup;
  MwFastBuffer<float> Xs;
  MwFastBuffer<float> Ys;
  MwFastBuffer<quat> RefPoseGlobalJoints;
  MwFastBuffer<quat> JointOffsets;
  MwId RefPoseName;
  MwFastBuffer<MwId> PoseNames;
};

struct CPlugAnimPoseGroup : public CMwNod {
  CPlugAnimPoseGroup();

  MwId Name;
  CPlugAnimChannelGroup* ChannelGroup;
  uint cPose;
  MwFastBuffer<vec2> PoseCoords;
  MwFastBuffer<MwId> PoseNames;
  bool IsDifference;
  MwId DifferenceFromClip;
};

struct CPlugAnimGraphStack : public CMwNod {
  CPlugAnimGraphStack();

  MwId Name;
  MwFastBuffer<CPlugAnimGraph*> Graphs;
};

struct CPlugAnimClipEditionPose : public CMwNod {
  CPlugAnimClipEditionPose();

};

struct CPlugAnimVariantGroup : public CMwNod {
  CPlugAnimVariantGroup();

};

struct CPlugAnimSpotModel {
  MwId Name;
  EPlugAnimSpotType SpotType;
};

// userName: 'AdnModel'
// File extension: 'AdnModel.gbx'
struct CPlugAdnModel : public CMwNod {
  CPlugAdnModel();

  MwFastBuffer<CPlugAdnPartInstance> PartInstances;
  MwFastBuffer<NPlugAdn_STag> Tags;
};

// userName: 'AdnProject'
// File extension: 'AdnProject.gbx'
struct CPlugAdnProject : public CMwNod {
  CPlugAdnProject();

  CSystemFidsFolder* RootFolder;
  CPlugMetaData* DefaultPartShader;
  CPlugMetaData* DefaultSkinShader;
  CPlugAnimImport* DefaultClipImport;
  string TextureNameStopAtString;
  MwFastBuffer<NPlugAdn_STagFromName> TagsFromNames;
  uint PartShaderDefaultCount;
  MwFastBuffer<CPlugAdnRandomGroup*> RandomGroups;
};

// userName: 'AdnRandomGen'
// File extension: 'AdnRandomGen.gbx'
struct CPlugAdnRandomGen : public CMwNod {
  CPlugAdnRandomGen();

  MwId Id;
  uint RandSeed;
  uint cModel;
  uint cDispModel;
  MwArrayInPlaceDyn<SPlugAdnRandomGenSet> Sets;
};

struct CPlugAnimImport : public CMwNod {
  CPlugAnimImport();

  NPlugAnim_SImportParams Params;
  MwFastBuffer<NPlugAnim_SSkelImport> Skels;
  MwFastBuffer<NPlugAnim_SChannelGroupImport> ChannelGroups;
  void ClipsDisableAll();
  MwFastBuffer<NPlugAnim_SClipImport> Clips;
  MwFastBuffer<NPlugAnim_SPoseGridImport> PoseGrids;
};

// userName: 'PlugMaterial_VertexIndex'
// File extension: 'MaterialVId.Gbx'
struct CPlugMaterial_VertexIndex : public CMwNod {
  CPlugMaterial_VertexIndex();

  CPlugMaterial* MaterialInstance;
  CPlugMaterial* MaterialModel;
  CPlugBitmap* BaseColor;
  CPlugBitmap* Normal;
  CPlugBitmap* RoughMetal;
  CPlugBitmap* BlendMask;
  CPlugBitmap* PyX2;
  CPlugBitmap* PyH2;
  MwSArray<CPlugMaterial*> Materials;
};

struct CPlugAdnRandomGroup : public CMwNod {
  CPlugAdnRandomGroup();

  string Name;
  MwFastBuffer<NPlugAdn_STag> Tags;
  MwFastBuffer<SPlugRandomMetaData> RandomMetaDatas;
};

// userName: 'PlugDynaObjectModel'
// File extension: 'DynaObject.Gbx'
struct CPlugDynaObjectModel : public CMwNod {
  CPlugDynaObjectModel();

  bool IsStatic;
  bool DynamizeOnSpawn;
  CPlugSolid2Model* Mesh;
  CPlugSurface* StaticShape;
  CPlugSurface* DynaShape;
  CPlugAnimLocSimple* LocAnim;
  bool LocAnimIsPhysical;
  float BreakSpeedKmh; // Range: 0 - 200
  float Mass; // Range: 1 - 1000
  float LightAliveDurationSc_Min; // Range: 0 - 10
  float LightAliveDurationSc_Max; // Range: 0 - 10
  CPlugDynaWaterModel* WaterModel;
  float StaticShape_BoxSizeX;
  float StaticShape_BoxSizeY;
  float StaticShape_BoxSizeZ;
  float DynaShape_BoxSizeX;
  float DynaShape_BoxSizeY;
  float DynaShape_BoxSizeZ;
  GmBoxAligned StaticShape_AABB;
};

// userName: 'Prefab'
// File extension: 'Prefab.Gbx'
struct CPlugPrefab : public CMwNod {
  CPlugPrefab();

  MwFastBuffer<NPlugPrefab_SEntRef> Ents;
  NFastBucketAlloc_SAllocator ParamsData;
};

// userName: 'VehicleStyles'
// File extension: 'VehicleStyles.gbx'
struct CPlugVehicleVisStyles : public CMwNod {
  CPlugVehicleVisStyles();

  MwFastBuffer<CPlugVehicleVisStyleRandomGroup*> RandomGroups;
  string UnwantedTrafficVehiclesMatchName;
};

struct CPlugVehicleVisStyleRandomGroup : public CMwNod {
  CPlugVehicleVisStyleRandomGroup();

  MwId Name;
  string MatchName;
  float GroupProba;
  MwFastBuffer<SPlugVehicleVisStyle> Styles;
  void SetDefault();
};

struct CPlugCitizenModel : public CMwNod {
  CPlugCitizenModel();

  CPlugSolid2Model* Mesh;
  CPlugAnimFile* Anim;
};

// userName: 'FxAnimFromTexture1dArray'
// File extension: 'FxAnimFromTex1DArray.Gbx'
struct CPlugFxAnimFromTexture1dArray : public CMwNod {
  CPlugFxAnimFromTexture1dArray();

  MwFastBuffer<NPlugFxAnimFromTexture1dArray_SElem> Elems;
};

// userName: 'TagFidCache'
struct CPlugAdnTagFidCache : public CMwNod {
  CPlugAdnTagFidCache();

  MwFastBuffer<NPlugAdn_STagFid> TagFids;
};

// userName: 'MetaData'
// File extension: 'MetaData.gbx'
struct CPlugMetaData : public CMwNod {
  CPlugMetaData();

  CPlugMetaData* ParentData;
};

// userName: 'PlugImageArray'
// File extension: 'ImageArray.Gbx'
struct CPlugImageArray : public CMwNod {
  CPlugImageArray();

  CSystemFidsFolder* m_Folder;
  CSystemFidsFolder* m_Folder_TextureDecals;
  CPlugMaterial_VertexIndex* Material_VId;
  float m_MaskScale; // Range: 1 - 4
  bool m_MaskSmooth;
  uint m_BlendMaxCount;
  MwFastBuffer<NPlugImageArray_SElem> Layers;
};

struct CPlugVehicleCameraHelicoModel : public CPlugCamControlModel {
  CPlugVehicleCameraHelicoModel();

};

struct CPlugRecastPolyMeshData : public CMwNod {
  CPlugRecastPolyMeshData();

};

// userName: 'VFXFile'
// File extension: 'VFX.Gbx'
struct CPlugVFXFile : public CMwNod {
  CPlugVFXFile();

  MwClassId ContextClassId;
  CPlugVFXNode_Graph Graph;
};

struct CPlugSymlink : public CMwNod {
  CPlugSymlink();

  wstring Path;
};

// userName: 'UIConfig'
// File extension: 'gbx.json'
struct CPlugAnimRigUIConfig : public CMwNod {
  CPlugAnimRigUIConfig();

  MwFastBuffer<SPlugAnimUIRigNodeTypeConfig> TypeConfigs;
};

// userName: 'AdnRandomGenList'
// File extension: 'AdnRandomGenList.gbx'
struct CPlugAdnRandomGenList : public CMwNod {
  CPlugAdnRandomGenList();

  MwFastBuffer<CPlugAdnRandomGen*> Datas;
};

struct CPlugLightDyna {
  RGBAColor Color_Srgb;
  float Intensity;
  float DiffuseIntensity;
};

// userName: 'StaticObjectModel'
// File extension: 'StaticObject.Gbx'
struct CPlugStaticObjectModel : public CMwNod {
  CPlugStaticObjectModel();

  CPlugSolid2Model* Mesh;
  CSystemFidFile* const MeshFidForReload;
  CPlugSurface* Shape;
  CSystemFidFile* const ShapeFidForReload;
};

// userName: 'LightmapCustom'
// File extension: 'LightmapCustom.Gbx'
struct CPlugLightMapCustom : public CMwNod {
  CPlugLightMapCustom();

  CSystemFidsFolder* Folder;
};

// userName: 'FxSystem'
// File extension: 'FxSys.Gbx'
struct CPlugFxSystem : public CMwNod {
  CPlugFxSystem();

  MwClassId const ContextClassId;
  MwClassId const ExtraContextClassId;
  CPlugFxSystemNode* const RootNode;
  MwFastBuffer<SPlugGraphVar> Vars;
};

struct CPlugGameSkinAndFolder : public CMwNod {
  CPlugGameSkinAndFolder();

  MwId IdName;
  CPlugGameSkin* Remapping;
  CPlugGameSkin* Remapping_NoTrackWall_Cache;
  CSystemFidsFolder* RemapFolder;
};

// userName: 'MaterialColorTable'
// File extension: 'gbx.json'
struct CPlugMaterialColorTargetTable : public CMwNod {
  CPlugMaterialColorTargetTable();

  UnknownType Colors;
  UnknownType Colors_Blind;
  UnknownType TM_Stunt;
  UnknownType TM_Stunt_Blind;
  UnknownType Classic;
  UnknownType Classic_ColorBlind;
  UnknownType Stunt;
  UnknownType Stunt_ColorBlind;
  UnknownType Red;
  UnknownType Red_ColorBlind;
  UnknownType Orange;
  UnknownType Orange_ColorBlind;
  UnknownType Yellow;
  UnknownType Yellow_ColorBlind;
  UnknownType Lime;
  UnknownType Lime_ColorBlind;
  UnknownType Green;
  UnknownType Green_ColorBlind;
  UnknownType Cyan;
  UnknownType Cyan_ColorBlind;
  UnknownType Blue;
  UnknownType Blue_ColorBlind;
  UnknownType Purple;
  UnknownType Purple_ColorBlind;
  UnknownType Pink;
  UnknownType Pink_ColorBlind;
  UnknownType White;
  UnknownType White_ColorBlind;
  UnknownType Black;
  UnknownType Black_ColorBlind;
};

struct CPlugDynaWaterModel : public CMwNod {
  CPlugDynaWaterModel();

};

// userName: 'PlacementPatch'
// File extension: 'PlacementPatch.Gbx'
struct CPlugPlacementPatch : public CPlugRoadChunk {
  CPlugPlacementPatch();

};

// userName: 'RoadChunkTraffic'
// File extension: 'TrafficRoadChunk.Gbx'
struct CPlugRoadChunkTraffic : public CPlugRoadChunk {
  CPlugRoadChunkTraffic();

};

// userName: 'RoadChunkCitizen'
// File extension: 'CitizenRoadChunk.Gbx'
struct CPlugRoadChunkCitizen : public CPlugRoadChunk {
  CPlugRoadChunkCitizen();

};

struct CPlugAnimJointExprGroup : public CMwNod {
  CPlugAnimJointExprGroup();

  MwFastBuffer<NPlugAnim_SJointExpr> JointExprs;
};

// userName: 'ModelKitDataBaseDesc'
// File extension: 'gbx.json'
struct NPlugModelKit_SDataBaseDesc : public CMwNod {
  NPlugModelKit_SDataBaseDesc();

  MwSArray<NPlugModelKit_SPartPaths> Parts;
  MwSArray<NPlugModelKit_SOptionDesc> Options;
  MwSArray<NPlugModelKit_SOptionValDesc> DefaultOptions;
  MwSArray<NPlugModelKit_SOptionArrayDesc> IgnoreOptions;
  MwSArray<NPlugModelKit_SOptionArrayDesc> DevOptions;
  MwSArray<NPlugModelKit_SModelDesc> Models;
  MwSArray<SConstString> Resources;
};

// userName: 'ModelKitDataBase'
// File extension: 'gbx'
struct NPlugModelKit_SDataBase : public CMwNod {
  NPlugModelKit_SDataBase();

  MwSArray<NPlugModelShading_SMaterialFiles> Materials;
  MwSArray<NPlugModelKit_SPartFiles> PartFiles;
  MwSArray<NPlugModelKit_SPartFilesVersion> AllPartVersions;
  MwSArray<NPlugModelKit_SPartFilesVersions> Parts;
  MwSArray<NPlugModelKit_SOption> Options;
  MwSArray<NPlugModelKit_SOptionVal> DefaultOptions;
  MwSArray<NPlugModelKit_SOptionArray> IgnoreOptions;
  MwSArray<NPlugModelKit_SOptionArray> DevOptions;
  MwSArray<CSystemFidFile*> Resources;
  MwSArray<NPlugModelKit_SModelFiles> Models;
};

struct NPlugTrigger_SWaypoint : public CMwNod {
  NPlugTrigger_SWaypoint();

  EGameItemWaypointType Type;
  CPlugSurface* TriggerShape;
  bool NoRespawn;
};

struct NPlugTrigger_SSpecial : public CMwNod {
  NPlugTrigger_SSpecial();

  CPlugSurface* TriggerShape;
  bool IsMergeable;
};

// userName: 'Spawn'
// File extension: 'Spawn.gbx'
struct CPlugSpawnModel : public CMwNod {
  CPlugSpawnModel();

  iso4 Loc;
  float TorqueX;
  uint TorqueDuration;
  vec3 DefaultGravitySpawn;
};

// userName: 'EditorHelper'
// File extension: 'EditorHelper.gbx'
struct CPlugEditorHelper : public CMwNod {
  CPlugEditorHelper();

  CSystemFidFile* PrefabFid;
};

struct SPlugFlockEmitterState {
  uint SpawnCount;
  float Radius;
  float Power;
  bool IsRepulsor;
  bool IsLandingArea;
  iso4 Loc;
};

// userName: 'MaterialWaterArray'
// File extension: 'WaterArray.gbx'
struct CPlugMaterialWaterArray : public CMwNod {
  CPlugMaterialWaterArray();

  MwFastBuffer<NPlugMaterial_SWater> Waters;
};

// File extension: 'PlugWeather.Gbx'
struct CPlugWeather : public CMwNod {
  CPlugWeather();

  CPlugClouds* Clouds;
  CPlugMaterial* MaterialSky_Night;
  CPlugMaterial* MaterialSky_SunRise;
  CPlugMaterial* MaterialSky_Day;
  CPlugMaterial* MaterialSky_SunFall;
  CPlugMaterial* MaterialSea0;
  CPlugMaterial* MaterialSea1;
  CPlugFileImg* ImageLightAmb;
  CPlugFileImg* ImageLightDirSun;
  CPlugFileImg* ImageLightDirMoon;
  CPlugFileImg* ImageLightDirDblSided;
  CPlugBitmap* BitmapFlareSun;
  float FlareAngularSizeSun;
  CPlugBitmap* BitmapFlareMoon;
  float FlareAngularSizeMoon;
  CPlugBitmap* BitmapSkyGradV;
  CPlugFileImg* ImageFogColor;
  CPlugFileImg* ImageSeaColor;
  CPlugParticleEmitterModel* EmitterInFog;
  CPlugBitmap* BitmapRainFid;
  CMwNod* SceneFxFid;
  float CameraFarZ;
  bool FogByVertex;
  vec3 FogRGB;
  enum class CPlugWeather::EGxFogFormula {
    None = 0,
    Exp = 1,
    Exp2 = 2,
    Linear = 3,
  };
  CPlugWeather::EGxFogFormula FogFormula;
  enum class CPlugWeather::EGxFogSpace {
    CameraFarZ = 0,
    World = 1,
  };
  CPlugWeather::EGxFogSpace FogSpace;
  float FogLinearStart;
  float FogLinearEnd;
  float FogExpDensity;
  vec3 DayFogColor;
  float DayFogStart;
  float DayFogEnd;
  float DayFogDensity;
  GxFogBlender* FogBlender;
  void SeaTwkResetToShaderDefaults();
  float SeaTwkReflecIntensNight; // Range: 0 - 1
  float SeaTwkReflecIntensDay; // Range: 0 - 1
  float SeaTwkReflecIntensMidNight; // Range: 0 - 1
  float SeaTwkReflecIntensMidDay; // Range: 0 - 1
  float SeaTwkReflecIntensTMNight; // Range: 0 - 1
  float SeaTwkReflecIntensTMDay; // Range: 0 - 1
  float SeaTwkReflecIntensTMmidNight; // Range: 0 - 1
  float SeaTwkReflecIntensTMmidDay; // Range: 0 - 1
  vec3 SeaTwkWaterColor_Night;
  vec3 SeaTwkWaterColor_Day;
};

// File extension: 'PlugPuffLull.Gbx'
struct CPlugPuffLull : public CMwNod {
  CPlugPuffLull();

  float TileSizeInWorld;
  float PuffWDMax; // Range: 0 - 1
  float LullWDMax; // Range: 0 - 1
  uint GenCount;
  float GenPuffRatio; // Range: 0 - 1
  vec2 GenSize;
  float GenPuffWDMin; // Range: 0 - 1
  float GenPuffWDMax; // Range: 0 - 1
  float GenLullWDMin; // Range: 0 - 1
  float GenLullWDMax; // Range: 0 - 1
  string GenLifeTimeMin;
  string GenLifeTimeMax;
  float BlendPuff; // Range: 0 - 1
  float BlendLull; // Range: 0 - 1
  float Combine2nd8th; // Range: 0 - 1
  CPlugMaterial* MaterialPuff;
  CPlugMaterial* MaterialLull;
};

// File extension: 'PlugClouds.Gbx'
struct CPlugClouds : public CMwNod {
  CPlugClouds();

  CPlugCloudsSolids* Solids;
  CPlugFileImg* ImageColorMin;
  CPlugFileImg* ImageColorMax;
  CPlugMaterial* MaterialModelToForce;
  CPlugCloudsParam* CloudsParam;
};

struct CPlugDayTime : public CMwNod {
  CPlugDayTime();

};

// File extension: 'PlugCloudsParam.Gbx'
struct CPlugCloudsParam : public CMwNod {
  CPlugCloudsParam();

  UnnamedEnum HeightCenter;
  vec2 HeightCenterXZ;
  float BottomNearZ;
  MwFastBuffer<float> PointDists;
  MwFastBuffer<float> PointHeights;
  float BottomFarZ;
  float SpeedScale;
  UnnamedEnum Lighting;
  bool MaterialUseT3b;
};

// File extension: 'PlugCloudsSolids.Gbx'
struct CPlugCloudsSolids : public CMwNod {
  CPlugCloudsSolids();

  vec2 GridSizeXZ;
  const MwFastBuffer<CPlugSolid*> SolidFids;
};

struct CPlugCurveEnvelopeDeprec : public CMwNod {
  CPlugCurveEnvelopeDeprec();

};

// userName: 'CurveSimple'
// File extension: 'gbx'
struct CPlugCurveSimpleNod : public CMwNod {
  CPlugCurveSimpleNod();

  NPlugCurve_SSimpleCurveInPlace7 Curve;
};

// userName: 'BitmapArrayBuilder'
// File extension: 'BitmapArrayBuilder.Gbx'
struct CPlugBitmapArrayBuilder : public CMwNod {
  CPlugBitmapArrayBuilder();

  MwFastBuffer<CPlugBitmap*> BitmapsInArray;
  CPlugBitmap* GeneratedArray;
  nat3 ArrayTexelDims;
};

// userName: 'ItemPlacementClass'
// File extension: 'ItemPlacementClass.Gbx'
struct NPlugItemPlacement_SClass : public CMwNod {
  NPlugItemPlacement_SClass();

  MwFastBuffer<MwId> CompatibleIdGroups;
  MwId SizeGroup;
  bool AlwaysUp;
  bool AlignToInterior;
  bool AlignToWorldDir;
  vec3 WorldDir;
  MwFastBuffer<NPlugItemPlacement_SPatchLayout> PatchLayouts;
  MwFastBuffer<uint> GroupCurPatchLayouts;
  uint CurVariant;
};

// userName: 'Podium'
// File extension: 'Podium.gbx'
struct CPlugPodium : public CMwNod {
  CPlugPodium();

  CPlugMediaClipList* ClipList;
};

// File extension: 'MediaClipList.Gbx'
struct CPlugMediaClipList : public CMwNod {
  CPlugMediaClipList();

  const MwFastBuffer<CMwNod*> MediaClipFids;
};

// userName: 'Rig'
// File extension: 'Rig.gbx'
struct NPlugAnim_SRig : public CMwNod {
  NPlugAnim_SRig();

  MwSArray<MwId> ControlNames;
  MwSArray<uint> ControlPoseOffsets;
  uint PoseByteSize;
  NPlugAnim_SRigPose RefPose;
  MwFastArray<uint8> Data;
};

// userName: 'RigToSkel'
// File extension: 'RigToSkel.gbx'
struct NPlugAnim_SRigToSkel : public CMwNod {
  NPlugAnim_SRigToSkel();

  MwFastBuffer<NPlugAnim_SRigToSkelNode*> Nodes;
};

struct NPlugAnim_SRigToSkelNode : public CMwNod {
  NPlugAnim_SRigToSkelNode();

};

struct NPlugAnim_SRigToSkelNode_Chain2 : public NPlugAnim_SRigToSkelNode {
  NPlugAnim_SRigToSkelNode_Chain2();

  MwId RigControl_Target;
  MwId RigControl_Pole;
  MwId SkelJoint_Begin;
  MwId SkelJoint_End;
};

struct NPlugAnim_SRigToSkelNode_Fixed : public NPlugAnim_SRigToSkelNode {
  NPlugAnim_SRigToSkelNode_Fixed();

  MwId SkelJoint;
};

struct NPlugAnim_SRigToSkelNode_SetDOV : public NPlugAnim_SRigToSkelNode {
  NPlugAnim_SRigToSkelNode_SetDOV();

  MwId RigControl_DOV;
  MwId RigControl_Left;
  MwId SkelJoint;
};

struct NPlugAnim_SRigToSkelNode_SetPos : public NPlugAnim_SRigToSkelNode {
  NPlugAnim_SRigToSkelNode_SetPos();

  MwId RigControl_Pos;
  MwId SkelJoint;
};

} // namespace Plug

namespace Scene {

struct CSceneEngine : public CMwEngine {
  CSceneEngine();

};

struct CScene : public CMwNod {
  MwFastBuffer<CSceneMobil*> Mobils;
  MwFastBuffer<CSceneLight*> Lights;
  ISceneVis* const SceneVis;
  IScenePhy* const ScenePhy;
  IScenePhy* const ServerScenePhy;
  CMwNod* const MoodBlender;
  NHmsZone_SPImp* const HmsZonePImp;
  CSceneSector* Sector;
};

// File extension: 'Scene2d.Gbx'
struct CScene2d : public CScene {
  CScene2d();

  CSceneSector* const Sector;
  vec2 OverlayMin;
  vec2 OverlayMax;
  MwFastBuffer<CSceneLight*> Lights;
};

struct CSceneLayout : public CMwNod {
  CSceneLayout();

  MwFastBuffer<NSceneLayout_SLight> Lights;
  MwFastBuffer<NSceneLayout_SItem> Items;
  CSystemFidFile* WeatherModel;
  float CameraFarZ;
  float EdFocusZ;
  float EdLensSize;
  CSceneFxNod* SceneFxNod;
  CPlugBitmap* BitmapWaterFog;
  CPlugBitmap* BitmapCubeReflectHardSpecA;
  CPlugBitmap* BitmapCubeReflectHdrAlpha2;
};

struct CSceneSector : public CMwNod {
  CSceneSector();

  CHmsZone* Zone;
  CScene* const Scene;
};

struct CSceneObject : public CMwNod {
  CScene* Scene;
  void SetLocation(iso4 Location, CSceneSector* Sector);
};

struct CSceneMgrGUI : public CMwNod {
};

struct CSceneLocation : public CSceneObject {
  CSceneLocation();

  iso4 Location;
};

struct CScenePoc : public CSceneObject {
  CHmsPoc* HmsPoc;
  bool IsActive;
};

// File extension: 'SceneLight.Gbx'
struct CSceneLight : public CScenePoc {
  CSceneLight();

  GxLight* Light;
};

struct CSceneVehicleVisState {
  float InputSteer; // Range: -1 - 1
  bool InputIsBraking;
  float InputGasPedal; // Range: 0 - 1
  float InputBrakePedal; // Range: 0 - 1
  float InputVertical; // Range: -1 - 1
  ESceneVehicleVisReactorBoostLvl ReactorBoostLvl;
  ESceneVehicleVisReactorBoostType ReactorBoostType;
  vec3 ReactorAirControl;
  bool ReactorInputsX;
  vec3 WorldCarUp;
  bool IsGroundContact;
  bool IsReactorGroundMode;
  bool IsWheelsBurning;
  float GroundDist; // Range: -1 - 15
  const EPlugSurfaceMaterialId FLGroundContactMaterial;
  float FLSteerAngle; // Range: -0.523599 - 0.523599
  float FLWheelRot; // Range: 0 - 6.28319
  float FLWheelRotSpeed;
  float FLDamperLen; // Range: 0 - 2
  float FLSlipCoef; // Range: 0 - 1
  float FLIcing01; // Range: 0 - 1
  float FLTireWear01; // Range: 0 - 1
  float FLBreakNormedCoef; // Range: 0 - 1
  const EPlugSurfaceMaterialId FRGroundContactMaterial;
  float FRSteerAngle; // Range: -0.523599 - 0.523599
  float FRWheelRot; // Range: 0 - 6.28319
  float FRWheelRotSpeed;
  float FRDamperLen; // Range: 0 - 2
  float FRSlipCoef; // Range: 0 - 1
  float FRIcing01; // Range: 0 - 1
  float FRTireWear01; // Range: 0 - 1
  float FRBreakNormedCoef; // Range: 0 - 1
  const EPlugSurfaceMaterialId RRGroundContactMaterial;
  float RRSteerAngle; // Range: -0.523599 - 0.523599
  float RRWheelRot; // Range: 0 - 6.28319
  float RRWheelRotSpeed;
  float RRDamperLen; // Range: 0 - 2
  float RRSlipCoef; // Range: 0 - 1
  float RRIcing01; // Range: 0 - 1
  float RRTireWear01; // Range: 0 - 1
  float RRBreakNormedCoef; // Range: 0 - 1
  const EPlugSurfaceMaterialId RLGroundContactMaterial;
  float RLSteerAngle; // Range: -0.523599 - 0.523599
  float RLWheelRot; // Range: 0 - 6.28319
  float RLWheelRotSpeed;
  float RLDamperLen; // Range: 0 - 2
  float RLSlipCoef; // Range: 0 - 1
  float RLIcing01; // Range: 0 - 1
  float RLTireWear01; // Range: 0 - 1
  float RLBreakNormedCoef; // Range: 0 - 1
  uint CurGear;
  bool EngineOn;
  bool IsTurbo;
  float TurboTime;
  uint RaceStartTime;
  vec3 Position;
  const vec3 Left;
  const vec3 Up;
  const vec3 Dir;
  vec3 WorldVel;
  float FrontSpeed;
  float BulletTimeNormed; // Range: 0 - 1
  float SimulationTimeCoef;
  float AirBrakeNormed; // Range: 0 - 1
  float SpoilerOpenNormed; // Range: 0 - 1
  float WingsOpenNormed; // Range: 0 - 1
  bool IsTopContact;
  float WetnessValue01;
  float WaterImmersionCoef;
  float WaterOverDistNormed;
  vec3 WaterOverSurfacePos;
  uint8 DiscontinuityCount;
  UnknownType CamGrpStates;
};

struct CSceneCharVisState {
  vec3 Pos;
  quat RotRef;
  float Walk;
  float Strafe;
  vec3 Vel;
  vec3 WishMove;
  UnnamedEnum CharPhyState;
  const UnnamedEnum CharPhyState2;
  float AimPitch; // Range: -1.5708 - 1.5708
  float AimYaw; // Range: -3.14159 - 3.14159
  bool Speaking;
  bool HackIsInternalCam;
  bool IsAttractor1;
  bool IsAttractor2;
  vec3 AttractorPos1;
  vec3 AttractorPos2;
  EMeleeState MeleeState;
  EMeleeHitType MeleeHitType;
  float SnapAimYawMeleeFight; // Range: -3.14159 - 3.14159
  uint StunnedDuration;
  EParryType ParryType;
  float WaterLevel;
  EPlugSeatType SeatType;
  int8 SeatTypeNum;
  float Tiredness;
};

// File extension: 'Mobil.Gbx'
struct CSceneMobil : public CSceneObject {
  CSceneMobil();

  CSceneMobil* const Model;
  bool IsVisible;
  CHmsItem* Item;
  CPlugSolid* Solid;
  CPlugSolid* SolidModel;
  void Show();
  void Hide();
  uint CastedShadows;
  bool SelfShadow;
};

struct CSceneCloudSystem : public CMwNod {
  bool IsVisible;
  bool AutoSizeFarZ;
  float ScaleFarZ_Visibility; // Range: 0 - 1
  uint InstCountX;
  uint InstCountZ;
  float WindVortexSpeed;
  bool WindVortexUseHyperbolicSpeed;
  float WindSpeed;
  float WindDir; // Range: 0 - 6.28
  bool FadeAlpha;
  bool ForceSize;
  vec2 GridSizeXZ;
  bool MaterialUseT3b;
  const MwFastBuffer<CPlugShader*> Shaders;
  const UnnamedEnum OutputMode;
};

struct CScenePickerManager : public CMwNod {
  CHmsPicker* const Picker;
};

struct CSceneVehicleVis {
  CSceneVehicleVisState* AsyncState;
  CPlugVehicleVisModel* Model;
  float Turbo; // Range: 0 - 1
};

struct CSceneCharVis {
  CSceneCharVisState State;
};

// File extension: 'SceneFx.Gbx'
struct CSceneFxColors : public CSceneFxCompo {
  CSceneFxColors();

  float ParamInverseRGB; // Range: 0 - 1
  float ParamHue; // Range: 0 - 1
  float ParamSaturation; // Range: -1 - 1
  float ParamBrightness; // Range: -1 - 1
  float ParamContrast; // Range: -1 - 1
  vec3 ParamModulateRGB;
  float ParamModulateR; // Range: 0 - 1
  float ParamModulateG; // Range: 0 - 1
  float ParamModulateB; // Range: 0 - 1
  vec3 ParamBlendRGB;
  float ParamBlendAlpha; // Range: 0 - 1
  bool ParamUserEnable;
  float UserIntensity; // Range: 0 - 1
  float CloudsIntensity; // Range: 0 - 1
  float ZFarIntensity; // Range: 0 - 1
  float ParamZFar_StartZ;
  float ParamZFar_StopZ;
  float ParamZFarInverseRGB; // Range: 0 - 1
  float ParamZFarHue; // Range: 0 - 1
  float ParamZFarSaturation; // Range: -1 - 1
  float ParamZFarBrightness; // Range: -1 - 1
  float ParamZFarContrast; // Range: -1 - 1
  vec3 ParamZFarModulateRGB;
  float ParamZFarModulateR; // Range: 0 - 1
  float ParamZFarModulateG; // Range: 0 - 1
  float ParamZFarModulateB; // Range: 0 - 1
  vec3 ParamZFarBlendRGB;
  float ParamZFarBlendAlpha; // Range: 0 - 1
  CPlugShaderApply* ShaderAutoSaturateWithLight;
  CPlugBitmap* BitmapDepth;
  CPlugBitmap* BitmapSbch;
  vec2 BitmapTcScale;
  vec2 BitmapTcPeriod;
  vec2 BitmapRangeR;
  vec2 BitmapRangeG;
  vec2 BitmapRangeB;
  vec2 BitmapRangeA;
  CPlugShader* ShaderColorGrading;
};

// File extension: 'SceneFx.Gbx'
struct CSceneFxSuperSample : public CSceneFxCompo {
  CSceneFxSuperSample();

};

struct CSceneLocationCamera : public CSceneLocation {
  CSceneLocationCamera();

  float StereoScreenDist;
  vec2 FovRectMin;
  vec2 FovRectMax;
};

// File extension: 'SceneFx.Gbx'
struct CSceneFxFlares : public CSceneFxCompo {
  CSceneFxFlares();

  uint FlarePerBlock;
};

// File extension: 'SceneFxNod.Gbx'
struct CSceneFxNod : public CMwNod {
  CSceneFxNod();

  bool IsActive;
  CSceneFx* Fx;
  CSceneFxNod* const NodInput;
};

// File extension: 'SceneFx.Gbx'
struct CSceneFxBloom : public CSceneFxCompo {
  CSceneFxBloom();

  uint BlurSize;
  float Intensity; // Range: 0 - 1
  float DataBlend; // Range: 0 - 1
  bool DualData;
  const MwFastBuffer<CSceneFxBloomData*> Datas;
  UnnamedEnum RadialInput;
  float RadialIntens; // Range: 0 - 1
  uint m_RadialProjQuality;
  float RadialRadius2d;
  float RadialRadius3d;
  vec2 RadialOffset2d;
  vec3 RadialOffset3d;
};

struct CSceneFxBloomData : public CMwNod {
  CSceneFxBloomData();

  float HighInvExponent; // Range: 0 - 1
  float FakeHdrExponent; // Range: 0 - 1
  float FakeHdrMin; // Range: 0 - 1
};

// File extension: 'SceneConfig.Gbx'
struct CSceneConfig : public CMwNod {
  CSceneConfig();

};

struct CSceneConfigVision : public CMwNod {
  CSceneConfigVision();

};

// File extension: 'SceneFx.Gbx'
struct CSceneFxStereoscopy : public CSceneFxCompo {
  CSceneFxStereoscopy();

  UnnamedEnum Output;
  bool ExternalControl;
  const float EyeSeparation;
  float SeparationUserScale; // Range: 0 - 1
  const float SeparationGameScale; // Range: 0 - 1
  uint MarginPixelCount;
  UnnamedEnum SplitRatio;
  UnnamedEnum AnaglyphColor;
  float AnaglyphColorFactor; // Range: 0 - 1
  CPlugShaderApply* Shader_Interlaced;
  CPlugShaderApply* Shader_DebugBlend;
};

// File extension: 'SceneFx.Gbx'
struct CSceneFxHeadTrack : public CSceneFxCompo {
  CSceneFxHeadTrack();

};

// File extension: 'SceneFx.Gbx'
struct CSceneFx : public CMwNod {
  float Intensity; // Range: 0 - 1
  bool WantPreLoad;
};

// File extension: 'SceneFx.Gbx'
struct CSceneFxOverlay : public CSceneFx {
  CSceneFxOverlay();

  CPlugShader* Shader;
};

struct CSceneFxCompo : public CSceneFx {
};

struct CSceneFxDepthOfField : public CSceneFxCompo {
  CSceneFxDepthOfField();

};

// File extension: 'SceneFx.Gbx'
struct CSceneFxCameraBlend : public CSceneFxCompo {
  CSceneFxCameraBlend();

  bool CaptureEnable;
  float CaptureWeight; // Range: 0 - 1
};

// userName: 'SFxBlur'
// File extension: 'SceneFx.Gbx'
struct CSceneFxBlur : public CSceneFxCompo {
  CSceneFxBlur();

};

// File extension: 'SceneFx.Gbx'
struct CSceneFxDistor2d : public CSceneFxCompo {
  CSceneFxDistor2d();

  CPlugBitmap* BitmapDistor;
  float DistorScale;
};

// File extension: 'SceneFx.Gbx'
struct CSceneFxEdgeBlender : public CSceneFxCompo {
  CSceneFxEdgeBlender();

  float EdgeDotMax; // Range: 0 - 1
  float EdgeDepthMax;
  uint GutterPassCount;
  float GutterWeight; // Range: 0 - 1
  CPlugShaderApply* ShaderEdgeDetect;
  CPlugShaderApply* ShaderEdgeGutter;
};

// File extension: 'CarMarksModel.Gbx'
struct CSceneVehicleCarMarksModel : public CMwNod {
  CSceneVehicleCarMarksModel();

  bool Disabled;
  MwFastBuffer<CSceneVehicleCarMarksModelSub*> Models;
};

// File extension: 'CarMarksModelSub.Gbx'
struct CSceneVehicleCarMarksModelSub : public CMwNod {
  CSceneVehicleCarMarksModelSub();

  float SampleMinDist;
  float Width;
  float WidthCoefForceX;
  float WidthCoefForceZ;
  float WidthMax;
  float Alpha;
  float AlphaCoefForceX;
  float AlphaCoefForceZ;
  float AlphaMax;
  float GroundOffset;
  float TexVPerMeter;
  float FadeDist;
  bool CondGroundContact;
  bool CondSliding;
  UnnamedEnum CondMaterialId;
  float CondForceXGreaterThan;
  float CondForceZGreaterThan;
  CPlugMaterial* Material;
  bool DebugDisplay;
};

struct CSceneVehicleCarMarksSamples : public CMwNod {
  CSceneVehicleCarMarksSamples();

  string Name;
  const uint ByteSize;
  bool Disabled;
  MwFastBuffer<uint> Stops;
};

// File extension: 'SceneFx.Gbx'
struct CSceneFxCellEdge : public CSceneFxCompo {
  CSceneFxCellEdge();

  uint BlurTexelCount;
  float BlurCenterOverEdges;
  CPlugShaderApply* ShaderEdgeDetect;
};

struct CSceneFxMgr : public CMwNod {
};

struct CSceneTriggerAction : public CMwNod {
};

// userName: 'Bullet'
struct CSceneBulletVis : public CMwNod {
};

struct CSceneSolid2Vis : public CMwNod {
};

// userName: 'Wagon'
struct CSceneWagonPhy {
};

// userName: 'Wagon'
struct CSceneWagonVis {
};

// userName: 'Bullet'
struct CSceneBulletPhy : public CMwNod {
};

struct CSceneGunPhy {
};

struct CSceneGunVis {
};

// userName: 'TrafficPhyRailEnt'
struct SSceneTrafficPhyRailEnt : public CMwNod {
};

// userName: 'TrafficPhyDynaEnt'
struct SSceneTrafficPhyDynaEnt : public CMwNod {
};

// userName: 'TrafficPhy'
struct CSceneTrafficPhy : public CMwNod {
};

struct CSceneVehicleVisVFXExtraContext {
  bool IsVisible;
  uint AudioGroupHandle;
  vec3 LAmbient;
};

} // namespace Scene

namespace System {

struct CSystemEngine : public CMwEngine {
  CSystemEngine();

  void SynchUpdate();
};

struct CSystemMouse : public CNodSystem {
  CSystemMouse();

};

struct CSystemKeyboard : public CNodSystem {
  CSystemKeyboard();

};

struct CSystemWindow : public CNodSystem {
  CSystemWindow();

  const uint SizeX; // Range: 0 - 64000
  const uint SizeY; // Range: 0 - 64000
  const uint PosX; // Range: 0 - 64000
  const uint PosY; // Range: 0 - 64000
  const float Ratio;
  const bool HasSizeChanged;
  const bool HasPosChanged;
  string StatusString;
};

struct CSystemConfig : public CMwNod {
  CSystemConfig();

  NSystemConfig::EGamePackQuality GamePackQuality;
  bool IsSafeMode;
  CSystemConfigDisplay* Display;
  CSystemConfigDisplay* DisplaySafe;
  bool NetworkUseProxy;
  string NetworkProxyUrl;
  uint NetworkServerPort; // Range: 0 - 65535
  uint NetworkP2PServerPort; // Range: 0 - 65535
  uint NetworkClientPort;
  CSystemConfig::ENetworkSpeed NetworkSpeed;
  uint NetworkCustomDownload;
  uint NetworkCustomUpload;
  bool NetworkForceUseLocalAddress;
  string NetworkForceServerAddress;
  uint NetworkServerBroadcastLength;
  bool NetworkTestInternetConnection;
  bool NetworkUseNatUPnP;
  string NetworkLastUsedMSAddress;
  string NetworkLastUsedMSPath;
  bool FileTransferEnableDownload;
  bool FileTransferEnableUpload;
  bool EnableLocators;
  bool AutoUpdateFromLocator;
  bool FileTransferEnableAvatarDownload;
  bool FileTransferEnableAvatarUpload;
  bool FileTransferEnableAvatarLocators;
  bool FileTransferEnableMapDownload;
  bool FileTransferEnableMapUpload;
  bool FileTransferEnableMapLocators;
  bool FileTransferEnableMapModDownload;
  bool FileTransferEnableMapModUpload;
  bool FileTransferEnableMapModLocators;
  bool FileTransferEnableMapSkinDownload;
  bool FileTransferEnableMapSkinUpload;
  bool FileTransferEnableMapSkinLocators;
  bool FileTransferEnableTagDownload;
  bool FileTransferEnableTagUpload;
  bool FileTransferEnableTagLocators;
  bool FileTransferEnableVehicleSkinDownload;
  bool FileTransferEnableVehicleSkinUpload;
  bool FileTransferEnableVehicleSkinLocators;
  bool FileTransferEnableUnknownTypeDownload;
  bool FileTransferEnableUnknownTypeUpload;
  bool FileTransferEnableUnknownTypeLocators;
  bool IsIgnorePlayerSkins;
  bool IsSkipRollingDemo;
  bool GameProfileEnableMulti;
  string GameProfileName;
  uint PlayerInfoDisplaySize; // Range: 1 - 5
  CSystemConfig::EPlayerInfoDisplayType PlayerInfoDisplayType;
  bool DisableReplayRecording;
  CSystemConfig::ETmCarQuality TmCarQuality;
  CSystemConfig::ETmCarParticlesQuality TmCarParticlesQuality;
  CSystemConfig::EPlayerShadow PlayerShadow;
  CSystemConfig::EPlayerOcclusion PlayerOcclusion;
  CSystemConfig::ETmOpponents TmOpponents;
  uint TmMaxOpponents;
  CSystemConfig::ETmBackgroundQuality TmBackgroundQuality;
  uint SmMaxPlayerResimStepPerFrame;
  bool AudioEnabled;
  string AudioDevice_Oal;
  bool AudioMuteWhenAppUnfocused;
  float AudioSoundVolume;
  float AudioSoundLimit_Scene;
  float AudioSoundLimit_Ui;
  float AudioMusicVolume;
  CSystemConfig::EAudioQuality AudioGlobalQuality;
  bool AudioAllowEFX;
  bool AudioAllowHRTF;
  bool AudioDisableDoppler;
  bool InputsAlternateMethod;
  bool InputsCaptureKeyboard;
  bool InputsFreezeUnusedAxes;
  bool InputsEnableRumble;
  bool InputsEnableJoysticks;
  bool InputsDisableFreeCamPadControl;
  CSystemConfig::EAdvertising Advertising_Enabled;
  float Advertising_TunningCoef;
  bool Advertising_DisabledByUser;
  bool EnableCrashLogUpload;
  string BlackListUrl;
  string BadWordListUrl;
  string AntiCheatServerUrl;
  string DesiredLanguageId;
  bool SkipIntro;
  string VoiceChat_Device_In;
  string VoiceChat_Device_Out;
  float VoiceChat_SpeakerVolume;
  NSystemConfig::EVoiceDetectionMode VoiceChat_VoiceDetection_Mode;
  float VoiceChat_VoiceDetection_Sensitivity;
  bool VoiceChat_VoiceDetection_Auto;
  float VoiceChat_VoiceDetection;
};

struct CSystemMemoryMonitor : public CMwNod {
  CSystemMemoryMonitor();

};

struct CNodSystem : public CMwNod {
  CNodSystem();

};

struct CSystemFidsFolder : public CMwNod {
  const MwFastBuffer<CSystemFidsFolder*> Trees;
  const MwFastBuffer<CSystemFidFile*> Leaves;
  const uint ByteSize;
  const uint ByteSizeEd;
  const wstring DirName;
  const wstring FullDirName;
  CSystemFidsFolder* const ParentFolder;
};

struct CSystemFidFile : public CMwNod {
  CMwNod* const Nod;
  bool Text;
  bool Compressed;
  const uint ByteSize;
  const uint ByteSizeEd;
  const string TimeWrite;
  bool IsReadOnly;
  const wstring FileName;
  const wstring FullFileName;
  const wstring ShortFileName;
  CSystemFidsFolder* const ParentFolder;
  CSystemFidFile* const Container;
  void CopyToFileRelative(string RelFileName, bool FailIfExists);
  bool OSCheckIfExists();
};

struct CSystemFidsDrive : public CSystemFidsFolder {
  CSystemFidsDrive();

};

// Description: "API for platform calls."
struct CSystemPlatformScript : public CMwNod {
  enum class CSystemPlatformScript::ESystemPlatform {
    None = 0,
    Steam = 1,
    UPlay = 2,
    PS4 = 3,
    XBoxOne = 4,
    PS5 = 5,
    XBoxSeries = 6,
    Stadia = 7,
    Luna = 8,
  };
  enum class CSystemPlatformScript::ESystemSkuIdentifier {
    Unknown = 0,
    EU = 1,
    US = 2,
    JP = 3,
    CN = 4,
  };
  const CSystemPlatformScript::ESystemPlatform Platform; // Maniascript
  const CSystemPlatformScript::ESystemSkuIdentifier SkuIdentifier; // Maniascript
  const string ExeVersion; // Maniascript
  const uint CurrentLocalDate; // Maniascript
  const string CurrentLocalDateText; // Maniascript
  const wstring CurrentTimezone; // Maniascript
  const string CurrentTimezoneTimeOffset; // Maniascript
  const string ExtraTool_Info; // Maniascript
  wstring ExtraTool_Data; // Maniascript
  void ClipboardSet(wstring ClipboardText); // Maniascript
  void FlashWindow(); // Maniascript
  const bool IsWritingUserSave; // Maniascript
};

// File extension: 'RefBuffer.Gbx'
struct CSystemDependenciesList : public CMwNod {
  CSystemDependenciesList();

  MwFastBuffer<CMwNod*> Files;
  MwFastBuffer<CSystemFidsFolder*> Folders;
  const MwFastBuffer<uint> Folders_ClassIds;
  CMwRefBuffer* FillFromRefBuffer;
  void CleanRedondancies();
};

struct CSystemConfigDisplay : public CMwNod {
  CSystemConfigDisplay();

  bool AllowVRR;
  CSystemConfigDisplay::EShowPerformance ShowPerformance;
  CSystemConfigDisplay::ERenderingApi RenderingApi;
  wstring Adapter;
  CSystemConfigDisplay::EDisplayMode DisplayMode;
  nat2 ScreenSizeFS;
  CSystemConfigDisplay::EWindowSize ScreenSizeWin;
  uint RefreshRate;
  CSystemConfigDisplay::EDisplaySync DisplaySync;
  bool StereoByDefault;
  bool StereoAdvanced;
  CSystemConfigDisplay::ETripleBuffer TripleBuffer;
  bool Customize;
  CSystemConfigDisplay::EPreset Preset;
  CSystemConfigDisplay::EAntialias Antialiasing;
  CSystemConfigDisplay::EDeferredAA DeferredAA;
  CSystemConfigDisplay::EShaderQ ShaderQuality;
  ETextureQuality TexturesQuality;
  CSystemConfigDisplay::EFilterAnisoQ FilterAnisoQ;
  CSystemConfigDisplay::EZClip ZClip;
  CSystemConfigDisplay::EZClipAuto ZClipAuto;
  uint ZClipNbBlock;
  CSystemConfigDisplay::EGeometryQ GeometryQuality;
  float GeomLodScaleZ; // Range: 0.5 - 8
  CSystemConfigDisplay::EEverywhereReflect ReflectEverywhere;
  CSystemConfigDisplay::EWaterReflect WaterReflect;
  bool WaterGeomStadium;
  CSystemConfigDisplay::EVehicleReflect VehicleReflect;
  uint VehicleReflectMaxCount;
  bool Decals_3D__TextureDecals_;
  bool Decals_2D__TextureDecals_;
  CSystemConfigDisplay::EFxBloomHdr FxBloomHdr;
  CSystemConfigDisplay::EFxMotionBlur FxMotionBlur;
  float FxMotionBlurIntens; // Range: 0 - 1
  CSystemConfigDisplay::EFxBlur FxBlur;
  CSystemConfigDisplay::ELightMapSizeMax LM_SizeMax;
  CSystemConfigDisplay::ELightMapQuality LM_Quality;
  bool LM_QUltra;
  bool LM_iLight;
  CSystemConfigDisplay::EScreenShotExt ScreenShotExt;
  CSystemConfigDisplay::EShadows Shadows;
  CSystemConfigDisplay::EGpuSync GpuSync0;
  CSystemConfigDisplay::EGpuSync GpuSync1;
  CSystemConfigDisplay::EGpuSync GpuSync2;
  CSystemConfigDisplay::EGpuSync GpuSync3;
  uint GpuSyncTimeOut;
  uint MaxFps;
  bool EmulateCursorGDI;
  bool OptimPartDynaGeom;
  bool DisableZBufferRange;
  bool DisableWindowedAntiAlias;
  bool DisableHdrCubeRenderMipMap;
  bool EnableFullscreenGDI;
  CSystemConfigDisplay::ELightFromMap LightFromMap;
  bool EnableCheckLags;
  float AgpUseFactor; // Range: 0 - 1
  float ParticleMaxGpuLoadMs;
  bool AsyncRender;
  bool MultiThread;
  uint ThreadCountMax;
  bool Automatic_Enabled;
  uint Automatic_MinFps;
};

struct CSystemPackManager : public CMwNod {
};

struct CSystemPackDesc : public CMwNod {
  const wstring Name;
  const string Checksum;
  const wstring FileName;
  CSystemFidFile* const Fid;
  const wstring LocatorFileName;
  const string Url;
  const bool AutoUpdate;
};

struct CSystemNodWrapper : public CMwNod {
  CSystemNodWrapper();

};

struct CSystemData : public CMwNod {
  string Url;
  CSystemPackDesc* const PackDesc;
  CMwNod* Data;
};

struct CSystemFidContainer : public CMwNod {
};

} // namespace System

namespace Vision {

struct CVisionViewport : public CHmsViewport {
  bool EmulateCursor;
  const MwFastBuffer<wstring> Display_Win32DeviceNames;
  void ReGen_TextureGenUVs();
  void GetPerformance();
  bool Alpha01BlendEdges;
  uint Alpha01ClipRef_NoAtoC;
  const float TimeGpuMs_Total;
  const float TimeGpuMs_Particles;
  const uint cPerf_BufferRefract;
  const float Fill01_BufferRefract; // Range: 0 - 1
  bool AsyncRender;
  void FreeTextureMem();
  void RestoreTextureMem();
  CMwNod* VisionResourceFile;
  CPlugBitmap* FogCameraTransmittances;
  bool ShotHqLighting_Enable;
  CMwNod* ShotHqLightingParam;
  bool ShadowCache_Enable;
  SVisShadowCacheMgr* const ShadowCacheMgr;
  bool EnableVolumetricFog;
  float FogTexMaxDepth; // Range: 5 - 200
  float FogSliceDistribCoef; // Range: 0.001 - 8
  float ScatteringCoef; // Range: 0 - 5
  float AbsorptionCoef; // Range: 0 - 5
  float FogAmbientTweak; // Range: 0 - 1
  float ExtinctionTweak; // Range: 0 - 1
  float PhaseAnisotropy; // Range: 0 - 1
  float HeightDensity0; // Range: 0 - 5
  float HeightScalingCoef; // Range: 0 - 5
  vec3 Albedo;
  float NoiseMaxDensity; // Range: 0 - 2
  float NoiseScalingCoef; // Range: 0 - 5
  UnnamedEnum ColorBlindnessCorrection;
  float ColorBlindnessCorrection_Intensity; // Range: 0 - 1
};

struct CVisionViewportNull : public CVisionViewport {
};

// File extension: 'VisionResourceFile.Gbx'
struct CVisionResourceFile : public CMwNod {
  CVisionResourceFile();

  CVisionResourceShaders* Shaders;
  CPlugBitmap* Bitmap_DeferredDiffuseAmbient;
  float BlurDepthTestMaxDist;
  CPlugShaderApply* Shader3DVolumetricFog;
  CPlugShaderApply* ShaderT3Sea;
  CPlugShaderApply* ShaderShadowFakeQuad;
  CPlugShaderApply* ShaderSoapBubble;
  CPlugShaderApply* ShaderDecal2d_CV_DefDecal;
  CPlugShaderApply* ShaderRasterCircularGauge;
  CPlugFont* FontDebugText;
  CPlugShaderApply* FillVtxColor_sRGB;
  CPlugShaderApply* BlackNoSpec;
  CPlugShaderApply* ShaderEnergyPrismAA;
  CPlugShaderApply* ShaderDownSize2x2AvgInLdr;
  CPlugShaderApply* ShaderDownSize3x3AvgInLdr;
  CPlugShaderApply* ShaderColorDepthAlphaDown2x2_Min;
  CPlugShaderApply* ShaderGeomToReflectCubeMap;
  CPlugShaderApply* ShaderGeomToReflectCubeDist;
  CPlugShaderApply* ShaderGeomToFaceNormal;
  CPlugShaderApply* ShaderGeomToFaceNormal_Alpha01;
  CPlugShaderApply* ShaderGeomFakeShadows;
  CPlugShaderApply* ShaderBoxesRasterBitmapBlendPreMod;
  CPlugShaderApply* ShaderT3DynaSpriteDiffuse;
  CPlugShaderApply* ShaderRasterDistor2d;
  CPlugShaderApply* ShaderLightFromMap;
  CPlugShaderApply* ShaderSmVortexParticle;
  CPlugBitmap* BitmapHyperZ;
  CPlugBitmap* BitmapDeferredZ;
  CPlugBitmap* BitmapDeferredPixelNormalInC;
  CPlugBitmap* BitmapDeferredVertexNormalInC;
  CPlugBitmap* BitmapDeferredFaceNormalInC;
  CPlugBitmap* BitmapDeferredMDiffuse;
  CPlugBitmap* BitmapDeferredMSpecular;
  CPlugBitmap* BitmapDeferredLightMask;
  CPlugBitmap* BitmapDeferredPreShade;
  CPlugBitmap* BitmapWaterNormalDecals;
  CPlugBitmap* BitmapWaterRefract;
  CPlugBitmap* BitmapWaterReflect;
  CPlugBitmap* BitmapWaterPlaneId;
  CPlugBitmap* BitmapWaterCaustics;
  CPlugBitmap* BitmapPostFxCopyHdrA;
  CPlugBitmap* BitmapSwapR8;
  CPlugBitmap* BitmapSwapR16;
  CPlugBitmap* BitmapShadowLDir0;
  CPlugBitmap* BitmapShadowLDir0_Texture1;
  CPlugBitmap* BitmapShadowStaticSprite;
  CPlugBitmap* BitmapShadowStaticSprite_T1;
  CPlugBitmap* BitmapShadowMaskDepth;
  CPlugBitmap* BitmapShadowPssmDepth;
  CPlugBitmap* BitmapMotion;
  CPlugBitmap* BitmapHeightField;
  CPlugBitmap* BitmapCloudsOcc;
  CPlugBitmap* BitmapProbeGrid;
  CPlugBitmap* BitmapProbeGridSpread1;
  CPlugBitmap* BitmapInvFresnelPC3;
  CPlugBitmap* BitmapSeaSxSySzPC3;
  CPlugBitmap* BitmapHyperZ_Peel0;
  CPlugBitmap* BitmapToneMap_AutoScale;
  CPlugBitmap* BitmapPreLightGenTx;
  CPlugBitmap* BitmapPreLightGenTy;
  CPlugBitmap* BitmapPreLightGenTz;
  CPlugBitmap* BitmapLmDiffuse;
  CPlugBitmap* BitmapDefaultFresnelFaceBump;
  CPlugBitmap* BitmapReflectCubeHdr;
  CPlugBitmap* BitmapReflectCubeHdrLQ;
  CPlugBitmap* BitmapImageSpaceReflectCube;
  CPlugBitmap* BitmapCopyRtAfterCameras;
  CPlugBitmap* BitmapTranslucentDown;
  CPlugBitmap* BitmapOverlayBlur;
  CPlugBitmap* BitmapDynaHueIntensGauge;
  CPlugBitmap* BitmapDynaTcSxyTzw;
  CPlugBitmap* BitmapDynaTcSxyTzw_RenderOverlay;
  CPlugBitmap* BitmapPerlinPerm2D;
  CPlugBitmap* BitmapPerlinGradOfPerm1D;
  CPlugBitmap* BitmapPerlinGrad3OfPerm2D;
  CPlugBitmap* BitmapPerlinGrad4OfPerm2D;
  CPlugBitmap* BitmapNoise3d_Fog0_32;
  CPlugBitmap* BitmapNoise3d_Fog0_64;
  CPlugFxLensFlareArray* FxLensFlareArray;
  CPlugShaderApply* Shader2dLightHistoryUpdate;
  CPlugBitmap* BitmapLmLListIndexLow;
  CPlugBitmap* BitmapLmLListIndexHigh;
  CPlugBitmap* BitmapTileList;
  CPlugBitmap* BitmapTileMask;
  CPlugBitmap* BitmapTileGpu;
  CPlugBitmap* BitmapIceBubbles;
  CPlugBitmap* BitmapIceColorMaskCrackRoughness;
  CPlugBitmap* BitmapIceSubsurfaceHeightSpeckle;
  CPlugBitmap* BitmapIce_D;
  CPlugBitmap* BitmapCube_N;
  CPlugBitmap* BitmapIce_Roughness;
  CPlugBitmap* BitmapCube_D;
  CPlugBitmap* BitmapSpecIE_to_Roughness;
  CPlugBitmap* EnergyHueIntens_Additif;
  CPlugBitmap* EnergyHueIntens_Blend;
  CPlugBitmap* RandVec3_2d;
};

struct CDx11Viewport : public CVisionViewport {
  const string FeatureLevel;
  bool ZBuffer_Near1_Far0;
};

struct CVisionResourceShaders : public CMwNod {
  CPlugShader* DeferredFogLighting_Inter0;
  CPlugShader* DeferredFogLighting_Inter1;
  CPlugShader* DeferredWaterFog0;
  CPlugShader* DeferredWaterFog1;
  CPlugShader* DeferredWaterFog2;
  CPlugShader* DeferredWaterFog3;
};

struct SVisShadowCacheMgr {
  nat2 c2TexelShadow;
  uint Stat_cShadowUsed;
  uint Stat_cShadowFace;
  uint Stat_cSharedFace;
  uint Stat_cClippedFace;
  uint Cfg_cShadowMax;
  float Cfg_cPixelInTexelMax;
  float Cfg_cTexelByMeterMax;
  float Cfg_cPixelBorder;
  bool Cfg_DoRotateSpots;
  bool Cfg_DoScissorSpotOuter;
  bool Cfg_DoMergeSplittedY;
  bool Cfg_DoFaceCameraClip;
  bool Cfg_IsDynaOnly;
  bool Cfg_SwitchInvertY;
  bool Cfg_BasicAlloc;
  uint cLightIn;
  CPlugBitmap* MapShadowCache;
};

// File extension: 'SceneFx.Gbx'
struct CVisPostFx_ToneMapping : public CSceneFxCompo {
  CVisPostFx_ToneMapping();

};

// File extension: 'SceneFx.Gbx'
struct CVisPostFx_MotionBlur : public CSceneFxCompo {
  CVisPostFx_MotionBlur();

};

// File extension: 'SceneFx.Gbx'
struct CVisPostFx_BloomHdr : public CSceneFxCompo {
  CVisPostFx_BloomHdr();

  bool BloomIntensUseCurve;
  float MinIntensInBloomSrc;
  uint DownCount;
  const bool PreDown4x4_Else2x2;
  uint BlurSizeDown1_2x2;
  const uint BlurSizeDown1;
  const uint BlurSizeDown2;
  const uint BlurSizeDown3;
  const uint BlurSizeDown4;
  const uint BlurSizeDown5;
  float BlurSizeScaleByLevel;
  float BlurCenterOverEdges;
  float UpScale;
  float UpScaleAlpha;
  UnnamedEnum BlurTexAddress;
  bool UseFakedFlares;
  float StreaksSrcMin; // Range: 0 - 1
  uint StreaksRotationCount;
  float StreaksRotation; // Range: 0 - 360
  uint StreaksPassCount;
  CPlugShaderApply* FidShaderBloomSelectFilterDown2;
  CPlugShaderApply* FidShaderBloomSelectFilterDown4;
  CPlugShaderApply* ShaderBloomFinal;
  CPlugShaderApply* ShaderDistorImageSpace;
  CPlugShaderApply* ShaderDistorGetDepthIntens;
  CPlugShaderApply* ShaderDistorAtDepth;
  CPlugShaderApply* ShaderHorizonBlur;
  bool DistorEnable;
  uint DistorPeriodTime;
  float DistorRedSelectRatio; // Range: 0 - 2
  bool DistorUseDepth;
};

} // namespace Vision

namespace Audio {

struct CAudioPort : public CMwNod {
  bool IsEnabled;
  float SoundVolume; // Range: -40 - 0
  float MusicVolume; // Range: -40 - 0
  bool MuteWhenUnfocused;
  UnnamedEnum SettingQuality;
  float SettingSoundHdrFactor; // Range: 0 - 1
  bool SettingDisableDoppler;
  uint SettingUpdatePeriod; // Range: 0 - 1000
  uint SettingSoundsPerUpdate;
  uint SettingMaxSimultaneousSounds;
  const MwFastBuffer<CAudioZone*> Zones;
  const MwFastBuffer<CAudioListener*> Listeners;
  const MwFastBuffer<CAudioSource*> Sources;
  const MwFastBuffer<CAudioBufferKeeper*> BufferKeepers;
  const uint Manager_AllocatedVoices;
  const uint Manager_AvailableVoices;
  CPlugAudioBalance* const BaseBalanceLoud;
  CPlugAudioBalance* const BaseBalanceSoft;
  CPlugAudioBalance* const CurBalance;
  const float CurScriptLimitMusicVolumedB;
  const float CurScriptLimitSceneSoundVolumedB;
  const float CurScriptLimitUiSoundVolumedB;
  float CurSoundVolumeRms;
  float CurSoundHdrCorrection;
  int ForceEnableMusicCount;
  bool MuteSounds_FromLoadScreen;
  uint IsCapturing;
  const MwFastBuffer<CPlugFileSnd*> CapturedFileSnds;
  CPlugFileText* ModifierXmlFile;
  void ParseModifierXml();
  void LoadExternalSoundParam();
  void SaveExternalSoundParam();
  const string AnalyzerEstimatedVolume;
  const float AnalyzerAudioTimePerSec;
  const float AnalyzerAudioTimePerSecUpdate;
  const float AnalyzerAudioTimePerSecStream;
  const uint AnalyzerStreamBytesPerSec;
  const uint AnalyzerAudioLongestSlice;
  const uint AnalyzerTotalKeepersMem;
  const string AnalyzerNbSoundsStr;
  const uint AnalyzerNbFadingSounds;
  const uint AnalyzerNbAutoDuckedSounds;
  const uint AnalyzerNbStreamingSounds;
  const uint AnalyzerNbDestoyedPerUpdate;
  const uint AnalyzerNbCreatedPerUpdate;
};

struct CAudioPortNull : public CAudioPort {
};

struct CAudioSoundImplem : public CMwNod {
  CAudioSource* const Source;
  const bool IsActuallyPlaying;
  void RefreshStaticProperties();
  float PlayCursor;
  float PlayCursorUi; // Range: 0 - 1
};

struct CAudioBufferKeeper : public CMwNod {
  CPlugFileSnd* const PlugFileSnd;
  const uint HwAllocSize;
  const bool IsStreaming;
  const bool IsSilent;
  const uint NbUses;
};

struct CAudioListener : public CMwNod {
  CPlugAudioEnvironment* AudioEnvironment;
  float InsideCoef;
  float EnclosedCoef;
  float UnderwaterCoef;
  MwFastBuffer<uint> AudioGroupHandlesInFocus;
  iso4 Location;
  vec3 Velocity;
  bool ApplyHmdCorrection;
};

struct CAudioZone : public CMwNod {
  const MwFastBuffer<CAudioZoneSource*> Sources;
  const MwFastBuffer<CAudioListener*> Listeners;
};

struct CAudioScriptManager : public CMwNod {
  enum class CAudioScriptManager::ELibSound {
    Alert = 0,
    ShowDialog = 1,
    HideDialog = 2,
    ShowMenu = 3,
    HideMenu = 4,
    Focus = 5,
    Valid = 6,
    Start = 7,
    Countdown = 8,
    Victory = 9,
    ScoreIncrease = 10,
    Checkpoint = 11,
  };
  CAudioScriptSound* CreateSound(string Url); // Maniascript
  CAudioScriptSound* CreateSoundEx(string Url, float VolumedB, bool IsMusic, bool IsLooping, bool IsSpatialized); // Maniascript
  void DestroySound(CAudioScriptSound* Sound); // Maniascript
  const MwFastBuffer<CAudioScriptSound*> Sounds; // Maniascript
  CAudioScriptMusic* CreateMusic(string Url); // Maniascript
  void DestroyMusic(CAudioScriptMusic* Music); // Maniascript
  void PlaySoundEvent(CAudioScriptSound* Sound, float VolumedB); // Maniascript
  void PlaySoundEventUrl(string Url, float VolumedB); // Maniascript
  void PlaySoundLibrary(CAudioScriptManager::ELibSound Sound, uint SoundVariant, float VolumedB); // Maniascript
  void PlayDelayedSoundEvent(CAudioScriptSound* Sound, float VolumedB, int Delay); // Maniascript
  void PlayDelayedSoundEventUrl(string Url, float VolumedB, int Delay); // Maniascript
  void PlayDelayedSoundLibrary(CAudioScriptManager::ELibSound Sound, uint SoundVariant, float VolumedB, int Delay); // Maniascript
  void PlaySoundEventMix(CAudioScriptSound* Sound, float VolumedB, vec3 PanRadiusLfe); // Maniascript
  void PlaySoundEventMixUrl(string Url, float VolumedB, vec3 PanRadiusLfe); // Maniascript
  void PlayDelayedSoundEventMix(CAudioScriptSound* Sound, float VolumedB, vec3 PanRadiusLfe, int Delay); // Maniascript
  void PlayDelayedSoundEventMixUrl(string Url, float VolumedB, vec3 PanRadiusLfe, int Delay); // Maniascript
  void ClearAllDelayedSoundsEvents(); // Maniascript
  bool ForceEnableMusic; // Maniascript
  float LimitMusicVolumedB; // Maniascript
  float LimitSceneSoundVolumedB; // Maniascript
  float LimitUiSoundVolumedB; // Maniascript
};

struct CAudioScriptSound : public CMwNod {
  void Play(); // Maniascript
  void Stop(); // Maniascript
  const bool IsPlaying; // Maniascript
  const bool DownloadInProgress; // Maniascript
  float Volume; // Maniascript
  float FadeDuration; // Maniascript
  float VolumedB; // Maniascript
  float Pitch; // Maniascript
  vec3 RelativePosition; // Maniascript
  vec3 PanRadiusLfe; // Maniascript
  float PlayCursor; // Maniascript
  const float PlayLength; // Maniascript
};

struct CAudioScriptMusic : public CAudioScriptSound {
  enum class CAudioScriptMusic::EUpdateMode {
    Cut = 0,
    OnNextBar = 1,
    OnNextHalfBar = 2,
    OnNextBeat = 3,
  };
  MwFastArray<float> Tracks_Volume; // Maniascript
  MwFastArray<float> Tracks_VolumedB; // Maniascript
  const MwFastArray<wstring> Tracks_Name; // Maniascript
  const MwFastArray<float> Tracks_Length; // Maniascript
  const uint Tracks_Count; // Maniascript
  const float BeatsPerMinute; // Maniascript
  const float BeatDuration; // Maniascript
  const uint BeatsPerBar; // Maniascript
  CAudioScriptMusic::EUpdateMode UpdateMode; // Maniascript
  bool Dbg_ForceIntensity; // Maniascript
  bool Dbg_ForceSequential; // Maniascript
  bool Dbg_ForceRandom; // Maniascript
  float LPF_CutoffRatio; // Range: 0 - 1 // Maniascript
  float LPF_Q; // Range: 0.1 - 20 // Maniascript
  float HPF_CutoffRatio; // Range: 0 - 1 // Maniascript
  float HPF_Q; // Range: 0.1 - 20 // Maniascript
  float FadeTracksDuration; // Maniascript
  float FadeFiltersDuration; // Maniascript
  void MuteAllTracks(); // Maniascript
  void UnmuteAllTracks(); // Maniascript
  void NextVariant(); // Maniascript
  void NextVariant2(bool IsIntensityDecreasing); // Maniascript
  void EnableSegment(wstring SegmentName); // Maniascript
  bool UseNewImplem; // Maniascript
};

struct CAudioZoneSource : public CMwNod {
  enum class CAudioZoneSource::EAudioBalanceGroup {
    Auto = 0,
    Music = 1,
    Menus = 2,
    Ambiance = 3,
    Player = 4,
    Bengs = 5,
    Guns = 6,
    BackingDirect = 7,
    Trails = 8,
    GameUI = 9,
    Custom1 = 10,
    Custom2 = 11,
    OtherPlayers = 12,
    ImpactWarning = 13,
    Environment = 14,
  };
  CPlugSound* PlugSound;
  float PriorityAdjustement;
  bool UseLowQuality;
  vec3 VolumicSize;
  CAudioZoneSource::EAudioBalanceGroup BalanceGroup;
  uint AudioGroupHandle;
  bool Play;
  bool Stop;
  float Volume; // Range: 0 - 2
  float Pitch; // Range: 0 - 5
  uint Variant;
  uint MaxVariants;
  float RpmOrSpeed;
  float Accel; // Range: 0 - 1
  uint Gear;
  uint SurfaceId;
  float SkidIntensity;
  float SkidSpeedKmh;
  CMwNod* const AudioSourceMain;
  CMwNod* const AudioSourceBacking;
};

struct CAudioSource : public CMwNod {
  enum class CAudioSource::EAudioBalanceGroup {
    Auto = 0,
    Music = 1,
    Menus = 2,
    Ambiance = 3,
    Player = 4,
    Bengs = 5,
    Guns = 6,
    BackingDirect = 7,
    Trails = 8,
    GameUI = 9,
    Custom1 = 10,
    Custom2 = 11,
    OtherPlayers = 12,
    ImpactWarning = 13,
    Environment = 14,
  };
  const bool IsPlaying;
  void Play();
  void Stop();
  CPlugSound* const PlugSound;
  bool UseLowQuality;
  float PriorityAdjustement;
  CAudioSource::EAudioBalanceGroup BalanceGroup;
  uint AudioGroupHandle;
  float PlayCursor;
  float PlayCursorUi; // Range: 0 - 1
  float VolumedB; // Range: -60 - 6
  float LfeSenddB; // Range: -60 - 6
  float Pitch; // Range: 0.01 - 5
  float PanAngleDeg; // Range: -180 - 180
  float Radius;
  vec3 Position;
  iso4 Loc;
  vec3 Velocity;
  CAudioSoundImplem* const Implementation;
};

struct CAudioSourceMusic : public CAudioSource {
  MwFastArray<float> TracksVolume;
  MwFastArray<float> TracksVolumedB;
  float LPF_CutoffRatio; // Range: 0 - 1
  float LPF_Q; // Range: 0.1 - 20
  float HPF_CutoffRatio; // Range: 0 - 1
  float HPF_Q; // Range: 0.1 - 20
  float FadeTrackDuration;
  float FadeFiltersDuration;
  void MuteAllTracks();
  void UnmuteAllTracks();
};

struct CAudioSourceEngine : public CAudioSource {
  float RpmNormalised; // Range: 0 - 1
  float Rpm;
  float Accel; // Range: 0 - 1
  float VehicleSpeed;
  float Alpha; // Range: -1 - 1
  uint Gear;
};

struct CAudioSourceSurface : public CAudioSource {
  enum class CAudioSourceSurface::ESurfId {
    Concrete = 0,
    Pavement = 1,
    Grass = 2,
    Ice = 3,
    Metal = 4,
    Sand = 5,
    Dirt = 6,
    Turbo_Deprecated = 7,
    DirtRoad = 8,
    Rubber = 9,
    SlidingRubber = 10,
    Test = 11,
    Rock = 12,
    Water = 13,
    Wood = 14,
    Danger = 15,
    Asphalt = 16,
    WetDirtRoad = 17,
    WetAsphalt = 18,
    WetPavement = 19,
    WetGrass = 20,
    Snow = 21,
    ResonantMetal = 22,
    GolfBall = 23,
    GolfWall = 24,
    GolfGround = 25,
    Turbo2_Deprecated = 26,
    Bumper_Deprecated = 27,
    NotCollidable = 28,
    FreeWheeling_Deprecated = 29,
    TurboRoulette_Deprecated = 30,
    WallJump = 31,
    MetalTrans = 32,
    Stone = 33,
    Player = 34,
    Trunk = 35,
    TechLaser = 36,
    SlidingWood = 37,
    PlayerOnly = 38,
    Tech = 39,
    TechArmor = 40,
    TechSafe = 41,
    OffZone = 42,
    Bullet = 43,
    TechHook = 44,
    TechGround = 45,
    TechWall = 46,
    TechArrow = 47,
    TechHook2 = 48,
    Forest = 49,
    Wheat = 50,
    TechTarget = 51,
    PavementStair = 52,
    TechTeleport = 53,
    Energy = 54,
    TechMagnetic = 55,
    TurboTechMagnetic_Deprecated = 56,
    Turbo2TechMagnetic_Deprecated = 57,
    TurboWood_Deprecated = 58,
    Turbo2Wood_Deprecated = 59,
    FreeWheelingTechMagnetic_Deprecated = 60,
    FreeWheelingWood_Deprecated = 61,
    TechSuperMagnetic = 62,
    TechNucleus = 63,
    TechMagneticAccel = 64,
    MetalFence = 65,
    TechGravityChange = 66,
    TechGravityReset = 67,
    RubberBand = 68,
    Gravel = 69,
    Hack_NoGrip_Deprecated = 70,
    Bumper2_Deprecated = 71,
    NoSteering_Deprecated = 72,
    NoBrakes_Deprecated = 73,
    RoadIce = 74,
    RoadSynthetic = 75,
    Green = 76,
    Plastic = 77,
    DevDebug = 78,
    Free3 = 79,
  };
  CAudioSourceSurface::ESurfId SurfaceId;
  float SpeedNormalised; // Range: 0 - 1
  float SpeedKmh;
  float SkidIntensity; // Range: 0 - 1
  float SkidSpeedKmh;
};

struct CAudioSourceMulti : public CAudioSource {
  uint Variant;
  uint MaxVariants;
};

struct CAudioSourceMood : public CAudioSource {
  float InsideFactor; // Range: 0 - 1
};

struct CAudioSourceGauge : public CAudioSource {
  float Ratio; // Range: 0 - 1
  float FullRatio;
  float GradingRatio;
  float Rate;
};

struct COalAudioPort : public CAudioPort {
  const wstring OpenALDllPath;
  string DeviceName;
  bool SettingAllowEFX;
  bool SettingAllowHRTF;
  bool SettingForceStereo;
  const uint NbMaxSounds;
  void EnumerateDevicesNames_WriteToLog();
  void EnumerateDevices();
  const bool EFXEnabled;
  const uint Manager_SpacialVoices;
  const uint Manager_DirectVoices;
  CPlugAudioEnvironment* const CurrentEnvironment;
};

struct COalAudioBufferKeeper : public CAudioBufferKeeper {
};

} // namespace Audio

namespace Script {

struct CScriptSetting : public CMwNod {
  const UnnamedEnum Type;
  const string Name;
  bool BooleanVal;
  int IntegerVal;
  float RealVal;
  wstring TextVal;
  vec2 Vec2Val;
  vec3 Vec3Val;
  int2 Int2Val;
  int3 Int3Val;
};

struct CScriptTraitsPersistent : public CMwNod {
  CScriptTraitsPersistent();

  const uint PersistentTraitsCount;
};

struct CScriptTraitsMetadata : public CMwNod {
  CScriptTraitsMetadata();

  const uint MetadataTraitsCount;
  void ClearMetadata();
};

struct CScriptInterfacableValue : public CMwNod {
  const UnnamedEnum Type;
  bool BooleanVal;
  int IntegerVal;
  float RealVal;
  wstring TextVal;
  vec2 Vec2Val;
  vec3 Vec3Val;
  int2 Int2Val;
  int3 Int3Val;
  bool Bool1;
  string String1;
  wstring StringInt1;
};

struct CScriptBaseEvent : public CScriptBaseConstEvent {
  const bool HasBeenPassed; // Maniascript
  const bool HasBeenDiscarded; // Maniascript
};

// Description: "An event"
struct CScriptBaseConstEvent : public CMwNod {
  bool HasBeenProcessed; // Maniascript
};

struct CScriptPoison : public CMwNod {
};

} // namespace Script

namespace Net {

struct CNetNod : public CMwNod {
};

struct CNetServerInfo : public CMwNod {
  const string GameID;
  const string GameVersion;
  const string SessionTitleId;
  const string HostName;
  const string LocalIP;
  const uint LocalUDPPort;
  const uint LocalTCPPort;
  const string RemoteIP;
  const uint RemoteUDPPort;
  const uint RemoteTCPPort;
};

struct CNetClientInfo : public CMwNod {
  const string GameID;
  const string GameVersion;
  const string HostName;
  const string LocalIP;
  const string RemoteIP;
};

struct CNetFormTimed : public CNetNod {
};

struct CNetFormQuerrySessions : public CNetNod {
  CNetFormQuerrySessions();

};

struct CNetFormEnumSessions : public CNetNod {
  CNetFormEnumSessions();

};

struct CNetFormPing : public CNetFormTimed {
  CNetFormPing();

};

struct CNetServer : public CMwNod {
  const uint P2PPort;
  const bool AcceptConnections;
  const uint NbrNewConnections;
  const uint NbrConnections;
  const uint NbrConnectionsDone;
  const uint NbrConnectionsPending;
  const uint NbrConnectionsDisconnecting;
  const uint SendingDataRate;
  const uint TCPSendingDataRate;
  const uint UDPSendingDataRate;
  const uint ReceivingDataRate;
  const uint TCPReceivingDataRate;
  const uint UDPReceivingDataRate;
  const uint ReceptionPacketTotal;
  const uint UDPReceptionPacketTotal;
  const uint TCPReceptionPacketTotal;
  const uint SendingPacketTotal;
  const uint UDPSendingPacketTotal;
  const uint TCPSendingPacketTotal;
  const uint ReceptionNodTotal;
  const uint UDPReceptionNodTotal;
  const uint TCPReceptionNodTotal;
  const uint SendingNodTotal;
  const uint UDPSendingNodTotal;
  const uint TCPSendingNodTotal;
};

struct CNetClient : public CMwNod {
  const MwFastBuffer<CNetConnection*> Connections;
  const uint NbrNewConnections;
  const uint NbrConnections;
  const uint NbrConnectionsInProgress;
  const uint NbrConnectionsDone;
  const uint NbrConnectionsDisconnecting;
  const uint SendingDataRate;
  const uint TCPSendingDataRate;
  const uint UDPSendingDataRate;
  const uint ReceivingDataRate;
  const uint TCPReceivingDataRate;
  const uint UDPReceivingDataRate;
  const uint ReceptionPacketTotal;
  const uint UDPReceptionPacketTotal;
  const uint TCPReceptionPacketTotal;
  const uint SendingPacketTotal;
  const uint UDPSendingPacketTotal;
  const uint TCPSendingPacketTotal;
  const uint ReceptionNodTotal;
  const uint UDPReceptionNodTotal;
  const uint TCPReceptionNodTotal;
  const uint SendingNodTotal;
  const uint UDPSendingNodTotal;
  const uint TCPSendingNodTotal;
  const int TimeLatestHumanPing;
  const int TimeLatestGamePing;
  const float TimeLatestEpsilon;
  const float TimeSmoothedEpsilon;
  const uint LatestTimeSynchronization;
  const uint LatestTimeSyncReception;
  bool TimeNotifyDiscontinuity;
  uint PrevDiscontinuityTime;
  const float TimeCorrectionWeight;
  float TimeThreshold;
  float TimeSmoothing;
  uint TimeLookahead;
};

struct CNetConnection : public CMwNod {
  const bool ClientToServer;
  CMwNod* Info;
  const uint TCPPort;
  const uint UDPPort;
  const uint State;
  const bool Broken;
  const bool ConnectionTCP;
  const bool ConnectionWaiting;
  const bool ConnectionRequest;
  const bool TestingUDP;
  const bool Connected;
  const bool CanReceiveTCP;
  const bool CanSendTCP;
  const bool CanReceiveUDP;
  const bool CanSendUDP;
  const uint TCPEmissionQueue;
  const bool WasUDPPacketDropped;
  const bool IsTCPSaturated;
  const uint LatestNetworkActivity;
  const uint LatestUDPActivity;
  const uint ReceptionPacketTotal;
  const uint UDPReceptionPacketTotal;
  const uint TCPReceptionPacketTotal;
  const uint SendingPacketTotal;
  const uint UDPSendingPacketTotal;
  const uint TCPSendingPacketTotal;
  const uint ReceptionNodTotal;
  const uint UDPReceptionNodTotal;
  const uint TCPReceptionNodTotal;
  const uint SendingNodTotal;
  const uint UDPSendingNodTotal;
  const uint TCPSendingNodTotal;
  const uint ReceptionPacketLoss;
  const uint ReceptionPacketTotalWithoutLoss;
  const uint ReceptionCounter;
  const uint SendingCounter;
  void Disconnect();
};

struct CNetFormConnectionAdmin : public CNetNod {
  CNetFormConnectionAdmin();

};

struct CNetHttpClient : public CMwNod {
  string Server;
  uint Port;
  uint Context;
  string ProxyUrl;
  const string LastError;
  string LanguageHeader;
  string UserAgentHeader;
  const MwFastBuffer<CNetHttpResult*> Requests;
};

struct CNetHttpResult : public CMwNod {
  const string Server;
  const string Path;
  enum class CNetHttpResult::EKind {
    Download = 0,
    Upload = 1,
  };
  const CNetHttpResult::EKind Kind;
  enum class CNetHttpResult::EStatus {
    Connecting = 0,
    Request = 1,
    Downloading = 2,
    Done = 3,
    Error = 4,
  };
  const CNetHttpResult::EStatus Status;
  const uint ExpectedTransfertSize;
  const string ContentType;
  const string ContentEncoding;
  const uint CurrentSize;
  const uint HttpError;
};

struct CNetMasterServer : public CMwNod {
  string GameVersion;
  const MwFastBuffer<CNetMasterServerUserInfo*> MasterServerUserInfos;
  const MwFastBuffer<CNetMasterServerDownload*> Downloads;
  const MwFastBuffer<CNetMasterServerDownload*> CurrentDownloads;
  const MwFastBuffer<CNetHttpClient*> HttpDownloadClients;
  const MwFastBuffer<CNetMasterServerUptoDateCheck*> UpToDateChecks;
  const MwFastBuffer<CNetMasterServerUptoDateCheck*> CurrentUpToDateChecks;
  const MwFastBuffer<CNetHttpClient*> HttpCheckUpToDateClients;
};

struct CNetMasterHost : public CMwNod {
  string NetHostName;
  const string TitleIdName;
};

struct CNetFileTransfer : public CMwNod {
  const MwFastBuffer<CNetIPSource*> IPSources;
  const MwFastBuffer<CNetFileTransferUpload*> Uploads;
  const MwFastBuffer<CNetFileTransferDownload*> Downloads;
  const MwFastBuffer<CNetFileTransferDownload*> TerminatedDownloads;
  const bool DownloadEnabled;
  const bool UploadEnabled;
  const bool WaitForDownload;
  const string IPAddress;
  const uint P2PKey;
  const uint ClientUId;
  const bool IsServer;
  const uint UploadRate;
  const uint DownloadRate;
  const uint NbOfCurrentUls;
  const uint SendBandwidthLimit;
  const uint TotalSendingSize;
  const uint NbSendChannelLeft;
  const uint NbSendChannelToRestore;
  const uint NbSendChannelUsed;
  const uint FirstTimeNotEnoughSendBandwidth;
  const bool IsEmissionSaturated;
  const uint NbOfCurrentDls;
  const uint ReceiveBandwidthLimit;
  const uint TotalReceivingSize;
  const uint NbReceiveChannelLeft;
  const uint NbReceiveChannelToRestore;
  const uint NbReceiveChannelUsed;
  const uint FirstTimeNotEnoughReceiveBandwidth;
  const uint MaxDownloadRateLeft;
  const uint MaxDownloadChannelLeft;
  const uint MaxUploadRateLeft;
  const uint MaxUploadChannelLeft;
  const uint MinSizeTransfer;
  const uint MaxSizeTransfer;
  const uint MaxDownloads;
  const uint MaxUploads;
  const uint MaxChannelPerTransfer;
  const uint NewConnectionTimeoutForDownload;
  const uint NewConnectionTimeoutForUpload;
  const uint SendMsgThroughServerTimeoutForDownload;
  const uint SendMsgThroughServerTimeoutForUpload;
  const uint LastUpdateTime;
  const uint UpdateDelta;
  const uint WriteOnDiskError;
  CNetServer* const Server;
  CNetClient* const Client;
  CNetMasterServer* const MasterServer;
  CSystemPackManager* const PackManager;
  CSystemFidsFolder* const DiskCacheDir;
};

struct CNetMasterServerInfo : public CMwNod {
  string Addr;
  string Path;
  string Name;
  wstring Message;
  bool LadderEnabled;
};

struct CNetFileTransferNod : public CMwNod {
  const wstring Name;
  const string Checksum;
  CSystemFidFile* const File;
  const uint TotalSize;
  const uint CurrentOffset;
};

struct CNetFileTransferForm : public CNetNod {
  CNetFileTransferForm();

};

struct CNetFileTransferDownload : public CNetFileTransferNod {
  const uint IdDownload;
  const UnnamedEnum DownloadState;
  CSystemPackDesc* const PackDesc;
  const uint PriorityLevel;
  const uint PriorityFlags;
  const MwFastBuffer<CNetSource*> Sources;
  const uint NbOfEffectiveSources;
  CNetSource* const ActiveSource;
  const MwFastBuffer<CNetURLSource*> UrlSources;
  CNetURLSource* const ActiveUrlSource;
  const string Url;
  const string LastUrlUsed;
  const bool AcceptP2P;
  const bool AcceptUrl;
  const bool SkipServerSource;
  const bool IsNearFinished;
  const wstring TempFileName;
  const uint LastWriteTimeout;
  CNetMasterServerUptoDateCheck* const PackDescUpToDateCheck;
  const bool PackDescUpToDateChecked;
  const uint LastDataMsgTime;
  const uint LastDataReception;
  const uint LastDataWrite;
  const uint SendEfficiency;
  const uint InstantaneousEfficiency;
  const uint AverageEfficiency;
};

struct CNetFileTransferUpload : public CNetFileTransferNod {
  const uint IdUpload;
  const uint IdDownload;
  const uint IdSource;
  const UnnamedEnum UploadState;
  CNetIPSource* const IPSource;
  CNetConnection* const Connection;
  const uint DownloadPriorityLevel;
  const uint PriorityLevel;
  const uint NbChannelsUsed;
  const uint NbChannelsUsedValidated;
  const uint TimeOut;
  const uint ValidityTimeOut;
  const bool IsActive;
  const bool MustBeActive;
  const uint LastActiveTime;
  const bool IsConnecting;
  const bool FirstToConnect;
  const bool MustSendReqAck;
  const bool ReqAckSent;
  const uint UploadAttempt;
  const uint MsgAttempt;
  bool ForceCancel;
  const uint ReadOffset;
  const uint LastMessageTime;
  const uint LastReadTimeOut;
  const uint LastSentTime;
  const uint LastDataComplete;
  const uint InstantaneousEfficiency;
  const uint AverageEfficiency;
};

struct CNetSource : public CMwNod {
  CNetSource();

  const uint IdSource;
  const uint IdUpload;
  const UnnamedEnum SourceState;
  CNetFileTransferDownload* const Download;
  CNetIPSource* const SourceAddress;
  CNetConnection* const Connection;
  const uint LastMessageTime;
  const uint TimeOut;
  const uint NbChannelsUsed;
  const uint LastNbChannelsUsedProposition;
  const bool TestedAtLeastOnce;
  const bool OwnsFile;
  const bool IsConnecting;
  const bool FirstToConnect;
  const bool ReqSent;
  const bool MustSendRequest;
  const bool MustSendUploadAck;
  const bool ForceSending;
  const bool HasReceivedUrl;
  bool InterruptTransfer;
  bool ForceCancel;
  const uint SourceAttempt;
  const uint MsgAttempt;
};

struct CNetIPC : public CMwNod {
  CNetIPC();

  const uint Port;
  const string VersionString;
  uint MaxPacketSize;
};

struct CNetFormRpcCall : public CNetNod {
  CNetFormRpcCall();

};

struct CNetUPnP : public CMwNod {
};

struct CWebServicesTaskWait : public CWebServicesTaskSequence {
};

struct CNetMasterServerRequest : public CMwNod {
};

struct CNetIPSource : public CMwNod {
  const string RemoteIP;
  const uint DownloadRate;
  const uint UploadRate;
  const bool IsServer;
  const bool IsUploadEnabled;
  const uint P2PKey;
  const uint ClientUId;
  CNetConnection* const GameConnection;
  CNetConnection* const ConnectionFrom;
  CNetConnection* const ConnectionTo;
  const uint ConnectTimeOut;
  const uint LastConnectionTime;
  const uint LastContactTime;
  const uint ConnectionFromTimeOut;
  const uint ConnectionToTimeOut;
  const bool IsConnecting;
  const uint ConnectionAttempt;
  const bool ConnectionTested;
  const bool ConnectionPossible;
  const uint ThroughServerAttempt;
  const bool ThroughServerTested;
  const bool ThroughServerPossible;
  const bool CanBeConnectedBy;
  const MwFastBuffer<CNetSource*> SourcesUsingConnectionFrom;
  const MwFastBuffer<CNetFileTransferUpload*> UploadsUsingConnectionFrom;
  const MwFastBuffer<CNetSource*> SourcesUsingConnectionTo;
  const MwFastBuffer<CNetFileTransferUpload*> UploadsUsingConnectionTo;
  bool ForceCancel;
};

struct CNetMasterServerUptoDateCheck : public CMwNod {
  const string Server;
  const string Path;
  const uint Port;
  const uint PriorityLevel;
  const bool IsUpToDate;
  const uint ReturnedError;
  CNetHttpResult* const HttpResult;
  const bool IsFinished;
};

struct CNetURLSource : public CMwNod {
  const UnnamedEnum URLSourceState;
  CNetFileTransferDownload* const Download;
  CNetFileTransferDownload* const MasterServerDownload;
  const string Url;
};

// Description: "Manager for HTTP requests"
struct CNetScriptHttpManager : public CMwNod {
  CNetScriptHttpRequest* CreateGet(string Url); // Maniascript
  CNetScriptHttpRequest* CreateGet2(string Url, bool UseCache); // Maniascript
  CNetScriptHttpRequest* CreateGet3(string Url, bool UseCache, string AdditionalHeaders); // Maniascript
  CNetScriptHttpRequest* CreatePost(string Url, string Resource); // Maniascript
  CNetScriptHttpRequest* CreatePost2(string Url, string Resource, string AdditionalHeaders); // Maniascript
  CNetScriptHttpRequest* CreatePostFile(string Url, wstring FileName, string AdditionalHeaders); // Maniascript
  CNetScriptHttpRequest* CreatePut(string Url, string Resource, string AdditionalHeaders); // Maniascript
  void Destroy(CNetScriptHttpRequest* Request); // Maniascript
  bool IsValidUrl(string Url); // Maniascript
  const MwNodPool<CNetScriptHttpRequest*> Requests; // Maniascript
  const uint SlotsAvailable; // Maniascript
  const MwNodPool<CNetScriptHttpEvent*> PendingEvents; // Maniascript
  bool AutomaticHeaders_Timezone; // Maniascript
};

// Description: "An HTTP request."
struct CNetScriptHttpRequest : public CMwNod {
  const string Url; // Maniascript
  const wstring Result; // Maniascript
  const uint StatusCode; // Maniascript
  const bool IsCompleted; // Maniascript
};

struct CNetXmpp_Timer : public CMwNod {
  CNetXmpp_Timer();

};

struct CNetMasterServerDownload : public CMwNod {
  const string Server;
  const string Path;
  const uint Port;
  enum class CNetMasterServerDownload::EContext {
    None = 0,
    Browser = 1,
    Menu = 2,
    Screen = 3,
    InGameInterface = 4,
  };
  const CNetMasterServerDownload::EContext Context;
  const uint Offset;
  enum class CNetMasterServerDownload::EUploadMethod {
    None = 0,
    Post = 1,
    Put = 2,
  };
  const CNetMasterServerDownload::EUploadMethod UploadMethod;
  const uint TotalSize;
  const uint DownloadStartTime;
  const uint DownloadingTime;
  const uint PriorityLevel;
  const uint ReturnedError;
  const string RequestHeaders;
  const uint Timeout;
  CNetHttpResult* const HttpResult;
  const bool IsFinished;
  const bool IsPaused;
};

struct CWebServicesTaskWaitMultiple : public CWebServicesTaskSequence {
};

struct CNetFormNewPing : public CNetNod {
  CNetFormNewPing();

};

// Description: "Masterserver user info."
struct CNetMasterServerUserInfo : public CMwNod {
  enum class CNetMasterServerUserInfo::EFirstPartySignInState {
    Unknown = 0,
    NotSignedUp = 1,
    SignedOut = 2,
    SignedIn = 3,
  };
  enum class CNetMasterServerUserInfo::EMasterServerConnectionStatus {
    NotConnected = 0,
    Connecting = 1,
    Connected = 2,
    Disconnecting = 3,
  };
  enum class CNetMasterServerUserInfo::EMasterServerConnectionDetailedStatus {
    NotConnected = 0,
    CheckingNetworkAvailability = 1,
    CheckingMasterServerConnexion = 2,
    WaitingCredentials = 3,
    CheckingMasterServerWaitingQueue = 4,
    WaitingOnMasterServerQueue = 5,
    WaitingSubscriptionCredentials = 6,
    CheckingSubscriptionCredentials = 7,
    WaitingSubscriptionInfo = 8,
    Subscribe = 9,
    OpeningSession = 10,
    Identifying = 11,
    GettingZones = 12,
    WaitingNewZone = 13,
    ChangingZone = 14,
    GettingOnlineProfile = 15,
    SynchronizingProfileChunks = 16,
    AssociatingKey = 17,
    UpdatingProfile = 18,
    GettingBannedCryptedChecksumsList = 19,
    GettingTitleInfo = 20,
    ConnectingToUbiServices = 21,
    FinalizingConnection = 22,
    Connected = 23,
    Disconnecting = 24,
  };
  enum class CNetMasterServerUserInfo::EMasterServerConnectionError {
    NetworkNotAvailable = 0,
    FirstPartySignedOut = 1,
    NoUbiServicesSession = 2,
    MasterServerDisabled = 3,
    MasterServerInSafeMode = 4,
    KillSwitchEnabled = 5,
    BetaBanned = 6,
    BetaNotApproved = 7,
    NoUplayPC = 8,
  };
  const int SysUserId;
  const string Login; // Maniascript
  const wstring DisplayName; // Maniascript
  const string Password;
  const string NewPassword;
  CNetMasterServerInfo* const MasterServerInfo;
  const CNetMasterServerUserInfo::EMasterServerConnectionStatus ConnectionStatus; // Maniascript
  const string LastConnectionErrorType; // Maniascript
  const string LastConnectionErrorCode; // Maniascript
  const wstring LastConnectionErrorDescription; // Maniascript
};

struct CWebServicesTask : public CMwNod {
};

struct CWebServicesTaskSequence : public CWebServicesTask {
};

// Description: "Asynchronous task result."
struct CWebServicesTaskResult : public CMwNod {
  const bool IsProcessing; // Maniascript
  const bool HasSucceeded; // Maniascript
  const bool HasFailed; // Maniascript
  const bool IsCanceled; // Maniascript
  const string ErrorType; // Maniascript
  const string ErrorCode; // Maniascript
  const wstring ErrorDescription; // Maniascript
  void Cancel(); // Maniascript
};

struct CWebServicesTaskScheduler : public CMwNod {
};

struct CNetUbiServices : public CMwNod {
};

struct CNetUbiServicesTask : public CWebServicesTaskSequence {
};

struct CNetUbiServicesTask_CreateSession : public CNetUbiServicesTask {
};

struct CNetUbiServicesTask_DeleteSession : public CNetUbiServicesTask {
};

struct CNetUbiServicesTask_Party_GetMaxMemberLimit : public CNetUbiServicesTask {
};

struct CNetUbiServicesTask_Party_SetMaxMemberLimit : public CNetUbiServicesTask {
};

struct CWebServicesTask_Party_SetMaxMemberLimit : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Empty : public CWebServicesTaskVoid {
};

struct CNetUplayPC : public CMwNod {
};

struct CNetNadeoServicesTask_GetAccountIdFromWebServicesIdentity : public CNetNadeoServicesRequestTask {
};

struct CNetUbiServicesTask_Profile_RetrieveProfileInfoList : public CNetUbiServicesTask {
};

struct CWebServicesTask_GetDisplayNameFromWebServicesUserId : public CWebServicesTaskSequence {
};

struct CWebServicesTask_GetDisplayNameFromWebServicesIdentity : public CWebServicesTaskSequence {
};

struct CNetMasterServerRequestTask : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_Bool : public CWebServicesTaskResult {
  const bool Value; // Maniascript
};

struct CWebServicesTaskResult_String : public CWebServicesTaskResult {
  const string Value; // Maniascript
};

struct CNetMasterServerTask_GetClientConfigUrls : public CNetMasterServerRequestTask {
};

struct CNetUplayPCTask_Overlay_ShowMicroApp : public CWebServicesTaskSequence {
};

struct CNetUbiServicesTask_SendNotification : public CNetUbiServicesTask {
};

struct CNetUbiServicesTask_CheckNewNotification : public CWebServicesTaskSequence {
};

struct CWebServicesTask_CheckNetworkAvailability : public CWebServicesTaskSequence {
};

struct CNetUplayPCUserInfo : public CMwNod {
};

struct CWebServicesTask_GetFriendList : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_WSFriendList : public CWebServicesTaskResult {
};

struct CNetMasterServerTask_GetApplicationConfig : public CNetMasterServerRequestTask {
};

struct CWebServicesTaskResult_ClientConfig : public CWebServicesTaskResult {
};

struct CNetMasterServerTask_GetWaitingParams : public CNetMasterServerRequestTask {
};

struct CNetMasterServerTask_CheckLoginForSubscribe : public CNetMasterServerRequestTask {
};

struct CNetMasterServerTask_Subscribe : public CNetMasterServerRequestTask {
};

struct CNetMasterServerTask_OpenSession : public CNetMasterServerRequestTask {
};

struct CNetMasterServerTask_Connect : public CNetMasterServerRequestTask {
};

struct CWebServicesTaskResult_Session_Get : public CWebServicesTaskResult {
  const string SessionId; // Maniascript
  const string ServerLogin; // Maniascript
  const string ServerPassword; // Maniascript
};

struct CWebServicesTask_OpenNewsLink : public CWebServicesTaskSequence {
};

struct CNetUbiServicesTask_GetNews : public CNetUbiServicesTask {
};

struct CNetMasterServerTask_Session_Get : public CWebServicesTaskSequence {
};

struct CNetMasterServerTask_Session_JoinOrCreate : public CWebServicesTaskSequence {
};

struct CNetMasterServerTask_Session_Leave : public CWebServicesTaskSequence {
};

struct CNetMasterServerTask_Session_InviteBuddy : public CWebServicesTaskSequence {
};

struct CNetUbiServicesTask_Party_UpdateInvitation : public CNetUbiServicesTask {
};

struct CWebServicesTask_Party_CancelInvitation : public CWebServicesTaskSequence {
};

struct CWebServicesTask_GetUserPrestigeSelected : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_AddSubscriptionFromPSN : public CNetNadeoServicesRequestTask {
};

struct CNetMasterServerTask_GetFeatureTimeLimit : public CNetMasterServerRequestTask {
};

struct CWebServicesTaskResult_Natural : public CWebServicesTaskResult {
  const uint Value; // Maniascript
};

struct CNetMasterServerTask_CheckFeatureTimeLimit : public CNetMasterServerRequestTask {
};

struct CNetMasterServerTask_SetFeatureTimeUse : public CNetMasterServerRequestTask {
};

struct CWebServicesTaskResult_PlayerFeatureLimitList : public CWebServicesTaskResult {
};

struct CWebServicesTask_GetStatList : public CWebServicesTaskSequence {
};

struct CWebServicesTask_GetBlockList : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_StringInt : public CWebServicesTaskResult {
  const wstring Value; // Maniascript
};

struct CNetUbiServicesTask_GetUnsentEvents : public CNetUbiServicesTask {
};

struct CNetUbiServicesTask_RefreshSession : public CNetUbiServicesTask {
};

struct CNetNadeoServicesTask_GetAllPrestigeList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSAccountSkinFavorite : public CWebServicesTaskResult {
};

struct CNetUbiServicesTask_RetrieveBetaUserInfo : public CNetUbiServicesTask {
};

struct CNetUbiServicesTask_AcceptNDA : public CNetUbiServicesTask {
};

struct CNetUplayPCTask_Achievement_Unlock : public CWebServicesTaskSequence {
};

struct CWebServicesTaskVoid : public CWebServicesTask {
};

struct CNetMasterServerTask_ImportAccount : public CNetMasterServerRequestTask {
};

struct CNetMasterServerTask_ImportAccount_IsFinished : public CNetMasterServerRequestTask {
};

struct CWebServicesTaskResult_OpenSession : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_StringList : public CWebServicesTaskResult {
  const MwFastBuffer<string> Values; // Maniascript
};

struct CWebServicesTaskResult_StringIntList : public CWebServicesTaskResult {
  const MwFastBuffer<wstring> Values; // Maniascript
};

// Description: "An HTTP event."
struct CNetScriptHttpEvent : public CMwNod {
  enum class CNetScriptHttpEvent::EType {
    RequestComplete = 0,
  };
  const CNetScriptHttpEvent::EType Type; // Maniascript
  CNetScriptHttpRequest* const Request; // Maniascript
};

struct CNetNadeoServices : public CMwNod {
};

struct CNetNadeoServicesUserInfo : public CMwNod {
};

struct CNetNadeoServicesRequestTask : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesRequest : public CMwNod {
};

struct CWebServicesTaskResult_UrlConfig : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_AuthenticateCommon : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetClientConfig : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_AuthenticateWithBasicCredentials : public CNetNadeoServicesTask_AuthenticateCommon {
};

struct CNetNadeoServicesTask_AuthenticateWithUbiServices : public CNetNadeoServicesTask_AuthenticateCommon {
};

struct CNetNadeoServicesTask_AuthenticateWithUnsecureAccountId : public CNetNadeoServicesTask_AuthenticateCommon {
};

struct CWebServicesTask_GetWebIdentityFromWebServicesUserId : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_Connect : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetAccountClientSignature : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_SetClientCaps : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_Signature : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetEncryptedPackageAccountKey : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetZones : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSZoneList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesRequestManager : public CMwNod {
};

struct CNetNadeoServicesTask_GetAccountClientUrls : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetAccountZone : public CNetNadeoServicesRequestTask {
};

struct CWebServices : public CMwNod {
};

// Description: "WebServices user info."
struct CWebServicesUserInfo : public CMwNod {
  const MwId Id; // Maniascript
  const int SysUserId;
  const string WebServicesUserId; // Maniascript
  const string NadeoServicesAccountId; // Maniascript
};

struct CWebServicesTask_Connect : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_GetApiRequests : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_ConnectToNadeoServices : public CWebServicesTaskSequence {
};

struct CWebServicesTask_UpdateClientConfig : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_NSWebServicesIdentityList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetWebServicesIdentityFromAccountId : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_ApplicationConfig : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetProfileChunkList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_SetProfileChunk : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSProfileChunk : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_DeleteProfileChunk : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetAccountStationList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetEncryptedPackageList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSEncryptedPackageList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetClientFileList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSClientFileList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetEncryptedPackageVersionList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSEncryptedPackageVersionList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetEncryptedPackageVersionCryptKey : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSEncryptedPackageVersionCryptKey : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_LoadStation : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSStation : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_UnloadStation : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetAccountLadderInfo : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSLadderAccountInfo : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetAuthenticationToken : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetAccountMapRecordList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSMapRecordList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_SetMapRecordAttempt : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetAccountGroupList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetAccountPolicyRuleValueList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSPolicyRuleValueList : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_NSAccountZone : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_SetAccountZone : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetClientUpdaterFile : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetAccountEncryptedPackageList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetAccountTitleList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSTitleList : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_NSTitle : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_SetTitle : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSEncryptedPackage : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_SetEncryptedPackage : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_CreateEncryptedPackageVersion : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSMap : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_SetMap : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_SetCampaign : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSMapList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetMapList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSCampaignList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetCampaignList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSEncryptedPackageVersionWithCryptKey : public CWebServicesTaskResult {
};

struct CNetUbiServicesTask_RequestUserLegalOptinsStatus : public CNetUbiServicesTask {
};

struct CNetNadeoServicesTask_GetAccountClientPluginList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_AddMapListToCampaign : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_RemoveMapListFromCampaign : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetMap : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetAccountMapList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetAccountSkinFavoriteList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSAccountSkinFavoriteList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetSeason : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSSeason : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_NSSeasonList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetSeasonList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_SetSeason : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_AddMapListToSeason : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_RemoveMapListFromSeason : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetAccountSeasonList : public CNetNadeoServicesRequestTask {
};

struct CNetUplayPCTask_JoinSession : public CWebServicesTaskSequence {
};

struct CNetUplayPCTask_LeaveSession : public CWebServicesTaskSequence {
};

struct CNetUplayPCTask_ShowInviteUI : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_NSServer : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_SetServer : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetServer : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_DeleteServer : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_CheckSubscription : public CWebServicesTaskSequence {
};

struct CNetUplayPCTask_GetUserConsumableItemList : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_UPCConsumableItemList : public CWebServicesTaskResult {
};

struct CWebServicesTask_UpdateUserConfig : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_AddSubscription : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_CreateSkin : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSSkin : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetSkinList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSSkinList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetCreatorSkinList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_SetAccountSkin : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSAccountSkin : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetAccountSkinList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSAccountSkinList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_RefreshNadeoServicesAuthenticationToken : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_AddClientLog : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetAccountXp : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSAccountXp : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_AddToWaitingQueue : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSWaitingInfo : public CWebServicesTaskResult {
};

struct CWebServicesTask_CheckLoginExists : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_CheckLoginExists : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_CreateAccount : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_CreateUserAccount : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_AddAccountPasswordReset : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_SetAccountPassword : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_SendResetPasswordRequest : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_SetAccountPresence : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSAccountPresence : public CWebServicesTaskResult {
};

struct CWebServicesTask_Disconnect : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_DeleteAccountPresence : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_DisconnectFromNadeoServices : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_AddAccountSkinFavorite : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_RemoveAccountSkinFavorite : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetSkin : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_CheckWaitingQueue : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_UnsetAccountSkin : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSAccountTrophyLastYearSummary : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetAccountTrophyLastYearSummary : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetTrophySettings : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSTrophySettings : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetUserAccount : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSUserAccount : public CWebServicesTaskResult {
};

struct CWebServicesTask_SetUserZone : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_GetAccountDisplayNameList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_SetTrophyLiveTimeAttackAchievementResult : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSAccountTrophyGainList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_UpdateUserAccount : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_ResetPassword : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_ResetAccountPassword : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_SetTrophyCompetitionMatchAchievementResult : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSAccountTrophyGainHistory : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetAccountTrophyGainHistory : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSItemCollection : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_SetItemCollection : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSItemCollectionList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetItemCollection : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetItemCollectionList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetAccountItemCollectionList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSItemCollectionVersion : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_NSItemCollectionVersionList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_CreateItemCollectionVersion : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetItemCollectionVersion : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetItemCollectionVersionList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSAccountItemCollectionFavorite : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_NSAccountItemCollectionFavoriteList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_AddAccountItemCollectionFavorite : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetAccountItemCollectionFavoriteList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_RemoveAccountItemCollectionFavorite : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSUpload : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_CreateUpload : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetUpload : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_UploadPart : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_Upload : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_SetItemCollectionActivityId : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetClub : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSClubList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetClubList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetItemCollectionListByUniqueIdentifierList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_UploadSessionReplay : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_WSFriendInfoList : public CWebServicesTaskResult {
};

struct CWebServicesTask_RetrieveFriendList : public CWebServicesTaskSequence {
};

struct CNetUplayPCTask_GetFriendList : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_UPCFriendList : public CWebServicesTaskResult {
};

struct CNetUbiServicesTask_GetFriendList : public CNetUbiServicesTask {
};

struct CWebServicesTaskResult_NSSquad : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_CreateSquad : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetSquad : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_AddSquadInvitation : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_RemoveSquadInvitation : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_AcceptSquadInvitation : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_DeclineSquadInvitation : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_RemoveSquadMember : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_SetSquadLeader : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_LeaveSquad : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetAccountSquad : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSAccountZoneList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetAccountZoneList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSAccountSubscriptionList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetAccountSubscriptionList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetAccountSkinListByAccountList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSAccountClubTag : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetAccountClubTag : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_SetAccountClubTag : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_GetUserClubTag : public CWebServicesTaskSequence {
};

struct CWebServicesTask_SetUserClubTag : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_NSAccountClubTagList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetAccountClubTagList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_GetUserClubTagList : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_NSDriverBotGroupList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetDriverBotGroupList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_AddDriverBotGroupList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_GetSeasonPlayableList : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_GetAccountPlayableSeasonList : public CNetNadeoServicesRequestTask {
};

struct CNetUbiServicesTask_Profile_RetrieveUplayProfileInfoList : public CNetUbiServicesTask {
};

struct CNetUbiServicesTask_Profile_RetrieveProfileInfoListFromPlatformTypeAndUserId : public CNetUbiServicesTask {
};

struct CWebServicesTask_GetUserZone : public CWebServicesTaskSequence {
};

struct CWebServicesTask_GetUserZoneList : public CWebServicesTaskSequence {
};

struct CWebServicesTask_GetZoneList : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_WSZoneList : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_WSZonePtrList : public CWebServicesTaskResult {
};

struct CWebServicesTask_Permission_CheckCrossPlay : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Permission_CheckPlayMultiplayerAsync : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Permission_CheckPlayMultiplayerMode : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Permission_CheckPlayMultiplayerSession : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Permission_CheckPrivilegeForAllUsers : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Permission_CheckTargetedUseUserCreatedContent : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Permission_CheckTargetedUseUserCreatedContentForAllUsers : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Permission_CheckTargetedViewUserGameHistory : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Permission_CheckViewOnlinePresence : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Permission_CheckUseUserCreatedContent : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_CheckTargetedPrivilegeResult : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_WSNewsList : public CWebServicesTaskResult {
};

struct CWebServicesTask_GetUserNewsList : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_UbiServicesNewsList : public CWebServicesTaskResult {
};

struct CWebServicesTask_GetMapList : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_WSMapPtrList : public CWebServicesTaskResult {
};

struct CWebServicesTask_StartMapRecordAttempt : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_NSMapRecordAttempt : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_CreateMapRecordSecureAttempt : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_StopMapRecordAttempt : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_PatchMapRecordSecureAttempt : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSMapRecordSecureAttempt : public CWebServicesTaskResult {
};

struct CNetUbiServicesTask_GetStatList : public CNetUbiServicesTask {
};

struct CWebServicesTaskResult_UbiServicesStatList : public CWebServicesTaskResult {
};

struct CWebServicesTask_GetFirstPartyAchievementList : public CWebServicesTaskSequence {
};

struct CWebServicesTask_UpdateFirstPartyAchievementCompletion : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_WSPrestigeList : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_WSUserPrestigeList : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_WSUserPrestige : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_UPCAchievementCompletionList : public CWebServicesTaskResult {
};

struct CNetUplayPCTask_Achievement_GetCompletionList : public CWebServicesTaskSequence {
};

struct CWebServicesTask_UserProfile_GetAvatarUrl : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_WSBlockedUserList : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_NSAccountPrestigeList : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_NSCurrentAccountPrestige : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_NSPrestigeList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetAccountPrestigeList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetCurrentAccountPrestige : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetPrestigeList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetPrestigeListFromModeYearType : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_SetAccountPrestigeCurrent : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_GetUserPrestigeList : public CWebServicesTaskSequence {
};

struct CWebServicesTask_SetUserPrestigeSelected : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_UnsetAccountPrestigeCurrent : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_GetPrestigeList : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_WSPrestige : public CWebServicesTaskResult {
};

struct CWebServicesTask_GetPrestige : public CWebServicesTaskSequence {
};

struct CNetUbiServicesTask_PlayerPreferences_SetStandardPreferences : public CNetUbiServicesTask {
};

struct CNetUbiServicesTask_PlayerPreferences_GetStandardPreferences : public CNetUbiServicesTask {
};

struct CWebServicesTaskResult_UbiServicesPlayerPreferencesStandard : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_NSAccountPrestige : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetAccountPrestigeCurrent : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_GetUserPrestigeSelectedForUser : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_GetCurrentAccountPrestigeList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_GetUserPrestigeSelectedForUserList : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_UbiServicesPartyInfo : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_WSPartyInfo : public CWebServicesTaskResult {
};

struct CWebServicesTask_Party_RetrievePartyInfo : public CWebServicesTaskSequence {
};

struct CNetUbiServicesTask_Party_GetPartyInfo : public CNetUbiServicesTask {
};

struct CNetUbiServicesTask_Party_CreateParty : public CNetUbiServicesTask {
};

struct CWebServicesTask_Party_Create : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Party_Leave : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Party_RenewPartyExpiration : public CWebServicesTaskSequence {
};

struct CNetUbiServicesTask_Party_AutoRemovePartyMemberOnDisconnect : public CNetUbiServicesTask {
};

struct CWebServicesTaskResult_UbiServicesPartyInvitationList : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_UbiServicesPartyJoinRequestList : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_UbiServicesPartyMemberList : public CWebServicesTaskResult {
};

struct CNetUbiServicesTask_Party_GetPartyInvitationList : public CNetUbiServicesTask {
};

struct CNetUbiServicesTask_Party_GetPartyJoinRequestList : public CNetUbiServicesTask {
};

struct CNetUbiServicesTask_Party_GetPartyMemberList : public CNetUbiServicesTask {
};

struct CNetUbiServicesTask_Party_LeaveParty : public CNetUbiServicesTask {
};

struct CNetUbiServicesTask_Party_RenewExpiration : public CNetUbiServicesTask {
};

struct CWebServicesTask_Party_RetrievePartyInvitationList : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Party_RetrievePartyJoinRequestList : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Party_RetrievePartyMemberList : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_WSPartyInvitationList : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_WSPartyJoinRequestList : public CWebServicesTaskResult {
};

struct CWebServicesTaskResult_WSPartyMemberList : public CWebServicesTaskResult {
};

struct CWebServicesTask_Party_RequestAutoRemovePartyMemberOnDisconnect : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_WSPartyCompleteInfo : public CWebServicesTaskResult {
};

struct CWebServicesTask_Party_RetrievePartyCompleteInfo : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_GetVisualNotificationUrlInfo : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_UbiServicesVisualNotificationUrlInfo : public CWebServicesTaskResult {
};

struct CWebServicesTask_UbisoftConnect_Show : public CWebServicesTaskSequence {
};

struct CWebServicesTask_UserProfile_ShowUbisoftConnectProfile : public CWebServicesTaskSequence {
};

struct CNetUbiServicesTask_Party_GetFirstPartySessionInfo : public CNetUbiServicesTask {
};

struct CNetUbiServicesTask_Party_UpdateParty : public CNetUbiServicesTask {
};

struct CWebServicesTask_Party_Update : public CWebServicesTaskSequence {
};

struct CNetUbiServicesTask_Party_UpdateLockState : public CNetUbiServicesTask {
};

struct CWebServicesTask_Party_SetLocked : public CWebServicesTaskSequence {
};

struct CNetUbiServicesTask_Profile_RetrieveProfileInfoListFromPlatform : public CNetUbiServicesTask {
};

struct CWebServicesTask_GetWebServicesUserIdFromWebIdentity : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_GetMapRecordList : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_AddClientDebugInfo : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTask_Preference_RetrieveUserPreference : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Event_AddMapSession : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_AddTelemetryMapSession : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_GetMapVote : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_VoteMap : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_Activity_CreateMatch : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_Activity_ReportMatchResult : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_WSMapRecordList : public CWebServicesTaskResult {
};

struct CWebServicesTask_GetMapRecordListByMapRecordContextAndUserList : public CWebServicesTaskSequence {
};

struct CNetUbiServicesTask_Party_ChangeFirstPartySessionId : public CNetUbiServicesTask {
};

struct CNetNadeoServicesTask_Activity_UpdateMatch : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSDriverBotGroupAddLimit : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_AddDriverBotGroupList_GetLimit : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_UbiServicesBlockList : public CWebServicesTaskResult {
};

struct CNetUbiServicesTask_Blocklist_Get : public CNetUbiServicesTask {
};

struct CWebServicesTaskResult_UbiServicesProfileConsent : public CWebServicesTaskResult {
};

struct CNetUbiServicesTask_PlayerConsents_GetConsent : public CNetUbiServicesTask {
};

struct CWebServicesTaskResult_UbiServicesProfileAcceptanceList : public CWebServicesTaskResult {
};

struct CNetUbiServicesTask_PlayerConsents_GetAcceptanceList : public CNetUbiServicesTask {
};

struct CWebServicesTask_Permission_GetPlayerInteractionRestriction : public CWebServicesTaskSequence {
};

struct CWebServicesTask_Permission_GetPlayerInteractionStatusList : public CWebServicesTaskSequence {
};

struct CWebServicesTaskResult_Integer : public CWebServicesTaskResult {
  const int Value; // Maniascript
};

struct CWebServicesTask_RetrieveUserPrestigeLevelList : public CWebServicesTaskSequence {
};

struct CWebServicesTask_RetrievePrestigeInfoList : public CWebServicesTaskSequence {
};

struct CNetNadeoServicesTask_GetAccountMapFavoriteList : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSAccountMapFavoriteList : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_GetAccountMapFavoriteListByMapUid : public CNetNadeoServicesRequestTask {
};

struct CNetNadeoServicesTask_AddAccountMapFavorite : public CNetNadeoServicesRequestTask {
};

struct CWebServicesTaskResult_NSAccountMapFavorite : public CWebServicesTaskResult {
};

struct CNetNadeoServicesTask_RemoveAccountMapFavorite : public CNetNadeoServicesRequestTask {
};

} // namespace Net

namespace Input {

struct CInputPort : public CMwNod {
  const UnnamedEnum InputsMode;
  const string CurrentActionMap;
  bool IsDoingIME;
  const bool IsFocused;
  UnnamedEnum MouseVisibility;
  float RumbleIntensity; // Range: 0 - 2
  float CenterSpringIntensity; // Range: 0 - 1
  float ForceFeedbackIntensity; // Range: 0 - 1
  bool PollingEnabled;
  const bool DeviceHasBeenHotPlugged;
  const uint DevicePlugEventCount;
  void DeviceHotPlugUpdate();
  const MwFastBuffer<CInputDevice*> ConnectedDevices;
  uint MaxSampleRate;
  uint MinHistoryLength;
  const uint EventInStoreCount;
  bool IgnoreFocusForGamePads;
  uint FakeInputLagAvg; // Range: 0 - 500
  float FakeInputLagVar; // Range: 0 - 1
  const bool IsPadModuleExclusive;
  const MwFastBuffer<CInputScriptPad*> Script_Pads;
  uint AutoRepeat_InitialDelay;
  uint AutoRepeat_Period;
  const bool StatsDInputOverflowedLastFrame;
  const uint StatsDInputOverflowCount;
  const uint StatsDInputEventsLastFrame;
  const uint StatsDInputEventsCount;
  const uint StatsLatestEvent;
  const uint StatsDInputTimedEventsCount;
  const uint StatsDInputWrongTimestampAhead;
  const float StatsDInputWrongTimestampRatioAhead;
  const uint StatsDInputWrongTimestampLate;
  const float StatsDInputWrongTimestampRatioLate;
  const float StatsDInputWrongTimestampAvgDelta;
  const uint StatsNbMappedInputsReceived;
};

struct CInputPortDx8 : public CInputPort {
  CInputPortDx8();

};

struct CInputPortNull : public CInputPort {
};

struct CInputScriptEvent : public CMwNod {
  enum class CInputScriptEvent::EType {
    PadButtonPress = 0,
  };
  enum class CInputScriptEvent::EButton {
    Left = 0,
    Right = 1,
    Up = 2,
    Down = 3,
    A = 4,
    B = 5,
    X = 6,
    Y = 7,
    L1 = 8,
    R1 = 9,
    LeftStick = 10,
    RightStick = 11,
    Menu = 12,
    View = 13,
    LeftStick_Left = 14,
    LeftStick_Right = 15,
    LeftStick_Up = 16,
    LeftStick_Down = 17,
    RightStick_Left = 18,
    RightStick_Right = 19,
    RightStick_Up = 20,
    RightStick_Down = 21,
    L2 = 22,
    R2 = 23,
    None = 24,
  };
  const CInputScriptEvent::EType Type; // Maniascript
  CInputScriptPad* const Pad; // Maniascript
  const CInputScriptEvent::EButton Button; // Maniascript
  const bool IsAutoRepeat; // Maniascript
  const uint KeyCode; // Maniascript
  const string KeyName; // Maniascript
};

struct CInputBindingsConfig : public CMwNod {
  CInputBindingsConfig();

};

struct CInputDevice : public CMwNod {
  const uint UserData;
  const wstring InstanceName;
  const MwId InstanceId;
  const wstring DeviceModelName;
  const MwId DeviceModelId;
  const bool IsDisabled;
  const bool InputNotAvailable;
  const bool IsUnPlugged;
  const bool MustBePolled;
  const bool CanRumble;
  const uint ObjectCount;
  void ReadHardwareCurState();
};

struct CInputDeviceMouse : public CInputDevice {
  const bool IsInsideWindow;
  const vec2 PosInViewport;
  const vec2 KineticScrollVel;
};

struct CInputDeviceDx8Mouse : public CInputDeviceMouse {
};

struct CInputDeviceDx8Keyboard : public CInputDevice {
};

struct CInputDeviceDx8Pad : public CInputDevice {
  void DevSwitchUserIdInvalid();
};

struct CInputReplay : public CMwNod {
  CInputReplay();

  const uint NbEvents;
};

// Description: "Input devices."
struct CInputScriptManager : public CMwNod {
  enum class CInputScriptManager::EButton {
    Left = 0,
    Right = 1,
    Up = 2,
    Down = 3,
    A = 4,
    B = 5,
    X = 6,
    Y = 7,
    L1 = 8,
    R1 = 9,
    LeftStick = 10,
    RightStick = 11,
    Menu = 12,
    View = 13,
    LeftStick_Left = 14,
    LeftStick_Right = 15,
    LeftStick_Up = 16,
    LeftStick_Down = 17,
    RightStick_Left = 18,
    RightStick_Right = 19,
    RightStick_Up = 20,
    RightStick_Down = 21,
    L2 = 22,
    R2 = 23,
    None = 24,
  };
  enum class CInputScriptManager::EPadType {
    Keyboard = 0,
    Mouse = 1,
    Generic = 2,
    XBox = 3,
    PlayStation = 4,
    Vive = 5,
  };
  const MwFastBuffer<CInputScriptEvent*> PendingEvents; // Maniascript
  const uint Now; // Maniascript
  const uint Period; // Maniascript
  const MwFastBuffer<CInputScriptPad*> Pads; // Maniascript
  const vec2 MousePos; // Maniascript
  const vec2 MouseKineticScrollVel; // Maniascript
  const bool MouseLeftButton; // Maniascript
  const bool MouseRightButton; // Maniascript
  const bool MouseMiddleButton; // Maniascript
  const MwFastBuffer<vec2> TouchPoints_Cur; // Maniascript
  const MwFastBuffer<vec2> TouchPoints_Init; // Maniascript
  wstring GetPadButtonPlaygroundBinding(CInputScriptPad* Pad, CInputScriptManager::EButton Button); // Maniascript
  wstring GetPadButtonCurrentBinding(CInputScriptPad* Pad, CInputScriptManager::EButton Button); // Maniascript
  wstring GetPadButtonBinding(CInputScriptPad* Pad, CInputScriptManager::EButton Button); // Maniascript
  string GetActionBindingRaw(CInputScriptPad* Pad, string ActionMap, string ActionId); // Maniascript
  wstring GetActionBinding(CInputScriptPad* Pad, string ActionMap, string ActionId); // Maniascript
  wstring GetActionDisplayName(string ActionMap, string ActionId); // Maniascript
  const bool JapanStylePadButtons; // Maniascript
  bool ExclusiveMode; // Maniascript
  bool IsKeyPressed(int KeyCode); // Maniascript
  uint Dbg_AutoRepeat_InitialDelay; // Maniascript
  uint Dbg_AutoRepeat_Period; // Maniascript
  CInputScriptPad* const LatestActivePad; // Maniascript
  const CInputScriptManager::EPadType LatestActivePadType; // Maniascript
  const uint TimeSinceLatestActivity; // Maniascript
  const uint TimeSinceLatestMouseActivity; // Maniascript
  const uint TimeSinceLatestTouchActivity; // Maniascript
  const uint TimeSinceLatestKeyboardActivity; // Maniascript
  const uint TimeSinceLatestPadActivity; // Maniascript
};

// Description: "game controller."
struct CInputScriptPad : public CMwNod {
  enum class CInputScriptPad::EButton {
    Left = 0,
    Right = 1,
    Up = 2,
    Down = 3,
    A = 4,
    B = 5,
    X = 6,
    Y = 7,
    L1 = 8,
    R1 = 9,
    LeftStick = 10,
    RightStick = 11,
    Menu = 12,
    View = 13,
    LeftStick_Left = 14,
    LeftStick_Right = 15,
    LeftStick_Up = 16,
    LeftStick_Down = 17,
    RightStick_Left = 18,
    RightStick_Right = 19,
    RightStick_Up = 20,
    RightStick_Down = 21,
    L2 = 22,
    R2 = 23,
    None = 24,
  };
  enum class CInputScriptPad::EPadType {
    Keyboard = 0,
    Mouse = 1,
    Generic = 2,
    XBox = 3,
    PlayStation = 4,
    Vive = 5,
  };
  const int ControllerId; // Maniascript
  const MwId UserId; // Maniascript
  const CInputScriptPad::EPadType Type; // Maniascript
  const wstring ModelName; // Maniascript
  const uint IdleDuration; // Maniascript
  const uint Left; // Maniascript
  const uint Right; // Maniascript
  const uint Up; // Maniascript
  const uint Down; // Maniascript
  const uint A; // Maniascript
  const uint B; // Maniascript
  const uint X; // Maniascript
  const uint Y; // Maniascript
  const uint L1; // Maniascript
  const uint R1; // Maniascript
  const uint LeftStickBut; // Maniascript
  const uint RightStickBut; // Maniascript
  const uint Menu; // Maniascript
  const uint View; // Maniascript
  const float LeftStickX; // Range: -1 - 1 // Maniascript
  const float LeftStickY; // Range: -1 - 1 // Maniascript
  const float RightStickX; // Range: -1 - 1 // Maniascript
  const float RightStickY; // Range: -1 - 1 // Maniascript
  const float L2; // Range: 0 - 1 // Maniascript
  const float R2; // Range: 0 - 1 // Maniascript
  const MwFastBuffer<CInputScriptPad::EButton> ButtonEvents; // Maniascript
  void ClearRumble(); // Maniascript
  void AddRumble(uint Duration, float LargeMotor, float SmallMotor); // Maniascript
  void SetColor(vec3 Color); // Maniascript
};

} // namespace Input

namespace Xml {

// Description: "Tool for parsing document"
struct CXmlScriptParsingManager : public CMwNod {
  CXmlScriptParsingDocumentXml* Create(wstring Contents); // Maniascript
  CXmlScriptParsingDocumentXml* Create2(wstring Contents, bool GenerateText, bool GenerateTextRaw, bool GenerateTextResursive); // Maniascript
  void Destroy(CXmlScriptParsingDocumentXml* Document); // Maniascript
  void Destroy2(CXmlScriptParsingDocumentJson* Document); // Maniascript
  CXmlScriptParsingDocumentXml* Parse_Xml(wstring Contents); // Maniascript
  CXmlScriptParsingDocumentXml* Parse_Xml2(wstring Contents, bool GenerateText, bool GenerateTextRaw, bool GenerateTextResursive); // Maniascript
  CXmlScriptParsingDocumentXml* Parse_Json(wstring Contents); // Maniascript
  CXmlScriptParsingDocumentJson* Parse_JsonEx(wstring Contents); // Maniascript
  void Parse_Destroy(CXmlScriptParsingDocumentXml* Document); // Maniascript
  void Parse_Destroy2(CXmlScriptParsingDocumentJson* Document); // Maniascript
  uint DocumentsSlotsLimit; // Maniascript
  const MwFastBuffer<CXmlScriptParsingDocumentXml*> Documents; // Maniascript
  const MwFastBuffer<CXmlScriptParsingDocumentXml*> DocumentsXml; // Maniascript
  const MwFastBuffer<CXmlScriptParsingDocumentJson*> DocumentsJson; // Maniascript
  MwId Compose_Start_Xml(bool Compact); // Maniascript
  MwId Compose_Start_Json(bool Compact); // Maniascript
  void Compose_End(MwId ComposerId); // Maniascript
  void Compose_Destroy(MwId ComposerId); // Maniascript
  void Compose_Node_Open(MwId ComposerId, wstring Name); // Maniascript
  void Compose_Node_Close(MwId ComposerId); // Maniascript
  void Compose_Array_Open(MwId ComposerId, wstring Name); // Maniascript
  void Compose_Array_Close(MwId ComposerId); // Maniascript
  void Compose_Attribute_Text(MwId ComposerId, wstring Name, wstring Value); // Maniascript
  void Compose_Attribute_Integer(MwId ComposerId, wstring Name, int Value); // Maniascript
  void Compose_Attribute_Real(MwId ComposerId, wstring Name, float Value); // Maniascript
  void Compose_Attribute_Boolean(MwId ComposerId, wstring Name, bool Value); // Maniascript
  void Compose_Value(MwId ComposerId, wstring Name, wstring Value); // Maniascript
  wstring Compose_GetResult(MwId ComposerId); // Maniascript
};

struct CXmlScriptParsingDocumentXml : public CMwNod {
  const string TextContents; // Maniascript
  CXmlScriptParsingNodeXml* const Root; // Maniascript
  const MwFastBuffer<CXmlScriptParsingNodeXml*> Nodes; // Maniascript
  CXmlScriptParsingNodeXml* GetFirstChild(string Name); // Maniascript
};

struct CXmlScriptParsingNodeXml : public CMwNod {
  const wstring Name; // Maniascript
  const wstring TextContents; // Maniascript
  const string TextRawContents; // Maniascript
  const string TextRecursiveContents; // Maniascript
  const MwFastBuffer<CXmlScriptParsingNodeXml*> Children; // Maniascript
  wstring GetAttributeText(wstring Name, wstring DefaultValue); // Maniascript
  int GetAttributeInteger(wstring Name, int DefaultValue); // Maniascript
  float GetAttributeReal(wstring Name, float DefaultValue); // Maniascript
  bool GetAttributeBoolean(wstring Name, bool DefaultValue); // Maniascript
  CXmlScriptParsingNodeXml* GetFirstChild(string Name); // Maniascript
};

struct CXmlScriptParsingDocumentJson : public CMwNod {
  const string TextContents; // Maniascript
  CXmlScriptParsingNodeJson* const Root; // Maniascript
  const MwFastBuffer<CXmlScriptParsingNodeJson*> Nodes; // Maniascript
  CXmlScriptParsingNodeJson* GetFirstChild(string Name); // Maniascript
};

struct CXmlScriptParsingNodeJson : public CMwNod {
  enum class CXmlScriptParsingNodeJson::ENodeType {
    Object = 0,
    Array = 1,
    Value = 2,
  };
  const MwFastBuffer<wstring> Keys; // Maniascript
  const MwFastBuffer<CXmlScriptParsingNodeJson*> Children; // Maniascript
  const wstring AsText; // Maniascript
  const int AsInteger; // Maniascript
  const float AsReal; // Maniascript
  const bool AsBool; // Maniascript
  const CXmlScriptParsingNodeJson::ENodeType NodeType; // Maniascript
  CXmlScriptParsingNodeJson* GetFirstChild(string Name); // Maniascript
};

} // namespace Xml

namespace TrackMania {

// File extension: 'ManiaPlanet.Gbx'
struct CTrackMania : public CGameManiaPlanet {
  CTrackMania();

  void ScanDiskForChallenges();
  void ScanDiskForReplays();
  const MwFastBuffer<CTrackManiaMatchSettings*> MatchSettings;
};

// File extension: 'TrackManiaMenus.Gbx'
struct CTrackManiaMenus : public CGameCtnMenusManiaPlanet {
  CTrackManiaMenus();

  CGameCtnChallenge* const SelectedChallenge;
  uint LastPage;
  const UnnamedEnum Medal;
  const uint CampaignChallengeNumber;
  wstring TestComment;
  uint LastScore;
  uint LastRecord;
  UnnamedEnum NetworkGameMode;
  const wstring NetworkGameModeName;
  uint MedalsCount;
  bool TestDifficulty0;
  bool TestDifficulty1;
  bool TestDifficulty2;
  bool TestDifficulty3;
  bool TestDifficulty4;
  bool TestQuality0;
  bool TestQuality1;
  bool TestQuality2;
  bool TestQuality3;
  bool TestQuality4;
  void OnChooseAvatar1();
  void OnChooseAvatar2();
  void OnChooseAvatar3();
  void OnChooseAvatar4();
  void OnChooseAvatar5();
  void OnChooseAvatar6();
  void OnChooseAvatar7();
  void OnChooseAvatar8();
  void OnChooseProfile1();
  void OnChooseProfile2();
  void OnChooseProfile3();
  void OnChooseProfile4();
  void OnChooseProfile5();
  void OnChooseProfile6();
  void OnChooseProfile7();
  void OnChooseProfile8();
  void MenuChoosePlaylistHotseat();
  void MenuChoosePlaylistNetwork();
  void MenuChoosePlaylist_Init();
  void MenuChoosePlaylist_Clean();
  void MenuChoosePlaylist_OnPlaylistSelected();
  void MenuChoosePlaylist_OnBack();
  void MenuChoosePlaylist_OnRefresh();
  const uint MenuChoosePlaylist_PlaylistsCount;
  void MenuChoosePlaylist_OnOfficialTracks();
  void MenuChoosePlaylist_OnMyTracks();
  void MenuChoosePlaylist_OnDownloadedTracks();
  void MenuChooseChallenge_OnSaveSettings();
  void MenuChooseChallenge_OnSaveSettings_OnYes();
  void MenuChooseChallenge_OnSaveSettings_DoSave();
  void MenuCampaignChallenges();
  void MenuCampaignChallenges_OnBack();
  void MenuCampaignChallenges_OnChallengeCardSelected();
  const wstring MenuCampaignChallenges_Location;
  const uint MenuCampaignChallenges_Level;
  CPlugBitmap* MenuCampaignChallenges_ZoneLogoBitmap;
  void MenuCampaignChallenges_OnChallengeGridPrevPage();
  void MenuCampaignChallenges_OnChallengeGridNextPage();
  void MenuCampaignChallenges_OnChallengeCardRemoved();
  void MenuCampaignChallenges_OnChallengeCardRemovedConfirmed();
  void MenuCampaignChallenges_OnToggleLadder();
  void MenuReplayEditor();
  void MenuMultiPlayerNetworkCreate_ChooseScriptName();
  void MenuMultiPlayerNetworkCreate_OpenDialogScriptName_Yes();
  void MenuMultiPlayer_OnLan();
  void MenuMultiPlayer_OnInternet();
  void MenuMultiLocal_OnBrowseChallenge();
  void MenuMultiLocal_OnBrowseReplay();
  void DialogLowFpsWarn_OnOk();
  void DialogLowFpsWarn_OnCancel();
  bool DialogLowFpsWarn_NeverAskAgain;
  void ShowMenus();
  void ShowDialogs();
  const uint CatalogChapterTotalCoppers;
};

struct CTrackManiaNetwork : public CGameManiaPlanetNetwork {
  CTrackManiaPlayerInfo* PlayerInfo;
  uint SpectatorTimeshift; // Range: 0 - 60000
};

struct CTrackManiaNetworkServerInfo : public CGameCtnNetServerInfo {
  enum class CTrackManiaNetworkServerInfo::EGameMode {
    _GameMode_Script = 0, // |GameMode|Script
    _GameMode_Rounds = 1, // |GameMode|Rounds
    _GameMode_Time_Attack = 2, // |GameMode|Time Attack
    _GameMode_Team = 3, // |GameMode|Team
    _GameMode_Laps = 4, // |GameMode|Laps
    _GameMode_Cup = 5, // |GameMode|Cup
    _GameMode_Stunts = 6, // |GameMode|Stunts
  };
  enum class CTrackManiaNetworkServerInfo::EGameMode_Script {
    Script = 0,
    Rounds = 1,
    TimeAttack = 2,
    Team = 3,
    Laps = 4,
    Cup = 5,
    Stunts = 6,
  };
  const CTrackManiaNetworkServerInfo::EGameMode CurGameMode;
  const CTrackManiaNetworkServerInfo::EGameMode_Script CurGameMode_Script;
  const wstring CurGameModeStr;
  const uint CurRoundPointsLimit;
  const uint CurRoundForcedLaps;
  const bool CurRoundUseNewRules;
  const uint CurRoundPointsLimitNewRules;
  const uint CurTeamPointsLimit;
  const uint CurTeamMaxPoints;
  const bool CurTeamUseNewRules;
  const uint CurTeamPointsLimitNewRules;
  const uint CurTimeAttackLimit;
  const uint CurTimeAttackSynchStartPeriod;
  const uint CurLapsNbLaps;
  const uint CurLapsTimeLimit;
  const uint CurEswcCupPointsLimit;
  const uint CurEswcCupRoundsPerChallenge;
  const uint CurEswcCupNbWinners;
  const uint CurEswcCupWarmUpDuration;
  const uint CurChatTime;
  const uint CurFinishTimeout;
  const uint CurAllWarmUpDuration;
  const bool CurDisableRespawn;
  const uint CurForceMaxOpponents;
  const wstring CurScriptRelName;
  CTrackManiaNetworkServerInfo::EGameMode NextGameMode;
  CTrackManiaNetworkServerInfo::EGameMode_Script NextGameMode_Script;
  uint NextRoundPointsLimit;
  uint NextRoundForcedLaps;
  bool NextRoundUseNewRules;
  uint NextRoundPointsLimitNewRules;
  uint NextTeamPointsLimit;
  uint NextTeamMaxPoints;
  bool NextTeamUseNewRules;
  uint NextTeamPointsLimitNewRules;
  uint NextTimeAttackLimit;
  uint NextTimeAttackSynchStartPeriod;
  uint NextLapsNbLaps;
  uint NextLapsTimeLimit;
  uint NextEswcCupPointsLimit;
  uint NextEswcCupRoundsPerChallenge;
  uint NextEswcCupNbWinners;
  uint NextEswcCupWarmUpDuration;
  uint NextChatTime;
  uint NextFinishTimeout;
  uint NextAllWarmUpDuration;
  bool NextDisableRespawn;
  uint NextForceMaxOpponents;
  wstring NextScriptRelName;
};

struct CTrackManiaPlayerInfo : public CGamePlayerInfo {
};

struct CTrackManiaMatchSettings : public CGameFid {
  wstring Comment;
  bool IsSolo;
  bool IsHotSeat;
  bool IsLan;
  bool IsInternet;
  uint SortIndex;
  bool RandomMapOrder;
  const uint NbChallenges;
  const UnnamedEnum Network_GameMode;
  UnnamedEnum HotSeat_GameMode;
  const uint HotSeat_TimeLimit;
  const uint HotSeat_Rounds;
  const MwFastBuffer<CGameCtnChallenge*> ChallengeInfos;
};

// File extension: '.Frame.Gbx'
struct CTrackManiaControlCheckPointList : public CControlFrame {
  CTrackManiaControlCheckPointList();

  CControlStyle* StyleName;
  CControlStyle* StyleRank;
  CControlStyle* StyleTime;
  CControlFrame* CardModel;
};

// File extension: 'Frame.Gbx'
struct CTrackManiaControlPlayerInfoCard : public CTrackManiaControlCard {
  CTrackManiaControlPlayerInfoCard();

  CPlugBitmap* Avatar;
};

// File extension: 'Frame.Gbx'
struct CTrackManiaControlCard : public CGameControlCard {
  CTrackManiaControlCard();

};

// File extension: 'Frame.Gbx'
struct CTrackManiaControlMatchSettingsCard : public CTrackManiaControlCard {
  CTrackManiaControlMatchSettingsCard();

  const wstring StrName;
  const wstring StrComment;
  const uint ChallengesCount;
  const uint Medals;
  const wstring StrTracks;
  CPlugBitmap* BmpBannerGrey;
  CPlugBitmap* BmpBanner;
};

struct CGamePlayerProfileChunk_TrackManiaSettings : public CGamePlayerProfileChunk {
  CGamePlayerProfileChunk_TrackManiaSettings();

  uint RespawnCount;
  uint RestartCount;
  bool IsShowPlayerGhost;
  bool NadeoGhostsUnlockMessage;
  MwFastBuffer<bool> IsDisplayRaceHelp;
};

} // namespace TrackMania

namespace ShootMania {

// File extension: 'TitleCore.Gbx'
struct CShootMania : public CGameManiaTitleCore {
  CShootMania();

  CScene2d* ArenaInterfaceUIFid;
  CSmArenaResource* ArenaResourcesFid;
  CGamePlaygroundResources* const ArenaWorld;
  CSmServer* const Server;
  const MwFastBuffer<CSmClient*> Clients;
};

struct SSmPlayerVis {
};

// Description: "Ingame Manialink API for ShootMania."
struct CSmArenaInterfaceManialinkScripHandler : public CGameScriptHandlerPlaygroundInterface {
  CSmArenaInterfaceManialinkScripHandler();

  uint ArenaNow; // Maniascript
  CSmScriptPlayer* const InputPlayer; // Maniascript
  CSmScriptPlayer* const GUIPlayer; // Maniascript
  const MwFastBuffer<CSmScriptPlayer*> Players; // Maniascript
  const MwFastBuffer<CGameScriptVehicle*> Vehicles; // Maniascript
  const MwFastBuffer<CSmArenaScore*> Scores; // Maniascript
  const MwFastBuffer<int> ClanScores; // Maniascript
  bool HideResumePlayingButton; // Maniascript
  const MwFastBuffer<CSmScriptMapBase*> MapBases; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks_PlayerSpawn; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks_Gauge; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks_Sector; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks_BotPath; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks_ObjectAnchor; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks_Gate; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks_Foundation; // Maniascript
};

struct CSmArenaClient : public CGamePlaygroundCommon {
  CGameCtnChallenge* const Map;
  CSmArena* const Arena;
  NSmArenaInterface_SMgr* const ArenaInterface;
  float PredictionSmoothConvergenceSpeed;
  NPlugCurve_SSimpleCurveInPlace7 PredictionSmoothFromExtrapolationDuration;
  uint PredictionSimDurationMax;
  float LocalSmoothMin;
  uint LocalSmoothMinInputLag;
  float LocalSmoothMax;
  uint LocalSmoothMaxInputLag;
  float LocalSmoothConvergenceSpeed;
  float LocalSmoothCoefConvergenceSpeed;
  uint LocalSmoothCoefConvergenceDelay;
  const float LocalSmooth;
  const float LocalSmoothCoef;
  float PredictionSmooth;
};

// Description: "Scoring info for ShootMania players."
struct CSmArenaScore : public CGamePlaygroundScore {
  CSmArenaScore();

  uint TeamNum; // Maniascript
  int Points; // Maniascript
  int RoundPoints; // Maniascript
  uint NbRespawnsRequested; // Maniascript
  const uint NbEliminationsInflicted; // Maniascript
  const uint NbEliminationsTaken; // Maniascript
  const uint DamageInflicted; // Maniascript
  const uint DamageTaken; // Maniascript
  MwSArray<uint> BestRaceTimes; // Maniascript
  MwSArray<uint> PrevRaceTimes; // Maniascript
  MwSArray<uint> BestLapTimes; // Maniascript
  MwSArray<uint> PrevLapTimes; // Maniascript
  uint BestRaceNbRespawns; // Maniascript
  uint PrevRaceNbRespawns; // Maniascript
  uint NbEliminationsInflicted_Ed;
  uint NbEliminationsTaken_Ed;
  uint DamageInflicted_Ed;
  uint DamageTaken_Ed;
};

// Description: "An action"
struct CSmActionInstance : public CGameAction {
  CGameActionModel* const Model;
  const uint Now; // Maniascript
  const uint Variant1; // Maniascript
  const uint Variant2; // Maniascript
  const uint Variant3; // Maniascript
  const MwSArray<CSmScriptPlayer*> Players; // Maniascript
  CSmScriptPlayer* const Owner; // Maniascript
  CGameScriptVehicle* const OwnerVehicle; // Maniascript
  const bool IsActive; // Maniascript
  const bool IsBound; // Maniascript
  const bool IsInitialFrame; // Maniascript
  uint Energy; // Maniascript
  uint EnergyMax; // Maniascript
  uint EnergyCost; // Maniascript
  bool EnergyReload; // Maniascript
  float AmmoGain; // Maniascript
  MwId State_EntityId1; // Maniascript
  int State_Integer1; // Maniascript
  int State_Integer2; // Maniascript
  bool State_Boolean1; // Maniascript
  const MwFastBuffer<CSmActionInstanceEvent*> PendingEvents; // Maniascript
  bool IsJumping; // Maniascript
  bool IsGliding; // Maniascript
  bool IsAttractor; // Maniascript
  bool IsFlying; // Maniascript
  bool IsSliding; // Maniascript
  bool IsRunning; // Maniascript
  bool IsFrozen; // Maniascript
  bool IsSneaking; // Maniascript
  bool IsFreeLooking; // Maniascript
  bool HasNoPlayerCollision; // Maniascript
  void SendRulesEvent(wstring Param1, MwFastBuffer<wstring>& Param2, CGameScriptEntity* Shooter, CGameScriptEntity* Victim); // Maniascript
  MwId Anim_GetModelId(wstring ModelName); // Maniascript
  MwId Anim_PlayAtLocation(MwId AnimModelId, vec3 Position, vec3 Direction); // Maniascript
  MwId Anim_PlayOnPlayer(MwId AnimModelId, CSmScriptPlayer* Player); // Maniascript
  void Anim_Stop(MwId AnimId); // Maniascript
  MwId Projectile_GetModelId(wstring ModelName); // Maniascript
  MwId Projectile_CreateAtLocation(MwId ProjectileModelId, CSmScriptPlayer* PlayerToIgnore, vec3 InitialPosition, vec3 InitialDirection, vec3 InitialVelocity); // Maniascript
  MwId Projectile_CreateOnPlayer(MwId ProjectileModelId, CSmScriptPlayer* Shooter); // Maniascript
  uint Cooldown; // Maniascript
  bool Cooldown_IsReady(); // Maniascript
  void Cooldown_Start(); // Maniascript
  MwId Shield_CreateAtLocation(vec3 Position, vec3 Direction); // Maniascript
  MwId Shield_CreateOnPlayer(CSmScriptPlayer* ShieldOwner); // Maniascript
  void Shield_Destroy(MwId ShieldId); // Maniascript
  bool Shield_Exists(MwId ShieldId); // Maniascript
  uint Shield_GetArmor(MwId ShieldId); // Maniascript
  void Shield_SetArmor(MwId ShieldId, uint ShieldArmor); // Maniascript
  bool Shield_GetIsActive(MwId ShieldId); // Maniascript
  void Shield_SetIsActive(MwId ShieldId, bool ShieldIsActive); // Maniascript
  uint Shield_GetArmorMax(MwId ShieldId); // Maniascript
  uint Shield_GetTickReload(MwId ShieldId); // Maniascript
  uint Shield_GetCooldown(MwId ShieldId); // Maniascript
  void Vehicle_TriggerTurbo(); // Maniascript
  void Vehicle_TriggerTurboBrake(); // Maniascript
  bool Focus_Request(); // Maniascript
  void Focus_Release(); // Maniascript
  const bool Focus_Active; // Maniascript
  const bool Trigger0; // Maniascript
  const bool Trigger1; // Maniascript
  const bool Trigger2; // Maniascript
  const bool Trigger3; // Maniascript
  const bool Trigger4; // Maniascript
  const bool Trigger5; // Maniascript
  const bool Trigger6; // Maniascript
  const bool Trigger7; // Maniascript
  const bool Trigger8; // Maniascript
  const bool Trigger9; // Maniascript
  const bool Trigger10; // Maniascript
  const bool Trigger11; // Maniascript
  const bool Trigger12; // Maniascript
  const bool Trigger13; // Maniascript
  int Focused_Scroll; // Maniascript
  int Focused_Scroll_Max; // Maniascript
};

struct CSmNetForm : public CGameNetForm {
  CSmNetForm();

};

struct CSmPlayer : public CGamePlayer {
  CSmPlayer();

  CGameScriptPlayer* ScriptAPI;
  bool UseDelayedVisuals;
  bool TrustClientSimu;
  uint EdClan;
  bool Speaking;
  bool SkippedInputs;
  float LinearHue; // Range: 0 - 1
  vec3 LinearHueSrgb;
  const int SpawnIndex;
  TSceneUId FocusableEntitySceneUId;
  CSmArenaScore* const Score;
  const int StartTime;
  int EndTime;
  uint SpawnableObjectModelIndex;
  bool TrustClientSimu_Client_IsTrustedState;
  uint TrustClientSimu_ServerOverrideCount;
  uint CurrentLaunchedRespawnLandmarkIndex;
  uint CurrentStoppedRespawnLandmarkIndex;
  uint8 Flags;
};

struct CSmArenaInterfaceUI : public CGamePlaygroundInterface {
  CSmArenaInterfaceUI();

};

// Description: "API for ShootMania bots"
struct CSmPlayerDriver : public CMwNod {
  enum class CSmPlayerDriver::ESmDriverBehaviour {
    Static = 0,
    Turret = 1,
    Scripted = 2,
    IA = 3,
    Patrol = 4,
    Escape = 5,
    Saunter = 6,
    Orbit = 7,
    Follow = 8,
  };
  enum class CSmPlayerDriver::ESmDriverPathState {
    Static = 0,
    None = 1,
    Computing = 2,
    Simple = 3,
    Full = 4,
    Incomplete = 5,
    InFlock = 6,
  };
  enum class CSmPlayerDriver::ESmDriverPatrolMode {
    OneTrip = 0,
    BackAndForth = 1,
    Loop = 2,
  };
  enum class CSmPlayerDriver::ESmAttackFilter {
    All = 0,
    AllPlayers = 1,
    AllBots = 2,
    AllOpposite = 3,
    OppositePlayers = 4,
    OppositeBots = 5,
    Nobody = 6,
  };
  CSmPlayerDriver::ESmDriverBehaviour Behaviour; // Maniascript
  bool PursueTarget; // Maniascript
  float AggroRadius; // Maniascript
  float ShootRadius; // Maniascript
  float TargetMinDistance; // Maniascript
  float DisengageDistance; // Maniascript
  float PathSpeedCoef; // Maniascript
  float Accuracy; // Maniascript
  uint ReactionTime; // Maniascript
  uint ShootPeriodMin; // Maniascript
  uint ShootPeriodMax; // Maniascript
  bool RocketAnticipation; // Maniascript
  float Fov; // Maniascript
  float TargetDetectionFov; // Range: 0 - 360 // Maniascript
  CSmPlayerDriver::ESmAttackFilter AttackFilter; // Maniascript
  CSmScriptPlayer* const Target; // Maniascript
  const bool IsStuck; // Maniascript
  bool IsFlying; // Maniascript
  const CSmPlayerDriver::ESmDriverPathState PathState; // Maniascript
  CSmScriptPlayer* const Owner; // Maniascript
  void Scripted_Move(vec3 Goal); // Maniascript
  void Scripted_MoveDelta(vec3 Delta); // Maniascript
  void Scripted_MoveAndAim(vec3 Goal); // Maniascript
  void Scripted_MoveDeltaAndAim(vec3 Delta); // Maniascript
  void Scripted_Aim(vec3 Goal); // Maniascript
  void Scripted_AimDelta(float DeltaYaw, float DeltaPitch); // Maniascript
  void Scripted_RequestAction(); // Maniascript
  void Scripted_RequestGunTrigger(); // Maniascript
  void Scripted_RequestInput(CSmArenaRulesMode::EActionInput Input); // Maniascript
  int PathOffset; // Maniascript
  float Agressivity; // Maniascript
  bool UseOldShootingSystem; // Maniascript
  CSmPlayerDriver::ESmDriverPatrolMode Patrol_Mode; // Maniascript
  void Patrol_Restart();
  CGameScriptMapBotPath* Patrol_Path; // Maniascript
  vec3 Escape_AnchorPoint; // Maniascript
  float Escape_DistanceSafe; // Maniascript
  float Escape_DistanceMinEscape; // Maniascript
  float Escape_DistanceMaxEscape; // Maniascript
  vec3 Saunter_AnchorPoint; // Maniascript
  uint Saunter_BaseChillingTime; // Maniascript
  uint Saunter_ChillingTimeDelta; // Maniascript
  float Saunter_Radius; // Maniascript
  vec3 Orbit_AnchorPoint; // Maniascript
  float Orbit_Radius; // Maniascript
  bool Orbit_IsClockWise; // Maniascript
  bool Scripted_ForceAimInMoveDir; // Maniascript
  vec3 Follow_Offset; // Maniascript
  bool Follow_ForceAimInLeaderDir; // Maniascript
  bool Follow_HasLeader; // Maniascript
  CSmScriptPlayer* Follow_Leader; // Maniascript
  vec3 Follow_AnchorPoint; // Maniascript
  CSmScriptPlayer* ForcedTarget; // Maniascript
  MwFastBuffer<CSmScriptPlayer*> TargetsToAvoid; // Maniascript
  bool PredictJump; // Maniascript
  bool UsePathFinding; // Maniascript
  enum class CSmArenaRulesMode::EActionInput {
    Weapon = 0,
    Secondary = 1,
    Movement = 2,
    QuickAccess1 = 3,
    QuickAccess2 = 4,
    QuickAccess3 = 5,
    QuickAccess4 = 6,
    QuickAccess5 = 7,
    QuickAccess6 = 8,
    QuickAccess7 = 9,
    QuickAccess8 = 10,
    QuickAccess9 = 11,
    QuickAccess0 = 12,
    Consumable1 = 13,
    Consumable2 = 14,
    None = 15,
  };
  CSmArenaRulesMode::EActionInput OnStuckInput; // Maniascript
  float MaxPathDivergence; // Maniascript
  float TargetWorldRadius; // Maniascript
  float TargetWorldPriority; // Range: 0 - 1 // Maniascript
  uint LostTargetChaseDuration; // Maniascript
  MwFastBuffer<CSmPlayerDriver*> FlockPartners; // Maniascript
  float FlockRadius; // Maniascript
  float FlockFov; // Range: 0 - 360 // Maniascript
  float FlockCohesionWeight; // Maniascript
  float FlockAlignmentWeight; // Maniascript
  float FlockSeparationWeight; // Maniascript
};

// Description: "An action event"
struct CSmActionInstanceEvent : public CScriptBaseConstEvent {
  enum class CSmActionInstanceEvent::EType {
    OnHitPlayer = 0,
    OnProjectileEnd = 1,
    OnProjectileDestroyed = 2,
    OnHitObject = 3,
    OnHitShield = 4,
    OnHitVehicle = 5,
    OnShieldEnd = 6,
    OnHitTurret = 7,
    OnInputChange = 8,
    OnFocusedScroll = 9,
  };
  enum class CSmActionInstanceEvent::EInputChange {
    IsActive = 0,
    Focused_Main = 1,
    Focused_Secondary = 2,
    Focused_QuickAccess1 = 3,
    Focused_QuickAccess2 = 4,
    Focused_QuickAccess3 = 5,
    Focused_QuickAccess4 = 6,
    Focused_QuickAccess5 = 7,
    Focused_QuickAccess6 = 8,
    Focused_QuickAccess7 = 9,
    Focused_QuickAccess8 = 10,
    Focused_QuickAccess9 = 11,
    Focused_QuickAccess0 = 12,
    Focused_Consumable1 = 13,
    Focused_Consumable2 = 14,
  };
  const CSmActionInstanceEvent::EType Type; // Maniascript
  const MwId ProjectileModelId; // Maniascript
  const int ProjectileStartDate; // Maniascript
  const int Damage; // Maniascript
  const uint ContextId; // Maniascript
  const vec3 Position; // Maniascript
  const vec3 Direction; // Maniascript
  const vec3 Normal; // Maniascript
  CGameScriptEntity* const Victim; // Maniascript
  const uint Shield; // Maniascript
  const CSmActionInstanceEvent::EInputChange Input; // Maniascript
  const int ScrollDelta; // Maniascript
};

struct CSmArenaRules : public CMwNod {
  enum class CSmArenaRules::EGameplay {
    Default = 0,
    Mp3Beta0 = 1,
  };
  const MwFastBuffer<CSmArenaScore*> Scores;
  CSmArenaRulesMode* RulesMode;
  uint RulesStateStartTime;
  uint RulesStateEndTime;
  uint RulesStateTeam1Score;
  uint RulesStateTeam2Score;
  uint RulesStateTeam3Score;
  uint RulesStateTeam4Score;
  bool FeatureArmorBar;
  bool FeatureUseTeamClans;
  bool FeatureForcedClans;
  bool FeatureUseMultiClans;
  bool FeatureStamina;
  bool FeatureWallJump;
  bool FeatureSameWallJump;
  bool FeatureRocketJump;
  bool FeatureFallDamage;
  bool FeatureGuns;
  bool FeaturePvECollisions;
  bool FeaturePvPCollisions;
  bool FeaturePvPWeapons;
  bool FeatureProtectClanmates;
  bool FeatureLaserVsBullets;
  bool FeatureLaserSkewering;
  bool FeaturePlayerTagging;
  bool FeatureBeaconsWithRecipients;
  bool FeatureInteractiveScreensIn3d;
  bool FeatureGunSpecial;
  bool FeatureAmmoBonusOnHit;
  bool FeatureAllies;
  CSmArenaRules::EGameplay Gameplay;
  bool FeatureCharCanFly;
  bool FeatureLaunchedCheckpoints;
  bool FeatureStunts;
  bool FeatureCanReTriggerLatestWaypoint;
};

struct CSmChallengeParameters : public CGameCtnChallengeParameters {
  CSmChallengeParameters();

  void SetDefaults();
};

// Description: "An instance of an object."
struct CSmObject : public CGameScriptEntity {
  enum class CSmObject::EStatus {
    OnPlayer = 0,
    OnAnchor = 1,
    InWorld = 2,
    Unspawned = 3,
  };
  const CSmObject::EStatus Status; // Maniascript
  const MwId ModelId; // Maniascript
  void SetAnchor(CGameScriptMapObjectAnchor* ObjectAnchor); // Maniascript
  void SetPlayer(CSmScriptPlayer* Player); // Maniascript
  void SetPosition(vec3 Position); // Maniascript
  void SetPositionAndVel(vec3 Position, vec3 Vel); // Maniascript
  void SetUnspawned(); // Maniascript
  CSmScriptPlayer* const Player; // Maniascript
  CSmScriptMapLandmark* const AnchorLandmark; // Maniascript
  const vec3 Position; // Maniascript
  const vec3 Vel; // Maniascript
  uint MachineState; // Maniascript
  bool Throwable; // Maniascript
  const MwId ThrowLatestPlayerId; // Maniascript
};

struct CSmAnalyzer : public CGameAnalyzer {
  const float LocalEntitySmoothCoef;
};

struct CSmArenaPhysics : public CMwNod {
  IScenePhy* const ScenePhy;
};

struct CSmClient : public CMwNod {
  CSmClient();

  CSmArenaClient* const ArenaClient;
  CSmArenaRules* const Rules;
};

struct CSmArenaServer : public CMwNod {
  CSmArena* const Arena;
};

struct CSmArena : public CMwNod {
  CSmArenaPhysics* const ArenaPhysics;
  CSmArenaRules* const Rules;
  CGamePlaygroundResources* const Resources;
  const MwFastBuffer<CSmPlayer*> Players;
  const MwFastBuffer<CSmObject*> Objects;
  const MwFastBuffer<CSmScriptMapBase*> MapBases;
  const MwFastBuffer<CSmScriptMapGate*> MapGates;
  const MwFastBuffer<CSmScriptMapGauge*> MapGauges;
  const MwFastBuffer<CGameScriptMapLandmark*> MapLandmarks;
  const MwFastBuffer<CGameScriptMapSector*> MapSectors;
  const MwFastBuffer<CGameScriptMapSpawn*> MapPlayerSpawns;
  const MwFastBuffer<CGameScriptMapBotPath*> MapBotPaths;
  const MwFastBuffer<CGameScriptMapBotSpawn*> MapBotSpawns;
  const MwFastBuffer<CGameScriptMapObjectAnchor*> MapObjectAnchors;
  const MwFastBuffer<CGameScriptMapVehicleAnchor*> MapVehicleAnchors;
  const MwFastBuffer<CGameScriptMapWaypoint*> MapWaypoints;
  void FakePlayers_AllStatic();
  void FakePlayers_AllPatrol();
  void FakePlayers_AllBackAndForth();
  void FakePlayers_AllOneTrip();
};

// Description: "Rules API for ShootMania gamemodes."
struct CSmArenaRulesMode : public CGamePlaygroundScript {
  enum class CSmArenaRulesMode::ESmScoreSortOrder {
    TotalPoints = 0,
    RoundPoints = 1,
    EliminationsInflicted = 2,
    EliminationsTaken = 3,
    Respawns = 4,
    DamageInflicted = 5,
    DamageTaken = 6,
    BestRace = 7,
    BestLap = 8,
    PrevRace = 9,
    Name = 10,
    LadderRank = 11,
  };
  enum class CSmArenaRulesMode::EWeapon {
    Laser = 0,
    Nucleus = 1,
    Arrow = 2,
    Rocket = 3,
    Missile = 4,
  };
  enum class CSmArenaRulesMode::EActionSlot {
    Slot_A = 0,
    Slot_B = 1,
    Slot_C = 2,
    Slot_D = 3,
    Slot_E = 4,
    Slot_F = 5,
    Slot_G = 6,
    Slot_H = 7,
  };
  enum class CSmArenaRulesMode::EActionInput {
    Weapon = 0,
    Secondary = 1,
    Movement = 2,
    QuickAccess1 = 3,
    QuickAccess2 = 4,
    QuickAccess3 = 5,
    QuickAccess4 = 6,
    QuickAccess5 = 7,
    QuickAccess6 = 8,
    QuickAccess7 = 9,
    QuickAccess8 = 10,
    QuickAccess9 = 11,
    QuickAccess0 = 12,
    Consumable1 = 13,
    Consumable2 = 14,
    None = 15,
  };
  enum class CSmArenaRulesMode::EGameplay {
    Default = 0,
    Mp3Beta0 = 1,
  };
  enum class CSmArenaRulesMode::ERespawnBehaviour {
    Custom = 0,
    DoNothing = 1,
    GiveUpBeforeFirstCheckpoint = 2,
    AlwaysGiveUp = 3,
    AlwaysRespawn = 4,
  };
  enum class CSmArenaRulesMode::ECheckpointBehaviour {
    Custom = 0,
    Default = 1,
    InfiniteLaps = 2,
  };
  enum class CSmArenaRulesMode::EGiveUpBehaviour {
    Custom = 0,
    DoNothing = 1,
    GiveUp = 2,
  };
  enum class CSmArenaRulesMode::EBrowseActionWheel {
    EBrowseActionWheelSet1 = 0,
    EBrowseActionWheelSet2 = 1,
    EBrowseActionWheelSet3 = 2,
    EBrowseActionWheelSet4 = 3,
    EBrowseActionWheelSet5 = 4,
    EBrowseActionWheelSet6 = 5,
    EBrowseActionWheelSet7 = 6,
    EBrowseActionWheelSet8 = 7,
    EBrowseActionWheelSet9 = 8,
    EBrowseActionWheelInc = 9,
    EBrowseActionWheelDec = 10,
    EBrowseActionWheelNone = 11,
  };
  enum class CSmArenaRulesMode::EVehicleTransformType {
    Reset = 0,
    CarSnow = 1,
    CarRally = 2,
    CarDesert = 3,
  };
  uint StartTime; // Maniascript
  uint EndTime; // Maniascript
  uint SpawnInvulnerabilityDuration; // Maniascript
  uint UnspawnAnimDuration; // Maniascript
  uint SpawnDelayDuration; // Maniascript
  CSmArenaRulesMode::ERespawnBehaviour RespawnBehaviour; // Maniascript
  CSmArenaRulesMode::ECheckpointBehaviour CheckpointBehaviour; // Maniascript
  CSmArenaRulesMode::EGiveUpBehaviour GiveUpBehaviour; // Maniascript
  bool GiveUpBehaviour_RespawnAfter; // Maniascript
  bool GiveUpBehaviour_SkipAfterFinishLine; // Maniascript
  uint LapCountOverride; // Maniascript
  bool UseMultiClans; // Maniascript
  bool UseClans; // Maniascript
  bool UseForcedClans; // Maniascript
  bool UsePvECollisions; // Maniascript
  bool UsePvPCollisions; // Maniascript
  bool UseGuns; // Maniascript
  bool UsePvPWeapons; // Maniascript
  bool UseInterractiveScreensIn3d; // Maniascript
  bool UseLaserVsBullets; // Maniascript
  bool UseLaserSkewering; // Maniascript
  bool UsePlayerTagging; // Maniascript
  bool UseBeaconsWithRecipients; // Maniascript
  bool UseAmmoBonusOnHit; // Maniascript
  bool UseSameWallJump; // Maniascript
  bool UseDefaultActionEvents; // Maniascript
  bool UseLaunchedCheckpoints; // Maniascript
  bool ReadonlyLaunchedCheckpoints; // Maniascript
  bool CharCanFly; // Maniascript
  bool TrustClientSimu_FallbackToLatestReceivedState; // Maniascript
  bool UseProtectClanmates; // Maniascript
  bool UseAllies; // Maniascript
  bool UseStunts; // Maniascript
  bool CanReTriggerLatestWaypoint; // Maniascript
  bool DisableDefaultSkinPreload; // Maniascript
  uint GameplayVersion; // Maniascript
  CSmArenaRulesMode::EGameplay Gameplay; // Maniascript
  float Mood_GetTimeOfDay(); // Maniascript
  void Mood_SetTimeOfDay(float Time01); // Maniascript
  uint Mood_GetDayDuration(); // Maniascript
  bool Mood_IsNight(); // Maniascript
  float OffZoneRadius; // Maniascript
  float OffZoneRadiusSpeed; // Maniascript
  MwId OffZoneCenterLandmarkId; // Maniascript
  const uint PlayersNbTotal; // Maniascript
  const uint PlayersNbAlive; // Maniascript
  const uint PlayersNbDead; // Maniascript
  const uint ClansNbTotal; // Maniascript
  const uint ClansNbAlive; // Maniascript
  const uint ClansNbDead; // Maniascript
  const MwFastBuffer<uint> ClansNbPlayers; // Maniascript
  const MwFastBuffer<uint> ClansNbPlayersAlive; // Maniascript
  const MwFastBuffer<CSmScriptPlayer*> Players; // Maniascript
  const MwFastBuffer<CSmScriptPlayer*> BotPlayers; // Maniascript
  const MwFastBuffer<CSmScriptPlayer*> Spectators; // Maniascript
  const MwFastBuffer<CSmScriptPlayer*> AllPlayers; // Maniascript
  const MwFastBuffer<CSmArenaRulesEvent*> PendingEvents; // Maniascript
  CSmScriptPlayer* GetPlayerFromUI(CGamePlaygroundUIConfig* UI); // Maniascript
  CSmScriptPlayer* GetPlayerFromLogin(string Login); // Maniascript
  const MwFastBuffer<CSmScriptMapBase*> MapBases; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks_PlayerSpawn; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks_Gauge; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks_Sector; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks_BotPath; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks_BotSpawn; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks_ObjectAnchor; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks_Gate; // Maniascript
  const MwFastBuffer<CSmScriptMapLandmark*> MapLandmarks_Foundation; // Maniascript
  uint UiScoresPointsLimit; // Maniascript
  const MwFastBuffer<CSmArenaScore*> Scores; // Maniascript
  int Clan1Score; // Maniascript
  int Clan2Score; // Maniascript
  MwSArray<int> ClanScores; // Maniascript
  void Save_Request(wstring FileName); // Maniascript
  void Load_Request(wstring FileName); // Maniascript
  void SetClubLogoUrl(string ClubLogoUrl); // Maniascript
  void SetDecoImageUrl_DecalSponsor4x1(string ImageUrl); // Maniascript
  void SetDecoImageUrl_Screen2x1(string ImageUrl); // Maniascript
  void SetDecoImageUrl_Screen16x9(string ImageUrl); // Maniascript
  void SetDecoImageUrl_Screen8x1(string ImageUrl); // Maniascript
  void SetDecoImageUrl_Screen16x1(string ImageUrl); // Maniascript
  void PassOn(CSmArenaRulesEvent* Event); // Maniascript
  void Discard(CSmArenaRulesEvent* Event); // Maniascript
  void SpawnPlayer(CSmScriptPlayer* Player, int ClanNum, int Armor, CGameScriptMapSpawn* PlayerSpawn, int ActivationDate); // Maniascript
  void SpawnPlayer_InTurret(CSmScriptPlayer* Player, int ClanNum, CGameScriptTurret* Turret, int ActivationDate); // Maniascript
  void SpawnPlayer_InVehicle(CSmScriptPlayer* Player, int ClanNum, CGameScriptVehicle* Vehicle, int ActivationDate); // Maniascript
  void SpawnPlayer_InVehicle2(CSmScriptPlayer* Player, int ClanNum, CGameScriptVehicle* Vehicle, int SlotIndex, int ActivationDate); // Maniascript
  void SpawnBotPlayer_AtPlayerSpawn(CSmScriptPlayer* Player, int ClanNum, int Armor, CGameScriptMapSpawn* PlayerSpawn, int ActivationDate); // Maniascript
  void SpawnBotPlayer_AtBotPath(CSmScriptPlayer* Player, int ClanNum, int Armor, CGameScriptMapBotPath* BotPath, int ActivationDate); // Maniascript
  void SpawnBotPlayer_AtBotSpawn(CSmScriptPlayer* Player, int ClanNum, int Armor, CGameScriptMapBotSpawn* BotSpawn, int ActivationDate); // Maniascript
  void SpawnBotPlayer_FromOwner(CSmScriptPlayer* Player, CSmScriptPlayer* Owner, int Armor, vec3 Offset, int ActivationDate); // Maniascript
  void UnspawnPlayer(CSmScriptPlayer* Player); // Maniascript
  const CSmArenaRulesMode::ESmScoreSortOrder Scores_SortCriteria; // Maniascript
  bool Scores_AutoUpdateLadderRank; // Maniascript
  bool Scores_AutoUploadPersonalBests; // Maniascript
  void Scores_SetSortCriteria(CSmArenaRulesMode::ESmScoreSortOrder Criteria); // Maniascript
  void Scores_ComputeLadderRank(CSmArenaRulesMode::ESmScoreSortOrder Criteria); // Maniascript
  void Scores_ClearAll(); // Maniascript
  void Score_ClearPrevRace(CSmArenaScore* Score); // Maniascript
  void Score_ClearPrevLap(CSmArenaScore* Score); // Maniascript
  void Score_Clear(CSmArenaScore* Score); // Maniascript
  void ClearScores(); // Maniascript
  void SetPlayerClan(CSmScriptPlayer* Player, int ClanNum); // Maniascript
  void SetPlayerWeapon(CSmScriptPlayer* Player, CSmArenaRulesMode::EWeapon DefaultWeapon, bool AutoSwitchWeapon); // Maniascript
  void SetPlayerWeaponAvailable(CSmScriptPlayer* Player, CSmArenaRulesMode::EWeapon Weapon, bool Available); // Maniascript
  void SetPlayerAllWeaponAvailable(CSmScriptPlayer* Player, bool Available); // Maniascript
  void SetPlayerReloadAllWeapons(CSmScriptPlayer* Player, bool ReloadAllWeapons); // Maniascript
  void SetPlayerAmmo(CSmScriptPlayer* Player, CSmArenaRulesMode::EWeapon Weapon, int Count); // Maniascript
  int GetPlayerAmmo(CSmScriptPlayer* Player, CSmArenaRulesMode::EWeapon Weapon); // Maniascript
  void AddPlayerAmmo(CSmScriptPlayer* Player, CSmArenaRulesMode::EWeapon Weapon, float DeltaCount); // Maniascript
  void SetPlayerAmmoMax(CSmScriptPlayer* Player, CSmArenaRulesMode::EWeapon Weapon, int Count); // Maniascript
  int GetPlayerAmmoMax(CSmScriptPlayer* Player, CSmArenaRulesMode::EWeapon Weapon); // Maniascript
  int GetWeaponIndex(CSmArenaRulesMode::EWeapon Weapon); // Maniascript
  int GetWeaponNum(CSmArenaRulesMode::EWeapon Weapon); // Maniascript
  bool CanRespawnPlayer(CSmScriptPlayer* Player); // Maniascript
  void RespawnPlayer(CSmScriptPlayer* Player); // Maniascript
  void SetNbBotPlayers(int NbClan1, int NbClan2); // Maniascript
  void SetNbFakePlayers(int NbClan1, int NbClan2); // Maniascript
  CSmScriptPlayer* CreateBotPlayer(MwId ModelId, int ClanNum); // Maniascript
  void DestroyBotPlayer(CSmScriptPlayer* BotPlayer); // Maniascript
  void DestroyAllBotPlayers(); // Maniascript
  bool Bots_AutoSpawn; // Maniascript
  bool Bots_AutoRespawn; // Maniascript
  bool Bots_AutoIgnoreBotEvents; // Maniascript
  bool ForceNavMapsComputation; // Maniascript
  bool DelayedVisuals_Enabled; // Maniascript
  bool CrudeExtrapolation_AllowDelay; // Maniascript
  void ScriptedBot_Move(CSmScriptPlayer* BotPlayer, vec3 Goal); // Maniascript
  void ScriptedBot_MoveDelta(CSmScriptPlayer* BotPlayer, vec3 Delta); // Maniascript
  void ScriptedBot_MoveAndAim(CSmScriptPlayer* BotPlayer, vec3 Goal); // Maniascript
  void ScriptedBot_MoveDeltaAndAim(CSmScriptPlayer* BotPlayer, vec3 Delta); // Maniascript
  void ScriptedBot_Aim(CSmScriptPlayer* BotPlayer, vec3 Goal); // Maniascript
  void ScriptedBot_AimDelta(CSmScriptPlayer* BotPlayer, float DeltaYaw, float DeltaPitch); // Maniascript
  void ScriptedBot_RequestAction(CSmScriptPlayer* BotPlayer); // Maniascript
  void ScriptedBot_RequestGunTrigger(CSmScriptPlayer* BotPlayer); // Maniascript
  MwId ActionGetModelId(wstring ActionName); // Maniascript
  void ActionLoad(CSmScriptPlayer* Player, CSmArenaRulesMode::EActionSlot ActionSlot, MwId ModelId); // Maniascript
  void ActionLoad_MeleeMode(CSmScriptPlayer* Player, CSmArenaRulesMode::EActionSlot ActionSlot, MwId ModelId, bool MeleeMode); // Maniascript
  void ActionLoad_Vehicle(CGameScriptVehicle* Vehicle, int VehicleSlotIndex, CSmArenaRulesMode::EActionSlot ActionSlot, MwId ModelId); // Maniascript
  void ActionLoad_Turret(CGameScriptTurret* Turret, CSmArenaRulesMode::EActionSlot ActionSlot, MwId ModelId); // Maniascript
  uint Action_GetCooldown(CSmScriptPlayer* Player, CSmArenaRulesMode::EActionInput ActionInput); // Maniascript
  uint Action_GetRemainingCooldown(CSmScriptPlayer* Player, CSmArenaRulesMode::EActionInput ActionInput); // Maniascript
  void ActionBind(CSmScriptPlayer* Player, CSmArenaRulesMode::EActionSlot ActionSlot, CSmArenaRulesMode::EActionInput ActionInput); // Maniascript
  void ActionUnBind(CSmScriptPlayer* Player, CSmArenaRulesMode::EActionInput ActionInput); // Maniascript
  void ActionSetVariant1(CSmScriptPlayer* Player, CSmArenaRulesMode::EActionSlot ActionSlot, uint ActionVariant); // Maniascript
  void ActionSetVariant2(CSmScriptPlayer* Player, CSmArenaRulesMode::EActionSlot ActionSlot, uint ActionVariant); // Maniascript
  void ActionSetVariant3(CSmScriptPlayer* Player, CSmArenaRulesMode::EActionSlot ActionSlot, uint ActionVariant); // Maniascript
  void LoadAction(CSmScriptPlayer* Player, CSmArenaRulesMode::EActionSlot ActionSlot, MwId ModelId, uint ActionVariant); // Maniascript
  void BindAction(CSmScriptPlayer* Player, CSmArenaRulesMode::EActionSlot ActionSlot, CSmArenaRulesMode::EActionInput ActionInput); // Maniascript
  const MwFastBuffer<CSmObject*> Objects; // Maniascript
  CSmObject* ObjectCreate(MwId ModelId); // Maniascript
  void ObjectDestroy(CSmObject* Object); // Maniascript
  void ObjectDestroyAll(); // Maniascript
  void SetSpawnableObjectModel(CSmScriptPlayer* Player, wstring ObjectModelName, float StaminaCoef); // Maniascript
  wstring GetObjectStringId(CSmObject* Object); // Maniascript
  wstring GetObjectModelName(CSmObject* Object); // Maniascript
  void RemoveShieldArmor(uint VictimShieldId, uint Damage); // Maniascript
  void Replay_SaveAttackScore(CSmScriptPlayer* Player, int Score); // Maniascript
  void Replay_SaveDefenseScore(CSmScriptPlayer* Player, int Score); // Maniascript
  void Replay_SaveTeamScore(uint Team, int Score); // Maniascript
  void Replay_SavePlayerOfInterest(CSmScriptPlayer* Player); // Maniascript
  void Replay_SaveWinner(CSmScriptPlayer* Player); // Maniascript
  void Replay_SaveInterface(); // Maniascript
  void Player_BeginNewRace(CSmScriptPlayer* Player, uint AbsoluteTime); // Maniascript
  void Player_BeginNewLap(CSmScriptPlayer* Player, uint RaceTime); // Maniascript
  void Player_RemoveLastWaypointTime(CSmScriptPlayer* Player); // Maniascript
  void Player_AddWaypointTime(CSmScriptPlayer* Player, uint Time, CSmScriptMapLandmark* Landmark); // Maniascript
  void Player_SetFinishTime(CSmScriptPlayer* Player, uint Time, CSmScriptMapLandmark* Landmark); // Maniascript
  void Player_ClearRaceWaypointTimes(CSmScriptPlayer* Player); // Maniascript
  void Player_ClearLapWaypointTimes(CSmScriptPlayer* Player); // Maniascript
  void Player_ClearCurrentLapNumber(CSmScriptPlayer* Player); // Maniascript
  void Player_SetCurRaceAsBestRace(CSmScriptPlayer* Player); // Maniascript
  void Player_SetCurRaceAsPrevRace(CSmScriptPlayer* Player); // Maniascript
  void Player_SetCurLapAsBestLap(CSmScriptPlayer* Player); // Maniascript
  void Player_SetCurLapAsPrevLap(CSmScriptPlayer* Player); // Maniascript
  void Player_SetPlayerCurRaceAsScoreBestRace(CSmScriptPlayer* Player); // Maniascript
  void Player_SetPlayerCurRaceAsScorePrevRace(CSmScriptPlayer* Player); // Maniascript
  void Player_SetPlayerCurLapAsScoreBestLap(CSmScriptPlayer* Player); // Maniascript
  void Player_SetPlayerCurLapAsScorePrevLap(CSmScriptPlayer* Player); // Maniascript
  void Player_SetPlayerPrevLapAsScoreBestLap(CSmScriptPlayer* Player); // Maniascript
  void Player_SetPlayerPrevLapAsScorePrevLap(CSmScriptPlayer* Player); // Maniascript
  CGameGhostMgrScript* const GhostMgr; // Maniascript
  void Ghosts_SetStartTime(int StartTime); // Maniascript
  void Ghosts_SetMaxAlpha(float MaxAlpha); // Maniascript
  CGameGhostScript* Ghost_RetrieveFromPlayer(CSmScriptPlayer* Player); // Maniascript
  CGameGhostScript* Ghost_RetrieveFromPlayer2(CSmScriptPlayer* Player, bool TruncateLaunchedCheckpointsRespawns); // Maniascript
  void Ghost_CopyToScoreBestRaceAndLap(CGameGhostScript* Ghost, CSmArenaScore* Score); // Maniascript
  MwId Ghost_Add(CGameGhostScript* Ghost, bool IsGhostLayer); // Maniascript
  MwId Ghost_Add_Deprecated(CGameGhostScript* Ghost); // Maniascript
  MwId Ghost_AddWithOffset(CGameGhostScript* Ghost, bool IsGhostLayer, int Offset); // Maniascript
  bool Ghost_IsReplayOver(MwId GhostInstanceId); // Maniascript
  bool Ghost_IsVisible(MwId GhostInstanceId); // Maniascript
  vec3 Ghost_GetPosition(MwId GhostInstanceId); // Maniascript
  void Ghost_Remove(MwId GhostInstanceId); // Maniascript
  void Ghost_RemoveAll(); // Maniascript
  void GhostRecorder_RecordingStart(CSmScriptPlayer* Player); // Maniascript
  void GhostRecorder_RecordingStartAtTime(CSmScriptPlayer* Player, int StartTime); // Maniascript
  CGameGhostScript* GhostRecorder_RecordingEnd(CSmScriptPlayer* Player); // Maniascript
  void GhostRecorder_SetEnabled(CSmScriptPlayer* Player, bool Value); // Maniascript
  void GhostRecorder_Ghosts_Select(CSmScriptPlayer* Player); // Maniascript
  void GhostRecorder_Clear(CSmScriptPlayer* Player); // Maniascript
  const MwFastBuffer<CGameGhostScript*> GhostRecorder_Ghosts; // Maniascript
  CSmScriptMapLandmark* GhostDriver_Playlist_GetNextSpawn(CGameScriptPlayer* Player); // Maniascript
};

struct CSmServer : public CMwNod {
  CSmServer();

  CSmArenaServer* const ArenaServer;
  CSmArenaRules* const Rules;
  uint ClientInputsMaxLatency;
  uint DbgMinInputDelay;
  uint DbgDelaySendSnapshots;
};

// Description: "Event recieved by ShootMania gamemodes."
struct CSmArenaRulesEvent : public CScriptBaseEvent {
  enum class CSmArenaRulesEvent::EType {
    Unknown = 0,
    OnShoot = 1,
    OnHit = 2,
    OnNearMiss = 3,
    OnArmorEmpty = 4,
    OnCapture = 5,
    OnShotDeny = 6,
    OnFallDamage = 7,
    OnCommand = 8,
    OnPlayerAdded = 9,
    OnPlayerRemoved = 10,
    OnPlayerRequestRespawn = 11,
    OnActionCustomEvent = 12,
    OnActionEvent = 13,
    OnPlayerTouchesObject = 14,
    OnPlayerThrowsObject = 15,
    OnPlayerTriggersSector = 16,
    OnPlayerTriggersWaypoint = 17,
    OnPlayerRequestActionChange = 18,
    OnVehicleArmorEmpty = 19,
    OnVehicleCollision = 20,
    OnVehicleVsVehicleCollision = 21,
    OnPlayerRequestItemInteraction = 22,
    OnStuntFigure = 23,
  };
  enum class CSmArenaRulesEvent::EActionSlot {
    Slot_A = 0,
    Slot_B = 1,
    Slot_C = 2,
    Slot_D = 3,
    Slot_E = 4,
    Slot_F = 5,
    Slot_G = 6,
    Slot_H = 7,
  };
  enum class CSmArenaRulesEvent::EActionInput {
    Weapon = 0,
    Secondary = 1,
    Movement = 2,
    QuickAccess1 = 3,
    QuickAccess2 = 4,
    QuickAccess3 = 5,
    QuickAccess4 = 6,
    QuickAccess5 = 7,
    QuickAccess6 = 8,
    QuickAccess7 = 9,
    QuickAccess8 = 10,
    QuickAccess9 = 11,
    QuickAccess0 = 12,
    Consumable1 = 13,
    Consumable2 = 14,
    None = 15,
  };
  const CSmArenaRulesEvent::EType Type; // Maniascript
  CSmScriptPlayer* const Player; // Maniascript
  int Damage; // Maniascript
  const uint VictimShield; // Maniascript
  CGameScriptEntity* const VictimEntity; // Maniascript
  int ShooterPoints; // Maniascript
  CGameScriptEntity* const ShooterEntity; // Maniascript
  const uint ShooterClan; // Maniascript
  const float Height; // Maniascript
  const uint UserData; // Maniascript
  const vec3 ItemPosition; // Maniascript
  const float MissDist; // Maniascript
  const uint WeaponNum; // Maniascript
  const bool ShooterUsedAction; // Maniascript
  const uint ShooterWeaponNum; // Maniascript
  const CSmArenaRulesEvent::EActionSlot ShooterActionSlot; // Maniascript
  const wstring ShooterActionId; // Maniascript
  const bool VictimUsedAction; // Maniascript
  const uint VictimWeaponNum; // Maniascript
  const CSmArenaRulesEvent::EActionSlot VictimActionSlot; // Maniascript
  const wstring VictimActionId; // Maniascript
  const CSmArenaRulesEvent::EActionSlot ActionSlot; // Maniascript
  const CSmArenaRulesEvent::EActionInput ActionInput; // Maniascript
  const wstring ActionId; // Maniascript
  const wstring Param1; // Maniascript
  const MwFastBuffer<wstring> Param2; // Maniascript
  CSmObject* const Object; // Maniascript
  const uint WaypointTime; // Maniascript
  const uint WaypointLapTime; // Maniascript
  const bool IsFinish; // Maniascript
  const bool IsNewLap; // Maniascript
  CSmScriptMapLandmark* const Landmark; // Maniascript
  CGamePlayerInfo* const User; // Maniascript
  const bool PlayerWasSpawned; // Maniascript
  const bool PlayerWasDisconnected; // Maniascript
  const bool PlayerWasInLadderMatch; // Maniascript
  const vec3 PlayerLastPosition; // Maniascript
  const vec3 PlayerLastAimDirection; // Maniascript
  const bool GiveUp; // Maniascript
  const bool RegressRespawn; // Maniascript
  const string CommandName; // Maniascript
  const bool CommandValueBoolean; // Maniascript
  const int CommandValueInteger; // Maniascript
  const float CommandValueReal; // Maniascript
  const wstring CommandValueText; // Maniascript
  const vec2 CommandValueVec2; // Maniascript
  const vec3 CommandValueVec3; // Maniascript
  const int2 CommandValueInt2; // Maniascript
  const int3 CommandValueInt3; // Maniascript
  const int ActionChange; // Maniascript
  NSceneVehiclePhy_SStuntFigure* const StuntFigure; // Maniascript
};

struct CSmEditorPluginMapType : public CGameEditorPluginMapMapType {
  CSmArenaRulesMode* const Mode; // Maniascript
  void AuthorTime_SetFromGhost(CGameGhostScript* Ghost); // Maniascript
  void SetAuthorTimeAndGhost(CGameGhostScript* Ghost); // Maniascript
  const MwFastBuffer<CSmScriptPlayer*> AllPlayers; // Maniascript
  const MwFastBuffer<CSmScriptPlayer*> Players; // Maniascript
};

// File extension: 'SmArenaResource.Gbx'
struct CSmArenaResource : public CMwNod {
  CSmArenaResource();

  CScene2d* ScreenIn3d;
  CPlugBitmap* BitmapCoolDownCutOff;
  CPlugBitmap* BitmapConsumablesIcon_Off;
  CPlugShader* ShaderCrossHairHueShift;
  CPlugGameSkin* GameSkinCrosshairs;
  CPlugSound* SoundOffZone;
  CPlugSound* SoundTeleport;
  CPlugSound* UISound_StartMatch;
  CPlugSound* UISound_EndMatch;
  CPlugSound* UISound_StartRound;
  CPlugSound* UISound_EndRound;
  CPlugSound* UISound_PhaseChange;
  CPlugSound* UISound_TieBreakPoint;
  CPlugSound* UISound_TiePoint;
  CPlugSound* UISound_VictoryPoint;
  CPlugSound* UISound_Capture;
  CPlugSound* UISound_TimeOut;
  CPlugSound* UISound_Notice;
  CPlugSound* UISound_Warning;
  CPlugSound* UISound_PlayerEliminated;
  CPlugSound* UISound_PlayerHit;
  CPlugSound* UISound_Checkpoint;
  CPlugSound* UISound_Finish;
  CPlugSound* UISound_Record;
  CPlugSound* UISound_ScoreProgress;
  CPlugSound* UISound_RankChange;
  CPlugSound* UISound_Bonus;
  CPlugSound* UISound_FirstHit;
  CPlugSound* UISound_Combo;
  CPlugSound* UISound_PlayersRemaining;
  CPlugSound* UISound_Custom1;
  CPlugSound* UISound_Custom2;
  CPlugSound* UISound_Custom3;
  CPlugSound* UISound_Custom4;
  CPlugSound* SoundUISplashVersus;
  CPlugSound* SoundUIRetractableShow;
  CPlugSound* SoundUIRetractableHide;
  CPlugSound* SoundUITimeLeftWarning;
  CPlugSound* SoundUITimeTick;
  CPlugSound* SoundUITimeOut;
  CPlugSound* SoundUIGaugeHealth;
  CPlugSound* SoundUIGaugeStamina;
  CPlugSound* SoundUIGaugeEnergy;
  CControlFrame* CardPlayerScore_Small;
  CControlFrame* CardPlayerScore_Small_Rev;
  CControlFrame* CardPlayerScore_Big;
  CControlFrame* CardPlayerScore_Big_Rev;
  CControlFrame* CardPlayerScore_Medium;
  CControlFrame* CardPlayerScore_Medium_Rev;
  CFuncEnum* GameplayIcons;
  CPlugParticleEmitterModel* LaserSight_EmitterModel_deprecated;
  CGameActionFxResources* ActionFxResources;
  CPlugParticleMaterialImpactModel* Destructibles_MaterialImpactModel;
  CPlugDestructibleFx* DestructiblesFx;
  CMwRefBuffer* RefBuffer;
  float AdditionalListenerDistanceNbSquares;
};

struct CSmModuleScoresTable : public CGamePlaygroundModuleClientScoresTable {
};

struct CSmScriptMapBase : public CMwNod {
  uint Clan; // Maniascript
  bool IsActive; // Maniascript
  const uint NumberOfCollectors; // Maniascript
};

struct CSmScriptMapGate : public CMwNod {
  uint Clan; // Maniascript
  bool Automatic; // Maniascript
  bool ManualClosed; // Maniascript
  const bool AutoClosed; // Maniascript
  bool AutoIsActive; // Maniascript
  int AutoCloseDelay; // Maniascript
  int AutoOpenSpeed; // Maniascript
};

struct CSmScriptMapGauge : public CMwNod {
  uint Clan; // Maniascript
  float ValueReal; // Range: 0 - 1 // Maniascript
  uint Value; // Maniascript
  uint Max; // Maniascript
  int Speed; // Maniascript
  bool Captured; // Maniascript
};

struct CSmScriptMapLandmark : public CGameScriptMapLandmark {
  CSmScriptMapBase* const Base; // Maniascript
  CSmScriptMapGate* const Gate; // Maniascript
  CSmScriptMapGauge* const Gauge; // Maniascript
};

// Description: "A Shootmania player."
struct CSmScriptPlayer : public CGameScriptPlayer {
  enum class CSmScriptPlayer::ESpawnStatus {
    NotSpawned = 0,
    Spawning = 1,
    Spawned = 2,
  };
  enum class CSmScriptPlayer::EPost {
    Char = 0,
    None = 1,
    CarDriver = 2,
    Reserved1 = 3,
    Reserved2 = 4,
    Reserved3 = 5,
    Reserved4 = 6,
    Reserved5 = 7,
    Reserved6 = 8,
    Reserved7 = 9,
    Reserved8 = 10,
    Reserverd9 = 11,
    Reserved10 = 12,
    Reserved11 = 13,
    Reserved12 = 14,
    Reserved13 = 15,
  };
  CSmArenaScore* const Score; // Maniascript
  const CSmScriptPlayer::ESpawnStatus SpawnStatus; // Maniascript
  const int StartTime; // Maniascript
  int EndTime; // Maniascript
  const MwFastBuffer<uint> RaceWaypointTimes; // Maniascript
  const MwFastBuffer<uint> LapWaypointTimes; // Maniascript
  const MwFastBuffer<uint> CurrentLapWaypointTimes; // Maniascript
  const MwFastBuffer<uint> PreviousLapWaypointTimes; // Maniascript
  const uint CurrentLapNumber; // Maniascript
  const int CurrentRaceTime; // Maniascript
  const int CurrentLapTime; // Maniascript
  const int LapStartTime; // Maniascript
  const uint CurrentRaceRespawns; // Maniascript
  const uint CurrentLapRespawns; // Maniascript
  const CSmScriptPlayer::EPost Post; // Maniascript
  float AmmoGain; // Range: 0 - 10 // Maniascript
  float AmmoPower; // Range: 0.1 - 10 // Maniascript
  const bool AutoSwitchWeapon; // Maniascript
  const uint CurWeapon; // Maniascript
  const uint CurAmmo; // Maniascript
  const uint CurAmmoMax; // Maniascript
  const uint CurAmmoUnit; // Maniascript
  uint ActionWheelSelectedSlotIndex; // Maniascript
  uint Armor; // Maniascript
  uint ArmorMax; // Maniascript
  uint ArmorGain; // Maniascript
  uint ArmorReplenishGain; // Maniascript
  float ArmorPower; // Range: 0.1 - 10 // Maniascript
  uint Stamina; // Maniascript
  const uint StaminaMaxValue; // Maniascript
  float StaminaMax; // Range: 0.1 - 3 // Maniascript
  float StaminaGain; // Range: 0 - 1 // Maniascript
  float StaminaPower; // Range: 0.1 - 1 // Maniascript
  float Energy; // Maniascript
  float SpeedPower; // Range: 0.1 - 1 // Maniascript
  float JumpPower; // Range: 0.1 - 1 // Maniascript
  bool AllowWallJump; // Maniascript
  bool AllowProgressiveJump; // Maniascript
  bool UseAlternateWeaponVisual; // Maniascript
  MwId ForceModelId; // Maniascript
  bool IsHighlighted; // Maniascript
  float EnergyLevel; // Range: 0 - 1 // Maniascript
  vec3 ForceColor; // Maniascript
  string Dossard; // Maniascript
  string Dossard_Number; // Maniascript
  string Dossard_Trigram; // Maniascript
  vec3 Dossard_Color; // Maniascript
  const float GetLinearHue; // Range: 0 - 1 // Maniascript
  float ForceLinearHue; // Maniascript
  bool ForceLightTrail; // Maniascript
  bool HasShield; // Maniascript
  const bool IsInVehicle; // Maniascript
  bool IsStuck; // Maniascript
  float ThrowSpeed; // Range: 0 - 500 // Maniascript
  const int CurrentClan; // Maniascript
  const uint IdleDuration; // Maniascript
  const bool IsEntityStateAvailable; // Maniascript
  const vec3 Position; // Maniascript
  const float AimYaw; // Maniascript
  const float AimPitch; // Maniascript
  const float AimRoll; // Maniascript
  const vec3 AimDirection; // Maniascript
  const vec3 UpDirection; // Maniascript
  const vec3 LeftDirection; // Maniascript
  const vec3 Velocity; // Maniascript
  const float Speed; // Maniascript
  const bool IsUnderground; // Maniascript
  const bool IsTouchingGround; // Maniascript
  const bool IsInAir; // Maniascript
  const bool IsInWater; // Maniascript
  const bool IsInOffZone; // Maniascript
  const bool IsOnTech; // Maniascript
  const bool IsOnTechGround; // Maniascript
  const bool IsOnTechLaser; // Maniascript
  const bool IsOnTechArrow; // Maniascript
  const bool IsOnTechNucleus; // Maniascript
  const bool IsOnTechArmor; // Maniascript
  const bool IsOnTechSafeZone; // Maniascript
  const bool IsOnTechTeleport; // Maniascript
  const bool IsOnTechNoWeapon; // Maniascript
  const bool IsPowerJumpActivable; // Maniascript
  const bool IsTeleportActivable; // Maniascript
  const bool IsAttractorActivable; // Maniascript
  const uint NbActiveAttractors; // Maniascript
  const bool IsRunning; // Maniascript
  const bool IsCapturing; // Maniascript
  CSmScriptMapLandmark* const CapturedLandmark; // Maniascript
  const MwFastBuffer<CSmObject*> Objects; // Maniascript
  CGameScriptVehicle* const Vehicle; // Maniascript
  const bool IsFakePlayer; // Maniascript
  const bool IsBot; // Maniascript
  bool UseDelayedVisuals; // Maniascript
  bool UseCrudeExtrapolation; // Maniascript
  bool TrustClientSimu; // Maniascript
  CSmPlayerDriver* const Driver; // Maniascript
  const float AccelCoef; // Range: 0 - 1 // Maniascript
  const float ControlCoef; // Range: 0 - 1 // Maniascript
  const float GravityCoef; // Range: 0 - 1 // Maniascript
  const float AdherenceCoef; // Range: 0 - 1 // Maniascript
  const float Upwardness; // Maniascript
  const float Distance; // Maniascript
  const uint DisplaySpeed; // Maniascript
  const float InputSteer; // Maniascript
  const float InputGasPedal; // Maniascript
  const bool InputIsBraking; // Maniascript
  const float EngineRpm; // Maniascript
  const int EngineCurGear; // Maniascript
  const float EngineTurboRatio; // Maniascript
  const uint WheelsContactCount; // Maniascript
  const uint WheelsSkiddingCount; // Maniascript
  const uint FlyingDuration; // Maniascript
  const uint SkiddingDuration; // Maniascript
  const float SkiddingDistance; // Maniascript
  const float FlyingDistance; // Maniascript
  NSceneVehiclePhy_SStuntStatus* const Stunt; // Maniascript
  const uint HandicapNoGasDuration; // Maniascript
  const uint HandicapForceGasDuration; // Maniascript
  const uint HandicapNoBrakesDuration; // Maniascript
  const uint HandicapNoSteeringDuration; // Maniascript
  const uint HandicapNoGripDuration; // Maniascript
};

struct CSmModuleManager : public CGamePlaygroundModuleManagerClient {
};

struct CSmOffZone : public CMwNod {
};

struct CSmActionMgr : public CGameMgrAction {
  CSmPlayer* Ed_Player;
  CSmActionInstance* const WeaponAction;
};

struct CSmNetFormBroadcastable : public CGameNetForm {
  CSmNetFormBroadcastable();

};

struct CSmArenaInterfaceManialinkScriptHandler_ReadOnly : public CGameScriptHandlerPlaygroundInterface_ReadOnly {
  uint ArenaNow; // Maniascript
  CSmScriptPlayer* const InputPlayer; // Maniascript
  CSmScriptPlayer* const GUIPlayer; // Maniascript
  const MwSArray<CSmScriptPlayer*> Players; // Maniascript
  const MwSArray<CGameScriptVehicle*> Vehicles; // Maniascript
  const MwSArray<CSmArenaScore*> Scores; // Maniascript
  const MwSArray<CSmScriptMapBase*> MapBases; // Maniascript
  const MwSArray<CSmScriptMapLandmark*> MapLandmarks; // Maniascript
  const MwSArray<CSmScriptMapLandmark*> MapLandmarks_PlayerSpawn; // Maniascript
  const MwSArray<CSmScriptMapLandmark*> MapLandmarks_Gauge; // Maniascript
  const MwSArray<CSmScriptMapLandmark*> MapLandmarks_Sector; // Maniascript
  const MwSArray<CSmScriptMapLandmark*> MapLandmarks_BotPath; // Maniascript
  const MwSArray<CSmScriptMapLandmark*> MapLandmarks_ObjectAnchor; // Maniascript
  const MwSArray<CSmScriptMapLandmark*> MapLandmarks_Gate; // Maniascript
  const MwSArray<CSmScriptMapLandmark*> MapLandmarks_Foundation; // Maniascript
};

} // namespace ShootMania

namespace GameData {

struct CGameCtnCollector : public CMwNod {
  CGameCtnCollector();

  enum class CGameCtnCollector::EProdState {
    Aborted = 0,
    GameBox = 1,
    DevBuild = 2,
    Release = 3,
  };
  UnnamedEnum CollectionId;
  wstring CollectionId_Text;
  MwId Author;
  wstring NameE;
  const wstring Name; // Maniascript
  wstring Description;
  bool IsInternal;
  bool IsAdvanced;
  wstring PageName; // Maniascript
  uint CatalogPosition;
  CGameCtnCollector::EProdState ProdState;
  const CGameCtnCollector::EProdState ProdState_Auto;
  CPlugFileImg* IconFid;
  bool IconUseAutoRender;
  uint IconQuarterRotationY;
  float IconPhase01; // Range: 0 - 1
  uint InterfaceNumber; // Maniascript
  CPlugBitmap* const Icon; // Maniascript
  CMwNod* const ArticlePtr;
  const string SkinDirectory; // Maniascript
};

// File extension: 'Item.Gbx'
struct CGameItemModel : public CGameCtnCollector {
  CGameItemModel();

  enum class CGameItemModel::EnumItemType {
    _ItemType_Undefined = 0, // |ItemType|Undefined
    _ItemType_Decoration = 1, // |ItemType|Decoration
    _ItemType_Undefined = 2, // |ItemType|Undefined
    _ItemType_Bot = 3, // |ItemType|Bot
    _ItemType_Vehicle = 4, // |ItemType|Vehicle
    _ItemType_Undefined = 5, // |ItemType|Undefined
    _ItemType_Undefined = 6, // |ItemType|Undefined
    _ItemType_Undefined = 7, // |ItemType|Undefined
    _ItemType_Undefined = 8, // |ItemType|Undefined
    _ItemType_Undefined = 9, // |ItemType|Undefined
    _ItemType_Undefined = 10, // |ItemType|Undefined
    _ItemType_Undefined = 11, // |ItemType|Undefined
    _ItemType_Undefined = 12, // |ItemType|Undefined
    _ItemType_Undefined = 13, // |ItemType|Undefined
    _ItemType_Undefined = 14, // |ItemType|Undefined
    _ItemType_Undefined = 15, // |ItemType|Undefined
  };
  enum class CGameItemModel::EnumWaypointType {
    Start = 0,
    Finish = 1,
    Checkpoint = 2,
    None = 3,
    StartFinish = 4,
    Dispenser = 5,
  };
  enum class CGameItemModel::EnumDefaultCam {
    None = 0,
    Default = 1,
    Free = 2,
    Spectator = 3,
    Behind = 4,
    Close = 5,
    Internal = 6,
    Helico = 7,
    FirstPerson = 8,
    ThirdPerson = 9,
    ThirdPersonTop = 10,
    Iso = 11,
    IsoFocus = 12,
    Dia3 = 13,
    Board = 14,
    MonoScreen = 15,
    Rear = 16,
    Debug = 17,
    _1 = 18, // 1
    _2 = 19, // 2
    _3 = 20, // 3
    Alt1 = 21,
    Orbital = 22,
    Decals = 23,
    Snap = 24,
    NearOpponents = 25,
    MapThumbnail = 26,
  };
  enum class CGameItemModel::EnumInventoryItemClass {
    Weapon = 0,
    Movement = 1,
    Consumable = 2,
    Armor = 3,
  };
  const CGameItemModel::EnumItemType ItemTypeForUI;
  wstring Name;
  const CGameItemModel::EnumItemType ItemTypeEmbedded;
  const MwFastBuffer<CGameActionModel*> Actions;
  CGameItemPlacementParam* DefaultPlacementParam_Head;
  CGameItemPlacementParam* DefaultPlacementParam_Content;
  CGameItemPlacementParam* DefaultPlacementParam_Dbg;
  CPlugFileImg* Icon;
  CMwNod* EntityModelEdition;
  CMwNod* EntityModel;
  CPlugVFXFile* VFX;
  CPlugGameSkinAndFolder* MaterialModifier;
  CMwNod* const VisModel;
  CMwNod* VisModelCustom;
  DataRef DefaultSkinFileRef;
  DataRef ArchetypeRef;
  CGameItemModel* ArchetypeFid;
  CPlugMediaClipList* PodiumClipList;
  CPlugMediaClipList* IntroClipList;
  CMwNod* const PhyModel;
  CMwNod* PhyModelCustom;
  bool HasPath;
  bool CanFly;
  bool IsStart;
  bool IsCheckpoint;
  bool IsFinish;
  bool IsStartFinish;
  CGameItemModel::EnumItemType ItemTypeE;
  string SkinDirNameCustom;
  CPlugFileFidContainer* DefaultSkinFid;
  void InitFromArchetype();
  const MwFastBuffer<CMwNod*> Cameras;
  CGameItemModel::EnumDefaultCam DefaultCam;
  MwId DefaultWeaponName;
  MwFastArray<CPlugFileZip*> NadeoSkinsFids;
  CGameItemModel::EnumWaypointType WaypointType;
  vec3 GroundPoint;
  float PainterGroundMargin;
  float OrbitalCenterHeightFromGround;
  float OrbitalRadiusBase;
  float OrbitalPreviewAngle;
  CGameCtnMacroBlockInfo* IconMacroBlockInfo;
  bool DisableLightmap;
  bool DisableAutoCreateSound;
};

// Description: ""
struct CGameModulePlaygroundPlayerStateComponentModel : public CMwNod {
  enum class CGameModulePlaygroundPlayerStateComponentModel::EType {
    Invalid = 0,
    Gauge = 1,
    List = 2,
  };
  enum class CGameModulePlaygroundPlayerStateComponentModel::EModel {
    Invalid = 0,
    WeaponGauge = 1,
    ArmorGauge = 2,
    StaminaGauge = 3,
    ActionGauge = 4,
    WeaponList = 5,
  };
  const CGameModulePlaygroundPlayerStateComponentModel::EType Type; // Maniascript
  string Name; // Maniascript
  vec2 Position;
  float PosX; // Maniascript
  float PosY; // Maniascript
  float ZIndex; // Maniascript
  float Scale; // Maniascript
  float Rotation; // Maniascript
};

// File extension: 'GameObjectPhyModel.Gbx'
struct CGameObjectPhyModel : public CMwNod {
  CGameObjectPhyModel();

  DataRef MoveShape;
  CPlugSurface* MoveShapeFid;
  DataRef HitShape;
  DataRef TriggerShape;
  CPlugDynaModel* DynaModel;
  CPlugDynaPointModel* DynaPointModel;
  CPlugCharPhySpecialProperty* SpecialProperties;
  uint FirePeriod;
  DataRef ActionModel;
  const MwFastBuffer<CGameActionModel*> Actions;
  MwFastBuffer<CPlugTriggerAction*> Triggers;
  enum class CGameObjectPhyModel::EGameObjectPhyModelProgam {
    None = 0,
    Target = 1,
    Turret = 2,
  };
  CGameObjectPhyModel::EGameObjectPhyModelProgam Program;
  enum class CGameObjectPhyModel::EGameObjectPhyModelPersistence {
    OnPlayerUnspawn = 0,
    OnPlayerRemoved = 1,
    NeverRemove = 2,
  };
  CGameObjectPhyModel::EGameObjectPhyModelPersistence Persistence;
  bool CanStopEnemyBullet;
  bool CanStopEnemy;
  float ThrowSpeed;
  float ThrowAngularSpeed;
  float ScaleCoefMax;
  float StaminaSpawnCoef;
  uint Armor;
  bool HasALifeTime;
  uint LifeTimeDuration;
  uint TimeBeforeDome;
  bool HealEnabled;
  uint HealArmorGainPerSecond;
  bool ShieldEnabled;
  uint ShieldDomeArmor;
  bool ThrowableEnabled;
  float ThrowableDamageCoef;
  uint ThrowableStunnedDuration;
  uint ThrowableCooldownBetweenTakeAndThrow;
  float StaminaThrowCost;
  bool BumperEnabled;
  uint BumperMaxInstancesSpawnable;
  bool MagnetEnabled;
};

// File extension: 'GameObjectVisModel.Gbx'
struct CGameObjectVisModel : public CMwNod {
  CGameObjectVisModel();

  DataRef Mesh;
  CPlugSolid2Model* MeshShadedFid;
  CPlugAnimLocSimple* LocAnim;
  vec3 DomeShaderColor;
  wstring SoundRef_Spawn;
  wstring SoundRef_Unspawn;
  wstring SoundRef_Grab;
  wstring SoundRef_Smashed;
  wstring SoundRef_Permanent;
  CPlugSound* Sound_Spawn;
  CPlugSound* Sound_Unspawn;
  CPlugSound* Sound_Grab;
  CPlugSound* Sound_Smashed;
  CPlugSound* Sound_Permanent;
  iso4 SoundLoc_Permanent;
  CPlugParticleEmitterModel* ParticleModel_Alive;
  vec3 ParticleModelPos_Alive;
  CPlugSolid2Model* MeshShaded;
  wstring SmashParticleRef;
  CPlugParticleEmitterModel* DestroyParticleModel;
  DataRef VisEntFx;
};

// File extension: 'Action.Gbx'
struct CGameActionModel : public CMwNod {
  CGameActionModel();

  enum class CGameActionModel::EInventoryItemClass {
    Weapon = 0,
    Movement = 1,
    Consumable = 2,
    Armor = 3,
  };
  enum class CGameActionModel::EBehavior {
    Projectile = 0,
    Timeline = 1,
    Script = 2,
    ExternalActions = 3,
  };
  enum class CGameActionModel::EExternalActions {
    Run = 0,
    Climb = 1,
    Fly = 2,
    Attractor1 = 3,
    Sneak = 4,
    Slide = 5,
    Gliding = 6,
    Attractor2 = 7,
    Parachute = 8,
    SnapAim = 9,
    Grapnel = 10,
    SuitcaseTeleport = 11,
    PowerBoots = 12,
    JetPack = 13,
    Aim = 14,
    Bumper = 15,
    SuitcaseShield = 16,
    SuitcaseHeal = 17,
    SelectedSuitcase = 18,
    NewRun = 19,
    ForceField = 20,
    Ice = 21,
  };
  string ActionName;
  DataRef Icon;
  CGameActionModel::EBehavior Behaviour;
  CGameActionModel::EInventoryItemClass InventoryItemClass;
  const wstring Description;
  bool UseableInWater;
  bool GroundedOwnerOnly;
  bool StaticOwnerOnly;
  bool CancelledByTechNoWeapon;
  bool UseVehicleGuns;
  bool UseVehicleLockedEnt;
  uint Cooldown;
  uint EnergyCost;
  uint EnergyMax;
  bool EnergyReload;
  uint DirectHitDamage;
  uint ExplosionDamage;
  float ExplosionDamage_Radius;
  float ExplosionDamage_RadiusAttenuation;
  CPlugScriptWithSettings* Script;
  DataRef Icon_Focused_Validate;
  DataRef Icon_Focused_Cancel;
  MwFastBuffer<CPlugCustomBulletModel*> Projectiles;
  MwFastBuffer<CPlugCustomBeamModel*> Beams;
  DataRef CrossHair;
  CPlugAnimFile* Anim;
  bool IsAnimPartial;
  MwFastBuffer<CGameItemModel*> Items;
  CGameActionModel::EExternalActions ExternalAction;
  MwFastBuffer<CPlugBulletModel*> BulletModels_Nadeo;
  CPlugBulletModel* const CompatBullet_Nadeo;
  const uint ParticleBlockCount;
  void LogParticleBlockModelIds();
};

struct CGameWaypointSpecialProperty : public CMwNod {
  CGameWaypointSpecialProperty();

  string Tag;
  uint Order;
  void LinkedCheckpointToggle();
};

// File extension: 'GameActionFxResources.Gbx'
struct CGameActionFxResources : public CMwNod {
  CGameActionFxResources();

  CSystemFidsFolder* ParticleEmitterSubModels_Continuous;
  CSystemFidsFolder* ParticleEmitterSubModels_Splash;
  CSystemFidsFolder* MeshFolder;
  CSystemFidsFolder* AudioFxLibraryFolder;
  CSystemFidsFolder* BeamEmitterSubModels;
  CMwRefBuffer* BulletModelsLibrary;
};

// File extension: 'Gate.gbx'
struct CGameGateModel : public CMwNod {
  CGameGateModel();

  CPlugSurface* Shape;
  CGameArmorModel* Armor;
  CPlugSound* GateOpenSound;
  CPlugSound* GateSound;
  CPlugSound* GateCloseSound;
};

// File extension: 'Teleporter.gbx'
struct CGameTeleporterModel : public CMwNod {
  CGameTeleporterModel();

  CPlugSpawnModel* Spawn;
  CPlugSurface* TriggerShape;
  vec3 CenterPos;
};

// File extension: 'Armor.gbx'
struct CGameArmorModel : public CMwNod {
  CGameArmorModel();

  uint ArmorMax;
  uint ReplenishDelayAfterDamage;
};

// File extension: 'CaptureZone.gbx'
struct CGameCaptureZoneModel : public CMwNod {
  CGameCaptureZoneModel();

  iso4 Loc;
  float Radius;
  float Height;
};

// File extension: 'Turbine.gbx'
struct CGameTurbineModel : public CMwNod {
  CGameTurbineModel();

  iso4 Loc;
  float SilencerBubble_RadiusMute;
  float SilencerBubble_RadiusFade;
  bool SilencerBubble_OnlyPlayerSounds;
};

struct CGameGhostTMData : public CMwNod {
  CGameGhostTMData();

};

struct CGameModulePlaygroundModel : public CGameModuleModelCommon {
  DataRef Icon;
  float SizeX; // Range: 0 - 320
  float SizeY; // Range: 0 - 180
  CPlugFileText* ManialinkCode;
};

struct CGameModuleNodForPropertyList : public CMwNod {
  CGameModuleNodForPropertyList();

};

// File extension: 'Module.Gbx'
struct CGameModuleMenuModel : public CGameModuleModelCommon {
  CGameModuleMenuModel();

  CPlugFileTextScript* ManiaAppScript;
  MwFastBuffer<CGameModuleMenuPageModel*> Pages; // Maniascript
  CGameModuleMenuPageModel* AddPage(wstring PageUrl); // Maniascript
  void AddLink(CGameModuleMenuPageModel* ParentPage, CGameModuleMenuPageModel* ChildPage); // Maniascript
  string MenuScript; // Maniascript
};

struct CGameModuleModelCommon : public CMwNod {
  const uint LastEditionTime;
};

// File extension: 'Module.Gbx'
struct CGameModulePlaygroundInventoryModel : public CGameModulePlaygroundModel {
  CGameModulePlaygroundInventoryModel();

  string Title;
  uint SlotCapacity;
  MwFastBuffer<CGameModuleNodForPropertyList*> Categories;
};

// File extension: 'Module.Gbx'
struct CGameModulePlaygroundScoresTableModel : public CGameModulePlaygroundModel {
  CGameModulePlaygroundScoresTableModel();

  MwFastBuffer<CGameModuleNodForPropertyList*> Columns;
  bool UseTeams;
  bool DisplayTeamScores;
  string Title;
  bool UsePlayerDarkening;
  bool UseSettings;
  bool SortSpectatorsLast;
  bool SortDisconnectedLast;
  bool DisplayLocalPlayerInfo;
  bool DisplayServerName;
  bool DisplayColumnLegends;
  uint ColumnCount; // Range: 1 - 10
  uint LineCount; // Range: 1 - 10
  float ColumnMargin;
  DataRef GameModeIcon;
  DataRef SmStormBackground;
  DataRef TmStadiumBackground;
  DataRef TmCanyonBackground;
  DataRef TmValleyBackground;
  DataRef TmLagoonBackground;
  DataRef TmNextBackground;
  DataRef SmHeaderTeam1Bg;
  DataRef SmHeaderTeam2Bg;
  DataRef TmHeaderTeam1Bg;
  DataRef TmHeaderTeam2Bg;
  DataRef TmNextHeader;
  DataRef TmNextFooter;
  DataRef TmNextFooterMedal;
  DataRef SmPlayerCardSquareQuad;
  DataRef SmPlayerCardBackgroundQuad;
  DataRef SmPlayerCardBackgroundQuadRevert;
  DataRef TmPlayerCardSquareQuad;
  DataRef TmPlayerCardBackgroundQuad;
  DataRef TmPlayerCardBackgroundQuadRevert;
  DataRef Echelon1;
  DataRef Echelon2;
  DataRef Echelon3;
  DataRef Echelon4;
  DataRef Echelon5;
  DataRef Echelon6;
  DataRef Echelon7;
  DataRef Echelon8;
  DataRef Echelon9;
  DataRef Echelon1Rev;
  DataRef Echelon2Rev;
  DataRef Echelon3Rev;
  DataRef Echelon4Rev;
  DataRef Echelon5Rev;
  DataRef Echelon6Rev;
  DataRef Echelon7Rev;
  DataRef Echelon8Rev;
  DataRef Echelon9Rev;
  DataRef Arrow_Back;
  DataRef RankGold;
  DataRef RankGold_Effects;
  DataRef RankSilver;
  DataRef RankSilver_Effects;
  DataRef RankBronze;
  DataRef RankNormal;
  DataRef SmallMedalGold;
  DataRef SmallMedalSilver;
  DataRef SmallMedalBronze;
  DataRef SmallMedalNadeo;
  string HeaderStyle;
  string PlayerInfoLocalRankStyle;
  string PlayerCardLocalRankStyle;
  float HeaderScale;
  float PlayerInfoLocalRankScale;
  float PlayerCardLocalRankScale;
  string DefaultTeam1Name;
  string DefaultTeam2Name;
};

// File extension: 'Module.Gbx'
struct CGameModulePlaygroundStoreModel : public CGameModulePlaygroundModel {
  CGameModulePlaygroundStoreModel();

  string Title;
  string Subtitle;
  DataRef PowerUpImageLvl1;
  DataRef PowerUpImageLvl2;
  DataRef PowerUpImageLvl3;
  DataRef PowerUpImageLvl4;
  DataRef PowerUpImageLvl5;
  MwFastBuffer<CGameActionModel*> Actions;
  MwFastBuffer<CGameItemModel*> Items;
};

// Description: ""
// File extension: 'Module.Gbx'
struct CGameModulePlaygroundHudModel : public CGameModuleModelCommon {
  CGameModulePlaygroundHudModel();

  CPlugFileText* Manialink;
  DataRef Inventory;
  DataRef Store;
  DataRef Chrono;
  DataRef ScoresTable;
  DataRef SpeedMeter;
  DataRef PlayerState;
  DataRef TeamState;
  MwId SubModuleAdd(wstring ModulePath, string ModuleName); // Maniascript
  MwId SubModuleSetNameAndId(MwId ModuleId, string NewModuleName); // Maniascript
  void SubModuleRemove(MwId ModuleId); // Maniascript
  bool SubModuleIsContextActive_Deprec(MwId ModuleId, MwId ContextId); // Maniascript
  void SubModuleSetContextIsActive_Deprec(MwId ModuleId, MwId ContextId, bool IsActive); // Maniascript
  const MwFastBuffer<MwId> SubModuleIds; // Maniascript
  void SubModuleRetrieve(MwId ModuleId); // Maniascript
  CGameModulePlaygroundHudModelModule* SubModule; // Maniascript
  MwId ContextAdd(string ContextName); // Maniascript
  MwId ContextSetId(MwId ContextId, string NewContextName); // Maniascript
  void ContextRemove(MwId ContextId); // Maniascript
  const MwSArray<MwId> ContextsIds; // Maniascript
  const MwFastBuffer<CGameModulePlaygroundHudModelModule*> SubModules; // Maniascript
  CGameModulePlaygroundHudModelModule* NewSubModule(wstring ModulePath); // Maniascript
  void DeleteSubModule(CGameModulePlaygroundHudModelModule* SubModule); // Maniascript
  bool SubModuleIsContextActive(CGameModulePlaygroundHudModelModule* SubModule, MwId ContextId); // Maniascript
  void SubModuleSetContextIsActive(CGameModulePlaygroundHudModelModule* SubModule, MwId ContextId, bool IsActive); // Maniascript
  const uint EditorContextIndex; // Maniascript
};

// File extension: 'Module.Gbx'
struct CGameModuleMenuPageModel : public CGameModuleModelCommon {
  CGameModuleMenuPageModel();

  CPlugFileText* Manialink;
  MwFastBuffer<CGameModuleNodForPropertyList*> Components;
  string ManialinkText; // Maniascript
};

// File extension: 'Editor.Gbx'
struct CGameEditorModel : public CMwNod {
  CGameEditorModel();

  enum class CGameEditorModel::EEditorType {
    ActionMaker = 0,
    ChallengeEditor = 1,
    ItemEditor = 2,
    InterfaceDesigner = 3,
    MeshModeler = 4,
  };
  const CGameEditorModel::EEditorType EditorType;
  bool AutoGenerateHelp;
  bool AutoCreateTooltipsFromBindings;
  CGameManiaAppTextSet* MainPluginTextSet;
  MwFastBuffer<string> SubPluginIds;
  const MwFastBuffer<CGameManiaAppTextSet*> SubPluginTextSets;
};

// File extension: ''
struct CGameVehicleModel : public CMwNod {
  CGameVehicleModel();

  UnnamedType PhyModel;
  MwFastBuffer<CGameObjectModel*> SpawnableObjects;
  CPlugVehicleVisModel* VisModel;
};

struct CGameObjectModel : public CMwNod {
  CGameObjectModel();

  wstring ScriptId;
  CGameObjectPhyModel* Phy;
  CGameObjectVisModel* Vis;
  CGameObjectModel* SlaveShieldDome;
  CGameObjectModel* SlaveHealDome;
  wstring m_InventoryParams_InventoryName;
  wstring m_InventoryParams_InventoryDescription;
  EGameInventoryItemClass m_InventoryParams_InventoryItemClass;
  uint m_InventoryParams_InventoryOccupation;
};

// File extension: 'Module.Gbx'
struct CGameModulePlaygroundChronoModel : public CGameModulePlaygroundModel {
  CGameModulePlaygroundChronoModel();

  float SizeX; // Range: 0 - 320
  float SizeY; // Range: 0 - 180
  float TextSize; // Range: 0 - 180
};

// File extension: 'Module.Gbx'
struct CGameModulePlaygroundSpeedMeterModel : public CGameModulePlaygroundModel {
  CGameModulePlaygroundSpeedMeterModel();

  enum class CGameModulePlaygroundSpeedMeterModel::ESpeedUnit {
    Undefined = 0,
    MetersPerSecond = 1,
    KmPerHour = 2,
    MilesPerHour = 3,
  };
  float GlobalScale; // Range: 0 - 5
  uint MeterNbGauge; // Range: 0 - 3
  float MeterSize;
  float MeterClipOffset;
  float MeterOpacity; // Range: 0 - 1
  float MeterFullCircleValue; // Range: 0.001 - 1e+06
  float MeterCircleMaxAngle; // Range: 0 - 360
  float MeterCircleOffsetAngle; // Range: -360 - 360
  bool MeterShowBackground;
  bool MeterShowSpeedBar;
  string MeterBGColor;
  string MeterInnerColor;
  string MeterMiddleColor;
  string MeterOuterColor;
  CGameModulePlaygroundSpeedMeterModel::ESpeedUnit SpeedForcedUnit;
  float SpeedValueScale; // Range: 0 - 5
  bool SpeedShowUnit;
  bool SpeedShowFooter;
  float SpeedUnitScale; // Range: 0 - 5
  bool SpeedShowCustomText;
  string SpeedCustomText;
  bool DistanceShowValue;
  bool DistanceShowFooter;
  DataRef GaugeBGImage;
  DataRef MeterInnerImage;
  DataRef MeterMiddleImage;
  DataRef MeterOuterImage;
  DataRef SpeedBarImage;
  DataRef FooterKMHLineImage;
  DataRef FooterKMHUnitImage;
  DataRef FooterMiHUnitImage;
  DataRef FooterMSUnitImage;
  DataRef FootLineDistanceImage;
};

// userName: 'PlaceParam'
// File extension: 'PlaceParam.Gbx'
struct CGameItemPlacementParam : public CMwNod {
  CGameItemPlacementParam();

  MwFastBuffer<vec3> PivotPositions;
  bool SwitchPivotManually;
  float FlyStep;
  float FlyOffset;
  bool GhostMode;
  float GridSnap_HStep;
  float GridSnap_VStep;
  float GridSnap_HOffset;
  float GridSnap_VOffset;
  float PivotSnap_Distance;
  bool YawOnly;
  bool NotOnObject;
  bool AutoRotation;
  vec3 Cube_Center;
  float Cube_Size;
  MwFastBuffer<vec3> m_PivotPositions;
  void AddPivotPosition();
  void AppendPivotPositionsFromMagnets();
  void RemoveLastPivotPosition();
  void RemoveAllPivotPositions();
  MwFastBuffer<quat> PivotRotations;
  void AddPivotRotation_Deprecated();
  void RemoveLastPivotRotation();
  void RemoveAllPivotRotations();
  MwFastBuffer<GmTransYawPitchRollDeg> m_MagnetLocs_Degrees;
  void AddMagnetLoc();
  void AddMagnetLoc_Left();
  void AddMagnetLoc_Right();
  void AddMagnetLoc_Up();
  void AddMagnetLoc_Down();
  void AddMagnetLoc_Front();
  void AddMagnetLoc_Back();
  void RemoveLastMagnetLoc();
  void RemoveAllMagnetLocs();
  bool HasPath;
  bool IsFreelyAnchorable;
  NPlugItemPlacement_SClass* PlacementClass;
};

// Description: ""
// File extension: 'Module.Gbx'
struct CGameModulePlaygroundPlayerStateModel : public CGameModulePlaygroundModel {
  CGameModulePlaygroundPlayerStateModel();

  bool DynamicPreview; // Maniascript
  float PreviewSpeed; // Maniascript
  MwFastBuffer<CGameModulePlaygroundPlayerStateComponentModel*> ComponentsForDesign;
  MwId ComponentAddDefaultName(CGameModulePlaygroundPlayerStateComponentModel::EModel ComponentModel); // Maniascript
  MwId ComponentAddCustomName(CGameModulePlaygroundPlayerStateComponentModel::EModel ComponentModel, string ComponentName); // Maniascript
  MwId ComponentSetNameAndId(MwId ComponentId, string NewComponentName); // Maniascript
  void ComponentRemove(MwId ComponentId); // Maniascript
  const MwFastBuffer<MwId> ComponentIds; // Maniascript
  void ComponentRetrieve(MwId ComponentId); // Maniascript
  CGameModulePlaygroundPlayerStateComponentModel* Component; // Maniascript
  const MwFastBuffer<CGameModulePlaygroundPlayerStateComponentModel*> Components; // Maniascript
  CGameModulePlaygroundPlayerStateComponentModel* NewComponent(CGameModulePlaygroundPlayerStateComponentModel::EModel ComponentModel); // Maniascript
  void DeleteComponent(CGameModulePlaygroundPlayerStateComponentModel* Component); // Maniascript
};

// File extension: 'Module.Gbx'
struct CGameModulePlaygroundTeamStateModel : public CGameModulePlaygroundModel {
  CGameModulePlaygroundTeamStateModel();

  enum class CGameModulePlaygroundTeamStateModel::ECardAlign {
    Vertical = 0,
    Horizontal = 1,
    Custom = 2,
  };
  float GlobalScale; // Range: 0 - 100
  uint PlayerPerTeam; // Range: 1 - 1000
  bool ShowTeamNames;
  bool ShowTeamEmblems;
  bool ShowTeamScores;
  bool ShowTeamBackgrounds;
  bool UseCustomTeamBackground;
  DataRef CustomTeamBackground;
  vec2 FirstPlayerPosition;
  CGameModulePlaygroundTeamStateModel::ECardAlign PlayerCardsAlign;
  vec2 NextPlayerPositionOffset;
  bool ShowPlayerNames;
  bool ShowPlayerAvatars;
  bool ShowPlayerEchelons;
  bool ShowPlayerManiaStars;
  bool ShowPlayerTools;
  bool ShowPlayerTags;
  bool ShowPlayerScores;
  bool ShowPlayerRoundScores;
  float PlayerGaugesHeight; // Range: 0 - 100
  bool ShowAllyArmorGauges;
  bool ShowAllyWeaponGauges;
  bool ShowAllyStaminaGauges;
  bool ShowEnemyTeam;
  bool ShowEnemyArmorGauges;
  bool ShowEnemyWeaponGauges;
  bool ShowEnemyStaminaGauges;
  bool UseCustomActiveBackground;
  DataRef CustomActiveBackground;
  bool UseCustomActiveColor;
  vec3 ActiveColor;
  float ActiveColorOpacity; // Range: 0 - 1
  bool UseCustomInactiveBackground;
  DataRef CustomInactiveBackground;
  bool UseCustomInactiveColor;
  vec3 InactiveColor;
  float InactiveColorOpacity; // Range: 0 - 1
  DataRef Team1Emblem;
  DataRef Team1Outline;
  bool Team1UseDefaultColors;
  vec3 Team1PrimaryColor;
  float Team1PrimaryColorOpacity; // Range: 0 - 1
  vec3 Team1SecondaryColor;
  float Team1SecondaryColorOpacity; // Range: 0 - 1
  DataRef Team2Emblem;
  DataRef Team2Outline;
  bool Team2UseDefaultColors;
  vec3 Team2PrimaryColor;
  float Team2PrimaryColorOpacity; // Range: 0 - 1
  vec3 Team2SecondaryColor;
  float Team2SecondaryColorOpacity; // Range: 0 - 1
  DataRef Echelon1Image;
  DataRef Echelon2Image;
  DataRef Echelon3Image;
  DataRef Echelon4Image;
  DataRef Echelon5Image;
  DataRef Echelon6Image;
  DataRef Echelon7Image;
  DataRef Echelon8Image;
  DataRef Echelon9Image;
  DataRef Echelon1RevImage;
  DataRef Echelon2RevImage;
  DataRef Echelon3RevImage;
  DataRef Echelon4RevImage;
  DataRef Echelon5RevImage;
  DataRef Echelon6RevImage;
  DataRef Echelon7RevImage;
  DataRef Echelon8RevImage;
  DataRef Echelon9RevImage;
};

struct CGameModulePlaygroundPlayerStateGaugeModel : public CGameModulePlaygroundPlayerStateComponentModel {
  CGameModulePlaygroundPlayerStateGaugeModel();

  enum class CGameModulePlaygroundPlayerStateGaugeModel::ESource {
    Weapon_current = 0, // Weapon current
    Weapon_max = 1, // Weapon max
    Weapon_grading = 2, // Weapon grading
    Armor_current = 3, // Armor current
    Armor_max = 4, // Armor max
    Armor_grading = 5, // Armor grading
    Stamina_current = 6, // Stamina current
    Stamina_max = 7, // Stamina max
    Stamina_grading = 8, // Stamina grading
    X = 9,
    Y = 10,
    Z = 11,
    Reserved9 = 12,
    Reserved10 = 13,
    Reserved11 = 14,
    Reserved12 = 15,
    Reserved13 = 16,
    Reserved14 = 17,
    Reserved15 = 18,
    Reserved16 = 19,
    Reserved17 = 20,
    Constant = 21,
    Action_current = 22, // Action current
    Action_max = 23, // Action max
    Action_grading = 24, // Action grading
    Action_current_cooldown = 25, // Action current cooldown
    Action_max_cooldown = 26, // Action max cooldown
  };
  enum class CGameModulePlaygroundPlayerStateGaugeModel::EDistanceUnit {
    Default = 0,
    Meter = 1,
    Kilometer = 2,
    Mile = 3,
  };
  enum class CGameModulePlaygroundPlayerStateGaugeModel::ESpeedUnit {
    Default = 0,
    m_s = 1, // m/s
    km_h = 2, // km/h
    Mi_h = 3, // Mi/h
  };
  enum class CGameModulePlaygroundPlayerStateGaugeModel::EDirection {
    Right = 0,
    Top_right = 1, // Top right
    Top = 2,
    Top_left = 3, // Top left
    Left = 4,
    Bottom_left = 5, // Bottom left
    Bottom = 6,
    Bottom_right = 7, // Bottom right
    Center = 8,
  };
  enum class CGameModulePlaygroundPlayerStateGaugeModel::EActionInput {
    Activable1 = 0,
    Activable2 = 1,
    Activable3 = 2,
    Activable4 = 3,
    Activable5 = 4,
    Activable6 = 5,
    Activable7 = 6,
    Activable8 = 7,
    Activable9 = 8,
    Activable0 = 9,
    Consumable1 = 10,
    Consumable2 = 11,
  };
  CGameModulePlaygroundPlayerStateGaugeModel::EActionInput ValueActionInput;
  bool ValueActionIsCooldown;
  CGameModulePlaygroundPlayerStateGaugeModel::ESource CurrentValueSource;
  CGameModulePlaygroundPlayerStateGaugeModel::EDistanceUnit CurrentValueDistanceUnit;
  CGameModulePlaygroundPlayerStateGaugeModel::ESpeedUnit CurrentValueSpeedUnit;
  float CurrentValueCustom;
  CGameModulePlaygroundPlayerStateGaugeModel::ESource MaxValueSource;
  CGameModulePlaygroundPlayerStateGaugeModel::EDistanceUnit MaxValueDistanceUnit;
  CGameModulePlaygroundPlayerStateGaugeModel::ESpeedUnit MaxValueSpeedUnit;
  float MaxValueCustom;
  bool HideWhenFull;
  float HideDelay;
  bool ShowGauge;
  vec2 GaugePosition;
  float GaugeRotation;
  vec2 GaugeSize;
  CGameModulePlaygroundPlayerStateGaugeModel::ESource GradingSource;
  CGameModulePlaygroundPlayerStateGaugeModel::EDistanceUnit GradingDistanceUnit;
  CGameModulePlaygroundPlayerStateGaugeModel::ESpeedUnit GradingSpeedUnit;
  float GradingCustom;
  bool GaugeUseDefaultColor;
  vec3 GaugeCustomColor;
  bool ShowValue;
  CGameModulePlaygroundPlayerStateGaugeModel::EDirection ValuePosition;
  vec2 ValueRelativePosition;
  float ValueRotation;
  float CurrentValueSize;
  bool ShowMaxValue;
  float MaxValueSize;
  bool ValueUseDefaultColor;
  vec3 ValueCustomColor;
  bool ShowBackground;
  vec2 BackgroundRelativePosition;
  float BackgroundRotation;
  vec2 BackgroundSize;
  DataRef BackgroundImage;
  bool BackgroundUseDefaultColor;
  vec3 BackgroundCustomColor;
  bool ShowIcon;
  CGameModulePlaygroundPlayerStateGaugeModel::EDirection IconPosition;
  vec2 IconRelativePosition;
  float IconRotation;
  float IconScale;
  DataRef IconImage;
  bool IconUseDefaultColor;
  vec3 IconCustomColor;
  bool IconClipFromValue;
  bool ShowIconBackground;
  float IconBackgroundScale;
  DataRef IconBackgroundImage;
  bool IconUpdateFromWeapon;
  bool IconUpdateFromStamina;
  bool IconUpdateFromAction;
  bool IconUpdateSetCustomImages;
  const MwFastBuffer<string> WeaponCustomIconsNames;
  const MwFastBuffer<wstring> WeaponCustomIconsUrls;
  DataRef StaminaJumpImage;
  DataRef StaminaRunImage;
  DataRef StaminaAttractorImage;
  DataRef StaminaSneakImage;
  DataRef StaminaSlideImage;
  DataRef StaminaBackFlipImage;
  DataRef StaminaFrontFlipImage;
  DataRef StaminaSpotImage;
  DataRef StaminaTeleportImage;
  DataRef StaminaInteractImage;
  DataRef StaminaPowerJumpImage;
  DataRef StaminaScopeImage;
  bool ShowCustomText;
  string CustomText;
  CGameModulePlaygroundPlayerStateGaugeModel::EDirection CustomTextPosition;
  vec2 CustomTextRelativePosition;
  float CustomTextRotation;
  float CustomTextSize;
  bool CustomTextUseDefaultColor;
  vec3 CustomTextCustomColor;
};

// File extension: 'PixelArt.Gbx'
struct CGamePixelArtModel : public CMwNod {
  CGamePixelArtModel();

};

// File extension: 'BlockItem.Gbx'
struct CGameBlockItem : public CMwNod {
  CGameBlockItem();

  MwId ArchetypeBlockInfoId_GameBox;
  UnnamedEnum ArchetypeBlockInfoCollectionId;
  wstring ArchetypeBlockInfoCollectionId_Text;
  MwFastBuffer<CPlugCrystal*> BlockInfoMobilSkins_Crystals;
  MwFastBuffer<CPlugStaticObjectModel*> BlockInfoMobilSkins_StaticObjModels;
  MwFastBuffer<CPlugSurface*> BlockInfoMobilSkins_TriggerShapes;
  MwFastBuffer<uint> BlockInfoMobilSkins_MobilIds;
  MwFastBuffer<vec3> BlockInfoMobilSkins_SpawnLocTranss;
  MwFastBuffer<float> BlockInfoMobilSkins_SpawnLocYaws;
  MwFastBuffer<float> BlockInfoMobilSkins_SpawnLocPitchs;
  MwFastBuffer<float> BlockInfoMobilSkins_SpawnLocRolls;
  MwFastBuffer<vec3> BlockInfoMobilSkins_SectorPoss;
  const MwId ArchetypeBlockInfoId;
  const MwFastBuffer<CPlugCrystal*> CustomizedVariants;
};

struct CGameCommonItemEntityModelEdition : public CMwNod {
  CGameCommonItemEntityModelEdition();

  enum class CGameCommonItemEntityModelEdition::EnumItemType {
    Undefined = 0,
    Ornament = 1,
    PickUp = 2,
    Character = 3,
    Vehicle = 4,
    Spot = 5,
    Cannon = 6,
    Group = 7,
    Decal = 8,
    Turret = 9,
    Wagon = 10,
    Block = 11,
    EntitySpawner = 12,
    DeprecV = 13,
    Procedural = 14,
    Generic = 15,
  };
  enum class CGameCommonItemEntityModelEdition::EnumInventoryItemClass {
    Weapon = 0,
    Movement = 1,
    Consumable = 2,
    Armor = 3,
  };
  const CGameCommonItemEntityModelEdition::EnumItemType ItemType;
  CPlugCrystal* MeshCrystal;
  CPlugParticleEmitterModel* DestroyParticleModel;
  float Mass;
  CGameActionModel* PickupActionModel;
  const MwFastBuffer<CGameActionModel*> TriggeredActions;
  MwFastBuffer<CPlugTriggerAction*> Triggers;
  wstring InventoryName;
  wstring InventoryDescription;
  CGameCommonItemEntityModelEdition::EnumInventoryItemClass InventoryItemClass;
  uint InventoryOccupation;
};

struct CGameCommonItemEntityModel : public CMwNod {
  CGameCommonItemEntityModel();

  CMwNod* VisModel;
  CMwNod* PhyModel;
  CMwNod* StaticObject;
  CPlugSurface* TriggerShape;
  iso4 SpawnLoc;
};

struct CGameCharacterModel : public CMwNod {
  CGameCharacterModel();

  UnnamedType Phy;
  CPlugCharVisModel* Vis;
  float LockMaxDist;
  MwFastBuffer<CGameObjectModel*> SpawnableObjects;
};

struct CGameManiaAppTextSet : public CMwNod {
  CGameManiaAppTextSet();

  CPlugFileTextScript* const MainScript;
  const MwFastBuffer<string> LayerManialinkNames;
  const MwFastBuffer<CPlugFileText*> LayerManialinks;
};

struct CGameObjectPhyCompoundModel : public CMwNod {
  CGameObjectPhyCompoundModel();

};

struct CGameModulePlaygroundHudModelModule : public CMwNod {
  CGameModulePlaygroundHudModelModule();

  string ModuleName; // Maniascript
  float PosX; // Maniascript
  float PosY; // Maniascript
  float ZIndex; // Maniascript
  float Scale; // Maniascript
  const wstring ModulePath; // Maniascript
  uint ContextsMask;
};

struct CGameModulePlaygroundPlayerStateListModel : public CGameModulePlaygroundPlayerStateComponentModel {
  CGameModulePlaygroundPlayerStateListModel();

  enum class CGameModulePlaygroundPlayerStateListModel::ESource {
    Weapon_current = 0, // Weapon current
    Weapon_max = 1, // Weapon max
    Weapon_grading = 2, // Weapon grading
    Armor_current = 3, // Armor current
    Armor_max = 4, // Armor max
    Armor_grading = 5, // Armor grading
    Stamina_current = 6, // Stamina current
    Stamina_max = 7, // Stamina max
    Stamina_grading = 8, // Stamina grading
    X = 9,
    Y = 10,
    Z = 11,
    Reserved9 = 12,
    Reserved10 = 13,
    Reserved11 = 14,
    Reserved12 = 15,
    Reserved13 = 16,
    Reserved14 = 17,
    Reserved15 = 18,
    Reserved16 = 19,
    Reserved17 = 20,
    Constant = 21,
    Action_current = 22, // Action current
    Action_max = 23, // Action max
    Action_grading = 24, // Action grading
    Action_current_cooldown = 25, // Action current cooldown
    Action_max_cooldown = 26, // Action max cooldown
  };
  enum class CGameModulePlaygroundPlayerStateListModel::EDistanceUnit {
    Default = 0,
    Meter = 1,
    Kilometer = 2,
    Mile = 3,
  };
  enum class CGameModulePlaygroundPlayerStateListModel::ESpeedUnit {
    Default = 0,
    m_s = 1, // m/s
    km_h = 2, // km/h
    Mi_h = 3, // Mi/h
  };
  uint ItemCount;
  vec2 ItemPositionOffset;
  float FocusScale;
  DataRef Weapon0Image;
  DataRef Weapon1Image;
  DataRef Weapon2Image;
  DataRef Weapon3Image;
  vec2 ItemIconSize;
  bool IconUseDefaultColor;
  vec3 IconCustomColor;
  bool ShowItemValue;
  vec2 ItemValuePosition;
  float ItemValueSize;
  bool ValueUseDefaultColor;
  vec3 ValueCustomColor;
  bool ShowBackground;
  float BackgroundBorderSize;
  bool BackgroundUseDefaultColor;
  vec3 BackgroundCustomColor;
  float BackgroundOpacity; // Range: 0 - 1
  DataRef BackgroundImage;
};

struct CGameModuleEditorModel : public CGameModuleModelCommon {
  CGameModuleEditorModel();

  float SizeX; // Range: 0 - 320
  float SizeY; // Range: 0 - 180
  CPlugFileText* ManialinkCode;
};

struct CGameModuleEditorGraphEditionModel : public CGameModuleEditorModel {
  CGameModuleEditorGraphEditionModel();

};

} // namespace GameData

namespace Meta {

struct NPlugPrefab_SLodGroup {
  GmBoxAligned BBox;
  UnknownType LodMaxDistAtFov90s;
};

struct CPlugAnimTimingFixedPeriod {
  uint cFrame;
  uint FramePeriod;
  bool Looping;
};

struct NPlugAnim_SChannelGroupImport {
  MwId Name;
  MwId SkelName;
  MwId RootJointName;
  string ExcludeJointMatchName;
  string IncludeOnlyJointMatchName;
  MwFastBuffer<NPlugAnim_SChannelGroupJointImport> Joints;
};

struct NPlugAdn_STagFromName {
  string MatchName;
  NPlugAdn::ETagCat Cat;
  NPlugAdn_STag Tag;
};

struct NPlugAdn_STagFid {
  CSystemFidFile* Fid;
  MwSArray<NPlugAdn_STag> RequiredTags;
  MwSArray<NPlugAdn_STag> Tags;
  float Proba;
};

struct CPlugAnimClipFlags {
  float WorldSpeedKmh;
  float MinPlaySpeed;
  float MaxPlaySpeed;
  bool Looping;
  bool IsDifference;
  bool RootRotation;
  bool FootCasting;
  bool FootCentering;
  float PlaySpeed;
  bool IsPartial;
  MwId DifferenceFromClip;
  MwId VariantGroup;
  EPlugAnimClipAlignOffset AlignOffset;
};

struct CPlugAnimFileXml {
};

struct CPlugAdnPartInstance {
  CPlugAdnPart* Part;
  CPlugMetaData* Shader;
};

// userName: 'Shader'
// File extension: 'AdnShader.gbx'
struct CPlugAdnShader_Part {
  CMwNod* Normal;
  CMwNod* Mask;
  CMwNod* BaseColorMotif;
  CMwNod* Zone0_Normal;
  CMwNod* Zone0_Rough;
  CMwNod* Zone1_Normal;
  CMwNod* Zone1_Rough;
  CMwNod* Zone2_Normal;
  CMwNod* Zone2_Rough;
  RGBAColor BaseColor_Motif;
  uint iChannelMotif;
  GmSimi2 TcToTcMotif;
  bool IsMotifWrap;
  RGBAColor BaseColor_Zone0;
  RGBAColor BaseColor_Modulate_Zone0;
  float DeltaRoughness_Modulate_Zone0; // Range: -0.5 - 0.5
  float TcZoneScale_Zone0; // Range: 0 - 150
  RGBAColor BaseColor_Zone1;
  RGBAColor BaseColor_Modulate_Zone1;
  float DeltaRoughness_Modulate_Zone1; // Range: -0.5 - 0.5
  float TcZoneScale_Zone1; // Range: 0 - 150
  RGBAColor BaseColor_Zone2;
  RGBAColor BaseColor_Modulate_Zone2;
  float DeltaRoughness_Modulate_Zone2; // Range: -0.5 - 0.5
  float TcZoneScale_Zone2; // Range: 0 - 150
  RGBAColor BaseColor_Metal;
  float Roughness_Metal; // Range: 0 - 1
};

// userName: 'Shader'
// File extension: 'AdnShader.gbx'
struct CPlugAdnShader_Skin {
  CMwNod* BaseColor0;
  CMwNod* BaseColor1;
  CMwNod* Normal0;
  CMwNod* Rough0;
  RGBAColor BaseColor1_;
  uint iChannelColor1;
};

struct CPlugAnimGraphState {
  MwId Name;
  CPlugAnimNode* BaseNode;
  CPlugAnimNodeAim* AimNode;
  CPlugAnimRootYaw RootYaw;
};

struct CPlugAnimGraphTransition {
  UnknownType FromStates;
  UnknownType ExcludeFromStates;
  MwArrayInPlaceDyn<SPlugAnimGraphStateSelectCond> SelectConds;
  CPlugAnimTransition Params;
  MwId ToState;
  string CondExpr;
  uint EvalPriority;
};

struct CPlugAnimNode {
};

struct NSceneFlock_SBirdPhysicsParams {
  float PitchSpeed;
  float PitchMax;
  float YawSpeed;
  float YawMax;
  float RollSpeed;
  float RollMax;
  float PitchUpSpeed;
  float RollUpSpeed;
  float OtherEmitterBirdsInfluence;
  float BirdDistanceToSpeed;
  float GroundRepulsorDistanceToSpeed;
  float RepulsorDistanceToSpeed;
  float MainEmitterDistanceToSpeed;
  float GroundAvoidDistOffset;
};

struct CPlugAnimNodeAim : public CPlugAnimNode {
  MwId PoseGrid;
};

struct SPlugSkelSocket {
  MwId Name;
  uint16 JointIndex;
  iso4 LocInJoint;
};

struct NHmsMgrParticle_SMgr {
  const uint ModelCount;
};

struct NPlugAnim_SImportParams {
  float Scale;
  bool TransformIso4MaxToGbx;
  uint RootRotQuarterX;
  bool RemoveNamespace;
  string AnimFileImportPath;
  string SkelImportPath;
  string ClipImportPath;
  string ChannelGroupImportPath;
  string PoseGridImportPath;
  string FbxRoot;
};

struct NPlugAnim_SClipImport {
  bool Enabled;
  MwId Name;
  MwId TakeName;
  MwId ChannelGroupName;
  MwId VariantGroup;
  bool Looping;
  float WorldSpeedKmh;
  float PlaySpeed;
  float MinPlaySpeed;
  float MaxPlaySpeed;
  bool FootCasting;
  bool FootCentering;
  bool RootRotation;
  MwId DifferenceFromClip;
  MwId RootNodeName;
  bool RemoveRootMotionAndExtractWorldSpeed;
};

struct GmSurf {
  EGmSurfType GmSurfType;
  vec3 GameplayMainDir;
};

struct GmSurfPrimitive : public GmSurf {
  EPlugSurfaceMaterialId SurfaceIds_PhysicId;
  EPlugSurfaceGameplayId SurfaceIds_GameplayId;
  uint16 MaterialIndex;
};

struct GmSurfSphere : public GmSurfPrimitive {
  float Radius;
};

struct GmSurfSphereLocated : public GmSurfPrimitive {
  vec3 Center;
  float Radius;
};

struct GmSurfEllipsoid : public GmSurfPrimitive {
  vec3 Scale;
};

struct SGmSurfMultiSphereElem {
  vec3 Pos;
  float Radius;
};

struct GmSurfMultiSphere : public GmSurfPrimitive {
  MwArrayInPlaceDyn<SGmSurfMultiSphereElem> Spheres;
};

struct GmSurfVCylinder : public GmSurfPrimitive {
  float Height;
  float Radius;
};

struct GmSurfCylinder : public GmSurfPrimitive {
  float RadiusY;
  float RadiusXZ;
};

struct GmSurfBox : public GmSurfPrimitive {
  GmBoxAligned AABB;
};

struct SGmConvexFace {
  vec4 Plane;
  uint iFaceNormal;
  uint FaceNormalDir;
  MwSArray<uint> Verts;
};

struct SGmConvexEdge {
  uint Verts_0_;
  uint Verts_1_;
  uint iEdgeDir;
};

struct IGmConvexPoly {
  MwSArray<vec3> Verts;
  MwSArray<vec3> FaceNormals;
  MwSArray<vec3> EdgeDirs;
  MwSArray<SGmConvexFace> Faces;
  MwSArray<SGmConvexEdge> Edges;
};

struct GmSurfConvexPolyhedron : public GmSurfPrimitive {
  IGmConvexPoly ConvexPoly;
  GmBoxAligned AABB;
};

struct GmSurfCapsule : public GmSurfPrimitive {
  vec3 SphereCenter;
  vec3 Dir;
  float Radius;
  float Length;
};

struct GmSurfPlane : public GmSurfPrimitive {
  vec4 PlaneEq;
};

struct GmSurfMeshTri {
  uint VertIndices_0_;
  uint VertIndices_1_;
  uint VertIndices_2_;
  EPlugSurfaceMaterialId SurfaceIds_PhysicId;
  EPlugSurfaceGameplayId SurfaceIds_GameplayId;
  uint16 MaterialIndex;
};

struct GmSurfMesh : public GmSurf {
  MwFastBuffer<vec3> m_Verts;
  MwFastBuffer<GmSurfMeshTri> m_Tris;
};

struct GmSurfCompound : public GmSurf {
  MwFastBuffer<iso4> SurfLocs;
  MwFastBuffer<GmSurf*> Surfs;
  MwFastBuffer<uint16> SurfJoints;
};

struct GmSurfCompoundInstance : public GmSurf {
  GmSurfCompound* Compound;
  MwFastBuffer<iso4> SurfLocs;
};

struct GmSurfCircle : public GmSurf {
  vec3 Circle_Center;
  float Circle_Radius;
  vec3 Circle_Normal;
};

struct SPlugAnimGraphStateSelectCond {
  EPlugAnimGraphStateSelectCond Cond;
  MwId Value;
};

struct CPlugAnimTransition {
  uint Duration;
  EPlugStateMachineTransitionType Type;
  EPlugStateMachineTransitionCurve Curve;
};

struct CPlugAnimRootYaw {
  EPlugAnimRootYawType Type;
  float YawSpeedDegPerSec;
  float YawDeltaDeg;
};

struct CPlugAnimNodeJump : public CPlugAnimNode {
  CPlugAnimNodeClip Start;
  CPlugAnimNodeClip Phase1;
  CPlugAnimNodeClip Phase2;
  CPlugAnimNodeClip EndLoop;
};

struct CPlugAnimNodeSequence : public CPlugAnimNode {
  MwArrayInPlaceDyn<CPlugAnimNodeClip> Clips;
};

struct CPlugAnimNodeProceduralAttractor : public CPlugAnimNode {
  SPlugAnimNodeProceduralAttractorParams Params;
};

struct SPlugAnimNodeLocoNode {
  CPlugAnimNodeClip Clip;
  CPlugAnimNodeAim Aim;
  CPlugAnimRootYaw RootYaw;
  EPlugAnimLocoNodeType Type;
  float YawDistMinDeg;
  float YawDistMaxDeg;
  float SpeedMinKmh;
  float SpeedMaxKmh;
};

struct SPlugAnimNodeProceduralAttractorParams {
  float Clav_XAngleFromPitch;
  float Clav_ZAngleFromPitch;
  float Neck_YAngleFromPitch;
  float Spine1_YAngleFromPitch;
  float Spine_YAngleFromPitch;
  float Head_ZAngleFromPitch;
  float ForeArm_ZAngleFromPitch;
  float Hand_ZAngleFromPitch;
  float Hips_XAngleCoef;
  float Hips_YAngleCoef;
  float Hips_ZAngleCoef;
  float Hand1_XAngle;
  float Hand2_XAngle;
  float RootYawSpeedDeg;
  float AttractorYawDistMaxDeg;
};

struct CPlugAnimNodeClip : public CPlugAnimNode {
  MwId ClipName;
  uint ClipStartTime;
};

struct CPlugAnimNodeLocoGroup : public CPlugAnimNode {
  CPlugAnimTransition Transition;
  MwArrayInPlaceDyn<SPlugAnimNodeLocoNode> LocoNodes;
};

struct NPlugImageArray_SElem {
  wstring Name;
  int Priority;
  int Mask_Priority;
  vec2 Py_UV_Size__m_;
  vec2 Py_UV_Offset__m_;
  float Py_Rotation; // Range: -90 - 90
  float Py_Pxz_Blend_StartAngle; // Range: 0 - 90
  float Py_Pxz_Blend_EndAngle; // Range: 0 - 90
  vec2 Pxz_UV_Size__m_;
  vec2 Pxz_UV_Offset__m_;
  float Px_Pz_Blend_StartAngle; // Range: 0 - 90
  float Px_Pz_Blend_EndAngle; // Range: 0 - 90
};

struct NPlugAnim_SPoseGridImport {
  MwId Name;
  MwId ChannelGroupName;
  MwId RefPoseFbxName;
  MwId RefPoseTakeName;
  uint RefPoseFrame;
  UnknownType Xs;
  UnknownType Ys;
  MwFastBuffer<NPlugAnim_SPoseGridImportPose> Poses;
};

struct GmSurfSphericalShell : public GmSurfPrimitive {
  float InnerRadius;
  float OuterRadius;
  bool SkipInToOut;
};

struct CGameMenuScene : public CMwNod {
};

struct SSceneWagonState {
  float Speed;
};

struct CPlugAnimGraphNode_JointInertia : public CPlugGraphNode {
  CPlugAnimGraphNode_JointInertia();

  MwId Joint;
  string Active_Expr;
};

struct MwClassId {
  uint ClassId;
};

struct NSceneConstruction_SConstructionVis {
};

struct NSceneProp_SPropVis {
};

struct CPlugAnimGraphNode_AirTrajectoryPrediction : public CPlugGraphNode {
  CPlugAnimGraphNode_AirTrajectoryPrediction();

  string Period_Expr;
  string Max_Duration_Expr;
  MwId Next_Pos_VarName;
  MwId Next_Vel_VarName;
  MwId Time_To_Impact_VarName;
  MwId Impact_Normal_VarName;
};

struct SSceneDestructibleVisGroupState {
};

struct SSceneRopeVisEnt {
};

struct SSceneGateSpecial {
  NHmsCollision_SItem* TriggerCollItem;
  EPlugSurfaceGameplayId GameplayId;
  vec3 WorldFrontDir;
  uint iMerged;
};

struct NPlugAnim_SSkelImport {
  MwId Name;
  MwId TakeName;
  MwId RootJointName;
  string ExcludeJointMatchName;
  string IncludeOnlyJointMatchName;
  MwFastBuffer<NPlugAnim_SSkelImportJointLod> JointLods;
  UnknownType LodMaxDists;
};

struct CPlugAnimNodeBlend2d : public CPlugAnimNode {
  MwFastBuffer<float> Xs;
  MwFastBuffer<float> Ys;
  string XExpr;
  string YExpr;
  MwFastBuffer<CPlugAnimNodeClip> GridNodes;
};

struct CPlugVegetMaterialVariation : public CMwNod {
  CPlugVegetMaterialVariation();

  RGBAColor Color0;
  RGBAColor Color1;
  RGBAColor Color2;
  float Variation_max_per_vertex; // Range: 0 - 1
  float Variation_max_per_instance; // Range: 0 - 1
};

struct CPlugVegetSubSurfaceParams : public CMwNod {
  CPlugVegetSubSurfaceParams();

  bool Enable;
  float Power; // Range: 0 - 10
  float Scale; // Range: 0.1 - 2
  float Distortion; // Range: 0 - 1
  RGBAColor Data_SubSurfaceColor;
};

struct SSceneDestructibleVisState {
};

struct NPlugDyna_SConstraintModel {
  NPlugDyna::EDynaConstraintType Type;
  float Spring_Length;
  float Spring_DampingRatio;
  float Spring_FreqHz;
};

struct NPlugDyna_SForceFieldModel {
  NPlugDyna::EForceFieldType Type;
  float Radius;
  NPlugDyna_SConstraintModel ConstraintModel;
};

struct NPlugPrefab_SEntRef {
  SConstString Name;
  CSystemFidFile* const ModelFid;
  CMwNod* Model;
  SMetaPtr Params;
  GmTransQuat Location;
  uint LodGroupId;
};

struct SPlugVehicleVisStyle {
  RGBAColor Color;
  float Proba;
};

struct NHmsMgrInstDyna_SMgr {
  const uint cModel;
  const uint cInstance;
  const uint cLocalChunk;
  const uint cOrphans;
  bool Cfg_LodPerChunk_Enabled;
  float Cfg_LodPerChunk_MinDist_InChunk;
  float Cfg_LodPerChunk_DistToAdd_InChunk;
};

struct CGameManialinkNavigationScriptHandler : public CMwNod {
  CGameManialinkNavigationScriptHandler();

  bool m_SendNavigationEvent; // Maniascript
  bool m_InternalNavigationFocus; // Maniascript
  bool m_AutoNavigation; // Maniascript
  void ResetGlobalSoloGroups(); // Maniascript
  MwFastBuffer<CEventMenuNavigation*> PendingEvents; // Maniascript
  void SetFocusedControl(CGameManialinkControl* Control); // Maniascript
  void ChangeControlTarget(CGameManialinkControl* Control, string Action, string NewTargetId); // Maniascript
  CGameManialinkControl* GetControlTarget(CGameManialinkControl* Control, CGameManialinkScriptEvent::EMenuNavAction Action); // Maniascript
  void EnableGroup(string NavGroup, bool Enable); // Maniascript
  void EnableInputs(CGameManialinkControl* Control, bool Enable); // Maniascript
  CGameManialinkControl* GetFocusedControl(string NavGroup); // Maniascript
  CGameManialinkControl* GetPreviousFocusedControl(string NavGroup); // Maniascript
  void Lock(CGameManialinkControl* Control, bool Lock); // Maniascript
  bool IsLocked(CGameManialinkControl* Control); // Maniascript
  string GetControlGroup(CGameManialinkControl* Control); // Maniascript
  void SetControlGroupName(CGameManialinkControl* Control, string NavGroup); // Maniascript
  void SetGroupParent(string ChildGroup, string ParentGroup); // Maniascript
  void SetChildrenGroupParent(MwFastBuffer<wstring>& ChildrenGroups, string ParentGroup); // Maniascript
  void EnableNavigation(bool Enable); // Maniascript
  bool IsNavigationControl(CGameManialinkControl* Control); // Maniascript
  void ApplyInput(CGameManialinkFrame* Frame, CGameManialinkScriptEvent::EMenuNavAction Action); // Maniascript
  void SetGlobalSoloGroups(MwFastBuffer<wstring>& GlobalSoloGroups); // Maniascript
  void AddToGlobalSoloGroups(MwFastBuffer<wstring>& GlobalSoloGroups); // Maniascript
  bool IsGlobalSoloGroupsEmpty(); // Maniascript
  void RemoveFromGlobalSoloGroups(MwFastBuffer<wstring>& GlobalSoloGroups); // Maniascript
  bool IsInGlobalSoloGroups(string Group); // Maniascript
};

struct CScriptEvent : public CMwNod {
  CScriptEvent();

};

struct CEventMenuNavigation : public CScriptEvent {
  CEventMenuNavigation();

};

struct CEventMenuNavigationOnAction : public CEventMenuNavigation {
  CEventMenuNavigationOnAction();

  CGameManialinkControl* From; // Maniascript
  CGameManialinkControl* To; // Maniascript
  bool IsMouse; // Maniascript
  string NavGroup; // Maniascript
  enum class CGameManialinkScriptEvent::EMenuNavAction {
    Up = 0,
    Right = 1,
    Left = 2,
    Down = 3,
    Select = 4,
    Cancel = 5,
    PageUp = 6,
    PageDown = 7,
    AppMenu = 8,
    Action1 = 9,
    Action2 = 10,
    Action3 = 11,
    Action4 = 12,
    ScrollUp = 13,
    ScrollDown = 14,
  };
  CGameManialinkScriptEvent::EMenuNavAction Input; // Maniascript
};

struct CPlugVegetTreeModel : public CMwNod {
  CPlugVegetTreeModel();

  NPlugVeget_STreeModel Data;
};

struct NPlugFxAnimFromTexture1dArray_SElem {
  CPlugFileImg* Image;
  float PeriodSc_Min;
  float PeriodSc_Max;
};

// userName: 'PlugGrassMatterArray'
// File extension: 'PlugGrassMatterArray.Gbx'
struct CPlugGrassMatterArray : public CPlug {
  CPlugGrassMatterArray();

  NPlugGrass_SMatterArray MatterArray;
};

// userName: 'PlugPainterLayer'
// File extension: 'PlugPainterLayer.Gbx'
struct NPlugPainterLayer_SList : public CPlug {
  NPlugPainterLayer_SList();

  MwFastBuffer<NPlugPainterLayer_SElem*> m_LayersList;
};

struct CPlugGraphNode : public CMwNod {
  CPlugGraphNode();

  MwId Name;
};

struct CPlugAnimGraphNode_JointRotateFrom : public CPlugGraphNode {
  CPlugAnimGraphNode_JointRotateFrom();

  MwId Joint;
  MwId Source_Joint;
  vec3 XYZ_Coef;
};

struct CPlugAnimGraphNode_JointRotate : public CPlugGraphNode {
  CPlugAnimGraphNode_JointRotate();

  MwId Joint;
  string Rotation_Expr;
  bool Vector;
  vec3 Axis;
  bool Apply_On_X;
  bool Apply_On_Y;
  bool Apply_On_Z;
  bool Force_Rotation_Order;
  GmFunc::ERotationOrder Rotation_Order;
  NPlugAnim::EGraphNodeOperationType Operator;
  NPlugAnim::EGraphNodeAxisWorkspace Axis_Workspace;
};

struct CPlugAnimGraphNode_JointAlignTo : public CPlugGraphNode {
  CPlugAnimGraphNode_JointAlignTo();

  MwId Joint;
  MwId Align_To_Joint;
  bool Scale;
  bool Reciprocally_Aligned;
  string Rotation_Expr;
};

struct CPlugAnimGraphNode_JointTranslate : public CPlugGraphNode {
  CPlugAnimGraphNode_JointTranslate();

  MwId Joint;
  string Translation_Expr;
  bool Vector;
  vec3 Axis;
  bool Apply_On_X;
  bool Apply_On_Y;
  bool Apply_On_Z;
  NPlugAnim::EGraphNodeOperationType Operator;
  NPlugAnim::EGraphNodeAxisWorkspace Axis_Workspace;
};

struct CPlugAnimGraphNode_JointIK2 : public CPlugGraphNode {
  CPlugAnimGraphNode_JointIK2();

  MwId Begin_Joint;
  MwId Middle_Joint;
  MwId End_Joint;
  string IK2_Weight_Expr;
  bool Keep_End_Joint_Global_Rotation;
  NPlugAnim::EGraphNodeIK2SolverMode SolverMode;
  string PoleVector_Expr;
  string SolverAuto_Weight_Expr;
  bool SoftIK;
  string SoftIK_Weight_Expr;
};

struct CPlugAnimGraphNode_JointKeepRefGlobalRot : public CPlugGraphNode {
  CPlugAnimGraphNode_JointKeepRefGlobalRot();

  MwId Joint;
  bool UpdateChilds;
};

struct CPlugAnimGraphNode_JointTranslateDistConstraint : public CPlugGraphNode {
  CPlugAnimGraphNode_JointTranslateDistConstraint();

  MwId Joint;
  MwId Joint_Dist_Constraint_A;
  MwId Joint_Dist_Constraint_B;
};

struct CPlugAnimGraphNode_LocalToGlobal : public CPlugGraphNode {
  CPlugAnimGraphNode_LocalToGlobal();

};

struct CPlugAnimGraphNode_Blend : public CPlugGraphNode {
  CPlugAnimGraphNode_Blend();

  string Weight_Expr;
  bool Is_Switch;
  bool Is_Normalized_Uniform;
  uint Input_Count;
  UnknownType Input_Weights;
};

struct CPlugAnimGraphNode_ClipPlay : public CPlugGraphNode {
  CPlugAnimGraphNode_ClipPlay();

  MwId Clip_Name;
  MwId ClipGroup_Name;
  string Play_Time_Expr;
  bool Play_Time_Is_Normalized;
  string Play_Speed_Expr;
  string Start_Value_On_Activate__Expr_;
  bool Start_Value_Is_Normalized;
  bool Reset_On_Activate;
  NPlugAnim::EClipPlayAdditiveType Additive_Type;
};

struct CPlugAnimGraphNode_Blend2d : public CPlugGraphNode {
  CPlugAnimGraphNode_Blend2d();

};

struct CPlugAnimGraphNode_Graph : public CPlugGraphNode_Graph {
  CPlugAnimGraphNode_Graph();

  int8 Max_Lod;
  string Skip_Expr;
  MwFastBuffer<SPlugAnimClipGroup> Clip_Groups;
  MwFastBuffer<string> Macros;
};

struct CPlugAnimGraphNode_StateMachine : public CPlugGraphNode_StateMachine {
  CPlugAnimGraphNode_StateMachine();

  MwFastBuffer<SPlugStateMachine_State> States;
  MwFastBuffer<SPlugStateMachine_Transition> Transitions;
  bool Reset_To_Default_State_On_Activate;
};

struct CPlugAnimGraphNode_LodSwitch : public CPlugGraphNode {
  CPlugAnimGraphNode_LodSwitch();

};

struct CPlugAnimGraphNode_Group : public CPlugGraphNode_Group {
  CPlugAnimGraphNode_Group();

};

struct CPlugAnimGraphNode_RefLocalPose : public CPlugGraphNode {
  CPlugAnimGraphNode_RefLocalPose();

};

struct CPlugAnimGraphNode_GraphOutput : public CPlugGraphNode {
  CPlugAnimGraphNode_GraphOutput();

};

struct CPlugVFXNode : public CMwNod {
  CPlugVFXNode();

  MwId Name;
  bool Enabled;
};

struct CPlugVFXNode_Emit : public CPlugVFXNode {
  CPlugVFXNode_Emit();

  MwId JointName;
  vec3 Normal;
  string PositionExpr;
  string ConditionExpr;
  bool Force_activation;
};

struct CPlugVFXNode_EmitterModel : public CPlugVFXNode {
  CPlugVFXNode_EmitterModel();

  CPlugParticleEmitterModel* BaseModel;
  bool EnablePropagation;
  uint PropagationParams_cGenPoint;
  float Propagation_Con_eHalfAngle__Rad_;
  float PropagationParams_MaxDist;
};

struct CPlugVFXNode_Graph : public CPlugVFXNode {
  CPlugVFXNode_Graph();

  MwFastBuffer<CPlugVFXNode*> Nodes;
  MwFastBuffer<SPlugVFXNodeConnection> Connections;
};

struct CPlugVFXNode_SubEmitterModel : public CPlugVFXNode {
  CPlugVFXNode_SubEmitterModel();

  CPlugParticleGpuSpawn* Spawn;
  CPlugParticleGpuModel* Model;
};

struct CPlugAnimGraphNode_ExtractMotion : public CPlugGraphNode {
  CPlugAnimGraphNode_ExtractMotion();

  MwId Joint_Name;
  NPlugAnim::EExtractMotionMode Mode;
  MwId Trans_VarName;
  MwId Rot_VarName;
  bool Force_Rotation_Order;
  GmFunc::ERotationOrder Rotation_Order;
  bool Remove_From_Pose;
  bool Extract_From_Linked_ModeInst;
  bool Extract_From_A_pose;
  MwId Status_VarName;
};

struct CPlugAnimGraphNode_RefGlobalPose : public CPlugGraphNode {
  CPlugAnimGraphNode_RefGlobalPose();

};

struct NPlugItemPlacement_SPlacement {
  uint iLayout;
  MwSArray<NPlugItemPlacement_SPlacementOption> Options;
};

struct CPlugAnimGraphNode_SetVar : public CPlugGraphNode {
  CPlugAnimGraphNode_SetVar();

  string Active_Expr;
  bool Set_To_Default_Value_When_Not_Active;
  bool Set_Only_On_Activation;
  MwId Variable_name;
  string Variable_Expr;
  bool Update_Before_Recursion_Call;
};

struct CPlugAnimGraphNode_ClipGroupPlay : public CPlugGraphNode {
  CPlugAnimGraphNode_ClipGroupPlay();

  MwFastBuffer<MwId> Clip_Names;
  string Play_Speed_Expr;
  bool Reset_On_Activate;
  bool Loop_On_Last_Clip;
};

struct CPlugVFXNode_VortexEmitterModel : public CPlugVFXNode {
  CPlugVFXNode_VortexEmitterModel();

  string Radius;
  string cVortexPerCircle;
  string Strength;
  uint LifeTime;
  string SpeedCoef;
  float AxisRandomness;
  uint EmitPeriod;
  string Normal;
  float Loc_cPixelMin;
};

struct CPlugVFXNode_EmissionGroup : public CPlugVFXNode {
  CPlugVFXNode_EmissionGroup();

  vec4 ColPlaneEq;
  bool InteractWithAllGroups;
};

struct CPlugAnimGraphNode_SetSkel : public CPlugGraphNode {
  CPlugAnimGraphNode_SetSkel();

  MwId Skel_Name;
};

struct CPlugAnimGraphNode_LayeredBlend : public CPlugGraphNode {
  CPlugAnimGraphNode_LayeredBlend();

  MwId ChannelGroup_Name;
  string Layer_Weight_Expr;
  NPlugAnim::ELayeredBlendType Blend_Type;
  NPlugAnim::ELayeredBlendContext Trans;
  NPlugAnim::ELayeredBlendContext Rot;
};

struct CPlugAnimGraphNode_GlobalToLocal : public CPlugGraphNode {
  CPlugAnimGraphNode_GlobalToLocal();

};

struct CPlugAnimGraphNode_PoseGrid : public CPlugGraphNode {
  CPlugAnimGraphNode_PoseGrid();

  MwId PoseGrid_Name;
  string Weight_Expr;
  string X_Expr;
  string Y_Expr;
};

struct CPlugAnimGraphNode_GraphInput : public CPlugGraphNode {
  CPlugAnimGraphNode_GraphInput();

};

// userName: 'AnimSceneClip'
// File extension: 'AnimSceneClip.Gbx'
struct NPlugAnim_SSceneClip : public CMwNod {
  NPlugAnim_SSceneClip();

  MwSArray<NPlugAnim_SEntTrack> EntTracks;
  MwSArray<NPlugAnim_SEntClip*> EntClips;
  int Duration;
};

struct NPlugDynaObjectModel_SInstanceParams {
  float PeriodSc;
  float PeriodScMax;
  uint TextureId;
  bool IsKinematic;
  bool CastStaticShadow;
  float Phase01;
  float Phase01Max;
};

struct NSceneLight_SLight {
  vec3 Pos;
  RGBAColor Color;
};

// userName: 'NPlugItemPlacement::SGroups'
// File extension: 'ItemPlacementGroups.Gbx'
struct NPlugItemPlacement_SGroups : public CMwNod {
  NPlugItemPlacement_SGroups();

  MwFastBuffer<NPlugItemPlacement_SIdGroup> IdGroups;
  MwFastBuffer<NPlugItemPlacement_SSizeGroup> SizeGroups;
};

struct CPlugAnimGraphNode_JointLock : public CPlugGraphNode {
  CPlugAnimGraphNode_JointLock();

  MwId Joint;
  NPlugAnim::EGraphNodeLockWorkspace Lock_Workspace;
  MwId Custom_Joint;
  NPlugAnim::EGraphNodeLockType Lock_Type;
  vec3 Weight_Axis_Translate;
  vec3 Weight_Axis_Rotate;
  bool Force_Rotation_Order;
  GmFunc::ERotationOrder Rotation_Order;
  string Weight_Expr;
  string Active_Expr;
};

struct NSceneDestructibleVis_SMgr {
};

// userName: 'VariantList'
// File extension: 'VariantList.Gbx'
struct NPlugItem_SVariantList : public CMwNod {
  NPlugItem_SVariantList();

  MwSArray<NPlugItem_SVariant> Variants;
};

struct CPlugGraphNode_Graph : public CPlugGraphNode {
  CPlugGraphNode_Graph();

  const MwFastBuffer<CPlugGraphNode*> Nodes;
  const MwFastBuffer<SPlugGraphNodeConnection> Connections;
  MwFastBuffer<SPlugGraphVar> Vars;
};

struct CPlugGraphNode_StateMachine : public CPlugGraphNode {
  CPlugGraphNode_StateMachine();

};

struct CPlugGraphNode_Group : public CPlugGraphNode {
  CPlugGraphNode_Group();

  string Text;
  RGBAColor Color;
  bool MoveContent;
  vec2 Size;
};

struct CPlugFxSystemNode : public CMwNod {
  CPlugFxSystemNode();

  MwId Name;
};

struct CPlugFxSystemNode_Parallel : public CPlugFxSystemNode {
  CPlugFxSystemNode_Parallel();

  const MwFastBuffer<CPlugFxSystemNode*> Children;
};

struct CPlugFxSystemNode_Condition : public CPlugFxSystemNode {
  CPlugFxSystemNode_Condition();

  string ConditionExpr;
  CPlugFxSystemNode* const Child;
};

struct CPlugFxSystemNode_ParticleEmitter : public CPlugFxSystemNode {
  CPlugFxSystemNode_ParticleEmitter();

  CPlugParticleEmitterModel* Model;
  MwId JointName;
  bool DOVAndUpAreLocalSpace;
  string LocalOffsetExpr;
  string WorldOffsetExpr;
  string UpExpr;
  string DOVExpr;
  string LinearVelInWExpr;
  string SpawnFreqModifierExpr;
  string ScaleExpr;
  string OpacityExpr;
  string LAmbientExpr;
  string WaterTopExpr;
  string LinearHue01;
  string HueLightness;
};

struct CPlugFxSystemNode_SubFxSystem : public CPlugFxSystemNode {
  CPlugFxSystemNode_SubFxSystem();

  CPlugFxSystem* FxSystem;
};

struct CPlugFxSystemNode_UpdateVar : public CPlugFxSystemNode {
  CPlugFxSystemNode_UpdateVar();

  MwId VarName;
  bool ResetToDefaultIfInactive;
  string UpdateVarExpr;
};

struct CPlugFxSystemNode_SoundEmitter : public CPlugFxSystemNode {
  CPlugFxSystemNode_SoundEmitter();

  CPlugSound* Model;
  MwId JointName;
  EAudioBalanceGroup AudioBalanceGroup;
  string AudioGroupHandleExpr;
  bool PlayOnce;
  string VolumeExpr;
  float FadeOffDuration;
  string PitchExpr;
  string Surface_SurfaceIdExpr;
  string Surface_SpeedKmhExpr;
  string Surface_SkidIntensityExpr;
  string Surface_SkidSpeedKmhExpr;
};

struct NPlugDyna_SPrefabConstraintParams {
  uint Ent1;
  uint Ent2;
  vec3 Pos1;
  vec3 Pos2;
};

// userName: 'KinematicConstraint'
// File extension: 'KinematicConstraint.Gbx'
struct NPlugDyna_SKinematicConstraint : public CMwNod {
  NPlugDyna_SKinematicConstraint();

  NPlugDyna_SAnimFunc01 TransAnimFunc;
  NPlugDyna::EAxis TransAxis;
  float TransMin;
  float TransMax;
  NPlugDyna_SAnimFunc01 RotAnimFunc;
  NPlugDyna::EAxis RotAxis;
  float AngleMinDeg;
  float AngleMaxDeg;
  NPlugDyna::EShaderTcType ShaderTcType;
  NPlugDyna_SAnimFuncNat ShaderTcAnimFunc;
  NFuncShaderLayerUV_STransSubTextureIn ShaderTcData_TransSub;
};

struct NSceneDynaVis_SMgr {
};

struct CPlugAnimGraphNode_DebugHelper : public CPlugGraphNode {
  CPlugAnimGraphNode_DebugHelper();

  string Point_A_Expr;
  string Point_B_Expr;
};

struct CPlugAnimGraphNode_AssertVar : public CPlugGraphNode {
  CPlugAnimGraphNode_AssertVar();

  string Variable;
  string Expected_value;
  NPlugAnim::EGraphNodeAssertVarType Type_of_the_var_to_assert;
  string Epsilon;
};

struct CPlugAnimGraphNode_Funnel : public CPlugGraphNode {
  CPlugAnimGraphNode_Funnel();

  NPlugAnim::EFunnelType Funnel;
};

struct CPlugAnimGraphNode_ExtractUnit : public CPlugGraphNode {
  CPlugAnimGraphNode_ExtractUnit();

  MwId Joint_Name;
  NPlugAnim::EExtractMotionMode Mode;
  NPlugAnim::EExtractUnitTarget Motion_targeted;
  NPlugAnim::EExtractUnitAxis Axis;
  bool Force_Rotation_Order;
  GmFunc::ERotationOrder Rotation_Order;
  MwId VarName;
  MwId Status_VarName;
  bool Remove_From_Pose;
  bool Extract_From_Linked_ModelInst;
  bool Extract_From_A_pose;
};

struct SGameCtnBlockInfoVariant_UsedData {
  void UpdateLists();
  void ClearLists();
  MwFastBuffer<CGameCtnBlockInfoClip*> UsedClips;
  MwFastBuffer<CGameCtnBlockInfo*> UsedPillars;
  MwFastBuffer<CGameCtnBlockInfoPylon*> UsedPylons;
};

struct CPlugAnimGraphNode_SetJointExpr : public CPlugGraphNode {
  CPlugAnimGraphNode_SetJointExpr();

  MwId Joint_Expr_Group_Name;
};

struct CPlugAnimGraphNode_JointTransConstraint : public CPlugGraphNode {
  CPlugAnimGraphNode_JointTransConstraint();

  MwId Joint;
  UnknownType Array_of_Joints;
  UnknownType Array_of_Weights;
  uint Input_Count;
  uint Input_Count;
};

struct CPlugAnimGraphNode_JointRotConstraint : public CPlugGraphNode {
  CPlugAnimGraphNode_JointRotConstraint();

  MwId Joint;
  UnknownType Array_of_Joints;
  UnknownType Array_of_Weights;
  uint Input_Count;
  uint Input_Count;
};

struct NPlugItemPlacement_SPlacementGroup {
  MwSArray<NPlugItemPlacement_SPlacement> Placements;
};

struct NPlugStaticObjectModel_SInstanceParams {
  float Phase01;
};

struct CPlugShaderCBufferStatic : public CMwNod {
  CPlugShaderCBufferStatic();

};

struct NGameArenaState_SGameState {
  SGamePlaygroundUIConfig::EUISequence UISequence;
  vec3 DecorationColorHSL;
};

struct SPlugVehiclePhyRestStateValues {
  float DamperRestStateValue;
  float OffsetFromShapeScale;
};

struct CPlugAnimGraphNode_AvatarV3_Global : public CPlugGraphNode {
  CPlugAnimGraphNode_AvatarV3_Global();

  NPlugAnim_SAvatarAnimV3_Model_Global Model;
};

struct CPlugAnimGraphNode_AvatarV3_Locomotion : public CPlugGraphNode {
  CPlugAnimGraphNode_AvatarV3_Locomotion();

};

struct CPlugAnimGraphNode_AvatarV3_Jump : public CPlugGraphNode {
  CPlugAnimGraphNode_AvatarV3_Jump();

};

struct CPlugAnimGraphNode_AvatarV3_Swim : public CPlugGraphNode {
  CPlugAnimGraphNode_AvatarV3_Swim();

};

struct CPlugAnimGraphNode_AvatarV3_Idle : public CPlugGraphNode {
  CPlugAnimGraphNode_AvatarV3_Idle();

};

struct CPlugAnimGraphNode_AvatarV0 : public CPlugGraphNode {
  CPlugAnimGraphNode_AvatarV0();

  NPlugAnim_SAvatarAnimV0_Model Model;
};

struct CPlugAnimGraphNode_AvatarPoseEditor : public CPlugGraphNode {
  CPlugAnimGraphNode_AvatarPoseEditor();

  NPlugAnim_SPoseEditor_Model Model;
};

struct CPlugAnimGraphNode_AvatarV3_Seated : public CPlugGraphNode {
  CPlugAnimGraphNode_AvatarV3_Seated();

};

struct CPlugAnimGraphNode_AvatarV3_Resting : public CPlugGraphNode {
  CPlugAnimGraphNode_AvatarV3_Resting();

};

// userName: 'World'
// File extension: 'World.Gbx'
struct SSmWorldSaveFile : public CMwNod {
  SSmWorldSaveFile();

  wstring ChallengeUId;
  wstring ModeName;
  const uint DelayedSceneSnapshotSize;
  const uint ReplicaSnapshotSize;
};

struct CPlugAnimGraphNode_Avatar_Climb : public CPlugGraphNode {
  CPlugAnimGraphNode_Avatar_Climb();

};

} // namespace Meta

namespace MetaNotPersistent {

struct NHmsZone_SPImp {
  MwFastBuffer<NHmsZone_SLightStatic> LightStatic_Lights;
  MwFastBuffer<NHmsZone_SLightDyna2> LightDynamic_Lights;
  MwFastBuffer<NHmsZone_SLightDynaFrustum> LightDynamic_Frustum_Lights;
  MwFastBuffer<NHmsZone_SLightDir> LightDir_Lights;
  GxLightAmbient* LightAmbient;
};

struct CHmsZoneVPacker {
  nat3 m_c3Cell;
  const float __used_cells;
  const float __objects_usedCell;
  const float __packs_usedCell;
  const float __KVert_pack;
  const float __outside_objects;
  const uint ObjectInsides;
  const uint ObjectOutsides;
  const uint UnpackableVSpace;
  void _BBox_Update_();
};

struct SHmsZoneVisionCst {
  SHmsPostFxState PostFx;
  NSysCfgVision_SNvHBAO* Option_NvHBAO;
  float m_FxSmSelfIllumScaleGamePlay;
  float PlugFxHdrScalesT3_ScaleBlockEnergy;
  float PlugFxHdrScalesT3_ScaleBlockSelfIllum;
  float PlugFxHdrScalesT3_ScaleParticle;
  float PlugFxHdrScalesT3_ScalePlayer;
  float LocalLightScale;
  bool SpriteSortIsRadial;
  float m_SmOffZoneRadius_Radius; // Range: 8 - 32
  vec2 m_SmOffZoneRadius_CenterXZ;
  CHmsMoodBlender* MoodBlender;
  float FxDayTime_TimeFactor;
  float FxDayTime_NightOnlySI_GlobalScale;
  NHmsLightMapBlender_SMgr* LightmapBlender;
};

struct NPlugBitmap_SAtlasCubeIn {
  uint c2Pixel_x;
  uint c2Pixel_y;
  uint cSkipMip_ImageId; // Range: 1 - 4
  MwFastBuffer<NPlugBitmap_SAtlasCubeInElem> Elems;
};

struct ISceneVis {
  IScenePhy* ScenePhy;
  IScenePhy* ServerScenePhy;
  CScene* HackScene;
  CSceneFxMgr* SceneFxMgr;
  float CameraFarZ;
  vec3 CameraClearColor;
  MwArrayInPlaceDyn<SMetaPtr> MgrMetaPtrs;
};

struct IScenePhy {
  UnnamedType const CharPhy;
  NSceneDyna_SMgr* const Dyna;
  NHmsCollision_SMgr* const Collision;
};

struct NSceneWeather_SMgr : public CMwNod {
  NSceneWeather_SMgr();

  CPlugFxLightning* FxLightning;
  CMwNod* CloudSystem;
  float Params_SiteLatitude; // Range: -90 - 90
  float TimeRemapped; // Range: 0 - 1
  NSceneWeather::EClearMode ClearMode;
  float Params_SunIntensity;
  float Params_MoonIntensity;
  bool Params_EnableUpdate;
  bool EnableMoodArray;
};

struct NSceneVehiclePhy_SStuntStatus {
  uint LatestFigureTime; // Maniascript
  NSceneVehiclePhy_SStuntFigure LatestFigure; // Maniascript
  bool IsInFigure; // Maniascript
  bool IsNoAirControl; // Maniascript
  bool IsEpicAirControl; // Maniascript
  bool IsMasterAirControl; // Maniascript
  uint8 ChainCounter; // Maniascript
  uint16 ChainDelay; // Maniascript
  const uint TimeLeftForStuntCombo; // Maniascript
  const uint TimeElapsedSinceLastStunt; // Maniascript
};

struct NSceneVehiclePhy_SStuntFigure {
  enum class NSceneVehiclePhy_SStuntFigure::EStuntName {
    None = 0,
    BasicJump = 1,
    Flip = 2,
    Backflip = 3,
    Spin = 4,
    Aerial = 5,
    AlleyOop = 6,
    Roll = 7,
    Corkscrew = 8,
    SpinOff = 9,
    Rodeo = 10,
    FlipFlap = 11,
    Twister = 12,
    FreeStyle = 13,
    SpinningChaos = 14,
    FlippingMix = 15,
    RollingMadness = 16,
  };
  NSceneVehiclePhy_SStuntFigure::EStuntName Name; // Maniascript
  uint8 Combo; // Maniascript
  uint16 Angle; // Maniascript
  uint16 Points; // Maniascript
  float Factor; // Maniascript
  bool StraightLanding; // Maniascript
  bool ReverseLanding; // Maniascript
  bool PerfectLanding; // Maniascript
  bool MasterJump; // Maniascript
  bool MasterLanding; // Maniascript
  bool EpicLanding; // Maniascript
  bool Wreck; // Maniascript
};

struct NGameLoadProgress_SMgr : public CMwNod {
  NGameLoadProgress_SMgr();

  bool HoldScreenGet_DownloadData;
  bool HoldScreenGet_ChallengeData;
  bool HoldScreenGet_PreparePlayground;
  bool HoldScreenGet_WaitKeyPress;
  bool HoldScreenGet_RulesScript;
  bool HoldScreenGet_EditorScript;
  NGameLoadProgress::EState State;
  CControlFrame* FrameManialink;
  CScene2d* Overlay;
};

struct NGameScriptChat_SManager {
  MwFastBuffer<NGameScriptChat_SContext*> Contextes; // Maniascript
};

struct NGameHud3d_SMgr {
  SGameHud3dParams Params;
};

struct NGameMenuSkinChooser_SMgr : public CMwNod {
  NGameMenuSkinChooser_SMgr();

  CGameCtnMenuProfileScene* ProfileScene;
  CPlugCamControlModel* CamModel;
  string NextSkinOptions;
};

struct CWebServicesTaskResult_GhostDriver_UploadLimits : public CWebServicesTaskResult {
  MwFastBuffer<SWebServicesTaskResult_GhostDriver_UploadLimit> Limits; // Maniascript
};

struct CWebServicesTaskResult_GhostDriver_Download : public CWebServicesTaskResult {
  MwFastBuffer<SWebServicesTaskResult_GhostDriver_Download_Team> Teams; // Maniascript
};

struct NPlugMaterial_SWater {
};

struct NGameCollection_SCustomizableDeco {
  NPlugSkin_SCustomizableBitmap DecalSponsor1x1Big;
  NPlugSkin_SCustomizableBitmap DecalSponsor4x1;
  NPlugSkin_SCustomizableBitmap DecorationScreen16x9;
  NPlugSkin_SCustomizableBitmap DecorationScreen8x1;
  NPlugSkin_SCustomizableBitmap DecorationScreen16x1;
};

struct CNotification : public CMwNod {
  CNotification();

  const string Type; // Maniascript
};

struct CNotification_Prestige : public CNotification {
  CNotification_Prestige();

};

struct CNotification_Squad : public CNotification {
  CNotification_Squad();

};

struct CNotification_PrestigeEarned : public CNotification_Prestige {
  CNotification_PrestigeEarned();

  const string CategoryType; // Maniascript
  const uint CategoryLevel; // Maniascript
  const NWebServicesPrestige::EPrestigeMode Mode; // Maniascript
  const string PrestigeId; // Maniascript
  const uint PrestigeLevel; // Maniascript
  const wstring RewardDisplayName; // Maniascript
  const string RewardFileUrl; // Maniascript
  const string RewardThumbnailUrl; // Maniascript
  const string RewardType; // Maniascript
  const string SkinOptions; // Maniascript
  const uint StatCurrentValue; // Maniascript
  const uint StatValueForNextLevel; // Maniascript
  const uint StatValueForCurrentLevel; // Maniascript
  const uint TimeStamp; // Maniascript
  const uint Year; // Maniascript
};

struct CNotification_SquadDeleted : public CNotification_Squad {
  CNotification_SquadDeleted();

  const string SquadId; // Maniascript
  const string SquadType; // Maniascript
  const uint TimeStamp; // Maniascript
};

struct CNotification_SquadInvitationAccepted : public CNotification_Squad {
  CNotification_SquadInvitationAccepted();

  const string InvitedAccountId; // Maniascript
  const string InvitedCountryFlagUrl; // Maniascript
  const wstring InvitedDisplayName; // Maniascript
  const bool InvitedIsCommunicationRestricted; // Maniascript
  const bool InvitedIsFirstPartyDisplayName; // Maniascript
  const string InvitedSkinOptions; // Maniascript
  MwSArray<CSkinInfo> InvitedSkinList; // Maniascript
  const string InvitedWebServicesUserId; // Maniascript
  const string SquadId; // Maniascript
  const string SquadType; // Maniascript
  const uint TimeStamp; // Maniascript
};

struct CNotification_SquadInvitationAdded : public CNotification_Squad {
  CNotification_SquadInvitationAdded();

  const string InvitedAccountId; // Maniascript
  const string InvitedCountryFlagUrl; // Maniascript
  const wstring InvitedDisplayName; // Maniascript
  const bool InvitedIsFirstPartyDisplayName; // Maniascript
  const string InvitedWebServicesUserId; // Maniascript
  const string InviterAccountId; // Maniascript
  const string InviterCountryFlagUrl; // Maniascript
  const wstring InviterDisplayName; // Maniascript
  const bool InviterIsFirstPartyDisplayName; // Maniascript
  const string InviterWebServicesUserId; // Maniascript
  const string SquadId; // Maniascript
  const string SquadType; // Maniascript
  const uint TimeStamp; // Maniascript
};

struct CNotification_SquadInvitationCanceled : public CNotification_Squad {
  CNotification_SquadInvitationCanceled();

  const string CancelerAccountId; // Maniascript
  const wstring CancelerCountryFlagUrl; // Maniascript
  const wstring CancelerDisplayName; // Maniascript
  const bool CancelerIsFirstPartyDisplayName; // Maniascript
  const string CancelerWebServicesUserId; // Maniascript
  const string InvitedAccountId; // Maniascript
  const string InvitedCountryFlagUrl; // Maniascript
  const wstring InvitedDisplayName; // Maniascript
  const bool InvitedIsFirstPartyDisplayName; // Maniascript
  const string InvitedWebServicesUserId; // Maniascript
  const string SquadId; // Maniascript
  const string SquadType; // Maniascript
  const uint TimeStamp; // Maniascript
};

struct CNotification_SquadInvitationCanceledForExitingPlayer : public CNotification_Squad {
  CNotification_SquadInvitationCanceledForExitingPlayer();

  const string InvitedAccountId; // Maniascript
  const string InvitedCountryFlagUrl; // Maniascript
  const wstring InvitedDisplayName; // Maniascript
  const bool InvitedIsFirstPartyDisplayName; // Maniascript
  const string InvitedWebServicesUserId; // Maniascript
  const string SquadId; // Maniascript
  const string SquadType; // Maniascript
  const uint TimeStamp; // Maniascript
};

struct CNotification_SquadInvitationCanceledForFullSquad : public CNotification_Squad {
  CNotification_SquadInvitationCanceledForFullSquad();

  const string InvitedAccountId; // Maniascript
  const string InvitedCountryFlagUrl; // Maniascript
  const wstring InvitedDisplayName; // Maniascript
  const bool InvitedIsFirstPartyDisplayName; // Maniascript
  const string InvitedWebServicesUserId; // Maniascript
  const string SquadId; // Maniascript
  const string SquadType; // Maniascript
  const uint TimeStamp; // Maniascript
};

struct CNotification_SquadInvitationDeclined : public CNotification_Squad {
  CNotification_SquadInvitationDeclined();

  const string InvitedAccountId; // Maniascript
  const string InvitedCountryFlagUrl; // Maniascript
  const wstring InvitedDisplayName; // Maniascript
  const bool InvitedIsFirstPartyDisplayName; // Maniascript
  const string InvitedWebServicesUserId; // Maniascript
  const string SquadId; // Maniascript
  const string SquadType; // Maniascript
  const uint TimeStamp; // Maniascript
};

struct CNotification_SquadInvitationReceived : public CNotification_Squad {
  CNotification_SquadInvitationReceived();

  const string InviterAccountId; // Maniascript
  const string InviterCountryFlagUrl; // Maniascript
  const wstring InviterDisplayName; // Maniascript
  const bool InviterIsFirstPartyDisplayName; // Maniascript
  const string InviterWebServicesUserId; // Maniascript
  const string SquadId; // Maniascript
  const string SquadType; // Maniascript
  const uint TimeStamp; // Maniascript
};

struct CNotification_SquadLockStateUpdated : public CNotification_Squad {
  CNotification_SquadLockStateUpdated();

  const string SquadId; // Maniascript
  const string SquadType; // Maniascript
  const bool IsLocked; // Maniascript
  const uint TimeStamp; // Maniascript
};

struct CNotification_SquadMemberAdded : public CNotification_Squad {
  CNotification_SquadMemberAdded();

  const string MemberAccountId; // Maniascript
  const string MemberCountryFlagUrl; // Maniascript
  const wstring MemberDisplayName; // Maniascript
  const bool MemberIsCommunicationRestricted; // Maniascript
  const bool MemberIsFirstPartyDisplayName; // Maniascript
  const string MemberSkinOptions; // Maniascript
  MwSArray<CSkinInfo> MemberSkinList; // Maniascript
  const string MemberWebServicesUserId; // Maniascript
  const string SquadId; // Maniascript
  const string SquadType; // Maniascript
  const uint TimeStamp; // Maniascript
};

struct CNotification_SquadMemberKicked : public CNotification_Squad {
  CNotification_SquadMemberKicked();

  const string KickerAccountId; // Maniascript
  const string KickerCountryFlagUrl; // Maniascript
  const wstring KickerDisplayName; // Maniascript
  const bool KickerIsFirstPartyDisplayName; // Maniascript
  const string KickerWebServicesUserId; // Maniascript
  const string MemberAccountId; // Maniascript
  const string MemberCountryFlagUrl; // Maniascript
  const wstring MemberDisplayName; // Maniascript
  const bool MemberIsFirstPartyDisplayName; // Maniascript
  const string MemberWebServicesUserId; // Maniascript
  const string SquadId; // Maniascript
  const string SquadType; // Maniascript
  const uint TimeStamp; // Maniascript
};

struct CNotification_SquadMemberRemoved : public CNotification_Squad {
  CNotification_SquadMemberRemoved();

  const string MemberAccountId; // Maniascript
  const string MemberCountryFlagUrl; // Maniascript
  const wstring MemberDisplayName; // Maniascript
  const bool MemberIsFirstPartyDisplayName; // Maniascript
  const string MemberWebServicesUserId; // Maniascript
  const string SquadId; // Maniascript
  const string SquadType; // Maniascript
  const uint TimeStamp; // Maniascript
};

struct CNotification_SquadUpdated : public CNotification_Squad {
  CNotification_SquadUpdated();

  const string SquadId; // Maniascript
  const string SquadType; // Maniascript
  const string LeaderAccountId; // Maniascript
  const string LeaderWebServicesUserId; // Maniascript
  const bool WasForcedToLeaveDueToCrossPlaySetting; // Maniascript
  const uint TimeStamp; // Maniascript
};

struct CSeasonMapInfo : public CMwNod {
  CSeasonMapInfo();

  const string MapId; // Maniascript
  const MwId MapUid; // Maniascript
  const uint BronzeScore; // Maniascript
  const uint SilverScore; // Maniascript
  const uint GoldScore; // Maniascript
  const uint AuthorScore; // Maniascript
  const uint TimeStamp; // Maniascript
};

struct CSeason : public CMwNod {
  CSeason();

  const string CreatorAccountId; // Maniascript
  const string CreatorWebServicesUserId; // Maniascript
  const string GameMode; // Maniascript
  const string GameModeCustomData; // Maniascript
  const string MapRecordType; // Maniascript
  const string Id; // Maniascript
  const wstring Name; // Maniascript
  const uint CreationTimeStamp; // Maniascript
  const uint StartTimeStamp; // Maniascript
  const uint EndTimeStamp; // Maniascript
  MwSArray<CSeasonMapInfo> MapInfoList; // Maniascript
};

struct CAccountTrophyGain : public CMwNod {
  CAccountTrophyGain();

  const string AccountId; // Maniascript
  const string WebServicesUserId; // Maniascript
  const uint T1Count; // Maniascript
  const uint T2Count; // Maniascript
  const uint T3Count; // Maniascript
  const uint T4Count; // Maniascript
  const uint T5Count; // Maniascript
  const uint T6Count; // Maniascript
  const uint T7Count; // Maniascript
  const uint T8Count; // Maniascript
  const uint T9Count; // Maniascript
  const uint TimeStamp; // Maniascript
};

struct CAccountTrophyGainForHistory : public CAccountTrophyGain {
  CAccountTrophyGainForHistory();

  CTrophyAchievement* TrophyAchievement; // Maniascript
};

struct CAccountTrophyGainForHistory_CompetitionMatch : public CAccountTrophyGainForHistory {
  CAccountTrophyGainForHistory_CompetitionMatch();

  const uint Rank; // Maniascript
  CTrophyAchievement_CompetitionMatch* TrophyAchievement_CompetitionMatch; // Maniascript
};

struct CAccountTrophyGainForHistory_CompetitionRanking : public CAccountTrophyGainForHistory {
  CAccountTrophyGainForHistory_CompetitionRanking();

  const uint Rank; // Maniascript
  CTrophyAchievement_CompetitionRanking* TrophyAchievement_CompetitionRanking; // Maniascript
};

struct CAccountTrophyGainForHistory_LiveMatch : public CAccountTrophyGainForHistory {
  CAccountTrophyGainForHistory_LiveMatch();

  const uint Rank; // Maniascript
  CTrophyAchievement_LiveMatch* TrophyAchievement_LiveMatch; // Maniascript
};

struct CAccountTrophyGainForHistory_SoloMedal : public CAccountTrophyGainForHistory {
  CAccountTrophyGainForHistory_SoloMedal();

  const uint Level; // Maniascript
  const uint PreviousLevel; // Maniascript
  CTrophyAchievement_SoloMedal* TrophyAchievement_SoloMedal; // Maniascript
};

struct CAccountTrophyGainForHistory_SoloRanking : public CAccountTrophyGainForHistory {
  CAccountTrophyGainForHistory_SoloRanking();

  const uint Rank; // Maniascript
  CTrophyAchievement_SoloRanking* TrophyAchievement_SoloRanking; // Maniascript
};

struct CAccountTrophyLastYearSummary : public CMwNod {
  CAccountTrophyLastYearSummary();

  const string AccountId; // Maniascript
  const string WebServicesUserId; // Maniascript
  const uint T1Count; // Maniascript
  const uint T2Count; // Maniascript
  const uint T3Count; // Maniascript
  const uint T4Count; // Maniascript
  const uint T5Count; // Maniascript
  const uint T6Count; // Maniascript
  const uint T7Count; // Maniascript
  const uint T8Count; // Maniascript
  const uint T9Count; // Maniascript
  const uint TimeStamp; // Maniascript
};

struct CTrophyAchievement : public CMwNod {
  CTrophyAchievement();

  const string TrophyAchievementId; // Maniascript
  const string TrophyAchievementType; // Maniascript
};

struct CTrophyAchievement_CompetitionMatch : public CTrophyAchievement {
  CTrophyAchievement_CompetitionMatch();

  const string CompetitionId; // Maniascript
  const wstring CompetitionMatchInfo; // Maniascript
  const wstring CompetitionName; // Maniascript
  const wstring CompetitionStage; // Maniascript
  const wstring CompetitionStageStep; // Maniascript
  const string CompetitionType; // Maniascript
  const string ServerId; // Maniascript
};

struct CTrophyAchievement_CompetitionRanking : public CTrophyAchievement {
  CTrophyAchievement_CompetitionRanking();

  const string CompetitionId; // Maniascript
  const wstring CompetitionName; // Maniascript
  const wstring CompetitionStage; // Maniascript
  const wstring CompetitionStageStep; // Maniascript
  const string CompetitionType; // Maniascript
};

struct CTrophyAchievement_LiveMatch : public CTrophyAchievement {
  CTrophyAchievement_LiveMatch();

  const uint Duration; // Maniascript
  const string GameMode; // Maniascript
  const string GameModeCustomData; // Maniascript
  const bool IsOfficial; // Maniascript
  const string ServerId; // Maniascript
};

struct CTrophyAchievement_SoloMedal : public CTrophyAchievement {
  CTrophyAchievement_SoloMedal();

  const string SoloMedalAchievementType; // Maniascript
};

struct CTrophyAchievement_SoloRanking : public CTrophyAchievement {
  CTrophyAchievement_SoloRanking();

  const string MapId; // Maniascript
  const string SeasonId; // Maniascript
  const string SoloRankingAchievementType; // Maniascript
};

struct CTrophySoloMedalAchievementLevelSettings : public CMwNod {
  CTrophySoloMedalAchievementLevelSettings();

  const string Level; // Maniascript
  const uint T1Count; // Maniascript
  const uint T2Count; // Maniascript
  const uint T3Count; // Maniascript
  const uint T4Count; // Maniascript
  const uint T5Count; // Maniascript
  const uint T6Count; // Maniascript
  const uint T7Count; // Maniascript
  const uint T8Count; // Maniascript
  const uint T9Count; // Maniascript
};

struct CTrophySoloMedalAchievementSettings : public CMwNod {
  CTrophySoloMedalAchievementSettings();

  const string Type; // Maniascript
  CTrophySoloMedalAchievementLevelSettings* AllBronzeLevelSettings; // Maniascript
  CTrophySoloMedalAchievementLevelSettings* AllSilverLevelSettings; // Maniascript
  CTrophySoloMedalAchievementLevelSettings* AllGoldLevelSettings; // Maniascript
  CTrophySoloMedalAchievementLevelSettings* AllAuthorLevelSettings; // Maniascript
};

struct CMapRecord : public CMwNod {
  CMapRecord();

  const string AccountId; // Maniascript
  const string WebServicesUserId; // Maniascript
  const string MapId; // Maniascript
  const MwId MapUid; // Maniascript
  const string ScopeType; // Maniascript
  const string ScopeId; // Maniascript
  const string GameMode; // Maniascript
  const string GameModeCustomData; // Maniascript
  const uint Score; // Maniascript
  const uint Time; // Maniascript
  const uint RespawnCount; // Maniascript
  const uint Timestamp; // Maniascript
  const uint Medal; // Maniascript
  const uint MultiAsyncLevel; // Maniascript
  const uint SkillPoints; // Maniascript
  const wstring FileName; // Maniascript
  const string ReplayUrl; // Maniascript
};

struct NGameScriptChat_SContext {
  MwVirtualArray<NGameScriptChat_SHistory> Histories; // Maniascript
  NGameScriptChat_SHistory* History_Create(wstring Filter, uint MaxSize); // Maniascript
  void History_Destroy(NGameScriptChat_SHistory* History); // Maniascript
};

struct NGameScriptChat_SHistory {
  const MwVirtualArray<NGameScriptChat_SEntry> Entries; // Maniascript
  uint Window_Size; // Maniascript
  uint Window_Offset; // Maniascript
  const uint Window_OffsetMax; // Maniascript
  MwFastBuffer<NGameScriptChat_SEvent*> PendingEvents; // Maniascript
};

struct NGameScriptChat_SEntry {
  SConstStringInt const Text; // Maniascript
  SConstString const SenderLogin; // Maniascript
  SConstStringInt const SenderDisplayName; // Maniascript
  SConstStringInt const SenderFullDisplayName; // Maniascript
  SConstStringInt const SenderFullDisplayName_ForTTS; // Maniascript
  SConstStringInt const SenderTeamColorText; // Maniascript
  const bool IsSystemMessage; // Maniascript
  const EChatScope ChatScope; // Maniascript
};

struct NGameScriptChat_SEvent {
};

struct NGameScriptChat_SEvent_HistoryChange : public NGameScriptChat_SEvent {
};

struct NGameScriptChat_SEvent_NewEntry : public NGameScriptChat_SEvent {
  NGameScriptChat_SEntry Entry; // Maniascript
};

struct CGameDirectLinkScript {
};

struct CGameDirectLinkScript_Home : public CGameDirectLinkScript {
};

struct CGameDirectLinkScript_NewMap : public CGameDirectLinkScript {
};

struct CGameDirectLinkScript_OfficialCampaign : public CGameDirectLinkScript {
};

struct CGameDirectLinkScript_JoinServer : public CGameDirectLinkScript {
  const string ServerId; // Maniascript
  const bool IsSpectator; // Maniascript
};

struct CGameDirectLinkScript_TrackOfTheDay : public CGameDirectLinkScript {
};

struct CGameDirectLinkScript_Ranked : public CGameDirectLinkScript {
};

struct CGameDirectLinkScript_Royal : public CGameDirectLinkScript {
};

struct CGameDirectLinkScript_ArcadeServer : public CGameDirectLinkScript {
};

struct CGameDirectLinkScript_Hotseat : public CGameDirectLinkScript {
};

struct CGameDirectLinkScript_Splitscreen : public CGameDirectLinkScript {
};

struct CGameDirectLinkScript_Garage : public CGameDirectLinkScript {
};

struct CGameDirectLinkScript_WaitingPage : public CGameDirectLinkScript {
  const string Reason; // Maniascript
};

struct CGameDirectLinkScript_JoinSession : public CGameDirectLinkScript {
  const string SessionId; // Maniascript
  const bool IsFirstPartySession; // Maniascript
  const uint Context; // Maniascript
};

struct CGameUserManagerScript_VoiceChatEvent : public CMwNod {
  CGameUserManagerScript_VoiceChatEvent();

};

struct CGameUserManagerScript_VoiceChatEvent_UserChange_IsSpeaking : public CGameUserManagerScript_VoiceChatEvent {
  CGameUserManagerScript_VoiceChatEvent_UserChange_IsSpeaking();

  CGameUserVoiceChat* const User; // Maniascript
  const bool NewValue; // Maniascript
};

struct CGameUserManagerScript_VoiceChatEvent_UserChange_IsConnected : public CGameUserManagerScript_VoiceChatEvent {
  CGameUserManagerScript_VoiceChatEvent_UserChange_IsConnected();

  CGameUserVoiceChat* const User; // Maniascript
  const bool NewValue; // Maniascript
};

struct CGameUserManagerScript_VoiceChatEvent_UserChange_IsMuted : public CGameUserManagerScript_VoiceChatEvent {
  CGameUserManagerScript_VoiceChatEvent_UserChange_IsMuted();

  CGameUserVoiceChat* const User; // Maniascript
  const bool NewValue; // Maniascript
  const bool ChangePending; // Maniascript
};

struct CGameUserManagerScript_VoiceChatEvent_SpeakingHasChanged : public CGameUserManagerScript_VoiceChatEvent {
  CGameUserManagerScript_VoiceChatEvent_SpeakingHasChanged();

};

struct CGameUserManagerScript_VoiceChatEvent_Message : public CGameUserManagerScript_VoiceChatEvent {
  CGameUserManagerScript_VoiceChatEvent_Message();

  CGameUserVoiceChat* const Sender; // Maniascript
  CGameUserVoiceChat* const Destination; // Maniascript
  const wstring Message; // Maniascript
  const string Lang_BCP47; // Maniascript
};

struct CGameUserManagerScript_VoiceChatEvent_DisplayUI : public CGameUserManagerScript_VoiceChatEvent {
  CGameUserManagerScript_VoiceChatEvent_DisplayUI();

  const bool NewValue; // Maniascript
};

struct NGameEditorMap_SExperimentalFeatures {
  bool IsAirMappingModeAvailable;
  bool IsGhostModeAvailable;
  bool AreFreeModesAvailable;
  bool IsFreeDestructibleVoxelsModeAvailable;
  bool IsBlockAdvisorEnabled;
  bool IsBlockLinkEnabled;
  bool IsAutoAirMappingEnabled;
  uint AutoAirMapping_MaxPillarCount;
  bool DeleteBeforePlacingBlocks;
  bool DisplaySkinsInInventory;
  uint AutoSavePeriod;
  bool ShowNextMapElemParamIMGUIWindow;
  bool ShowMagnetsInItemCursor;
  float MagnetSnapDistance;
  bool SaveLaunchedCheckpointsInMap;
};

struct NPlugItemPlacement_STag {
  SConstString const Type;
};

struct CAdvertisingManager {
  enum class CAdvertisingManager::ELiveAdSystem {
    Nadeo = 0,
    UGC = 1,
  };
  MwSArray<CAdvertisingSlot> AdSlots_Nadeo; // Maniascript
  MwSArray<CAdvertisingSlot> AdSlots_UGC; // Maniascript
  bool AdSystem_GetAvailability(CAdvertisingManager::ELiveAdSystem AdSystem); // Maniascript
  void AdSystem_SetImpressionParams(CAdvertisingManager::ELiveAdSystem AdSystem, float MinDuration_seconds, float MaxAngle_deg, float MinScreenSurface01, float MinAdVisibleSurface01); // Maniascript
  int AdSlot_FetchImpressions(CAdvertisingManager::ELiveAdSystem AdSystem, MwId SlotId); // Maniascript
  float AdSlot_FetchDisplayDuration(CAdvertisingManager::ELiveAdSystem AdSystem, MwId SlotId); // Maniascript
  void AdSlot_SetNextAd(CAdvertisingManager::ELiveAdSystem AdSystem, MwId SlotId, int AdUid, string AdImageUrl); // Maniascript
};

struct NGameMiniMap_SMgr : public CMwNod {
  NGameMiniMap_SMgr();

};

struct CGameVoiceChatConfigScript {
  enum class CGameVoiceChatConfigScript::ESyncMode {
    Default = 0,
    Disabled = 1,
    Manual = 2,
    Clan = 3,
    Squad = 4,
  };
  CGameUserVoiceChat::EMuteSetting NewRemoteUser_DefaultMuteSetting; // Maniascript
  CGameVoiceChatConfigScript::ESyncMode SyncMode; // Maniascript
  string Manual_Channel; // Maniascript
  void Manual_ClearUsers(); // Maniascript
  CGameUserVoiceChat* Manual_UserAdd_Proc(string WebServicesUserId); // Maniascript
};

struct SWebServicesTaskResult_GhostDriver_Download_Ghost {
  CGameCtnGhost* DbgGhost;
  CGameGhostScript* Ghost; // Maniascript
};

struct SWebServicesTaskResult_GhostDriver_Download_Member {
  MwFastBuffer<SWebServicesTaskResult_GhostDriver_Download_Ghost> Ghosts; // Maniascript
};

struct SWebServicesTaskResult_GhostDriver_Download_Team {
  uint TeamLevel; // Maniascript
  MwFastBuffer<SWebServicesTaskResult_GhostDriver_Download_Member> Members; // Maniascript
  MwFastBuffer<SWebServicesTaskResult_GhostDriver_Download_Ghost> Ghosts; // Maniascript
};

struct SWebServicesTaskResult_GhostDriver_UploadLimit {
  uint TeamLevel; // Maniascript
  uint Limit; // Maniascript
};

struct CGameDataFileTask_GhostDriver : public CWebServicesTaskSequence {
};

struct CGameDataFileTask_GhostDriver_UploadLimits : public CGameDataFileTask_GhostDriver {
};

struct CGameDataFileTask_GhostDriver_Upload : public CGameDataFileTask_GhostDriver {
};

struct CGameDataFileTask_GhostDriver_Download : public CGameDataFileTask_GhostDriver {
};

struct CFriend : public CMwNod {
  CFriend();

  const string AccountId; // Maniascript
  const string CountryFlagUrl; // Maniascript
  const wstring DisplayName; // Maniascript
  const bool IsFirstPartyDisplayName; // Maniascript
  const string PlatformType; // Maniascript
  const string Presence; // Maniascript
  const string Relationship; // Maniascript
  const string WebServicesUserId; // Maniascript
};

struct CSkinInfo : public CMwNod {
  CSkinInfo();

  const wstring Name; // Maniascript
  const string Type; // Maniascript
  const string Url; // Maniascript
};

struct CSquadInvitation : public CMwNod {
  CSquadInvitation();

  const string AccountId; // Maniascript
  const string CountryFlagUrl; // Maniascript
  const wstring DisplayName; // Maniascript
  const bool IsFirstPartyDisplayName; // Maniascript
  const string PrestigeSkinOptions; // Maniascript
  MwSArray<CSkinInfo> SkinList; // Maniascript
};

struct CSquadMember : public CMwNod {
  CSquadMember();

  const string AccountId; // Maniascript
  const string CountryFlagUrl; // Maniascript
  const wstring DisplayName; // Maniascript
  const bool IsFirstPartyDisplayName; // Maniascript
  const string PrestigeSkinOptions; // Maniascript
  MwSArray<CSkinInfo> SkinList; // Maniascript
};

struct CSquad : public CMwNod {
  CSquad();

  const uint CreationTimeStamp; // Maniascript
  const string LeaderAccountId; // Maniascript
  const string LeaderWebServicesUserId; // Maniascript
  const string Id; // Maniascript
  const bool IsLocked; // Maniascript
  const wstring Name; // Maniascript
  const uint Size; // Maniascript
  const string Type; // Maniascript
  const uint UpdateTimeStamp; // Maniascript
  MwSArray<CSquadInvitation> InvitationList; // Maniascript
  MwSArray<CSquadMember> MemberList; // Maniascript
};

struct CNewsLink : public CMwNod {
  CNewsLink();

  const string Type; // Maniascript
  const string Param; // Maniascript
  const wstring ActionName; // Maniascript
  const wstring ActionDescription; // Maniascript
};

struct CNews : public CMwNod {
  CNews();

  const string Id; // Maniascript
  const string Type; // Maniascript
  const string Placement; // Maniascript
  const string Locale; // Maniascript
  const wstring Title; // Maniascript
  const wstring Body; // Maniascript
  const string MediaUrl; // Maniascript
  const string MediaType; // Maniascript
  const string PublicationDate; // Maniascript
  const uint Priority; // Maniascript
  const uint DisplayTime; // Maniascript
  MwSArray<CNewsLink> LinkList; // Maniascript
};

struct CPrestige : public CMwNod {
  CPrestige();

  const string CategoryType; // Maniascript
  const uint CategoryLevel; // Maniascript
  const bool IsUnlocked; // Maniascript
  const NWebServicesPrestige::EPrestigeMode Mode; // Maniascript
  const string PrestigeId; // Maniascript
  const uint PrestigeLevel; // Maniascript
  const wstring RewardDisplayName; // Maniascript
  const string RewardFileUrl; // Maniascript
  const string RewardThumbnailUrl; // Maniascript
  const string RewardType; // Maniascript
  const string SkinOptions; // Maniascript
  const uint StatCurrentValue; // Maniascript
  const uint StatValueForNextLevel; // Maniascript
  const uint StatValueForCurrentLevel; // Maniascript
  const uint TimeStamp; // Maniascript
  const uint Year; // Maniascript
};

struct CUserPrestige : public CMwNod {
  CUserPrestige();

  const string CategoryType; // Maniascript
  const uint CategoryLevel; // Maniascript
  const NWebServicesPrestige::EPrestigeMode Mode; // Maniascript
  const string PrestigeId; // Maniascript
  const uint PrestigeLevel; // Maniascript
  const string SkinOptions; // Maniascript
  const uint TimeStamp; // Maniascript
  const uint Year; // Maniascript
};

struct CGameCtnMenuProfileScene : public CGameMenuScene {
};

struct CNadeoServicesMap : public CMwNod {
  CNadeoServicesMap();

  const string AuthorAccountId; // Maniascript
  const string AuthorWebServicesUserId; // Maniascript
  const wstring AuthorDisplayName; // Maniascript
  const bool AuthorIsFirstPartyDisplayName; // Maniascript
  const uint AuthorScore; // Maniascript
  const uint BronzeScore; // Maniascript
  const string CollectionName; // Maniascript
  const bool CreatedWithGamepadEditor; // Maniascript
  const bool CreatedWithSimpleEditor; // Maniascript
  const wstring FileName; // Maniascript
  const string FileUrl; // Maniascript
  const uint GoldScore; // Maniascript
  const string Id; // Maniascript
  const bool IsPlayable; // Maniascript
  const wstring Name; // Maniascript
  const uint SilverScore; // Maniascript
  const wstring Style; // Maniascript
  const string SubmitterAccountId; // Maniascript
  const string SubmitterWebServicesUserId; // Maniascript
  const string ThumbnailUrl; // Maniascript
  const uint TimeStamp; // Maniascript
  const wstring Type; // Maniascript
  const string Uid; // Maniascript
};

struct CNadeoServicesItemCollection : public CMwNod {
  CNadeoServicesItemCollection();

  const string ActivityId; // Maniascript
  const uint CreationTimeStamp; // Maniascript
  const string CreatorAccountId; // Maniascript
  const string CreatorWebServicesUserId; // Maniascript
  const wstring CreatorDisplayName; // Maniascript
  const bool CreatorIsFirstPartyDisplayName; // Maniascript
  const wstring ClubId; // Maniascript
  CNadeoServicesItemCollectionVersion* CurrentVersion; // Maniascript
  const wstring DisplayName; // Maniascript
  const bool HasVersionList; // Maniascript
  const string Id; // Maniascript
  const bool IsCurrentVersionNull; // Maniascript
  const wstring Name; // Maniascript
  const string Type; // Maniascript
  const uint UpdateTimeStamp; // Maniascript
  MwSArray<CNadeoServicesItemCollectionVersion> VersionList; // Maniascript
};

struct CNadeoServicesSkin : public CMwNod {
  CNadeoServicesSkin();

  const string Checksum; // Maniascript
  const string CreatorAccountId; // Maniascript
  const string CreatorWebServicesUserId; // Maniascript
  const wstring CreatorDisplayName; // Maniascript
  const bool CreatorIsFirstPartyDisplayName; // Maniascript
  const wstring DisplayName; // Maniascript
  const wstring FileName; // Maniascript
  const string FileUrl; // Maniascript
  const string Id; // Maniascript
  const wstring Name; // Maniascript
  const string ThumbnailUrl; // Maniascript
  const uint TimeStamp; // Maniascript
  const string Type; // Maniascript
};

struct CZone : public CMwNod {
  CZone();

  const string Id; // Maniascript
  const wstring Name; // Maniascript
  const string ParentId; // Maniascript
  const wstring Path; // Maniascript
  const wstring FullPath; // Maniascript
  const string FlagUrl; // Maniascript
  const string CountryFlagUrl; // Maniascript
  const wstring Model_CarSport_SkinName; // Maniascript
  const string Model_CarSport_SkinUrl; // Maniascript
  const wstring Model_CharacterPilot_SkinName; // Maniascript
  const string Model_CharacterPilot_SkinUrl; // Maniascript
};

struct CNadeoServicesItemCollectionVersion : public CMwNod {
  CNadeoServicesItemCollectionVersion();

  const string Checksum; // Maniascript
  const uint CreationTimeStamp; // Maniascript
  const string Id; // Maniascript
  const uint UpdateTimeStamp; // Maniascript
  const string Url; // Maniascript
};

struct NSmArenaInterface_SMgr {
};

struct NPlugCurve_SSimpleCurveInPlace7 : public NPlugCurve_SSimpleCurve {
};

struct NSceneTransitoryBlockingSurfs_SMgr {
};

struct NSceneLightMapCloud_SServer {
};

struct NSceneLightMapCloud_SClient {
};

struct NScenePicking_SMgr {
  MwFastBuffer<NScenePicking_SPickable> Pickables;
};

struct NSceneGateSpecial_SMgr {
  MwFastBuffer<SSceneGateSpecial> GateSpecials;
};

struct NSceneTimeshift_SMgr {
};

struct NSceneRecorder_SMgr {
};

struct NSmArenaPhysics_SMgr {
};

struct NSmPlayerVis_SMgr {
};

struct NSmPlayerPhy_SMgr {
};

struct NSmArenaVis_SMgr {
};

struct NGameConstraintPhy_SMgr {
};

struct NGameObjectPhy_SMgr {
};

struct NGameArenaState_SMgr {
};

struct NGameArenaVis_SMgr {
};

struct NGamePrefab_SMgr {
  NGamePrefab_SGridParams GridParams;
  MwFastBuffer<NGamePrefab_SInst> Insts;
  NFastBlockAlloc_SAllocator EntLists_BlockAlloc;
  NFastBlockAlloc_SAllocator GridEntSpawners_BlockAlloc;
};

struct NGamePrefab_SInst {
  NGamePrefab_SInstanceCreateParams Params;
  NGamePrefab_SSceneEntListElem* EntList;
};

struct NGamePrefab_SGridEntSpawner {
};

struct NGamePrefabPhy_SMgr {
  MwFastBuffer<NGamePrefabPhy_SInst> Insts;
};

struct NGamePrefabPhy_SInst {
  CPlugPrefab* Prefab;
  NGamePrefabPhy_SSceneEntListElem* EntList;
};

struct NSceneKinematicVis_SMgr {
  MwFastBuffer<NSceneKinematicVis_SConstraint> Constraints;
  const MwSArray<NSceneKinematicVis_SSharedSignal*> SharedSignals;
};

struct NSceneKinematicVis_SConstraint {
};

struct NGameActionFxVis_SMgr {
};

struct NGameShieldVis_SMgr {
};

struct NGameShieldPhy_SMgr {
};

struct NGameGateVis_SMgr {
  uint ArmorAnimTime;
  float ArmorAnimExp;
};

struct NGameGatePhy_SMgr {
};

struct NGameWaypoint_SMgr {
  const MwSArray<NGameWaypoint_STrigger*> Triggers;
  const MwSArray<NGameWaypoint_SSpawn*> Spawns;
  const uint StartCount;
  CPlugMediaClipList* DefaultSpawnClipList;
};

struct NGameWaypoint_SSpawn {
};

struct NGameItem_SMgr {
};

struct NGameMediaClip_SMgr {
};

struct NGameSlotVis_SMgr {
};

struct NGameSlotPhy_SMgr {
  float Tunings_CoefFriction;
  float Tunings_CoefAcceleration;
  float Tunings_Sensibility;
};

struct NGameVehiclePhy_SMgr {
};

struct NGameTurretVis_SMgr {
};

struct NGameTurretPhy_SMgr {
};

struct NGameObjectVis_SMgr {
  MwFastBuffer<CGameObjectModel*> ModelLib;
};

struct NGamePodiumVis_SMgr {
  NGamePodiumVis_SConfig Config;
};

struct NGamePodium_SMgr {
};

struct NGamePodium_SPodium {
  CPlugMediaClipList* ClipList;
};

struct NGameCamera_SMgr {
  MwFastBuffer<NGameCamera_SCamSys*> CamSystems;
};

struct NGameMapPhy_SMgr {
};

struct NGameMgrMap_SMgr {
  NFastBucketAlloc_SAllocator NonEditorBlockContentAlloc;
  const MwSArray<NGameMgrMap_SMapInst*> MapInsts;
  NFastBlockAlloc_SAllocator ItemInstsAlloc_BlockAlloc;
  NFastBlockAlloc_SAllocator BlockInstsAlloc_BlockAlloc;
  NFastBucketAlloc_SAllocator NonEditorBlockContentAlloc;
  const uint BlockInMapUIdToBlockInstMem;
  const uint ItemInMapUIdToItemInstMem;
};

struct NGameGhostClips_SMgr {
  MwFastBuffer<NGameGhostClips_SClipPlayerGhost> Ghosts;
};

struct NGameGhost_SMgr {
};

struct NSceneItemPlacement_SMgr {
};

struct NSceneItemPlacement_SZone {
};

struct NSceneBulletPhy_SMgr {
  uint BulletRemovalDelay;
  bool LaserVsBullet;
  bool LaserSkewering;
};

struct NSceneDestructiblePhy_SMgr {
};

struct SSceneDestructibleDynaPhy {
};

struct SSceneDestructiblePhy {
};

struct NSceneDyna_SMgr {
  MwFastBuffer<NSceneDyna_SKinematicConstraint> KinematicConstraints;
  const MwSArray<NSceneDyna_SKinematicSharedSignal*> KinematicSharedSignals;
  NSceneDyna_SItemArrayDyn Items;
};

struct NSceneDyna_SKinematicConstraint {
};

struct NSceneAdvert_SMgr {
  MwFastBuffer<NSceneAdvert_SVisibilityTracker*> Trackers;
  MwSArray<NSceneAdvert_SAdSystemParams> AdSystemParams;
  uint MaxRayBillboardRaycastPerFrame;
};

struct NSceneMapColoring_SMgr {
  CPlugBitmap* BitmapTeamEmblem;
};

struct NSceneModelKit_SMgr {
};

struct NSceneProp_SMgrPhy {
};

struct NSceneProp_SProp {
};

struct NSceneProp_SMgrVis {
};

struct NSceneConstruction_SMgrPhy {
};

struct NSceneConstruction_SConstruction {
};

struct NSceneConstruction_SMgrVis {
};

struct NSceneRopeVis_SMgr {
};

struct NSceneRopePhy_SMgr {
};

struct SSceneRopePhyEnt {
};

struct NSceneEditorHelper_SMgr {
  bool IsVisible;
};

struct NSceneEditorHelper_SHelper {
};

struct NSceneFlock_SMgr {
  bool IsPlaying;
  NSceneFlock_SBirdPhysicsParams BirdPhysicsParams;
};

struct NSceneAnim_SMgr {
  MwFastBuffer<NSceneAnim_SModel*> Models;
  MwFastBuffer<NSceneAnim_SModelInst> ModelInsts;
};

struct NSceneSolid2Vis_SMgr {
};

struct NSceneSound_SMgr {
};

struct NSceneSound_SSource {
};

struct NSceneSpectatorVis_SMgr {
  NSceneSpectatorVis_SInput Input;
  const uint ActualSpectatorCount;
  const uint SpawnPoss_Count;
};

struct NSceneParticleVis_SMgr {
};

struct NSceneCitizenNetwork_SMgr {
};

struct NSceneCitizenNetwork_SChunk {
};

struct NSceneRoad_SMgr {
};

struct NSceneRoad_SChunk {
};

struct NSceneRail_SMgr {
};

struct NSceneBulletVis_SMgr {
};

struct NSceneFxSystem_SMgr {
  MwFastBuffer<NSceneFxSystem_SFxSystemInstance> Instances;
  MwFastBuffer<NSceneFxSystem_SRuntimeModel*> RuntimeModels;
};

struct NSceneFxSystem_SFxSystemInstance {
  NSceneFxSystem_SRuntimeModel* RuntimeModel;
  iso4 Loc;
  float LodZMin;
};

struct NSceneWind_SMgrPhy {
};

struct NSceneTrainVis_SMgr {
};

struct NSceneTrainPhy_SMgr {
};

struct NSceneCharVis_SMgr {
  MwFastBuffer<CMwNod*> CharModels;
  const MwSArray<CSceneCharVis*> CharViss;
  CMwRefBuffer CustomSkinModels;
  CPlugAnimFile* CinematicLib;
  NSceneVisEntFx_STactical FxTactical;
  uint LodMaxVert;
};

struct NSceneVehicleVis_SMgr {
};

struct NSceneLight_SMgr {
};

struct NScenePathFinding_SMgr {
  NScenePathFinding_SNavMeshBuildParams NavMeshBuildParams;
  NScenePathFinding_STileCacheHandler TileCacheFromChunksHandler;
  MwFastBuffer<NScenePathFinding_SPathRequestState> PathRequests;
  MwFastBuffer<NScenePathFinding_SDynamicPathState> DynamicPaths;
};

struct NSceneTrafficVis_SMgr {
};

struct NSceneTrafficPhy_SMgr {
};

struct NSceneVFX_SMgr {
};

struct NSceneVFX_SVFXInstance {
};

struct NSceneEdEntWrapper_SMgr {
};

struct NSceneEdEntWrapper_SWrapper {
};

struct NHmsForestVis_SMgr {
};

struct NHmsForestVis_STree {
};

struct NHmsMgrInstDyna2_SMgr {
  const uint cModel;
  const uint cInstance;
  bool InstAABBT_Order;
};

struct SHmsLodGroup {
};

struct SHmsInstDyna {
};

struct SHmsMeshStatic {
};

struct NHmsZone_SMgrLightDynamic {
};

struct NHmsZone_SLightDyna {
  vec3 LinearRGB;
  float Radius;
  iso4 Location;
};

struct NHmsCollision_SMgr {
  void UpdateStatic();
  const uint AllocatedByteCount;
  const uint GetUsedByteCount;
  const uint cItemStatic;
  const uint cItemContinuous;
  const uint cItemDiscrete;
  const uint cSurfaceStatic;
  const uint cSurfaceContinuous;
  const uint cSurfaceDiscrete;
  vec3 WarpExclusionBoxCenter;
  vec3 WarpExclusionBoxHalf;
};

struct NHmsCollision_SItem {
};

struct NScenePicking_SPickable {
  GmTransQuat Loc;
};

struct GmTransQuat {
  vec3 Trans;
  quat Quat;
};

struct NPlugPainterLayer_SElem {
  MwId m_ID;
  NPlugPainterLayer::EPlugPainterLayer_Type m_Type;
  bool IsVisible;
  wstring m_Name;
};

struct NPlugPainterLayer_SLayerVFX {
};

struct NPlugPainterLayer_SLayerDraw {
  MwFastBuffer<NPlugPainterLayer_SBitmapGroup*> m_Bitmaps;
  float Opacity;
};

struct NPlugPainterLayer_SBitmapGroup {
  DataRef m_MainBitmap;
  DataRef m_OpacityBitmap;
  DataRef m_RoughBitmap;
};

struct NPlugAnim_SSkelPose {
  uint8 iLod;
  NPlugAnim::EPoseType Type;
  MwSArray<GmTransQuat> Joints;
  MwSArray<float> Floats;
};

struct NPlugAnim_SIKSkelPoseN {
  MwSArray<NPlugAnim_SIKChainPoseN> Chains;
};

struct NPlugAnim_SIKChainPoseN {
  vec3 A;
  vec3 B;
  vec3 PoleVector;
  quat BRot;
};

struct NPlugAnim_SIKChainPose {
  vec3 A;
  vec3 B;
  vec3 PoleVector;
};

struct NPlugAnim_SIKSkelMappingCache {
  MwSArray<NPlugAnim_SIKChainMappingCache> Chains;
};

struct NPlugAnim_SIKChainMappingCache {
  uint16 AJoint;
  uint16 BJoint;
  vec3 PoleVectorPos;
  uint16 PoleVectorJoint;
  vec3 PoleVectorRef;
  uint16 BRotJoint;
};

struct NPlugAnim_SIKSkelMapping {
  MwSArray<NPlugAnim_SIKChainMapping> Chains;
};

struct NPlugAnim_SIKChainMapping {
  MwId Name;
  MwId AJoint;
  MwId BJoint;
  vec3 PoleVectorPos;
  MwId PoleVectorJoint;
  vec3 PoleVectorRef;
  MwId BRotJoint;
};

struct NPlugAnim_SIKSkel {
  MwSArray<MwId> ChainNames;
};

struct SMetaPtr {
};

struct NHmsMgrVolume_SFogSimuParameters {
  float EmissionMultiplier; // Range: 0 - 5
  float DissipationFactor; // Range: 0 - 5
  float DensityMax; // Range: 0 - 10
  float GFactor; // Range: 0 - 1
};

struct NFuncShaderLayerUV_STransSubTextureIn {
  uint NbSubTexture;
  uint NbSubTexturePerLine;
  uint NbSubTexturePerColumn;
  bool TopToBottom;
};

struct TSceneUId {
};

struct NGameAdvert_Simu_SMgr {
};

struct NGameAdvert_Anzu_SMgr {
};

struct Nat256 {
};

struct NGameAdvert_Test_SMgr {
  bool TestVideos;
  bool TestManialink;
  CPlugBitmap* Texture_Manialink;
};

struct EditorMeshPickingIdentifier {
};

struct NGamePrefab_SInstanceCreateParams {
  CPlugPrefab* Prefab;
  GmTransQuat Loc;
  uint IdForLightMap;
  float Phase;
};

struct NGamePrefab_SSceneEntListElem {
  NGamePrefab_SSceneEntListElem* Next;
};

struct NGamePrefab_SGridParams {
  bool Enabled;
  float CellSize;
  float CellSpawnDist;
};

struct NFastBlockAlloc_SAllocator {
  uint UsedBlockCount;
  const uint AllocatedBytes;
};

struct NGamePrefabPhy_SSceneEntListElem {
  NGamePrefabPhy_SSceneEntListElem* Next;
};

struct NGamePrefabPhy_SInstanceCreateParams {
  CPlugPrefab* Prefab;
  float Phase;
};

struct NSceneKinematicVis_SSharedSignal {
  NPlugDyna_SKinematicConstraint* Model;
  float Phase;
  uint cRef;
};

struct NGamePodiumVis_SConfig {
  bool Hud3d;
  bool AvatarsEnabled;
  bool SkinsEnabled;
  bool CustomCamSys;
  bool PilotsOutsideVehicleIfPossible;
};

struct NGamePodiumVis_SSequence {
  CGameCtnMediaClipPlayer* ClipPlayer;
};

struct SGameHud3dParams {
  bool Players;
  bool Markers;
  bool DisplayAvatar;
  bool DisplayName;
  bool DisplayBg;
  bool DisplayEchelon;
  bool DoMaxDistanceFade;
  float DisplaySize;
  vec3 PlayerBaseTint;
  float PlayerBaseAlpha;
  vec3 MarkerBaseTint;
  float MarkerBaseAlpha;
  float BaseSkew;
  float WorldCartoucheAltitude;
  float ScreenCartoucheAltitude;
  bool ForceNameShadow;
  float NameClipLengthSizeRatio;
  float NameRatioXY;
  float LineWidthRatio;
  float GaugeYPos;
  float GaugeHeight;
  float GaugeWidth;
  uint VisibilityFadeDuration;
  uint VisibilityTestPeriod;
};

struct NGameMenuSkinChooser_SDispParams {
  bool IsNight;
  bool IsDamage;
  UnknownType IsDamageZone;
  UnknownType DamageZoneAmounts;
  bool IsDirtBody;
  bool IsDirtWheel;
  bool IsOpenParts;
  bool IsDetachedParts;
  bool IsBrake;
  bool IsTurbo;
  bool IsSkeleton;
  bool IsDead;
};

struct NGameCamera_SCamSys {
};

struct NGameScriptDebugger_SDebuggerMgr {
  void RemoveAll();
  MwFastBuffer<NGameScriptDebugger_SWatchedVariable> WatchV1;
  MwFastBuffer<NGameScriptDebugger_SWatchedVariableV2> WatchV2;
};

struct NGameScriptDebugger_SWatchedVariable {
  void Pause();
  void Resume();
  const wstring Name;
  const MwFastBuffer<wstring> Values;
  const wstring FileName;
};

struct NGameScriptDebugger_SWatchedVariableV2 {
  SConstString const Name;
  const MwSArray<wstring> Values;
};

struct NGameVideoSource_SMgr {
  MwArrayInPlaceDyn<SConstString> ApiNames;
  MwArrayInPlaceDyn<SConstString> SourceNames;
  string SelApiName;
  string SelSourceName;
  bool BitmapUsed;
  CPlugBitmap* Bitmap;
};

struct CAdvertisingSlot {
  MwId Id; // Maniascript
  float DisplayRatio; // Maniascript
  int CurrentAdUid; // Maniascript
  int NextAdUid; // Maniascript
};

struct NGameAdvert_Live_SMgr {
  NGameAdvert_Live_SAdsManager* AdsManager;
};

struct NGameAdvert_Live_SAdsManager {
  MwArrayInPlaceDyn<NGameAdvert_Live_SAdSlotInternal> AdSlotsInternal;
  NGameAdvert_Live_SImpressionParams ImpressionParams;
};

struct NGameAdvert_Live_SAdSlotInternal {
  CPlugBitmap* CurrentImage;
  float ImpressionDuration;
  uint Impressions;
};

struct NGameAdvert_Live_SImpressionParams {
  float MinDuration;
  float MaxAngleDeg;
  float MinScreenSurface01;
  float MinAdVisibleSurface01;
};

struct NChatLog_SChatHistory {
  MwSArray<NChatLog_SChatMsg> Msgs;
};

struct NChatLog_SChatMsg {
  SConstString Text;
  SConstString Sender;
  SConstStringInt SenderDisplayName;
  bool IsTargeted;
  uint8 ChatOption;
  uint8 SpecialCode;
  uint Timestamp;
};

struct NFastBucketAlloc_SAllocator {
  const uint cBucket;
  const uint AllocatedSize;
  const uint UsedSize;
};

struct NGameMgrMap_SMapInst {
  uint BlockInsts_Count;
  uint ItemInsts_Count;
};

// userName: 'ShootIconSetting'
// File extension: '.gbx.json'
struct NGameIconShooter_SSceneSetting {
  float LDirSun_AltitudeDeg; // Range: 0 - 90
  float LDirSun_AzimutDeg; // Range: -180 - 180
  RGBAColor LDirSun_RGB;
  float LDirSun_Scale;
  RGBAColor LAmbient_RGB;
  float LAmbient_Scale;
  CSystemFidFile* CubeMapFid;
  float ExposureBias; // Range: -2 - 2
};

struct NGameIconShooter_SCameraSetting {
  float AltitudeDeg; // Range: 0 - 90
  float AzimutDeg; // Range: -180 - 180
  float FovDeg; // Range: 0 - 100
};

struct CGameCtnMediaBlockGhostTM_SuperSKeyVal {
  enum class CGameCtnMediaBlockGhostTM_SuperSKeyVal::EForceLight {
    Auto = 0,
    On = 1,
    Off = 2,
  };
  wstring Ghost_Name;
  const wstring Skin_Name;
  void Select_Skin();
  vec3 Trail_Color;
  float Trail_Intensity; // Range: 0 - 1
  const uint Race_Time;
  float Start_Offset;
  bool Force_Hue;
  CGameCtnMediaBlockGhostTM_SuperSKeyVal::EForceLight Lights;
};

struct NGameMediaBlockEntity_SuperSKeyVal {
  enum class NGameMediaBlockEntity_SuperSKeyVal::EForceLight {
    Auto = 0,
    On = 1,
    Off = 2,
  };
  wstring Ghost_Name;
  const wstring Skin_Name;
  void Select_Skin();
  void Reset_Skin();
  string SkinOptions;
  vec3 Trail_Color;
  float Trail_Intensity; // Range: 0 - 1
  float SelfIllum_Intensity;
  float Start_Offset;
  bool Force_Hue;
  NGameMediaBlockEntity_SuperSKeyVal::EForceLight Lights;
  const string Race_Time;
};

struct CGameCtnMediaBlockOpponentVisibility_SuperSKeyVal {
  enum class CGameCtnMediaBlockOpponentVisibility_SuperSKeyVal::EOpponentVisibility {
    Hidden = 0,
    Ghost = 1,
    Opaque = 2,
  };
  CGameCtnMediaBlockOpponentVisibility_SuperSKeyVal::EOpponentVisibility Visibility;
};

struct CGameCtnMediaBlockDirtyLens_SKeyVal {
  float FxIntensity; // Range: 0 - 1
};

struct CGameCtnMediaBlockCameraEffectScript_SKeyVal {
  float A;
  float B;
  float C;
};

struct CGameCtnMediaBlockVehicleLight_SuperSKeyVal {
  wstring Target;
};

struct CGameCtnMediaBlockFog_SKeyVal {
  float Distance;
  float FogIntensity; // Range: 0 - 1
  float SkyFogIntensity; // Range: 0 - 1
  vec3 LinearRGB;
  float CloudsOpacity; // Range: 0 - 1
  float CloudsSpeedScale;
};

struct CGameCtnMediaBlockTimeSpeed_SKeyVal {
  float Value;
};

struct CGameCtnMediaBlockBloomHdr_SKeyVal {
  float FxIntensity;
  float StreaksIntensity;
  float StreaksAttenuation;
};

struct CGameCtnMediaBlockToneMapping_SKeyVal {
  float ExposureBias;
  float MaxHDR;
  float LightTrailScale;
};

struct CGameCtnMediaBlockDOF_SKeyVal {
  wstring Target;
  vec3 TargetRelativePos;
  float FocusZ; // Range: 0.1 - 5000
  float LensSize; // Range: 0.001 - 10
};

struct CGameCtnMediaBlock3dStereo_SKeyVal {
  float Separation; // Range: 0 - 1
  float Up_to_max; // Range: 0 - 1
  float ScreenDist; // Range: 0.5 - 500
};

struct CGameCtnMediaBlockTransitionFade_SuperSKeyVal {
  vec3 Color;
  float Opacity; // Range: 0 - 1
};

struct CGameCtnMediaBlockColorGrading_SuperSKeyVal {
  const wstring Current_grade;
  void Change_grade();
  float Intensity; // Range: 0 - 1
};

struct CGameCtnMediaBlockText_SuperSKeyVal {
  wstring Text;
  vec3 Position;
  vec2 Size;
  float Rotation;
  vec3 Color;
  float Opacity; // Range: 0 - 1
};

struct CGameCtnMediaBlockSpectators_SKeyVal {
  float Value; // Range: 0 - 100
};

struct CGameCtnMediaBlockSound_SuperSKeyVal {
  const wstring Current_sound;
  void Change_sound();
  bool Music;
  bool Looping;
  uint Play_count;
  float Volume; // Range: 0 - 1
  vec3 Position;
  bool Audio_To_Speech;
  wstring Target;
};

struct CGameCtnMediaBlockMusicEffect_SKeyVal {
  float MusicVolume; // Range: 0 - 1
  float SoundVolume; // Range: 0 - 1
};

struct CGameCtnMediaBlockImage_SuperSKeyVal {
  const wstring Current_image;
  void Change_image();
  vec3 Position;
  vec2 Size;
  float Rotation;
  float Opacity; // Range: 0 - 1
};

struct CGameCtnMediaBlockCameraEffectShake_SKeyVal {
  float Intensity;
  float Speed;
};

struct CGameCtnMediaBlockCameraCustom_SKeyVal {
  wstring Target;
  wstring Anchor;
  bool AnchorClipEntVisible;
  bool UseAnchorOrientation;
  CGameCtnMediaBlockCameraCustom::ECamInterp Interp;
  vec3 Val_Trans;
  vec3 Rotation__Pitch__Yaw__Roll_;
  float Roll;
  float Val_Fov;
  float Val_NearZ;
  vec3 Val_TargetTrans;
  float Val_PssmDistScale;
  vec3 TangentLeft_Trans;
  vec3 TangentRight_Trans;
};

struct CGameCtnMediaBlockCameraPath_SKeyVal {
  wstring Target;
  wstring Anchor;
  bool AnchorGhostVisible;
  bool UseAnchorOrientation;
  float Weight;
  vec3 Translation;
  vec3 Rotation__Pitch__Yaw__Roll_;
  float Fov;
  float NearZ;
  vec3 TargetTranslation;
};

struct CGameCtnMediaBlockCameraOrbital_SKeyVal {
  wstring Target;
  vec3 Target_pos__x_y_z_;
  vec3 TargetOffset;
  float Latitude;
  float Longitude;
  float Radius;
  float Field_of_view;
};

struct CGameCtnMediaBlockTime_SKeyVal {
  float Value;
  float Tangent;
};

struct CGameCtnMediaBlockFxCameraBlend_SKeyVal {
  float Capture_Weight; // Range: 0 - 1
};

struct CGameCtnMediaBlockFxColors_SKeyVal {
  float FxIntensity; // Range: 0 - 1
  bool EditingFar;
  float NearZ;
  float ColorParam_Hue; // Range: 0 - 1
  float ColorParam_Saturation; // Range: -1 - 1
  float ColorParam_Brightness; // Range: -1 - 1
  float ColorParam_Contrast; // Range: -1 - 1
  float ColorParam_InverseRGB; // Range: 0 - 1
  float ColorParam_ModulateRGB_r; // Range: 0 - 1
  float ColorParam_ModulateRGB_g; // Range: 0 - 1
  float ColorParam_ModulateRGB_b; // Range: 0 - 1
  float IntensityFarZ; // Range: 0 - 1
  float FarZ;
  float ColorParamFarZ_Hue; // Range: 0 - 1
  float ColorParamFarZ_Saturation; // Range: -1 - 1
  float ColorParamFarZ_Brightness; // Range: -1 - 1
  float ColorParamFarZ_Contrast; // Range: -1 - 1
  float ColorParamFarZ_InverseRGB; // Range: 0 - 1
  float ColorParamFarZ_ModulateRGB_r; // Range: 0 - 1
  float ColorParamFarZ_ModulateRGB_g; // Range: 0 - 1
  float ColorParamFarZ_ModulateRGB_b; // Range: 0 - 1
};

struct SGameItemModelTree {
  bool IsFolder;
  MwId Name;
  SConstString Collection;
  EGameCollectorSearchScope SearchIn;
  MwSArray<SGameItemModelTree> Childs;
};

struct SGameBlockInfoTree {
  bool IsFolder;
  MwId Name;
  MwSArray<SGameBlockInfoTree> Childs;
};

struct SGameBlockInfoGroup {
  MwId GroupId;
  MwFastBuffer<MwId> BlockIds;
};

struct NPlugSkin_SCustomizableBitmap {
  CPlugBitmap* Bitmap;
};

struct NGameGhostClips_SClipPlayerGhost {
  TSceneUId VisEntUId;
  CGameCtnMediaClip* Clip;
  CGameCtnGhost* GhostModel;
  CGameAvatar* GhostAvatarCache;
  CGameLeague* GhostZoneCountryCache;
};

struct NGameMapItemPlacement_SMapData {
  MwFastBuffer<NGameMapItemPlacement_SZone*> Zones;
};

struct NGameMapItemPlacement_SZone {
  NGameMapItemPlacement_SZoneId Id;
};

struct NGameMapItemPlacement_SZoneId {
  CGameCtnBlock* BlockWithZone;
  uint iZoneInBlock;
  uint iLayout;
};

struct SGameCtnIdentifier {
  MwId Id;
  MwId Collection;
  MwId Author;
};

struct GmTransYawPitchRollDeg {
  vec3 Trans;
  float Trans_x;
  float Trans_y;
  float Trans_z;
  float YawDeg;
  float PitchDeg;
  float RollDeg;
};

struct NHmsLightMap_SCptBackgnd {
  const nat2 c2TexelLM;
  const MwFastBuffer<CPlugBitmap*> BitmapLM_Accums;
  const float Progress__;
};

struct CVisionVideoDecode {
  nat2 m_c2PixelRGBA;
  CPlugFileVideo* m_PlugFileVideo;
};

struct NSysCfgVision_SSubsurfaceScattering {
  bool IsEnabled;
  float Profile_RadiusInMeter;
  RGBAColor Profile_Strength;
  RGBAColor Profile_Falloff;
  uint Separable_SampleQuality;
  float Separable_FilterPixelCountMin;
};

struct CVisionHmsZone {
  NVisionHmsZone_SWaterPlaneVisibility* m_WaterPlaneVisibility;
};

struct NVisionHmsZone_SWaterPlaneVisibility {
  UnknownType Planes_Visibility01;
  UnknownType Planes_MapIntensU8;
  UnknownType LastSelecteds_PlaneId;
  UnknownType LastSelecteds_I01;
  const float LastSelected0_I01; // Range: 0 - 1
};

struct NVis_SBenchShader {
};

struct NFilmicTone_PowerSegments_SCurveParamsUser {
  float m_toeStrength;
  float m_toeLength;
  float m_shoulderStrengthFStops;
  float m_shoulderLength;
  float m_shoulderAngle;
};

struct NSysCfgVision_SVisionImpostor {
};

struct NSysCfgVision_SVisionForest {
};

struct NSysCfgVision_SLightmap {
  float AmbientBumpiness; // Range: 1 - 8
  bool CacheSmall_BumpIntensYYY;
  uint CacheSmall_BumpIntensYYY_Q100;
  float CacheSmall_ProbeAmbientRGB_Q100;
  float CacheSmall_ProbeAmbientSH1_Q100;
  float CacheSmall_ProbeSkyVisible_Q100;
  bool AdaptPeelingToReceiverSize;
  bool Split2d_UvAlloc;
  bool DynaUV;
  bool DynaUV_DeferAllocUV;
  bool CustomLM1_IsEnabled;
  float Preview_AmbientScaleYm; // Range: 0 - 1
  float Preview_AmbientScaleYp; // Range: 0 - 1
};

struct NSysCfgVision_SCptInGameplay {
  uint Max_Ms_Frame_CPU;
  float Max_Ms_Frame_GPU;
  float MOp_ms_GPU_effective;
  bool Gpu_UseEffectivePerfRatio;
  float WaitForGpuMaxPowerStateSc;
  float BlendDurationSc;
  uint Incremental_DirGroupCount;
  NSysCfgVision_SCptInGame_Local Local;
  NSysCfgVision_SCptInGame_Global Global_Cloud;
};

struct NSysCfgVision_SCptInGame_Local {
  bool IsEnabled;
  EHmsLightMapQuality Quality;
  uint StartDelayMs;
  bool Cancel;
  bool WarpSky_UseCubeApprox;
  float LocalBackGND_DistFactor;
};

struct NSysCfgVision_SCptInGame_Global {
  bool IsEnabled;
  bool PartialCpt;
  EHmsLightMapQuality Quality;
  float CptStart_DistMin;
  float CptStart_DelaySc;
  uint PeelSizeForced;
  float ValidDist_Start;
  float ValidDist_Stop;
  NHmsLightMap_SCptBackgnd* State;
};

struct NSysCfgVision_SNvHBAO {
  bool Enabled;
  float Radius; // Range: 0.5 - 2
  float Bias; // Range: 0 - 0.5
  float LargeScaleAO; // Range: 0 - 2
  float SmallScaleAO; // Range: 0 - 2
  float PowerExponent; // Range: 1 - 4
};

struct NSysCfgVision_SSSAmbOcc {
  bool IsEnabled;
  bool UseHomeMadeHBAO;
  bool Display_AO;
  float ImageSize;
  float WorldSize;
  float Exponent;
  uint BlurTexelCount;
  bool DelayGrassFences;
  bool NvHBAO_Enabled;
  float NvHBAO_Radius; // Range: 0.5 - 2
  float NvHBAO_Bias; // Range: 0 - 0.5
  float NvHBAO_LargeScaleAO; // Range: 0 - 2
  float NvHBAO_SmallScaleAO; // Range: 0 - 2
  float NvHBAO_PowerExponent; // Range: 1 - 4
  bool NvHBAO_BigScale_Enabled;
  float NvHBAO_BigScale_Radius; // Range: 8 - 32
  float NvHBAO_BigScale_Bias; // Range: 0 - 0.5
  float NvHBAO_BigScale_LargeScaleAO; // Range: 0 - 2
  float NvHBAO_BigScale_SmallScaleAO; // Range: 0 - 2
  float NvHBAO_BigScale_PowerExponent; // Range: 1 - 4
};

struct NSysCfgVision_SDeferred {
  bool Enable;
  bool ForceForwardMSAA;
  bool MsaaUseZPrePass;
  bool FaceNormalFromDepth;
  bool LightEnable;
  bool LightCullFrustum;
  bool LightScissor;
  bool LightDepthTest;
  bool LightShowFill;
  bool LightShadowEnable;
  bool LightForceMultiProjector;
  bool ShadowVolEnable;
  bool ShadowVolShowPixelFill;
  bool ShadowVolUseBestZFunc;
  bool FakeOccEnable;
  bool DecalEnable;
  bool DecalEnableStaticGeom;
  bool DecalEnableStaticBox;
  bool DecalEnableDynamicGeom;
  bool DecalEnableDynamicBox;
  bool DecalSpecularEnable;
  bool DecalPNormalEnable;
  bool DecalStaticCache;
  NSysCfgVision::SDeferred::EDecalZFunc DecalForceZFunc;
  bool DecalUseStencil;
  uint DecalStaticBatchCountX256;
  bool DecalVisibleCheckDist;
  bool DecalVisibleCheckSize;
  float DecalVisibleMaxDistAtFov90;
  float DecalVisibleMinScreenHeight;
  float DecalVisibleFadeFactor;
  NSysCfgVision::SDeferred::EDecalDbg DecalDbg;
  bool DecalUseCameraSpace;
  bool FogEnable;
  bool FogVolumeEnable;
  bool FogVolumeTranslucent;
  bool FogVolumeNearZ;
  bool FogVolumeShowGeom;
  bool BurnEnable;
  bool BurnUseStencil;
  bool CameraMotion;
  bool ReadMotion;
};

struct CSysFidNodRefBase {
  CSystemFidFile* m_FidFile;
};

struct NSysPlatform_SDirectLink {
  NSysPlatform::EDirectLinkType Type;
  SConstString JoinServer_ServerId;
  NSysPlatform::EDirectLink_Join_Role JoinServer_Role;
  SConstString JoinSession_SessionId;
  bool JoinSession_IsFirstPartySession;
  uint JoinSession_Context;
};

struct SScenePhy_DelayedScene {
  IScenePhy* ScenePhy;
};

struct NSceneVisEntFx_STactical {
  float Radius;
  float Size;
  float MaxY;
  float MinY;
  float MinDotN;
  float Intens;
};

struct NSceneVisEntFx_SVisibilityState {
  NSceneVisEntFx::EVisible eVisible;
  float GhostTransparency;
  uint8 IsFirstPerson;
  uint8 IsInternalCam;
};

struct NSceneDyna_SMgrParams {
  bool EnableMultiThread;
  bool EnableContactCache;
  NSceneDyna_SSleepingParams DefaultSleepingParams;
  NSceneDyna_SSolverParams Solver;
  bool UseNarrowPhaseOutputV2;
  bool RandomizeContacts;
  bool RandomizeConstraints;
};

struct NSceneDyna_SSleepingParams {
  float SleepingVel;
  float SleepingAngVel;
};

struct NSceneDyna_SSolverParams {
  uint FrictionStaticIterCount;
  uint FrictionDynaIterCount;
  uint VelocityIterCount;
  uint PositionIterCount;
  float DepenImpulseFactor;
  float MaxDepenVel;
  bool EnablePositionConstraint;
  float AllowedPen;
  uint VelBiasMode;
  bool UseConstraints2;
  float MinVelocityForRestitution;
};

struct NSceneLayout_SLight {
  MwId Name;
  GmTransQuat Loc;
};

struct NSceneLayout_SItem {
  MwId Name;
  GmTransQuat Loc;
  CPlugSolid* SolidModel;
  CPlugSolid2Model* Mesh;
};

struct SSceneDestructibleRuntimePhy {
};

struct SSceneDestructibleFrozenPhy {
};

struct NSceneDyna_SKinematicSharedSignal {
  NPlugDyna_SKinematicConstraint* Model;
  float Phase;
  uint cRef;
};

struct NSceneDyna_SItemArrayDyn {
  MwFastBuffer<NSceneDyna_SItemState> States;
};

struct NSceneDyna_SItemState {
  vec3 Vel;
  vec3 AngularVel;
};

struct NSceneDyna_SItemCreateParams {
};

struct NSceneDyna_SKinematicConstraintCreateParams {
};

struct NSceneAdvert_SVisibilityTracker {
  uint DurationMs;
  float ThisFrameMaxCosAngle;
  const float ThisFrameAngleDeg;
  float ThisFrameAdVisibleSurface01;
  float ThisFrameScreenSurface01;
};

struct NSceneAdvert_SAdSystemParams {
  SConstString DbgName;
  bool GenerateVisibilities;
  bool DoVisibilityChecks;
  bool GenerateImpressions;
  float MinCosAngle;
  float MinAdVisibleSurface01;
  uint MinDurationMs;
};

struct SSceneMapColorisable01VisState {
  float Hue;
  float Gauge;
};

struct SSceneMapBaseVisState {
  float Hue;
  float Intensity;
};

struct NSceneAnim_SModel {
  const uint cRef;
  CPlugSkel* const PrimarySkel;
  CPlugAnimFile* const AnimFile;
  const uint GraphModelSize;
  const uint GraphInstanceSize;
  const uint cGraphNode;
};

struct NSceneAnim_SModelInst {
  NSceneAnim_SModel* Model;
  NSceneAnim_SModelInstInput Input;
  NPlugAnim_SSkelPose SkelPose;
};

struct NSceneAnim_SEdModelInstContext {
  MwSArray<SMetaPtr> Ptrs;
};

struct NSceneAnim_SNodeState_StateMachine {
  uint16 State;
  uint16 PrevState;
  uint StateStartTime;
};

struct NSceneAnim_SClipPlayer {
  NSceneAnim_SModel* Model;
  NSceneAnim_SClipPlayerInput Input;
};

struct NSceneAnim_SClipPlayerInput : public NSceneAnim_SInstInputBase {
};

struct NSceneAnim_SInstInputBase {
  int Lod; // Range: -1 - 2
};

struct NSceneAnim_SModelInstInput : public NSceneAnim_SInstInputBase {
  MwArrayInPlaceDyn<SMetaPtr> Contexts;
};

struct NSceneAnim_SNodeUpdateContextParams {
  uint Time;
  float Period;
  uint iLod;
  vec3 BulletHit_Relative_Pos;
  vec3 BulletHit_Incoming_Dir;
  bool BulletHit_Is_DirectHit;
};

struct NSceneSpectatorVis_SInput {
  uint SpectatorCount;
  float SpectatorPercent;
};

struct NSceneSpectatorVis_SLocGenOptions {
  NSceneSpectatorVis_SPositionGenOptions PointsOpts;
  NSceneSpectatorVis_SOrientationOpts OrientationOpts;
};

struct NSceneSpectatorVis_SPositionGenOptions {
  bool SortByTrajDist;
  bool GenerateNewPoints;
  bool RemoveExtraPoints;
  bool AvoidStaticColOverlap;
  uint cTargetPoints;
  float MinDistBetweenPoints; // Range: 0.1 - 1
  bool FindNearest_UseOptim;
  float FindNearest_Optim_DecimationDist; // Range: 0 - 100
};

struct NSceneSpectatorVis_SOrientationOpts {
  NSceneSpectatorVis::EOrientationMode Mode;
  bool WatchVisible_UseDecimateTraj;
  float WatchVisible_MaxDist; // Range: 0 - 1000
  float WatchVisible_MinDist; // Range: 0 - 10
  float DecimationInterval; // Range: 0 - 20
};

struct NSceneFxSystem_SRuntimeModel {
  const uint cRef;
  CPlugFxSystem* const FxSystem;
  CPlugSkel* const Skel;
  const uint Size;
  const uint InstanceSize;
};

struct NSceneFxSystem_SFxSystemInstanceInput {
  MwArrayInPlaceDyn<SMetaPtr> Contexts;
};

struct NSceneFxSystem_SFxSystemInstanceExpressionsUpdateParams {
  uint Time;
  float Period;
  float LodZMin;
};

struct NMotionPrediction_SParams {
  float KeepLocalLinearVelMinX;
  float KeepLocalLinearVelMaxX;
  float KeepLocalLinearVelMinY;
  float KeepLocalLinearVelMaxY;
};

struct CSceneVehicleVisParams {
};

struct NSceneAnim_SChar {
  NSceneAnim_SModel* Model;
  NSceneAnim_SCharInput Input;
  MwArrayInPlaceDyn<NSceneAnim_SGraphInstance> Layers;
  NSceneAnim_SCharState State;
};

struct NSceneAnim_SCharInput {
  vec3 Pos;
  vec3 Vel;
  vec3 WishMove;
  float AngularVel;
  float AimPitch; // Range: -1.5708 - 1.5708
  float AimYaw; // Range: -3.14159 - 3.14159
  float CrouchCoef; // Range: 0 - 1
  bool Aiming;
  bool Looking;
  bool Freezed;
  float DbgX; // Range: -1 - 1
  float DbgY; // Range: -1 - 1
  MwArrayInPlaceDyn<SMetaPtr> Contexts;
};

struct NSceneAnim_SGraphInstance {
  CPlugAnimGraph* Model;
  CPlugAnimGraphState* StateModel;
  NSceneAnim_SGraphState* StateInstance;
  float StateWeight;
  CPlugAnimGraphState* PrevStateModel;
  NSceneAnim_SGraphState* PrevStateInstance;
  CPlugAnimGraphTransition* TransitionModel;
  float TransitionTime;
};

struct NSceneAnim_SCharState {
  float CrouchCoef;
};

struct NSceneAnim_SGraphUpdateParams {
  uint Time;
  float RootYaw;
  float VelYaw;
  float WorldSpeedKmh;
};

struct NSceneAnim_SGraphState {
  uint StartTime;
  uint StateTime;
  NSceneAnim_SNode* Node;
};

struct NSceneAnim_SNode {
};

struct NSceneAnim_SNodeAim : public NSceneAnim_SNode {
};

struct NSceneAnim_SNodeProceduralAttractor : public NSceneAnim_SNode {
};

struct NSceneAnim_SNodeBlend2d : public NSceneAnim_SNode {
};

struct NSceneAnim_SNodeLocoGroup : public NSceneAnim_SNode {
  uint iLocoNode;
  uint iLocoNodePrev;
  NSceneAnim_SNodeClip Clip;
  NSceneAnim_SNodeClip ClipPrev;
  float ClipTransitionTime;
  float ClipWeight;
  float TurnClipStartRootYaw;
  float ClipTotalPlayingTime;
};

struct NSceneAnim_SNodeClip : public NSceneAnim_SNode {
  float ClipTimeSec;
  NSceneAnim_SRootYaw RootYaw;
};

struct NSceneAnim_SNodeSequence : public NSceneAnim_SNode {
  uint iSequenceClip;
  NSceneAnim_SNodeClip Clip;
};

struct NSceneAnim_SNodeJump : public NSceneAnim_SNode {
  uint iPhase;
  NSceneAnim_SNodeClip Clip;
};

struct NSceneAnim_SRootYaw {
  float Cur;
  float Target;
};

struct NScenePathFinding_SNavMeshBuildParams {
  float CellSize;
  float CellHeight;
  float AgentHeight;
  float AgentRadius;
  float AgentMaxClimb;
  float RegionMinSize;
  float RegionMergeSize;
  float EdgeMaxLen;
  float EdgeMaxError;
  float DetailSampleDist;
  float DetailSampleMaxError;
  float AgentMaxSlope;
  int VertsPerPoly;
  bool MonotonePartitioning;
  uint TileCountX;
  uint TileCountZ;
  float Epsilon;
  uint TileCount;
  vec3 AABBMin;
  vec3 AABBMax;
  float TileSizeX;
  float TileSizeZ;
  bool NoOffsetTest;
};

struct NScenePathFinding_STileCacheHandler {
  const uint Chunks_count;
  const uint Tiles_count;
  const uint Obstacles_count;
  const uint Pending_modified_count;
  const uint Pending_active_count;
  const uint Pending_inactive_count;
};

struct NScenePathFinding_SPathRequestState {
  NScenePathFinding_SPathRequestInput Input;
  NScenePathFinding_SPathRequestOutput Output;
  NScenePathFinding_SPathRequestData Data;
};

struct NScenePathFinding_SDynamicPathState {
  vec3 EndPoint;
  uint InputUpdateDate;
  vec3 CurrentPos;
  uint PathUpdateDate;
  uint PathPointIndex;
  MwFastBuffer<vec3> PathPoints;
  uint FirstModifiedTileIndex;
  uint FirstUpdatedTileIndex;
  MwFastBuffer<int2> PathTiles;
  MwFastBuffer<uint> PathTilesFirstPointIndex;
};

struct NScenePathFinding_SAgentModel {
  NScenePathFinding_SAgentSize AgentSize;
  uint iNavMesh;
  uint RefCount;
};

struct NScenePathFinding_SAgentSize {
  float AgentRadius;
  float AgentHeight;
  float ClimbHeight;
};

struct NScenePathFinding_SPathRequestInput {
  uint AgentModel;
  vec3 StartPoint;
  vec3 EndPoint;
  uint16 PolygonExcludeFlags;
  uint LinkExcludeFlags;
  bool ComputeSimplePath;
  bool ComputeDetailPath;
  bool ComputeCrossesDynamicArea;
};

struct NScenePathFinding_SPathRequestOutput {
  NScenePathFinding::EPathRequestStatus Status;
  MwSArray<vec3> SimplePath;
  MwSArray<vec3> DetailPath;
  MwSArray<uint> SimpleInDetailIndices;
  bool CrossesDynamicArea;
};

struct NScenePathFinding_SPathRequestData {
  MwFastBuffer<vec3> SimplePath;
  MwFastBuffer<vec3> DetailPath;
  MwFastBuffer<uint> SimpleInDetailIndices;
  bool CrossesDynamicArea;
};

struct NSceneVFX_SVFXInstanceInput {
  MwArrayInPlaceDyn<SMetaPtr> Contexts;
};

struct NSceneVFX_SVFXInstanceNodeUpdateContext {
  uint Time;
  float Period;
};

struct SPlugExpr {
  MwSArray<uint8> ByteCode;
  MwSArray<uint8> Constants;
};

struct NPlugGunLock_SParams {
  bool IsActive;
  bool UseFrustum;
  float Frustum_YXRatio;
  float Frustum_MaxZ;
  bool CanChangeTarget;
  float Frustum_FovX; // Range: 0 - 90
  float Frustum_FovY; // Range: 0 - 90
  uint TimeToLock;
  CPlugCurveSimpleNod* LockProgressMultiplierFromLockValue;
  bool AutoAim_Active;
  float AutoAim_Rate;
};

struct SPlugGraphNodeConnection {
  uint InputNode;
  uint InputSlot;
  uint OutputNode;
  uint OutputSlot;
};

struct SPlugGraphVar {
  MwId Name;
  NPlugExpr::EType Type;
  bool Bool8Val;
  int8 Int8Val;
  int16 Int16Val;
  int Int32Val;
  float FloatVal; // Range: 0 - 1
  float MinVal;
  float MaxVal;
  vec3 Float3Val;
  int8 Enum8Val;
  int16 Enum16Val;
  int Enum32Val;
  vec2 Float2Val;
  vec4 Float4Val;
  NGmUnit::EConvertMethod UIConvIntToExt;
};

struct SPlugStateMachine_State {
  MwId Name;
  CPlugGraphNode_Graph* Graph;
  bool Is_Default_Transition;
  SPlugStateMachine_TransitionData Default_Transition_Type;
};

struct SPlugStateMachine_TransitionData {
  string Cond_Expr;
  EPlugStateMachineTransitionType Type;
  float Duration;
  EPlugStateMachineTransitionCurve Curve;
};

struct SPlugStateMachine_Transition {
  string Cond_Expr;
  EPlugStateMachineTransitionType Type;
  float Duration;
  EPlugStateMachineTransitionCurve Curve;
  uint Eval_Priority;
};

struct NPlugGpuHlsl_D3D12_SRootSign {
  MwArrayInPlaceDyn<NPlugGpuHlsl_D3D12_NRootSign_SParam> Params;
  const uint cSampler;
};

struct NPlugGpuHlsl_D3D12_NRootSign_SParam {
  const UnknownType Type;
  SConstString const Stage;
};

struct SPlugAnimUIRigNodeTypeConfig {
  EPlugAnimGraphNodeType Type;
  RGBAColor Color;
};

struct NPlugAnim_SPoseEditor_Model {
  GmTransQuat Head;
  GmTransQuat UpperSpine;
  GmTransQuat MiddleSpine;
  GmTransQuat LowerSpine;
  GmTransQuat LUpperArm;
  GmTransQuat LLowerArm;
  GmTransQuat LUpperLeg;
  GmTransQuat LLowerLeg;
  GmTransQuat LFoot;
  GmTransQuat RUpperArm;
  GmTransQuat RLowerArm;
  GmTransQuat RUpperLeg;
  GmTransQuat RLowerLeg;
  GmTransQuat RFoot;
  void Export_pose_as_C___code();
};

struct NPlugAnim_SAvatarAnimV0_Model {
  NPlugAnim_SAvatarAnimV0_Model_Locomotion Locomotion;
  NPlugAnim_SAvatarAnimV0_Model_Tiredness Tiredness;
  NPlugAnim_SAvatarAnimV0_Model_Firing Firing;
  NPlugAnim_SAvatarAnimV0_Model_WeightInfluence WeightInfluence;
  MwId VarId_WalkCycle;
  NPlugCurve_SRichCurveInPlace4 Temp_RichCurve;
};

struct NPlugAnim_SAvatarAnimV3_Model_Global {
  NPlugAnim_SAvatarAnimV3_Model_Locomotion Locomotion;
  NPlugAnim_SAvatarAnimV3_Model_Jump Jump;
  NPlugAnim_SAvatarAnimV3_Model_Swim Swim;
  NPlugAnim_SAvatarAnimV3_Model_Idle Idle;
  NPlugAnim_SAvatarAnimV3_Model_Fire Fire;
  NPlugAnim_SAvatarAnim_Model_Rest Rest;
  NPlugAnim_SAvatarAnim_Model_Climb Climb;
  float Legs_Thickness;
};

struct NPlugAnim_SAvatarAnimV3_Model_Locomotion {
  bool ForceGaitModel;
  MwId GaitModelIdToUse;
  bool UseRayCastsToPreventFeetFromPenetratingTheGround;
  MwArrayInPlaceDyn<NPlugAnim_SAvatarAnimV3_Model_Gait> GaitModels;
  MwArrayInPlaceDyn<NPlugAnim_SGaitSelector> GaitSelectors;
  NPlugAnim_SLocomotionRotationModel BodyRotation;
};

struct NPlugAnim_SAvatarAnimV3_Model_Jump {
  float AnimDuration;
  float K0ToK1Transition_NormedStartTime;
  float K0ToK1Transition_NormedDuration;
  float K1ToK2Transition_NormedStartTime;
  float K1ToK2Transition_NormedDuration;
  float BodyRotationVelocity_DegPerSec;
};

struct NPlugAnim_SAvatarAnimV3_Model_Swim {
  float MasterFreq;
  float Head_Pitch_Degrees;
  float Head_RollAmplitude_Degrees;
  float Arms_SpreadAngle_Degrees;
  float Arms_Phi_Degrees;
  float Body_RollAmplitude_Degrees;
  float Body_BuoyancyAmplitude;
  float Legs_FreqCoef;
  float Legs_Amplitude_Degrees;
  float Legs_SpreadAngle_Degrees;
};

struct NPlugAnim_SAvatarAnimV3_Model_Idle {
  float Head_YawMax_Degrees;
  float Arms_Frequency;
  float UpperArms_RxMax_Degrees;
  float LowerArms_RxMax_Degrees;
  float Breathing_Frequency;
  float UpperSpine_TxStabilizationFactor;
  float UpperSpine_TzStabilizationFactor;
  float UpperSpine_RotationStabilizationFactor;
  float MiddleSpine_RxMax_Degrees;
  float MiddleSpine_TyMax;
  float MiddleSpine_YawMax_Degrees;
  float LowerSpine_RxMin_Degrees;
  float LowerSpine_RxMax_Degrees;
  float LowerSpine_RyMin_Degrees;
  float LowerSpine_RyMax_Degrees;
  float LowerSpine_TxMax;
  float LowerSpine_TyMax;
  float LowerSpine_TzMax;
  float Legs_Frequency;
};

struct NPlugAnim_SAvatarAnimV3_Model_Fire {
  uint AnimDuration;
  float AnimBlendNormalizedDuration;
};

struct NPlugAnim_SAvatarAnim_Model_Rest {
  float SeatInSlopeDirection_MinElevationGain;
  float HunchCoef; // Range: 0 - 1
  float LaidBackCoef; // Range: 0 - 1
  float Arms_SpreadCoefMax; // Range: 0 - 1
  float Legs_SwingFrequency;
  float Legs_SpreadAngleMax_Degrees;
  float UpperLegs_SwingAmplitude_Degrees;
  float LowerLegs_SwingAmplitude_Degrees;
  float SpinningDuration;
  float SittingDuration;
};

struct NPlugAnim_SAvatarAnim_Model_Climb {
  vec2 Left_hand_offset_from_left_shoulder;
  vec2 Right_hand_offset_from_right_shoulder;
  vec2 Left_foot_offset_from_left_hip;
  vec2 Right_foot_offset_from_right_hip;
  float HoldDistance;
  float ReachDuration;
  float ReachElevation;
  float ReachDeviation;
};

struct NPlugAnim_SAvatarAnimV3_Model_Gait {
  MwId Id;
  float FrequencyMin;
  float FrequencyMax;
  float Head_RxStabilizationFactor;
  float Head_RyStabilizationFactor;
  float Head_RzStabilizationFactor;
  float Head_PitchCoef;
  float Head_PitchMax_Degrees;
  float Head_PitchMin_Degrees;
  float Head_YawMax_Degrees;
  float UpperArms_RxMin_Degrees;
  float UpperArms_RxMax_Degrees;
  float UpperArms_RyMax_Degrees;
  float UpperArms_Rz_Degrees;
  float LowerArms_RxFreqCoef;
  float LowerArms_RxMax_Degrees;
  float MiddleSpine_RxStabilizationFactor;
  float MiddleSpine_RyStabilizationFactor;
  float MiddleSpine_RzStabilizationFactor;
  float MiddleSpine_YawMax_Degrees;
  float LowerSpine_TyOffset;
  float LowerSpine_TyAmplitude;
  float LowerSpine_RxOffset_Degrees;
  float LowerSpine_RxAmplitude_Degrees;
  float LowerSpine_RyOffset_Degrees;
  float LowerSpine_RyAmplitude_Degrees;
  float LowerSpine_RzOffset_Degrees;
  float LowerSpine_RzAmplitude_Degrees;
  float LowerSpine_TiltRxCoef;
  float LowerSpine_TiltRxMax_Degrees;
  float LowerSpine_TiltRzCoef;
  float LowerSpine_TiltRzMax_Degrees;
  float UpperLegs_RxMin_Degrees;
  float UpperLegs_RxMax_Degrees;
  float UpperLegs_Ry_Degrees;
  float UpperLegs_RzMin_Degrees;
  float UpperLegs_RzMax_Degrees;
  float LowerLegs_Phi;
  float Feet_GroundPhaseNormedStartTimes_Nat_EAvatarSide__Left__;
  float Feet_GroundPhaseNormedStartTimes_Nat_EAvatarSide__Right__;
  float Feet_GroundPhaseNormedDurations_Nat_EAvatarSide__Left__;
  float Feet_GroundPhaseNormedDurations_Nat_EAvatarSide__Right__;
  float Feet_TyOffsetMax;
  float Feet_TyOffsetToMaxNormedDt;
  float Strafe_LowerSpineAngleMax_Degrees;
  float Strafe_ForwardToReverseHysteresisAngle_Degrees;
};

struct NPlugAnim_SGaitSelector {
  float MinHorizontalVelocity_KmH;
  float MaxHorizontalVelocity_KmH;
  MwId GaitModelId;
};

struct NPlugAnim_SLocomotionRotationModel {
  float LowerBodyYawAngularVelocity_DegPerSec;
  float UpperBodyYawAngularVelocity_DegPerSec;
  float Strafe_LowerBodyYawAngularVelocity_DegPerSec;
  float Strafe_UpperBodyYawAngularVelocity_DegPerSec;
};

struct NPlugAnim_SAvatarAnimV0_Model_Locomotion {
  float BaseAmplitude_Min_Degrees;
  float BaseAmplitude_Max_Degrees;
  float BaseAmplitude_VelocityCoef;
  float BaseFreq_Max;
  float BaseFreq_VelocityCoef;
  float LowerLeg_Phi;
  float UpperArmAmplitude_Coef;
  float UpperArmRz_Coef_Degrees;
  float UpperArmRz_Min_Degrees;
  float UpperArmRz_Max_Degrees;
  float LowerArmAmplitude_Coef;
  float LowerSpineAmplitude_Coef;
  float LowerLegAmplitude_Coef;
  float UpperLegOffset_Coef;
  float LowerSpineUpAndDownAmplitude_Coef;
  float LowerSpineUpAndDownAmplitude_Min;
  float LowerSpineUpAndDownAmplitude_Max;
  float LowerSpineLRAmplitude_Coef;
  float LowerSpineLRAmplitude_Min;
  float LowerSpineLRAmplitude_Max;
  float AnimAcceleration;
  float BackToIdle_Duration;
  float TiltWhenTurning_Global_Max_Degrees;
  float TiltWhenTurning_Global_Coef;
  float TiltWhenTurning_Global_Max_Degrees;
  float TiltWhenTurning_Global_Coef;
  float TiltWhenTurning_LowerSpineRz_Max_Degrees;
  float TiltWhenTurning_LowerSpineRz_Coef;
  float TiltWhenTurning_LowerSpineRx_Max_Degrees;
  float TiltWhenTurning_LowerSpineRx_Coef;
  float Head_Pitch_Max_Degrees;
  float Head_Pitch_Min_Degrees;
  float Head_Pitch_MinMaxInvCoef;
};

struct NPlugAnim_SAvatarAnimV0_Model_Tiredness {
  float Tiredness;
};

struct NPlugAnim_SAvatarAnimV0_Model_Firing {
  uint AnimDuration;
  float AnimBlendNormalizedDuration;
};

struct NPlugAnim_SAvatarAnimV0_Model_WeightInfluence {
};

struct NPlugCurve_SRichCurveInPlace4 : public NPlugCurve_SRichCurve {
};

struct SPlugAnimClipGroup {
  MwId Name;
  bool Is_Synchro;
};

struct NPlugItem_SVariant {
  MwSArray<NPlugItemPlacement_STag> Tags;
  CMwNod* EntityModel;
  bool HiddenInManualCycle;
  CSystemFidFile* const EntityModelFidForReload;
};

struct NPlugItemPlacement_SPatchLayoutDeprec {
  NPlugItemPlacement::EStartPos StartPos;
  NPlugItemPlacement::EFillMode FillMode;
  uint FixedStepCount; // Range: 1 - 10
  NPlugItemPlacement::EFillDir FillOrder;
  NPlugItemPlacement::ESiblingDistribution SiblingDistribution;
  uint UVSpacing_x; // Range: 0 - 10
  uint UVSpacing_y; // Range: 0 - 10
  bool CursorCentered;
};

struct NPlugItemPlacement_SPatchLayout {
  NPlugItemPlacement::EFillDir FillDir;
  uint ItemCount; // Range: 0 - 10
  float ItemSpacing; // Range: 0 - 16
  NPlugItemPlacement::EAlign FillAlign;
  float FillBorderOffset;
  float Altitude;
  float NormedPos; // Range: 0 - 1
  float DistFromNormedPos;
  MwFastBuffer<MwId> OnlyOnGroups;
};

struct NPlugItemPlacement_SIdGroup {
  MwId Name;
};

struct NPlugItemPlacement_SSizeGroup {
  MwId Name;
  vec3 Size;
};

struct NPlugItemPlacement_SPlacementOption {
  MwSArray<NPlugItemPlacement_STag> RequiredTags;
};

struct NPlugItemPlacement_STypeDef {
  SConstString Name;
  MwSArray<SConstString> Values;
};

struct NPlugItemPlacement_STable {
  UnknownType Types;
};

struct NPlugItemPlacement_STypeInternal {
  SConstString Name;
  UnknownType Values;
};

struct NPlugBulletModel_SElectroPulse {
  float Value;
};

struct NPlugBulletModel_SDamage {
  int DirectHitValue;
  int Value;
  float Radius;
  float RadiusAttenuation;
};

struct NPlugBulletModel_SBlow {
  float Value;
  float Radius;
  float RadiusAttenuation;
  float VerticalScale;
};

struct NPlugAnim_SRigPose {
};

struct SPlugSpawnState {
  iso4 BaseLoc;
  CPlugSpawnModel* SpawnModel;
};

struct GmBoxAligned {
  vec3 m_Center;
  vec3 m_HalfDiag;
};

struct SCamShakeParams {
  uint Duration;
  uint FadeDuration;
  float PosOffset;
  float PosFreq;
  float RotOffsetDeg;
  float RotFreq;
  float ImpactIntensityMin;
};

struct NPlugAnim_SSkelImportJointLod {
  string JointMatchName;
  bool IncludeChildren;
  uint8 LodValue;
};

struct NPlugAnim_SPoseGridImportPose {
  uint X;
  uint Y;
  MwId FbxName;
  MwId TakeName;
  uint Frame;
};

struct NPlugAnim_SChannelGroupJointImport {
  MwId Name;
  float Weight;
};

struct SMwIdRefSkel : public SMwIdRefBase {
};

struct SMwIdRefRig : public SMwIdRefBase {
};

struct SMwIdRefRigToSkel : public SMwIdRefBase {
};

struct SMwIdRefChannelGroup : public SMwIdRefBase {
};

struct SMwIdRefJointExprGroup : public SMwIdRefBase {
};

struct SMwIdRefClip : public SMwIdRefBase {
};

struct NPlugAnim_SJointExpr {
  MwId JointName;
  NPlugAnim::EJointChannel Channel;
  SConstString Expr;
};

struct SFastRange {
  uint Start;
  uint Count;
};

struct GmTransYaw {
  vec3 Trans;
  float Yaw;
};

struct NPlugAnim_SAssetFiles {
  MwSArray<SMwIdFid> Skels;
  MwSArray<SMwIdFid> Clips;
  MwSArray<SMwIdFid> ChannelGroups;
  MwSArray<SMwIdFid> JointExprGroups;
  MwSArray<SMwIdFid> Rigs;
  MwSArray<SMwIdFid> RigToSkels;
};

struct SMwIdFid {
  MwId Id;
  CSystemFidFile* Fid;
};

struct NPlugAnim_SAssetPaths {
  MwSArray<SMwIdString> Skels;
  MwSArray<SMwIdString> Clips;
  MwSArray<SMwIdString> ChannelGroups;
  MwSArray<SMwIdString> JointExprGroups;
  MwSArray<SMwIdString> Rigs;
  MwSArray<SMwIdString> RigToSkels;
};

struct SMwIdString {
  MwId Id;
  SConstString String;
};

struct SMwIdRefBase {
  MwId Id;
  CSystemFidFile* Fid;
  CMwNod* NodRef;
};

// userName: 'AdnTag'
struct NPlugAdn_STag {
  NPlugAdn_STagType Type;
  NPlugAdn_STagValue Value;
};

struct SPlugRandomMetaData {
  CPlugMetaData* MetaData;
  float Proba;
};

struct SPlugAdnRandomGenSet {
  MwFastBuffer<NPlugAdn_STag> RequiredTags;
  MwFastBuffer<NPlugAdn_STag> ExcludedTags;
  MwFastBuffer<NPlugAdn_STag> AnimRequiredTags;
};

struct NPlugAdn_STagType {
  uint Index;
};

struct NPlugAdn_STagValue {
  uint Index;
};

struct NPlugAdn_SContext {
  MwFastBuffer<NPlugAdn_STagFid> AllParts;
  MwFastBuffer<NPlugAdn_STagFid> AllClips;
};

struct NPlugAdn_STagTypeDef {
  SConstString Name;
  MwSArray<SConstString> Values;
  bool IsUnique;
  float PresenceProba;
  RGBAColor Color;
};

struct NPlugAdn_STagTable {
  UnknownType Types;
  NFastBucketAlloc_SAllocator BucketAlloc;
};

struct NPlugAdn_STagTypeInternal {
  SConstString String;
  UnknownType Values;
  float PresenceProba;
  RGBAColor Color;
  const bool IsInDb;
};

struct NPlugAdn_STagValueInternal {
  SConstString String;
  const bool IsInDb;
};

struct NPlugMapAI_SNode {
};

struct NPlugMapAI_SNode_SpawnCitizen : public NPlugMapAI_SNode {
  MwId CitizenTypeId;
  MwArrayInPlaceDyn<NPlugAdn_STag> RequiredTags;
  MwArrayInPlaceDyn<NPlugAdn_STag> ExcludedTags;
};

struct NPlugTVScreen_SModelInfo {
};

struct NPlugTVScreen_SSkinModel {
};

struct NPlugClassicSkin_SSkinModel_SFxSystem {
  float EmitterLinearHue01;
  float EmitterHueLightness;
};

struct NPlugClassicSkin_SSkinModel {
  NPlugClassicSkin_SSkinModel_SFxSystem FxSystem;
};

struct NPlugMatRemap_SSkinModel {
};

struct NPlugParallaxScreen_SSkinModel {
  MwArrayInPlaceDyn<NPlugParallaxScreen_SSkinModel_Layer> Layers;
};

struct NPlugParallaxScreen_SSkinModel_Layer {
  string Name;
  float Depth;
  bool HasVerticalTiling;
};

struct NPlugSkinManialink_SSkinModel {
  string ManialinkFileName;
};

struct NPlugSkinnedModel_SMgr {
  MwFastBuffer<NPlugSkinnedModel_SModel*> AllModels;
  MwFastBuffer<NPlugSkinnedModel_SSkin*> AllSkins;
  const uint LightBitmapsCount;
  bool AutoDestroyUnusedModels;
  void DestroyUnusedModels();
};

struct NPlugSkinnedModel_SModel {
  CMwNod* BaseModel;
  CMwNod* SkinnedModel;
  uint UsageCount;
  NPlugSkinnedModel_SSkin* Skin;
};

struct NPlugSkinnedModel_SSkin {
  CPlugGameSkin* SkinDesc;
  CSystemPackDesc* BasePackDesc;
  CSystemPackDesc* ForegroundPackDesc;
  uint UsageCount;
  SMetaPtr SkinModel;
};

struct NPlugSkinnedModel_SImage_SSampler {
  EGxTexAddress AddressU;
  EGxTexAddress AddressV;
};

struct NPlugSkinnedModel_SImage {
  NPlugSkinnedModel_SImage_SSampler Sampler;
};

struct NPlugModelKit_SModelLib {
  MwFastBuffer<NPlugModelKit_SModelAssets> Models;
  bool AutoCleanUnusedModels;
  void CleanUnusedModels();
};

struct NPlugModelKit_SModelAssets {
  MwSArray<NPlugModelKit_SPartAssets> PartAssets;
  MwSArray<uint8> FullSignature;
  uint cRef;
};

struct NPlugModelKit_SPartAssets {
  CPlugSolid2Model* MeshShaded;
  CPlugAnimFile* AnimFile;
  uint iParentPart;
  uint16 ParentPartJoint;
  uint8 UseParentRig;
};

struct NPlugModelShading_SMaterialFiles {
};

struct NPlugModelKit_SPartFiles {
  CSystemFidFile* Mesh;
  MwSArray<uint> MeshMaterials;
  CSystemFidFile* AnimFile;
  NPlugAnim_SAssetFiles AnimAssetFiles;
  CSystemFidFile* FxSystem;
};

struct NPlugModelKit_SPartFilesVersion {
  MwSArray<uint8> FullOptionVals;
  uint Files;
};

struct NPlugModelKit_SPartFilesVersions {
  SConstString Name;
  SFastRange RangeInAllPartVersions;
  SConstString ParentPart;
  SConstString ParentPartJoint;
  bool UseParentRig;
  SConstString JointAddPrefix;
};

struct NPlugModelKit_SOption {
  SConstString Name;
  NPlugModelKit_SOptionVal OptionCond;
  MwSArray<SConstString> Vals;
  bool IsUsed;
  NPlugModelKit::EOptionAutoFill AutoFill;
};

struct NPlugModelKit_SOptionVal {
  uint8 iOption;
  uint8 iVal;
};

struct NPlugModelKit_SOptionArray {
  MwSArray<NPlugModelKit_SOptionVal> Options;
};

struct NPlugModelKit_SModelFiles {
  MwSArray<NPlugModelKit_SOptionVal> Conds;
  MwSArray<uint> Parts;
};

struct NPlugModelKit_SPartPaths {
  SConstString Name;
  SConstString Mesh;
  SConstString Shading;
  MwSArray<NPlugModelKit_SFolder> ShadingFolders;
  SConstString AnimFile;
  MwSArray<NPlugModelKit_SFolder> AnimFolders;
  SConstString FxSystem;
  SConstString ParentPart;
  SConstString ParentPartJoint;
  bool UseParentRig;
  SConstString JointAddPrefix;
};

struct NPlugModelKit_SOptionDesc {
  SConstString Name;
  NPlugModelKit_SOptionValDesc OptionCond;
  NPlugModelKit::EOptionAutoFill AutoFill;
  MwSArray<SConstString> Vals;
};

struct NPlugModelKit_SOptionValDesc {
  SConstString Name;
  SConstString Val;
};

struct NPlugModelKit_SOptionArrayDesc {
  MwSArray<NPlugModelKit_SOptionValDesc> Options;
};

struct NPlugModelKit_SModelDesc {
  MwSArray<NPlugModelKit_SOptionValDesc> OptionConds;
  MwSArray<SConstString> PartNames;
  bool MergeMeshes;
};

struct NPlugModelKit_SFolder {
  SConstString Path;
  bool Optional;
};

struct NPlugPrefab_SEnt_Void {
  CMwNod* Model;
  GmTransQuat Location;
};

struct NPlugRoadChunk_SParams {
  EPlugRoadChunkType ChunkType;
  EPlugRoadWayType WayType;
  EPlugRoadType RoadType;
  EPlugRoadDensity RoadDensity;
  bool IsInsertion;
  bool OneWayFreeDir;
  bool IsTwoWays;
  bool Taggable;
  bool PatchUseEndVertsForUs;
  uint cLanePerWay;
  float RightOffset;
  float LeftOffset;
  MwId GroupId;
  quat InteriorOffset;
  MwId CitizenTypeId;
};

struct SPlugTrafficLight {
  uint8 iWay;
  uint8 iGroup;
};

struct NPlugAnim_SEntTrack {
  NPlugAnim_SEntDesc EntDesc;
  MwSArray<NPlugAnim_SEntBlock> EntBlocks;
};

struct NPlugAnim_SEntClip {
  MwClassId EntClassId;
  MwSArray<NPlugAnim_SPropClip> Props;
};

struct NPlugAnim_SEntDesc {
  MwClassId ClassId;
  SConstString Name;
};

struct NPlugAnim_SEntBlock {
  uint iEntClip;
  int Begin;
  int End;
  int OffsetAtBegin;
};

struct NPlugAnim_SPropClip {
  NPlugAnim_SPropDesc Desc;
  NPlugCurve_SRichCurve Curve;
};

struct NPlugAnim_SPropDesc {
  SConstString Name;
  NPlugAnim::EPropType Type;
};

struct NPlugCurve_SRichCurve {
  MwSArray<NPlugCurve_SRichCurveKey> Keys;
};

struct NPlugNoise_SParams {
  float ScaleX;
  float ScaleY;
  float ScaleZ;
  float ScaleT;
  float PeriodX;
  float PeriodY;
  float PeriodZ;
  float PeriodT;
  float ValMin;
  float ValMax;
};

struct NPlugModelLodMesh_SRange {
  float DistMin;
  float DistMax;
  MwFastBuffer<CPlugModelMesh*> Meshes;
  MwFastBuffer<iso4> MeshLocations;
};

struct NPlugGrass_SMatterArray {
  MwFastBuffer<NPlugGrass_SModel> Models;
  MwArrayInPlaceDyn<NPlugGrass_SMatter> Matters;
};

struct NPlugGrass_SModel {
  CSystemFidFile* Fid;
  float RandomScale; // Range: 0 - 1
};

struct NPlugGrass_SMatter {
  string Name;
  MwFastBuffer<NPlugGrass_SMatterModel> Models;
};

struct NPlugGrass_SMatterModel {
  float Spacing;
};

struct GmSurfaceIds {
  EPlugSurfaceMaterialId PhysicId;
  EPlugSurfaceGameplayId GameplayId;
};

struct NPlugSkel_SJointLodSetup {
  string JointMatchName;
  bool IncludeChildren;
  uint8 MaxLod;
};

struct NPlugFilePreloader_SPreloadDesc {
  MwSArray<NPlugFilePreloader_SPreloadDescGroup> Groups;
};

struct NPlugFilePreloader_SPreloadDescGroup {
  SConstStringInt Container;
  SConstString BatchName;
  MwSArray<NPlugFilePreloader_SPreloadDescFid> Fids;
};

struct NPlugFilePreloader_SPreloadDescFid {
  NPlugFilePreloader::EAllocType AllocType;
  SConstStringInt FileName;
  uint Size;
};

struct NPlugModelShading_SText_SShading_STmCarPrestige {
  MwArrayInPlaceDyn<NPlugModelShading_SText_SShading_STmCarPrestige_SMedal> Medals;
};

struct NPlugModelShading_SText_SShading_STmCarPrestige_SMedal {
  UnknownType Matters_BaseColor;
  UnknownType Matters_Rough;
  UnknownType Matters_RoughId;
  UnknownType Matters_Metal;
  RGBAColor Decal_BaseColor;
  uint8 Decal_Rough;
  uint8 Decal_Metal;
};

struct NPlugModelShading_SText_SShading_SSampler {
  MwId Name;
  EGxTexAddress AddressU;
  EGxTexAddress AddressV;
};

struct NPlugModelShading_SText_SShading_SColor {
  MwId Name;
  RGBAColor Value;
};

struct NPlugModelShading_SText_SShading_SConstant {
  MwId Name;
  float Value;
};

struct NPlugModelShading_SText_SShading {
  MwFastBuffer<NPlugModelShading_SText_SShading_SConstant> Constants;
  MwFastBuffer<NPlugModelShading_SText_SShading_SColor> Colors;
  MwFastBuffer<NPlugModelShading_SText_SShading_SSampler> Samplers;
  NPlugModelShading_SText_SShading_STmCarPrestige* TmCarPrestige;
  int SortIndex;
  string DecalBitmapRender;
};

struct NPlugModelShading_SText {
  NPlugModelShading_SText_SShading Shading;
};

struct NPlugModelShading_STextureSlotFile {
  CSystemFidFile* Texture;
};

struct NPlugModelShading_SDecalRender_StadiumSaisons {
};

struct SPlugParticleLaserEnergyStyle {
  vec3 ColorCenter;
  vec3 ColorBorder;
  float ColorLerpPow;
  float ColorCenterMultiplier;
  float ColorBorderMultiplier;
};

struct SPlugParticleLaserEnergyModel {
  float Length;
  float Radius;
  SPlugParticleLaserEnergyStyle Style;
};

struct SPlugParticleEmitStateFromImpactModel {
  float DirNormalCoef;
  float DirTangentialCoef;
};

struct SPlugParticleSimulatedSmokeModel {
  uint LifeSpanMs;
  float InfluenceRadius;
  float Gravity;
  NPlugCurve_SSimpleCurveInPlace7 ParticleSizeOverLife;
  NPlugCurve_SSimpleCurveInPlace7 ParticleAlphaOverLife;
  float ParticleBaseSize;
  float ParticleSizeRandomness;
  float ParticlePositionRandomRadius;
  float ParticlePositionRandomness;
  float ParticleLifeDistance;
  vec3 ParticleBaseColor;
  NPlugCurve_SSimpleCurveInPlace7 FilamentStrengthOverLife;
  float FilamentBaseStrength;
  float FilamentPositionRandomRadius;
  float FilamentPositionRandomness;
};

struct SPlugParticlePrecalcModel {
  bool Enabled;
  uint PartCount;
  uint SampleRate;
};

struct SPlugParticlePhysicsModel {
  float WeightBirth;
  float WeightBirthVariation;
  float FluidFrictionBirth;
  float FluidFrictionBirthVariation;
  bool FluidFrictionBirthUseIntensity;
  float FluidFrictionBirthIntensityBase;
  bool RelativeToEmitter;
  bool CollisionEnabled;
  float CollisionBounce;
  float CollisionDamper;
  float CollisionRadius;
};

struct SPlugParticleRenderModel {
  EPlugParticleMultiStateRenderMode MultiStateRenderMode;
  EPlugParticleStandardRenderMode StandardRenderMode;
  bool SortSprites;
  float ViewDistMax;
  CPlugMaterial* Material;
  CPlugShader* Shader;
  CPlugSolid2Model* Mesh;
  wstring MeshRef;
  bool MeshUseInstancing;
  EPlugMeshInstanceType MeshInstanceType;
  NPlugCurve_SSimpleCurveInPlace7 MeshScaleOverLife;
  NPlugCurve_SSimpleCurveInPlace7 MeshRampMinOverLife;
  NPlugCurve_SSimpleCurveInPlace7 MeshRampMaxOverLife;
  iso4 MeshLocOffset;
  bool MeshLocUseScale;
  bool MeshLocScaleFromIntensity;
  uint CircularTrailVertPerPartCount;
  bool LightTrail_Hack_ImmobileEmitter;
  float SizeBirthRatioXY;
  vec2 SpritePivotPoint;
  EPlugParticleTextureAtlas TextureAtlas;
  uint TextureAtlasDimX;
  uint TextureAtlasDimY;
  uint TextureAtlasCount;
  bool TextureAtlasTopToBottom;
  uint TextureAtlasFixedIndex;
  uint TextureAtlasAnimationPeriodMs;
  bool TextureAtlasAnimationRandomPhase;
  NPlugCurve_SSimpleCurveInPlace4 IntensityFilter;
  float SizeBirth;
  float SizeBirthVariation;
  bool SizeBirthUseEmissionZone;
  float SizeBirthEmissionZoneScale;
  bool SizeBirthUseIntensity;
  NPlugCurve_SSimpleCurveInPlace7 SizeBirthFromSpeedKmh;
  float SizeK;
  NPlugCurve_SSimpleCurveInPlace4 SizeOverLife;
  bool SizeUseSizeX;
  NPlugCurve_SSimpleCurveInPlace4 SizeXOverLife;
  float BeamLengthSpeedScale;
  float BeamLengthMax;
  float RollSpeedBirth;
  float RollSpeedBirthVariation;
  NPlugCurve_SColorGradient ColorGradient;
  EPlugParticleColorGradient ColorGradientUse;
  bool ColorModulateWithTransparency;
  EPlugParticleRenderColor ColorType;
  bool ColorBirthUseIntensity;
  float TransparencyBirth;
  float TransparencyBirthVariation;
  bool TransparencyBirthUseIntensity;
  NPlugCurve_SSimpleCurveInPlace4 TransparencyOverLife;
  float UScaleDist;
  float VScaleDist;
  float WaterSplashVelY;
  CGxLightBall* Light;
  NPlugCurve_SSimpleCurveInPlace4 LightRadiusOverLife;
  NPlugCurve_SColorGradient LightColorOverLife;
  bool LightUseEmitterHue;
  bool LightColorModulateWithTransparency;
};

struct NPlugCurve_SSimpleCurveInPlace4 : public NPlugCurve_SSimpleCurve {
};

struct NPlugCurve_SColorGradient {
  float Xs_1_;
  float Xs_2_;
  vec3 Ys_0_;
  vec3 Ys_1_;
  vec3 Ys_2_;
  vec3 Ys_3_;
};

struct SPlugParticleLifeModel {
  float Life;
  float LifeVariation;
  uint LifePeriodMs_OnePart;
};

struct SPlugParticleSpawnModel {
  uint StartTime;
  uint EndTime;
  void _Set_Null_EndTime_();
  EPlugParticleSpawnCond Cond;
  float Period;
  float MinDist;
  iso4 LocOffset;
  NPlugCurve_SSimpleCurveInPlace4 IntensityFromSpawnNormLife;
};

struct SPlugVFXNodeConnection {
  uint InputNode;
  uint InputSlot;
  uint OutputNode;
  uint OutputSlot;
};

struct NPlugVeget_STreeModel {
  float Impostor_Lod_Dist;
  NPlugImpostor::EPlaneComputeMode Impostor_Plane_Mode;
  float ReductionRatio01; // Range: 0 - 0.3
  float Params_AngleMax_RotXZ_Deg; // Range: 0 - 20
  bool Params_EnableRandomRotationY;
  MwArrayInPlaceDyn<NPlugVeget_STreeLodModel> LodModels;
  UnknownType LodMaxDists;
  MwArrayInPlaceDyn<NPlugVeget_SMaterial> Materials;
  bool Params_Force_No_Collision;
  bool Params_Impostor_AllPlanesVisible;
  bool Params_ReceivesPSSM;
  NPlugVeget_STreeLeafPropagationModel Propagation;
};

struct NPlugVeget_STreeLodModel {
  MwArrayInPlaceDyn<NPlugVeget_SShadedGeom> SGs;
};

struct NPlugVeget_SMaterial {
  CPlugMaterial* PlugMat;
  CPlugFileImg* Color;
  CPlugFileImg* Normal;
  CPlugFileImg* Roughness;
  CPlugVegetMaterialVariation* Veget_Variation;
  CPlugVegetSubSurfaceParams* Veget_SubSurface;
};

struct NPlugVeget_STreeLeafPropagationModel {
  CPlugBitmap* Render_BaseColor;
  CPlugBitmap* Render_Normal;
  nat2 Render_c2AtlasGrid;
  float Render_LeafSize;
  uint Phy_cGroundGenPoint;
  uint Phy_cLeafPerSecond;
  float Phy_ConeHalfAngleDeg;
  float Phy_EmissionSpawnRadius;
  bool Phy_Enable;
  UnknownType EmissionPoss;
};

struct NPlugVeget_SShadedGeom {
  CPlugVisual* Visual;
  uint16 iModelMat;
};

struct NPlugDyna_SAnimFunc01 {
  MwArrayInPlaceDyn<NPlugDyna_SAnimFunc01Base> SubFuncs;
};

struct NPlugDyna_SAnimFuncNat {
  MwArrayInPlaceDyn<NPlugDyna_SAnimFuncNatBase> SubFuncs;
};

struct NPlugDyna_SAnimFuncNatBase {
  uint DurationMs;
  uint Value;
};

struct NPlugDyna_SAnimFunc01Base {
  NPlugDyna::EAnimFuncBase FuncBase;
  bool InverseY;
  uint DurationMs;
};

struct SPlugDynaWaterModel {
};

struct SCBuffer {
};

struct NPlugCurve_SRichCurveInPlace7 : public NPlugCurve_SRichCurve {
};

struct NPlugCurve_SRichCurveKey {
  NPlugCurve::EInterpMode InterpMode;
  NPlugCurve::ETangentMode TangentMode;
  float InTangent;
  float InWeight;
  float OutTangent;
  float OutWeight;
  float x;
  float y;
};

struct NPlugCurve_SSimpleCurve {
};

struct NPlugBitmap_SAtlasCubeInElem {
  uint i2Pixel_Start_x;
  uint i2Pixel_Start_y;
  uint i2Pixel_End_x;
  uint i2Pixel_End_y;
  float CellCount_x;
  float CellCount_y;
  float Depth;
  float RatioYx;
};

struct NProbabilityDistributionSampler_SAliasTable {
  MwSArray<float> Weights;
  MwSArray<uint> Alias;
  MwSArray<float> Probs;
};

struct GmTransYawPitchRoll {
  vec3 Trans;
  float Yaw;
  float Pitch;
  float Roll;
};

struct GmTransYawPitch {
  vec3 Trans;
  float Yaw;
  float Pitch;
};

struct NFastPoolVirtual_SPoolBase {
  uint UsedElemCount;
  NatAdr const AllocatedBytes;
};

struct GmBoxNat3 {
  nat3 Min;
  nat3 Max;
};

struct SGmSmoothReal3Model {
  float Delta;
  uint TimeUp;
  uint TimeDown;
};

struct TPlayerUId {
};

struct NHmsCollision_SItemCreateParams {
};

struct NHmsLightMapBlender_SMgr {
  MwArrayInPlaceDyn<NHmsLightMapBlender_SFrame> Frames;
};

struct NHmsLightMapBlender_SFrame {
  uint CacheFrameId;
  UnknownType Bitmaps;
  CPlugBitmap* Bitmap_Sprite;
  CPlugBitmap* Bitmap_LocalLight;
};

struct NHmsLightMapCache_SFrame {
  const float ReplayTime;
  const float MaxHDR_Mood;
  const float MaxHDR;
  const vec3 MaxHDR_HBasisScaled234;
  const float BounceFactor;
  const float SkyFactor;
  const bool SkyUseClouds;
  const bool StoreLAmbient;
  const NHmsLightMapCache::ELocalLight_Storage LocalLight_Storage;
  const NHmsLightMapCache::ELocalLight_Switch LocalLight_Switch;
  const vec3 LAmbient;
  const float LAmbientScale;
  const EHmsLightMapBump Bump;
};

struct NHmsZoneOverlay_SClipRect {
  vec2 Rect_min;
  vec2 Rect_max;
  uint In3dQuadIndex;
  bool IsUsed;
};

struct SHmsFxBloomHdr {
  float Dyna_Intensity;
  float Dyna_BloomIntensity;
  float Dyna_StreaksIntensity;
  float Dyna_LensDirtIntens; // Range: 0 - 1
  float Dyna_StreaksAttenuation; // Range: 0 - 1
  CPlugCurveSimpleNod* IntensAtHdrNorm_Func;
};

struct SHmsFxToneMap {
  bool Dyna_AutoExposureEnable;
  float Dyna_ExposureStaticBase; // Range: -2 - 2
  float Dyna_ExposureShootingBias; // Range: -2 - 2
  float 50__seconds; // Range: 0 - 1
  float Dyna_AutoExpMinExposure;
  float Dyna_AutoExpMaxExposure;
  float KeyValue; // Range: 0.01 - 0.5
  CPlugCurveSimpleNod* KeyValueAtAvgLumi_Func;
  EPlugMoodFxTMFilmCurve Dyna_FilmCurve;
};

struct SHmsPostFxState {
  SHmsFxToneMap ToneMap;
  SHmsFxBloomHdr BloomHdr;
  float MotionBlur2d_Intensity01; // Range: 0 - 1
};

struct NHmsLightMap_SComputePImp {
  bool ShowOverlap;
  CPlugPointsInSphereOpt* PointsInSphereOpt;
  CPlugBitmap* const BitmapLM_LightWeight;
  CPlugBitmap* const BitmapLM_LListUV;
  CPlugBitmap* const BitmapLM_LListW;
  CPlugBitmap* const BitmapShadow;
  CPlugBitmap* const BitmapLM_MDiffuse;
  CPlugBitmap* const BitmapSM_DepthToPeel;
  CPlugBitmap* const BitmapSM_ColorPeeled;
  CPlugBitmap* const BitmapLM_ILightDir;
  CPlugBitmap* const BitmapLM_ILightInput;
  CPlugBitmap* const BitmapLM_Mask;
  CPlugBitmap* const BitmapLM_Temp_Accum;
  CPlugBitmap* const BitmapSprite3x3_ILightOutput;
  CPlugBitmap* const BitmapSprite_ILightDir;
  CPlugBitmap* const BitmapSprite_PosAndRadius;
  CPlugBitmap* const BitmapProbeGeom_BBoxInvHDiag_ThroughBounce;
  CPlugBitmap* const BitmapLMSS_LBumpIntens;
  CPlugBitmap* const BitmapSpriteSS_InLightAtt;
  CPlugBitmap* const ProbeGridCpt_Bitmap_Accum;
  CPlugBitmap* const ProbeGridCpt_Bitmap_Accum_SkyVis;
  CPlugBitmap* const ProbeGridCpt_Bitmap_ILightDir;
  bool EnableShadows;
};

struct NHmsLightMap_SPImp {
  CSystemPackDesc* const CachePackDesc;
  CSystemPackDesc* const CachePackDescBumpAvg;
  CHmsLightMapCache* const Cache;
  const EHmsLightMapCacheSize CacheSize;
  CHmsLightMapCache* const CacheSmall;
  const MwId CacheId_IdCollection;
  const MwId CacheId_IdDecoration;
  const string ChallengeJointId;
  const uint cBlock;
  const uint cBlockSprite;
  const uint cBlockSH;
  const float DeCompGrayScale;
  const float DeCompMinLinear;
};

struct CHmsSolidVisCst_TmCar_SPrestige {
  uint Year;
  CHmsSolidVisCst_TmCar::SPrestige::EMedal Medal;
  CHmsSolidVisCst_TmCar::SPrestige::ESeasonOrType SeasonOrType;
  uint Level;
  EPrestigeMode Mode;
  EPrestigeSeason Season;
};

struct CHmsSolid2 {
  CPlugSolid2Model* m_Model;
  MwSArray<iso4> m_JointToSolids;
  MwSArray<bool> m_JointVisibles;
  MwSArray<float> m_DamageZoneAmounts;
};

struct NHmsZone_SLightDir {
  GxLightDirectional* gxLight;
  CHmsLight* HmsLight;
};

struct NHmsZone_SLightDynaFrustum : public NHmsZone_SLightDyna {
  CGxLightFrustum* gxLight;
  CPlugBitmap* BitmapProjector;
};

struct NHmsZone_SLightDyna2 {
};

struct NHmsZone_SLightStatic {
  CPlugLight* PlugLight;
  CGxLightBall* gxLight;
  iso4 Location;
};

struct NHmsZone_SMgrMeshStatic {
  MwFastBuffer<NHmsZone_NHmsMeshStatic_SModelRef> Models;
  const uint cInstance;
};

struct NHmsZone_NHmsMeshStatic_SModelRef {
  CPlugSolid2Model* ModelForZone;
};

struct CControlEffectSimi_SKeyVal {
  vec2 Simi2_Pos;
  float Simi2_Rot;
  vec2 Simi2_Scale;
  float Depth;
  float Opacity;
  float ColorBlend;
  vec3 Color;
};

} // namespace MetaNotPersistent

// Enums is not really an engine, it's a hardcoded separate thing pretending to be an engine.
namespace Enums {

enum class EPlugSurfaceGameplayId {
  None = 0,
  Turbo = 1,
  Turbo2 = 2,
  TurboRoulette = 3,
  FreeWheeling = 4,
  NoGrip = 5,
  NoSteering = 6,
  ForceAcceleration = 7,
  Reset = 8,
  SlowMotion = 9,
  Bumper = 10,
  Bumper2 = 11,
  ReactorBoost_Legacy = 12,
  Fragile = 13,
  ReactorBoost2_Legacy = 14,
  Bouncy = 15,
  NoBrakes = 16,
  Cruise = 17,
  ReactorBoost_Oriented = 18,
  ReactorBoost2_Oriented = 19,
  VehicleTransform_Reset = 20,
  VehicleTransform_CarSnow = 21,
  VehicleTransform_CarRally = 22,
  VehicleTransform_CarDesert = 23,
  XXX_Null = 24, // XXX Null
};

enum class NPlugPainterLayer::EPlugPainterLayer_Type {
  Draw = 0,
  VFX = 1,
  Skidmarks = 2,
  Tyres = 3,
};

enum class NPlugAnim::EPoseType {
  Parent = 0,
  Global = 1,
  Additive = 2,
};

enum class NSystemConfig::EVoiceDetectionMode {
  PushToTalk = 0,
  AutoDetect = 1,
  Manual = 2,
};

enum class SGamePlaygroundUIConfig::EUISequence {
  None = 0,
  Playing = 1,
  Intro = 2,
  Outro = 3,
  Podium = 4,
  CustomMTClip = 5,
  EndRound = 6,
  PlayersPresentation = 7,
  UIInteraction = 8,
  RollingBackgroundIntro = 9,
  CustomMTClip_WithUIInteraction = 10,
  Finish = 11,
};

enum class NWebServicesPrestige::EPrestigeMode {
  Unknown = 0,
  Ranked = 1,
  Royal = 2,
  Season = 3,
};

enum class ESpectatorCameraType {
  _SpectatorCam_Replay = 0, // |SpectatorCam|Replay
  _SpectatorCam_Follow = 1, // |SpectatorCam|Follow
  _SpectatorCam_Free = 2, // |SpectatorCam|Free
};

enum class ESpectatorTargetMode {
  _SpectatorCam_Manual = 0, // |SpectatorCam|Manual
  _SpectatorCam_Auto = 1, // |SpectatorCam|Auto
};

enum class EChatScope {
  ToEveryone = 0,
  ToSpectatorCurrent = 1,
  ToSpectatorAll = 2,
  ToTeam = 3,
  ToYouOnly = 4,
};

enum class CGameCtnMediaBlockEditorTriangles::EEditMode {
  Edit_vertex = 0, // Edit vertex
  Create_triangle = 1, // Create triangle
  Delete_vertex = 2, // Delete vertex
};

enum class CGameCtnMediaBlockCameraCustom::ECamInterp {
  None = 0,
  CatmullNRom = 1, // Catmull-Rom
  Linear = 2,
  Hermite_Curve = 3, // Hermite Curve
};

enum class EGameCollectorSearchScope {
  Undefined = 0,
  Game = 1,
  TitleMaker = 2,
  Title = 3,
  User = 4,
};

enum class EDummyCollectionIdent {
  Dummy = 0,
};

enum class EGhostNameLogoType {
  None = 0,
  LocalPlateform = 1,
  Ubisoft = 2,
};

enum class NGameLoadProgress::EState {
  Disabled = 0,
  Displayed = 1,
};

enum class EGameInventoryItemClass {
  Weapon = 0,
  Movement = 1,
  Consumable = 2,
  Armor = 3,
};

enum class CSystemConfigDisplay::EShowPerformance {
  _ShowPerformance_None = 0, // |ShowPerformance|None
  _ShowPerformance_Minimal = 1, // |ShowPerformance|Minimal
  _ShowPerformance_Fps = 2, // |ShowPerformance|Fps
  _ShowPerformance_FpsPBars = 3, // |ShowPerformance|Fps+Bars
  _ShowPerformance_FpsPBarsPLegends = 4, // |ShowPerformance|Fps+Bars+Legends
};

enum class CSystemConfigDisplay::ERenderingApi {
  D3D9 = 0,
  D3D11 = 1,
  Vulkan = 2,
};

enum class CSystemConfigDisplay::EDisplayMode {
  _DisplayMode_FullScreen = 0, // |DisplayMode|FullScreen
  _DisplayMode_Windowed = 1, // |DisplayMode|Windowed
  _DisplayMode_WindowedFull = 2, // |DisplayMode|WindowedFull
};

enum class CSystemConfigDisplay::EWindowSize {
  _800x450 = 0, // 800x450
  _1024x576 = 1, // 1024x576
  _1280x720 = 2, // 1280x720
  _1600x900 = 3, // 1600x900
  _1920x1080 = 4, // 1920x1080
};

enum class CSystemConfigDisplay::EDisplaySync {
  _DisplaySync_None = 0, // |DisplaySync|None
  _DisplaySync_1_Interval = 1, // |DisplaySync|1 Interval
  _DisplaySync_2_Intervals = 2, // |DisplaySync|2 Intervals
  _DisplaySync_3_Intervals = 3, // |DisplaySync|3 Intervals
};

enum class CSystemConfigDisplay::ETripleBuffer {
  _TripleBuffering_Off = 0, // |TripleBuffering|Off
  _TripleBuffering_On = 1, // |TripleBuffering|On
};

enum class CSystemConfigDisplay::EPreset {
  _DisplayPreset_None = 0, // |DisplayPreset|None
  _DisplayPreset_Very_Fast = 1, // |DisplayPreset|Very Fast
  _DisplayPreset_Fast = 2, // |DisplayPreset|Fast
  _DisplayPreset_Nice = 3, // |DisplayPreset|Nice
  _DisplayPreset_Very_Nice = 4, // |DisplayPreset|Very Nice
};

enum class CSystemConfigDisplay::EAntialias {
  _Antialiasing_None = 0, // |Antialiasing|None
  _Antialiasing__2_samples = 1, // |Antialiasing| 2 samples
  _Antialiasing__4_samples = 2, // |Antialiasing| 4 samples
  _Antialiasing__6_samples = 3, // |Antialiasing| 6 samples
  _Antialiasing__8_samples = 4, // |Antialiasing| 8 samples
  _Antialiasing_16_samples = 5, // |Antialiasing|16 samples
};

enum class CSystemConfigDisplay::EDeferredAA {
  _DeferredAntialiasing_None = 0, // |DeferredAntialiasing|None
  _DeferredAntialiasing__FXAA = 1, // |DeferredAntialiasing| FXAA
  _DeferredAntialiasing__TXAA = 2, // |DeferredAntialiasing| TXAA
};

enum class CSystemConfigDisplay::EShaderQ {
  _ShaderQuality_Very_Fast = 0, // |ShaderQuality|Very Fast
  _ShaderQuality_Fast = 1, // |ShaderQuality|Fast
  _ShaderQuality_Nice = 2, // |ShaderQuality|Nice
  _ShaderQuality_Very_Nice = 3, // |ShaderQuality|Very Nice
};

enum class ETextureQuality {
  _TexturesQuality_Very_Low = 0, // |TexturesQuality|Very Low
  _TexturesQuality_Low = 1, // |TexturesQuality|Low
  _TexturesQuality_Medium = 2, // |TexturesQuality|Medium
  _TexturesQuality_High = 3, // |TexturesQuality|High
};

enum class CSystemConfigDisplay::EFilterAnisoQ {
  _MaxFiltering_Bilinear = 0, // |MaxFiltering|Bilinear
  _MaxFiltering_Trilinear = 1, // |MaxFiltering|Trilinear
  _MaxFiltering_Anisotropic__2x = 2, // |MaxFiltering|Anisotropic  2x
  _MaxFiltering_Anisotropic__4x = 3, // |MaxFiltering|Anisotropic  4x
  _MaxFiltering_Anisotropic__8x = 4, // |MaxFiltering|Anisotropic  8x
  _MaxFiltering_Anisotropic_16x = 5, // |MaxFiltering|Anisotropic 16x
  _MaxFiltering_Aniso_16x_everywhere = 6, // |MaxFiltering|Aniso 16x everywhere
};

enum class CSystemConfigDisplay::EZClip {
  _ZClip_Disable = 0, // |ZClip|Disable
  _ZClip_Auto = 1, // |ZClip|Auto
  _ZClip_Fixed = 2, // |ZClip|Fixed
};

enum class CSystemConfigDisplay::EZClipAuto {
  _ZClipAuto_Near = 0, // |ZClipAuto|Near
  _ZClipAuto_Medium = 1, // |ZClipAuto|Medium
  _ZClipAuto_Far = 2, // |ZClipAuto|Far
};

enum class CSystemConfigDisplay::EGeometryQ {
  _GeometryQuality_Very_Fast = 0, // |GeometryQuality|Very Fast
  _GeometryQuality_Fast = 1, // |GeometryQuality|Fast
  _GeometryQuality_Nice = 2, // |GeometryQuality|Nice
  _GeometryQuality_Very_Nice = 3, // |GeometryQuality|Very Nice
};

enum class CSystemConfigDisplay::EEverywhereReflect {
  _EverywhereReflect_None = 0, // |EverywhereReflect|None
  _EverywhereReflect_Enabled = 1, // |EverywhereReflect|Enabled
};

enum class CSystemConfigDisplay::EWaterReflect {
  _WaterReflect_Very_Fast = 0, // |WaterReflect|Very Fast
  _WaterReflect_Fast = 1, // |WaterReflect|Fast
  _WaterReflect_Nice = 2, // |WaterReflect|Nice
  _WaterReflect_Very_Nice = 3, // |WaterReflect|Very Nice
};

enum class CSystemConfigDisplay::EVehicleReflect {
  _VehicleReflect_Low = 0, // |VehicleReflect|Low
  _VehicleReflect_HighInReplay = 1, // |VehicleReflect|HighInReplay
  _VehicleReflect_High = 2, // |VehicleReflect|High
};

enum class CSystemConfigDisplay::EFxBloomHdr {
  _FxBloomHdr_None = 0, // |FxBloomHdr|None
  _FxBloomHdr_Medium = 1, // |FxBloomHdr|Medium
  _FxBloomHdr_High = 2, // |FxBloomHdr|High
};

enum class CSystemConfigDisplay::EFxMotionBlur {
  _FxMotionBlur_Off = 0, // |FxMotionBlur|Off
  _FxMotionBlur_On = 1, // |FxMotionBlur|On
};

enum class CSystemConfigDisplay::EFxBlur {
  _FxBlur_Off = 0, // |FxBlur|Off
  _FxBlur_On = 1, // |FxBlur|On
};

enum class CSystemConfigDisplay::ELightMapSizeMax {
  _LightMapSizeMax_Auto = 0, // |LightMapSizeMax|Auto
  _LightMapSizeMax_1k_2 = 1, // |LightMapSizeMax|1k^2
};

enum class CSystemConfigDisplay::ELightMapQuality {
  _LightMapQuality_None = 0, // |LightMapQuality|None
  _LightMapQuality_Very_Fast = 1, // |LightMapQuality|Very Fast
  _LightMapQuality_Fast = 2, // |LightMapQuality|Fast
  _LightMapQuality_Default = 3, // |LightMapQuality|Default
  _LightMapQuality_High = 4, // |LightMapQuality|High
};

enum class CSystemConfigDisplay::EScreenShotExt {
  JPG = 0,
  WebP = 1,
  TGA = 2,
};

enum class CSystemConfigDisplay::EShadows {
  _Shadows_None = 0, // |Shadows|None
  _Shadows_Minimum = 1, // |Shadows|Minimum
  _Shadows_Medium = 2, // |Shadows|Medium
  _Shadows_High = 3, // |Shadows|High
  _Shadows_Very_High = 4, // |Shadows|Very High
};

enum class CSystemConfigDisplay::EGpuSync {
  _GpuSync_None = 0, // |GpuSync|None
  _GpuSync_3_Frames = 1, // |GpuSync|3 Frames
  _GpuSync_2_Frames = 2, // |GpuSync|2 Frames
  _GpuSync_1_Frame = 3, // |GpuSync|1 Frame
  _GpuSync_Immediate = 4, // |GpuSync|Immediate
};

enum class CSystemConfigDisplay::ELightFromMap {
  _LightFromMap_None = 0, // |LightFromMap|None
  _LightFromMap_My_Vehicle = 1, // |LightFromMap|My Vehicle
  _LightFromMap_All_Vehicles = 2, // |LightFromMap|All Vehicles
};

enum class EHmsLightMapQuality {
  VFast = 0,
  Fast = 1,
  Default = 2,
  High = 3,
  Ultra = 4,
};

enum class NSysCfgVision::SDeferred::EDecalZFunc {
  None = 0,
  L_ = 1, // <=
  G_ = 2, // >=
};

enum class NSysCfgVision::SDeferred::EDecalDbg {
  None = 0,
  PixelFill = 1,
  ShowBoxes = 2,
};

enum class NSysPlatform::EDirectLinkType {
  JoinServer = 0,
  EditNewMap = 1,
  OfficialCampaign = 2,
  TrackOfTheDay = 3,
  Ranked = 4,
  Royal = 5,
  ArcadeServer = 6,
  Hotseat = 7,
  Splitscreen = 8,
  Garage = 9,
  JoinSession = 10,
  MainUserSignOutFirstParty = 11,
};

enum class NSysPlatform::EDirectLink_Join_Role {
  Player = 0,
  Spectator = 1,
};

enum class NSystemConfig::EGamePackQuality {
  _Quality_Unknown = 0, // |Quality|Unknown
  _Quality_LD = 1, // |Quality|LD
  _Quality_HD = 2, // |Quality|HD
};

enum class CSystemConfig::ENetworkSpeed {
  Custom = 0,
  _100Kbps = 1,
  _1Mbps = 2,
  _10Mbps = 3,
  _100Mbps = 4,
};

enum class CSystemConfig::EPlayerInfoDisplayType {
  Name = 0,
  Avatar = 1,
  Avatar_and_Name = 2, // Avatar and Name
};

enum class CSystemConfig::ETmCarQuality {
  _CarQuality_All_Low = 0, // |CarQuality|All Low
  _CarQuality_Medium__Low_Opponents = 1, // |CarQuality|Medium, Low Opponents
  _CarQuality_High__Medium_Opponents = 2, // |CarQuality|High, Medium Opponents
  _CarQuality_All_High = 3, // |CarQuality|All High
};

enum class CSystemConfig::ETmCarParticlesQuality {
  _CarParticlesQuality_All_Low = 0, // |CarParticlesQuality|All Low
  _CarParticlesQuality_All_Medium = 1, // |CarParticlesQuality|All Medium
  _CarParticlesQuality_High_Medium_Opponents = 2, // |CarParticlesQuality|High,Medium Opponents
  _CarParticlesQuality_All_High = 3, // |CarParticlesQuality|All High
};

enum class CSystemConfig::EPlayerShadow {
  _PlayerShadow_None = 0, // |PlayerShadow|None
  _PlayerShadow_Me = 1, // |PlayerShadow|Me
  _PlayerShadow_All = 2, // |PlayerShadow|All
};

enum class CSystemConfig::EPlayerOcclusion {
  _PlayerOcclusion_None = 0, // |PlayerOcclusion|None
  _PlayerOcclusion_Me = 1, // |PlayerOcclusion|Me
  _PlayerOcclusion_All = 2, // |PlayerOcclusion|All
};

enum class CSystemConfig::ETmOpponents {
  _Opponents_Always_Visible = 0, // |Opponents|Always Visible
  _Opponents_Hide_Too_Close = 1, // |Opponents|Hide Too Close
};

enum class CSystemConfig::ETmBackgroundQuality {
  _BackgroundQuality_Low = 0, // |BackgroundQuality|Low
  _BackgroundQuality_Medium = 1, // |BackgroundQuality|Medium
  _BackgroundQuality_High = 2, // |BackgroundQuality|High
};

enum class CSystemConfig::EAudioQuality {
  _AudioQuality_Low = 0, // |AudioQuality|Low
  _AudioQuality_Normal = 1, // |AudioQuality|Normal
  _AudioQuality_High = 2, // |AudioQuality|High
};

enum class CSystemConfig::EAdvertising {
  Disabled = 0,
  Configurable = 1,
  Forced = 2,
};

enum class NSceneVisEntFx::EVisible {
  None = 0,
  Solid = 1,
  Ghost = 2,
  Cloaked = 3,
};

enum class EMeleeState {
  None = 0,
  BlowStart = 1,
  BlowHit = 2,
  BlowRecovery = 3,
  Stun = 4,
  Dodge = 5,
  Parry = 6,
};

enum class EMeleeHitType {
  None = 0,
  Light = 1,
  Medium = 2,
  Heavy = 3,
};

enum class EParryType {
  None = 0,
  Light = 1,
  Medium = 2,
  Heavy = 3,
};

enum class EPlugSeatType {
  Pilot = 0,
  Passenger = 1,
};

enum class ESceneCharVisNotice {
  Jump = 0,
  JumpChain1 = 1,
  JumpChain2 = 2,
  WallJump = 3,
  DoubleJump = 4,
  Land = 5,
  Spawn = 6,
  InjuredLight = 7,
  InjuredEliminatedShot = 8,
  InjuredEliminated = 9,
  Fire = 10,
  JumpPadSmall = 11,
  Attractor = 12,
  GestureTaunt = 13,
  Turbo = 14,
  SpeedUp = 15,
  FreeWheeling = 16,
  Evolution = 17,
  BulletHit = 18,
  BackFlipJump = 19,
  FrontFlipJump = 20,
  ZoomIn = 21,
  ZoomOut = 22,
  MissingStamina = 23,
  GesturePointTo = 24,
  JumpPadMed = 25,
  JumpPadBig = 26,
  TechJump = 27,
  TeleportUnspawn = 28,
  TeleportSpawn = 29,
  FallDamage = 30,
  MeleeHit = 31,
  GestureVictory = 32,
};

enum class EPlugSurfaceMaterialId {
  Concrete = 0,
  Pavement = 1,
  Grass = 2,
  Ice = 3,
  Metal = 4,
  Sand = 5,
  Dirt = 6,
  Turbo_Deprecated = 7,
  DirtRoad = 8,
  Rubber = 9,
  SlidingRubber = 10,
  Test = 11,
  Rock = 12,
  Water = 13,
  Wood = 14,
  Danger = 15,
  Asphalt = 16,
  WetDirtRoad = 17,
  WetAsphalt = 18,
  WetPavement = 19,
  WetGrass = 20,
  Snow = 21,
  ResonantMetal = 22,
  GolfBall = 23,
  GolfWall = 24,
  GolfGround = 25,
  Turbo2_Deprecated = 26,
  Bumper_Deprecated = 27,
  NotCollidable = 28,
  FreeWheeling_Deprecated = 29,
  TurboRoulette_Deprecated = 30,
  WallJump = 31,
  MetalTrans = 32,
  Stone = 33,
  Player = 34,
  Trunk = 35,
  TechLaser = 36,
  SlidingWood = 37,
  PlayerOnly = 38,
  Tech = 39,
  TechArmor = 40,
  TechSafe = 41,
  OffZone = 42,
  Bullet = 43,
  TechHook = 44,
  TechGround = 45,
  TechWall = 46,
  TechArrow = 47,
  TechHook2 = 48,
  Forest = 49,
  Wheat = 50,
  TechTarget = 51,
  PavementStair = 52,
  TechTeleport = 53,
  Energy = 54,
  TechMagnetic = 55,
  TurboTechMagnetic_Deprecated = 56,
  Turbo2TechMagnetic_Deprecated = 57,
  TurboWood_Deprecated = 58,
  Turbo2Wood_Deprecated = 59,
  FreeWheelingTechMagnetic_Deprecated = 60,
  FreeWheelingWood_Deprecated = 61,
  TechSuperMagnetic = 62,
  TechNucleus = 63,
  TechMagneticAccel = 64,
  MetalFence = 65,
  TechGravityChange = 66,
  TechGravityReset = 67,
  RubberBand = 68,
  Gravel = 69,
  Hack_NoGrip_Deprecated = 70,
  Bumper2_Deprecated = 71,
  NoSteering_Deprecated = 72,
  NoBrakes_Deprecated = 73,
  RoadIce = 74,
  RoadSynthetic = 75,
  Green = 76,
  Plastic = 77,
  DevDebug = 78,
  Free3 = 79,
  XXX_Null = 80, // XXX Null
};

enum class NSceneWeather::EClearMode {
  Fixed = 0,
  Fog = 1,
};

enum class NSceneSpectatorVis::EOrientationMode {
  None = 0,
  WatchNearestTrajPoint = 1,
  WatchNearestVisibleTrajPoint = 2,
};

enum class ESceneVehicleVisReactorBoostLvl {
  None = 0,
  Lvl1 = 1,
  Lvl2 = 2,
};

enum class ESceneVehicleVisReactorBoostType {
  None = 0,
  Up = 1,
  Down = 2,
  UpAndDown = 3,
};

enum class ESceneVehicleVisEvent {
  GlassSmash = 0,
  Horn = 1,
  Turbo = 2,
  ImpactBody = 3,
  ImpactWheelFront = 4,
  ImpactWheelBack = 5,
  BeginFreeWheeling = 6,
  EndFreeWheeling = 7,
  AirSpawnRelease = 8,
  PartDetached = 9,
  ImpactDetachedPart = 10,
  BeginNoGrip = 11,
  EndNoGrip = 12,
  BeginBulletTime = 13,
  EndBulletTime = 14,
  BeginBoost_1 = 15,
  EndBoost = 16,
  _Unused0 = 17,
  BeginBoost_2 = 18,
  ResetHandicaps = 19,
  TriggeredWaypoint = 20,
  BeginFragile = 21,
  _Unused1 = 22,
  _Unused2 = 23,
  VehicleTransform = 24,
  WallBounce = 25,
};

enum class NScenePathFinding::EPathRequestStatus {
  Unknown = 0,
  Invalid = 1,
  Pending = 2,
  Processed = 3,
  Success = 4,
  Failed = 5,
};

enum class EPlugStateMachineTransitionType {
  Immediate = 0,
  Frozen = 1,
  Blend = 2,
};

enum class EPlugStateMachineTransitionCurve {
  HardHard = 0,
  HardSoft = 1,
  SoftSoft = 2,
  SoftHard = 3,
};

enum class NPlugExpr::EType {
  Bool8 = 0,
  Int32 = 1,
  Float = 2,
  Float3 = 3,
  Void = 4, // void
  Enum32 = 5,
  Int8 = 6,
  Int16 = 7,
  Enum8 = 8,
  Enum16 = 9,
  Int3 = 10,
  Float2 = 11,
  Float4 = 12,
};

enum class NGmUnit::EConvertMethod {
  None = 0,
  DegToRad = 1,
  RadToDeg = 2,
  KnotToMs = 3,
  MsToKnot = 4,
};

enum class NPlugGpuHlsl_D3D12::NRootSign::EParamType {
  Draw_CBuffer = 0,
  Shader_CBuffer = 1,
  Shader_Table_Sampler = 2,
  Shader_Table_SrvUav = 3,
  Scene_CBuffer = 4,
  Scene_TableSrv = 5,
};

enum class EPlugAnimGraphNodeType {
  JointRotateFrom = 0,
  JointRotate = 1,
  JointAlignTo = 2,
  JointTranslate = 3,
  JointIK2 = 4,
  JointKeepRefGlobalRot = 5,
  JointTranslateDistConstraint = 6,
  LocalToGlobal = 7,
  ClipPlay = 8,
  Blend = 9,
  Blend2d = 10,
  FinalPose = 11,
  StateMachine = 12,
  Graph = 13,
  LodSwitch = 14,
  Group = 15,
  RefLocalPose = 16,
  GraphOutput = 17,
  ExtractMotion = 18,
  RefGlobalPose = 19,
  SetVar = 20,
  ClipGroupPlay = 21,
  SetSkel = 22,
  LayeredBlend = 23,
  GlobalToLocal = 24,
  PoseGrid = 25,
  GraphInput = 26,
  JointLock = 27,
  JointInertia = 28,
  AirTrajectoryPrediction = 29,
  DebugHelper = 30,
  AssertVar = 31,
  Funnel = 32,
  ExtractUnit = 33,
  SetJointExpr = 34,
  JointTransConstraint = 35,
  JointRotConstraint = 36,
  AvatarV3_Global = 37,
  AvatarV3_Locomotion = 38,
  AvatarV3_Jump = 39,
  AvatarV3_Swim = 40,
  AvatarV3_Idle = 41,
  AvatarV0 = 42,
  AvatarPoseEditor = 43,
  AvatarV3_Seated = 44,
  AvatarV3_Resting = 45,
  Avatar_Climb = 46,
};

enum class NPlugAnim::EExtractMotionMode {
  Absolute = 0,
  Additive_Delta = 1, // Additive Delta
};

enum class NPlugAnim::EExtractUnitTarget {
  Translation = 0,
  Rotation = 1,
};

enum class NPlugAnim::EExtractUnitAxis {
  XAxis = 0,
  YAxis = 1,
  ZAxis = 2,
};

enum class GmFunc::ERotationOrder {
  XYZ = 0,
  XZY = 1,
  YXZ = 2,
  YZX = 3,
  ZXY = 4,
  ZYX = 5,
};

enum class NPlugAnim::EFunnelType {
  EvaluateOnce = 0,
  UpdateOnce = 1,
  Combined = 2,
};

enum class NPlugAnim::EGraphNodeAssertVarType {
  Bool = 0,
  Float = 1,
  Float3 = 2,
  String = 3,
};

enum class NPlugAnim::ELayeredBlendType {
  Override = 0,
  Additive = 1,
};

enum class NPlugAnim::ELayeredBlendContext {
  Local = 0,
  Global = 1,
  Ignore = 2,
};

enum class NPlugAnim::EGraphNodeOperationType {
  Set = 0,
  Add = 1,
  Sub = 2,
  Mult = 3,
};

enum class NPlugAnim::EGraphNodeAxisWorkspace {
  Global = 0,
  Parent = 1,
  Local = 2,
};

enum class NPlugAnim::EGraphNodeLockWorkspace {
  World = 0,
  Global = 1,
  Parent = 2,
  Custom_Joint = 3, // Custom Joint
};

enum class NPlugAnim::EGraphNodeLockType {
  Translation_And_Rotation = 0, // Translation And Rotation
  Translation = 1,
  Rotation = 2,
};

enum class NPlugAnim::EGraphNodeIK2SolverMode {
  SolverAuto = 0,
  PoleVectorExpr = 1,
};

enum class NPlugAnim::EClipPlayAdditiveType {
  None = 0,
  From_First_Frame = 1, // From First Frame
  From_Skel_Ref = 2, // From Skel Ref
};

enum class NPlugItemPlacement::EStartPos {
  Cursor = 0,
  Zero = 1,
  CursorUZoneCenterV = 2,
  ZoneCenterUCursorV = 3,
  ZoneCenterUV = 4,
};

enum class NPlugItemPlacement::EFillMode {
  FixedStepCount = 0,
  AvailableSpace = 1,
};

enum class NPlugItemPlacement::EFillDir {
  U = 0,
  V = 1,
};

enum class NPlugItemPlacement::ESiblingDistribution {
  None = 0,
  Alternate = 1,
  All = 2,
};

enum class NPlugItemPlacement::EAlign {
  Center = 0,
  Begin = 1,
  End = 2,
};

enum class CPlugBulletPhyModel::EBulletType {
  Projectile = 0,
  Laser = 1,
  Beam = 2,
  TriLaser = 3,
  ProjectileHoming = 4,
  Spear = 5,
  ProjectileGuidedMouse = 6,
  ProjectileGuidedKeyboard = 7,
  ProjectileHovering = 8,
  Particle = 9,
  ProjectilePretargeted = 10,
  LaserSlash = 11,
};

enum class CPlugBulletPhyModel::EExplosionOccultationTest {
  Binary = 0,
  Smoother = 1,
};

enum class CPlugBulletPhyModel::EFireBulletPatternMode {
  NoPattern = 0,
  GrowingOffset = 1,
  Spiral = 2,
  Hatchet = 3,
  Net = 4,
  ChaosSpiral = 5,
};

enum class EAudioBalanceGroup {
  Auto = 0,
  Music = 1,
  Menus = 2,
  Ambiance = 3,
  Player = 4,
  Bengs = 5,
  Guns = 6,
  BackingDirect = 7,
  Trails = 8,
  GameUI = 9,
  Custom1 = 10,
  Custom2 = 11,
  OtherPlayers = 12,
  ImpactWarning = 13,
  Environment = 14,
};

enum class EImportMeshType {
  Static = 0,
  Dyna = 1,
  DynaSkelGeneric = 2,
  DynaSkelHumano = 3,
  DynaSkelFlat = 4,
  Vehicle = 5,
};

enum class EPlugModelShadingType {
  Dyna0 = 0,
  Static = 1,
  Char = 2,
  Vehicle = 3,
  MenuBox3d = 4,
};

enum class CPlugCustomBeamModel::ECustomBeamType {
  Laser = 0,
  Beam = 1,
  Slash = 2,
};

enum class EPlugCustomBulletModelType {
  Simple = 0,
  Homing = 1,
  GuidedKeyboard = 2,
  GuidedMouse = 3,
  Hovering = 4,
  Pretargeted = 5,
};

enum class EPlugAnimSpotType {
  _3Clips = 0, // 3Clips
  _1Clip = 1, // 1Clip
};

enum class EPlugAnimClipAlignOffset {
  None = 0,
  Cling = 1,
  Water = 2,
};

enum class EPlugAnimGraphBlending {
  Normal = 0,
  Additive = 1,
};

enum class EPlugAnimGraphStateSelectCond {
  ActionIs = 0,
  Action2Is = 1,
  StateIs = 2,
  StateIsFinished = 3,
  StateIsDiffFrom = 4,
  Action3Is = 5,
  Action1IsEvent = 6,
  Action2IsEvent = 7,
};

enum class EPlugAnimRootYawType {
  AimHysteresis = 0,
  Aim = 1,
  Vel = 2,
  Cling = 3,
  DontChange = 4,
  UseStartYaw = 5,
};

enum class EPlugAnimLocoNodeType {
  Move = 0,
  Idle = 1,
  Turn = 2,
  Skid = 3,
  Wait = 4,
};

enum class NPlugAnim::EJointChannel {
  Tx = 0,
  Ty = 1,
  Tz = 2,
  Rx = 3,
  Ry = 4,
  Rz = 5,
};

enum class NPlugAdn::ETagCat {
  Tag = 0,
  Required = 1,
};

enum class EAudioFilterType {
  Null = 0,
  LowPass = 1,
  HighPass = 2,
  BandPass = 3,
  HighShelf = 4,
  LowShelf = 5,
  Peaking = 6,
};

enum class EGxTexAddress {
  Wrap = 0,
  Mirror = 1,
  Clamp = 2,
  Border = 3,
};

enum class NPlugModelKit::EOptionAutoFill {
  None = 0,
  IsExtraFolders = 1,
};

enum class EPlugRoadChunkType {
  Road = 0,
  Intersection = 1,
};

enum class EPlugRoadWayType {
  OneWay = 0,
  TwoWays = 1,
};

enum class EPlugRoadType {
  Undefined = 0,
  Country_Lane = 1, // Country Lane
  City = 2,
  Road = 3,
  InterState = 4,
  SpeedWay = 5,
  HighWay = 6,
};

enum class EPlugRoadDensity {
  Very_low = 0, // Very low
  Low = 1,
  Medium = 2,
  High = 3,
  Very_high = 4, // Very high
};

enum class EPlugFlockType {
  Bird = 0,
  Pig = 1,
};

enum class EGameItemWaypointType {
  Start = 0,
  Finish = 1,
  Checkpoint = 2,
  None = 3,
  StartFinish = 4,
  Dispenser = 5,
};

enum class NPlugAnim::EPropType {
  Float = 0,
  Float3 = 1,
  Quat = 2,
  RGBAColor = 3,
  Bool = 4,
  Int8 = 5,
  Int16 = 6,
  Int32 = 7,
};

enum class EGmSurfType {
  Sphere = 0,
  Ellipsoid = 1,
  Plane = 2,
  QuadHeight = 3,
  TriangleHeight = 4,
  Polygon = 5,
  Box = 6,
  Mesh = 7,
  VCylinder = 8,
  MultiSphere = 9,
  ConvexPolyhedron = 10,
  Capsule = 11,
  Circle = 12,
  Compound = 13,
  SphereLocated = 14,
  CompoundInstance = 15,
  Cylinder = 16,
  SphericalShell = 17,
  Voxel = 18,
  Diggable = 19,
};

enum class NPlugFilePreloader::EAllocType {
  Persistent = 0,
  Transient = 1,
  Unknown = 2,
};

enum class EPlugParticleMultiStateRenderMode {
  LineNormal = 0,
  LineWideWorld = 1,
  LineWideScreen = 2,
  QuadCenterLeft = 3,
  QuadUp = 4,
  WaterSplash = 5,
  LightTrail = 6,
  GrassMarks = 7,
};

enum class EPlugParticleStandardRenderMode {
  QuadCamera = 0,
  WaterSplash = 1,
  QuadSpeed = 2,
  LinesSpeedCamera = 3,
  Mesh = 4,
  Triangle = 5,
  LightOnly = 6,
  Beam = 7,
  CameraWaterDroplets = 8,
  Quad = 9,
};

enum class EPlugMeshInstanceType {
  Color_Iso4 = 0,
  VolumeExplosion = 1,
};

enum class EPlugParticleTextureAtlas {
  None = 0,
  Fixed = 1,
  Random = 2,
  AnimatedRandom = 3,
  AnimatedSynchro = 4,
};

enum class EPlugParticleColorGradient {
  RandomConstantColor = 0,
  ColorOverLife = 1,
};

enum class EPlugParticleRenderColor {
  RGB = 0,
  RGBPHue = 1, // RGB+Hue
  Hue00 = 2,
  LAmbientLocals = 3,
};

enum class EPlugParticleSpawnCond {
  Active_FixedPeriod = 0, // Active&FixedPeriod
  Active_MinDist = 1, // Active&MinDist
  Active = 2,
  None = 3,
  Active_FixedDist = 4, // Active&FixedDist
};

enum class EPlugParticleEmitterSubModel {
  Standard = 0,
  MultiState = 1,
  OneParticle = 2,
  VortexSim_Particle = 3,
  VortexSim_VortexFilament = 4,
  VortexSim_Repulsor = 5,
  Chain = 6,
  Gpu = 7,
  Energy = 8,
};

enum class NPlugImpostor::EPlaneComputeMode {
  Bounding_Box_Border = 0, // Bounding Box Border
  Bounding_Box_Center = 1, // Bounding Box Center
  Trunk_Bounding_Box_Center = 2, // Trunk Bounding Box Center
  Zero = 3,
  Horizontal_Plane_Bottom = 4, // Horizontal Plane Bottom
};

enum class EPlugAnimSimpleFunc {
  Linear = 0,
  Sin = 1,
};

enum class NPlugDyna::EAxis {
  x = 0,
  y = 1,
  z = 2,
};

enum class NPlugDyna::EShaderTcType {
  None = 0,
  TransSubTexture = 1,
};

enum class NPlugDyna::EAnimFuncBase {
  Constant = 0,
  Linear = 1,
  EaseInQuad = 2,
  EaseOutQuad = 3,
  EaseInOutQuad = 4,
};

enum class EPlugDynaSleepingMethod {
  None = 0,
  LowLinearVel_AngularVel = 1,
  LowLinearVel = 2,
  Reserved0 = 3,
};

enum class NPlugDyna::EDynaConstraintType {
  PivotJoint = 0,
  Spring = 1,
  FixedPivot = 2,
  Gear = 3,
};

enum class NPlugDyna::EForceFieldType {
  Magnet = 0,
  _01 = 1,
};

enum class NPlugCurve::EInterpMode {
  Constant = 0,
  Linear = 1,
  Cubic = 2,
};

enum class NPlugCurve::ETangentMode {
  Smooth = 0,
  Broken = 1,
  Auto = 2,
};

enum class CHmsLightMapCache::ESortMode {
  None = 0,
  HDiagCenter = 1,
};

enum class CHmsLightMapCache::EAllocMode {
  _64_2 = 0, // 64^2
  _64_2PUseFree = 1, // 64^2+UseFree
  BestSizePUseFree = 2, // BestSize+UseFree
};

enum class EPlugGpuPlatform {
  _00 = 0, // 
  D3D11 = 1,
  pf3 = 2,
  pf4 = 3,
  pf5 = 4,
  pf6 = 5,
  pf7 = 6,
  pf8 = 7,
};

enum class EHmsLightMapCompress {
  Ldr_DXT1 = 0, // Ldr DXT1
  sRGB_Hyper_DXT1 = 1, // sRGB Hyper DXT1
  Hyper_sRGB_DXT1 = 2, // Hyper sRGB DXT1
  Scale_sRGB_DXT1 = 3, // Scale sRGB DXT1
};

enum class CHmsLightMapCache::EVersion {
  Invalid = 0,
  _2011_07_19_Beta1 = 1, // 2011_07_19_Beta1
  _2011_07_21_Beta1 = 2, // 2011_07_21_Beta1
  _2011_07_26_Beta1d = 3, // 2011_07_26_Beta1d
  _2011_08_04_Beta2a = 4, // 2011_08_04_Beta2a
  _2011_08_08_Beta3a = 5, // 2011_08_08_Beta3a
  _2014_03_14_Update3_Storm = 6, // 2014_03_14_Update3_Storm
  _2017_03_07_ManiaPlanet4 = 7, // 2017_03_07_ManiaPlanet4
  _2020_03_25_Beta1 = 8, // 2020_03_25_Beta1
};

enum class CHmsLightMapCache::EQualityVer {
  UltraMapperUnalignWith1k = 0,
  BounceShadowFiltered = 1,
  TinyAlloc_16b = 2,
  ShadowCube_GeomToEyeLengthBias = 3,
  BlockLight_WrongRotations = 4,
  R11G11B10F_No_BounceFactor = 5,
  HBasis_LQ_SignSqrt_BIntensScales = 6,
  ProbeGrid_HdrScaleAmbient = 7,
  ModelSplit2_GmPackReal2_V0 = 8,
  ShadowLQ = 9,
  UnmappedBlock_FullCovering = 10,
  StadiumColorisableBounces = 11,
  Item_Prefab_MultiMesh = 12,
  ShaderQLow_NeedTgtPixel = 13,
  Current = 14,
};

enum class EHmsLightMapBump {
  TxTyTz = 0,
  TxTyTz_Intens = 1,
  None = 2,
  HBasis_Color = 3,
  HBasis_Intens = 4,
};

enum class NHmsLightMapCache::ELocalLight_Storage {
  None = 0,
  All = 1,
  Only_rgb_accum = 2, // Only rgb accum
};

enum class NHmsLightMapCache::ELocalLight_Switch {
  On = 0,
  Off = 1,
  Unknown = 2,
};

enum class CHmsZoneOverlay::EPickEnableMode {
  Disabled = 0,
  Foreground = 1,
  Always = 2,
};

enum class EHmsOverlayAdaptRatio {
  None = 0,
  ReSizeX = 1,
  ReSizeY = 2,
  HMD_ScaleX = 3,
  ShrinkToKeepRatio = 4,
  ShrinkToKeepRatio_OnlyWider = 5,
};

enum class EPlugMoodFxTMFilmCurve {
  Disable = 0,
  Preset1 = 1,
  Preset2 = 2,
  Preset3 = 3,
  Preset4 = 4,
};

enum class EHmsLightMapCacheSize {
  VerySmallTest = 0,
  VerySmall = 1,
  Local2k = 2,
};

enum class CHmsSolidVisCst_TmCar::SPrestige::EMedal {
  Bronze = 0,
  Silver = 1,
  Gold = 2,
  Master = 3,
};

enum class CHmsSolidVisCst_TmCar::SPrestige::ESeasonOrType {
  Spring = 0,
  Summer = 1,
  Fall = 2,
  Winter = 3,
  Royal = 4,
  Ranked = 5,
};

enum class EPrestigeMode {
  Season = 0,
  Royal = 1,
  Ranked = 2,
};

enum class EPrestigeSeason {
  Spring = 0,
  Summer = 1,
  Fall = 2,
  Winter = 3,
};

enum class CHmsCamera::EViewportRatio {
  None = 0,
  FovY = 1,
  FovX = 2,
};

} // namespace Enums
