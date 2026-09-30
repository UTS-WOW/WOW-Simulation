using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;

namespace Naval
{
    /// <summary>
    /// The whole interface, built from code at runtime: fleet selection screen, domination status bar,
    /// message log, selected ship readout, fleet roster, RTS command panel, direct-control action bar,
    /// minimap and result screen. Buttons are hit tested against their rects, so no EventSystem is needed.
    /// </summary>
    public class UIManager : MonoBehaviour
    {
        public static UIManager I { get; private set; }

        class UIButton
        {
            public RectTransform rect;
            public Image bg;
            public Text label;
            public System.Action onClick;
            public System.Func<bool> enabled;
            public System.Func<bool> highlighted;
            public Color color;
        }

        /// <summary>Drag-anywhere-on-the-track integer slider, hit tested like the buttons.</summary>
        class UISlider
        {
            public RectTransform track;
            public Image fill;
            public Text readout;
            public int min, max;
            public System.Func<int> get;
            public System.Action<int> set;
            public System.Func<bool> visible;
            /// <summary>Turns the raw value into the readout text. Not every slider counts ships.</summary>
            public System.Func<int, string> format;
            public bool dragging;
        }

        Canvas _canvas;
        Font _font;
        readonly List<UIButton> _buttons = new List<UIButton>();
        readonly List<UISlider> _sliders = new List<UISlider>();
        static readonly List<RectTransform> _blockers = new List<RectTransform>();

        // top bar
        Text _modeText, _objectiveText, _timerText, _alliedScore, _enemyScore, _speedText, _weatherText, _zoneText, _fleetCountText;

        // control mode banner
        RectTransform _bannerPanel;
        Text _bannerText;

        // ship panel
        GameObject _shipPanel;
        Text _shipTitle, _shipVitals, _shipSensors, _shipTargetText;
        Image _shipHealthFill, _shipFuelFill, _shipAmmoFill, _shipTorpFill, _shipDcFill;
        readonly Image[] _systemFills = new Image[7];

        // fleet roster
        RectTransform _fleetPanel;
        class FleetEntry
        {
            public Ship ship;
            public RectTransform rect;
            public Image bg, healthFill, statusPip;
            public Text label;
        }
        readonly List<FleetEntry> _fleet = new List<FleetEntry>();

        // command panel (RTS) and action bar (direct)
        RectTransform _commandPanel;
        RectTransform _actionBar;
        class ActionSlot
        {
            public RectTransform rect;
            public Image bg, cooldownFill, activeGlow;
            public Text key, label, charges;
        }
        readonly ActionSlot[] _slots = new ActionSlot[7];
        Text _ammoLabel, _subLabel;
        Image _subBatteryFill, _throttleFill;
        Text _throttleLabel;

        // messages
        Text _messageText;
        readonly List<(string text, float time, Team team)> _messages = new List<(string, float, Team)>();

        // overlays
        GameObject _deployPanel, _endPanel, _helpPanel, _menuPanel;
        GameObject _editorPanel;
        Text _editorStatus, _editorCounts, _editorLoadList;
        string _editorNameEntry = "New Scenario";
        Text _endTitle, _endBody, _deployText, _menuTotalText;
        Text _warningText;
        float _warningTimer;

        RectTransform _minimapRect;
        FleetSetup _menuSetup = FleetSetup.Default();
        MapConfig _menuMap = MapConfig.ForPreset(MapPreset.OceanArchipelago);
        Text _aiDebugText;
        Text _mapSummaryText;
        Text _policyStatusText;

        public bool HelpVisible { get; private set; }

        public static UIManager Create(Transform parent)
        {
            var go = new GameObject("UI");
            go.transform.SetParent(parent, false);
            var u = go.AddComponent<UIManager>();
            I = u;
            u.Build();
            return u;
        }

        // ------------------------------------------------------------------ construction

        void Build()
        {
            _font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (_font == null) _font = Resources.GetBuiltinResource<Font>("Arial.ttf");

            var canvasGo = new GameObject("Canvas", typeof(RectTransform));
            canvasGo.transform.SetParent(transform, false);
            _canvas = canvasGo.AddComponent<Canvas>();
            _canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            _canvas.sortingOrder = 100;
            var scaler = canvasGo.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920f, 1080f);
            scaler.matchWidthOrHeight = 0.5f;

            BuildTopBar();
            BuildBanner();
            BuildMessageLog();
            BuildShipPanel();
            BuildFleetPanel();
            BuildCommandPanel();
            BuildActionBar();
            BuildMinimap();
            BuildWarning();
            BuildDeployPanel();
            BuildEditorPanel();
            BuildEndPanel();
            BuildHelpPanel();
            BuildFleetMenu();

            GameEvents.OnMessage += OnMessage;
        }

        void OnDestroy()
        {
            GameEvents.OnMessage -= OnMessage;
            _blockers.Clear();
        }

        static readonly Color PanelBg = new Color(0.05f, 0.08f, 0.12f, 0.88f);
        static readonly Color TextMain = new Color(0.85f, 0.92f, 0.97f);
        static readonly Color TextDim = new Color(0.58f, 0.68f, 0.76f);
        static readonly Color Accent = new Color(0.35f, 0.78f, 1f);
        static readonly Color AlliedCol = new Color(0.35f, 0.78f, 1f);
        static readonly Color EnemyCol = new Color(1f, 0.42f, 0.38f);

        RectTransform Panel(string name, Transform parent, Vector2 anchorMin, Vector2 anchorMax,
                            Vector2 offsetMin, Vector2 offsetMax, Color? color = null, bool blocker = true)
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(parent, false);
            var rt = (RectTransform)go.transform;
            rt.anchorMin = anchorMin; rt.anchorMax = anchorMax;
            rt.offsetMin = offsetMin; rt.offsetMax = offsetMax;
            var img = go.AddComponent<Image>();
            img.color = color ?? PanelBg;
            if (blocker) _blockers.Add(rt);
            return rt;
        }

        Image Bar(string name, Transform parent, Vector2 anchorMin, Vector2 anchorMax,
                  Vector2 offsetMin, Vector2 offsetMax, Color fill)
        {
            var bg = Panel(name + "Bg", parent, anchorMin, anchorMax, offsetMin, offsetMax, new Color(0f, 0f, 0f, 0.55f), false);
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(bg, false);
            var rt = (RectTransform)go.transform;
            rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one;
            rt.offsetMin = new Vector2(1f, 1f); rt.offsetMax = new Vector2(-1f, -1f);
            rt.pivot = new Vector2(0f, 0.5f);
            var img = go.AddComponent<Image>();
            img.color = fill;
            return img;
        }

        Text Label(string name, Transform parent, string content, int size, TextAnchor anchor, Color color,
                   Vector2 anchorMin, Vector2 anchorMax, Vector2 offsetMin, Vector2 offsetMax, FontStyle style = FontStyle.Normal)
        {
            var go = new GameObject(name, typeof(RectTransform));
            go.transform.SetParent(parent, false);
            var rt = (RectTransform)go.transform;
            rt.anchorMin = anchorMin; rt.anchorMax = anchorMax;
            rt.offsetMin = offsetMin; rt.offsetMax = offsetMax;
            var t = go.AddComponent<Text>();
            t.font = _font;
            t.text = content;
            t.fontSize = size;
            t.alignment = anchor;
            t.color = color;
            t.fontStyle = style;
            t.horizontalOverflow = HorizontalWrapMode.Overflow;
            t.verticalOverflow = VerticalWrapMode.Overflow;
            t.raycastTarget = false;
            return t;
        }

        UIButton Button(string text, Transform parent, Vector2 anchorMin, Vector2 anchorMax,
                        Vector2 offsetMin, Vector2 offsetMax, System.Action onClick,
                        System.Func<bool> enabled = null, System.Func<bool> highlighted = null, Color? tint = null)
        {
            var rt = Panel("Btn_" + text, parent, anchorMin, anchorMax, offsetMin, offsetMax,
                tint ?? new Color(0.12f, 0.19f, 0.26f, 0.95f), false);
            var label = Label("Label", rt, text, 15, TextAnchor.MiddleCenter, TextMain,
                Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero);
            var b = new UIButton
            {
                rect = rt,
                bg = rt.GetComponent<Image>(),
                label = label,
                onClick = onClick,
                enabled = enabled,
                highlighted = highlighted,
                color = tint ?? new Color(0.12f, 0.19f, 0.26f, 0.95f)
            };
            _buttons.Add(b);
            return b;
        }

        // ------------------------------------------------------------------ top bar

        void BuildTopBar()
        {
            var bar = Panel("TopBar", _canvas.transform, new Vector2(0f, 1f), new Vector2(1f, 1f),
                new Vector2(0f, -54f), new Vector2(0f, 0f), new Color(0.04f, 0.07f, 0.11f, 0.94f));

            _modeText = Label("Mode", bar, "DOMINATION", 18, TextAnchor.UpperLeft, Accent,
                new Vector2(0f, 0f), new Vector2(0f, 1f), new Vector2(16f, 6f), new Vector2(260f, -4f), FontStyle.Bold);
            _objectiveText = Label("Objective", bar, "", 12, TextAnchor.LowerLeft, TextDim,
                new Vector2(0f, 0f), new Vector2(0f, 1f), new Vector2(16f, 4f), new Vector2(560f, -22f));

            // centre block: allied score | countdown | enemy score
            _alliedScore = Label("Allied", bar, "0", 26, TextAnchor.MiddleRight, AlliedCol,
                new Vector2(0.5f, 0f), new Vector2(0.5f, 1f), new Vector2(-230f, 0f), new Vector2(-90f, 0f), FontStyle.Bold);
            _timerText = Label("Timer", bar, "15:00", 30, TextAnchor.MiddleCenter, TextMain,
                new Vector2(0.5f, 0f), new Vector2(0.5f, 1f), new Vector2(-85f, 0f), new Vector2(85f, 0f), FontStyle.Bold);
            _enemyScore = Label("Enemy", bar, "0", 26, TextAnchor.MiddleLeft, EnemyCol,
                new Vector2(0.5f, 0f), new Vector2(0.5f, 1f), new Vector2(90f, 0f), new Vector2(230f, 0f), FontStyle.Bold);

            _zoneText = Label("Zones", bar, "", 13, TextAnchor.UpperCenter, TextMain,
                new Vector2(0.5f, 0f), new Vector2(0.5f, 1f), new Vector2(-260f, 2f), new Vector2(260f, -34f));

            _fleetCountText = Label("Fleets", bar, "", 13, TextAnchor.MiddleRight, TextMain,
                new Vector2(1f, 0f), new Vector2(1f, 1f), new Vector2(-560f, 0f), new Vector2(-330f, 0f));
            _weatherText = Label("Weather", bar, "", 12, TextAnchor.MiddleRight, TextDim,
                new Vector2(1f, 0f), new Vector2(1f, 1f), new Vector2(-330f, 0f), new Vector2(-232f, 0f));

            _speedText = Label("SpeedLabel", bar, "SPEED", 11, TextAnchor.UpperRight, TextDim,
                new Vector2(1f, 0f), new Vector2(1f, 1f), new Vector2(-226f, 6f), new Vector2(-8f, -4f));

            float x = -222f;
            AddSpeedButton(bar, "||", 0f, ref x);
            AddSpeedButton(bar, "1x", 1f, ref x);
            AddSpeedButton(bar, "2x", 2f, ref x);
            AddSpeedButton(bar, "4x", 4f, ref x);
            AddSpeedButton(bar, "8x", 8f, ref x);
        }

        void AddSpeedButton(RectTransform bar, string label, float speed, ref float x)
        {
            float w = 42f;
            Button(label, bar, new Vector2(1f, 0f), new Vector2(1f, 1f),
                new Vector2(x, 6f), new Vector2(x + w, -22f),
                () => GameManager.I.SetSpeed(speed), null,
                () => GameManager.I != null && Mathf.Approximately(GameManager.I.GameSpeed, speed));
            x += w + 3f;
        }

        void BuildBanner()
        {
            _bannerPanel = Panel("Banner", _canvas.transform, new Vector2(0.5f, 1f), new Vector2(0.5f, 1f),
                new Vector2(-300f, -86f), new Vector2(300f, -58f), new Color(0.06f, 0.12f, 0.18f, 0.8f), false);
            _bannerText = Label("BannerText", _bannerPanel, "", 15, TextAnchor.MiddleCenter, Accent,
                Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero, FontStyle.Bold);
        }

        void BuildMessageLog()
        {
            var p = Panel("Messages", _canvas.transform, new Vector2(0f, 1f), new Vector2(0f, 1f),
                new Vector2(12f, -262f), new Vector2(400f, -96f), new Color(0.04f, 0.07f, 0.11f, 0.45f), false);
            _messageText = Label("Log", p, "", 14, TextAnchor.UpperLeft, TextMain,
                Vector2.zero, Vector2.one, new Vector2(10f, 6f), new Vector2(-8f, -6f));
            _messageText.horizontalOverflow = HorizontalWrapMode.Wrap;

            // AI reasoning readout, only while debug draw is on (F1)
            var dbg = Panel("AIDebug", _canvas.transform, new Vector2(0f, 1f), new Vector2(0f, 1f),
                new Vector2(12f, -352f), new Vector2(430f, -266f), new Color(0.03f, 0.06f, 0.1f, 0.6f), false);
            _aiDebugText = Label("AIDebugText", dbg, "", 13, TextAnchor.UpperLeft, new Color(0.8f, 0.9f, 0.75f),
                Vector2.zero, Vector2.one, new Vector2(10f, 6f), new Vector2(-8f, -6f));
            _aiDebugText.horizontalOverflow = HorizontalWrapMode.Wrap;
            dbg.gameObject.SetActive(false);
        }

        void BuildWarning()
        {
            var p = Panel("Warning", _canvas.transform, new Vector2(0.5f, 1f), new Vector2(0.5f, 1f),
                new Vector2(-260f, -128f), new Vector2(260f, -92f), new Color(0.5f, 0.08f, 0.06f, 0.85f), false);
            _warningText = Label("WarnText", p, "", 20, TextAnchor.MiddleCenter, new Color(1f, 0.9f, 0.85f),
                Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero, FontStyle.Bold);
            p.gameObject.SetActive(false);
        }

        // ------------------------------------------------------------------ ship panel

        void BuildShipPanel()
        {
            var p = Panel("ShipPanel", _canvas.transform, new Vector2(0f, 0f), new Vector2(0f, 0f),
                new Vector2(12f, 12f), new Vector2(432f, 250f));
            _shipPanel = p.gameObject;

            _shipTitle = Label("Title", p, "NO SHIP SELECTED", 18, TextAnchor.UpperLeft, Accent,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(12f, -32f), new Vector2(-12f, -6f), FontStyle.Bold);

            _shipHealthFill = Bar("Health", p, new Vector2(0f, 1f), new Vector2(1f, 1f),
                new Vector2(12f, -52f), new Vector2(-12f, -34f), new Color(0.35f, 0.9f, 0.45f));
            Label("HealthCap", p, "HULL", 11, TextAnchor.UpperLeft, TextDim,
                new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(14f, -50f), new Vector2(80f, -36f));

            _shipVitals = Label("Vitals", p, "", 14, TextAnchor.UpperLeft, TextMain,
                new Vector2(0f, 1f), new Vector2(0.55f, 1f), new Vector2(12f, -126f), new Vector2(-4f, -56f));
            _shipSensors = Label("Sensors", p, "", 14, TextAnchor.UpperLeft, TextMain,
                new Vector2(0.5f, 1f), new Vector2(1f, 1f), new Vector2(4f, -126f), new Vector2(-12f, -56f));

            Label("FuelCap", p, "FUEL", 11, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 0f), new Vector2(0f, 0f), new Vector2(12f, 96f), new Vector2(56f, 112f));
            _shipFuelFill = Bar("Fuel", p, new Vector2(0f, 0f), new Vector2(0.5f, 0f),
                new Vector2(58f, 96f), new Vector2(-8f, 112f), new Color(0.9f, 0.7f, 0.3f));

            Label("AmmoCap", p, "AMMO", 11, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0.5f, 0f), new Vector2(0.5f, 0f), new Vector2(6f, 96f), new Vector2(52f, 112f));
            _shipAmmoFill = Bar("Ammo", p, new Vector2(0.5f, 0f), new Vector2(1f, 0f),
                new Vector2(54f, 96f), new Vector2(-12f, 112f), new Color(0.8f, 0.85f, 0.9f));

            Label("TorpCap", p, "TORP", 11, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 0f), new Vector2(0f, 0f), new Vector2(12f, 76f), new Vector2(56f, 92f));
            _shipTorpFill = Bar("Torp", p, new Vector2(0f, 0f), new Vector2(0.5f, 0f),
                new Vector2(58f, 76f), new Vector2(-8f, 92f), new Color(0.4f, 0.95f, 0.85f));

            Label("DcCap", p, "D.C.", 11, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0.5f, 0f), new Vector2(0.5f, 0f), new Vector2(6f, 76f), new Vector2(52f, 92f));
            _shipDcFill = Bar("Dc", p, new Vector2(0.5f, 0f), new Vector2(1f, 0f),
                new Vector2(54f, 76f), new Vector2(-12f, 92f), new Color(0.6f, 0.8f, 1f));

            string[] names = { "HULL", "ENG", "STEER", "MAIN", "SEC", "SENS", "PROP" };
            for (int i = 0; i < 7; i++)
            {
                int col = i % 4, row = i / 4;
                float x0 = 12f + col * 102f;
                float y0 = 34f - row * 22f;
                Label("SysLbl" + i, p, names[i], 10, TextAnchor.MiddleLeft, TextDim,
                    new Vector2(0f, 0f), new Vector2(0f, 0f), new Vector2(x0, y0), new Vector2(x0 + 40f, y0 + 16f));
                _systemFills[i] = Bar("Sys" + i, p, new Vector2(0f, 0f), new Vector2(0f, 0f),
                    new Vector2(x0 + 40f, y0 + 3f), new Vector2(x0 + 96f, y0 + 13f), new Color(0.4f, 0.85f, 0.6f));
            }

            _shipTargetText = Label("TargetText", p, "", 13, TextAnchor.MiddleLeft, new Color(1f, 0.6f, 0.5f),
                new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(12f, 116f), new Vector2(-12f, 134f));
        }

        // ------------------------------------------------------------------ fleet roster

        void BuildFleetPanel()
        {
            _fleetPanel = Panel("FleetPanel", _canvas.transform, new Vector2(0f, 0f), new Vector2(0f, 0f),
                new Vector2(12f, 262f), new Vector2(232f, 800f));
            Label("FleetTitle", _fleetPanel, "TASK FORCE", 14, TextAnchor.UpperLeft, Accent,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(10f, -24f), new Vector2(-10f, -4f), FontStyle.Bold);
        }

        void RebuildFleetEntries()
        {
            for (int i = 0; i < _fleet.Count; i++)
                if (_fleet[i].rect != null) Destroy(_fleet[i].rect.gameObject);
            _fleet.Clear();

            var ships = ShipRegistry.OfTeam(Team.Player);
            // 18 hulls: a compact two-line-per-ship row would not fit, so rows are tight
            for (int i = 0; i < ships.Count; i++)
            {
                var s = ships[i];
                float y = -28f - i * 27f;
                var rt = Panel("Fleet" + i, _fleetPanel, new Vector2(0f, 1f), new Vector2(1f, 1f),
                    new Vector2(6f, y - 24f), new Vector2(-6f, y), new Color(0.09f, 0.14f, 0.2f, 0.9f), false);

                var pip = Panel("Pip", rt, new Vector2(0f, 0f), new Vector2(0f, 1f),
                    new Vector2(0f, 0f), new Vector2(4f, 0f), Teams.Color(Team.Player), false);

                var label = Label("Name", rt, s.ClassTag + " " + Shorten(s.shipName), 12, TextAnchor.MiddleLeft, TextMain,
                    new Vector2(0f, 0.4f), new Vector2(1f, 1f), new Vector2(9f, 0f), new Vector2(-6f, -1f));

                var fill = Bar("Hp", rt, new Vector2(0f, 0f), new Vector2(1f, 0f),
                    new Vector2(9f, 4f), new Vector2(-6f, 10f), new Color(0.35f, 0.9f, 0.45f));

                _fleet.Add(new FleetEntry { ship = s, rect = rt, bg = rt.GetComponent<Image>(), healthFill = fill, label = label, statusPip = pip.GetComponent<Image>() });
            }
        }

        static string Shorten(string n)
        {
            if (string.IsNullOrEmpty(n)) return n;
            int space = n.IndexOf(' ');
            return space >= 0 && space < n.Length - 1 ? n.Substring(space + 1) : n;
        }

        // ------------------------------------------------------------------ command panel (RTS)

        void BuildCommandPanel()
        {
            _commandPanel = Panel("CommandPanel", _canvas.transform, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f),
                new Vector2(-330f, 12f), new Vector2(330f, 214f));

            Label("CmdTitle", _commandPanel, "FLEET COMMAND", 13, TextAnchor.UpperLeft, Accent,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(10f, -20f), new Vector2(-10f, -2f), FontStyle.Bold);

            string[,] labels =
            {
                { "Attack Mv (C)", "Patrol (V)", "Escort (B)", "Focus (T)", "Stop (Spc)" },
                { "Hold (H)", "Astern (R)", "Retreat (G)", "To Port (Y)", "Repair (E)" },
                { "Smoke (Q)", "Dive (Z)", "Surface (X)", "Hold Fire", "Select All" },
                { "Line Ahead", "Abreast", "Wedge", "Circle", "Screen" }
            };

            System.Action[,] actions =
            {
                {
                    () => SetPending(PendingCommand.AttackMove),
                    () => SetPending(PendingCommand.Patrol),
                    () => SetPending(PendingCommand.Follow),
                    () => SetPending(PendingCommand.FocusTarget),
                    () => CommandSystem.I.Stop()
                },
                {
                    () => CommandSystem.I.Hold(),
                    () => CommandSystem.I.Reverse(),
                    () => CommandSystem.I.Retreat(),
                    () => CommandSystem.I.ReturnToPort(),
                    () => CommandSystem.I.DamageControl()
                },
                {
                    () => CommandSystem.I.Smoke(),
                    () => CommandSystem.I.Dive(),
                    () => CommandSystem.I.Surface(),
                    () => CommandSystem.I.ToggleHoldFire(),
                    () => SelectionManager.I.SelectAll()
                },
                {
                    () => CommandSystem.I.SetFormation(FormationType.LineAhead),
                    () => CommandSystem.I.SetFormation(FormationType.LineAbreast),
                    () => CommandSystem.I.SetFormation(FormationType.Wedge),
                    () => CommandSystem.I.SetFormation(FormationType.Circle),
                    () => CommandSystem.I.SetFormation(FormationType.DefensiveScreen)
                }
            };

            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 5; c++)
                {
                    float x0 = 10f + c * 128f;
                    float y1 = -26f - r * 44f;
                    var action = actions[r, c];
                    int rr = r, cc = c;
                    var btn = Button(labels[r, c], _commandPanel, new Vector2(0f, 1f), new Vector2(0f, 1f),
                        new Vector2(x0, y1 - 40f), new Vector2(x0 + 122f, y1),
                        action,
                        () => SelectionManager.I != null && SelectionManager.I.Selected.Count > 0,
                        () => IsCommandHighlighted(rr, cc));
                    btn.label.fontSize = 13;
                }
        }

        bool IsCommandHighlighted(int row, int col)
        {
            var cs = CommandSystem.I;
            if (cs == null) return false;
            if (row == 0)
            {
                switch (col)
                {
                    case 0: return cs.Pending == PendingCommand.AttackMove;
                    case 1: return cs.Pending == PendingCommand.Patrol;
                    case 2: return cs.Pending == PendingCommand.Follow;
                    case 3: return cs.Pending == PendingCommand.FocusTarget;
                }
            }
            if (row == 3)
            {
                FormationType t = col == 0 ? FormationType.LineAhead : col == 1 ? FormationType.LineAbreast
                                : col == 2 ? FormationType.Wedge : col == 3 ? FormationType.Circle : FormationType.DefensiveScreen;
                return cs.CurrentFormation == t;
            }
            return false;
        }

        void SetPending(PendingCommand c)
        {
            if (CommandSystem.I != null) CommandSystem.I.SetPending(c);
        }

        // ------------------------------------------------------------------ action bar (direct)

        void BuildActionBar()
        {
            _actionBar = Panel("ActionBar", _canvas.transform, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f),
                new Vector2(-438f, 12f), new Vector2(438f, 150f));

            Label("BarTitle", _actionBar, "CONSUMABLES", 13, TextAnchor.UpperLeft, Accent,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(10f, -20f), new Vector2(-10f, -2f), FontStyle.Bold);

            _ammoLabel = Label("Ammo", _actionBar, "", 13, TextAnchor.UpperRight, new Color(1f, 0.85f, 0.4f),
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(10f, -20f), new Vector2(-10f, -2f), FontStyle.Bold);

            for (int i = 0; i < _slots.Length; i++)
            {
                float x0 = 12f + i * 123f;
                var slot = new ActionSlot();
                slot.rect = Panel("Slot" + i, _actionBar, new Vector2(0f, 1f), new Vector2(0f, 1f),
                    new Vector2(x0, -96f), new Vector2(x0 + 115f, -26f), new Color(0.11f, 0.17f, 0.24f, 0.95f), false);
                slot.bg = slot.rect.GetComponent<Image>();

                // cooldown sweep drawn as a bottom-up fill behind the text
                var cd = new GameObject("Cooldown", typeof(RectTransform));
                cd.transform.SetParent(slot.rect, false);
                var cdRt = (RectTransform)cd.transform;
                cdRt.anchorMin = Vector2.zero; cdRt.anchorMax = Vector2.one;
                cdRt.offsetMin = Vector2.zero; cdRt.offsetMax = Vector2.zero;
                cdRt.pivot = new Vector2(0.5f, 0f);
                slot.cooldownFill = cd.AddComponent<Image>();
                slot.cooldownFill.color = new Color(0.05f, 0.09f, 0.14f, 0.82f);
                slot.cooldownFill.raycastTarget = false;

                slot.key = Label("Key", slot.rect, "", 15, TextAnchor.UpperLeft, Accent,
                    new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(7f, -22f), new Vector2(27f, -3f), FontStyle.Bold);
                slot.label = Label("Name", slot.rect, "", 12, TextAnchor.MiddleCenter, TextMain,
                    new Vector2(0f, 0f), new Vector2(1f, 1f), new Vector2(4f, 14f), new Vector2(-4f, -20f));
                slot.label.horizontalOverflow = HorizontalWrapMode.Wrap;
                slot.charges = Label("Charges", slot.rect, "", 11, TextAnchor.LowerRight, TextDim,
                    new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(4f, 4f), new Vector2(-7f, 20f));

                _slots[i] = slot;
            }

            // engine telegraph
            Label("ThrottleCap", _actionBar, "THROTTLE", 10, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 0f), new Vector2(0f, 0f), new Vector2(12f, 8f), new Vector2(72f, 24f));
            _throttleFill = Bar("Throttle", _actionBar, new Vector2(0f, 0f), new Vector2(0f, 0f),
                new Vector2(74f, 9f), new Vector2(254f, 23f), new Color(0.45f, 0.9f, 0.6f));
            _throttleLabel = Label("ThrottleVal", _actionBar, "", 11, TextAnchor.MiddleLeft, TextMain,
                new Vector2(0f, 0f), new Vector2(0f, 0f), new Vector2(260f, 8f), new Vector2(400f, 24f));

            // submarine depth and battery
            _subLabel = Label("SubDepth", _actionBar, "", 11, TextAnchor.MiddleRight, new Color(0.5f, 0.9f, 1f),
                new Vector2(1f, 0f), new Vector2(1f, 0f), new Vector2(-300f, 8f), new Vector2(-150f, 24f));
            _subBatteryFill = Bar("Battery", _actionBar, new Vector2(1f, 0f), new Vector2(1f, 0f),
                new Vector2(-146f, 9f), new Vector2(-12f, 23f), new Color(0.4f, 0.9f, 1f));
        }

        // ------------------------------------------------------------------ minimap

        void BuildMinimap()
        {
            var frame = Panel("MinimapFrame", _canvas.transform, new Vector2(1f, 0f), new Vector2(1f, 0f),
                new Vector2(-274f, 12f), new Vector2(-12f, 274f), new Color(0.04f, 0.07f, 0.11f, 0.95f));

            var go = new GameObject("MinimapImage", typeof(RectTransform));
            go.transform.SetParent(frame, false);
            var rt = (RectTransform)go.transform;
            rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one;
            rt.offsetMin = new Vector2(6f, 6f); rt.offsetMax = new Vector2(-6f, -6f);
            var raw = go.AddComponent<RawImage>();
            raw.raycastTarget = false;
            _minimapRect = rt;

            Minimap.Create(transform, raw, rt);
        }

        // ------------------------------------------------------------------ fleet selection menu

        UISlider Slider(string name, Transform parent, Vector2 offMin, Vector2 offMax,
                        int min, int max, System.Func<int> get, System.Action<int> set,
                        System.Func<int, string> format = null)
        {
            var track = Panel(name + "Track", parent, new Vector2(0f, 1f), new Vector2(0f, 1f),
                offMin, offMax, new Color(0f, 0f, 0f, 0.5f), false);

            var fillGo = new GameObject(name + "Fill", typeof(RectTransform));
            fillGo.transform.SetParent(track, false);
            var frt = (RectTransform)fillGo.transform;
            frt.anchorMin = Vector2.zero; frt.anchorMax = Vector2.one;
            frt.offsetMin = new Vector2(2f, 2f); frt.offsetMax = new Vector2(-2f, -2f);
            frt.pivot = new Vector2(0f, 0.5f);
            var fill = fillGo.AddComponent<Image>();
            fill.color = new Color(0.2f, 0.55f, 0.75f, 0.95f);

            var readout = Label(name + "Val", track, "", 15, TextAnchor.MiddleCenter, TextMain,
                Vector2.zero, Vector2.one, Vector2.zero, Vector2.zero, FontStyle.Bold);

            var s = new UISlider { track = track, fill = fill, readout = readout, min = min, max = max,
                                   get = get, set = set, format = format };
            _sliders.Add(s);
            return s;
        }

        void BuildFleetMenu()
        {
            var p = Panel("FleetMenu", _canvas.transform, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f),
                new Vector2(-440f, -390f), new Vector2(440f, 390f), new Color(0.04f, 0.08f, 0.13f, 0.97f));
            _menuPanel = p.gameObject;

            Label("MenuTitle", p, "BATTLE SETUP", 28, TextAnchor.UpperCenter, Accent,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -50f), new Vector2(0f, -12f), FontStyle.Bold);

            // ---- fleet sizes -------------------------------------------------
            Label("PC", p, "YOUR FLEET", 14, TextAnchor.MiddleLeft, TextMain,
                new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(30f, -86f), new Vector2(170f, -60f), FontStyle.Bold);
            Slider("PlayerCount", p, new Vector2(176f, -86f), new Vector2(560f, -60f),
                FleetSetup.MinShips, FleetSetup.MaxShips,
                () => _menuSetup.playerShipCount, v => _menuSetup.playerShipCount = v);

            Label("EC", p, "ENEMY FLEET", 14, TextAnchor.MiddleLeft, TextMain,
                new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(30f, -120f), new Vector2(170f, -94f), FontStyle.Bold);
            Slider("EnemyCount", p, new Vector2(176f, -120f), new Vector2(560f, -94f),
                FleetSetup.MinShips, FleetSetup.MaxShips,
                () => _menuSetup.enemyShipCount, v => _menuSetup.enemyShipCount = v);

            Button("MATCH", p, new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(576f, -120f), new Vector2(676f, -60f),
                () => { _menuSetup.enemyShipCount = _menuSetup.playerShipCount; });

            // ---- composition mode --------------------------------------------
            Label("CM", p, "ENEMY COMPOSITION", 14, TextAnchor.MiddleLeft, TextMain,
                new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(30f, -160f), new Vector2(210f, -134f), FontStyle.Bold);
            var modes = new[] { FleetCompositionMode.Balanced, FleetCompositionMode.Custom, FleetCompositionMode.Mirror };
            var modeNames = new[] { "BALANCED", "CUSTOM SLOTS", "MIRROR MINE" };
            for (int i = 0; i < modes.Length; i++)
            {
                var m = modes[i];
                float x = 216f + i * 156f;
                Button(modeNames[i], p, new Vector2(0f, 1f), new Vector2(0f, 1f),
                    new Vector2(x, -162f), new Vector2(x + 148f, -132f),
                    () => { _menuSetup.compositionMode = m; }, null, () => _menuSetup.compositionMode == m);
            }

            // ---- per class allocation (custom mode) ---------------------------
            var classes = new[] { ShipClassType.Battleship, ShipClassType.Cruiser,
                                  ShipClassType.Destroyer, ShipClassType.Submarine };
            var blurbs = new[]
            {
                "Slow and sluggish, huge health and armour, devastating slow guns.",
                "Balanced, strong utility: hydroacoustic search and surveillance radar.",
                "Fastest and most agile, fragile, quick guns, torpedoes and smoke.",
                "Stealthy and fragile. Dives to hide, hunts with homing torpedoes."
            };

            for (int i = 0; i < classes.Length; i++)
            {
                var cls = classes[i];
                float y = -196f - i * 64f;

                Label("Cls" + i, p, ShipDatabase.ShortTag(cls) + "  " + cls.ToString().ToUpper(), 15, TextAnchor.UpperLeft, TextMain,
                    new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(30f, y - 20f), new Vector2(220f, y), FontStyle.Bold);
                Label("Blurb" + i, p, blurbs[i], 11, TextAnchor.UpperLeft, TextDim,
                    new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(30f, y - 36f), new Vector2(-360f, y - 18f));

                Button("-", p, new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(-352f, y - 30f), new Vector2(-320f, y),
                    () => { _menuSetup.Adjust(cls, -1); },
                    () => _menuSetup.compositionMode == FleetCompositionMode.Custom && _menuSetup.CountOf(cls) > 0);

                var count = Label("Count" + i, p, "", 18, TextAnchor.MiddleCenter, TextMain,
                    new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(-318f, y - 30f), new Vector2(-280f, y), FontStyle.Bold);
                count.name = "MenuCount_" + cls;

                Button("+", p, new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(-278f, y - 30f), new Vector2(-246f, y),
                    () => { _menuSetup.Adjust(cls, +1); },
                    () => _menuSetup.compositionMode == FleetCompositionMode.Custom);

                Button("COMMAND", p, new Vector2(1f, 1f), new Vector2(1f, 1f), new Vector2(-238f, y - 30f), new Vector2(-100f, y),
                    () => { _menuSetup.controlClass = cls; },
                    null, () => _menuSetup.controlClass == cls);
            }

            // ---- battlefield --------------------------------------------------
            Label("MapHdr", p, "BATTLEFIELD", 14, TextAnchor.MiddleLeft, TextMain,
                new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(30f, -486f), new Vector2(200f, -460f), FontStyle.Bold);

            var presets = new[] { MapPreset.OceanArchipelago, MapPreset.OpenSea, MapPreset.StraitClash };
            var presetNames = new[] { "ARCHIPELAGO", "OPEN SEA", "STRAIT CLASH" };
            for (int i = 0; i < presets.Length; i++)
            {
                var mp = presets[i];
                float x = 176f + i * 150f;
                Button(presetNames[i], p, new Vector2(0f, 1f), new Vector2(0f, 1f),
                    new Vector2(x, -488f), new Vector2(x + 142f, -458f),
                    () => { _menuMap = MapConfig.ForPreset(mp); },      // preset resets its own defaults
                    null, () => _menuMap.preset == mp);
            }

            // island density
            Label("DenHdr", p, "ISLANDS", 12, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(30f, -520f), new Vector2(120f, -496f));
            var densities = new[] { IslandDensity.Low, IslandDensity.Medium, IslandDensity.High, IslandDensity.Procedural };
            var densityNames = new[] { "LOW", "MED", "HIGH", "RANDOM" };
            for (int i = 0; i < densities.Length; i++)
            {
                var d = densities[i];
                float x = 122f + i * 74f;
                Button(densityNames[i], p, new Vector2(0f, 1f), new Vector2(0f, 1f),
                    new Vector2(x, -522f), new Vector2(x + 68f, -496f),
                    () => { _menuMap.islandDensity = d; }, null, () => _menuMap.islandDensity == d);
            }

            // weather
            Label("WxHdr", p, "WEATHER", 12, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(424f, -520f), new Vector2(510f, -496f));
            var weathers = new[] { WeatherType.Clear, WeatherType.Fog, WeatherType.Storm };
            var weatherNames = new[] { "CLEAR", "FOG", "STORM" };
            for (int i = 0; i < weathers.Length; i++)
            {
                var w = weathers[i];
                float x = 500f + i * 92f;
                Button(weatherNames[i], p, new Vector2(0f, 1f), new Vector2(0f, 1f),
                    new Vector2(x, -522f), new Vector2(x + 86f, -496f),
                    () => { _menuMap.weather = w; }, null, () => _menuMap.weather == w);
            }

            // capture radius, shown in metres (1 unit is about 10 m)
            Label("CapHdr", p, "CAP SIZE", 12, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(30f, -554f), new Vector2(120f, -530f));
            Slider("CapRadius", p, new Vector2(122f, -556f), new Vector2(420f, -530f),
                Mathf.RoundToInt(MapConfig.MinCaptureRadius), Mathf.RoundToInt(MapConfig.MaxCaptureRadius),
                () => Mathf.RoundToInt(_menuMap.captureRadius),
                v => _menuMap.captureRadius = v,
                v => (v * 10) + " m RADIUS");        // 1 world unit is about 10 m

            _mapSummaryText = Label("MapSummary", p, "", 12, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(430f, -556f), new Vector2(-30f, -530f));

            // ---- who drives each fleet: the rule AI, you, or a trained policy -----------
            Label("AiHdr", p, "ENEMY AI", 12, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(30f, -598f), new Vector2(120f, -574f));
            Button("RULE-BASED", p, new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(122f, -600f), new Vector2(252f, -572f),
                () => { _menuSetup.enemyController = ShipController.RuleAI; }, null,
                () => _menuSetup.enemyController == ShipController.RuleAI);
            Button("LEARNED", p, new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(258f, -600f), new Vector2(388f, -572f),
                () => { _menuSetup.enemyController = ShipController.Learned; }, PolicyAvailable,
                () => _menuSetup.enemyController == ShipController.Learned);
            Label("YouHdr", p, "YOUR FLEET", 12, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(424f, -598f), new Vector2(520f, -574f));
            Button("YOU COMMAND", p, new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(522f, -600f), new Vector2(662f, -572f),
                () => { _menuSetup.playerController = ShipController.Human; }, null,
                () => _menuSetup.playerController == ShipController.Human);
            Button("LEARNED", p, new Vector2(0f, 1f), new Vector2(0f, 1f), new Vector2(668f, -600f), new Vector2(798f, -572f),
                () => { _menuSetup.playerController = ShipController.Learned; }, PolicyAvailable,
                () => _menuSetup.playerController == ShipController.Learned);
            _policyStatusText = Label("PolicyStatus", p, "", 11, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(30f, -628f), new Vector2(-30f, -606f));

            _menuTotalText = Label("Total", p, "", 16, TextAnchor.MiddleCenter, TextMain,
                new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(0f, 96f), new Vector2(0f, 124f), FontStyle.Bold);

            Label("StartAs", p, "START AS", 12, TextAnchor.MiddleRight, TextDim,
                new Vector2(0f, 0f), new Vector2(0f, 0f), new Vector2(150f, 62f), new Vector2(238f, 88f));
            Button("FLEET COMMANDER", p, new Vector2(0f, 0f), new Vector2(0f, 0f), new Vector2(246f, 60f), new Vector2(426f, 90f),
                () => { _menuSetup.startAsCaptain = false; }, null, () => !_menuSetup.startAsCaptain);
            Button("SHIP CAPTAIN", p, new Vector2(0f, 0f), new Vector2(0f, 0f), new Vector2(434f, 60f), new Vector2(594f, 90f),
                () => { _menuSetup.startAsCaptain = true; }, null, () => _menuSetup.startAsCaptain);

            Button("RESET", p, new Vector2(0f, 0f), new Vector2(0f, 0f), new Vector2(30f, 18f), new Vector2(170f, 52f),
                () => { _menuSetup = FleetSetup.Default(); });

            // hand-authored battles for training live on their own screen
            Button("SCENARIO EDITOR", p, new Vector2(1f, 0f), new Vector2(1f, 0f),
                new Vector2(-210f, 18f), new Vector2(-30f, 52f),
                () => GameManager.I.EnterEditor(), null, null, new Color(0.18f, 0.26f, 0.40f, 0.95f));

            Button("LAUNCH BATTLE", p, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f), new Vector2(-130f, 18f), new Vector2(130f, 52f),
                () => GameManager.I.StartFromMenu(_menuSetup, 0, _menuMap),
                MenuIsValid, null, new Color(0.15f, 0.42f, 0.3f, 0.95f));

            _menuPanel.SetActive(false);
        }

        static bool PolicyAvailable() => RL.RLPolicyDriver.I != null && RL.RLPolicyDriver.I.Available;

        bool PolicyMissing() =>
            (_menuSetup.enemyController == ShipController.Learned || _menuSetup.playerController == ShipController.Learned) && !PolicyAvailable();

        bool MenuIsValid()
        {
            if (PolicyMissing()) return false;
            if (_menuSetup.compositionMode == FleetCompositionMode.Custom && _menuSetup.CustomTotal < 1) return false;
            if (_menuSetup.playerShipCount < FleetSetup.MinShips || _menuSetup.enemyShipCount < FleetSetup.MinShips) return false;
            return true;
        }

        // ------------------------------------------------------------------ overlays

        // ------------------------------------------------------------------ scenario editor

        void BuildEditorPanel()
        {
            // a frame rather than a full-screen sheet: the map has to stay visible and clickable
            var p = Panel("EditorPanel", _canvas.transform, Vector2.zero, Vector2.one,
                Vector2.zero, Vector2.zero, new Color(0f, 0f, 0f, 0f), false);
            _editorPanel = p.gameObject;

            // ---- top strip -------------------------------------------------
            var top = Panel("EdTop", p, new Vector2(0f, 1f), new Vector2(1f, 1f),
                new Vector2(0f, -46f), new Vector2(0f, 0f), new Color(0.04f, 0.08f, 0.13f, 0.96f));
            Label("EdTitle", top, "SCENARIO EDITOR", 20, TextAnchor.MiddleLeft, Accent,
                new Vector2(0f, 0f), new Vector2(0f, 1f), new Vector2(16f, 0f), new Vector2(240f, 0f), FontStyle.Bold);
            _editorCounts = Label("EdCounts", top, "", 13, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 0f), new Vector2(0f, 1f), new Vector2(250f, 0f), new Vector2(700f, 0f));

            Button("SAVE", top, new Vector2(1f, 0f), new Vector2(1f, 1f), new Vector2(-560f, 8f), new Vector2(-470f, -8f),
                () => SaveScenario());
            Button("LOAD", top, new Vector2(1f, 0f), new Vector2(1f, 1f), new Vector2(-462f, 8f), new Vector2(-372f, -8f),
                () => LoadScenario());
            Button("CLEAR", top, new Vector2(1f, 0f), new Vector2(1f, 1f), new Vector2(-364f, 8f), new Vector2(-274f, -8f),
                () => { if (ScenarioEditor.I != null) ScenarioEditor.I.ClearAll(); });
            Button("BACK", top, new Vector2(1f, 0f), new Vector2(1f, 1f), new Vector2(-266f, 8f), new Vector2(-176f, -8f),
                () => GameManager.I.EnterMenu(true));
            Button("PLAY SCENARIO", top, new Vector2(1f, 0f), new Vector2(1f, 1f), new Vector2(-168f, 8f), new Vector2(-12f, -8f),
                () => { if (ScenarioEditor.I != null) GameManager.I.BeginScenario(ScenarioEditor.I.Current); },
                () => ScenarioEditor.I != null && !ScenarioEditor.I.Current.IsEmpty,
                null, new Color(0.15f, 0.42f, 0.3f, 0.95f));

            // ---- left tool palette ------------------------------------------
            var left = Panel("EdTools", p, new Vector2(0f, 0f), new Vector2(0f, 1f),
                new Vector2(0f, 0f), new Vector2(212f, -46f), new Color(0.04f, 0.08f, 0.13f, 0.94f));

            Label("ToolHdr", left, "TOOL", 13, TextAnchor.MiddleLeft, Accent,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(14f, -30f), new Vector2(-10f, -10f), FontStyle.Bold);
            var tools = new[] { EditorTool.Select, EditorTool.PlaceShip, EditorTool.PlaceZone, EditorTool.PlaceIsland };
            var toolNames = new[] { "SELECT / MOVE", "PLACE SHIP", "PLACE ZONE", "PLACE ISLAND" };
            for (int i = 0; i < tools.Length; i++)
            {
                var t = tools[i];
                float y = -38f - i * 34f;
                Button(toolNames[i], left, new Vector2(0f, 1f), new Vector2(1f, 1f),
                    new Vector2(12f, y - 30f), new Vector2(-12f, y),
                    () => { if (ScenarioEditor.I != null) ScenarioEditor.I.Tool = t; },
                    null, () => ScenarioEditor.I != null && ScenarioEditor.I.Tool == t);
            }

            Label("SideHdr", left, "SIDE", 13, TextAnchor.MiddleLeft, Accent,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(14f, -200f), new Vector2(-10f, -180f), FontStyle.Bold);
            var teams = new[] { Team.Player, Team.Enemy };
            var teamNames = new[] { "FRIENDLY", "HOSTILE" };
            for (int i = 0; i < teams.Length; i++)
            {
                var t = teams[i];
                float x = 12f + i * 96f;
                Button(teamNames[i], left, new Vector2(0f, 1f), new Vector2(0f, 1f),
                    new Vector2(x, -238f), new Vector2(x + 88f, -208f),
                    () => { if (ScenarioEditor.I != null) ScenarioEditor.I.PaletteTeam = t; },
                    null, () => ScenarioEditor.I != null && ScenarioEditor.I.PaletteTeam == t);
            }

            Label("ClsHdr", left, "SHIP CLASS", 13, TextAnchor.MiddleLeft, Accent,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(14f, -272f), new Vector2(-10f, -252f), FontStyle.Bold);
            var classes = new[] { ShipClassType.Battleship, ShipClassType.Cruiser,
                                  ShipClassType.Destroyer, ShipClassType.Submarine };
            for (int i = 0; i < classes.Length; i++)
            {
                var c = classes[i];
                float y = -282f - i * 32f;
                Button(ShipDatabase.ShortTag(c) + "  " + c.ToString().ToUpper(), left,
                    new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(12f, y - 28f), new Vector2(-12f, y),
                    () => { if (ScenarioEditor.I != null) { ScenarioEditor.I.PaletteClass = c; ScenarioEditor.I.Tool = EditorTool.PlaceShip; } },
                    null, () => ScenarioEditor.I != null && ScenarioEditor.I.PaletteClass == c);
            }

            Label("EdHelp", left, "Left click places or picks.\nDrag to move, drag a rim to resize.\nRight-drag turns a ship.\nDelete removes the selection.",
                11, TextAnchor.UpperLeft, TextDim,
                new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(12f, 84f), new Vector2(-12f, 168f));

            _editorStatus = Label("EdStatus", left, "", 12, TextAnchor.LowerLeft, new Color(1f, 0.85f, 0.4f),
                new Vector2(0f, 0f), new Vector2(1f, 0f), new Vector2(12f, 12f), new Vector2(-12f, 76f));

            // ---- right settings inspector ------------------------------------
            var right = Panel("EdSettings", p, new Vector2(1f, 0f), new Vector2(1f, 1f),
                new Vector2(-236f, 0f), new Vector2(0f, -46f), new Color(0.04f, 0.08f, 0.13f, 0.94f));

            Label("SetHdr", right, "MATCH SETTINGS", 13, TextAnchor.MiddleLeft, Accent,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(14f, -30f), new Vector2(-10f, -10f), FontStyle.Bold);

            Label("WxHdr2", right, "WEATHER", 11, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(14f, -56f), new Vector2(-10f, -38f));
            var wx = new[] { WeatherType.Clear, WeatherType.Fog, WeatherType.Rain, WeatherType.Storm };
            for (int i = 0; i < wx.Length; i++)
            {
                var w = wx[i];
                float x = 12f + (i % 2) * 106f;
                float y = -62f - (i / 2) * 30f;
                Button(w.ToString().ToUpper(), right, new Vector2(0f, 1f), new Vector2(0f, 1f),
                    new Vector2(x, y - 26f), new Vector2(x + 98f, y),
                    () => { if (ScenarioEditor.I != null) ScenarioEditor.I.Current.weather = w; },
                    null, () => ScenarioEditor.I != null && ScenarioEditor.I.Current.weather == w);
            }

            Label("MapHdr2", right, "BATTLEFIELD", 11, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(14f, -140f), new Vector2(-10f, -122f));
            var presets = new[] { MapPreset.OceanArchipelago, MapPreset.OpenSea, MapPreset.StraitClash };
            var presetNames = new[] { "ARCHIPELAGO", "OPEN SEA", "STRAIT CLASH" };
            for (int i = 0; i < presets.Length; i++)
            {
                var mp = presets[i];
                float y = -146f - i * 30f;
                Button(presetNames[i], right, new Vector2(0f, 1f), new Vector2(1f, 1f),
                    new Vector2(12f, y - 26f), new Vector2(-12f, y),
                    () => { if (ScenarioEditor.I != null) ScenarioEditor.I.Current.preset = mp; },
                    null, () => ScenarioEditor.I != null && ScenarioEditor.I.Current.preset == mp);
            }

            Label("AiHdr", right, "ENEMY SKILL", 11, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(14f, -240f), new Vector2(-10f, -222f));
            var diffs = new[] { AIDifficulty.Recruit, AIDifficulty.Veteran, AIDifficulty.Elite };
            for (int i = 0; i < diffs.Length; i++)
            {
                var d = diffs[i];
                float x = 12f + i * 71f;
                Button(d.ToString().ToUpper().Substring(0, 3), right, new Vector2(0f, 1f), new Vector2(0f, 1f),
                    new Vector2(x, -272f), new Vector2(x + 63f, -246f),
                    () => { if (ScenarioEditor.I != null) ScenarioEditor.I.Current.aiDifficulty = d; },
                    null, () => ScenarioEditor.I != null && ScenarioEditor.I.Current.aiDifficulty == d);
            }

            Label("TimeHdr", right, "TIME LIMIT", 11, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(14f, -300f), new Vector2(-10f, -282f));
            Slider("EdTime", right, new Vector2(12f, -332f), new Vector2(224f, -306f), 5, 40,
                () => ScenarioEditor.I != null ? Mathf.RoundToInt(ScenarioEditor.I.Current.timeLimit / 60f) : 20,
                v => { if (ScenarioEditor.I != null) ScenarioEditor.I.Current.timeLimit = v * 60f; },
                v => v + " MIN");

            Label("SeedHdr", right, "MAP SEED", 11, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(14f, -360f), new Vector2(-10f, -342f));
            Slider("EdSeed", right, new Vector2(12f, -392f), new Vector2(224f, -366f), 1, 9999,
                () => ScenarioEditor.I != null ? ScenarioEditor.I.Current.seed : 1,
                v => { if (ScenarioEditor.I != null) ScenarioEditor.I.Current.seed = v; },
                v => "SEED " + v);

            Button("FOG OF WAR", right, new Vector2(0f, 1f), new Vector2(1f, 1f),
                new Vector2(12f, -430f), new Vector2(-12f, -400f),
                () => { if (ScenarioEditor.I != null) ScenarioEditor.I.Current.fogOfWar = !ScenarioEditor.I.Current.fogOfWar; },
                null, () => ScenarioEditor.I != null && ScenarioEditor.I.Current.fogOfWar);

            Label("SavedHdr", right, "SAVED SCENARIOS", 11, TextAnchor.MiddleLeft, TextDim,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(14f, -462f), new Vector2(-10f, -444f));
            _editorLoadList = Label("SavedList", right, "", 11, TextAnchor.UpperLeft, TextMain,
                new Vector2(0f, 0f), new Vector2(1f, 1f), new Vector2(14f, 12f), new Vector2(-12f, -468f));

            _editorPanel.SetActive(false);
        }

        void SaveScenario()
        {
            var ed = ScenarioEditor.I;
            if (ed == null) return;
            string err;
            if (ed.Current.Save(out err))
                GameEvents.RaiseMessage("Saved scenario '" + ed.Current.scenarioName + "'", Team.Neutral);
            else
                GameEvents.RaiseMessage("Could not save: " + err, Team.Neutral);
        }

        /// <summary>Loads the most recently saved scenario. The list on the right shows what is there.</summary>
        void LoadScenario()
        {
            var ed = ScenarioEditor.I;
            if (ed == null) return;
            var names = Scenario.ListSaved();
            if (names.Count == 0) { GameEvents.RaiseMessage("No saved scenarios yet", Team.Neutral); return; }

            // cycle through the saved files so repeated clicks walk the list
            int idx = names.IndexOf(Scenario.SafeFileName(ed.Current.scenarioName));
            string pick = names[(idx + 1 + names.Count) % names.Count];

            string err;
            var loaded = Scenario.Load(pick, out err);
            if (loaded == null) { GameEvents.RaiseMessage("Could not load: " + err, Team.Neutral); return; }
            ed.Current = loaded;
            GameEvents.RaiseMessage("Loaded scenario '" + loaded.scenarioName + "'", Team.Neutral);
        }

        void RefreshEditor()
        {
            var ed = ScenarioEditor.I;
            if (ed == null) return;
            var sc = ed.Current;
            _editorCounts.text = sc.scenarioName + "   |   " + sc.CountOf(Team.Player) + " friendly, " +
                                 sc.CountOf(Team.Enemy) + " hostile   |   " + sc.zones.Count + " zones   |   " +
                                 sc.islands.Count + " islands";
            _editorStatus.text = ed.Status;

            var names = Scenario.ListSaved();
            if (names.Count == 0) _editorLoadList.text = "(none saved yet)";
            else
            {
                var sb = new System.Text.StringBuilder();
                for (int i = 0; i < names.Count && i < 12; i++) sb.AppendLine("- " + names[i]);
                _editorLoadList.text = sb.ToString();
            }
        }

        void BuildDeployPanel()
        {
            var p = Panel("Deploy", _canvas.transform, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f),
                new Vector2(-360f, -190f), new Vector2(360f, 190f), new Color(0.04f, 0.08f, 0.13f, 0.95f));
            _deployPanel = p.gameObject;

            Label("DeployTitle", p, "FLEET DEPLOYMENT", 26, TextAnchor.UpperCenter, Accent,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -50f), new Vector2(0f, -12f), FontStyle.Bold);

            _deployText = Label("DeployBody", p, "", 15, TextAnchor.UpperLeft, TextMain,
                new Vector2(0f, 0f), new Vector2(1f, 1f), new Vector2(26f, 112f), new Vector2(-26f, -58f));
            _deployText.horizontalOverflow = HorizontalWrapMode.Wrap;

            Button("LINE AHEAD", p, new Vector2(0f, 0f), new Vector2(0f, 0f), new Vector2(26f, 64f), new Vector2(166f, 100f),
                () => GameManager.I.SetDeployFormation(FormationType.LineAhead), null,
                () => GameManager.I != null && GameManager.I.DeployFormation == FormationType.LineAhead);
            Button("LINE ABREAST", p, new Vector2(0f, 0f), new Vector2(0f, 0f), new Vector2(176f, 64f), new Vector2(316f, 100f),
                () => GameManager.I.SetDeployFormation(FormationType.LineAbreast), null,
                () => GameManager.I != null && GameManager.I.DeployFormation == FormationType.LineAbreast);
            Button("WEDGE", p, new Vector2(0f, 0f), new Vector2(0f, 0f), new Vector2(326f, 64f), new Vector2(456f, 100f),
                () => GameManager.I.SetDeployFormation(FormationType.Wedge), null,
                () => GameManager.I != null && GameManager.I.DeployFormation == FormationType.Wedge);
            Button("SCREEN", p, new Vector2(0f, 0f), new Vector2(0f, 0f), new Vector2(466f, 64f), new Vector2(606f, 100f),
                () => GameManager.I.SetDeployFormation(FormationType.DefensiveScreen), null,
                () => GameManager.I != null && GameManager.I.DeployFormation == FormationType.DefensiveScreen);

            Button("START BATTLE", p, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f), new Vector2(-120f, 16f), new Vector2(120f, 56f),
                () => GameManager.I.StartBattle(), null, null, new Color(0.15f, 0.42f, 0.3f, 0.95f));
        }

        void BuildEndPanel()
        {
            var p = Panel("End", _canvas.transform, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f),
                new Vector2(-320f, -190f), new Vector2(320f, 190f), new Color(0.04f, 0.08f, 0.13f, 0.96f));
            _endPanel = p.gameObject;

            _endTitle = Label("EndTitle", p, "VICTORY", 34, TextAnchor.UpperCenter, Accent,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -60f), new Vector2(0f, -14f), FontStyle.Bold);
            _endBody = Label("EndBody", p, "", 16, TextAnchor.UpperCenter, TextMain,
                new Vector2(0f, 0f), new Vector2(1f, 1f), new Vector2(24f, 70f), new Vector2(-24f, -70f));

            Button("NEW BATTLE", p, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f), new Vector2(-300f, 18f), new Vector2(-104f, 56f),
                () => GameManager.I.RestartSameMode(), null, null, new Color(0.15f, 0.42f, 0.3f, 0.95f));
            Button("CHANGE FLEET", p, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f), new Vector2(-98f, 18f), new Vector2(98f, 56f),
                () => GameManager.I.ReturnToMenu(), null, null, new Color(0.15f, 0.30f, 0.45f, 0.95f));
            Button("NEXT MODE", p, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f), new Vector2(104f, 18f), new Vector2(300f, 56f),
                () => GameManager.I.NextMode(), null, null, new Color(0.15f, 0.30f, 0.45f, 0.95f));

            _endPanel.SetActive(false);
        }

        void BuildHelpPanel()
        {
            var p = Panel("Help", _canvas.transform, new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.5f),
                new Vector2(-430f, -290f), new Vector2(430f, 290f), new Color(0.03f, 0.06f, 0.1f, 0.96f));
            _helpPanel = p.gameObject;

            Label("HelpTitle", p, "COMMAND REFERENCE", 22, TextAnchor.UpperCenter, Accent,
                new Vector2(0f, 1f), new Vector2(1f, 1f), new Vector2(0f, -44f), new Vector2(0f, -12f), FontStyle.Bold);

            string help =
                "TAB  switch between DIRECT CONTROL and FLEET COMMAND\n\n" +
                "DIRECT CONTROL\n" +
                "  W / S  engine telegraph        A / D  rudder        Space  all stop\n" +
                "  Mouse  train the guns          Left click  fire main battery\n" +
                "  Right click  torpedo spread    1-4  consumables    X  dive/surface (SS)\n\n" +
                "FLEET COMMAND\n" +
                "  Left click / drag  select      Shift+click  add     Double click  whole class\n" +
                "  1 / 2 / 3  select your LEFT, CENTRE and RIGHT squadrons (bound at deployment)\n" +
                "  Ctrl+1..9 rebind a group       Ctrl+A  select the whole fleet\n" +
                "  Right click  move / attack     Shift+right click  queue waypoint\n" +
                "  C attack move   V patrol   B escort   T focus fire   Space stop   H hold\n" +
                "  R astern   G retreat   Y return to port   Q smoke   E damage control\n" +
                "  F5-F9 formations   F10 break formation\n\n" +
                "CAMERA & TIME\n" +
                "  WASD / edge scroll (fleet mode)   Middle drag   Wheel zoom   F follow\n" +
                "  + / -  time compression 1x 2x 4x 8x     P pause     ` fleet overview\n\n" +
                "SYSTEM\n" +
                "  F1 debug draw   F2 reveal map   F3 nav grid   F4 this screen";

            Label("HelpBody", p, help, 15, TextAnchor.UpperLeft, TextMain,
                new Vector2(0f, 0f), new Vector2(1f, 1f), new Vector2(34f, 60f), new Vector2(-34f, -52f));

            Button("CLOSE (F4)", p, new Vector2(0.5f, 0f), new Vector2(0.5f, 0f), new Vector2(-90f, 16f), new Vector2(90f, 50f),
                () => ToggleHelp());

            _helpPanel.SetActive(false);
        }

        public void ToggleHelp()
        {
            HelpVisible = !HelpVisible;
            _helpPanel.SetActive(HelpVisible);
        }

        // ------------------------------------------------------------------ update

        void Update()
        {
            HandleSliders();
            HandleButtons();
            HandleMinimapInput();

            if (InputHub.KeyDown(Key.F4)) ToggleHelp();

            RefreshTopBar();
            RefreshBanner();
            RefreshShipPanel();
            RefreshFleetPanel();
            RefreshActionBar();
            RefreshMessages();
            RefreshWarning();
            RefreshAIDebug();
            RefreshPhasePanels();
        }

        void HandleSliders()
        {
            Vector2 mouse = InputHub.MousePosition;
            for (int i = 0; i < _sliders.Count; i++)
            {
                var s = _sliders[i];
                if (s.track == null || !s.track.gameObject.activeInHierarchy) { s.dragging = false; continue; }

                bool over = RectTransformUtility.RectangleContainsScreenPoint(s.track, mouse, null);
                if (InputHub.LeftDown && over) s.dragging = true;
                if (!InputHub.LeftHeld) s.dragging = false;

                if (s.dragging &&
                    RectTransformUtility.ScreenPointToLocalPointInRectangle(s.track, mouse, null, out Vector2 local))
                {
                    var r = s.track.rect;
                    float u = Mathf.Clamp01(Mathf.InverseLerp(r.xMin, r.xMax, local.x));
                    s.set(Mathf.RoundToInt(Mathf.Lerp(s.min, s.max, u)));
                }

                int v = s.get();
                float frac = Mathf.InverseLerp(s.min, s.max, v);
                s.fill.rectTransform.localScale = new Vector3(Mathf.Clamp01(frac), 1f, 1f);
                s.fill.color = s.dragging || over
                    ? new Color(0.32f, 0.72f, 0.95f, 0.95f) : new Color(0.2f, 0.55f, 0.75f, 0.95f);
                s.readout.text = s.format != null ? s.format(v) : v + " SHIPS";
            }
        }

        void HandleButtons()
        {
            Vector2 mouse = InputHub.MousePosition;
            bool click = InputHub.LeftDown;

            for (int i = 0; i < _buttons.Count; i++)
            {
                var b = _buttons[i];
                if (b.rect == null || !b.rect.gameObject.activeInHierarchy) continue;
                bool en = b.enabled == null || b.enabled();
                bool hover = RectTransformUtility.RectangleContainsScreenPoint(b.rect, mouse, null);
                bool hi = b.highlighted != null && b.highlighted();

                Color target = b.color;
                if (hi) target = new Color(0.22f, 0.45f, 0.6f, 0.95f);
                if (!en) target = new Color(b.color.r * 0.5f, b.color.g * 0.5f, b.color.b * 0.5f, 0.6f);
                else if (hover) target = Color.Lerp(target, new Color(0.4f, 0.7f, 0.9f, 1f), 0.45f);
                b.bg.color = target;
                b.label.color = en ? TextMain : new Color(0.5f, 0.55f, 0.6f);

                if (click && hover && en)
                {
                    AudioManager.PlayUI(SoundId.OrderConfirm, 0.5f);
                    b.onClick?.Invoke();
                }
            }
        }

        bool _minimapDragging;

        void HandleMinimapInput()
        {
            if (Minimap.I == null) return;
            Vector2 mouse = InputHub.MousePosition;

            if ((InputHub.LeftDown || (InputHub.LeftHeld && _minimapDragging)) && Minimap.I.ScreenToWorld(mouse, out Vector2 world))
            {
                _minimapDragging = true;
                if (ControlModeManager.I == null || !ControlModeManager.I.IsDirect) RTSCamera.I.FocusOn(world);
            }
            if (!InputHub.LeftHeld) _minimapDragging = false;

            if (InputHub.RightDown && Minimap.I.ScreenToWorld(mouse, out Vector2 target))
            {
                if (ControlModeManager.I == null || !ControlModeManager.I.IsDirect)
                    CommandSystem.I.Move(target, InputHub.Shift);
            }
        }

        void RefreshTopBar()
        {
            var gm = GameManager.I;
            if (gm == null) return;

            _modeText.text = gm.ModeName.ToUpper();
            _objectiveText.text = gm.ObjectiveText;

            float t = gm.TimeRemaining;
            _timerText.text = string.Format("{0:00}:{1:00}", Mathf.FloorToInt(t / 60f), Mathf.FloorToInt(t % 60f));
            _timerText.color = t <= 60f ? new Color(1f, 0.45f, 0.4f) : TextMain;

            _alliedScore.text = Mathf.RoundToInt(gm.PlayerScore).ToString();
            _enemyScore.text = Mathf.RoundToInt(gm.EnemyScore).ToString();

            int pAlive = ShipRegistry.AliveCount(Team.Player);
            int eAlive = ShipRegistry.AliveCount(Team.Enemy);
            _fleetCountText.text = "SHIPS  " + pAlive + " v " + eAlive;

            _weatherText.text = WeatherSystem.I != null ? WeatherSystem.I.Describe().ToUpper() : "CLEAR";
            _speedText.text = "SPEED  x" + gm.GameSpeed.ToString("0.#");

            var map = WorldMap.I;
            if (map != null && map.Zones.Count > 0)
            {
                var sb = new System.Text.StringBuilder();
                for (int i = 0; i < map.Zones.Count; i++)
                {
                    var z = map.Zones[i];
                    if (z == null) continue;
                    string col = z.Contested ? "#ffd23f" : z.Owner == Team.Player ? "#59c7ff" : z.Owner == Team.Enemy ? "#ff6b61" : "#b9c4cc";
                    int pct = Mathf.RoundToInt(z.CaptureFraction * 100f);
                    sb.Append("<color=").Append(col).Append(">").Append(z.zoneName);
                    if (z.Contested) sb.Append("!");
                    else if (pct > 0 && pct < 100) sb.Append(" ").Append(pct).Append("%");
                    sb.Append("</color>   ");
                }
                _zoneText.supportRichText = true;
                _zoneText.text = sb.ToString();
            }
        }

        void RefreshBanner()
        {
            var cm = ControlModeManager.I;
            var gm = GameManager.I;
            bool show = gm != null && gm.Phase == GamePhase.Battle;
            if (_bannerPanel.gameObject.activeSelf != show) _bannerPanel.gameObject.SetActive(show);
            if (!show) return;

            if (cm != null && cm.IsDirect)
            {
                var s = cm.Controlled;
                _bannerText.text = "DIRECT CONTROL  -  " + s.shipName + " (" + s.ClassTag + ")   [Tab] fleet command";
                _bannerText.color = new Color(0.55f, 1f, 0.75f);
            }
            else
            {
                _bannerText.text = "FLEET COMMAND   [Tab] take the helm";
                _bannerText.color = Accent;
            }
        }

        void RefreshShipPanel()
        {
            var sel = SelectionManager.I;
            var cm = ControlModeManager.I;
            Ship s = cm != null && cm.IsDirect ? cm.Controlled : (sel != null ? sel.Primary : null);

            if (s == null)
            {
                _shipTitle.text = "NO SHIP SELECTED";
                _shipVitals.text = "";
                _shipSensors.text = "";
                _shipTargetText.text = "";
                SetBar(_shipHealthFill, 0f); SetBar(_shipFuelFill, 0f); SetBar(_shipAmmoFill, 0f);
                SetBar(_shipTorpFill, 0f); SetBar(_shipDcFill, 0f);
                for (int i = 0; i < 7; i++) SetBar(_systemFills[i], 0f);
                return;
            }

            int extra = sel != null ? sel.Selected.Count - 1 : 0;
            _shipTitle.text = s.shipName + "  [" + s.ClassTag + "]" + (extra > 0 && (cm == null || !cm.IsDirect) ? "   +" + extra + " more" : "");

            SetBar(_shipHealthFill, s.HealthFraction);
            _shipHealthFill.color = s.HealthFraction > 0.6f ? new Color(0.35f, 0.9f, 0.45f)
                                  : s.HealthFraction > 0.3f ? new Color(1f, 0.8f, 0.25f) : new Color(1f, 0.35f, 0.3f);

            string throttle = s.Movement.Throttle > 0.05f ? "AHEAD " + Mathf.RoundToInt(s.Movement.Throttle * 100f) + "%"
                            : s.Movement.Throttle < -0.05f ? "ASTERN " + Mathf.RoundToInt(-s.Movement.Throttle * 100f) + "%"
                            : "STOP";

            _shipVitals.text =
                "HP  " + Mathf.RoundToInt(s.Damage.Health) + " / " + Mathf.RoundToInt(s.Damage.MaxHealth) + "\n" +
                "SPD " + s.SpeedKnots.ToString("F1") + " kn   " + throttle + "\n" +
                "HDG " + Mathf.RoundToInt(s.Heading).ToString("000") + "°   RUD " + (s.Movement.Rudder > 0.1f ? "STBD" : s.Movement.Rudder < -0.1f ? "PORT" : "MID") + "\n" +
                "STATE " + (s.AI != null ? s.AI.State.ToString().ToUpper() : "-") + "\n" +
                "CONDITION " + s.Damage.StatusLine();

            string det = s.Detection.SpottedByEnemy ? "SPOTTED" : "UNDETECTED";
            string depth = s.Submarine != null ? "\nDEPTH " + s.Submarine.DepthLabel() : "";
            string active = s.Abilities != null ? s.Abilities.StatusLine() : "";

            _shipSensors.text =
                "DETECTION " + det + "\n" +
                "SIGNATURE " + Mathf.RoundToInt(s.Detectability) + "\n" +
                "SPOT " + Mathf.RoundToInt(s.Detection.EffectiveSpotRange) +
                (s.Stats.sonarRange > 0f ? "\nSONAR " + Mathf.RoundToInt(s.Detection.EffectiveSonarRange) : "") +
                depth + (string.IsNullOrEmpty(active) ? "" : "\n" + active);

            _shipTargetText.text = s.CurrentTarget != null && !s.CurrentTarget.IsDead
                ? "ENGAGING " + s.CurrentTarget.shipName + "  (" + Mathf.RoundToInt(s.DistanceTo(s.CurrentTarget)) + ")"
                : (s.Weapons.HoldFire ? "WEAPONS TIGHT" : "NO TARGET");

            SetBar(_shipFuelFill, s.Resources.FuelFraction);
            SetBar(_shipAmmoFill, s.Resources.MainAmmoMax > 0 ? s.Resources.MainAmmo / (float)s.Resources.MainAmmoMax : 0f);
            SetBar(_shipTorpFill, s.Resources.TorpedoAmmoMax > 0 ? s.Resources.TorpedoAmmo / (float)s.Resources.TorpedoAmmoMax : 0f);
            SetBar(_shipDcFill, s.Damage.DamageControlReady ? 1f : 1f - Mathf.Clamp01(s.Damage.DamageControlCooldown / Mathf.Max(1f, s.Stats.damageControlCooldown)));
            _shipDcFill.color = s.Damage.DamageControlReady ? new Color(0.5f, 1f, 0.7f) : new Color(0.6f, 0.7f, 0.9f);

            for (int i = 0; i < 7; i++)
            {
                float v = s.Damage.SystemIntegrity((ShipSystem)i);
                SetBar(_systemFills[i], v);
                _systemFills[i].color = v > 0.66f ? new Color(0.4f, 0.85f, 0.6f)
                                     : v > 0.33f ? new Color(0.95f, 0.8f, 0.35f) : new Color(1f, 0.4f, 0.35f);
            }
        }

        static void SetBar(Image img, float fraction)
        {
            if (img == null) return;
            img.rectTransform.localScale = new Vector3(Mathf.Clamp01(fraction), 1f, 1f);
        }

        float _fleetRefreshTimer;

        void RefreshFleetPanel()
        {
            _fleetRefreshTimer -= Time.unscaledDeltaTime;
            var ships = ShipRegistry.OfTeam(Team.Player);
            if (_fleet.Count != ships.Count || _fleetRefreshTimer <= 0f)
            {
                if (_fleet.Count != ships.Count) RebuildFleetEntries();
                _fleetRefreshTimer = 0.5f;
            }

            Vector2 mouse = InputHub.MousePosition;
            bool click = InputHub.LeftDown;
            var cm = ControlModeManager.I;

            for (int i = 0; i < _fleet.Count; i++)
            {
                var e = _fleet[i];
                if (e.ship == null || e.ship.IsDead)
                {
                    e.bg.color = new Color(0.2f, 0.06f, 0.06f, 0.7f);
                    e.label.color = new Color(0.5f, 0.4f, 0.4f);
                    SetBar(e.healthFill, 0f);
                    continue;
                }

                SetBar(e.healthFill, e.ship.HealthFraction);
                e.healthFill.color = e.ship.HealthFraction > 0.6f ? new Color(0.35f, 0.9f, 0.45f)
                                   : e.ship.HealthFraction > 0.3f ? new Color(1f, 0.8f, 0.25f) : new Color(1f, 0.35f, 0.3f);

                bool hover = RectTransformUtility.RectangleContainsScreenPoint(e.rect, mouse, null);
                bool conning = cm != null && cm.IsDirect && cm.Controlled == e.ship;
                e.bg.color = conning ? new Color(0.14f, 0.38f, 0.3f, 0.95f)
                           : e.ship.Selected ? new Color(0.16f, 0.32f, 0.42f, 0.95f)
                           : hover ? new Color(0.14f, 0.22f, 0.3f, 0.95f) : new Color(0.09f, 0.14f, 0.2f, 0.9f);

                if (e.ship.Damage.FireStacks > 0) e.statusPip.color = new Color(1f, 0.5f, 0.15f);
                else if (e.ship.Damage.FloodingStacks > 0) e.statusPip.color = new Color(0.35f, 0.7f, 1f);
                else if (e.ship.Detection.SpottedByEnemy) e.statusPip.color = new Color(1f, 0.85f, 0.3f);
                else e.statusPip.color = Teams.Color(Team.Player);

                if (click && hover)
                {
                    // in direct control, clicking the roster takes the helm of that ship instead
                    if (cm != null && cm.IsDirect) cm.EnterDirect(e.ship);
                    else if (InputHub.Shift) SelectionManager.I.Toggle(e.ship);
                    else { SelectionManager.I.SelectOnly(e.ship); RTSCamera.I.FocusOn(e.ship.Position); }
                }
            }
        }

        void RefreshActionBar()
        {
            var cm = ControlModeManager.I;
            var gm = GameManager.I;
            bool direct = cm != null && cm.IsDirect && gm != null && gm.Phase == GamePhase.Battle;

            if (_actionBar.gameObject.activeSelf != direct) _actionBar.gameObject.SetActive(direct);
            bool rts = !direct && gm != null && (gm.Phase == GamePhase.Battle || gm.Phase == GamePhase.Deployment);
            if (_commandPanel.gameObject.activeSelf != rts) _commandPanel.gameObject.SetActive(rts);
            if (!direct) return;

            var ship = cm.Controlled;
            var ab = ship.Abilities;

            _ammoLabel.text = ship.Stats.mainBattery != null ? (ship.Weapons.UsingAP ? "AP LOADED" : "HE LOADED") : "";

            for (int i = 0; i < _slots.Length; i++)
            {
                var slot = _slots[i];
                var a = ab != null ? ab.GetSlot(i) : null;
                if (a == null)
                {
                    slot.rect.gameObject.SetActive(false);
                    continue;
                }
                slot.rect.gameObject.SetActive(true);
                slot.key.text = a.hotkey;
                slot.label.text = a.label;

                bool selectedAmmo = (a.id == AbilityId.ShellHE && !ship.Weapons.UsingAP) ||
                                    (a.id == AbilityId.ShellAP && ship.Weapons.UsingAP);
                bool subSurfaced = a.id == AbilityId.Dive && ship.Submarine != null && ship.Submarine.Depth != DepthState.Surface;

                if (a.IsActive) slot.bg.color = new Color(0.16f, 0.45f, 0.32f, 0.95f);
                else if (selectedAmmo || subSurfaced) slot.bg.color = new Color(0.2f, 0.36f, 0.5f, 0.95f);
                else if (!a.Ready && !a.IsToggle) slot.bg.color = new Color(0.14f, 0.14f, 0.18f, 0.95f);
                else slot.bg.color = new Color(0.11f, 0.17f, 0.24f, 0.95f);

                // cooldown sweep
                float fill = a.IsToggle ? 0f : (a.IsActive ? 0f : 1f - a.CooldownFraction);
                slot.cooldownFill.rectTransform.localScale = new Vector3(1f, Mathf.Clamp01(fill), 1f);

                if (a.IsActive) slot.charges.text = Mathf.CeilToInt(a.activeLeft) + "s";
                else if (a.cooldownLeft > 0f) slot.charges.text = Mathf.CeilToInt(a.cooldownLeft) + "s";
                else if (a.maxCharges > 0) slot.charges.text = a.chargesLeft + "/" + a.maxCharges;
                else if (a.id == AbilityId.Torpedoes || a.id == AbilityId.HomingTorpedoes)
                    slot.charges.text = ship.Weapons.TorpedoesReady ? "READY" : Mathf.CeilToInt(ship.Weapons.TorpedoReload) + "s";
                else slot.charges.text = "";
            }

            // engine telegraph
            float th = ship.Movement.Throttle;
            SetBar(_throttleFill, Mathf.Abs(th));
            _throttleFill.color = th >= 0f ? new Color(0.45f, 0.9f, 0.6f) : new Color(0.95f, 0.65f, 0.35f);
            _throttleLabel.text = (th > 0.05f ? "AHEAD " : th < -0.05f ? "ASTERN " : "STOP ") +
                                  (Mathf.Abs(th) > 0.05f ? Mathf.RoundToInt(Mathf.Abs(th) * 100f) + "%" : "") +
                                  "   " + ship.SpeedKnots.ToString("F0") + " kn";

            if (ship.Submarine != null)
            {
                _subLabel.text = ship.Submarine.DepthLabel() + "  BATT";
                SetBar(_subBatteryFill, ship.Submarine.BatteryFraction);
                _subBatteryFill.color = ship.Submarine.BatteryFraction > 0.3f
                    ? new Color(0.4f, 0.9f, 1f) : new Color(1f, 0.5f, 0.4f);
                _subBatteryFill.transform.parent.gameObject.SetActive(true);
            }
            else
            {
                _subLabel.text = "";
                _subBatteryFill.transform.parent.gameObject.SetActive(false);
            }
        }

        void OnMessage(string msg, Team team)
        {
            _messages.Add((msg, Time.unscaledTime, team));
            if (_messages.Count > 9) _messages.RemoveAt(0);
        }

        void RefreshMessages()
        {
            for (int i = _messages.Count - 1; i >= 0; i--)
                if (Time.unscaledTime - _messages[i].time > 14f) _messages.RemoveAt(i);

            var sb = new System.Text.StringBuilder();
            for (int i = 0; i < _messages.Count; i++)
            {
                var m = _messages[i];
                string color = m.team == Team.Player ? "#8fd6ff" : m.team == Team.Enemy ? "#ff8b7f" : "#cfd8dd";
                sb.Append("<color=").Append(color).Append(">").Append(m.text).Append("</color>\n");
            }
            _messageText.supportRichText = true;
            _messageText.text = sb.ToString();
        }

        /// <summary>Shows what each fleet commander currently believes, so the AI can be debugged.</summary>
        void RefreshAIDebug()
        {
            if (_aiDebugText == null) return;
            var panel = _aiDebugText.transform.parent.gameObject;
            bool show = DebugOverlay.Enabled;
            if (panel.activeSelf != show) panel.SetActive(show);
            if (!show) return;

            var enemy = BattleAssessment.For(Team.Enemy);
            var mine = BattleAssessment.For(Team.Player);
            var sb = new System.Text.StringBuilder();

            if (enemy != null)
            {
                sb.Append("ENEMY: ").Append(enemy.Describe()).Append('\n');
                sb.Append("  proj ").Append(Mathf.RoundToInt(enemy.ProjectedTheirs)).Append(" us / ")
                  .Append(Mathf.RoundToInt(enemy.ProjectedMine)).Append(" them")
                  .Append(enemy.ProjectedWin ? "  (they win)" : "  (we win)").Append('\n');
                sb.Append("  strength ").Append(enemy.StrengthRatio.ToString("F2"))
                  .Append("   contacts ").Append(enemy.Contacts.Count).Append('\n');
            }
            if (mine != null)
                sb.Append("OURS: strength ").Append(mine.StrengthRatio.ToString("F2"))
                  .Append("   contacts ").Append(mine.Contacts.Count);

            _aiDebugText.text = sb.ToString();
        }

        public void ShowWarning(string text, float duration = 2.5f)
        {
            _warningText.text = text;
            _warningTimer = duration;
            _warningText.transform.parent.gameObject.SetActive(true);
        }

        void RefreshWarning()
        {
            if (_warningTimer <= 0f) return;
            _warningTimer -= Time.unscaledDeltaTime;
            float pulse = 0.6f + Mathf.Sin(Time.unscaledTime * 10f) * 0.4f;
            _warningText.color = new Color(1f, 0.85f, 0.8f, pulse);
            if (_warningTimer <= 0f) _warningText.transform.parent.gameObject.SetActive(false);
        }

        void RefreshPhasePanels()
        {
            var gm = GameManager.I;
            if (gm == null) return;

            bool menu = gm.Phase == GamePhase.Menu;
            if (_menuPanel.activeSelf != menu) _menuPanel.SetActive(menu);
            if (menu) RefreshMenu();

            bool editing = gm.Phase == GamePhase.Editor;
            if (_editorPanel.activeSelf != editing) _editorPanel.SetActive(editing);
            if (editing) RefreshEditor();

            bool deploy = gm.Phase == GamePhase.Deployment;
            if (_deployPanel.activeSelf != deploy) _deployPanel.SetActive(deploy);
            if (deploy) _deployText.text = gm.DeploymentBriefing();

            bool over = gm.Phase == GamePhase.Victory || gm.Phase == GamePhase.Defeat;
            if (_endPanel.activeSelf != over) _endPanel.SetActive(over);
            if (over)
            {
                _endTitle.text = gm.Phase == GamePhase.Victory ? "VICTORY" : "DEFEAT";
                _endTitle.color = gm.Phase == GamePhase.Victory ? new Color(0.5f, 1f, 0.7f) : new Color(1f, 0.5f, 0.45f);
                _endBody.text = gm.ResultSummary;
            }

            // world HUD is meaningless on the menu screen and in the editor
            bool inWorld = !menu && !editing;
            if (_shipPanel.activeSelf != inWorld) _shipPanel.SetActive(inWorld);
            if (_fleetPanel.gameObject.activeSelf != (inWorld && !deploy)) _fleetPanel.gameObject.SetActive(inWorld && !deploy);
        }

        void RefreshMenu()
        {
            bool custom = _menuSetup.compositionMode == FleetCompositionMode.Custom;
            var classes = new[] { ShipClassType.Battleship, ShipClassType.Cruiser,
                                  ShipClassType.Destroyer, ShipClassType.Submarine };

            // in the automatic modes the per-class rows show what the generator will actually build
            var preview = custom ? null : FleetSetup.BalancedFor(_menuSetup.playerShipCount);

            for (int i = 0; i < classes.Length; i++)
            {
                var t = _menuPanel.transform.Find("MenuCount_" + classes[i]);
                if (t == null) continue;
                var txt = t.GetComponent<Text>();
                if (txt == null) continue;

                int n;
                if (custom) n = _menuSetup.CountOf(classes[i]);
                else { n = 0; for (int k = 0; k < preview.Count; k++) if (preview[k] == classes[i]) n++; }

                txt.text = n.ToString();
                txt.color = custom ? TextMain : TextDim;
            }

            int player = custom ? _menuSetup.CustomTotal : _menuSetup.playerShipCount;
            int enemy = _menuSetup.compositionMode == FleetCompositionMode.Mirror
                ? _menuSetup.enemyShipCount : _menuSetup.enemyShipCount;

            string mode = _menuSetup.compositionMode == FleetCompositionMode.Mirror ? "mirroring your composition"
                        : _menuSetup.compositionMode == FleetCompositionMode.Custom ? "your manual slots vs a balanced enemy"
                        : "balanced on both sides";

            if (_mapSummaryText != null)
                _mapSummaryText.text = _menuMap.LayoutName + "  -  " +
                    Mathf.RoundToInt(_menuMap.captureRadius * 10f) + " m caps";

            if (_policyStatusText != null)
                _policyStatusText.text = RL.RLPolicyDriver.I != null ? RL.RLPolicyDriver.I.Status : "no trained policy";

            bool ok = MenuIsValid();
            bool spectating = _menuSetup.playerController == ShipController.Learned;
            _menuTotalText.text = ok
                ? player + " v " + enemy + "   -   " + mode +
                  (spectating ? "   -   your fleet is flown by the trained policy" : "") +
                  (_menuSetup.startAsCaptain ? "   -   you take the helm of a " + _menuSetup.controlClass.ToString().ToUpper() : "")
                : (PolicyMissing() ? "No trained policy to fly a Learned fleet yet" : "Set at least one ship per side");
            _menuTotalText.color = ok ? new Color(0.5f, 1f, 0.7f) : new Color(1f, 0.75f, 0.4f);
        }

        // ------------------------------------------------------------------ helpers

        public static bool IsPointerOverUI(Vector2 screenPos)
        {
            for (int i = 0; i < _blockers.Count; i++)
            {
                var r = _blockers[i];
                if (r == null || !r.gameObject.activeInHierarchy) continue;
                if (RectTransformUtility.RectangleContainsScreenPoint(r, screenPos, null)) return true;
            }
            return false;
        }
    }
}
