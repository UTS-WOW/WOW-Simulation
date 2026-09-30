using UnityEngine;
using UnityEngine.InputSystem;

namespace Naval
{
    /// <summary>
    /// Developer visualisation: routes, sensor ranges, weapon envelopes, AI targets, velocity vectors,
    /// collision radii, torpedo tracks and the navigation grid.
    /// F1 debug draw, F2 reveal the map, F3 navigation grid, F4 help.
    /// </summary>
    public class DebugOverlay : MonoBehaviour
    {
        public static DebugOverlay I { get; private set; }

        public static bool Enabled { get; private set; }

        /// <summary>Lets the dev view switch the AI and objective layer on alongside its own drawing.</summary>
        public static void SetEnabled(bool on) => Enabled = on;
        public static bool ShowAll { get; private set; }
        public static bool ShowGrid { get; private set; }

        public static DebugOverlay Create(Transform parent)
        {
            var go = new GameObject("DebugOverlay");
            go.transform.SetParent(parent, false);
            var d = go.AddComponent<DebugOverlay>();
            I = d;
            return d;
        }

        float PS => RTSCamera.I != null ? RTSCamera.I.PixelScale : 0.15f;

        /// <summary>Show every ship regardless of fog of war (F2). Also used by replay rendering.</summary>
        public static void SetReveal(bool on)
        {
            ShowAll = on;
            if (FogOfWarRenderer.I != null) FogOfWarRenderer.I.Enabled = !on;
        }

        /// <summary>Turn the F1 debug drawing on or off from code.</summary>
        public static void SetDebugDraw(bool on) => Enabled = on;

        void Update()
        {
            if (InputHub.KeyDown(Key.F1))
            {
                Enabled = !Enabled;
                GameEvents.RaiseMessage("Debug draw " + (Enabled ? "ON" : "OFF"), Team.Neutral);
            }
            if (InputHub.KeyDown(Key.F2))
            {
                SetReveal(!ShowAll);
                GameEvents.RaiseMessage("Fog of war " + (ShowAll ? "DISABLED" : "ENABLED"), Team.Neutral);
            }
            if (InputHub.KeyDown(Key.F3))
            {
                ShowGrid = !ShowGrid;
                GameEvents.RaiseMessage("Navigation grid " + (ShowGrid ? "ON" : "OFF"), Team.Neutral);
            }

            if (ShowGrid) DrawGrid();
            if (!Enabled) return;

            DrawShips();
            DrawTorpedoes();
        }

        void DrawShips()
        {
            float ps = PS;
            var all = ShipRegistry.All;

            for (int i = 0; i < all.Count; i++)
            {
                var s = all[i];
                if (s == null || s.IsDead) continue;
                bool friendly = s.team == Team.Player;
                Color baseCol = Teams.Color(s.team);

                // collision radius
                LineDrawer.Circle(s.Position, s.Stats.length * 0.35f, 0.8f * ps, new Color(1f, 1f, 1f, 0.25f), 20);

                // detectability and sensor ranges
                LineDrawer.DashedCircle(s.Position, s.Detectability, 0.9f * ps, new Color(1f, 0.9f, 0.4f, 0.22f), 60);
                if (s.Detection != null)
                {
                    LineDrawer.DashedCircle(s.Position, s.Detection.EffectiveSpotRange, 0.9f * ps, new Color(0.5f, 0.9f, 1f, 0.16f), 60);
                    if (s.Stats.sonarRange > 0f)
                        LineDrawer.DashedCircle(s.Position, s.Detection.EffectiveSonarRange, 0.9f * ps, new Color(0.4f, 1f, 0.8f, 0.2f), 40);
                }

                // weapon envelope
                if (s.Stats.mainBattery != null)
                    LineDrawer.DashedCircle(s.Position, s.Stats.mainBattery.range, 0.8f * ps, new Color(1f, 0.45f, 0.3f, 0.14f), 72);

                // velocity vector
                LineDrawer.Arrow(s.Position, s.Position + s.Velocity * 12f, 1.2f * ps, new Color(0.4f, 1f, 0.5f, 0.8f), 6f * ps);

                // rudder / steering intent
                LineDrawer.Line(s.Position, s.Position + NavalMath.HeadingToVector(s.Movement.TargetHeading) * s.Stats.length,
                    1f * ps, new Color(1f, 1f, 0.3f, 0.6f));

                // AI target
                if (s.CurrentTarget != null && !s.CurrentTarget.IsDead)
                    LineDrawer.Line(s.Position, s.CurrentTarget.Position, 0.9f * ps, new Color(1f, 0.3f, 0.3f, 0.4f));

                // route and current waypoint
                var nav = s.Navigation;
                if (nav != null)
                {
                    Vector2 from = s.Position;
                    if (nav.Path != null)
                        for (int p = 0; p < nav.Path.Count; p++)
                        {
                            LineDrawer.Line(from, nav.Path[p], 1f * ps, new Color(0.3f, 0.8f, 1f, 0.55f));
                            LineDrawer.Cross(nav.Path[p], 4f * ps, 1f * ps, new Color(0.3f, 0.8f, 1f, 0.7f));
                            from = nav.Path[p];
                        }
                    if (nav.Waypoints.Count > 0)
                        LineDrawer.Cross(nav.Waypoints[0], 7f * ps, 1.4f * ps, new Color(0.9f, 1f, 0.3f, 0.9f));
                    if (nav.HasDestination)
                        LineDrawer.Circle(nav.CurrentDestination, 8f * ps, 1f * ps, new Color(0.9f, 1f, 0.3f, 0.5f), 16);
                }

                // AI state pip: colour codes the state machine
                Color stateCol = StateColor(s.AI != null ? s.AI.State : AIState.Idle);
                Vector2 pip = s.Position + new Vector2(0f, -s.Stats.length * 0.7f - 6f * ps);
                LineDrawer.FilledRect(pip - new Vector2(3f * ps, 3f * ps), pip + new Vector2(3f * ps, 3f * ps), stateCol);

                // assignment: where the commander wants this ship and what for
                if (s.AI != null && s.AI.HasStation)
                {
                    Color aCol = AssignmentColor(s.AI.Assignment);
                    LineDrawer.Dashed(s.Position, s.AI.StationPoint, 0.9f * ps, new Color(aCol.r, aCol.g, aCol.b, 0.3f), 12f, 10f);
                    LineDrawer.Circle(s.AI.StationPoint, 5f * ps, 1f * ps, new Color(aCol.r, aCol.g, aCol.b, 0.5f), 12);

                    // a cap sitter is tied to its ring
                    if (s.AI.AssignedZone != null &&
                        (s.AI.Assignment == AIAssignment.HoldCap || s.AI.Assignment == AIAssignment.ContestCap ||
                         s.AI.Assignment == AIAssignment.Decap))
                        LineDrawer.Dashed(s.Position, s.AI.AssignedZone.Position, 1.1f * ps,
                            new Color(aCol.r, aCol.g, aCol.b, 0.4f), 16f, 12f);
                }

                // local balance of force: green when this ship is winning its corner of the fight
                if (s.AI != null)
                {
                    float ratio = Mathf.Clamp(s.AI.LocalRatio, 0.2f, 2.5f);
                    Vector2 barAt = s.Position + new Vector2(0f, -s.Stats.length * 0.7f - 12f * ps);
                    float w = 18f * ps;
                    Color rc = ratio >= 1.15f ? new Color(0.4f, 1f, 0.5f, 0.7f)
                             : ratio <= 0.75f ? new Color(1f, 0.4f, 0.35f, 0.7f)
                                              : new Color(1f, 0.9f, 0.4f, 0.7f);
                    LineDrawer.Line(barAt - new Vector2(w, 0f), barAt + new Vector2(w * (ratio / 2.5f * 2f - 1f), 0f), 2f * ps, rc);
                }
            }
        }

        void DrawTorpedoes()
        {
            var ps = ProjectileSystem.I;
            if (ps == null) return;
            float p = PS;
            var torps = ps.Torpedoes;
            for (int i = 0; i < torps.Count; i++)
            {
                var t = torps[i];
                Vector2 dir = NavalMath.HeadingToVector(t.heading);
                LineDrawer.Dashed(t.pos, t.pos + dir * t.rangeLeft, 0.8f * p, new Color(1f, 0.4f, 0.4f, 0.35f), 10f, 8f);
            }
        }

        void DrawGrid()
        {
            var grid = NavGrid.I;
            var cam = RTSCamera.I;
            if (grid == null || cam == null) return;
            if (cam.Zoom > 420f) return;              // too far out to be readable

            Rect view = cam.WorldViewRect();
            grid.WorldToCell(new Vector2(view.xMin, view.yMin), out int x0, out int y0);
            grid.WorldToCell(new Vector2(view.xMax, view.yMax), out int x1, out int y1);

            float draft = 0.5f;
            var sel = SelectionManager.I != null ? SelectionManager.I.Primary : null;
            if (sel != null) draft = sel.Stats.draft;

            int drawn = 0;
            for (int y = y0; y <= y1 && drawn < 3000; y++)
                for (int x = x0; x <= x1 && drawn < 3000; x++)
                {
                    if (x < 0 || y < 0 || x >= grid.W || y >= grid.H) continue;
                    bool passable = grid.Passable(x, y, draft);
                    if (passable && grid.CoastDistance(x, y) > 2) continue;
                    Vector2 c = grid.CellCenter(x, y);
                    float h = grid.CellSize * 0.42f;
                    Color col = passable ? new Color(1f, 0.8f, 0.2f, 0.14f) : new Color(1f, 0.2f, 0.2f, 0.3f);
                    LineDrawer.Rect(c - new Vector2(h, h), c + new Vector2(h, h), 0.7f, col);
                    drawn++;
                }
        }

        public static Color AssignmentColor(AIAssignment a)
        {
            switch (a)
            {
                case AIAssignment.HoldCap: return new Color(0.4f, 1f, 0.6f);
                case AIAssignment.ContestCap: return new Color(1f, 0.85f, 0.3f);
                case AIAssignment.Decap: return new Color(1f, 0.35f, 0.25f);
                case AIAssignment.Screen: return new Color(0.5f, 0.85f, 1f);
                case AIAssignment.Hunt: return new Color(1f, 0.5f, 1f);
                case AIAssignment.Reserve: return new Color(0.7f, 0.7f, 0.8f);
                default: return new Color(0.85f, 0.85f, 0.85f);
            }
        }

        public static Color StateColor(AIState s)
        {
            switch (s)
            {
                case AIState.Attacking: return new Color(1f, 0.35f, 0.3f);
                case AIState.Tracking: return new Color(1f, 0.65f, 0.25f);
                case AIState.Searching: return new Color(0.5f, 0.8f, 1f);
                case AIState.Retreating: return new Color(1f, 0.4f, 0.9f);
                case AIState.Evading: return new Color(1f, 1f, 0.35f);
                case AIState.Patrolling: return new Color(0.4f, 0.9f, 0.7f);
                case AIState.Moving: return new Color(0.55f, 1f, 0.55f);
                case AIState.Repairing: return new Color(0.6f, 0.8f, 1f);
                case AIState.Disabled: return new Color(0.5f, 0.5f, 0.5f);
                case AIState.Sinking: return new Color(0.3f, 0.3f, 0.35f);
                default: return new Color(0.75f, 0.75f, 0.75f);
            }
        }
    }
}
