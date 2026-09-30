using UnityEngine;
using UnityEngine.InputSystem;

namespace Naval
{
    /// <summary>
    /// First-captain controls for the ship the player is conning: WASD works the engine order
    /// telegraph and the rudder, the mouse trains the guns, left click fires, 1-4 run the
    /// consumables and X takes a submarine up or down.
    /// </summary>
    public class DirectShipController : MonoBehaviour
    {
        public static DirectShipController I { get; private set; }

        public float ThrottleRate = 0.9f;      // how fast the telegraph moves per second
        public float RudderRate = 2.2f;

        float _rudderInput;
        float _fireHeldTime;

        public Vector2 AimWorld { get; private set; }
        public Ship AimedAt { get; private set; }

        public static DirectShipController Create(Transform parent)
        {
            var go = new GameObject("DirectShipController");
            go.transform.SetParent(parent, false);
            var d = go.AddComponent<DirectShipController>();
            I = d;
            return d;
        }

        void Update()
        {
            var cm = ControlModeManager.I;
            if (cm == null || !cm.IsDirect) return;
            if (GameManager.I != null && GameManager.I.Phase != GamePhase.Battle) return;

            var ship = cm.Controlled;
            if (ship == null || ship.IsDead || ship.Damage.IsSinking) return;

            float dt = Time.unscaledDeltaTime;   // controls stay responsive at 4x/8x time compression

            HandleHelm(ship, dt);
            HandleAiming(ship);
            HandleWeapons(ship);
            HandleAbilities(ship);
            DrawReticle(ship);
        }

        // ------------------------------------------------------------------ helm

        void HandleHelm(Ship ship, float dt)
        {
            var move = ship.Movement;

            // W / S work the engine order telegraph, not an instant velocity
            if (InputHub.Key(Key.W)) move.NudgeThrottle(ThrottleRate * dt);
            if (InputHub.Key(Key.S)) move.NudgeThrottle(-ThrottleRate * dt);

            // A / D swing the rudder; releasing both eases it amidships
            float target = 0f;
            if (InputHub.Key(Key.A)) target -= 1f;
            if (InputHub.Key(Key.D)) target += 1f;

            if (Mathf.Abs(target) > 0.01f)
                _rudderInput = Mathf.MoveTowards(_rudderInput, target, RudderRate * dt);
            else
                _rudderInput = Mathf.MoveTowards(_rudderInput, 0f, RudderRate * 1.4f * dt);

            move.SetRudder(_rudderInput);

            // quick orders
            if (InputHub.KeyDown(Key.Space)) { move.SetThrottle(0f); GameEvents.RaiseMessage(ship.shipName + ": all stop", Team.Player); }
        }

        // ------------------------------------------------------------------ gunnery

        void HandleAiming(Ship ship)
        {
            if (RTSCamera.I == null) return;
            AimWorld = RTSCamera.I.ScreenToWorld(InputHub.MousePosition);
            ship.Weapons.AimPoint = AimWorld;

            // if the reticle is sitting on a visible enemy, lead it automatically like a gunnery officer
            AimedAt = null;
            var enemies = ShipRegistry.OfTeam(Teams.Opponent(ship.team));
            float bestDist = 45f;
            for (int i = 0; i < enemies.Count; i++)
            {
                var e = enemies[i];
                if (e == null || e.IsDead) continue;
                if (e.Visual != null && !e.Visual.VisibleToPlayer) continue;
                float d = Vector2.Distance(AimWorld, e.Position);
                if (d < bestDist) { bestDist = d; AimedAt = e; }
            }

            if (AimedAt != null && ship.Stats.mainBattery != null)
            {
                if (NavalMath.Intercept(ship.Position, AimedAt.Position, AimedAt.Velocity,
                        ship.Stats.mainBattery.shellSpeed, out Vector2 lead, out float tof))
                    ship.Weapons.AimPoint = lead;
                ship.CurrentTarget = AimedAt;      // so secondaries and the HUD know what we are on
            }
        }

        void HandleWeapons(Ship ship)
        {
            if (UIManager.IsPointerOverUI(InputHub.MousePosition)) return;

            if (InputHub.LeftHeld) _fireHeldTime += Time.unscaledDeltaTime;
            else _fireHeldTime = 0f;

            // click or hold to keep the battery firing as it reloads
            if (InputHub.LeftDown || _fireHeldTime > 0.15f)
                ship.Weapons.RequestMainFire();

            // right click sends a torpedo spread along the reticle bearing
            if (InputHub.RightDown && ship.Stats.torpedoes != null)
            {
                if (!ship.Weapons.LaunchTorpedoesAt(AimWorld))
                {
                    if (!ship.Weapons.TorpedoesReady)
                        GameEvents.RaiseMessage(ship.shipName + ": tubes reloading", Team.Player);
                    else
                        GameEvents.RaiseMessage(ship.shipName + ": target outside the launch arc", Team.Player);
                }
            }
        }

        void HandleAbilities(Ship ship)
        {
            var ab = ship.Abilities;
            if (ab == null) return;

            // each consumable answers to the hotkey it declares (1-6 depending on the class)
            int num = InputHub.NumberRowDown();
            var slot = num >= 1 && !InputHub.Ctrl ? ab.GetByHotkey(num.ToString()) : null;
            if (slot != null && !ab.Use(slot.id) && !slot.Ready)
                GameEvents.RaiseMessage(slot.label + (slot.HasCharges ? " reloading" : " - no charges left"), Team.Player);

            if (InputHub.KeyDown(Key.X) && ship.Submarine != null) ab.Use(AbilityId.Dive);
            if (InputHub.KeyDown(Key.E)) ab.Use(AbilityId.DamageControl);
        }

        // ------------------------------------------------------------------ reticle

        void DrawReticle(Ship ship)
        {
            float ps = RTSCamera.I != null ? RTSCamera.I.PixelScale : 0.1f;
            var mb = ship.Stats.mainBattery;

            Color ready = ship.Weapons.MainReady ? new Color(0.55f, 1f, 0.7f, 0.9f) : new Color(1f, 0.8f, 0.4f, 0.6f);
            if (ship.Weapons.OnTarget) ready = ship.Weapons.MainReady ? new Color(0.5f, 1f, 0.6f, 1f) : ready;

            // crosshair on the cursor
            float r = 10f * ps;
            LineDrawer.Line(AimWorld + new Vector2(-r, 0f), AimWorld + new Vector2(-r * 0.35f, 0f), 1.4f * ps, ready);
            LineDrawer.Line(AimWorld + new Vector2(r * 0.35f, 0f), AimWorld + new Vector2(r, 0f), 1.4f * ps, ready);
            LineDrawer.Line(AimWorld + new Vector2(0f, -r), AimWorld + new Vector2(0f, -r * 0.35f), 1.4f * ps, ready);
            LineDrawer.Line(AimWorld + new Vector2(0f, r * 0.35f), AimWorld + new Vector2(0f, r), 1.4f * ps, ready);

            // the actual aim point (lead) when it differs from the cursor
            Vector2 aim = ship.Weapons.AimPoint;
            if ((aim - AimWorld).sqrMagnitude > 1f)
            {
                LineDrawer.Circle(aim, 6f * ps, 1.3f * ps, new Color(1f, 0.55f, 0.4f, 0.9f), 16);
                LineDrawer.Dashed(AimWorld, aim, 1f * ps, new Color(1f, 0.55f, 0.4f, 0.5f), 6f * ps, 5f * ps);
            }

            if (mb != null)
            {
                // maximum range ring, and a warning tint when the reticle is beyond it
                float dist = Vector2.Distance(ship.Position, AimWorld);
                Color ringCol = dist > mb.range ? new Color(1f, 0.4f, 0.35f, 0.25f) : new Color(1f, 0.85f, 0.5f, 0.18f);
                LineDrawer.DashedCircle(ship.Position, mb.range, 1.1f * ps, ringCol, 96);
            }

            // torpedo launch arc, so the player can see when the tubes will bear
            var td = ship.Stats.torpedoes;
            if (td != null && ship.Weapons.TorpedoesReady)
            {
                float half = td.launchArc * 0.5f;
                for (int side = -1; side <= 1; side += 2)
                {
                    float centre = ship.Heading + 90f * side;
                    for (int k = -1; k <= 1; k += 2)
                    {
                        float a = centre + half * k;
                        Vector2 d = NavalMath.HeadingToVector(a);
                        LineDrawer.Line(ship.Position + d * ship.Stats.length * 0.5f,
                                        ship.Position + d * td.range, 0.8f * ps,
                                        new Color(0.45f, 1f, 0.85f, 0.12f));
                    }
                }
            }
        }
    }
}
