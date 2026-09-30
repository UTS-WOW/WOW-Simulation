using System.Collections.Generic;
using UnityEngine;

namespace Naval
{
    public enum AbilityId
    {
        None,
        ShellHE, ShellAP,            // ammunition selection (instant toggles, no cooldown)
        Torpedoes, HomingTorpedoes,
        SmokeScreen, EngineBoost,
        HydroacousticSearch, SurveillanceRadar,
        DamageControl, RepairParty,
        SonarPing, Hydrophone,
        Dive,
        SpotterPlane, SubmarineSurveillance
    }

    /// <summary>One consumable slot: charges, cooldown, active duration.</summary>
    public class Ability
    {
        public AbilityId id;
        public string label;
        public string hotkey;
        public float cooldown;
        public float duration;      // 0 = instant effect
        public int maxCharges;      // 0 = unlimited

        public float cooldownLeft;
        public float activeLeft;
        public int chargesLeft;

        public bool IsActive => activeLeft > 0f;
        public bool IsToggle => id == AbilityId.ShellHE || id == AbilityId.ShellAP || id == AbilityId.Dive;
        public bool HasCharges => maxCharges <= 0 || chargesLeft > 0;
        public bool Ready => cooldownLeft <= 0f && !IsActive && HasCharges;
        public float CooldownFraction => cooldown <= 0f ? 1f : 1f - Mathf.Clamp01(cooldownLeft / cooldown);
        public float ActiveFraction => duration <= 0f ? 0f : Mathf.Clamp01(activeLeft / duration);
    }

    /// <summary>
    /// Consumables. Each class carries up to six slots on keys 1-6 (submarines also use X to dive).
    /// The same code path serves the player's action bar and the AI - the AI simply calls Use().
    /// </summary>
    public class ShipAbilities
    {
        readonly Ship _s;
        public readonly List<Ability> Slots = new List<Ability>();

        /// <summary>Currently loaded shell type. Ships without AP simply always use HE.</summary>
        public AbilityId ShellType { get; private set; } = AbilityId.ShellHE;

        // active effect state read by other systems
        public float SpeedMultiplier { get; private set; } = 1f;
        public float DetectionBonus { get; private set; } = 0f;      // extra spotting range
        public bool SeesThroughSmoke { get; private set; }
        /// <summary>Main battery range multiplier, from the spotter aircraft.</summary>
        public float GunRangeMultiplier { get; private set; } = 1f;
        /// <summary>Radius inside which contacts are spotted regardless of their concealment.</summary>
        public float AssuredDetectionRange { get; private set; }
        /// <summary>Whether that assured detection also reaches submerged boats.</summary>
        public bool AssuredDetectionSubmerged { get; private set; }
        public float RepairPerSecond { get; private set; }
        public bool SonarActive { get; private set; }

        public ShipAbilities(Ship s)
        {
            _s = s;
            Build(s.Stats.classType);
        }

        // Consumable radii, in world units (1 unit is about 10 m).
        public const float RadarRange = 1000f;        // 10.0 km
        public const float HydroRange = 500f;         //  5.0 km
        public const float HydrophoneRange = 700f;    //  7.0 km
        public const float SubSurveillanceRange = 900f; // 9.0 km
        /// <summary>Repair party heal, as a fraction of max HP per second.</summary>
        public const float RepairFraction = 0.005f;

        void Add(AbilityId id, string label, string key, float cooldown, float duration, int charges)
        {
            Slots.Add(new Ability
            {
                id = id,
                label = label,
                hotkey = key,
                cooldown = cooldown,
                duration = duration,
                maxCharges = charges,
                chargesLeft = charges
            });
        }

        void Build(ShipClassType cls)
        {
            switch (cls)
            {
                case ShipClassType.Destroyer:
                    Add(AbilityId.ShellHE, "HE Shells", "1", 0f, 0f, 0);
                    Add(AbilityId.ShellAP, "AP Shells", "2", 0f, 0f, 0);
                    Add(AbilityId.Torpedoes, "Torpedoes", "3", 0f, 0f, 0);
                    Add(AbilityId.SmokeScreen, "Smoke Screen", "4", 160f, 97f, 3);
                    Add(AbilityId.EngineBoost, "Engine Boost", "5", 120f, 120f, 3);
                    Add(AbilityId.DamageControl, "Damage Control", "6", 40f, 0f, 0);
                    break;

                case ShipClassType.Cruiser:
                    Add(AbilityId.ShellHE, "HE Shells", "1", 0f, 0f, 0);
                    Add(AbilityId.ShellAP, "AP Shells", "2", 0f, 0f, 0);
                    Add(AbilityId.SurveillanceRadar, "Radar", "3", 120f, 40f, 3);
                    Add(AbilityId.HydroacousticSearch, "Hydro Search", "4", 120f, 100f, 3);
                    Add(AbilityId.RepairParty, "Repair Party", "5", 80f, 28f, 3);
                    Add(AbilityId.DamageControl, "Damage Control", "6", 60f, 0f, 0);
                    break;

                case ShipClassType.Battleship:
                    Add(AbilityId.ShellHE, "HE Shells", "1", 0f, 0f, 0);
                    Add(AbilityId.ShellAP, "AP Shells", "2", 0f, 0f, 0);
                    Add(AbilityId.DamageControl, "Damage Control", "3", 80f, 20f, 0);
                    Add(AbilityId.RepairParty, "Repair Party", "4", 80f, 28f, 4);
                    Add(AbilityId.SpotterPlane, "Spotter Plane", "5", 240f, 100f, 4);
                    ShellType = AbilityId.ShellAP;      // battleships load AP by default
                    break;

                case ShipClassType.Submarine:
                    // the deck gun is surface-only, but it still needs an ammunition selection
                    Add(AbilityId.ShellHE, "HE Shells", "1", 0f, 0f, 0);
                    Add(AbilityId.HomingTorpedoes, "Homing Torps", "2", 0f, 0f, 0);
                    Add(AbilityId.SonarPing, "Sonar Ping", "3", 6.5f, 25f, 0);
                    Add(AbilityId.Hydrophone, "Hydrophone", "4", 60f, 30f, 4);
                    Add(AbilityId.SubmarineSurveillance, "Sub Surveillance", "5", 120f, 60f, 3);
                    Add(AbilityId.DamageControl, "Damage Control", "6", 40f, 15f, 3);
                    Add(AbilityId.Dive, "Dive / Surface", "X", 0f, 0f, 0);
                    break;

                default:
                    Add(AbilityId.DamageControl, "Damage Control", "3", 110f, 0f, 0);
                    break;
            }
        }

        public Ability Get(AbilityId id)
        {
            for (int i = 0; i < Slots.Count; i++) if (Slots[i].id == id) return Slots[i];
            return null;
        }

        public Ability GetSlot(int index) => index >= 0 && index < Slots.Count ? Slots[index] : null;

        /// <summary>The slot bound to a hotkey ("1".."6", "X"), or null.</summary>
        public Ability GetByHotkey(string key)
        {
            for (int i = 0; i < Slots.Count; i++) if (Slots[i].hotkey == key) return Slots[i];
            return null;
        }

        public bool Has(AbilityId id) => Get(id) != null;

        // ------------------------------------------------------------------ activation

        public bool UseSlot(int index)
        {
            var a = GetSlot(index);
            return a != null && Use(a.id);
        }

        public bool Use(AbilityId id)
        {
            var a = Get(id);
            if (a == null) return false;

            // ammunition and depth toggles are always available
            switch (id)
            {
                case AbilityId.ShellHE:
                case AbilityId.ShellAP:
                    ShellType = id;
                    if (_s.team == Team.Player && _s.Controller == ShipController.Human)
                        GameEvents.RaiseMessage(_s.shipName + ": loading " + (id == AbilityId.ShellHE ? "HE" : "AP"), Team.Player);
                    return true;

                case AbilityId.Torpedoes:
                case AbilityId.HomingTorpedoes:
                    // handled by the weapon system; the slot is a reminder of the key binding
                    return _s.Weapons.LaunchTorpedoesAt(_s.Weapons.AimPoint);

                case AbilityId.Dive:
                    if (_s.Submarine == null) return false;
                    if (_s.Submarine.Depth == DepthState.Surface) _s.Submarine.Dive();
                    else _s.Submarine.Surface();
                    return true;
            }

            if (!a.Ready) return false;

            a.cooldownLeft = a.cooldown;
            if (a.maxCharges > 0) a.chargesLeft--;
            if (a.duration > 0f) a.activeLeft = a.duration;

            switch (id)
            {
                case AbilityId.SmokeScreen:
                    if (!_s.Weapons.DeploySmoke())
                    {
                        a.cooldownLeft = 0f;
                        if (a.maxCharges > 0) a.chargesLeft++;
                        return false;
                    }
                    break;

                case AbilityId.EngineBoost:
                    Announce("engine boost engaged");
                    break;

                case AbilityId.HydroacousticSearch:
                    Announce("hydroacoustic search active");
                    AudioManager.PlayAt(SoundId.Sonar, _s.Position, 0.6f);
                    break;

                case AbilityId.SurveillanceRadar:
                    Announce("radar active");
                    AudioManager.PlayAt(SoundId.Sonar, _s.Position, 0.7f);
                    break;

                case AbilityId.DamageControl:
                    if (!_s.Damage.UseDamageControl())
                    {
                        // refund if the damage control party could not be sent away
                        a.cooldownLeft = 0f;
                        if (a.maxCharges > 0) a.chargesLeft++;
                        return false;
                    }
                    break;

                case AbilityId.RepairParty:
                    Announce("repair party working");
                    break;

                case AbilityId.SpotterPlane:
                    Announce("spotter plane away");
                    break;

                case AbilityId.SubmarineSurveillance:
                    Announce("submarine surveillance active");
                    AudioManager.PlayAt(SoundId.Sonar, _s.Position, 0.6f);
                    break;

                case AbilityId.SonarPing:
                    FireSonarPing();
                    break;

                case AbilityId.Hydrophone:
                    Announce("hydrophone sweep");
                    AudioManager.PlayAt(SoundId.Sonar, _s.Position, 0.5f);
                    break;
            }
            return true;
        }

        void Announce(string what)
        {
            if (_s.team == Team.Player) GameEvents.RaiseMessage(_s.shipName + ": " + what, Team.Player);
        }

        /// <summary>Submarine ping: marks a target so homing torpedoes can track it.</summary>
        void FireSonarPing()
        {
            ParticleFX.SonarPing(_s.Position, 120f);
            AudioManager.PlayAt(SoundId.Sonar, _s.Position, 0.8f);

            var enemies = ShipRegistry.OfTeam(Teams.Opponent(_s.team));
            float best = 420f;
            Ship hit = null;
            Vector2 aim = _s.Weapons.AimPoint;
            for (int i = 0; i < enemies.Count; i++)
            {
                var e = enemies[i];
                if (e == null || e.IsDead) continue;
                float d = Vector2.Distance(_s.Position, e.Position);
                if (d > 420f) continue;
                // prefer whatever the player is aiming at
                float score = d + Vector2.Distance(aim, e.Position) * 0.5f;
                if (score < best) { best = score; hit = e; }
            }

            if (hit != null)
            {
                hit.Detection.ApplySonarLock(24f);
                Announce("sonar lock on " + hit.shipName);
            }
        }

        // ------------------------------------------------------------------ tick

        public void Tick(float dt)
        {
            SpeedMultiplier = 1f;
            DetectionBonus = 0f;
            SeesThroughSmoke = false;
            GunRangeMultiplier = 1f;
            AssuredDetectionRange = 0f;
            AssuredDetectionSubmerged = false;
            RepairPerSecond = 0f;
            SonarActive = false;

            for (int i = 0; i < Slots.Count; i++)
            {
                var a = Slots[i];
                if (a.cooldownLeft > 0f) a.cooldownLeft = Mathf.Max(0f, a.cooldownLeft - dt);
                if (a.activeLeft > 0f)
                {
                    a.activeLeft = Mathf.Max(0f, a.activeLeft - dt);
                    ApplyActiveEffect(a, dt);
                    if (a.activeLeft <= 0f && a.id == AbilityId.RepairParty) Announce("repair party finished");
                }
            }
        }

        void ApplyActiveEffect(Ability a, float dt)
        {
            switch (a.id)
            {
                case AbilityId.EngineBoost:
                    SpeedMultiplier = 1.08f;
                    break;

                case AbilityId.SpotterPlane:
                    // the aircraft spots the fall of shot, extending the usable gun range
                    GunRangeMultiplier = 1.2f;
                    break;

                case AbilityId.SubmarineSurveillance:
                    // 9.0 km, and unlike anything else it finds boats at depth
                    AssuredDetectionRange = Mathf.Max(AssuredDetectionRange, SubSurveillanceRange);
                    AssuredDetectionSubmerged = true;
                    break;

                case AbilityId.HydroacousticSearch:
                    // 5.0 km against ships, and it hears through smoke and hull noise alike
                    DetectionBonus = Mathf.Max(DetectionBonus, HydroRange);
                    AssuredDetectionRange = Mathf.Max(AssuredDetectionRange, HydroRange);
                    AssuredDetectionSubmerged = true;
                    SeesThroughSmoke = true;
                    SonarActive = true;
                    break;

                case AbilityId.SurveillanceRadar:
                    // 10.0 km, and unlike hydro it reaches straight through islands
                    DetectionBonus = Mathf.Max(DetectionBonus, RadarRange);
                    AssuredDetectionRange = Mathf.Max(AssuredDetectionRange, RadarRange);
                    SeesThroughSmoke = true;
                    break;

                case AbilityId.Hydrophone:
                    // the boat's own passive set: 7.0 km, and it hears submerged contacts too
                    DetectionBonus = Mathf.Max(DetectionBonus, HydrophoneRange);
                    AssuredDetectionRange = Mathf.Max(AssuredDetectionRange, HydrophoneRange);
                    AssuredDetectionSubmerged = true;
                    SeesThroughSmoke = true;
                    SonarActive = true;
                    break;

                case AbilityId.RepairParty:
                    // heals a slice of the hull back, the classic battleship heal
                    RepairPerSecond = _s.Stats.maxHealth * RepairFraction;
                    _s.Damage.Heal(RepairPerSecond * dt);
                    break;
            }
        }

        public string StatusLine()
        {
            for (int i = 0; i < Slots.Count; i++)
                if (Slots[i].IsActive) return Slots[i].label.ToUpper() + " " + Mathf.CeilToInt(Slots[i].activeLeft) + "s";
            return "";
        }
    }
}
