using UnityEngine;

namespace Naval
{
    public enum DamageSource { Shell, Secondary, Torpedo, DepthCharge, Fire, Flooding, Collision, Ramming }

    /// <summary>
    /// Compartmentalised damage: structural hit points plus seven independent systems, fires,
    /// progressive flooding, a damage control party and a full sinking sequence.
    /// </summary>
    public class ShipDamage
    {
        readonly Ship _s;

        public float Health { get; private set; }
        public float MaxHealth => _s.Stats.maxHealth;
        public float HealthFraction => Mathf.Clamp01(Health / Mathf.Max(1f, MaxHealth));

        readonly float[] _systems = new float[7];
        readonly string[] _systemNames = { "Hull", "Engine", "Steering", "Main Guns", "Secondaries", "Sensors", "Propulsion" };

        public int FireStacks { get; private set; }
        public int FloodingStacks { get; private set; }
        const int MaxFires = 3;
        const int MaxFloods = 3;

        float[] _fireTimers = new float[MaxFires];
        float[] _floodTimers = new float[MaxFloods];

        public float DamageControlCooldown { get; private set; }
        public bool DamageControlReady => DamageControlCooldown <= 0f;
        public bool IsSinking { get; private set; }
        public float SinkTimer { get; private set; }
        public bool IsDisabled => _systems[(int)ShipSystem.Engine] < 0.12f && _systems[(int)ShipSystem.Propulsion] < 0.2f;

        public Ship LastAttacker { get; private set; }
        public float TimeSinceHit { get; private set; } = 999f;
        public float RecentDamageTaken { get; private set; }

        float _sinkFxTimer;

        public ShipDamage(Ship s)
        {
            _s = s;
            Health = s.Stats.maxHealth;
            for (int i = 0; i < _systems.Length; i++) _systems[i] = 1f;
        }

        public float SystemIntegrity(ShipSystem sys) => _systems[(int)sys];
        public string SystemName(ShipSystem sys) => _systemNames[(int)sys];

        // ------------------------------------------------------------------ damage entry points

        public void ApplyDamage(float amount, Ship attacker, DamageSource source, Vector2 hitPos)
        {
            if (IsSinking || _s.IsDead || amount <= 0f) return;

            Health -= amount;
            RecentDamageTaken += amount;
            TimeSinceHit = 0f;
            if (attacker != null) LastAttacker = attacker;

            GameEvents.RaiseDamaged(_s, amount, attacker);

            if (Health <= 0f)
            {
                Health = 0f;
                BeginSinking(attacker);
            }
        }

        /// <summary>Full armour interaction for a gun hit. Returns what the shell actually did.</summary>
        public HitResult ApplyShellHit(float damage, float penetration, Vector2 hitPos, float impactHeading,
                                       Ship attacker, DamageSource source, float fireChance, bool isAP = true,
                                       float overmatchThreshold = 0f, float ricochetStart = 45f, float ricochetAlways = 60f)
        {
            if (IsSinking || _s.IsDead) return HitResult.Miss;

            // angle between the shell path and the hull's beam: a bow-on target bounces shells
            float relative = Mathf.Abs(Mathf.DeltaAngle(_s.Heading, impactHeading));
            float obliquity = Mathf.Abs(Mathf.Cos(relative * Mathf.Deg2Rad));   // 1 = hitting bow/stern on, 0 = flat broadside
            float effectiveArmor = _s.Stats.armor * Mathf.Lerp(1f, 2.6f, obliquity);

            // The belt normal is perpendicular to the hull axis, so obliquity is the sine of the
            // impact angle measured off that normal: broadside is 0 degrees, bow-on is 90.
            float impactAngle = Mathf.Asin(Mathf.Clamp01(obliquity)) * Mathf.Rad2Deg;

            // Overmatch: a shell heavy enough for the plating in front of it is not turned by any
            // angle at all. This is why angling the bow saves a cruiser from most guns but not from
            // the heaviest battleship rifles.
            bool overmatch = overmatchThreshold > 0f && _s.Stats.armor <= overmatchThreshold;

            HitResult result;
            float applied;

            if (!isAP)
            {
                // High explosive: no citadels and no over-penetrations. It either burns through the
                // plating for full damage or splashes for a fraction, and it starts fires either way.
                bool through = penetration >= effectiveArmor;
                result = through ? HitResult.Penetration : HitResult.Shatter;
                applied = damage * (through ? 1f : 0.3f);
                ApplyDamage(applied, attacker, source, hitPos);
                DamageRandomSystem(hitPos, applied / MaxHealth * 3.6f);
                if (Random.value < fireChance) StartFire();
                return result;
            }

            // An overmatching shell skips the angle checks entirely and goes straight to the
            // penetration cases below.
            bool bounced = false;
            if (!overmatch && impactAngle >= ricochetStart)
            {
                // between the two angles the bounce is a coin weighted by how sharp the impact is
                float t = Mathf.InverseLerp(ricochetStart, Mathf.Max(ricochetStart + 0.01f, ricochetAlways), impactAngle);
                bounced = t >= 1f || Random.value < t;
            }

            if (bounced)
            {
                result = HitResult.Ricochet;
                applied = damage * 0.02f;
            }
            else if (!overmatch && penetration < effectiveArmor * 0.55f)
            {
                result = HitResult.Shatter;
                applied = damage * 0.06f;
            }
            else if (!overmatch && penetration < effectiveArmor)
            {
                result = HitResult.Shatter;
                applied = damage * 0.1f;
            }
            else if (_s.Stats.hasCitadel && penetration > _s.Stats.citadelArmor * 1.15f
                     && obliquity < 0.55f && Random.value < 0.35f)
            {
                // The citadel is checked before overpenetration on purpose. Real penetration figures
                // are hundreds of millimetres against plating measured in tens, so a plating ratio
                // test alone would call every heavy hit an overpenetration and no shell would ever
                // find a magazine.
                result = HitResult.Citadel;
                applied = damage * 2.1f;
            }
            else if (penetration > effectiveArmor * 5.5f && !_s.Stats.hasCitadel)
            {
                // Nothing inside worth arming the fuse for: the shell goes straight through. This is
                // why battleship rifles are a poor answer to a destroyer.
                result = HitResult.Overpenetration;
                applied = damage * 0.28f;
            }
            else
            {
                result = HitResult.Penetration;
                applied = damage;
            }

            ApplyDamage(applied, attacker, source, hitPos);

            if (result != HitResult.Ricochet && result != HitResult.Shatter)
            {
                DamageRandomSystem(hitPos, applied / MaxHealth * 3.2f);
                if (Random.value < fireChance * (result == HitResult.Citadel ? 1.6f : 1f)) StartFire();
            }

            return result;
        }

        public void ApplyTorpedoHit(float damage, Vector2 hitPos, Ship attacker, float floodChance)
        {
            if (IsSinking || _s.IsDead) return;
            // the anti-torpedo bulge absorbs a fraction of the warhead, per ship
            float reduction = 1f - Mathf.Clamp01(_s.Stats.torpedoProtection);
            ApplyDamage(damage * reduction, attacker, DamageSource.Torpedo, hitPos);
            DamageSystem(ShipSystem.Hull, 0.25f);
            if (Random.value < 0.5f) DamageSystem(ShipSystem.Propulsion, 0.35f);
            if (Random.value < floodChance) StartFlooding();
            if (Random.value < 0.25f) StartFlooding();
        }

        /// <summary>Restores hull points (repair party, port repairs). Never revives a sinking ship.</summary>
        public void Heal(float amount)
        {
            if (IsSinking || _s.IsDead || amount <= 0f) return;
            Health = Mathf.Min(MaxHealth, Health + amount);
        }

        public void DamageRandomSystem(Vector2 hitPos, float severity)
        {
            severity = Mathf.Clamp(severity, 0.02f, 0.8f);
            // where the shell landed decides what it wrecked
            float along = Vector2.Dot(hitPos - _s.Position, _s.Forward) / Mathf.Max(1f, _s.Stats.length * 0.5f);
            ShipSystem sys;
            float r = Random.value;
            if (along > 0.35f) sys = r < 0.5f ? ShipSystem.MainGuns : r < 0.75f ? ShipSystem.Sensors : ShipSystem.Hull;
            else if (along < -0.35f) sys = r < 0.45f ? ShipSystem.Propulsion : r < 0.75f ? ShipSystem.Steering : ShipSystem.Hull;
            else sys = r < 0.35f ? ShipSystem.Engine : r < 0.6f ? ShipSystem.SecondaryGuns : r < 0.8f ? ShipSystem.Sensors : ShipSystem.Hull;
            DamageSystem(sys, severity);
        }

        public void DamageSystem(ShipSystem sys, float amount)
        {
            int i = (int)sys;
            float before = _systems[i];
            _systems[i] = Mathf.Clamp01(_systems[i] - amount);
            if (before > 0.3f && _systems[i] <= 0.3f && _s.team == Team.Player)
                GameEvents.RaiseMessage(_s.shipName + ": " + _systemNames[i] + " badly damaged", Team.Player);
        }

        public void StartFire()
        {
            if (FireStacks >= MaxFires) return;
            _fireTimers[FireStacks] = Random.Range(24f, 38f);
            FireStacks++;
            AudioManager.PlayAt(SoundId.Fire, _s.Position, 0.5f);
            if (_s.team == Team.Player && FireStacks == 1)
                GameEvents.RaiseMessage(_s.shipName + " is on fire!", Team.Player);
        }

        public void StartFlooding()
        {
            if (FloodingStacks >= MaxFloods) return;
            _floodTimers[FloodingStacks] = Random.Range(40f, 70f);
            FloodingStacks++;
            if (_s.team == Team.Player && FloodingStacks == 1)
                GameEvents.RaiseMessage(_s.shipName + " is flooding!", Team.Player);
        }

        // ------------------------------------------------------------------ tick

        public void Tick(float dt)
        {
            TimeSinceHit += dt;
            RecentDamageTaken = Mathf.Max(0f, RecentDamageTaken - MaxHealth * 0.05f * dt);

            if (IsSinking) { SinkTick(dt); return; }

            if (DamageControlCooldown > 0f) DamageControlCooldown -= dt;

            // fires ------------------------------------------------------------
            for (int i = FireStacks - 1; i >= 0; i--)
            {
                _fireTimers[i] -= dt;
                if (_fireTimers[i] <= 0f)
                {
                    _fireTimers[i] = _fireTimers[FireStacks - 1];
                    FireStacks--;
                    continue;
                }
                ApplyDamage(_s.Stats.fireRate * dt, LastAttacker, DamageSource.Fire, _s.Position);
                if (Random.value < 0.35f * dt) DamageSystem(ShipSystem.SecondaryGuns, 0.03f);
                // fires spread if they are left burning
                if (FireStacks < MaxFires && Random.value < 0.012f * dt * FireStacks) StartFire();
            }

            // flooding ---------------------------------------------------------
            for (int i = FloodingStacks - 1; i >= 0; i--)
            {
                _floodTimers[i] -= dt;
                if (_floodTimers[i] <= 0f)
                {
                    _floodTimers[i] = _floodTimers[FloodingStacks - 1];
                    FloodingStacks--;
                    continue;
                }
                ApplyDamage(_s.Stats.floodRate * dt, LastAttacker, DamageSource.Flooding, _s.Position);
            }

            // passive system repair ---------------------------------------------
            float repair = _s.Stats.repairRate * 0.01f * dt;
            if (FireStacks == 0 && FloodingStacks == 0) repair *= 1.6f;
            for (int i = 0; i < _systems.Length; i++)
                if (_systems[i] < 1f) _systems[i] = Mathf.Min(1f, _systems[i] + repair);
        }

        void SinkTick(float dt)
        {
            SinkTimer -= dt;
            _sinkFxTimer -= dt;
            if (_sinkFxTimer <= 0f)
            {
                _sinkFxTimer = 0.18f;
                Vector2 p = _s.Position + Random.insideUnitCircle * _s.Stats.length * 0.4f;
                ParticleFX.Smoke(p, _s.Stats.length * 0.35f, new Color(0.12f, 0.12f, 0.13f, 0.9f), 5f, new Vector2(0f, 2f));
                if (Random.value < 0.4f) ParticleFX.Fire(p, _s.Stats.length * 0.2f);
                if (Random.value < 0.3f) ParticleFX.Splash(p, 1.5f);
            }
            if (SinkTimer <= 0f)
            {
                ParticleFX.Debris(_s.Position, _s.Stats.length * 0.6f);
                ParticleFX.Splash(_s.Position, _s.Stats.length * 0.35f);
                _s.Kill(LastAttacker);
            }
        }

        void BeginSinking(Ship killer)
        {
            if (IsSinking) return;
            IsSinking = true;
            SinkTimer = Mathf.Lerp(4.5f, 9f, Mathf.InverseLerp(1200f, 7500f, MaxHealth));
            LastAttacker = killer != null ? killer : LastAttacker;
            _s.Movement.SetRudder(Random.value < 0.5f ? -0.6f : 0.6f);
            ParticleFX.Explosion(_s.Position, _s.Stats.length * 0.22f);
            AudioManager.PlayAt(SoundId.Explosion, _s.Position, 1f);
            GameEvents.RaiseMessage(_s.shipName + " is going down!", _s.team);
        }

        // ------------------------------------------------------------------ repair

        public bool UseDamageControl()
        {
            if (!DamageControlReady || IsSinking) return false;
            // damage control parties draw on the fleet's repair supplies
            if (GameManager.I != null && !GameManager.I.TrySpendRepair(_s.team, 8f))
            {
                if (_s.team == Team.Player)
                    GameEvents.RaiseMessage(_s.shipName + ": repair supplies exhausted", Team.Player);
                return false;
            }
            DamageControlCooldown = _s.Stats.damageControlCooldown;
            FireStacks = 0;
            FloodingStacks = 0;
            Health = Mathf.Min(MaxHealth, Health + MaxHealth * _s.Stats.damageControlHeal);
            for (int i = 0; i < _systems.Length; i++)
                _systems[i] = Mathf.Min(1f, _systems[i] + 0.35f);
            AudioManager.PlayAt(SoundId.Repair, _s.Position, 0.6f);
            if (_s.team == Team.Player)
                GameEvents.RaiseMessage(_s.shipName + ": damage control party away", Team.Player);
            return true;
        }

        /// <summary>Heavy repair while docked - the only way to get badly mauled ships back to full.</summary>
        public void PortRepair(float dt)
        {
            if (IsSinking) return;
            Health = Mathf.Min(MaxHealth, Health + MaxHealth * 0.045f * dt);
            for (int i = 0; i < _systems.Length; i++)
                _systems[i] = Mathf.Min(1f, _systems[i] + 0.09f * dt);
            if (FireStacks > 0) FireStacks = 0;
            if (FloodingStacks > 0) FloodingStacks = 0;
            DamageControlCooldown = Mathf.Min(DamageControlCooldown, 3f);
        }

        public bool NeedsPort => HealthFraction < 0.35f || SystemIntegrity(ShipSystem.Engine) < 0.3f;

        public string StatusLine()
        {
            if (IsSinking) return "SINKING";
            if (FloodingStacks > 0 && FireStacks > 0) return "FIRE + FLOODING";
            if (FloodingStacks > 0) return "FLOODING";
            if (FireStacks > 0) return "ON FIRE";
            if (IsDisabled) return "DISABLED";
            return "OPERATIONAL";
        }
    }
}
