using System.Collections.Generic;
using UnityEngine;

namespace Naval
{
    /// <summary>What the fleet commander wants this ship to be doing.</summary>
    public enum AIAssignment
    {
        Station,      // hold the position given
        HoldCap,      // sit inside a zone we own and keep it
        ContestCap,   // get inside a zone we do not own and take it
        Decap,        // break up an enemy capture in progress and reset the meter
        Screen,       // stand between the objective and the enemy
        Hunt,         // prosecute a submarine, or lie in ambush
        Reserve       // stand off behind the line
    }

    /// <summary>
    /// Per ship decision making. Runs a state machine a few times a second and expresses its decisions
    /// as navigation orders and weapon requests - it never drives the hull directly.
    ///
    /// It reads the shared BattleAssessment for its team, so a ship knows what mode is being played,
    /// whether its side is winning, and whether it is locally outgunned right now.
    ///
    /// Player ships run a reduced version: they keep their own gunnery, damage control and torpedo
    /// dodging and will angle their armour when idle, but never override the orders the player gave.
    /// </summary>
    public class ShipAI
    {
        readonly Ship _s;

        public AIState State { get; private set; } = AIState.Idle;
        public string StateReason { get; private set; } = "";
        public Ship ManualTarget;                 // focus-fire order from the player or the commander
        public bool AutoEngage = true;
        public bool AutoEvade = true;
        public bool AutoDamageControl = true;
        public Vector2 StationPoint;              // assigned by the fleet commander
        public bool HasStation;
        public float Aggression = 1f;

        public AIAssignment Assignment = AIAssignment.Station;
        public CaptureZone AssignedZone;

        /// <summary>Shared team picture, republished by the fleet commander a couple of times a second.</summary>
        BattleAssessment _intel;
        /// <summary>Friendly versus known enemy strength around us. Above 1 means we are winning locally.</summary>
        public float LocalRatio { get; private set; } = 1f;

        float _decisionTimer;
        float _stateTimer;
        float _torpedoRunTimer;
        float _postAttackTimer;
        float _smokeTimer;
        float _coverTimer;
        Vector2 _coverPoint;
        readonly List<Ship> _visibleEnemies = new List<Ship>();

        public ShipAI(Ship s)
        {
            _s = s;
            _decisionTimer = Random.Range(0f, 0.4f);
        }

        public bool IsPlayerControlled => _s.Controller == ShipController.Human;
        public bool IsLearned => _s.Controller == ShipController.Learned;

        // ------------------------------------------------------------------ tick

        public void Tick(float dt)
        {
            _stateTimer += dt;
            if (_torpedoRunTimer > 0f) _torpedoRunTimer -= dt;
            if (_postAttackTimer > 0f) _postAttackTimer -= dt;
            if (_smokeTimer > 0f) _smokeTimer -= dt;
            if (_coverTimer > 0f) _coverTimer -= dt;

            if (_s.Damage.IsSinking) { SetState(AIState.Sinking, "hull lost"); return; }

            // torpedo dodging has to be checked at full rate
            if (AutoEvade) CheckIncomingTorpedoes();

            _decisionTimer -= dt;
            if (_decisionTimer > 0f) return;
            _decisionTimer = 1f / GameConfig.AIUpdateRate + Random.Range(-0.03f, 0.05f);

            _intel = BattleAssessment.For(_s.team);
            if (_intel != null) LocalRatio = _intel.LocalStrengthRatio(_s.Position, 380f);

            if (IsLearned)
            {
                // A learned policy chooses targets, consumables and movement itself. Only the
                // reflexes it was trained with stay on (AutoEvade above, damage control here).
                if (AutoDamageControl) ConsiderDamageControl();
                SetState(MapPlayerState(), "learned policy");
                return;
            }

            GatherEnemies();
            if (AutoDamageControl) ConsiderDamageControl();
            if (!IsPlayerControlled || !_s.IsDirectlyControlled) ConsiderConsumables();
            SelectTarget();

            if (IsPlayerControlled)
            {
                // player ships only reason about their own survival and gunnery
                if (_s.Navigation.Order == OrderType.AttackMove || _s.Navigation.Order == OrderType.Attack)
                    PlayerAttackAssist();
                // fighting smarter is free; steering is not, so we only angle when idle and unordered
                if (!_s.IsDirectlyControlled && OrderIsIdle) ConsiderAngling();
                SetState(MapPlayerState(), "player order");
                return;
            }

            Think();
        }

        AIState MapPlayerState()
        {
            if (_s.Navigation.IsEvading) return AIState.Evading;
            if (_s.Damage.IsDisabled) return AIState.Disabled;
            if (_s.CurrentTarget != null) return AIState.Attacking;
            switch (_s.Navigation.Order)
            {
                case OrderType.Patrol: return AIState.Patrolling;
                case OrderType.Move:
                case OrderType.AttackMove:
                case OrderType.Follow:
                case OrderType.ReturnToPort: return AIState.Moving;
                case OrderType.Retreat: return AIState.Retreating;
                default: return AIState.Idle;
            }
        }

        void PlayerAttackAssist()
        {
            // fire torpedoes automatically on an attack order when a good solution exists
            if (_s.CurrentTarget != null && _s.Weapons.TorpedoesReady)
            {
                float d = _s.DistanceTo(_s.CurrentTarget);
                var td = _s.Stats.torpedoes;
                if (td != null && d < td.range * 0.85f)
                    _s.Weapons.LaunchTorpedoesAtTarget(_s.CurrentTarget);
            }
        }

        void SetState(AIState s, string reason)
        {
            if (State != s) { State = s; _stateTimer = 0f; }
            StateReason = reason;
        }

        /// <summary>True when the player has not told this ship to go anywhere.</summary>
        bool OrderIsIdle
        {
            get
            {
                var o = _s.Navigation.Order;
                return o == OrderType.None || o == OrderType.Stop || o == OrderType.HoldPosition;
            }
        }

        /// <summary>
        /// Armour angling. A hull that is reloading gains nothing from showing its side, so it turns
        /// bow-on to the incoming fire and only swings back out when the salvo is ready. This is the
        /// single biggest survivability habit a real captain has.
        /// </summary>
        void ConsiderAngling()
        {
            if (_intel != null && !_intel.UsesAngling) return;
            if (_s.Stats.classType != ShipClassType.Battleship && _s.Stats.classType != ShipClassType.Cruiser) return;

            Ship threat = _s.Damage.TimeSinceHit < 10f ? _s.Damage.LastAttacker : _s.CurrentTarget;
            if (threat == null || threat.IsDead) return;

            float toThreat = NavalMath.VectorToHeading(threat.Position - _s.Position);
            bool loaded = _s.Weapons.MainReady || _s.Weapons.MainReloadFraction > 0.85f;

            // 30 degrees off the incoming line bounces shells; 75 unmasks the whole broadside
            float offset = loaded ? 72f : 30f;
            float side = Mathf.DeltaAngle(toThreat, _s.Heading) >= 0f ? 1f : -1f;
            _s.Movement.SteerToHeading(toThreat + offset * side);
        }

        // ------------------------------------------------------------------ perception

        void GatherEnemies()
        {
            if (DetectionSystem.I != null) DetectionSystem.I.GatherLiveContacts(_s.team, _visibleEnemies);
            else _visibleEnemies.Clear();
        }

        void CheckIncomingTorpedoes()
        {
            var ps = ProjectileSystem.I;
            if (ps == null) return;
            float notice = _s.Stats.torpedoes != null ? _s.Stats.torpedoes.detectRange : 55f;
            notice = Mathf.Max(notice, _s.Detection.EffectiveHydroRange * 0.6f);
            if (_s.Damage.SystemIntegrity(ShipSystem.Sensors) < 0.4f) notice *= 0.6f;

            var torps = ps.Torpedoes;
            for (int i = 0; i < torps.Count; i++)
            {
                var t = torps[i];
                if (t.team == _s.team) continue;
                Vector2 rel = _s.Position - t.pos;
                float dist = rel.magnitude;
                if (dist > notice) continue;

                Vector2 dir = NavalMath.HeadingToVector(t.heading);
                if (Vector2.Dot(rel.normalized, dir) < 0.55f) continue;      // not running at us

                // is it going to pass close?
                float cpa = NavalMath.DistanceToSegment(_s.Position, t.pos, t.pos + dir * Mathf.Min(t.rangeLeft, dist + 60f));
                if (cpa > _s.Stats.length * 1.6f) continue;

                ps.MarkTorpedoSpotted(t);
                if (_s.team == Team.Player && !_s.Navigation.IsEvading)
                {
                    GameEvents.RaiseTorpedoWarning(t.pos, _s.team);
                    GameEvents.RaiseMessage("TORPEDOES INBOUND - " + _s.shipName, Team.Player);
                    AudioManager.PlayAt(SoundId.Alarm, _s.Position, 0.8f);
                }
                _s.Navigation.Evade(t.pos);
                SetState(AIState.Evading, "torpedoes inbound");
                return;
            }
        }

        void ConsiderDamageControl()
        {
            var d = _s.Damage;
            if (!d.DamageControlReady) return;
            bool bad = d.FloodingStacks >= 1 && d.HealthFraction < 0.85f;
            bad |= d.FireStacks >= 2;
            bad |= d.FireStacks >= 1 && d.HealthFraction < 0.45f;
            bad |= d.SystemIntegrity(ShipSystem.Engine) < 0.25f;
            if (bad) d.UseDamageControl();
        }

        /// <summary>
        /// Situational use of the ship's consumables. Each class spends them the way a competent
        /// captain would: destroyers hide and sprint, cruisers light up what they cannot see,
        /// battleships heal, submarines ping for a firing solution.
        /// </summary>
        void ConsiderConsumables()
        {
            var ab = _s.Abilities;
            if (ab == null) return;

            float hp = _s.Damage.HealthFraction;
            bool spotted = _s.Detection.SpottedByEnemy;
            bool underFire = _s.Damage.TimeSinceHit < 8f;

            Ship nearest = null;
            float nearestDist = float.MaxValue;
            Ship nearestDD = null;
            float nearestDDDist = float.MaxValue;
            for (int i = 0; i < _visibleEnemies.Count; i++)
            {
                var e = _visibleEnemies[i];
                float d = _s.DistanceTo(e);
                if (d < nearestDist) { nearestDist = d; nearest = e; }
                if (e.Stats.classType == ShipClassType.Destroyer && d < nearestDDDist) { nearestDDDist = d; nearestDD = e; }
            }

            switch (_s.Stats.classType)
            {
                case ShipClassType.Destroyer:
                    // smoke to break contact when the shells start landing
                    if (spotted && underFire && hp < 0.8f && nearestDist < 500f)
                        ab.Use(AbilityId.SmokeScreen);
                    // sprint when running in for torpedoes or running away
                    if (State == AIState.Retreating || (State == AIState.Tracking && nearestDist > 220f))
                        ab.Use(AbilityId.EngineBoost);
                    break;

                case ShipClassType.Cruiser:
                    // hydro when something is close enough to be launching fish
                    if (nearestDist < 260f || (SmokeSystem.I != null && SmokeSystem.I.IsInsideSmoke(_s.Position)))
                        ab.Use(AbilityId.HydroacousticSearch);
                    // radar to burn a destroyer that is knife-fighting us
                    if (nearestDD != null && nearestDDDist < 420f)
                        ab.Use(AbilityId.SurveillanceRadar);
                    // or one we cannot see but are fairly sure is sitting on the point
                    else if (_intel != null && _intel.UsesInference &&
                             _intel.DarkThreatNear(_s.Position, 460f, ShipClassType.Destroyer))
                        ab.Use(AbilityId.SurveillanceRadar);
                    break;

                case ShipClassType.Battleship:
                    // heal once the fires are out, otherwise the repair is wasted
                    if (hp < 0.62f && _s.Damage.FireStacks == 0 && _s.Damage.FloodingStacks == 0)
                        ab.Use(AbilityId.RepairParty);
                    break;

                case ShipClassType.Submarine:
                    var td = _s.Stats.torpedoes;
                    if (nearest != null && td != null && nearestDist < td.range * 0.95f)
                        ab.Use(AbilityId.SonarPing);          // paints the target for homing fish
                    else if (_visibleEnemies.Count == 0)
                        ab.Use(AbilityId.Hydrophone);         // listen for something to hunt
                    break;
            }
        }

        // ------------------------------------------------------------------ targeting

        void SelectTarget()
        {
            if (ManualTarget != null && !ManualTarget.IsDead && CanEngage(ManualTarget))
            {
                _s.CurrentTarget = ManualTarget;
                return;
            }
            if (ManualTarget != null && ManualTarget.IsDead) ManualTarget = null;

            if (!AutoEngage)
            {
                // under direct control the player's reticle chooses the target
                if (!_s.IsDirectlyControlled) _s.CurrentTarget = null;
                return;
            }

            Ship best = null;
            float bestScore = float.MinValue;
            for (int i = 0; i < _visibleEnemies.Count; i++)
            {
                var e = _visibleEnemies[i];
                if (!CanEngage(e)) continue;
                float score = ScoreTarget(e);
                if (score > bestScore) { bestScore = score; best = e; }
            }
            _s.CurrentTarget = best;
        }

        /// <summary>
        /// The target this ship's own gunnery logic would pick right now, without assigning it. A
        /// learned policy's "auto" target uses this, so it only has to learn when to override it.
        /// </summary>
        public Ship PickGunTarget()
        {
            GatherEnemies();
            Ship best = null;
            float bestScore = float.MinValue;
            for (int i = 0; i < _visibleEnemies.Count; i++)
            {
                var e = _visibleEnemies[i];
                if (!CanEngage(e)) continue;
                float score = ScoreTarget(e);
                if (score > bestScore) { bestScore = score; best = e; }
            }
            return best;
        }

        bool CanEngage(Ship e)
        {
            if (e == null || e.IsDead || e.Damage.IsSinking) return false;
            if (e.Submarine != null && e.Submarine.IsSubmerged) return false;    // guns cannot touch a dived boat
            var mb = _s.Stats.mainBattery;
            if (mb == null) return false;
            if (mb.surfaceOnly && _s.Submarine != null && !_s.Submarine.CanUseDeckGun) return false;
            float d = _s.DistanceTo(e);
            return d <= mb.range * 1.15f;
        }

        float ScoreTarget(Ship e)
        {
            float d = _s.DistanceTo(e);
            var mb = _s.Stats.mainBattery;
            float score = 100f;

            // in-range and closer is better, but do not obsess over the nearest
            score -= d / Mathf.Max(1f, mb.range) * 45f;

            // class preference
            score += ClassPriority(_s.Stats.classType, e.Stats.classType);

            // finish off cripples
            score += (1f - e.HealthFraction) * 35f;
            if (e.Damage.FireStacks > 0) score += 8f;

            // securing a kill is worth more than any other consideration: a ship that dies now
            // stops shooting back, and half-damaged ships heal
            if (e.Damage.Health <= EstimatedSalvoDamage(e) * 1.15f) score += 120f;

            // reward targets we can actually hurt
            if (mb.penetration < e.Stats.armor * 0.9f) score -= 40f;

            // prefer whoever is shooting at us
            if (_s.Damage.LastAttacker == e && _s.Damage.TimeSinceHit < 12f) score += 22f;

            // isolated ships are safer to engage
            int friendsNear = ShipRegistry.InRadius(e.Position, 220f, e.team).Count;
            score -= friendsNear * 6f;

            // a target that is bow-on to us is hard to damage
            float aspect = Mathf.Abs(Mathf.DeltaAngle(e.Heading, NavalMath.VectorToHeading(_s.Position - e.Position)));
            if (aspect < 35f || aspect > 145f) score -= 12f;

            // transports are the whole point in escort missions
            if (e.Stats.classType == ShipClassType.Transport) score += 60f;

            // Chasing a wounded ship away from the objective is worth far less than holding the
            // objective (utility 30 against 90/95). A cripple running from the point is not worth
            // following; a healthy one sitting on the point is.
            if (AssignedZone != null)
            {
                float targetToZone = Vector2.Distance(e.Position, AssignedZone.Position);
                if (targetToZone < AssignedZone.radius * 1.2f)
                    score += BattleAssessment.UtilityResetEnemyCapTimer * 0.5f;      // he is on our point
                else if (targetToZone > AssignedZone.radius * 2.5f)
                    score -= (BattleAssessment.UtilityEnterEmptyFlagZone - BattleAssessment.UtilityPursueLowHpAwayFromFlag) * 0.4f;
            }

            return score;
        }

        static float ClassPriority(ShipClassType me, ShipClassType them)
        {
            switch (me)
            {
                case ShipClassType.Destroyer:
                    return them == ShipClassType.Destroyer ? 30f : them == ShipClassType.Submarine ? 25f
                         : them == ShipClassType.Cruiser ? 5f : -25f;      // avoid trading shells with a battleship
                case ShipClassType.Cruiser:
                    return them == ShipClassType.Destroyer ? 30f : them == ShipClassType.Cruiser ? 20f
                         : them == ShipClassType.Submarine ? 10f : -10f;
                case ShipClassType.Battleship:
                    return them == ShipClassType.Battleship ? 30f : them == ShipClassType.Cruiser ? 25f
                         : them == ShipClassType.Destroyer ? 10f : 0f;
                case ShipClassType.Submarine:
                    return them == ShipClassType.Battleship ? 40f : them == ShipClassType.Cruiser ? 25f
                         : them == ShipClassType.Destroyer ? -30f : 10f;
                default:
                    return 0f;
            }
        }

        // ------------------------------------------------------------------ enemy brain

        void Think()
        {
            var nav = _s.Navigation;

            if (_s.Damage.IsDisabled)
            {
                SetState(AIState.Disabled, "engines wrecked");
                nav.SpeedScale = 1f;
                return;
            }

            // ---- survival ---------------------------------------------------
            if (ShouldRetreat())
            {
                Retreat();
                return;
            }

            if (_s.Resources.NeedsResupply || _s.Damage.NeedsPort)
            {
                var port = WorldMap.I != null ? WorldMap.I.NearestPort(_s.Position, _s.team) : null;
                if (port != null)
                {
                    float d = Vector2.Distance(_s.Position, port.Position);
                    if (d < port.serviceRadius)
                    {
                        SetState(AIState.Repairing, "docked");
                        nav.OrderHold();
                        port.ServiceShip(_s, Time.deltaTime * 4f);
                        return;
                    }
                    if (nav.Order != OrderType.ReturnToPort)
                        nav.OrderReturnToPort(port.Position);
                    SetState(AIState.Moving, "returning to port");
                    return;
                }
            }

            // ---- submarines have their own playbook -------------------------
            if (_s.Submarine != null) { SubmarineThink(); return; }

            // ---- the objective comes before the fight -----------------------
            if (ObjectiveThink()) return;

            var target = _s.CurrentTarget;
            if (target == null)
            {
                // nothing in sight: hold the station the commander gave us, or sweep
                if (HasStation)
                {
                    float d = Vector2.Distance(_s.Position, StationPoint);
                    if (d > 90f)
                    {
                        if (nav.Order != OrderType.AttackMove || Vector2.Distance(nav.CurrentDestination, StationPoint) > 120f)
                            nav.OrderMove(StationPoint, false, OrderType.AttackMove);
                        SetState(AIState.Searching, "moving to station");
                    }
                    else
                    {
                        SetState(AIState.Patrolling, "on station");
                        if (nav.Order == OrderType.None) PatrolAround(StationPoint, 130f);
                    }
                }
                else SetState(AIState.Idle, "no contacts");

                nav.SpeedScale = 0.8f;
                return;
            }

            switch (_s.Stats.classType)
            {
                case ShipClassType.Destroyer: DestroyerThink(target); break;
                case ShipClassType.Cruiser: CruiserThink(target); break;
                case ShipClassType.Battleship: BattleshipThink(target); break;
                default: TransportThink(target); break;
            }
        }

        bool ShouldRetreat()
        {
            float h = _s.Damage.HealthFraction;
            if (h < 0.22f) return true;
            if (h < 0.4f && _s.Damage.FloodingStacks > 0) return true;
            if (h < 0.5f && _s.Damage.RecentDamageTaken > _s.Stats.maxHealth * 0.18f) return true;

            // badly outnumbered where we are standing, and not required to die on a point
            if (_intel != null && h < 0.75f && LocalRatio < _intel.DisengageRatio * 0.62f && !MustHoldGround)
                return true;

            return false;
        }

        /// <summary>A point that decides the match is worth dying on; an ordinary station is not.</summary>
        bool MustHoldGround
        {
            get
            {
                if (AssignedZone == null || _intel == null) return false;
                if (Assignment != AIAssignment.HoldCap && Assignment != AIAssignment.ContestCap) return false;
                for (int i = 0; i < _intel.Zones.Length; i++)
                    if (_intel.Zones[i].zone == AssignedZone) return _intel.Zones[i].mustHold;
                return false;
            }
        }

        /// <summary>
        /// Withdrawal. A damaged ship falls back behind the nearest friendly concentration and heals
        /// there - sailing all the way home would take it out of the battle entirely. Only a ship
        /// that is genuinely wrecked, dry or empty actually runs for port.
        /// </summary>
        void Retreat()
        {
            var nav = _s.Navigation;
            nav.SpeedScale = 1f;

            if (_smokeTimer <= 0f && _s.Abilities != null && _s.Abilities.Use(AbilityId.SmokeScreen))
                _smokeTimer = 30f;

            // battleships heal on the way out rather than after they arrive
            if (_s.Abilities != null) _s.Abilities.Use(AbilityId.RepairParty);

            bool needsYard = _s.Damage.HealthFraction < 0.18f || _s.Resources.NeedsResupply;
            Vector2 goal;

            if (needsYard && WorldMap.I != null && WorldMap.I.NearestPort(_s.Position, _s.team) != null)
            {
                goal = WorldMap.I.NearestPort(_s.Position, _s.team).Position;
                SetState(AIState.Retreating, "running for the yard");
            }
            else if (TryRallyPoint(out Vector2 rally))
            {
                goal = rally;
                SetState(AIState.Retreating, "falling back on support");
            }
            else
            {
                Vector2 away = Vector2.zero;
                for (int i = 0; i < _visibleEnemies.Count; i++)
                    away += (_s.Position - _visibleEnemies[i].Position).normalized;
                if (away.sqrMagnitude < 0.01f) away = -_s.Forward;
                goal = _s.Position + away.normalized * 500f;
                SetState(AIState.Retreating, "breaking contact");
            }

            // put an island between us and the guns if there is one to use
            if (_intel != null && _intel.UsesCover && BiggestThreat(out Ship threat) &&
                TryBreakLineOfSight(threat, out Vector2 cover))
                goal = cover;

            if (nav.Order != OrderType.Retreat || Vector2.Distance(nav.CurrentDestination, goal) > 200f)
                nav.OrderRetreat(WorldMap.I != null ? WorldMap.I.Clamp(goal) : goal);

            // destroyers cover their withdrawal with torpedoes
            if (_s.Weapons.TorpedoesReady && _s.CurrentTarget != null)
                _s.Weapons.LaunchTorpedoesAtTarget(_s.CurrentTarget);
        }

        /// <summary>Centre of the nearest group of healthier friends to fall back on.</summary>
        bool TryRallyPoint(out Vector2 rally)
        {
            rally = Vector2.zero;
            var friends = ShipRegistry.OfTeam(_s.team);
            Vector2 sum = Vector2.zero;
            int n = 0;
            for (int i = 0; i < friends.Count; i++)
            {
                var f = friends[i];
                if (f == null || f == _s || f.IsDead || f.Damage.IsSinking) continue;
                if (f.HealthFraction < _s.HealthFraction + 0.1f) continue;      // no point hiding behind a wreck
                if (_s.DistanceTo(f) > 900f) continue;
                sum += f.Position;
                n++;
            }
            if (n == 0) return false;

            Vector2 center = sum / n;
            // stop just short of them, on our side, rather than sailing into the middle of the group
            Vector2 dir = (center - _s.Position).normalized;
            rally = center - dir * 60f;
            return true;
        }

        bool BiggestThreat(out Ship threat)
        {
            threat = null;
            float best = 0f;
            for (int i = 0; i < _visibleEnemies.Count; i++)
            {
                var e = _visibleEnemies[i];
                if (e.Stats.mainBattery == null) continue;
                if (_s.DistanceTo(e) > e.Stats.mainBattery.range * 1.1f) continue;
                float t = e.Stats.mainBattery.damage * e.Stats.mainBattery.turrets;
                if (t > best) { best = t; threat = e; }
            }
            if (threat == null && _s.Damage.TimeSinceHit < 8f) threat = _s.Damage.LastAttacker;
            return threat != null;
        }

        /// <summary>
        /// Looks for somewhere nearby that an island would hide us from a particular enemy. Reuses the
        /// same line-of-sight test the detection system uses, so cover that works here really does
        /// break spotting.
        /// </summary>
        bool TryBreakLineOfSight(Ship threat, out Vector2 point)
        {
            point = Vector2.zero;
            if (threat == null) return false;

            // do not thrash: keep a cover point for a few seconds once we have one
            if (_coverTimer > 0f) { point = _coverPoint; return true; }

            Vector2 away = (_s.Position - threat.Position).normalized;
            for (int i = 0; i < 7; i++)
            {
                float spread = (i - 3) * 22f;
                Vector2 dir = NavalMath.Rotate(away, spread);
                for (float d = 110f; d <= 260f; d += 75f)
                {
                    Vector2 candidate = _s.Position + dir * d;
                    if (WorldMap.I != null) candidate = WorldMap.I.Clamp(candidate);
                    if (NavGrid.I != null && !NavGrid.I.PassableWorld(candidate, _s.Stats.draft)) continue;
                    if (DetectionSystem.HasLineOfSight(candidate, threat.Position, true)) continue;

                    _coverPoint = candidate;
                    _coverTimer = 6f;
                    point = candidate;
                    return true;
                }
            }
            return false;
        }

        /// <summary>
        /// Decapping: an enemy is on a point and the meter is running. Getting a hull inside the ring
        /// stops the count immediately, so that comes first - but the real job is killing or driving
        /// off the capper, and a capper sitting in his own smoke has to be flanked to be seen at all.
        /// </summary>
        bool DecapThink()
        {
            var z = AssignedZone;
            if (z == null) return false;

            var nav = _s.Navigation;
            float dist = Vector2.Distance(_s.Position, z.Position);
            bool inside = dist < z.radius * 0.9f;

            // HE resets modules and starts fires: far more reliable for breaking up a cap than
            // hoping for an armour-piercing citadel on a nimble destroyer
            if (_s.Abilities != null && _s.Abilities.Has(AbilityId.ShellHE))
                _s.Abilities.Use(AbilityId.ShellHE);

            // who is actually taking it from us?
            Ship capper = null;
            float bestD = z.radius * 1.3f;
            var foes = ShipRegistry.OfTeam(Teams.Opponent(_s.team));
            for (int i = 0; i < foes.Count; i++)
            {
                var e = foes[i];
                if (e == null || e.IsDead || e.Damage.IsSinking) continue;
                float d = Vector2.Distance(e.Position, z.Position);
                if (d < bestD) { bestD = d; capper = e; }
            }

            if (capper != null)
            {
                ManualTarget = capper;
                _s.CurrentTarget = capper;

                // if we cannot see him he is probably sitting in smoke: swing around the perimeter
                // to open a fresh line of sight rather than staring into the cloud
                bool canSee = DetectionSystem.I == null || DetectionSystem.I.IsVisible(capper, _s.team);
                if (!canSee && _intel != null && _intel.UsesInference)
                {
                    SetState(AIState.Searching, "flanking the smoke on " + z.zoneName);
                    Vector2 radial = (_s.Position - capper.Position).normalized;
                    Vector2 flank = new Vector2(-radial.y, radial.x) * ((_s.id % 2 == 0) ? 1f : -1f);
                    Vector2 around = capper.Position + (radial + flank).normalized * z.radius * 0.85f;
                    nav.SpeedScale = 1f;
                    nav.SteerDirect(WorldMap.I != null ? WorldMap.I.Clamp(around) : around);
                    return true;
                }
            }

            if (!inside)
            {
                // contesting stops the meter the instant we are in the circle
                SetState(AIState.Tracking, "resetting " + z.zoneName);
                nav.SpeedScale = 1f;
                Vector2 entry = z.Position + (_s.Position - z.Position).normalized * z.radius * 0.5f;
                nav.SteerDirect(WorldMap.I != null ? WorldMap.I.Clamp(entry) : entry);
                return true;
            }

            SetState(AIState.Attacking, "denying " + z.zoneName);
            float angle = NavalMath.VectorToHeading(_s.Position - z.Position) + 50f;
            Vector2 orbit = z.Position + NavalMath.HeadingToVector(angle) * z.radius * 0.6f;
            nav.SpeedScale = 0.8f;
            nav.SteerDirect(WorldMap.I != null ? WorldMap.I.Clamp(orbit) : orbit, 0.8f);
            return true;
        }

        /// <summary>Rough damage one of our salvos would do to a target, for kill securing.</summary>
        float EstimatedSalvoDamage(Ship target)
        {
            var mb = _s.Stats.mainBattery;
            if (mb == null) return 0f;
            float barrels = mb.turrets * mb.barrelsPerTurret * 0.5f;
            float hitRate = Mathf.Lerp(0.45f, 0.15f, Mathf.Clamp01(_s.DistanceTo(target) / Mathf.Max(1f, mb.range)));
            float pen = mb.penetration >= target.Stats.armor ? 1f : 0.35f;
            return mb.damage * barrels * hitRate * pen;
        }

        /// <summary>
        /// Cap play. A ship told to take or hold a point has to be inside the ring - this is what
        /// converts a contest into a capture. It orbits inside rather than parking, and gives the
        /// point up only if the local fight has become hopeless.
        /// </summary>
        bool ObjectiveThink()
        {
            if (Assignment == AIAssignment.Decap) return DecapThink();
            if (Assignment != AIAssignment.HoldCap && Assignment != AIAssignment.ContestCap) return false;
            var z = AssignedZone;
            if (z == null) return false;

            // The point is taken and nobody is trying to take it back, so there is nothing to be
            // gained by circling it. Release the ship to fight, screen or push the next objective.
            if (z.FullyCaptured && z.Owner == _s.team && !z.UnderAttack) return false;

            float disengage = _intel != null ? _intel.DisengageRatio : 0.6f;
            if (LocalRatio < disengage && _s.Damage.HealthFraction < 0.65f && !MustHoldGround)
                return false;                       // fall through to normal tactics, probably a retreat

            var nav = _s.Navigation;
            float dist = Vector2.Distance(_s.Position, z.Position);
            bool inside = dist < z.radius * 0.88f;

            if (!inside)
            {
                SetState(AIState.Tracking, "moving onto " + z.zoneName);
                nav.SpeedScale = 1f;
                Vector2 entry = z.Position + (_s.Position - z.Position).normalized * z.radius * 0.45f;
                nav.SteerDirect(WorldMap.I != null ? WorldMap.I.Clamp(entry) : entry);
                return true;
            }

            // inside the ring: keep moving so we are not an easy target, but never leave it
            SetState(z.Contested ? AIState.Attacking : AIState.Patrolling,
                     z.Contested ? "contesting " + z.zoneName : "holding " + z.zoneName);

            float angle = NavalMath.VectorToHeading(_s.Position - z.Position) + 55f;
            Vector2 orbit = z.Position + NavalMath.HeadingToVector(angle) * z.radius * 0.55f;
            nav.SpeedScale = 0.75f;
            nav.SteerDirect(WorldMap.I != null ? WorldMap.I.Clamp(orbit) : orbit, 0.75f);
            return true;
        }

        void PatrolAround(Vector2 center, float radius)
        {
            var pts = new List<Vector2>();
            for (int i = 0; i < 4; i++)
            {
                float a = i / 4f * Mathf.PI * 2f + Random.Range(0f, 1f);
                Vector2 p = center + new Vector2(Mathf.Cos(a), Mathf.Sin(a)) * radius;
                if (NavGrid.I != null) p = NavGrid.I.NearestNavigable(p, _s.Stats.draft);
                pts.Add(p);
            }
            _s.Navigation.OrderPatrol(pts);
        }

        // ------------------------------------------------------------------ class tactics

        void DestroyerThink(Ship target)
        {
            var nav = _s.Navigation;
            var mb = _s.Stats.mainBattery;
            var td = _s.Stats.torpedoes;
            float dist = _s.DistanceTo(target);
            bool heavy = target.Stats.classType == ShipClassType.Battleship || target.Stats.classType == ShipClassType.Cruiser;

            // hunting a submarine contact takes priority for a destroyer
            var sub = FindSubmarineContact();
            if (sub != null)
            {
                SetState(AIState.Tracking, "prosecuting submarine");
                nav.SpeedScale = 1f;
                Vector2 lead = sub.Position + sub.Velocity * 2f;
                nav.SteerDirect(lead, 1f, Vector2.Distance(_s.Position, lead) < 60f);
                return;
            }

            if (_postAttackTimer > 0f)
            {
                // disengage after a torpedo run
                SetState(AIState.Evading, "breaking off");
                Vector2 away = _s.Position + (_s.Position - target.Position).normalized * 300f;
                nav.SteerDirect(WorldMap.I.Clamp(away));
                if (dist < 260f && _s.Abilities != null) _s.Abilities.Use(AbilityId.SmokeScreen);
                return;
            }

            if (heavy && td != null && _s.Resources.TorpedoAmmo > 0)
            {
                // work into a flanking position for a torpedo attack
                float attackRange = td.range * 0.75f;
                SetState(AIState.Tracking, "torpedo run");
                nav.SpeedScale = 1f;

                if (_s.Weapons.LaunchTorpedoesAtTarget(target))
                {
                    _postAttackTimer = 14f;
                    return;
                }

                Vector2 toTarget = (target.Position - _s.Position).normalized;
                Vector2 flank = new Vector2(-toTarget.y, toTarget.x) * (((_s.id % 2) == 0) ? 1f : -1f);
                Vector2 approach = target.Position - toTarget * attackRange + flank * attackRange * 0.55f;
                nav.SteerDirect(WorldMap.I.Clamp(approach));

                // keep the guns busy only if we are not trying to stay hidden
                _s.Weapons.HoldFire = dist > mb.range * 0.85f || (!_s.Detection.SpottedByEnemy && dist > td.range);
                return;
            }

            // gun duel with a light target: fight at our maximum range
            SetState(AIState.Attacking, "gun action");
            _s.Weapons.HoldFire = false;
            float desired = mb.range * 0.8f;
            KeepRange(target, desired, 1f);

            if (_s.Detection.SpottedByEnemy && _s.Damage.HealthFraction < 0.6f && _s.Abilities != null)
                _s.Abilities.Use(AbilityId.SmokeScreen);

            if (td != null && dist < td.range * 0.8f)
                _s.Weapons.LaunchTorpedoesAtTarget(target);
        }

        void CruiserThink(Ship target)
        {
            var nav = _s.Navigation;
            var mb = _s.Stats.mainBattery;
            float dist = _s.DistanceTo(target);
            SetState(AIState.Attacking, "engaging");
            _s.Weapons.HoldFire = false;

            bool vsBattleship = target.Stats.classType == ShipClassType.Battleship;
            float desired = vsBattleship ? mb.range * 0.95f : mb.range * 0.72f;

            // support: do not stray too far from friends
            var friends = ShipRegistry.InRadius(_s.Position, 420f, _s.team);
            if (friends.Count <= 1 && HasStation && Vector2.Distance(_s.Position, StationPoint) > 350f)
            {
                SetState(AIState.Moving, "rejoining the fleet");
                nav.SteerDirect(StationPoint);
                return;
            }

            KeepRange(target, desired, vsBattleship ? 1f : 0.9f);

            var td = _s.Stats.torpedoes;
            if (td != null && dist < td.range * 0.7f)
                _s.Weapons.LaunchTorpedoesAtTarget(target);
        }

        void BattleshipThink(Ship target)
        {
            var mb = _s.Stats.mainBattery;
            float dist = _s.DistanceTo(target);
            SetState(AIState.Attacking, "main battery in action");
            _s.Weapons.HoldFire = false;

            // stay at long range where our armour and guns dominate
            float desired = mb.range * 0.72f;
            bool destroyersClose = false;
            for (int i = 0; i < _visibleEnemies.Count; i++)
            {
                var e = _visibleEnemies[i];
                if (e.Stats.classType == ShipClassType.Destroyer && _s.DistanceTo(e) < 220f) { destroyersClose = true; break; }
            }
            if (destroyersClose) desired = Mathf.Max(desired, mb.range * 0.85f);

            KeepRange(target, desired, 0.85f, 32f);
        }

        void TransportThink(Ship target)
        {
            SetState(AIState.Retreating, "unarmed");
            var nav = _s.Navigation;
            if (target != null && _s.DistanceTo(target) < 350f)
            {
                Vector2 away = _s.Position + (_s.Position - target.Position).normalized * 400f;
                nav.SteerDirect(WorldMap.I.Clamp(away));
            }
        }

        /// <summary>
        /// Hold a stand off distance while keeping the broadside pointed at the enemy - the core of
        /// naval positioning. angleOffset lets battleships angle their belt armour.
        /// </summary>
        void KeepRange(Ship target, float desired, float speedScale, float angleOffset = 0f)
        {
            var nav = _s.Navigation;

            // Posture and the local balance of force decide how close we are willing to fight.
            // Winning locally we press in, losing locally we open the range and buy time.
            float boldness = Aggression * Mathf.Clamp(LocalRatio, 0.55f, 1.6f);
            desired /= Mathf.Clamp(boldness, 0.6f, 1.5f);

            nav.SpeedScale = speedScale;

            Vector2 toMe = (_s.Position - target.Position).normalized;
            float dist = _s.DistanceTo(target);

            // Guns now out-range eyes by a wide margin - a battleship can shell 26 km but only sees
            // 14 km - so standing off at maximum gun range would mean never seeing anything to shoot
            // at. Hold a range we can spot from ourselves, unless someone else is already holding
            // the contact for us, in which case take the free reach.
            float spotLimit = Mathf.Min(target.Detectability, _s.Detection.EffectiveSpotRange);
            bool spottedByOthers = dist > spotLimit;
            if (!spottedByOthers) desired = Mathf.Min(desired, spotLimit * 0.95f);

            Vector2 goal;
            if (dist < desired * 0.8f)          // too close, open the range
                goal = target.Position + toMe * desired * 1.25f;
            else if (dist > desired * 1.15f)    // too far, close in
                goal = target.Position + toMe * desired * 0.9f;
            else
            {
                // in the sweet spot: circle the target so the guns keep bearing
                Vector2 tangent = new Vector2(-toMe.y, toMe.x) * (((_s.id % 2) == 0) ? 1f : -1f);
                goal = _s.Position + tangent * 120f + toMe * (dist - desired) * 0.5f;
            }

            if (angleOffset > 0f)
            {
                Vector2 dir = (goal - _s.Position).normalized;
                goal = _s.Position + NavalMath.Rotate(dir, angleOffset * (((_s.id % 2) == 0) ? 1f : -1f)) * 150f;
            }

            goal = ApplySupportDiscipline(goal);
            goal = ApplyTorpedoParanoia(goal);

            if (WorldMap.I != null) goal = WorldMap.I.Clamp(goal);
            nav.SteerDirect(goal, speedScale);
        }

        /// <summary>
        /// When a destroyer contact goes dark inside torpedo reach, the water it vanished into is
        /// probably full of fish. Rather than steaming a predictable straight line, the ship weaves -
        /// which is exactly what a human does when they lose a destroyer on the plot.
        /// </summary>
        Vector2 ApplyTorpedoParanoia(Vector2 goal)
        {
            if (_intel == null || !_intel.UsesInference) return goal;
            if (!_intel.DarkThreatNear(_s.Position, 420f, ShipClassType.Destroyer) &&
                !_intel.DarkThreatNear(_s.Position, 300f, ShipClassType.Submarine)) return goal;

            Vector2 dir = goal - _s.Position;
            if (dir.sqrMagnitude < 1f) return goal;

            // a slow weave either side of the intended track, offset per ship so a division does
            // not zig in unison and collide
            float phase = Time.time * 0.35f + _s.id * 1.7f;
            float weave = Mathf.Sin(phase) * 28f;
            return _s.Position + NavalMath.Rotate(dir, weave);
        }

        /// <summary>
        /// Stops a ship pushing so far ahead of its fleet that it gets focused down alone - the
        /// classic way an AI battleship throws itself away. It may not advance more than a set
        /// distance past the nearest friendly heavy hull.
        /// </summary>
        Vector2 ApplySupportDiscipline(Vector2 goal)
        {
            if (Aggression >= 1.5f) return goal;                 // desperate: everything forward
            if (_s.Stats.classType == ShipClassType.Destroyer) return goal;   // screening is their job

            Ship anchor = null;
            float bd = float.MaxValue;
            var friends = ShipRegistry.OfTeam(_s.team);
            for (int i = 0; i < friends.Count; i++)
            {
                var f = friends[i];
                if (f == null || f == _s || f.IsDead || f.Damage.IsSinking) continue;
                if (f.Stats.classType == ShipClassType.Destroyer || f.Stats.classType == ShipClassType.Submarine) continue;
                float d = _s.DistanceTo(f);
                if (d < bd) { bd = d; anchor = f; }
            }
            if (anchor == null) return goal;

            float leash = 300f * Mathf.Clamp(Aggression, 0.6f, 1.4f);
            Vector2 fromAnchor = goal - anchor.Position;
            if (fromAnchor.magnitude <= leash) return goal;
            return anchor.Position + fromAnchor.normalized * leash;
        }

        Ship FindSubmarineContact()
        {
            if (_s.Stats.asw == null || _s.Resources.ASWAmmo <= 0) return null;
            var enemies = ShipRegistry.OfTeam(Teams.Opponent(_s.team));
            Ship best = null; float bd = _s.Detection.EffectiveSonarRange * 1.6f;
            for (int i = 0; i < enemies.Count; i++)
            {
                var e = enemies[i];
                if (e.Submarine == null || e.IsDead) continue;
                if (DetectionSystem.I != null && !DetectionSystem.I.IsVisible(e, _s.team)) continue;
                float d = _s.DistanceTo(e);
                if (d < bd) { bd = d; best = e; }
            }
            return best;
        }

        // ------------------------------------------------------------------ submarine brain

        void SubmarineThink()
        {
            var sub = _s.Submarine;
            var nav = _s.Navigation;
            var td = _s.Stats.torpedoes;

            // any escort close by? then go deep and quiet
            Ship hunter = null;
            float hunterDist = float.MaxValue;
            for (int i = 0; i < _visibleEnemies.Count; i++)
            {
                var e = _visibleEnemies[i];
                if (e.Stats.asw == null) continue;
                float d = _s.DistanceTo(e);
                if (d < hunterDist) { hunterDist = d; hunter = e; }
            }

            bool threatened = hunter != null && hunterDist < 220f;
            bool beingPinged = _s.Detection.SpottedByEnemy;

            if (threatened || (beingPinged && sub.IsSubmerged))
            {
                SetState(AIState.Evading, "escort in contact");
                sub.SetDepth(DepthState.Deep);
                sub.SilentRunning = true;
                nav.SpeedScale = 0.55f;
                if (hunter != null)
                {
                    Vector2 away = _s.Position + (_s.Position - hunter.Position).normalized * 350f;
                    nav.SteerDirect(WorldMap.I.Clamp(away));
                }
                return;
            }

            // pick the juiciest target we know about
            Ship prey = null; float preyScore = float.MinValue;
            var enemies = ShipRegistry.OfTeam(Teams.Opponent(_s.team));
            for (int i = 0; i < enemies.Count; i++)
            {
                var e = enemies[i];
                if (e == null || e.IsDead || e.Damage.IsSinking) continue;
                float d = _s.DistanceTo(e);
                if (d > _s.Detection.EffectiveHydroRange * 1.4f) continue;      // passive sonar reach
                float sc = ClassPriority(ShipClassType.Submarine, e.Stats.classType) - d * 0.05f;
                if (sc > preyScore) { preyScore = sc; prey = e; }
            }

            if (prey == null)
            {
                // no contacts: recharge on the surface if it is safe, otherwise sweep at periscope depth
                if (sub.BatteryFraction < 0.5f && !beingPinged)
                {
                    sub.SetDepth(DepthState.Surface);
                    SetState(AIState.Repairing, "charging batteries");
                }
                else
                {
                    sub.SetDepth(DepthState.Periscope);
                    SetState(AIState.Searching, "searching");
                }
                sub.SilentRunning = false;
                nav.SpeedScale = 0.7f;
                if (HasStation && Vector2.Distance(_s.Position, StationPoint) > 120f)
                    nav.SteerDirect(StationPoint, 0.8f, true);
                return;
            }

            float dist = _s.DistanceTo(prey);
            float attackRange = td != null ? td.range * 0.8f : 200f;

            if (dist > attackRange)
            {
                SetState(AIState.Tracking, "closing on " + prey.ClassTag);
                sub.SetDepth(sub.BatteryFraction > 0.35f ? DepthState.Submerged : DepthState.Periscope);
                sub.SilentRunning = false;
                nav.SpeedScale = 1f;
                Vector2 intercept = prey.Position + prey.Velocity * (dist / Mathf.Max(0.5f, _s.Stats.maxSpeed)) * 0.5f;
                nav.SteerDirect(WorldMap.I.Clamp(intercept));
                return;
            }

            // in position: come to periscope depth and shoot
            SetState(AIState.Attacking, "attacking " + prey.ClassTag);
            sub.SetDepth(DepthState.Periscope);
            sub.SilentRunning = true;
            nav.SpeedScale = 0.5f;

            if (_s.Weapons.TorpedoesReady && sub.Depth == DepthState.Periscope)
            {
                if (_s.Weapons.LaunchTorpedoesAtTarget(prey))
                {
                    _postAttackTimer = 25f;
                    sub.SetDepth(DepthState.Deep);
                    SetState(AIState.Evading, "torpedoes away, going deep");
                    return;
                }
            }

            // manoeuvre onto a firing bearing (torpedo tubes want the target on the beam)
            Vector2 toPrey = (prey.Position - _s.Position).normalized;
            Vector2 firingPos = prey.Position - toPrey * attackRange * 0.7f
                                + new Vector2(-toPrey.y, toPrey.x) * attackRange * 0.35f;
            nav.SteerDirect(WorldMap.I.Clamp(firingPos), 0.6f);
        }
    }
}
