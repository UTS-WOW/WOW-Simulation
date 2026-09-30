using System.Collections.Generic;
using UnityEngine;

namespace Naval
{
    /// <summary>
    /// Turns orders into steering. Owns the waypoint queue, the computed route around land,
    /// terrain probing, ship-to-ship collision avoidance and evasive manoeuvres.
    /// </summary>
    public class ShipNavigation
    {
        readonly Ship _s;

        public OrderType Order { get; private set; } = OrderType.None;
        public Vector2 OrderPoint { get; private set; }
        public Ship OrderTarget { get; private set; }

        public readonly List<Vector2> Waypoints = new List<Vector2>();
        public List<Vector2> Path { get; private set; }
        public readonly List<Vector2> PatrolPoints = new List<Vector2>();
        int _patrolIndex;
        bool _patrolForward = true;

        public Ship FormationLeader;
        public Vector2 FormationOffset;

        public float SpeedScale = 1f;          // AI throttling (silent running, careful approach...)
        public bool HoldingPosition { get; private set; }

        object _pathHandle;
        Vector2 _pathGoal;
        float _repathTimer;
        float _stuckTimer;
        Vector2 _lastPos;
        float _progressTimer;

        float _evadeTimer;
        float _evadeHeading;
        public bool IsEvading => _evadeTimer > 0f;

        // AI supplied steering: overrides the standing order while it is fresh
        Vector2 _directPoint;
        float _directTimer;
        float _directCap = 1f;
        bool _directFinal;
        public bool IsDirectSteering => _directTimer > 0f;
        /// <summary>Where the current direct-steering leg is heading (valid while IsDirectSteering).</summary>
        public Vector2 DirectPoint => _directPoint;

        public float LastProbeAhead { get; private set; }
        public Vector2 SteerDebug { get; private set; }

        public ShipNavigation(Ship s)
        {
            _s = s;
            _lastPos = s.Position;
        }

        public bool HasDestination => Order == OrderType.Move || Order == OrderType.AttackMove ||
                                      Order == OrderType.Retreat || Order == OrderType.Patrol ||
                                      Order == OrderType.Follow || Order == OrderType.ReturnToPort;

        public Vector2 CurrentDestination
        {
            get
            {
                if (Waypoints.Count > 0) return Waypoints[0];
                if (Order == OrderType.Patrol && PatrolPoints.Count > 0) return PatrolPoints[_patrolIndex];
                return _s.Position;
            }
        }

        // ------------------------------------------------------------------ orders

        public void OrderMove(Vector2 point, bool queue = false, OrderType type = OrderType.Move)
        {
            _directTimer = 0f;
            if (!queue) { Waypoints.Clear(); Path = null; }
            Order = type;
            HoldingPosition = false;
            OrderTarget = null;
            Waypoints.Add(WorldMap.I != null ? WorldMap.I.Clamp(point) : point);
            OrderPoint = Waypoints[0];
            _repathTimer = 0f;
            FormationLeader = null;
        }

        public void OrderAttack(Ship target)
        {
            _directTimer = 0f;
            Order = OrderType.Attack;
            OrderTarget = target;
            HoldingPosition = false;
            Waypoints.Clear();
            Path = null;
            FormationLeader = null;
        }

        public void OrderPatrol(List<Vector2> points)
        {
            _directTimer = 0f;
            PatrolPoints.Clear();
            PatrolPoints.AddRange(points);
            _patrolIndex = 0;
            _patrolForward = true;
            Order = OrderType.Patrol;
            HoldingPosition = false;
            Waypoints.Clear();
            Path = null;
        }

        public void OrderFollow(Ship leader, Vector2 offset)
        {
            _directTimer = 0f;
            Order = OrderType.Follow;
            FormationLeader = leader;
            FormationOffset = offset;
            HoldingPosition = false;
            Waypoints.Clear();
            Path = null;
        }

        public void OrderStop()
        {
            _directTimer = 0f;
            Order = OrderType.Stop;
            Waypoints.Clear();
            PatrolPoints.Clear();
            Path = null;
            OrderTarget = null;
            FormationLeader = null;
            HoldingPosition = false;
        }

        public void OrderHold()
        {
            OrderStop();
            Order = OrderType.HoldPosition;
            OrderPoint = _s.Position;
            HoldingPosition = true;
        }

        public void OrderReverse()
        {
            _directTimer = 0f;
            Order = OrderType.Reverse;
            Waypoints.Clear();
            Path = null;
            HoldingPosition = false;
        }

        public void OrderRetreat(Vector2 point)
        {
            OrderMove(point, false, OrderType.Retreat);
        }

        public void OrderReturnToPort(Vector2 port)
        {
            OrderMove(port, false, OrderType.ReturnToPort);
        }

        public void ClearWaypointsKeepOrder()
        {
            Waypoints.Clear();
            Path = null;
        }

        /// <summary>
        /// Tactical steering from the AI. The AI thinks a few times a second, so the point is held
        /// and re-followed (with avoidance) every frame until the next decision arrives.
        /// It takes priority over the standing order while it is fresh.
        /// </summary>
        public void SteerDirect(Vector2 point, float throttleCap = 1f, bool isFinal = false, float hold = 0.5f)
        {
            _directPoint = point;
            _directCap = throttleCap;
            _directFinal = isFinal;
            _directTimer = hold;
        }

        public void CancelDirectSteering() => _directTimer = 0f;

        /// <summary>Called when a torpedo is spotted - turn to comb the wakes for a few seconds.</summary>
        public void Evade(Vector2 threatOrigin, float duration = 5.5f)
        {
            float toThreat = NavalMath.VectorToHeading(threatOrigin - _s.Position);
            // turning bow or stern on presents the smallest target
            float bow = Mathf.Abs(Mathf.DeltaAngle(_s.Heading, toThreat));
            _evadeHeading = bow < 90f ? toThreat : NavalMath.Wrap360(toThreat + 180f);
            _evadeTimer = Mathf.Max(_evadeTimer, duration);
        }

        // ------------------------------------------------------------------ tick

        public void Tick(float dt)
        {
            var move = _s.Movement;

            if (_evadeTimer > 0f)
            {
                _evadeTimer -= dt;
                move.SteerToHeading(ApplyAvoidance(_evadeHeading, dt));
                move.SetThrottle(1f);
                return;
            }

            if (_directTimer > 0f)
            {
                _directTimer -= dt;
                SteerTowards(_directPoint, dt, _directFinal, _directCap);
                return;
            }

            switch (Order)
            {
                case OrderType.Stop:
                    // Cut the engines but leave the helm alone: a stopped ship keeps whatever course
                    // was last commanded, which is what lets the AI angle its armour while parked.
                    move.SetThrottle(0f);
                    return;

                case OrderType.HoldPosition:
                    HoldStation(dt);
                    return;

                case OrderType.Reverse:
                    move.SetThrottle(-1f);
                    move.SteerToHeading(_s.Heading);
                    return;

                case OrderType.Attack:
                    if (OrderTarget == null || OrderTarget.IsDead) { OrderStop(); return; }
                    AttackStationKeeping(dt);
                    return;

                case OrderType.Follow:
                    FollowLeader(dt);
                    return;

                case OrderType.Patrol:
                    PatrolTick(dt);
                    return;

                case OrderType.None:
                    move.SetThrottle(0f);
                    return;
            }

            // Move / AttackMove / Retreat / ReturnToPort ------------------------
            if (Waypoints.Count == 0)
            {
                move.SetThrottle(0f);
                Order = OrderType.None;
                return;
            }

            Vector2 dest = Waypoints[0];
            float arrival = ArrivalRadius;
            if (Vector2.Distance(_s.Position, dest) < arrival)
            {
                Waypoints.RemoveAt(0);
                Path = null;
                if (Waypoints.Count == 0)
                {
                    Order = OrderType.None;
                    move.SetThrottle(0f);
                    return;
                }
            }

            FollowRoute(Waypoints[0], dt);
        }

        float ArrivalRadius => Mathf.Max(14f, _s.Stats.length * 1.2f);

        /// <summary>
        /// Attack orders do not mean "ram the target": close to a good gunnery range and then hold it,
        /// circling so the broadside keeps bearing.
        /// </summary>
        void AttackStationKeeping(float dt)
        {
            var mb = _s.Stats.mainBattery;
            float desired = mb != null ? mb.range * 0.72f : 120f;
            if (_s.Stats.torpedoes != null && mb == null) desired = _s.Stats.torpedoes.range * 0.7f;

            Vector2 toMe = (_s.Position - OrderTarget.Position);
            float dist = toMe.magnitude;
            if (dist < 1f) { _s.Movement.SetThrottle(0.3f); return; }
            toMe /= dist;

            Vector2 goal;
            if (dist > desired * 1.12f) goal = OrderTarget.Position + toMe * desired;
            else if (dist < desired * 0.8f) goal = OrderTarget.Position + toMe * desired * 1.2f;
            else
            {
                Vector2 tangent = new Vector2(-toMe.y, toMe.x) * ((_s.id % 2 == 0) ? 1f : -1f);
                goal = _s.Position + tangent * 110f;
            }

            if (WorldMap.I != null) goal = WorldMap.I.Clamp(goal);
            SteerTowards(goal, dt, false);
        }

        void HoldStation(float dt)
        {
            float d = Vector2.Distance(_s.Position, OrderPoint);
            if (d < ArrivalRadius * 1.4f)
            {
                // on station: engines stopped, helm free so the ship can angle toward the threat
                _s.Movement.SetThrottle(0f);
            }
            else SteerTowards(OrderPoint, dt, true);
        }

        void PatrolTick(float dt)
        {
            if (PatrolPoints.Count == 0) { Order = OrderType.None; return; }
            Vector2 p = PatrolPoints[_patrolIndex];
            if (Vector2.Distance(_s.Position, p) < ArrivalRadius * 1.5f)
            {
                Path = null;
                if (PatrolPoints.Count == 1) return;
                if (_patrolForward)
                {
                    _patrolIndex++;
                    if (_patrolIndex >= PatrolPoints.Count) { _patrolIndex = PatrolPoints.Count - 2; _patrolForward = false; }
                }
                else
                {
                    _patrolIndex--;
                    if (_patrolIndex < 0) { _patrolIndex = Mathf.Min(1, PatrolPoints.Count - 1); _patrolForward = true; }
                }
                _patrolIndex = Mathf.Clamp(_patrolIndex, 0, PatrolPoints.Count - 1);
            }
            FollowRoute(PatrolPoints[_patrolIndex], dt);
        }

        void FollowLeader(float dt)
        {
            if (FormationLeader == null || FormationLeader.IsDead)
            {
                Order = OrderType.None;
                FormationLeader = null;
                return;
            }
            Vector2 station = FormationLeader.Position + NavalMath.Rotate(FormationOffset, -FormationLeader.Heading);
            float d = Vector2.Distance(_s.Position, station);
            if (d < ArrivalRadius * 0.8f)
            {
                // matched up: hold the leader's course and speed
                _s.Movement.SteerToHeading(ApplyAvoidance(FormationLeader.Heading, dt));
                _s.Movement.SetThrottle(Mathf.Clamp01(FormationLeader.Movement.Throttle) * SpeedScale);
                return;
            }
            SteerTowards(station, dt, true, Mathf.Clamp(d / 90f, 0.35f, 1f));
        }

        // ------------------------------------------------------------------ route following

        void FollowRoute(Vector2 dest, float dt)
        {
            var grid = NavGrid.I;
            _repathTimer -= dt;

            bool needPath = false;
            if (grid != null)
            {
                if (Path == null && _pathHandle == null) needPath = true;
                else if ((dest - _pathGoal).sqrMagnitude > 40f * 40f) needPath = true;
                else if (_repathTimer <= 0f && Path != null && Path.Count > 0) needPath = false;
            }

            // straight water? then no route is needed at all
            if (grid != null && grid.LineOfWater(_s.Position, dest, _s.Stats.draft))
            {
                Path = null;
                if (_pathHandle != null) { grid.CancelRequest(_pathHandle); _pathHandle = null; }
                SteerTowards(dest, dt, true);
                return;
            }

            if (needPath && grid != null)
            {
                _pathGoal = dest;
                if (_pathHandle != null) grid.CancelRequest(_pathHandle);
                _pathHandle = grid.RequestPath(_s.Position, dest, _s.Stats.draft, OnPathReady);
                _repathTimer = 4f;
            }

            if (Path != null && Path.Count > 0)
            {
                // consume reached legs
                while (Path.Count > 1 && Vector2.Distance(_s.Position, Path[0]) < Mathf.Max(20f, _s.Stats.length))
                    Path.RemoveAt(0);
                SteerTowards(Path[0], dt, Path.Count == 1);
            }
            else
            {
                SteerTowards(dest, dt, true);
            }

            // stuck detection: no ground made good for a while -> force a fresh route
            _progressTimer += dt;
            if (_progressTimer > 3f)
            {
                float moved = Vector2.Distance(_s.Position, _lastPos);
                _lastPos = _s.Position;
                _progressTimer = 0f;
                if (moved < Mathf.Max(6f, _s.Stats.maxSpeed * 0.6f) && !_s.Movement.Aground)
                {
                    _stuckTimer += 1f;
                    if (_stuckTimer >= 2f)
                    {
                        _stuckTimer = 0f;
                        Path = null;
                        _repathTimer = 0f;
                        if (_pathHandle != null && grid != null) { grid.CancelRequest(_pathHandle); _pathHandle = null; }
                    }
                }
                else _stuckTimer = 0f;
            }
        }

        void OnPathReady(List<Vector2> path)
        {
            _pathHandle = null;
            if (path != null && path.Count > 0) Path = path;
        }

        /// <summary>Steer at a point, easing the throttle down when it is the final destination.</summary>
        public void SteerTowards(Vector2 point, float dt, bool isFinal, float throttleCap = 1f)
        {
            Vector2 to = point - _s.Position;
            float dist = to.magnitude;
            float desired = NavalMath.VectorToHeading(to);
            desired = ApplyAvoidance(desired, dt);
            _s.Movement.SteerToHeading(desired);

            float throttle = throttleCap * SpeedScale;

            if (isFinal)
            {
                // start slowing at the distance we need to stop from current speed
                float stopDist = (_s.Movement.Speed * _s.Movement.Speed) / (2f * Mathf.Max(0.05f, _s.Stats.deceleration));
                if (dist < stopDist + ArrivalRadius)
                    throttle *= Mathf.Clamp01(dist / Mathf.Max(1f, stopDist + ArrivalRadius));
            }

            // do not charge ahead while the bow is still swinging onto the new course
            float off = Mathf.Abs(Mathf.DeltaAngle(_s.Heading, desired));
            if (off > 60f) throttle *= Mathf.Lerp(1f, 0.45f, Mathf.InverseLerp(60f, 150f, off));

            _s.Movement.SetThrottle(Mathf.Clamp(throttle, -1f, 1f));
        }

        // ------------------------------------------------------------------ avoidance

        float ApplyAvoidance(float desiredHeading, float dt)
        {
            float adjusted = desiredHeading;
            adjusted = AvoidTerrain(adjusted);
            adjusted = AvoidShips(adjusted);
            SteerDebug = NavalMath.HeadingToVector(adjusted);
            return adjusted;
        }

        float AvoidTerrain(float heading)
        {
            var grid = NavGrid.I;
            if (grid == null) return heading;

            float speed = Mathf.Max(1f, Mathf.Abs(_s.Movement.Speed));
            float look = Mathf.Clamp(speed * 9f + _s.Stats.length * 1.6f, 40f, 260f);

            float ahead = ProbeDistance(heading, look);
            LastProbeAhead = ahead;
            if (ahead >= look) return heading;

            // scan for the most open bearing, preferring small course changes
            float best = heading;
            float bestScore = ahead - 100f;
            for (int sign = -1; sign <= 1; sign += 2)
                for (int step = 1; step <= 6; step++)
                {
                    float h = NavalMath.Wrap360(heading + sign * step * 15f);
                    float d = ProbeDistance(h, look);
                    float score = d - step * 6f;
                    if (score > bestScore) { bestScore = score; best = h; }
                }
            return best;
        }

        float ProbeDistance(float heading, float maxDist)
        {
            var grid = NavGrid.I;
            Vector2 dir = NavalMath.HeadingToVector(heading);
            float step = Mathf.Max(8f, grid.CellSize * 0.5f);
            for (float d = step; d <= maxDist; d += step)
            {
                Vector2 p = _s.Position + dir * d;
                if (WorldMap.I != null && !WorldMap.I.InBounds(p)) return d;
                if (!grid.PassableWorld(p, _s.Stats.draft)) return d;
            }
            return maxDist;
        }

        float AvoidShips(float heading)
        {
            float myR = _s.Stats.length * 0.6f;
            float look = Mathf.Max(45f, Mathf.Abs(_s.Movement.Speed) * 8f + myR * 2f);
            var near = ShipRegistry.AllInRadius(_s.Position, look, _s);
            if (near.Count == 0) return heading;

            Vector2 dir = NavalMath.HeadingToVector(heading);
            float steer = 0f;

            for (int i = 0; i < near.Count; i++)
            {
                var o = near[i];
                if (o.Submarine != null && o.Submarine.Depth == DepthState.Deep) continue;

                Vector2 rel = o.Position - _s.Position;
                float dist = rel.magnitude;
                if (dist < 0.01f) continue;

                float front = Vector2.Dot(rel.normalized, dir);
                if (front < 0.15f) continue;                      // only care about what is ahead

                float safe = myR + o.Stats.length * 0.6f + 14f;
                if (dist > safe * 2.4f) continue;

                float side = Vector2.Dot(new Vector2(dir.y, -dir.x), rel.normalized);  // +1 = starboard
                float urgency = Mathf.Clamp01(1f - (dist - safe) / (safe * 2.4f));
                // give way to starboard, but if they are dead ahead pick the freer side
                float turnDir = side > 0f ? -1f : 1f;
                if (Mathf.Abs(side) < 0.12f) turnDir = 1f;
                steer += turnDir * urgency * 45f;
            }

            return NavalMath.Wrap360(heading + Mathf.Clamp(steer, -70f, 70f));
        }
    }
}
