using UnityEngine;

namespace Naval
{
    /// <summary>
    /// Naval movement on Rigidbody2D. Throttle and rudder are commands; the hull answers through
    /// thrust, keel grip and rudder torque, so ships carry way, drift through turns, need water
    /// flowing over the rudder to steer at all, and can go astern.
    ///
    /// CommandTick runs per frame (control logic), PhysicsTick runs in FixedUpdate (forces).
    /// </summary>
    public class ShipMovement
    {
        readonly Ship _s;
        readonly Rigidbody2D _rb;

        /// <summary>Commanded throttle, -1 (full astern) .. 1 (flank).</summary>
        public float Throttle { get; private set; }
        /// <summary>Commanded rudder, -1 (hard port) .. 1 (hard starboard).</summary>
        public float RudderCommand { get; private set; }
        /// <summary>Actual rudder deflection, lags behind the command.</summary>
        public float Rudder { get; private set; }

        public float TargetHeading { get; set; }
        public bool HeadingControl { get; set; } = true;

        public bool Aground { get; private set; }
        public float AgroundTimer { get; private set; }
        /// <summary>Pressing against land: hard aground, or stopped at the bank by shoaling water ahead
        /// (the terrain bleeds the speed off before most ships ever run hard aground). Holds 1 s.</summary>
        public bool AgainstLand => _landTimer > 0f;
        float _landTimer;

        // sinking presentation
        float _listAngle, _sinkDepth;

        public ShipMovement(Ship s)
        {
            _s = s;
            _rb = s.Body;
            TargetHeading = s.Heading;
        }

        /// <summary>Signed speed along the bow: positive ahead, negative astern.</summary>
        public float Speed
        {
            get
            {
                if (_rb == null) return 0f;
                return Vector2.Dot(_rb.linearVelocity, _s.Forward);
            }
        }

        /// <summary>Sideways slip, useful for the AI and for wake effects.</summary>
        public float Drift => _rb != null ? Vector2.Dot(_rb.linearVelocity, _s.Starboard) : 0f;

        public float MaxSpeed
        {
            get
            {
                var st = _s.Stats;
                float v = st.maxSpeed;
                v *= Mathf.Lerp(0.25f, 1f, _s.Damage.SystemIntegrity(ShipSystem.Engine));
                v *= Mathf.Lerp(0.55f, 1f, _s.Damage.SystemIntegrity(ShipSystem.Propulsion));
                v *= 1f - Mathf.Min(0.45f, _s.Damage.FloodingStacks * 0.15f);
                if (_s.Submarine != null) v *= _s.Submarine.SpeedMultiplier;
                if (_s.Abilities != null) v *= _s.Abilities.SpeedMultiplier;
                if (_s.Resources != null && _s.Resources.FuelFraction <= 0.001f) v *= 0.25f;
                else if (_s.Resources != null && _s.Resources.FuelFraction < 0.12f) v *= 0.7f;
                return Mathf.Max(0.15f, v);
            }
        }

        public float MaxReverseSpeed => _s.Stats.reverseSpeed * Mathf.Lerp(0.3f, 1f, _s.Damage.SystemIntegrity(ShipSystem.Engine));

        public float TurnRate
        {
            get
            {
                float steering = Mathf.Lerp(0.2f, 1f, _s.Damage.SystemIntegrity(ShipSystem.Steering));
                // a ship with no way on cannot steer; effectiveness peaks around 40% of full speed
                float flow = Mathf.Clamp01(Mathf.Abs(Speed) / Mathf.Max(0.01f, MaxSpeed * 0.4f));
                return _s.Stats.turnRate * steering * flow;
            }
        }

        public void SetThrottle(float t) => Throttle = Mathf.Clamp(t, -1f, 1f);
        public void SetRudder(float r) { RudderCommand = Mathf.Clamp(r, -1f, 1f); HeadingControl = false; }

        public void SteerToHeading(float heading)
        {
            TargetHeading = NavalMath.Wrap360(heading);
            HeadingControl = true;
        }

        public void AllStop() => Throttle = 0f;

        /// <summary>Direct control: nudge the throttle in notches like an engine order telegraph.</summary>
        public void NudgeThrottle(float delta) => Throttle = Mathf.Clamp(Throttle + delta, -1f, 1f);

        // ------------------------------------------------------------------ per-frame control

        public void CommandTick(float dt)
        {
            var st = _s.Stats;

            if (HeadingControl)
            {
                float err = Mathf.DeltaAngle(_s.Heading, TargetHeading);
                if (Speed < -0.05f) err = -err;                     // steering reverses when going astern
                float gain = Mathf.Clamp(err / 22f, -1f, 1f);
                // damp the swing so big ships do not oscillate around their course
                float currentTurn = -_rb.angularVelocity;           // heading-space turn rate
                gain -= Mathf.Clamp(currentTurn / Mathf.Max(4f, st.turnRate) * 0.35f, -0.5f, 0.5f);
                RudderCommand = Mathf.Clamp(gain, -1f, 1f);
            }

            float shift = st.rudderShift * Mathf.Lerp(0.3f, 1f, _s.Damage.SystemIntegrity(ShipSystem.Steering));
            Rudder = Mathf.MoveTowards(Rudder, RudderCommand, shift * dt);

            if (AgroundTimer > 0f) AgroundTimer -= dt;
            Aground = AgroundTimer > 0f;
            if (_landTimer > 0f) _landTimer -= dt;
        }

        // ------------------------------------------------------------------ physics

        public void PhysicsTick(float dt)
        {
            if (_rb == null || dt <= 0f) return;
            if (_s.Damage != null && _s.Damage.IsSinking) { SinkPhysics(dt); return; }

            Vector2 fwd = _s.Forward;
            Vector2 stbd = _s.Starboard;
            Vector2 vel = _rb.linearVelocity;

            // ---- engines ------------------------------------------------------
            float targetSpeed = Throttle >= 0f ? Throttle * MaxSpeed : Throttle * MaxReverseSpeed;
            float forwardSpeed = Vector2.Dot(vel, fwd);
            float speedError = targetSpeed - forwardSpeed;

            float accelLimit = Mathf.Abs(targetSpeed) > Mathf.Abs(forwardSpeed) ? _s.Stats.acceleration : _s.Stats.deceleration;
            accelLimit *= Mathf.Lerp(0.35f, 1f, _s.Damage.SystemIntegrity(ShipSystem.Engine));

            // with no throttle the hull simply coasts to a stop on drag alone
            if (Mathf.Abs(Throttle) < 0.01f) accelLimit = _s.Stats.deceleration * 0.55f;

            float accel = Mathf.Clamp(speedError * 1.8f, -accelLimit, accelLimit);
            _rb.AddForce(fwd * (accel * _rb.mass), ForceMode2D.Force);

            // ---- keel grip: kill sideways slip so the hull tracks its bow ------
            float lateral = Vector2.Dot(vel, stbd);
            float grip = _s.Stats.classType == ShipClassType.Destroyer ? 3.4f : 2.6f;
            _rb.AddForce(-stbd * (lateral * grip * _rb.mass), ForceMode2D.Force);

            // ---- rudder -------------------------------------------------------
            float desiredTurn = Rudder * TurnRate;                  // deg/s in heading space
            if (forwardSpeed < 0f) desiredTurn = -desiredTurn;      // rudder reverses going astern
            float targetAngular = -desiredTurn;                     // Unity 2D is counter-clockwise positive
            float angularError = targetAngular - _rb.angularVelocity;

            // Torque needed to close a fixed fraction of the error this step. Deriving it from dt
            // keeps the response identical at 1x and at 8x time compression, where the physics step
            // is four times longer - a constant gain would oscillate.
            const float TurnGain = 0.35f;
            float torque = _rb.inertia * angularError * Mathf.Deg2Rad * TurnGain / dt;
            _rb.AddTorque(torque, ForceMode2D.Force);

            // ---- terrain ------------------------------------------------------
            ResolveTerrain(dt);

            // ---- keep inside the battle area ----------------------------------
            var map = WorldMap.I;
            if (map != null)
            {
                Vector2 p = _rb.position;
                Vector2 clamped = map.Clamp(p);
                if ((clamped - p).sqrMagnitude > 0.0001f)
                {
                    _rb.position = clamped;
                    _rb.linearVelocity = Vector2.Lerp(_rb.linearVelocity, Vector2.zero, 0.5f);
                }
            }
        }

        /// <summary>
        /// Land is not a collider - it is a height field. Ships that stand into shallow water take
        /// grounding damage and get pushed back down the depth gradient.
        /// </summary>
        void ResolveTerrain(float dt)
        {
            var map = WorldMap.I;
            if (map == null) return;

            float required = _s.Stats.draft * WorldMap.DraftToDepth;

            // look a little ahead so big ships feel the bottom before they are hard aground
            Vector2 probe = _rb.position + _s.Forward * Mathf.Max(2f, Speed * 0.8f);
            float depthAhead = map.SampleDepth(probe);
            float depthHere = map.SampleDepth(_rb.position);

            if (depthHere >= required && depthAhead >= required) return;
            _landTimer = 1f;

            Vector2 grad = map.DepthGradient(_rb.position, _s.Stats.length * 0.8f);
            if (grad.sqrMagnitude < 1e-6f) grad = -_s.Forward;
            grad.Normalize();

            if (depthHere < required)
            {
                // hard aground
                if (Mathf.Abs(Speed) > 0.6f && AgroundTimer <= 0f)
                {
                    float impact = Mathf.Abs(Speed) / Mathf.Max(0.1f, MaxSpeed);
                    _s.Damage.ApplyDamage(_s.Stats.maxHealth * 0.03f * impact, null, DamageSource.Collision, _s.Position);
                    _s.Damage.DamageSystem(ShipSystem.Hull, 0.12f * impact);
                    if (Random.value < 0.35f * impact) _s.Damage.StartFlooding();
                    ParticleFX.Splash(_rb.position + _s.Forward * _s.Stats.length * 0.4f, 3f);
                    AudioManager.PlayAt(SoundId.Impact, _s.Position, 0.8f);
                    if (_s.team == Team.Player)
                        GameEvents.RaiseMessage(_s.shipName + " has run aground!", Team.Player);
                }
                AgroundTimer = 1.5f;
                _rb.linearVelocity = Vector2.Lerp(_rb.linearVelocity, grad * 2f, 0.35f);
                _rb.AddForce(grad * (14f * _rb.mass), ForceMode2D.Force);
            }
            else
            {
                // shoaling water ahead: bleed speed and lean away from the bank
                _rb.AddForce(grad * (5f * _rb.mass), ForceMode2D.Force);
                _rb.linearVelocity *= 0.985f;
            }
        }

        // ------------------------------------------------------------------ sinking

        public void SinkTick(float dt)
        {
            _listAngle = Mathf.MoveTowards(_listAngle, 45f, 9f * dt);
            _sinkDepth = Mathf.MoveTowards(_sinkDepth, 1f, 0.22f * dt);
        }

        void SinkPhysics(float dt)
        {
            // engines are gone: the hull coasts and slews as it settles
            _rb.linearVelocity = Vector2.MoveTowards(_rb.linearVelocity, Vector2.zero, _s.Stats.deceleration * 0.4f * dt);
            _rb.angularVelocity = Mathf.MoveTowards(_rb.angularVelocity, Rudder * 6f, 4f * dt);
        }

        public float ListAngle => _listAngle;
        public float SinkProgress => _sinkDepth;
    }
}
