using UnityEngine;

namespace Naval
{
    /// <summary>
    /// A warship. The MonoBehaviour hosts the Rigidbody2D and hull collider; every gameplay subsystem
    /// is a plain class updated in a deterministic order so the simulation never depends on Unity's
    /// component execution order.
    ///
    /// Frame work (AI, navigation, gunnery, visuals) runs in Update.
    /// Physics (thrust, rudder torque, water drag) runs in FixedUpdate.
    /// </summary>
    [RequireComponent(typeof(Rigidbody2D))]
    public class Ship : MonoBehaviour
    {
        static int _nextId = 1;

        public int id;
        public Team team = Team.Player;
        public string shipName = "Ship";
        public ShipStats Stats;

        // Subsystems are plain classes owned by the ship and ticked in a fixed order.
        // They are deliberately not serialized - the simulation is built at runtime.
        [System.NonSerialized] public ShipMovement Movement;
        [System.NonSerialized] public ShipNavigation Navigation;
        [System.NonSerialized] public ShipDamage Damage;
        [System.NonSerialized] public ShipDetection Detection;
        [System.NonSerialized] public ShipWeapons Weapons;
        [System.NonSerialized] public ShipAbilities Abilities;
        [System.NonSerialized] public SubmarineSystem Submarine;      // null for surface ships
        [System.NonSerialized] public ShipResources Resources;
        [System.NonSerialized] public ShipVisual Visual;
        [System.NonSerialized] public ShipAI AI;

        public Rigidbody2D Body { get; private set; }
        public Collider2D HullCollider { get; private set; }

        public bool IsDead { get; private set; }
        public bool Selected { get; set; }
        public bool IsDirectlyControlled { get; set; }
        /// <summary>Who issues this ship's orders. Defaults to Human for the player, RuleAI for the enemy.</summary>
        public ShipController Controller = ShipController.Human;
        /// <summary>
        /// Throttle and rudder are being set from outside every decision (a low-level learned policy),
        /// so the autopilot must not overwrite them.
        /// </summary>
        public bool ExternalHelm { get; set; }
        /// <summary>The learned policy's last movement choice, repeated when it picks "keep" (-1 = none yet).</summary>
        [System.NonSerialized] public int LearnedMove = -1;
        /// <summary>When the learned policy last changed shell type (game time).</summary>
        [System.NonSerialized] public float LearnedAmmoSwitchTime = -999f;
        public int ControlGroup { get; set; } = -1;

        public Ship CurrentTarget;             // gunnery target
        public float TimeSinceSpawn { get; private set; }

        // ------------------------------------------------------------------ transform state

        /// <summary>World position, backed by the rigidbody so physics and gameplay never disagree.</summary>
        public Vector2 Position
        {
            get => Body != null ? Body.position : new Vector2(transform.position.x, transform.position.y);
            set
            {
                if (Body != null) Body.position = value;
                transform.position = new Vector3(value.x, value.y, VisualDepth);
            }
        }

        /// <summary>Compass heading in degrees, 0 = north. Sprites point up, so z-rotation = -heading.</summary>
        public float Heading
        {
            get => NavalMath.Wrap360(-(Body != null ? Body.rotation : transform.eulerAngles.z));
            set
            {
                float wrapped = NavalMath.Wrap360(value);
                if (Body != null) Body.rotation = -wrapped;
                else transform.rotation = Quaternion.Euler(0f, 0f, -wrapped);
            }
        }

        float VisualDepth => Submarine != null && Submarine.Depth != DepthState.Surface ? 0.2f : 0f;

        public Vector2 Forward => NavalMath.HeadingToVector(Heading);
        public Vector2 Starboard { get { Vector2 f = Forward; return new Vector2(f.y, -f.x); } }
        public Vector2 Velocity => Body != null ? Body.linearVelocity : Vector2.zero;
        public float Speed => Movement != null ? Movement.Speed : 0f;
        public float SpeedKnots => Mathf.Abs(Speed) * 19.4f;   // presentation only
        public bool IsSubmarine => Stats != null && Stats.IsSubmarine;
        public float HealthFraction => Damage != null ? Damage.HealthFraction : 1f;
        public bool IsSinking => Damage != null && Damage.IsSinking;
        public string ClassTag => ShipDatabase.ShortTag(Stats.classType);

        /// <summary>How visible this ship currently is (world units at which enemies can see it).</summary>
        public float Detectability => Detection != null ? Detection.CurrentDetectability : Stats.baseDetectability;

        // ------------------------------------------------------------------ lifecycle

        public static Ship Spawn(Transform parent, Team team, ShipClassType cls, Vector2 pos, float heading, string name = null)
        {
            var stats = ShipDatabase.Get(cls);
            var go = new GameObject("Ship");
            go.layer = NavalLayers.Ships;
            go.transform.SetParent(parent, false);

            var s = go.AddComponent<Ship>();
            s.id = _nextId++;
            s.team = team;
            s.Controller = team == Team.Player ? ShipController.Human : ShipController.RuleAI;
            s.Stats = stats;
            s.shipName = string.IsNullOrEmpty(name) ? ShipDatabase.NextName(team, cls) : name;
            go.name = s.shipName;

            s.BuildPhysics(pos, heading);

            s.Movement = new ShipMovement(s);
            s.Navigation = new ShipNavigation(s);
            s.Damage = new ShipDamage(s);
            s.Detection = new ShipDetection(s);
            s.Weapons = new ShipWeapons(s);
            s.Resources = new ShipResources(s);
            if (stats.IsSubmarine) s.Submarine = new SubmarineSystem(s);
            s.Abilities = new ShipAbilities(s);
            s.Visual = new ShipVisual(s);
            s.AI = new ShipAI(s);

            ShipRegistry.Register(s);
            GameEvents.RaiseSpawned(s);
            return s;
        }

        void BuildPhysics(Vector2 pos, float heading)
        {
            Body = gameObject.GetComponent<Rigidbody2D>();
            if (Body == null) Body = gameObject.AddComponent<Rigidbody2D>();

            Body.gravityScale = 0f;
            Body.bodyType = RigidbodyType2D.Dynamic;
            Body.collisionDetectionMode = CollisionDetectionMode2D.Continuous;
            Body.interpolation = RigidbodyInterpolation2D.Interpolate;
            Body.constraints = RigidbodyConstraints2D.None;
            // displacement: a battleship is an order of magnitude heavier than a destroyer
            Body.mass = Mathf.Max(1f, Stats.length * Stats.beam * 0.6f);
            // Hull resistance is modelled by the engine servo and the keel grip in ShipMovement, so
            // the built-in damping is only here for numerical stability. Real drag here would cap a
            // battleship below its rated speed, because its acceleration budget is tiny.
            Body.linearDamping = 0.05f;
            Body.angularDamping = 0.5f;

            var col = gameObject.AddComponent<CapsuleCollider2D>();
            col.direction = CapsuleDirection2D.Vertical;
            col.size = new Vector2(Stats.beam * 1.15f, Stats.length * 0.92f);
            col.offset = Vector2.zero;
            HullCollider = col;

            Body.position = pos;
            Body.rotation = -NavalMath.Wrap360(heading);
            transform.position = new Vector3(pos.x, pos.y, 0f);
        }

        void Update()
        {
            if (IsDead) return;
            float dt = Time.deltaTime;
            if (dt <= 0f) { Visual.Tick(0f); return; }

            TimeSinceSpawn += dt;

            Abilities.Tick(dt);
            Resources.Tick(dt);
            Submarine?.Tick(dt);
            Damage.Tick(dt);

            if (!Damage.IsSinking)
            {
                AI.Tick(dt);
                // a ship under direct control takes its helm orders from the player, not the autopilot
                if (!IsDirectlyControlled && !ExternalHelm) Navigation.Tick(dt);
                Movement.CommandTick(dt);
                Weapons.Tick(dt);
            }
            else
            {
                Movement.SinkTick(dt);
            }

            ApplyVisualDepth();
            Visual.Tick(dt);
        }

        void FixedUpdate()
        {
            if (IsDead || Movement == null) return;
            Movement.PhysicsTick(Time.fixedDeltaTime);
        }

        void ApplyVisualDepth()
        {
            var p = transform.position;
            float z = VisualDepth;
            if (!Mathf.Approximately(p.z, z)) transform.position = new Vector3(p.x, p.y, z);
        }

        // ------------------------------------------------------------------ collisions

        void OnCollisionEnter2D(Collision2D collision)
        {
            if (IsDead || Damage == null) return;
            var other = collision.collider.GetComponentInParent<Ship>();
            if (other == null) return;

            // Ramming: damage scales with the closing speed and the other hull's mass, and it does
            // not care whose side the other ship is on - shouldering a squadron mate out of the way
            // costs both of you plating. The threshold is low because ships now run at real speeds:
            // a destroyer's whole speed range is 0 to 2 units/second, so 1.2 would have meant only
            // head-on collisions ever registered.
            float closing = collision.relativeVelocity.magnitude;
            if (closing < 0.35f) return;

            float massRatio = other.Body != null && Body != null
                ? Mathf.Clamp(other.Body.mass / Mathf.Max(1f, Body.mass), 0.3f, 3f) : 1f;
            float dmg = Stats.maxHealth * 0.010f * closing * massRatio;

            Damage.ApplyDamage(dmg, other, DamageSource.Collision, Position);
            // a hard ram opens plates below the waterline
            if (closing > 1.6f && Random.value < 0.4f) Damage.StartFlooding();

            ParticleFX.Splash(collision.GetContact(0).point, 2.5f);
            AudioManager.PlayAt(SoundId.Impact, Position, 0.7f);

            if (team == Team.Player && closing > 0.8f)
                GameEvents.RaiseMessage(
                    other.team == team
                        ? shipName + " fouled " + other.shipName + " - both hulls damaged"
                        : shipName + " rammed " + other.shipName,
                    Team.Player);
        }

        // ------------------------------------------------------------------ death

        public void Kill(Ship killer)
        {
            if (IsDead) return;
            IsDead = true;
            Selected = false;
            IsDirectlyControlled = false;
            if (HullCollider != null) HullCollider.enabled = false;
            if (Body != null) Body.simulated = false;
            ShipRegistry.Unregister(this);
            GameEvents.RaiseDestroyed(this, killer);
            Visual.OnDestroyed();
            AudioManager.PlayAt(SoundId.Sinking, Position, 1f);
            Destroy(gameObject, 6f);
        }

        void OnDestroy()
        {
            ShipRegistry.Unregister(this);
        }

        // ------------------------------------------------------------------ helpers

        public bool IsHostileTo(Ship other) => other != null && other.team != team && other.team != Team.Neutral;

        public float DistanceTo(Ship other) => Vector2.Distance(Position, other.Position);

        /// <summary>Angle of a point relative to our bow, 0 = dead ahead, 180 = astern.</summary>
        public float BearingTo(Vector2 point)
        {
            float h = NavalMath.VectorToHeading(point - Position);
            return Mathf.Abs(Mathf.DeltaAngle(Heading, h));
        }

        public bool CanBeTargetedBy(Ship shooter)
        {
            if (IsDead || Damage.IsSinking) return false;
            // submerged boats can only be engaged by ASW weapons
            if (Submarine != null && Submarine.Depth != DepthState.Surface) return false;
            return true;
        }
    }

    /// <summary>Physics layers created at boot so shells and ships do not fight each other.</summary>
    public static class NavalLayers
    {
        public const int Ships = 0;      // default layer; ships collide with ships only
    }
}
