using UnityEngine;

namespace Naval.RL
{
    /// <summary>
    /// Command-line demo: skips the menu and starts a 6v6 domination battle with the trained policy
    /// flying one or both fleets, so it can be watched without clicking through the setup screen.
    ///
    ///   NavalTrainer.x86_64 -rlDemo both        policy vs policy
    ///   NavalTrainer.x86_64 -rlDemo enemy       you command, the policy is the enemy
    ///   NavalTrainer.x86_64 -rlDemo player      the policy flies your fleet against the rule AI
    ///   -rlDemoShips 6   fleet size       -rlDemoSeconds 60   quit after this long (for checks)
    /// </summary>
    public class RLDemo : MonoBehaviour
    {
        string _who = "both";
        int _ships = 6;
        float _quitAfter;
        int _frames;

        public static bool Requested => RLCommandLine.Has("-rlDemo");

        public static RLDemo Create(Transform parent)
        {
            var go = new GameObject("RLDemo");
            go.transform.SetParent(parent, false);
            var d = go.AddComponent<RLDemo>();
            d._who = RLCommandLine.Str("-rlDemo", "both").ToLowerInvariant();
            d._ships = Mathf.Clamp(RLCommandLine.Int("-rlDemoShips", 6), 1, FleetSetup.MaxShips);
            d._quitAfter = RLCommandLine.Int("-rlDemoSeconds", 0);
            return d;
        }

        void Start()
        {
            var driver = RLPolicyDriver.I;
            if (driver == null || !driver.Available)
            {
                Debug.LogWarning("[RL] demo: " + (driver != null ? driver.Status : "no policy driver") + " - the rule AI will fly both fleets");
            }
            bool learned = driver != null && driver.Available;
            var setup = FleetSetup.Default();
            setup.playerShipCount = setup.enemyShipCount = _ships;
            setup.playerController = learned && (_who == "both" || _who == "player") ? ShipController.Learned : ShipController.Human;
            setup.enemyController = learned && (_who == "both" || _who == "enemy") ? ShipController.Learned : ShipController.RuleAI;
            if (setup.playerController == ShipController.Human && _who == "player") setup.playerController = ShipController.RuleAI;
            GameManager.I.BeginTrainingMatch(GameMode.Domination, setup, MapConfig.ForPreset(MapPreset.OceanArchipelago),
                                             Random.Range(1, 999999), true);
            Debug.Log("[RL] demo: " + _ships + "v" + _ships + ", player fleet " + setup.playerController + ", enemy fleet " + setup.enemyController);
        }

        void Update()
        {
            _frames++;
            var gm = GameManager.I;
            if (_frames == 2 && gm.Phase == GamePhase.Deployment) gm.StartBattle();
            if (_quitAfter > 0f && gm.Phase == GamePhase.Battle && gm.BattleTime >= _quitAfter)
            {
                var d = RLPolicyDriver.I;
                Debug.Log("[RL] demo summary: " + (d != null ? d.DecisionsMade : 0) + " policy decisions, score " +
                          Mathf.RoundToInt(gm.PlayerScore) + " - " + Mathf.RoundToInt(gm.EnemyScore) +
                          ", fleet strength " + gm.FleetPoints + " - " + gm.EnemyFleetPoints);
                Application.Quit();
            }
        }
    }
}
