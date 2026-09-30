using System;
using UnityEngine;

namespace Naval
{
    /// <summary>Loose coupling hub - systems announce, UI and AI listen.</summary>
    public static class GameEvents
    {
        public static event Action<string, Team> OnMessage;
        public static event Action<Ship> OnShipSpawned;
        public static event Action<Ship, Ship> OnShipDestroyed;      // victim, killer (killer may be null)
        public static event Action<Ship, float, Ship> OnShipDamaged;   // victim, amount, attacker (may be null)
        public static event Action<Ship> OnSelectionChanged;
        public static event Action<Ship, Ship> OnContactGained;      // observer, contact
        public static event Action<Vector2, Team> OnTorpedoWarning;

        public static void RaiseMessage(string msg, Team team = Team.Neutral) => OnMessage?.Invoke(msg, team);
        public static void RaiseSpawned(Ship s) => OnShipSpawned?.Invoke(s);
        public static void RaiseDestroyed(Ship victim, Ship killer) => OnShipDestroyed?.Invoke(victim, killer);
        public static void RaiseDamaged(Ship s, float amount, Ship attacker) => OnShipDamaged?.Invoke(s, amount, attacker);
        public static void RaiseSelectionChanged(Ship s) => OnSelectionChanged?.Invoke(s);
        public static void RaiseContact(Ship observer, Ship contact) => OnContactGained?.Invoke(observer, contact);
        public static void RaiseTorpedoWarning(Vector2 pos, Team team) => OnTorpedoWarning?.Invoke(pos, team);

        public static void ClearAll()
        {
            OnMessage = null;
            OnShipSpawned = null;
            OnShipDestroyed = null;
            OnShipDamaged = null;
            OnSelectionChanged = null;
            OnContactGained = null;
            OnTorpedoWarning = null;
        }
    }
}
