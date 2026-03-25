using System;
using System.Collections.Generic;
using UnityEngine;

namespace Annabeth.Core
{
    [Serializable]
    public class TimerEntry
    {
        public string label = "Timer";
        public float durationSeconds = 300f;
        public float remainingSeconds;
        public bool isRunning;
    }

    /// <summary>
    /// Feature #15: Manages countdown timers and alarms.
    /// When a timer fires, invokes OnTimerFired so CompanionManager
    /// can trigger speech bubble + expression.
    /// </summary>
    public class AlarmTimerManager : MonoBehaviour
    {
        public event Action<TimerEntry> OnTimerFired;

        private readonly List<TimerEntry> _timers = new();

        public IReadOnlyList<TimerEntry> Timers => _timers;

        /// <summary>
        /// Add a new timer with given label and duration in seconds.
        /// Returns the created entry.
        /// </summary>
        public TimerEntry AddTimer(string label, float durationSeconds)
        {
            var entry = new TimerEntry
            {
                label = label,
                durationSeconds = durationSeconds,
                remainingSeconds = durationSeconds,
                isRunning = false
            };
            _timers.Add(entry);
            return entry;
        }

        public void RemoveTimer(TimerEntry entry) => _timers.Remove(entry);

        public void StartTimer(TimerEntry entry)
        {
            entry.remainingSeconds = entry.durationSeconds;
            entry.isRunning = true;
        }

        public void StopTimer(TimerEntry entry)
        {
            entry.isRunning = false;
        }

        public void ResetTimer(TimerEntry entry)
        {
            entry.remainingSeconds = entry.durationSeconds;
            entry.isRunning = false;
        }

        private void Update()
        {
            for (int i = _timers.Count - 1; i >= 0; i--)
            {
                var t = _timers[i];
                if (!t.isRunning) continue;

                t.remainingSeconds -= Time.deltaTime;
                if (t.remainingSeconds <= 0f)
                {
                    t.remainingSeconds = 0f;
                    t.isRunning = false;
                    Debug.Log($"[AlarmTimer] Timer fired: {t.label}");
                    OnTimerFired?.Invoke(t);
                }
            }
        }
    }
}
