using System;
using System.Collections;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Profiling;

namespace Annabeth.Core
{
    /// <summary>
    /// Periodic GC collection and working-set trim for a long-running desktop process.
    /// Ported from Mate-Engine MemoryTrim.cs + GCCollect.cs.
    /// Trims every 10 minutes when enabled, plus once at startup.
    /// </summary>
    public class MemoryOptimizer : MonoBehaviour
    {
#if UNITY_STANDALONE_WIN
        [DllImport("psapi.dll")]
        private static extern bool EmptyWorkingSet(IntPtr hProcess);

        [DllImport("kernel32.dll")]
        private static extern IntPtr GetCurrentProcess();
#endif

        [Header("Settings")]
        [SerializeField] private float trimIntervalSeconds = 600f; // 10 minutes
        [SerializeField] private float startupDelaySeconds = 10f;

        private Coroutine _autoTrimRoutine;

        private void Start()
        {
            ApplySettings();
        }

        /// <summary>Called by SettingsManager.ApplyAllSettings().</summary>
        public void ApplySettings()
        {
            bool enabled = SettingsManager.Instance != null
                && SettingsManager.Instance.data.enableAutoMemoryTrim;

            if (enabled && _autoTrimRoutine == null)
            {
                _autoTrimRoutine = StartCoroutine(AutoTrimRoutine());
            }
            else if (!enabled && _autoTrimRoutine != null)
            {
                StopCoroutine(_autoTrimRoutine);
                _autoTrimRoutine = null;
            }
        }

        /// <summary>Force an immediate trim. Useful after VRM swap.</summary>
        public void TrimNow()
        {
            long before = Profiler.GetMonoUsedSizeLong();

            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            Resources.UnloadUnusedAssets();

#if UNITY_STANDALONE_WIN
            EmptyWorkingSet(GetCurrentProcess());
#endif

            long after = Profiler.GetMonoUsedSizeLong();
            long freedKB = (before - after) / 1024;
            Debug.Log($"[MemoryOptimizer] Trim complete. Freed ~{freedKB}KB (Mono: {after / 1024}KB)");
        }

        private IEnumerator AutoTrimRoutine()
        {
            // Startup delay — let VRM and other assets finish loading first
            yield return new WaitForSeconds(startupDelaySeconds);
            TrimNow();

            while (true)
            {
                yield return new WaitForSeconds(trimIntervalSeconds);
                TrimNow();
            }
        }
    }
}
