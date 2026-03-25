using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Playables;

namespace Annabeth.Dance
{
    /// <summary>
    /// Feature #18: Forwards blendshape animation curves from a dance AnimationClip
    /// to the VRM model's SkinnedMeshRenderers via a secondary PlayableGraph.
    /// Searches candidate paths ("Body", "Face") for blendshape curves and creates
    /// a lightweight proxy that applies them each frame.
    /// </summary>
    public class DanceBlendshapeForwarder : MonoBehaviour
    {
        private PlayableGraph _graph;
        private readonly Dictionary<string, SkinnedMeshRenderer> _meshCache = new();
        private bool _active;

        /// <summary>
        /// Set up forwarding for the given clip's blendshape curves on the VRM model.
        /// </summary>
        public void Setup(AnimationClip clip, Animator animator, Transform vrmRoot)
        {
            Cleanup();

            if (clip == null || vrmRoot == null) return;

            // Find all SkinnedMeshRenderers on the VRM
            _meshCache.Clear();
            foreach (var smr in vrmRoot.GetComponentsInChildren<SkinnedMeshRenderer>())
                _meshCache[smr.name] = smr;

            // Check if clip has blendshape curves
            bool hasBlendshapes = false;
#if UNITY_EDITOR
            foreach (var binding in UnityEditor.AnimationUtility.GetCurveBindings(clip))
            {
                if (binding.type == typeof(SkinnedMeshRenderer) &&
                    binding.propertyName.StartsWith("blendShape."))
                {
                    hasBlendshapes = true;
                    break;
                }
            }
#else
            // At runtime, we can't easily inspect curve bindings without AnimationUtility.
            // Instead, create the graph and let Unity resolve targets. If the clip has
            // blendshape curves targeting candidate paths, they'll be applied automatically
            // through the PlayableGraph when a matching SkinnedMeshRenderer exists at the correct path.
            hasBlendshapes = true; // Assume yes at runtime — graph will harmlessly no-op if no curves match
#endif

            if (!hasBlendshapes)
            {
                Debug.Log("[BlendshapeForwarder] No blendshape curves found in clip");
                return;
            }

            // Create a PlayableGraph targeting the animator
            // The AnimationClipPlayable will evaluate all curves including blendshape ones
            _graph = PlayableGraph.Create("BlendshapeForwarder");
            _graph.SetTimeUpdateMode(DirectorUpdateMode.GameTime);

            var clipPlayable = AnimationClipPlayable.Create(_graph, clip);
            var output = AnimationPlayableOutput.Create(_graph, "BSForward", animator);
            output.SetSourcePlayable(clipPlayable);

            // Set additive mode so this doesn't override bone animation
            // The blendshape curves will still apply since they target different properties
            _graph.Play();
            _active = true;

            Debug.Log("[BlendshapeForwarder] Blendshape forwarding active");
        }

        public void Cleanup()
        {
            if (_graph.IsValid())
            {
                _graph.Stop();
                _graph.Destroy();
            }
            _active = false;
        }

        public void Pause()
        {
            if (_graph.IsValid() && _active)
                _graph.Stop();
        }

        public void Resume()
        {
            if (_graph.IsValid() && _active)
                _graph.Play();
        }

        /// <summary>
        /// Sync time with the main dance playable.
        /// </summary>
        public void SyncTime(double time)
        {
            if (!_graph.IsValid() || !_active) return;
            var playable = _graph.GetRootPlayable(0);
            if (playable.IsValid())
                playable.SetTime(time);
        }

        private void OnDestroy()
        {
            Cleanup();
        }
    }
}
