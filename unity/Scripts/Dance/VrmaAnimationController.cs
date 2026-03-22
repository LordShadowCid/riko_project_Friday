using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;
using UniGLTF;
using UniVRM10;

namespace Annabeth.Dance
{
    /// <summary>
    /// Loads and plays .vrma (VRM Animation) files at runtime.
    /// Uses UniVRM's VrmAnimationImporter and retargets to the active VRM model
    /// via Vrm10Runtime.VrmAnimation.
    /// </summary>
    public class VrmaAnimationController : MonoBehaviour
    {
        [Header("Settings")]
        [SerializeField] private string animationsFolder = "Animations";
        [SerializeField] private bool loop = true;

        private Vrm10Instance _vrm;
        private RuntimeGltfInstance _loadedGltf;
        private Vrm10AnimationInstance _currentAnimation;
        private Animation _legacyAnimation;
        private bool _isPlaying;

        public bool IsPlaying => _isPlaying;

        public void Initialize(Vrm10Instance vrm)
        {
            _vrm = vrm;
        }

        /// <summary>
        /// Load a .vrma file from StreamingAssets/Animations and play it.
        /// </summary>
        public async Task LoadAndPlay(string fileName, CancellationToken ct = default)
        {
            Stop();

            if (_vrm == null)
            {
                Debug.LogError("[VrmaAnimation] No VRM instance set. Call Initialize first.");
                return;
            }

            string fullPath = Path.Combine(Application.streamingAssetsPath, animationsFolder, fileName);
            if (!File.Exists(fullPath))
            {
                Debug.LogError($"[VrmaAnimation] File not found: {fullPath}");
                return;
            }

            try
            {
                Debug.Log($"[VrmaAnimation] Loading: {fileName}");

                // Parse the GLB data
                var data = new GlbFileParser(fullPath).Parse();

                // Import as VRM Animation
                using var importer = new VrmAnimationImporter(data);
                var instance = await importer.LoadAsync(new RuntimeOnlyAwaitCaller());

                _loadedGltf = instance;
                _currentAnimation = instance.GetComponent<Vrm10AnimationInstance>();
                _legacyAnimation = instance.GetComponent<Animation>();

                if (_currentAnimation == null)
                {
                    Debug.LogError("[VrmaAnimation] Loaded file has no Vrm10AnimationInstance component.");
                    CleanupLoaded();
                    return;
                }

                // Hide the box man visualization
                _currentAnimation.ShowBoxMan(false);

                // Configure looping on the legacy Animation component
                if (_legacyAnimation != null && _legacyAnimation.clip != null)
                {
                    _legacyAnimation.clip.wrapMode = loop ? WrapMode.Loop : WrapMode.Once;
                    _legacyAnimation.wrapMode = loop ? WrapMode.Loop : WrapMode.Once;
                    _legacyAnimation.Play();
                }

                // Assign to VRM runtime — this triggers automatic retargeting each frame
                _vrm.Runtime.VrmAnimation = _currentAnimation;

                _isPlaying = true;
                Debug.Log($"[VrmaAnimation] Playing: {fileName} (loop={loop})");
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[VrmaAnimation] Error loading {fileName}: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                CleanupLoaded();
            }
        }

        /// <summary>
        /// Stop the current animation and release resources.
        /// </summary>
        public void Stop()
        {
            if (_vrm != null && _vrm.Runtime != null)
            {
                _vrm.Runtime.VrmAnimation = null;
            }

            _isPlaying = false;
            CleanupLoaded();
        }

        /// <summary>
        /// Pause the current animation.
        /// </summary>
        public void Pause()
        {
            if (_legacyAnimation != null && _isPlaying)
            {
                // Legacy Animation doesn't have a simple Pause.
                // Reduce speed to 0 to freeze.
                foreach (AnimationState state in _legacyAnimation)
                {
                    state.speed = 0f;
                }
            }
        }

        /// <summary>
        /// Resume a paused animation.
        /// </summary>
        public void Resume()
        {
            if (_legacyAnimation != null && _isPlaying)
            {
                foreach (AnimationState state in _legacyAnimation)
                {
                    state.speed = 1f;
                }
            }
        }

        public void SetLoop(bool looping)
        {
            loop = looping;
            if (_legacyAnimation != null && _legacyAnimation.clip != null)
            {
                _legacyAnimation.clip.wrapMode = loop ? WrapMode.Loop : WrapMode.Once;
                _legacyAnimation.wrapMode = loop ? WrapMode.Loop : WrapMode.Once;
            }
        }

        private void CleanupLoaded()
        {
            if (_loadedGltf != null)
            {
                _loadedGltf.Dispose();
                _loadedGltf = null;
            }
            _currentAnimation = null;
            _legacyAnimation = null;
        }

        private void OnDestroy()
        {
            Stop();
        }
    }
}
