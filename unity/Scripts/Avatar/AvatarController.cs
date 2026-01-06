using System.Threading.Tasks;
using UnityEngine;
using UniVRM10;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Main avatar controller that coordinates all VRM components.
    /// Handles VRM loading and provides access to VRM instance.
    /// </summary>
    public class AvatarController : MonoBehaviour
    {
        [Header("VRM Settings")]
        [SerializeField] private string vrmPath = "Models/claire_avatar.vrm";
        [SerializeField] private bool loadOnStart = true;
        
        [Header("Components")]
        [SerializeField] private LipSyncController lipSyncController;
        [SerializeField] private EmotionController emotionController;
        [SerializeField] private BlinkController blinkController;
        [SerializeField] private EyeTrackingController eyeTrackingController;

        private Vrm10Instance _vrmInstance;
        private Vrm10RuntimeExpression _expression;
        private bool _isLoaded;

        public Vrm10Instance VrmInstance => _vrmInstance;
        public Vrm10RuntimeExpression Expression => _expression;
        public bool IsLoaded => _isLoaded;

        // Events
        public event System.Action<Vrm10Instance> OnVrmLoaded;

        private async void Start()
        {
            if (loadOnStart)
            {
                await LoadVRM(vrmPath);
            }
        }

        /// <summary>
        /// Load a VRM model from the StreamingAssets folder.
        /// </summary>
        public async Task LoadVRM(string path)
        {
            // Clean up existing VRM
            if (_vrmInstance != null)
            {
                Destroy(_vrmInstance.gameObject);
                _vrmInstance = null;
                _expression = null;
                _isLoaded = false;
            }

            try
            {
                string fullPath = System.IO.Path.Combine(Application.streamingAssetsPath, path);
                Debug.Log($"[AvatarController] Loading VRM from: {fullPath}");

                // Load VRM asynchronously
                _vrmInstance = await Vrm10.LoadPathAsync(fullPath,
                    canLoadVrm0X: true,
                    showMeshes: true,
                    materialGenerator: null,
                    vrmMetaInformationCallback: null,
                    ct: destroyCancellationToken);

                if (_vrmInstance == null)
                {
                    Debug.LogError("[AvatarController] Failed to load VRM!");
                    return;
                }

                // Position and parent the VRM
                _vrmInstance.transform.SetParent(transform, false);
                _vrmInstance.transform.localPosition = Vector3.zero;
                _vrmInstance.transform.localRotation = Quaternion.identity;

                // Get expression runtime
                _expression = _vrmInstance.Runtime.Expression;

                // Initialize sub-controllers
                InitializeControllers();

                _isLoaded = true;
                Debug.Log("[AvatarController] VRM loaded successfully!");
                OnVrmLoaded?.Invoke(_vrmInstance);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"[AvatarController] Error loading VRM: {e.Message}");
            }
        }

        private void InitializeControllers()
        {
            // Auto-find controllers if not assigned
            if (lipSyncController == null)
                lipSyncController = GetComponentInChildren<LipSyncController>();
            if (emotionController == null)
                emotionController = GetComponentInChildren<EmotionController>();
            if (blinkController == null)
                blinkController = GetComponentInChildren<BlinkController>();
            if (eyeTrackingController == null)
                eyeTrackingController = GetComponentInChildren<EyeTrackingController>();

            // Initialize with VRM instance
            lipSyncController?.Initialize(_vrmInstance);
            emotionController?.Initialize(_vrmInstance);
            blinkController?.Initialize(_vrmInstance);
            eyeTrackingController?.Initialize(_vrmInstance);
        }

        /// <summary>
        /// Get a humanoid bone transform.
        /// </summary>
        public Transform GetBone(HumanBodyBones bone)
        {
            if (!_isLoaded || _vrmInstance == null) return null;
            
            var animator = _vrmInstance.GetComponent<Animator>();
            return animator?.GetBoneTransform(bone);
        }

        private void OnDestroy()
        {
            if (_vrmInstance != null)
            {
                Destroy(_vrmInstance.gameObject);
            }
        }
    }
}
