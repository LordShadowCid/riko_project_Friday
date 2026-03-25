using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Animations;
using UnityEngine.Audio;
using UnityEngine.Playables;

namespace Annabeth.Dance
{
    /// <summary>
    /// Feature #16: Loads custom dance animations from .unity3d AssetBundles
    /// in StreamingAssets/CustomDances/. Each bundle may contain an AnimationClip
    /// and an AudioClip. Uses PlayableGraph for playback on the VRM Animator.
    /// </summary>
    public class CustomDanceLoader : MonoBehaviour
    {
        [Header("Settings")]
        [SerializeField] private string customDancesFolder = "CustomDances";

        public event Action<string> OnDanceStarted;
        public event Action OnDanceStopped;

        private Animator _animator;
        private AudioSource _audioSource;
        private PlayableGraph _graph;
        private AnimationClipPlayable _clipPlayable;
        private AudioClipPlayable _audioPlayable;
        private bool _isPlaying;

        private readonly List<string> _availableDances = new();
        private int _currentIndex = -1;

        public bool IsPlaying => _isPlaying;
        public IReadOnlyList<string> AvailableDances => _availableDances;
        public int CurrentIndex => _currentIndex;
        public float Progress => GetProgress();
        public float Volume
        {
            get => _audioSource != null ? _audioSource.volume : 1f;
            set { if (_audioSource != null) _audioSource.volume = value; }
        }

        private void Awake()
        {
            _audioSource = GetComponent<AudioSource>();
            if (_audioSource == null)
                _audioSource = gameObject.AddComponent<AudioSource>();
        }

        public void Initialize(Animator animator)
        {
            _animator = animator;
            ScanDances();
        }

        /// <summary>
        /// Scan the CustomDances folder for .unity3d AssetBundles.
        /// </summary>
        public void ScanDances()
        {
            _availableDances.Clear();
            string folder = Path.Combine(Application.streamingAssetsPath, customDancesFolder);
            if (!Directory.Exists(folder))
            {
                Directory.CreateDirectory(folder);
                return;
            }

            foreach (var file in Directory.GetFiles(folder, "*.unity3d"))
                _availableDances.Add(Path.GetFileNameWithoutExtension(file));

            Debug.Log($"[CustomDance] Found {_availableDances.Count} custom dances");
        }

        /// <summary>
        /// Load and play a custom dance by index.
        /// </summary>
        public void PlayDance(int index)
        {
            if (index < 0 || index >= _availableDances.Count) return;
            if (_animator == null) return;

            Stop();

            string bundlePath = Path.Combine(
                Application.streamingAssetsPath, customDancesFolder,
                _availableDances[index] + ".unity3d");

            AssetBundle bundle = null;
            try
            {
                bundle = AssetBundle.LoadFromFile(bundlePath);
                if (bundle == null)
                {
                    Debug.LogError($"[CustomDance] Failed to load bundle: {bundlePath}");
                    return;
                }

                var clip = bundle.LoadAsset<AnimationClip>(bundle.GetAllAssetNames()[0]);
                if (clip == null)
                {
                    // Try finding any AnimationClip
                    var clips = bundle.LoadAllAssets<AnimationClip>();
                    if (clips.Length > 0) clip = clips[0];
                }

                var audioClip = bundle.LoadAsset<AudioClip>(bundle.GetAllAssetNames()[0]);
                if (audioClip == null)
                {
                    var audioClips = bundle.LoadAllAssets<AudioClip>();
                    if (audioClips.Length > 0) audioClip = audioClips[0];
                }

                if (clip == null)
                {
                    Debug.LogError($"[CustomDance] No AnimationClip found in: {bundlePath}");
                    return;
                }

                // Create PlayableGraph
                _graph = PlayableGraph.Create("CustomDance");
                _graph.SetTimeUpdateMode(DirectorUpdateMode.GameTime);

                _clipPlayable = AnimationClipPlayable.Create(_graph, clip);
                var animOutput = AnimationPlayableOutput.Create(_graph, "AnimOutput", _animator);
                animOutput.SetSourcePlayable(_clipPlayable);

                // Audio if available
                if (audioClip != null && _audioSource != null)
                {
                    _audioPlayable = AudioClipPlayable.Create(_graph, audioClip, true);
                    var audioOutput = AudioPlayableOutput.Create(_graph, "AudioOutput", _audioSource);
                    audioOutput.SetSourcePlayable(_audioPlayable);
                }

                _graph.Play();
                _currentIndex = index;
                _isPlaying = true;
                OnDanceStarted?.Invoke(_availableDances[index]);
                Debug.Log($"[CustomDance] Playing: {_availableDances[index]}");
            }
            catch (Exception e)
            {
                Debug.LogError($"[CustomDance] Error loading {bundlePath}: {e.Message}");
            }
            finally
            {
                bundle?.Unload(false);
            }
        }

        public void PlayNext()
        {
            if (_availableDances.Count == 0) return;
            int next = (_currentIndex + 1) % _availableDances.Count;
            PlayDance(next);
        }

        public void PlayPrevious()
        {
            if (_availableDances.Count == 0) return;
            int prev = _currentIndex <= 0 ? _availableDances.Count - 1 : _currentIndex - 1;
            PlayDance(prev);
        }

        public void Pause()
        {
            if (_graph.IsValid() && _isPlaying)
                _graph.Stop();
        }

        public void Resume()
        {
            if (_graph.IsValid() && _isPlaying)
                _graph.Play();
        }

        public void Stop()
        {
            if (_graph.IsValid())
            {
                _graph.Stop();
                _graph.Destroy();
            }
            if (_isPlaying)
            {
                _isPlaying = false;
                _currentIndex = -1;
                OnDanceStopped?.Invoke();
            }
        }

        public void SetTime(float normalizedTime)
        {
            if (_graph.IsValid() && _clipPlayable.IsValid())
            {
                double duration = _clipPlayable.GetAnimationClip().length;
                _clipPlayable.SetTime(normalizedTime * duration);
            }
        }

        private float GetProgress()
        {
            if (!_graph.IsValid() || !_clipPlayable.IsValid()) return 0f;
            var clip = _clipPlayable.GetAnimationClip();
            if (clip == null || clip.length <= 0f) return 0f;
            return (float)(_clipPlayable.GetTime() % clip.length / clip.length);
        }

        private void OnDestroy()
        {
            if (_graph.IsValid())
                _graph.Destroy();
        }
    }
}
