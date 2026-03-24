using UnityEngine;

namespace Annabeth.Interaction
{
    /// <summary>
    /// Plays audio clips on touch reactions and window drag start/end.
    /// Based on Mate-Engine AvatarDragSoundHandler.cs — extended for touch.
    /// Volume and toggle controlled by SettingsManager.
    /// </summary>
    public class TouchSoundHandler : MonoBehaviour
    {
        [Header("Touch Sounds")]
        [SerializeField] private AudioClip[] touchSounds;

        [Header("Drag Sounds")]
        [SerializeField] private AudioClip dragStartSound;
        [SerializeField] private AudioClip dragEndSound;

        [Header("Pitch Variation")]
        [SerializeField, Range(0, 30)] private float pitchVariation = 10f;

        private AudioSource _source;

        private void Awake()
        {
            _source = gameObject.AddComponent<AudioSource>();
            _source.playOnAwake = false;
            _source.spatialBlend = 0f; // 2D sound
        }

        public void PlayTouchSound()
        {
            if (!IsEnabled() || touchSounds == null || touchSounds.Length == 0) return;
            var clip = touchSounds[Random.Range(0, touchSounds.Length)];
            PlayWithPitch(clip);
        }

        public void PlayDragStart()
        {
            if (!IsEnabled() || dragStartSound == null) return;
            PlayWithPitch(dragStartSound);
        }

        public void PlayDragEnd()
        {
            if (!IsEnabled() || dragEndSound == null) return;
            PlayWithPitch(dragEndSound);
        }

        private void PlayWithPitch(AudioClip clip)
        {
            if (_source == null || clip == null) return;
            _source.volume = Core.SettingsManager.Instance?.data.sfxVolume ?? 1f;
            float v = pitchVariation / 100f;
            _source.pitch = Random.Range(1f - v, 1f + v);
            _source.PlayOneShot(clip);
        }

        private bool IsEnabled()
        {
            var sm = Core.SettingsManager.Instance;
            return sm != null && sm.data.enableTouchSounds;
        }
    }
}
