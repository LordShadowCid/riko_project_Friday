using UnityEngine;

namespace Annabeth.Interaction
{
    /// <summary>
    /// Spawns particle effects at touch positions on the avatar.
    /// Hearts for head zone, sparkles for body zone.
    /// Toggle controlled by SettingsManager.data.enableParticles.
    /// Simplified from Mate-Engine AvatarParticleHandler.cs (no theme system).
    /// </summary>
    public class ParticleEffectHandler : MonoBehaviour
    {
        private ParticleSystem _heartParticles;
        private ParticleSystem _sparkleParticles;

        private void Start()
        {
            _heartParticles = CreateParticleSystem("HeartParticles",
                new Color(1f, 0.4f, 0.55f), startSize: 0.08f, burstCount: 8);
            _sparkleParticles = CreateParticleSystem("SparkleParticles",
                new Color(1f, 0.85f, 0.3f), startSize: 0.05f, burstCount: 12);
        }

        public void PlayAtPosition(Vector3 worldPos, bool isHeadZone)
        {
            if (!IsEnabled()) return;
            var ps = isHeadZone ? _heartParticles : _sparkleParticles;
            if (ps == null) return;
            ps.transform.position = worldPos;
            ps.Play();
        }

        private bool IsEnabled()
        {
            var sm = Core.SettingsManager.Instance;
            return sm != null && sm.data.enableParticles;
        }

        private ParticleSystem CreateParticleSystem(string goName, Color color,
            float startSize, int burstCount)
        {
            var go = new GameObject(goName);
            go.hideFlags = HideFlags.DontSave;
            go.transform.SetParent(transform, false);
            go.SetActive(false); // Prevent ParticleSystem from auto-playing
            var ps = go.AddComponent<ParticleSystem>();

            var main = ps.main;
            main.playOnAwake = false;
            main.duration = 0.5f;
            main.loop = false;
            main.startLifetime = 0.8f;
            main.startSpeed = 1.5f;
            main.startSize = startSize;
            main.startColor = color;
            main.maxParticles = burstCount * 2;
            main.simulationSpace = ParticleSystemSimulationSpace.World;
            main.gravityModifier = -0.3f; // float upward

            var emission = ps.emission;
            emission.rateOverTime = 0;
            emission.SetBursts(new[] { new ParticleSystem.Burst(0f, burstCount) });

            var shape = ps.shape;
            shape.shapeType = ParticleSystemShapeType.Sphere;
            shape.radius = 0.08f;

            var sizeOverLifetime = ps.sizeOverLifetime;
            sizeOverLifetime.enabled = true;
            sizeOverLifetime.size = new ParticleSystem.MinMaxCurve(1f,
                AnimationCurve.Linear(0, 1, 1, 0));

            var colorOverLifetime = ps.colorOverLifetime;
            colorOverLifetime.enabled = true;
            var gradient = new Gradient();
            gradient.SetKeys(
                new[] { new GradientColorKey(color, 0f), new GradientColorKey(color, 1f) },
                new[] { new GradientAlphaKey(1f, 0f), new GradientAlphaKey(0f, 1f) }
            );
            colorOverLifetime.color = gradient;

            // Use URP particle shader with built-in fallback
            var renderer = go.GetComponent<ParticleSystemRenderer>();
            var shader = Shader.Find("Universal Render Pipeline/Particles/Unlit")
                ?? Shader.Find("Particles/Standard Unlit")
                ?? Shader.Find("Hidden/Internal-Colored");
            if (shader != null)
            {
                var mat = new Material(shader);
                mat.color = color;
                renderer.material = mat;
            }

            ps.Stop();
            go.SetActive(true);
            return ps;
        }
    }
}
