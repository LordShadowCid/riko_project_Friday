using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using Annabeth.UI;

namespace Annabeth.Avatar
{
    /// <summary>
    /// Scrollable library panel listing available VRM models.
    /// Persists entries to avatars.json. Allows loading, removing, and adding new VRMs.
    /// Based on Mate-Engine AvatarLibraryMenu.cs — simplified, runtime-built UI.
    /// </summary>
    public class VrmModelLibrary : MonoBehaviour
    {
        private static bool _isLibraryOpen;
        public static bool IsLibraryOpen => _isLibraryOpen;

        private RectTransform _panelRect;
        private RectTransform _listContent;
        private Button _btnAddNew;
        private Button _btnClose;

        private AvatarController _avatarController;
        private List<LibraryEntry> _entries = new List<LibraryEntry>();
        private readonly List<GameObject> _entryRows = new List<GameObject>();

        private string LibraryPath =>
            Path.Combine(Application.persistentDataPath, "avatars.json");

        public void SetAvatarController(AvatarController ctrl) => _avatarController = ctrl;

        private void Awake()
        {
            BuildUI();
            LoadLibrary();
        }

        private void OnEnable()
        {
            _isLibraryOpen = true;
            RefreshList();
        }

        private void OnDisable()
        {
            _isLibraryOpen = false;
        }

        // ── UI Construction ─────────────────────────────────────

        private void BuildUI()
        {
            var panelSize = new Vector2(400, 460);
            _panelRect = UIFactory.CreatePanel(transform, "LibraryBg", panelSize);
            _panelRect.anchorMin = new Vector2(0.5f, 0.5f);
            _panelRect.anchorMax = new Vector2(0.5f, 0.5f);
            _panelRect.anchoredPosition = Vector2.zero;

            // Title
            var titleRow = new Vector2(380, 28);
            var title = UIFactory.CreateSectionHeader(_panelRect, "Character Library", titleRow);
            var titleRt = title.GetComponent<RectTransform>();
            titleRt.anchorMin = new Vector2(0.5f, 1f);
            titleRt.anchorMax = new Vector2(0.5f, 1f);
            titleRt.pivot = new Vector2(0.5f, 1f);
            titleRt.anchoredPosition = new Vector2(0, -6);

            // Scroll view for entries
            var svSize = new Vector2(panelSize.x - 12, panelSize.y - 90);
            var (_, content) = UIFactory.CreateScrollView(_panelRect, svSize);
            var svRt = content.parent.parent.GetComponent<RectTransform>();
            svRt.anchorMin = new Vector2(0.5f, 0.5f);
            svRt.anchorMax = new Vector2(0.5f, 0.5f);
            svRt.anchoredPosition = new Vector2(0, 8);
            _listContent = content;

            // Bottom buttons
            var btnSize = new Vector2(170, 30);

            _btnAddNew = UIFactory.CreateButton(_panelRect, "BtnAddNew", "Add New VRM", btnSize);
            var addRt = _btnAddNew.GetComponent<RectTransform>();
            addRt.anchorMin = new Vector2(0.25f, 0f);
            addRt.anchorMax = new Vector2(0.25f, 0f);
            addRt.pivot = new Vector2(0.5f, 0f);
            addRt.anchoredPosition = new Vector2(0, 8);

            _btnClose = UIFactory.CreateButton(_panelRect, "BtnClose", "Close", btnSize);
            var closeRt = _btnClose.GetComponent<RectTransform>();
            closeRt.anchorMin = new Vector2(0.75f, 0f);
            closeRt.anchorMax = new Vector2(0.75f, 0f);
            closeRt.pivot = new Vector2(0.5f, 0f);
            closeRt.anchoredPosition = new Vector2(0, 8);

            _btnAddNew.onClick.AddListener(OnAddNew);
            _btnClose.onClick.AddListener(OnClose);
        }

        // ── Library Persistence ─────────────────────────────────

        private void LoadLibrary()
        {
            _entries.Clear();

            if (File.Exists(LibraryPath))
            {
                try
                {
                    string json = File.ReadAllText(LibraryPath);
                    var wrapper = JsonUtility.FromJson<LibraryWrapper>(json);
                    if (wrapper?.entries != null)
                        _entries.AddRange(wrapper.entries);
                }
                catch (Exception e)
                {
                    Debug.LogError($"[VrmModelLibrary] Failed to load library: {e.Message}");
                }
            }

            // Ensure default model entry exists
            EnsureDefaultEntry();
        }

        private void SaveLibrary()
        {
            try
            {
                var wrapper = new LibraryWrapper { entries = _entries };
                string json = JsonUtility.ToJson(wrapper, true);
                File.WriteAllText(LibraryPath, json);
            }
            catch (Exception e)
            {
                Debug.LogError($"[VrmModelLibrary] Failed to save library: {e.Message}");
            }
        }

        private void EnsureDefaultEntry()
        {
            const string defaultRelPath = "Models/claire_avatar.vrm";
            bool found = false;
            foreach (var e in _entries)
            {
                if (e.isDefault) { found = true; break; }
            }

            if (!found)
            {
                _entries.Insert(0, new LibraryEntry
                {
                    displayName = "Claire (Default)",
                    filePath = defaultRelPath,
                    isDefault = true,
                    isRelativeToStreaming = true
                });
            }
        }

        // ── List Rendering ──────────────────────────────────────

        private void RefreshList()
        {
            // Destroy old rows
            foreach (var row in _entryRows)
                Destroy(row);
            _entryRows.Clear();

            for (int i = 0; i < _entries.Count; i++)
            {
                var entry = _entries[i];
                var row = CreateEntryRow(entry, i);
                _entryRows.Add(row);
            }
        }

        private GameObject CreateEntryRow(LibraryEntry entry, int index)
        {
            var rowSize = new Vector2(360, 46);
            var rowRt = UIFactory.CreatePanel(_listContent, $"Entry_{index}", rowSize,
                new Color(0.16f, 0.16f, 0.19f, 0.9f));
            var rowGo = rowRt.gameObject;

            // Name label
            var nameGo = new GameObject("Name");
            nameGo.transform.SetParent(rowRt, false);
            var nameRt = nameGo.AddComponent<RectTransform>();
            nameRt.anchorMin = new Vector2(0, 0);
            nameRt.anchorMax = new Vector2(0.6f, 1);
            nameRt.offsetMin = new Vector2(10, 2);
            nameRt.offsetMax = new Vector2(0, -2);
            var nameTxt = nameGo.AddComponent<Text>();
            nameTxt.text = entry.displayName;
            nameTxt.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf")
                ?? Font.CreateDynamicFontFromOSFont("Segoe UI", 14);
            nameTxt.fontSize = 14;
            nameTxt.color = UIFactory.TextColor;
            nameTxt.alignment = TextAnchor.MiddleLeft;

            // Load button
            var loadBtn = UIFactory.CreateButton(rowRt, "BtnLoad", "Load",
                new Vector2(60, 28), 12);
            var loadRt = loadBtn.GetComponent<RectTransform>();
            loadRt.anchorMin = new Vector2(1, 0.5f);
            loadRt.anchorMax = new Vector2(1, 0.5f);
            loadRt.pivot = new Vector2(1, 0.5f);
            loadRt.anchoredPosition = new Vector2(entry.isDefault ? -8 : -74, 0);

            int capturedIndex = index;
            loadBtn.onClick.AddListener(() => OnLoadEntry(capturedIndex));

            // Highlight if this is the currently loaded model
            string currentPath = Core.SettingsManager.Instance?.data.selectedModelPath ?? "";
            bool isCurrent = entry.isDefault
                ? string.IsNullOrEmpty(currentPath)
                : string.Equals(entry.filePath, currentPath, StringComparison.OrdinalIgnoreCase);

            if (isCurrent)
            {
                var img = rowRt.GetComponent<Image>();
                if (img != null)
                    img.color = new Color(0.20f, 0.28f, 0.20f, 0.9f);
            }

            // Remove button (not for default)
            if (!entry.isDefault)
            {
                var removeBtn = UIFactory.CreateButton(rowRt, "BtnRemove", "X",
                    new Vector2(28, 28), 12);
                var removeRt = removeBtn.GetComponent<RectTransform>();
                removeRt.anchorMin = new Vector2(1, 0.5f);
                removeRt.anchorMax = new Vector2(1, 0.5f);
                removeRt.pivot = new Vector2(1, 0.5f);
                removeRt.anchoredPosition = new Vector2(-8, 0);

                // Tint remove button reddish
                var rmColors = removeBtn.colors;
                rmColors.normalColor = new Color(0.45f, 0.20f, 0.20f, 1f);
                rmColors.highlightedColor = new Color(0.60f, 0.25f, 0.25f, 1f);
                rmColors.pressedColor = new Color(0.35f, 0.15f, 0.15f, 1f);
                removeBtn.colors = rmColors;

                removeBtn.onClick.AddListener(() => OnRemoveEntry(capturedIndex));
            }

            return rowGo;
        }

        // ── Actions ─────────────────────────────────────────────

        private async void OnLoadEntry(int index)
        {
            if (index < 0 || index >= _entries.Count) return;
            var entry = _entries[index];

            if (_avatarController == null)
                _avatarController = FindFirstObjectByType<AvatarController>();

            if (_avatarController == null)
            {
                Debug.LogError("[VrmModelLibrary] No AvatarController found!");
                return;
            }

            if (entry.isDefault)
            {
                // Load the bundled default model (relative path)
                await _avatarController.LoadVRM(entry.filePath);
                if (Core.SettingsManager.Instance != null)
                {
                    Core.SettingsManager.Instance.data.selectedModelPath = "";
                    Core.SettingsManager.Instance.SaveAll();
                }
            }
            else
            {
                // Load external model (absolute path)
                await _avatarController.LoadVRM(entry.filePath, isAbsolutePath: true);
                if (Core.SettingsManager.Instance != null)
                {
                    Core.SettingsManager.Instance.data.selectedModelPath = entry.filePath;
                    Core.SettingsManager.Instance.SaveAll();
                }
            }

            Debug.Log($"[VrmModelLibrary] Loaded: {entry.displayName}");
            RefreshList(); // Update highlight
        }

        private void OnRemoveEntry(int index)
        {
            if (index < 0 || index >= _entries.Count) return;
            if (_entries[index].isDefault) return;

            Debug.Log($"[VrmModelLibrary] Removed: {_entries[index].displayName}");
            _entries.RemoveAt(index);
            SaveLibrary();
            RefreshList();
        }

        private void OnAddNew()
        {
            string path = VrmFilePicker.OpenVrmFileDialog();
            if (string.IsNullOrEmpty(path)) return;

            // Check for duplicates
            foreach (var e in _entries)
            {
                if (string.Equals(e.filePath, path, StringComparison.OrdinalIgnoreCase))
                {
                    Debug.Log("[VrmModelLibrary] Model already in library.");
                    return;
                }
            }

            // Extract display name from filename
            string fileName = Path.GetFileNameWithoutExtension(path);
            var entry = new LibraryEntry
            {
                displayName = fileName,
                filePath = path,
                isDefault = false,
                isRelativeToStreaming = false,
                dateAdded = DateTime.Now.ToString("yyyy-MM-dd HH:mm")
            };

            _entries.Add(entry);
            SaveLibrary();
            RefreshList();

            Debug.Log($"[VrmModelLibrary] Added: {fileName}");
        }

        private void OnClose()
        {
            gameObject.SetActive(false);
        }

        private void OnDestroy()
        {
            _btnAddNew?.onClick.RemoveAllListeners();
            _btnClose?.onClick.RemoveAllListeners();
        }

        // ── Data Types ──────────────────────────────────────────

        [Serializable]
        public class LibraryEntry
        {
            public string displayName;
            public string filePath;
            public bool isDefault;
            public bool isRelativeToStreaming;
            public string dateAdded;
        }

        [Serializable]
        private class LibraryWrapper
        {
            public List<LibraryEntry> entries = new List<LibraryEntry>();
        }
    }
}
