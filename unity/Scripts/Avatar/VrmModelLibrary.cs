using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using UniVRM10;
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
        private RectTransform _panelRect;
        private RectTransform _listContent;
        private Button _btnAddNew;
        private Button _btnClose;

        private AvatarController _avatarController;
        private List<LibraryEntry> _entries = new List<LibraryEntry>();
        private readonly List<GameObject> _entryRows = new List<GameObject>();
        private bool _isLoading;

        /// <summary>Called when the Close button is pressed — RadialMenu hooks this to sync state.</summary>
        public event System.Action OnCloseRequested;

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
            ScanForNewModels();
            RefreshList();
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

        /// <summary>
        /// Auto-scan well-known directories for .vrm files and add any new ones
        /// to the library. Scans:
        ///   1) StreamingAssets/Models/  (built-in, shipped with build)
        ///   2) {exe_dir}/VRM_Models/   (user drop-in folder next to the exe)
        /// </summary>
        private void ScanForNewModels()
        {
            var scanDirs = new List<(string dir, bool isStreaming)>
            {
                (Path.Combine(Application.streamingAssetsPath, "Models"), true),
                (Path.Combine(Application.dataPath, "..", "VRM_Models"), false),
            };

            bool changed = false;

            foreach (var (dir, isStreaming) in scanDirs)
            {
                string resolvedDir = Path.GetFullPath(dir);
                if (!Directory.Exists(resolvedDir))
                {
                    // Create the user VRM_Models folder if it doesn't exist, so the user knows where to put files
                    if (!isStreaming)
                    {
                        try { Directory.CreateDirectory(resolvedDir); }
                        catch (Exception e) { Debug.LogWarning($"[VrmModelLibrary] Could not create {resolvedDir}: {e.Message}"); }
                    }
                    continue;
                }

                string[] vrmFiles;
                try { vrmFiles = Directory.GetFiles(resolvedDir, "*.vrm", SearchOption.TopDirectoryOnly); }
                catch (Exception e)
                {
                    Debug.LogWarning($"[VrmModelLibrary] Could not scan {resolvedDir}: {e.Message}");
                    continue;
                }

                foreach (string fullPath in vrmFiles)
                {
                    string normalizedFull = Path.GetFullPath(fullPath);

                    // Check if already in library (by full path or relative path)
                    bool alreadyExists = false;
                    foreach (var e in _entries)
                    {
                        string existingFull = e.isRelativeToStreaming
                            ? Path.GetFullPath(Path.Combine(Application.streamingAssetsPath, e.filePath))
                            : Path.GetFullPath(e.filePath);

                        if (string.Equals(existingFull, normalizedFull, StringComparison.OrdinalIgnoreCase))
                        {
                            alreadyExists = true;
                            break;
                        }
                    }

                    if (alreadyExists) continue;

                    string fileName = Path.GetFileNameWithoutExtension(fullPath);

                    if (isStreaming)
                    {
                        // Store as relative path for built-in models
                        string relPath = "Models/" + Path.GetFileName(fullPath);
                        _entries.Add(new LibraryEntry
                        {
                            displayName = fileName,
                            filePath = relPath,
                            isDefault = false,
                            isRelativeToStreaming = true,
                            dateAdded = DateTime.Now.ToString("yyyy-MM-dd HH:mm"),
                            author = ""
                        });
                    }
                    else
                    {
                        // Store absolute path for user-added models
                        _entries.Add(new LibraryEntry
                        {
                            displayName = fileName,
                            filePath = normalizedFull,
                            isDefault = false,
                            isRelativeToStreaming = false,
                            dateAdded = DateTime.Now.ToString("yyyy-MM-dd HH:mm"),
                            author = ""
                        });
                    }

                    Debug.Log($"[VrmModelLibrary] Auto-discovered: {fileName} from {resolvedDir}");
                    changed = true;
                }
            }

            if (changed)
                SaveLibrary();
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
            nameTxt.text = !string.IsNullOrEmpty(entry.author) && entry.author != "Unknown"
                ? $"{entry.displayName}  ({entry.author})"
                : entry.displayName;
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
            if (_isLoading) return;
            if (index < 0 || index >= _entries.Count) return;
            var entry = _entries[index];

            if (_avatarController == null)
                _avatarController = FindFirstObjectByType<AvatarController>();

            if (_avatarController == null)
            {
                Debug.LogError("[VrmModelLibrary] No AvatarController found!");
                return;
            }

            _isLoading = true;
            try
            {
                if (entry.isDefault)
                {
                    await _avatarController.LoadVRM(entry.filePath);
                    if (Core.SettingsManager.Instance != null)
                    {
                        Core.SettingsManager.Instance.data.selectedModelPath = "";
                        Core.SettingsManager.Instance.SaveAll();
                    }
                }
                else
                {
                    if (!File.Exists(entry.filePath))
                    {
                        Debug.LogWarning($"[VrmModelLibrary] File not found: {entry.filePath}");
                        return;
                    }
                    await _avatarController.LoadVRM(entry.filePath, isAbsolutePath: true);
                    if (Core.SettingsManager.Instance != null)
                    {
                        Core.SettingsManager.Instance.data.selectedModelPath = entry.filePath;
                        Core.SettingsManager.Instance.SaveAll();
                    }
                }

                Debug.Log($"[VrmModelLibrary] Loaded: {entry.displayName}");
                RefreshList();
            }
            catch (Exception e)
            {
                Debug.LogError($"[VrmModelLibrary] Load failed: {e.Message}");
            }
            finally
            {
                _isLoading = false;
            }
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

        private async void OnAddNew()
        {
            if (_isLoading) return;

            string path = VrmFilePicker.OpenVrmFileDialog();
            if (string.IsNullOrEmpty(path)) return;

            if (!File.Exists(path))
            {
                Debug.LogWarning($"[VrmModelLibrary] File not found: {path}");
                return;
            }

            // Check for duplicates
            foreach (var e in _entries)
            {
                if (string.Equals(e.filePath, path, StringComparison.OrdinalIgnoreCase))
                {
                    Debug.Log("[VrmModelLibrary] Model already in library.");
                    return;
                }
            }

            // Extract display name from filename initially
            string fileName = Path.GetFileNameWithoutExtension(path);
            string displayName = fileName;
            string author = "Unknown";

            // Try to extract VRM metadata
            _isLoading = true;
            try
            {
                var vrm10 = await UniVRM10.Vrm10.LoadPathAsync(path,
                    canLoadVrm0X: true, showMeshes: false);

                if (vrm10 != null)
                {
                    var meta = vrm10.Vrm?.Meta;
                    if (meta != null)
                    {
                        if (!string.IsNullOrEmpty(meta.Name))
                            displayName = meta.Name;
                        if (meta.Authors != null && meta.Authors.Count > 0
                            && !string.IsNullOrEmpty(meta.Authors[0]))
                            author = meta.Authors[0];
                    }
                    Destroy(vrm10.gameObject);
                }
            }
            catch (Exception e)
            {
                Debug.LogWarning($"[VrmModelLibrary] Could not read VRM metadata: {e.Message}");
            }
            finally
            {
                _isLoading = false;
            }

            var entry = new LibraryEntry
            {
                displayName = displayName,
                filePath = path,
                isDefault = false,
                isRelativeToStreaming = false,
                dateAdded = DateTime.Now.ToString("yyyy-MM-dd HH:mm"),
                author = author
            };

            _entries.Add(entry);
            SaveLibrary();
            RefreshList();

            Debug.Log($"[VrmModelLibrary] Added: {displayName} by {author}");
        }

        private void OnClose()
        {
            OnCloseRequested?.Invoke();
        }

        /// <summary>
        /// Feature #14: Get display info string ("ModelName by Author") for the currently loaded model.
        /// </summary>
        public string GetCurrentModelInfo()
        {
            string currentPath = Core.SettingsManager.Instance?.data.selectedModelPath ?? "";
            foreach (var e in _entries)
            {
                bool isCurrent = e.isDefault
                    ? string.IsNullOrEmpty(currentPath)
                    : string.Equals(e.filePath, currentPath, StringComparison.OrdinalIgnoreCase);
                if (isCurrent)
                {
                    if (!string.IsNullOrEmpty(e.author) && e.author != "Unknown")
                        return $"{e.displayName} by {e.author}";
                    return e.displayName;
                }
            }
            return "Unknown Model";
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
            public string author;
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
