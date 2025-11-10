# ==================================================
# FILE: ui/main_gui.py
# ==================================================
"""
Main GUI for Plugin Brain v3
- Uses new core modules for logging, scanning, and classification.
- Provides editable source (Installed DB) and target (Organized DB) paths.
- Includes an "Editor" tab for manual plugin correction (move operation).
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
from pathlib import Path
import threading
import json
import traceback

# --- FIX for NameError ---
from typing import List, Optional, Dict 
# -------------------------

# Import the new core modules
from core.logger import get_logger
from core.scanner import PluginScanner
from core.file_manager import FileAgent
from core.llm_core import llm_core

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class PluginBrainApp(ctk.CTk):
    
    CONFIG_FILE = Path("config.json")

    def __init__(self):
        super().__init__()
        self.title("Plugin Brain v3 - AI Multi-Agent Organizer")
        self.geometry("1000x800")
        
        # Get the logger
        self.logger = get_logger()
        
        # App state
        self.config = self._load_config()
        self.scanner: Optional[PluginScanner] = None
        self.file_agent: Optional[FileAgent] = None
        self.plugins: List[Dict] = [] # Holds all scanned plugin info dicts
        self.is_scanning = False
        self.is_classifying = False

        self._build_ui()
        
        # Auto-save config on exit
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.logger.info("Plugin Brain GUI Initialized.")

    def _load_config(self):
        """Loads config.json, creating it if it doesn't exist."""
        default = {
            "installed_path": r"C:\Users\dimbe\Documents\Image-Line\FL Studio\Presets\Plugin database\Installed",
            "target_root": r"C:\Users\dimbe\Documents\Image-Line\FL Studio\Presets\Plugin database"
        }
        if self.CONFIG_FILE.exists():
            try:
                config = json.loads(self.CONFIG_FILE.read_text(encoding="utf-8"))
                # Ensure all keys are present
                for key, value in default.items():
                    if key not in config:
                        config[key] = value
                return config
            except Exception as e:
                self.logger.error(f"Failed to load config.json: {e}. Using defaults.")
                return default
        
        # Write default config if it doesn't exist
        try:
            self.CONFIG_FILE.write_text(json.dumps(default, indent=2), encoding="utf-8")
        except Exception as e:
            self.logger.error(f"Failed to write default config: {e}")
        return default

    def _save_config(self):
        """Saves current config to config.json."""
        try:
            # Update config from UI fields before saving
            self.config["installed_path"] = self.installed_entry.get()
            self.config["target_root"] = self.target_entry.get()
            self.CONFIG_FILE.write_text(json.dumps(self.config, indent=2), encoding="utf-8")
            self.logger.info("Configuration saved.")
        except Exception as e:
            self.logger.error(f"Failed to save config: {e}")

    def _on_close(self):
        """Handles window close event."""
        self._save_config()
        self.destroy()

    # --- UI Building ---

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        # --- Paths Frame ---
        path_frame = ctk.CTkFrame(self)
        path_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        path_frame.grid_columnconfigure(1, weight=1)

        # Installed DB Path (Source)
        ctk.CTkLabel(path_frame, text="FL 'Installed' Path (Source):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.installed_entry = ctk.CTkEntry(path_frame)
        self.installed_entry.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        self.installed_entry.insert(0, self.config.get("installed_path", ""))
        ctk.CTkButton(path_frame, text="Browse...", width=100, command=self._browse_installed).grid(row=0, column=2, padx=10, pady=5)

        # Organized DB Path (Target)
        ctk.CTkLabel(path_frame, text="Organized DB Path (Target):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.target_entry = ctk.CTkEntry(path_frame)
        self.target_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.target_entry.insert(0, self.config.get("target_root", ""))
        ctk.CTkButton(path_frame, text="Browse...", width=100, command=self._browse_target).grid(row=1, column=2, padx=10, pady=5)

        # --- Action Frame ---
        action_frame = ctk.CTkFrame(self)
        action_frame.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        self.scan_button = ctk.CTkButton(action_frame, text="1. Scan Installed Plugins", command=self._start_scan)
        self.scan_button.pack(side="left", padx=10, pady=10)
        
        self.classify_button = ctk.CTkButton(action_frame, text="2. Classify & Organize All", command=self._start_classification)
        self.classify_button.pack(side="left", padx=10, pady=10)

        self.progress = ctk.CTkProgressBar(action_frame)
        self.progress.pack(side="left", fill="x", expand=True, padx=10, pady=10)
        self.progress.set(0)

        self.status_label = ctk.CTkLabel(action_frame, text="Ready.")
        self.status_label.pack(side="right", padx=10, pady=10)

        # --- Tab View ---
        self.tab_view = ctk.CTkTabview(self)
        self.tab_view.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        self.tab_view.add("Log")
        self.tab_view.add("Editor")

        # --- Log Tab ---
        self.log_text = ctk.CTkTextbox(self.tab_view.tab("Log"), state="disabled", font=("Consolas", 12))
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        # --- Editor Tab ---
        editor_frame = self.tab_view.tab("Editor")
        editor_frame.grid_columnconfigure(0, weight=1)
        editor_frame.grid_rowconfigure(1, weight=1)

        # Editor Controls
        editor_controls = ctk.CTkFrame(editor_frame)
        editor_controls.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        editor_controls.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(editor_controls, text="Filter:").grid(row=0, column=0, padx=5, pady=5)
        self.filter_entry = ctk.CTkEntry(editor_controls)
        self.filter_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.filter_entry.bind("<KeyRelease>", self._filter_plugin_list)

        # Editor Main Area
        editor_main = ctk.CTkFrame(editor_frame)
        editor_main.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        editor_main.grid_columnconfigure(0, weight=1)
        editor_main.grid_columnconfigure(1, weight=1)
        editor_main.grid_rowconfigure(0, weight=1)

        # Plugin List
        self.plugin_list = tk.Listbox(editor_main, font=("Segoe UI", 10), background="#2b2b2b", foreground="white", borderwidth=0, highlightthickness=0, selectbackground="#0078D7")
        self.plugin_list.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.plugin_list_scroll = ctk.CTkScrollbar(editor_main, command=self.plugin_list.yview)
        self.plugin_list_scroll.grid(row=0, column=1, sticky="ns")
        self.plugin_list.configure(yscrollcommand=self.plugin_list_scroll.set)
        self.plugin_list.bind("<<ListboxSelect>>", self._on_plugin_select)

        # Details & Correction Area
        correction_frame = ctk.CTkFrame(editor_main)
        correction_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        correction_frame.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(correction_frame, text="Plugin Details:").pack(anchor="w", padx=5)
        self.details_text = ctk.CTkTextbox(correction_frame, height=200, font=("Consolas", 11), state="disabled")
        self.details_text.pack(fill="x", expand=True, padx=5, pady=5)

        ctk.CTkLabel(correction_frame, text="Corrected Path (e.g., Effects/Reverb/Vendor/Plugin):").pack(anchor="w", padx=5)
        self.new_path_entry = ctk.CTkEntry(correction_frame)
        self.new_path_entry.pack(fill="x", padx=5, pady=5)

        self.move_button = ctk.CTkButton(correction_frame, text="Move Plugin to Corrected Path", command=self._editor_move_plugin, fg_color="red")
        self.move_button.pack(fill="x", padx=5, pady=10)
        
    # --- Logging ---
    
    def _log_gui(self, msg: str, level: str = "info"):
        """Thread-safe logging to the GUI text box."""
        def _append():
            self.log_text.configure(state="normal")
            self.log_text.insert("end", f"{msg}\n")
            self.log_text.configure(state="disabled")
            self.log_text.see("end")
        
        # Always call GUI updates from the main thread
        self.after(0, _append)
        
        # Also log to file
        if level == "info":
            self.logger.info(msg)
        elif level == "warning":
            self.logger.warning(msg)
        elif level == "error":
            self.logger.error(msg)
        else:
            self.logger.debug(msg)

    # --- Path Browsing ---

    def _browse_installed(self):
        path = filedialog.askdirectory(title="Select FL 'Installed' Database Path")
        if path:
            self.installed_entry.delete(0, "end")
            self.installed_entry.insert(0, path)
            self.config["installed_path"] = path

    def _browse_target(self):
        path = filedialog.askdirectory(title="Select Target (Organized) Database Root")
        if path:
            self.target_entry.delete(0, "end")
            self.target_entry.insert(0, path)
            self.config["target_root"] = path

    # --- Core Functions ---
    
    def _start_scan(self):
        """Starts the plugin scanning thread."""
        if self.is_scanning:
            self._log_gui("Scan already in progress.", "warning")
            return

        self.is_scanning = True
        self._set_ui_state("scanning")
        
        self.config["installed_path"] = self.installed_entry.get()
        self.scanner = PluginScanner([self.config["installed_path"]])
        
        self.plugins.clear()
        self.plugin_list.delete(0, "end")
        
        threading.Thread(target=self._scan_thread, daemon=True).start()

    def _scan_thread(self):
        """Worker thread for scanning plugins."""
        self._log_gui("Starting plugin scan...", "info")
        count = 0
        try:
            for plugin_info in self.scanner.scan():
                self.plugins.append(plugin_info)
                # Update listbox safely
                self.after(0, self._add_to_editor_list, plugin_info)
                count += 1
            self._log_gui(f"Scan complete. Found {count} .fst plugins.", "info")
        except Exception as e:
            self._log_gui(f"Scan failed: {e}\n{traceback.format_exc()}", "error")
            
        self.is_scanning = False
        self.after(0, lambda: self._set_ui_state("idle"))

    def _start_classification(self):
        """Starts the plugin classification thread."""
        if self.is_classifying:
            self._log_gui("Classification already in progress.", "warning")
            return
        if not self.plugins:
            self._log_gui("No plugins scanned. Please scan first.", "warning")
            return

        self.is_classifying = True
        self._set_ui_state("classifying")
        
        self.config["target_root"] = self.target_entry.get()
        self.file_agent = FileAgent(self.config["target_root"], dry_run=False)
        
        threading.Thread(target=self._classify_thread, daemon=True).start()

    def _classify_thread(self):
        """Worker thread for classifying plugins."""
        self._log_gui("Starting classification...", "info")
        total = len(self.plugins)
        count = 0
        
        for i, plugin_info in enumerate(self.plugins):
            try:
                # 1. Reason with LLM Core
                path_segments = llm_core.reason_path(plugin_info)
                
                # 2. Handle Skip (Native Plugins)
                if path_segments is None:
                    self._log_gui(f"Skipping native plugin: {plugin_info['name']}", "debug")
                    continue
                    
                # 3. Build Path
                dest_path = self.file_agent.build_path(path_segments)
                
                # 4. Transfer File (Copy)
                self.file_agent.transfer_plugin(plugin_info['path'], dest_path, move=False)
                
                # Store the classified path in metadata for the editor
                plugin_info['metadata']['classified_path'] = path_segments
                
                count += 1
                
            except Exception as e:
                self._log_gui(f"Failed to classify {plugin_info['name']}: {e}", "error")
            
            # Update progress bar safely
            progress_val = (i + 1) / total
            self.after(0, lambda p=progress_val: self.progress.set(p))
            
        self._log_gui(f"Classification complete. Organized {count} plugins.", "info")
        
        # After classifying, rebuild the list to show any new uncertain highlights
        self.after(0, self._filter_plugin_list)
        
        self.is_classifying = False
        self.after(0, lambda: self._set_ui_state("idle"))

    # --- Editor Functions ---

    def _add_to_editor_list(self, plugin_info: Dict):
        """Safely adds a plugin to the editor listbox."""
        display_name = plugin_info['name']
        version = plugin_info.get('version', 'Unknown')
        
        if version != 'Unknown':
            display_name += f" ({version})"
        
        listbox_index = self.plugin_list.size()
        self.plugin_list.insert("end", display_name)
        
        # Highlight uncertain items (check if path starts with _UNSURE)
        classified_path = plugin_info.get('metadata', {}).get('classified_path', [])
        if classified_path and classified_path[0] == "_UNSURE":
             self.plugin_list.itemconfig(listbox_index, {'bg': '#573A00', 'fg': '#FFD700'}) # Dark gold


    def _filter_plugin_list(self, event=None):
        """Filters the editor listbox based on the filter entry."""
        query = self.filter_entry.get().lower()
        self.plugin_list.delete(0, "end")
        
        for plugin_info in self.plugins:
            if query in plugin_info['name'].lower():
                self._add_to_editor_list(plugin_info)

    def _on_plugin_select(self, event=None):
        """Displays selected plugin details in the editor."""
        try:
            selected_indices = self.plugin_list.curselection()
            if not selected_indices:
                return
            
            # Find the full plugin_info object
            selected_name_full = self.plugin_list.get(selected_indices[0])
            selected_name = selected_name_full.split(" (")[0] # Get base name
            
            plugin_info = next((p for p in self.plugins if p['name'] == selected_name), None)

            if plugin_info:
                self.details_text.configure(state="normal")
                self.details_text.delete("1.0", "end")
                self.details_text.insert("1.0", json.dumps(plugin_info, indent=2))
                self.details_text.configure(state="disabled")
                
                # Pre-fill the new path entry with the classified path
                classified_path = plugin_info.get('metadata', {}).get('classified_path', [])
                # Join and remove _UNSURE prefix if present
                path_str = "/".join([p for p in classified_path if p != "_UNSURE"])
                self.new_path_entry.delete(0, "end")
                self.new_path_entry.insert(0, path_str)
            
        except Exception as e:
            self._log_gui(f"Error displaying details: {e}", "error")

    def _editor_move_plugin(self):
        """Moves the selected plugin to the new path (Editor correction)."""
        try:
            selected_indices = self.plugin_list.curselection()
            if not selected_indices:
                messagebox.showwarning("No Plugin Selected", "Please select a plugin from the list to move.")
                return
            
            # Find the full plugin_info object
            selected_name_full = self.plugin_list.get(selected_indices[0])
            selected_name = selected_name_full.split(" (")[0]
            
            plugin_info = next((p for p in self.plugins if p['name'] == selected_name), None)

            if not plugin_info:
                messagebox.showerror("Error", "Could not find plugin data. Please rescan.")
                return

            new_path_str = self.new_path_entry.get().strip()
            if not new_path_str:
                messagebox.showwarning("No Path Entered", "Please enter a new relative path in the 'Corrected Path' box (e.g., Effects/Reverb/Vendor/Plugin).")
                return

            if not self.file_agent:
                 self.file_agent = FileAgent(self.config["target_root"], dry_run=False)

            # Build new path and move the file
            path_segments = new_path_str.split('/')
            dest_path = self.file_agent.build_path(path_segments)
            
            self._log_gui(f"Editor Move: Moving {plugin_info['name']} to {dest_path}...", "info")
            
            # [cite_start]Perform the MOVE operation [cite: 502]
            self.file_agent.transfer_plugin(plugin_info['path'], dest_path, move=True)
            
            self._log_gui(f"Editor Move Successful: {plugin_info['name']}", "info")
            
            # Remove from list (it's no longer in 'Installed')
            self.plugin_list.delete(selected_indices[0])
            self.plugins = [p for p in self.plugins if p['path'] != plugin_info['path']]
            
        except Exception as e:
            self._log_gui(f"Editor Move Failed: {e}", "error")
            messagebox.showerror("Move Failed", f"Could not move plugin: {e}")

    # --- UI State ---
    
    def _set_ui_state(self, state: str):
        """Disables/Enables buttons based on app state."""
        if state == "scanning" or state == "classifying":
            self.scan_button.configure(state="disabled")
            self.classify_button.configure(state="disabled")
            self.move_button.configure(state="disabled")
            self.progress.start() # Indeterminate progress
            self.status_label.configure(text=f"{state.capitalize()}...")
        else: # idle
            self.scan_button.configure(state="normal")
            self.classify_button.configure(state="normal")
            self.move_button.configure(state="normal")
            self.progress.stop()
            self.progress.set(0)
            self.status_label.configure(text="Ready.")

if __name__ == "__main__":
    app = PluginBrainApp()
    app.mainloop()